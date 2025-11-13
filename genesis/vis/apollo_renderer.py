import math

import numpy as np
import torch

import genesis as gs
from genesis.repr_base import RBC
from genesis.constants import IMAGE_TYPE
from genesis.utils.misc import ti_to_torch
from genesis.utils.scene_exporter import SceneDescription
from genesis.utils.scene_exporter import _pos_to_y_up, _quat_to_y_up

try:
    from gs_apollo import ApolloRenderer as ApolloRendererImpl
except ImportError as e:
    gs.raise_exception_from("Failed to import Apollo renderer.", e)


# Helper functions
def get_cuda_device_uuid():
    cuda_device_index = torch.cuda.current_device()
    cuda_device_uuid_bytes = bytes(torch.cuda.get_device_properties(cuda_device_index).uuid.bytes)
    return cuda_device_uuid_bytes


def _merge_based_on_export_level(geom_pos, geom_quat, idx):
    geom_pos = torch.index_select(geom_pos, -2, idx)
    geom_quat = torch.index_select(geom_quat, -2, idx)
    return geom_pos, geom_quat


def _get_max_camera_resolution(cameras):
    if not cameras:
        return (1024, 1024)
    return max(camera.res for camera in cameras)


class Light:
    def __init__(self, pos, dir, color, intensity, directional, castshadow, cutoff, attenuation):
        self._pos = pos
        norm = math.sqrt(sum(x * x for x in dir))
        self._dir = tuple(x / norm for x in dir)
        self._color = color
        self._intensity = intensity
        self._directional = directional
        self._castshadow = castshadow
        self._cutoff = cutoff
        self._attenuation = attenuation

    @property
    def pos(self):
        return self._pos

    @property
    def dir(self):
        return self._dir

    @property
    def color(self):
        return self._color

    @property
    def intensity(self):
        return self._intensity

    @property
    def directional(self):
        return self._directional

    @property
    def castshadow(self):
        return self._castshadow

    @property
    def cutoffRad(self):
        return math.radians(self._cutoff)

    @property
    def cutoffDeg(self):
        return self._cutoff

    @property
    def attenuation(self):
        return self._attenuation


class ApolloRenderer(RBC):
    """
    This class is used to manage batch rendering
    """

    def __init__(self, visualizer, renderer_options, vis_options):
        self._visualizer = visualizer
        self._lights = gs.List()
        self._renderer = None
        self._t = -1
        self._mesh_transform_idx = None
        self._scene_description = None

        # save renderer options
        self._renderer_options = renderer_options

    def add_light(self, pos, dir, color, intensity, directional, castshadow, cutoff, attenuation):
        self._lights.append(Light(pos, dir, color, intensity, directional, castshadow, cutoff, attenuation))

    def build(self):
        """
        Build all cameras in the batch and initialize Moderona renderer
        """

        if gs.backend != gs.cuda:
            gs.raise_exception("BatchRenderer requires CUDA backend.")
        gpu_id = gs.device.index if gs.device.index is not None else 0

        # Extract the complete list of non-debug cameras
        self._cameras = gs.List([camera for camera in self._visualizer._cameras if not camera.debug])
        if not self._cameras:
            gs.raise_exception("Please add at least one camera when using BatchRender.")

        # Throw exception when there is no light
        if not self._lights:
            gs.raise_exception("Please add at least one light when using ApolloRenderer.")

        # Export the scene description and load it into the renderer
        self._scene_description = SceneDescription()
        self._scene_description.generate_from_scene(self._visualizer.scene)
        scene_description = self._scene_description.export_to_json_str()
        self._scene_description.export_to_file(self._renderer_options.scene_description_export_path)
        max_resolution = _get_max_camera_resolution(self._cameras)
        self._renderer = ApolloRendererImpl(
            self._renderer_options.app_mode,
            self._renderer_options.render_mode,
            self._renderer_options.debug_view,
            self._renderer_options.max_pt_depth,
            (
                max_resolution
                if self._renderer_options.app_mode == "batch_render"
                else self._renderer_options.window_size
            ),
        )
        self._renderer.load_scene_data(scene_description)
        self._mesh_transform_idx = self._scene_description._generate_mesh_transform_idx()

        self._is_mjcf_vgeom = torch.tensor(
            [isinstance(vgeom.entity.morph, gs.morphs.MJCF) for vgeom in self._visualizer.scene.rigid_solver.vgeoms],
            dtype=torch.bool,
            device=gs.device,
        )

        self._mjcf_link_indices = torch.tensor(
            [vgeom.link.idx for vgeom in self._visualizer.scene.rigid_solver.vgeoms], dtype=torch.long, device=gs.device
        ).unsqueeze(1)

        self._convert_to_y_up_list = [
            not isinstance(entity.morph, gs.morphs.Primitive)
            for entity in self._visualizer.scene.entities
            for vgeom in entity.vgeoms
        ]

    def update_scene(self):
        self._visualizer._context.update()

    def render(self, camera_index):
        """
        Render with the Apollo renderer, which currently doesn't support batch rendering.

        Returns
        -------
        rgb_arr : tuple of arrays
            The sequence of rgb images associated with each camera.
        """

        self._t = self._visualizer.scene.t

        # Update scene
        self.update_scene()

        # Capture animation
        if self._renderer_options.capture_animation:
            self._scene_description.capture_frame()

        # Render
        self._renderer.update(self._t)
        rgb = self._renderer.render(
            camera_index,
            *self._get_geom_pos_quat_tensor(self._visualizer.scene, self._mesh_transform_idx),
            *self._get_camera_pos_quat_numpy(self._cameras),
        )
        return rgb, None, None, None

    def destroy(self):
        # Only need to export scene description, with animation if capture animation is enabled
        if self._renderer_options.capture_animation:
            self._scene_description.export_to_file(self._renderer_options.scene_description_export_path)

        # Clear lights
        self._lights.clear()

        # Nuke renderer
        if self._renderer is not None:
            self._renderer.unload_scene()
            self._renderer.destroy()
            del self._renderer
            self._renderer = None

    def reset(self):
        self._t = -1

    # Helpers
    def _overwrite_mjcf_vgeoms_transforms(self, scene, geom_pos, geom_quat):
        # Overwrite positions and quaternions for MJCF vgeoms using torch operations
        if self._is_mjcf_vgeom.any():
            # Get link states (raw transforms, before y-up conversion)
            links_state_pos = scene.rigid_solver.get_links_pos()
            links_state_quat = scene.rigid_solver.get_links_quat()

            # Get the positions and quaternions from link states for MJCF vgeoms
            mjcf_link_pos = links_state_pos[self._mjcf_link_indices]
            mjcf_link_quat = links_state_quat[self._mjcf_link_indices]

            # Overwrite the positions and quaternions for MJCF vgeoms
            # Note: coordinate system conversion will be applied later in _get_geom_pos_quat_tensor
            geom_pos[self._is_mjcf_vgeom] = mjcf_link_pos[self._is_mjcf_vgeom]
            geom_quat[self._is_mjcf_vgeom] = mjcf_link_quat[self._is_mjcf_vgeom]

        return geom_pos, geom_quat

    def _get_geom_pos_quat_tensor(self, scene, idx):
        # Initial transforms
        geom_pos = ti_to_torch(scene.rigid_solver.vgeoms_state.pos)
        geom_quat = ti_to_torch(scene.rigid_solver.vgeoms_state.quat)

        # Overwrite transforms of mjcf vgeoms
        geom_pos, geom_quat = self._overwrite_mjcf_vgeoms_transforms(scene, geom_pos, geom_quat)

        # Convert to y-up
        geom_pos = _pos_to_y_up(geom_pos)
        geom_pos = geom_pos.transpose(0, 1)
        geom_quat = _quat_to_y_up(geom_quat, self._convert_to_y_up_list)
        geom_quat = geom_quat.transpose(0, 1)

        # # Select transforms, by merging transforms of entities that don't expand in scene description
        geom_pos = torch.index_select(geom_pos, -2, idx).contiguous()
        geom_quat = torch.index_select(geom_quat, -2, idx).contiguous()

        return geom_pos, geom_quat

    def _get_geom_pos_quat_numpy(self, scene, idx):
        geom_pos, geom_quat = self._get_geom_pos_quat_tensor(scene, idx)
        geom_pos = geom_pos.cpu().numpy()
        geom_quat = geom_quat.cpu().numpy()
        return geom_pos, geom_quat

    def _get_camera_pos_quat_tensor(self, cameras):
        camera_pos = torch.stack([camera.get_pos() for camera in cameras])
        camera_pos = _pos_to_y_up(camera_pos)
        camera_quat = torch.stack([camera.get_quat() for camera in cameras])
        camera_quat = _quat_to_y_up(camera_quat)
        return camera_pos, camera_quat

    def _get_camera_pos_quat_numpy(self, cameras):
        camera_pos, camera_quat = self._get_camera_pos_quat_tensor(cameras)
        camera_pos = camera_pos.cpu().numpy()
        camera_quat = camera_quat.cpu().numpy()
        return camera_pos, camera_quat

    @property
    def lights(self):
        return self._lights

    @property
    def cameras(self):
        return self._cameras
