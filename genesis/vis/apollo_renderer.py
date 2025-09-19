import math

import numpy as np
import torch

import genesis as gs
from genesis.repr_base import RBC
from genesis.constants import IMAGE_TYPE
from genesis.utils.misc import ti_to_torch
from genesis.utils.scene_exporter import SceneDescriptionExporter

try:
    from gs_apollo import ApolloRenderer as ApolloRendererImpl
except ImportError as e:
    gs.raise_exception_from("Failed to import Apollo renderer.", e)


def _transform_camera_quat(quat):
    # quat for Madrona needs to be transformed to y-forward
    w, x, y, z = torch.unbind(quat, dim=-1)
    return torch.stack([x + w, x - w, y - z, y + z], dim=-1) / math.sqrt(2.0)


def _make_tensor(data, *, dtype: torch.dtype = torch.float32):
    return torch.tensor(data, dtype=dtype, device=gs.device)


# Helper functions
def _wxyz_to_xyzw(wxyz):
    if isinstance(wxyz, tuple):
        return (wxyz[1], wxyz[2], wxyz[3], wxyz[0])
    else:
        # Handle multi-dimensional tensors by indexing along the last dimension
        return torch.stack([wxyz[..., 1], wxyz[..., 2], wxyz[..., 3], wxyz[..., 0]], dim=-1)


def pos_to_y_up(pos):
    # Swizzle to (X, Z, -Y)
    if isinstance(pos, tuple):
        return (pos[0], pos[2], -pos[1])
    else:
        return torch.stack([pos[..., 0], pos[..., 2], -pos[..., 1]], dim=-1)


def quat_to_y_up(quat, convert_to_y_up):
    if isinstance(quat, tuple):
        assert isinstance(convert_to_y_up, bool), f"convert_to_y_up must be a single boolean if quat is a tuple"
        if convert_to_y_up:
            x, y, z, w = quat
            divisor = math.sqrt(2.0)
            return _wxyz_to_xyzw(((x + w) / divisor, (x - w) / divisor, (z + y) / divisor, (z - y) / divisor))
        else:
            return _wxyz_to_xyzw(quat)
    elif quat.ndim == 1:
        assert isinstance(convert_to_y_up, bool), f"convert_to_y_up must be a single boolean if quat is 1D"
        if convert_to_y_up:
            w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
            # This is the same as transforming the quat with [0.7071068, -0.7071068, 0, 0]
            return _wxyz_to_xyzw(torch.stack([x + w, x - w, z + y, z - y], dim=-1) / math.sqrt(2.0))
        else:
            return _wxyz_to_xyzw(quat)
    else:
        convert_to_y_up_mask = torch.tensor(convert_to_y_up, dtype=torch.bool, device=quat.device)

        # Assert that the batch dimension matches
        assert (
            len(convert_to_y_up_mask) == quat.shape[0]
        ), f"convert_to_y_up_mask length ({len(convert_to_y_up_mask)}) must match quat batch dimension ({quat.shape[0]})"

        # Apply transformation only where convert_to_y_up is True
        result = quat.clone()

        # Apply transformation to all positions to convert to y-up at once
        if convert_to_y_up_mask.any():
            # Get indices where transformation should be applied
            convert_to_y_up_indices = torch.where(convert_to_y_up_mask)[0]

            # Apply transformation to all positions to convert to y-up
            w = quat[convert_to_y_up_indices, ..., 0]
            x = quat[convert_to_y_up_indices, ..., 1]
            y = quat[convert_to_y_up_indices, ..., 2]
            z = quat[convert_to_y_up_indices, ..., 3]
            result[convert_to_y_up_indices, :] = torch.stack([x + w, x - w, z + y, z - y], dim=-1) / math.sqrt(2.0)

        return _wxyz_to_xyzw(result)


def _get_geom_pos_quat_tensor(scene):
    geom_pos = pos_to_y_up(ti_to_torch(scene.rigid_solver.vgeoms_state.pos))
    geom_pos = geom_pos.transpose(0, 1).contiguous()
    convert_to_y_up_list = [not isinstance(entity.morph, gs.morphs.Primitive) for entity in scene.entities]
    geom_quat = quat_to_y_up(ti_to_torch(scene.rigid_solver.vgeoms_state.quat), convert_to_y_up_list)
    geom_quat = geom_quat.transpose(0, 1).contiguous()
    return geom_pos, geom_quat


def _get_geom_pos_quat_numpy(scene):
    geom_pos, geom_quat = _get_geom_pos_quat_tensor(scene)
    geom_pos = geom_pos.cpu().numpy()
    geom_quat = geom_quat.cpu().numpy()
    return geom_pos, geom_quat


def get_cuda_device_uuid():
    cuda_device_index = torch.cuda.current_device()
    cuda_device_uuid_bytes = bytes(torch.cuda.get_device_properties(cuda_device_index).uuid.bytes)
    return cuda_device_uuid_bytes


class Light:
    def __init__(self, pos, dir, color, intensity, directional, castshadow, cutoff, attenuation):
        self._pos = pos
        self._dir = tuple(dir / np.linalg.norm(dir))
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

        # Export the scene description and load it into the renderer
        scene_exporter = SceneDescriptionExporter(self._visualizer.scene)
        scene_description = scene_exporter.export_to_json_str()
        scene_exporter.export_to_file("scene_output/demo_with_apollo.json")
        self._renderer = ApolloRendererImpl()
        self._renderer.load_scene_data(scene_description)

    def update_scene(self):
        self._visualizer._context.update()

    def render(self):
        """
        Render with the Apollo renderer, which currently doesn't support batch rendering.

        Returns
        -------
        rgb_arr : tuple of arrays
            The sequence of rgb images associated with each camera.
        """

        self._t = self._visualizer.scene.t
        self._renderer.update(self._t)
        rgb = self._renderer.render(*_get_geom_pos_quat_numpy(self._visualizer.scene))
        return rgb, None, None, None

    def destroy(self):
        self._lights.clear()
        if self._renderer is not None:
            self._renderer.unload_scene()
            self._renderer.destroy()
            del self._renderer
            self._renderer = None

    def reset(self):
        self._t = -1

    @property
    def lights(self):
        return self._lights

    @property
    def cameras(self):
        return self._cameras
