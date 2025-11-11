import json
import os
import math
import random
import base64
from enum import Enum
import xml.etree.ElementTree as ET
from typing import Dict, List
import copy

import numpy as np
import torch
import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import ti_to_torch
from trimesh.visual.color import ColorVisuals

CURRENT_SCENE_DESCRIPTION_VERSION = 1


class RendererType(Enum):
    RASTERIZER = "rasterizer"
    APOLLO = "apollo"
    BATCH_RENDERER = "batch_renderer"
    RAYTRACER = "raytracer"


class ElementType(Enum):
    RIGID_ENTITY = "mesh_entities"
    CAMERA = "camera_entities"
    LIGHT = "light_entities"
    MATERIAL = "materials"
    SURFACE = "surfaces"


MORPH_TYPE_TO_CLASS = {
    "mesh": gs.morphs.Mesh,
    "urdf": gs.morphs.URDF,
    "mjcf": gs.morphs.MJCF,
    "box": gs.morphs.Box,
    "cylinder": gs.morphs.Cylinder,
    "capsule": gs.morphs.Cylinder,  # FIXME: Genesis does not support capsule yet.
    "sphere": gs.morphs.Sphere,
    "plane": gs.morphs.Plane,
    "terrain": gs.morphs.Terrain,
}

MATERIAL_TYPE_TO_CLASS = {
    "rigid": gs.materials.Rigid,
}

SURFACE_TYPE_TO_CLASS = {
    "glass": gs.surfaces.Glass,
    "plastic": gs.surfaces.Plastic,
    "metal": gs.surfaces.Metal,
    "bsdf": gs.surfaces.BSDF,
    "emission": gs.surfaces.Emission,
    "default": gs.surfaces.Default,
    "rough": gs.surfaces.Rough,
    "smooth": gs.surfaces.Smooth,
    "reflective": gs.surfaces.Reflective,
    "collision": gs.surfaces.Collision,
    "water": gs.surfaces.Water,
    "iron": gs.surfaces.Iron,
    "aluminum": gs.surfaces.Aluminium,
    "rough": gs.surfaces.Rough,
    "copper": gs.surfaces.Copper,
    "gold": gs.surfaces.Gold,
}

RENDERER_TYPE_TO_CLASS = {
    "rasterizer": gs.options.renderers.Rasterizer,
    "apollo": gs.options.renderers.ApolloRenderer,
    "batch_renderer": gs.options.renderers.BatchRenderer,
    "raytracer": gs.options.renderers.RayTracer,
}

SURFACE_PROPERTIES = [
    "ior",
    "double_sided",
    "subsurface",
    "metal_type",
]

# Texture properties
SURFACE_TEXTURES = [
    ("diffuse_texture", "color"),
    ("opacity_texture", "opacity"),
    ("roughness_texture", "roughness"),
    ("metallic_texture", "metallic"),
    ("emissive_texture", "emissive"),
    ("thickness_texture", "thickness"),
    ("normal_texture", "normal"),
    ("specular_texture", "specular"),
    ("transmission_texture", "transmission"),
]


# Helper functions
def _make_tensor(data, *, dtype: torch.dtype = torch.float32):
    return torch.tensor(data, dtype=dtype, device=gs.device)


def _wxyz_to_xyzw(wxyz):
    # Handle multi-dimensional tensors by indexing along the last dimension
    return torch.stack([wxyz[..., 1], wxyz[..., 2], wxyz[..., 3], wxyz[..., 0]], dim=-1)


def _xyzw_to_wxyz(xyzw):
    # Handle multi-dimensional tensors by indexing along the last dimension
    return torch.stack([xyzw[..., 3], xyzw[..., 0], xyzw[..., 1], xyzw[..., 2]], dim=-1)


def _pos_to_y_up(pos) -> torch.Tensor:
    # Swizzle to (X, Z, -Y)
    pos = torch.as_tensor(pos)
    return torch.stack([pos[..., 0], pos[..., 2], -pos[..., 1]], dim=-1)


def _pos_from_y_up(pos) -> torch.Tensor:
    # Swizzle to (X, -Z, Y)
    pos = torch.as_tensor(pos)
    return torch.stack([pos[..., 0], -pos[..., 2], pos[..., 1]], dim=-1)


def _quat_to_y_up(quat, y_up: bool = True) -> torch.Tensor:
    """
    Convert z-up quaternion to y-up by left-multiplying -90° about X.
    Input is interpreted as (w, x, y, z) and returned as (x, y, z, w).
    """
    # Handle tuple first (so the branch is reachable)
    quat = torch.as_tensor(quat)
    if quat.ndim == 1:
        assert isinstance(y_up, bool), "y_up must be a single boolean if quat is 1D"
        if y_up:
            w, x, y, z = quat
            quat = torch.stack([x + w, x - w, z + y, z - y], dim=-1) / math.sqrt(2.0)  # WXYZ
        return _wxyz_to_xyzw(quat)
    else:
        mask = torch.as_tensor(y_up, dtype=torch.bool, device=quat.device)
        assert len(mask) == quat.shape[0], f"y_up shape ({len(mask)}) must match quat batch dimension ({quat.shape[0]})"
        result = quat.clone()
        if mask.any():
            idx = torch.where(mask)[0]
            w = quat[idx, ..., 0]
            x = quat[idx, ..., 1]
            y = quat[idx, ..., 2]
            z = quat[idx, ..., 3]
            result[idx, :] = torch.stack([x + w, x - w, z + y, z - y], dim=-1) / math.sqrt(2.0)
        return _wxyz_to_xyzw(result)


def _quat_from_y_up(quat, y_up: bool = True) -> torch.Tensor:
    """
    Inverse of _quat_to_y_up: undo the -90° about X (i.e., apply +90° about X).
    Input is interpreted as (x, y, z, w) and returned as (w, x, y, z).
    """
    quat = torch.as_tensor(quat)
    if quat.ndim == 1:
        assert isinstance(y_up, bool), "_y_up must be a single boolean if quat is 1D"
        if y_up:
            x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
            quat = torch.stack([(w + x), (y - z), (y + z), (w - x)], dim=-1) / math.sqrt(2.0)  # XYZW
        return _xyzw_to_wxyz(quat)
    else:
        mask = torch.as_tensor(y_up, dtype=torch.bool, device=quat.device)
        assert len(mask) == quat.shape[0], f"y_up shape ({len(mask)}) must match quat batch dimension ({quat.shape[0]})"
        result = quat.clone()
        if mask.any():
            idx = torch.where(mask)[0]
            w = quat[idx, ..., 0]
            x = quat[idx, ..., 1]
            y = quat[idx, ..., 2]
            z = quat[idx, ..., 3]
            result[idx, :] = torch.stack([(w + x), (y - z), (y + z), (w - x)], dim=-1) / math.sqrt(2.0)
        return _xyzw_to_wxyz(result)


def _serialize_transforms(transforms: torch.Tensor) -> str:
    # Flatten and then encode in base64
    return base64.b64encode(transforms.cpu().numpy().flatten().tobytes()).decode("utf-8")


def _deserialize_transforms(b64: str, dtype=np.float32, shape=(-1, 3), device=None) -> torch.Tensor:
    buf = base64.b64decode(b64)
    arr = np.frombuffer(buf, dtype=dtype).copy()  # copy -> writable, avoids view-on-bytes issues
    arr = arr.reshape(shape)
    t = torch.from_numpy(arr)
    return t.to(device) if device is not None else t


def _get_basename_no_extension(path):
    return os.path.splitext(os.path.basename(path))[0]


class SceneDescription:
    # Geometry type to string mapping
    def __init__(self):
        self._scene = None
        self._json_content = None
        self._asset_dir = None

    def generate_from_scene(
        self,
        scene: gs.Scene,
        names: Dict[ElementType, list[str]] = None,
        asset_root_path: str = None,
    ):
        """
        Generate a scene description from a scene.

        Parameters
        ----------
        scene : gs.Scene
            The scene to generate a scene description from.
        names : Dict[ElementType, list[str]], optional
            The names of the objects (entities, cameras, lights...) in the scene.
            If not provided, the names will be generated automatically.
        asset_root_path : str, optional
            The root path of the assets.
            If not provided, the assets will be loaded from the default assets directory.
        """
        assert scene.is_built, "Scene must be built before generating scene description"
        self._scene = scene
        self._json_content = dict()
        self._asset_dir = gs.utils.get_assets_dir() if asset_root_path is None else asset_root_path
        self._generate_scene_desc(names)

    def load_from_file(self, file_path: str, build_scene: bool = True):
        """
        Load a scene description from a file.

        Parameters
        ----------
        file_path : str
            The path to the scene description file.
        build_scene : bool, optional
            Whether to instantly build the scene based on the scene description.
            If False, you need to use the returned initial arguments to build the scene.

        Returns
        -------
        names : Dict[ElementType, list[str]]
            The names of the objects (entities, cameras, lights...) in the scene.
        init_args : Dict[ElementType, dict]
            The initialization arguments for the objects (entities, cameras, lights...) in the scene.
        """
        if not os.path.exists(file_path):
            file_path = os.path.join(gs.utils.get_assets_dir(), file_path)

        with open(file_path, "r") as f:
            self._json_content = json.load(f)

        asset_dir = self._json_content.get("asset_root_path", ".")
        if os.path.isabs(asset_dir):
            self._asset_dir = asset_dir
        else:
            self._asset_dir = os.path.abspath(os.path.join(os.path.dirname(file_path), asset_dir))
        return self._load_scene_desc(build_scene)

    def capture_frame(self):
        """
        Capture the current frame from the scene.
        """
        frame = {
            "mesh_transforms": self._capture_entity_desc(),
            "camera_transforms": self._capture_camera_desc(),
        }

        frame_idx = self._scene.t
        animation_idx = self._get_animation_idx(frame_idx)
        if animation_idx is None:
            assert (
                not self._json_content["animation_frame"] or self._json_content["animation_frame"][-1] <= frame_idx
            ), "Frame index must be monotonically increasing"
            self._json_content["animation_frame"].append(frame_idx)
            self._json_content["scene_animation"].append(frame)
        else:
            self._json_content["scene_animation"][animation_idx] = frame

    def remove_frame(self, frame_idx: int):
        """
        Remove a frame from the scene description.

        Parameters
        ----------
        frame_idx : int
            The timestamp of the frame to remove.
        """
        animation_idx = self._get_animation_idx(frame_idx)
        if animation_idx is not None:
            self._json_content["scene_animation"].pop(animation_idx)
            self._json_content["animation_frame"].pop(animation_idx)

    def load_frame(
        self,
        frame_idx: int = None,
        animation_idx: int = None,
        load_scene: bool = True,
    ) -> dict:
        """
        Load a frame from the scene description.

        Parameters
        ----------
        frame_idx : int, optional
            The timestamp of the frame to load.
        animation_idx : int, optional
            The index of the animation to load.
        load_scene : bool, optional
            Whether to instantly build the scene based on the scene description.
            If False, you need to use the returned frame data to restore the scene.

        Returns
        -------
        frame_data : dict
            The frame data.
        """
        if not self._json_content.get("scene_animation", []):
            return None

        if animation_idx is None:
            if frame_idx is not None:
                animation_idx = self._get_animation_idx(frame_idx)
                if animation_idx is None:
                    return None
            else:
                animation_idx = len(self._json_content["scene_animation"]) - 1

        frame = self._json_content["scene_animation"][animation_idx]
        capture_dict = {
            ElementType.RIGID_ENTITY: self._restore_entity_desc(frame["mesh_transforms"]),
            ElementType.CAMERA: self._restore_camera_desc(frame["camera_transforms"]),
        }

        if load_scene:
            entities = self._scene.rigid_solver.entities
            entities_dict = self._json_content[ElementType.RIGID_ENTITY.value]
            for i, entity in enumerate(entities):
                entity_name = entities_dict[i]["name"]
                entity_frame = capture_dict[ElementType.RIGID_ENTITY][entity_name]
                entity.set_pos(entity_frame["pos"])
                entity.set_quat(entity_frame["quat"])
                if "qpos" in entity_frame:
                    entity.set_qpos(entity_frame["qpos"])

            cameras = [camera for camera in self._scene.visualizer.cameras if not camera.debug]
            cameras_dict = self._json_content[ElementType.CAMERA.value]
            for i, camera in enumerate(cameras):
                camera_name = cameras_dict[i]["name"]
                camera_frame = capture_dict[ElementType.CAMERA][camera_name]
                camera.set_pose(pos=camera_frame["pos"], lookat=camera_frame["lookat"])

        return capture_dict

    def export_to_file(self, export_path: str):
        """
        Export the scene description to a file.

        Parameters
        ----------
        export_path : str
            The path to the file to export the scene description to.
        """
        dir_path = os.path.dirname(export_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(export_path, "w") as f:
            json.dump(self._json_content, f, indent=4)

    def export_to_json_str(self) -> str:
        """
        Export the scene description to a JSON string.

        Returns
        -------
        scene_description : str
            The JSON string of the scene description.
        """
        return json.dumps(self._json_content, indent=4)

    def _get_animation_idx(self, frame_idx: int) -> int | None:
        return next(
            (aidx for aidx, fidx in enumerate(self._json_content["animation_frame"]) if fidx == frame_idx), None
        )

    def _get_rel_asset_path(self, path: str) -> str:
        return os.path.relpath(path, self._asset_dir)

    def _get_abs_asset_path(self, path: str) -> str:
        return os.path.abspath(os.path.join(self._asset_dir, path))

    def _generate_scene_desc(self, names: Dict[ElementType, list[str]] = None):
        # headers
        assert (
            self._scene.n_envs <= 1
        ), "Scene must have only one environment now."  # TODO: Support multiple environments
        self._json_content = {
            "version": CURRENT_SCENE_DESCRIPTION_VERSION,
            "num_environments": self._scene.n_envs,
            "asset_root_path": self._asset_dir,
            "frame_time": self._scene.sim_options.dt,
            "renderer": {
                "type": RendererType.RASTERIZER.value,
            },
            ElementType.RIGID_ENTITY.value: [],
            ElementType.CAMERA.value: [],
            ElementType.LIGHT.value: [],
            "scene_animation": [],
            "animation_frame": [],
        }

        # mesh entities
        for i, entity in enumerate(self._scene.rigid_solver.entities):
            entity_name = (
                self._generate_entity_name(entity.morph, i) if names is None else names[ElementType.RIGID_ENTITY][i]
            )
            self._json_content[ElementType.RIGID_ENTITY.value].append(self._generate_entity_desc(entity, entity_name))

        # camera entities
        for i, camera in enumerate(self._scene.visualizer.cameras):
            camera_name = self._generate_camera_name(camera, i) if names is None else names[ElementType.CAMERA][i]
            if not camera.debug:
                self._json_content[ElementType.CAMERA.value].append(self._generate_camera_desc(camera, camera_name))

        # light entities
        if hasattr(self._scene.visualizer, "apollo_renderer") and self._scene.visualizer.apollo_renderer is not None:
            for light in self._scene.visualizer.apollo_renderer.lights:
                self._json_content[ElementType.LIGHT.value].append(self._generate_light_desc(light))

        # environment map
        # TODO: Support environment map for Apollo renderer
        if self._scene.visualizer.raytracer is not None:
            self._json_content[ElementType.LIGHT.value].append(self._generate_environment_map_desc())

    def _load_scene_desc(self, build_scene: bool = True):
        init_args = {
            ElementType.RIGID_ENTITY: {},
            ElementType.CAMERA: {},
            ElementType.SURFACE: {},
        }

        renderer_options = self._load_render_options()
        # mesh entities
        entities_dict = self._json_content.get(ElementType.RIGID_ENTITY.value, {})
        entities_name = []
        surfaces_name = []
        for i, entity_dict in enumerate(entities_dict):
            if "name" not in entity_dict:
                entity_name = self._load_entity_name(entity_dict, i)
                entity_dict["name"] = entity_name
            entity_name = entity_dict["name"]
            surface_name = f"{entity_name}_surface"
            entities_name.append(entity_name)
            surfaces_name.append(surface_name)
            entity_args, surface = self._load_entity_desc(entity_dict)
            init_args[ElementType.RIGID_ENTITY][entity_name] = entity_args
            init_args[ElementType.SURFACE][surface_name] = surface

        # camera entities
        cameras_dict = self._json_content.get(ElementType.CAMERA.value, {})
        cameras_name = []
        for i, camera_dict in enumerate(cameras_dict):
            if "name" not in camera_dict:
                camera_name = self._load_camera_name(camera_dict, i)
                camera_dict["name"] = camera_name
            camera_name = camera_dict["name"]
            cameras_name.append(camera_name)
            camera_args = self._load_camera_desc(camera_dict)
            init_args[ElementType.CAMERA][camera_name] = camera_args

        # light entities
        lights_dict = self._json_content.get(ElementType.LIGHT.value, {})
        lights_name = []

        names = {
            ElementType.RIGID_ENTITY: entities_name,
            ElementType.CAMERA: cameras_name,
            ElementType.LIGHT: lights_name,
            ElementType.SURFACE: surfaces_name,
        }

        if build_scene:
            sim_args = {}
            if "frame_time" in self._json_content:
                sim_args["dt"] = self._json_content.get("frame_time")
            self._scene = gs.Scene(
                sim_options=gs.options.SimOptions(**sim_args),
                rigid_options=gs.options.RigidOptions(),
                renderer=renderer_options,
                show_viewer=False,
            )

            for entity_name in entities_name:
                entity_args = copy.deepcopy(init_args[ElementType.RIGID_ENTITY][entity_name])
                surface_name = entity_args["surface"]
                surface_args = init_args[ElementType.SURFACE][surface_name]
                entity_args["surface"] = surface_args
                self._scene.add_entity(**entity_args)

            for camera_name in cameras_name:
                camera_args = init_args[ElementType.CAMERA][camera_name]
                self._scene.add_camera(**camera_args)
            self._scene.build(n_envs=self._json_content.get("num_environments"))

        return names, init_args

    def _load_render_options(self) -> gs.options.renderers.RendererOptions:
        renderer_dict = self._json_content.get("renderer", {})
        renderer_type = RendererType(renderer_dict.get("type", RendererType.RASTERIZER.value))
        if renderer_type == RendererType.RASTERIZER:
            renderer_options = gs.options.renderers.Rasterizer()
        elif renderer_type == RendererType.APOLLO:
            renderer_options = gs.options.renderers.ApolloRenderer()
        elif renderer_type == RendererType.BATCH_RENDERER:
            renderer_options = gs.options.renderers.BatchRenderer()
        elif renderer_type == RendererType.RAYTRACER:
            renderer_options = gs.options.renderers.RayTracer()
        else:
            raise ValueError(f"Invalid renderer type: {renderer_type}")
        return renderer_options

    # Meshes
    def _should_export_at_geom_level(self, morph: gs.morphs.Morph) -> bool:
        return not isinstance(morph, (gs.morphs.Mesh, gs.morphs.Primitive))

    def _generate_entity_desc(self, entity, entity_name: str):
        entity_type = self._generate_entity_type(entity.morph)
        entity_surface = self._generate_surface(entity.surface, isinstance(entity.morph, gs.morphs.Plane))
        convert_to_y_up = isinstance(entity.morph, gs.morphs.Mesh)

        pos = entity.morph.pos
        entity_dict = {
            "name": entity_name,
            "entity_type": entity_type,
            "position": _pos_to_y_up(pos).tolist(),
            "rotation": _quat_to_y_up(entity.morph.quat, convert_to_y_up).tolist(),
            "qpos": entity.init_qpos.tolist(),
            "scale": self._generate_entity_scale(entity.morph),
            "collision": entity.morph.collision,
            "visualization": entity.morph.visualization,
            "material_override": entity_surface,
        }
        if not isinstance(entity.morph, gs.morphs.MJCF):
            entity_dict["fixed"] = entity.morph.fixed

        if isinstance(entity.morph, gs.morphs.FileMorph):
            file_path = entity.morph.file
            uri = self._get_rel_asset_path(file_path)
            entity_dict["uri"] = uri
            if self._should_export_at_geom_level(entity.morph):
                file_ext = os.path.splitext(file_path)[1].lower()
                if file_ext == ".urdf":
                    output_ext = ".jurdf"
                elif file_ext == ".xml":
                    output_ext = ".jxml"
                else:
                    raise ValueError(f"Unsupported file type: {file_ext} ")
                converted_uri = uri.replace(file_ext, output_ext)
                converted_file_path = file_path.replace(file_ext, output_ext)
                entity_dict["converted_uri"] = converted_uri

                gs.logger.warning(f"Exporting geoms to JSON file: {converted_file_path}")
                if not os.path.exists(converted_file_path):
                    self._export_geoms_to_json(entity, entity_dict, converted_file_path)

        entity_dict.update(self._generate_entity_extra_desc(entity.morph))

        return entity_dict

    def _load_entity_desc(self, entity_dict: dict) -> dict:
        morph_class = self._load_entity_morph(entity_dict)
        convert_to_y_up = issubclass(morph_class, gs.morphs.Mesh)

        entity_quat = _quat_from_y_up(entity_dict.get("rotation", [0.0, 0.0, 0.0, 1.0]), convert_to_y_up).tolist()
        morph_args = {
            "pos": _pos_from_y_up(entity_dict.get("position", [0.0, 0.0, 0.0])).tolist(),
            "quat": entity_quat,
            "collision": entity_dict.get("collision", True),
            "visualization": entity_dict.get("visualization", True),
        }
        if morph_class not in [gs.morphs.MJCF, gs.morphs.Plane]:
            morph_args["fixed"] = entity_dict.get("fixed", False)
        if issubclass(morph_class, gs.morphs.FileMorph):
            morph_args["file"] = self._get_abs_asset_path(entity_dict.get("uri"))
        morph_args.update(self._load_entity_scale(entity_dict))

        surface = self._load_surface(
            entity_dict.get("material_override", {})
        )  # FIXME: primitive in Genesis does not have uv now.
        entity_name = entity_dict.get("name")

        entity_args = {"morph": morph_class(**morph_args), "surface": f"{entity_name}_surface"}
        return entity_args, surface

    def _generate_mesh_transform_idx(self) -> list[int]:
        mesh_transform_idx = []
        entity_start_idx = 0
        for entity in self._scene.rigid_solver.entities:
            if self._should_export_at_geom_level(entity.morph):
                mesh_transform_idx += list(range(entity_start_idx, entity_start_idx + entity.n_vgeoms))
            else:
                mesh_transform_idx += [entity_start_idx]
            entity_start_idx += entity.n_vgeoms
        return _make_tensor(mesh_transform_idx, dtype=torch.int)

    def _capture_entity_desc(self) -> dict:
        if not self._scene.rigid_solver.is_active:
            return {}

        vgeoms_state_pos = ti_to_torch(self._scene.rigid_solver.vgeoms_state.pos).squeeze(1)
        vgeoms_state_quat = ti_to_torch(self._scene.rigid_solver.vgeoms_state.quat).squeeze(1)
        links_state_pos = ti_to_torch(self._scene.rigid_solver.links_state.pos).squeeze(1)
        links_state_quat = ti_to_torch(self._scene.rigid_solver.links_state.quat).squeeze(1)

        base_links_idx = _make_tensor(
            [entity.base_link_idx for entity in self._scene.rigid_solver.entities], dtype=torch.int
        )
        entity_pos = links_state_pos[base_links_idx]
        entity_quat = links_state_quat[base_links_idx]
        entity_qpos = ti_to_torch(self._scene.rigid_solver.qpos).squeeze(1)

        # Evaluate index mask
        geoms_pos = vgeoms_state_pos.clone()
        geoms_quat = vgeoms_state_quat.clone()
        is_mjcf_vgeom = _make_tensor(
            [isinstance(vgeom.entity.morph, gs.morphs.MJCF) for vgeom in self._scene.rigid_solver.vgeoms],
            dtype=torch.bool,
        )

        if is_mjcf_vgeom.any():
            # Collect indices for vgeoms that are MJCF
            mjcf_indices = torch.where(is_mjcf_vgeom)[0]
            link_indices = _make_tensor([vgeom.link.idx for vgeom in self._scene.rigid_solver.vgeoms], dtype=torch.int)
            link_indices = link_indices[mjcf_indices]
            geoms_pos[mjcf_indices] = links_state_pos[link_indices]
            geoms_quat[mjcf_indices] = links_state_quat[link_indices]

        # Convert to y-up
        convert_to_y_up_list = []
        for entity in self._scene.rigid_solver.entities:
            if isinstance(entity.morph, gs.morphs.Primitive):
                convert_to_y_up_list.append(False)
            else:
                n_vgeoms = len(entity.vgeoms)
                convert_to_y_up_list.extend([True] * n_vgeoms)
        convert_to_y_up_list = np.array(convert_to_y_up_list)

        geoms_pos = _pos_to_y_up(geoms_pos)
        geoms_quat = _quat_to_y_up(geoms_quat, convert_to_y_up_list)

        mesh_transform_idx = self._generate_mesh_transform_idx()
        geoms_pos = torch.index_select(geoms_pos, 0, mesh_transform_idx).contiguous()
        geoms_quat = torch.index_select(geoms_quat, 0, mesh_transform_idx).contiguous()

        transforms = {
            "pos": _serialize_transforms(geoms_pos),
            "quat": _serialize_transforms(geoms_quat),
            "entity_pos": _serialize_transforms(entity_pos),
            "entity_quat": _serialize_transforms(entity_quat),
            "entity_qpos": _serialize_transforms(entity_qpos),
        }
        return transforms

    def _restore_entity_desc(self, mesh_transforms) -> dict:
        entities_dict = self._json_content.get(ElementType.RIGID_ENTITY.value)
        entities_n_qs = [len(entity_dict.get("qpos")) for entity_dict in entities_dict]
        n_entities = len(entities_n_qs)
        n_qs = sum(entities_n_qs)
        if not n_entities:
            return {}

        entity_pos = _deserialize_transforms(mesh_transforms["entity_pos"], shape=(n_entities, -1, 3)).squeeze(1)
        entity_quat = _deserialize_transforms(mesh_transforms["entity_quat"], shape=(n_entities, -1, 4)).squeeze(1)
        entity_qpos = (
            _deserialize_transforms(mesh_transforms["entity_qpos"], shape=(n_qs, -1)).squeeze(1) if n_qs > 0 else None
        )

        entities_capture = {}
        q_start = 0
        for i, n_qs in enumerate(entities_n_qs):
            entity_name = entities_dict[i]["name"]
            entity_capture = {
                "pos": entity_pos[i].tolist(),
                "quat": entity_quat[i].tolist(),
            }
            if entity_qpos is not None:
                entity_capture["qpos"] = entity_qpos[q_start : q_start + n_qs].tolist()
            q_start += n_qs
            entities_capture[entity_name] = entity_capture

        return entities_capture

    def _generate_entity_extra_desc(self, morph: gs.morphs.Morph) -> dict:
        entity_dict = {}
        if isinstance(morph, gs.morphs.Plane):
            entity_dict["tiling"] = tuple(p / t for p, t in zip(morph.plane_size, morph.tile_size))
        return entity_dict

    def _generate_entity_type(self, morph: gs.morphs.Morph) -> str:
        if isinstance(morph, gs.morphs.FileMorph):
            if isinstance(morph, gs.morphs.Mesh):
                entity_type = gs.GEOM_TYPE.MESH.name.lower()
            else:
                entity_type = "group"
        elif isinstance(morph, gs.morphs.Primitive):
            entity_type = morph.__class__.__name__.lower()
        else:
            raise ValueError(f"Unknown entity type: {type(morph)}")
        return entity_type

    def _generate_entity_name(self, morph: gs.morphs.Morph, index: int) -> str:
        if isinstance(morph, gs.morphs.FileMorph):
            return _get_basename_no_extension(morph.file) + f"_{index}"
        else:
            return morph.__class__.__name__.lower() + f"_{index}"

    def _load_entity_name(self, entity_dict: dict, index: int) -> str:
        entity_class = self._load_entity_morph(entity_dict)
        if isinstance(entity_class, gs.morphs.FileMorph):
            return _get_basename_no_extension(entity_dict.get("uri")) + f"_{index}"
        else:
            return entity_class.__name__.lower() + f"_{index}"

    def _generate_vgeom_type_name(self, vgeom, entity_type: str) -> tuple[str, str]:
        if "mesh_path" in vgeom.metadata:
            vgeom_name = _get_basename_no_extension(vgeom.metadata["mesh_path"])
            vgeom_type = entity_type
        else:
            vgeom_name = vgeom_type = gs.GEOM_TYPE(vgeom.type).name.lower()
        return vgeom_type, vgeom_name

    def _generate_entity_scale(self, morph: gs.morphs.Morph) -> tuple:
        if isinstance(morph, gs.morphs.Primitive):
            if isinstance(morph, gs.morphs.Box):
                scale = (morph.size[0], morph.size[1], morph.size[2])
            elif isinstance(morph, gs.morphs.Sphere):
                scale = (morph.radius, morph.radius, morph.radius)
            elif isinstance(morph, gs.morphs.Cylinder):
                scale = (morph.radius, morph.radius, morph.height)
            elif isinstance(morph, gs.morphs.Plane):
                scale = (morph.plane_size[0], morph.plane_size[1], 1.0)
            else:
                scale = (1.0, 1.0, 1.0)
        elif isinstance(morph, gs.morphs.FileMorph):
            if isinstance(morph.scale, float):
                scale = (morph.scale, morph.scale, morph.scale)
            elif isinstance(morph.scale, tuple):
                scale = morph.scale
        else:
            scale = (1.0, 1.0, 1.0)

        # Swizzle y and z for now, until Apollo is z-up
        return (scale[0], scale[2], scale[1])

    def _load_entity_morph(self, entity_dict: dict) -> type[gs.morphs.Morph]:
        entity_type = entity_dict.get("entity_type")
        if entity_type == "group":
            uri = entity_dict.get("uri")
            if uri.endswith(".urdf"):
                morph_type = "urdf"
            elif uri.endswith(".xml"):
                morph_type = "mjcf"
            else:
                raise ValueError(f"Unknown uri format: {uri}")
        else:
            morph_type = entity_type
        return MORPH_TYPE_TO_CLASS[morph_type]

    def _load_entity_scale(self, entity_dict: dict) -> dict:
        args_dict = {}
        scale = entity_dict.get("scale", (1.0, 1.0, 1.0))
        scale = (scale[0], scale[2], scale[1])
        morph_class = self._load_entity_morph(entity_dict)

        if issubclass(morph_class, gs.morphs.Primitive):
            if issubclass(morph_class, gs.morphs.Box):
                args_dict["size"] = scale
            elif issubclass(morph_class, gs.morphs.Sphere):
                args_dict["radius"] = scale[0]
            elif issubclass(morph_class, gs.morphs.Cylinder):
                args_dict["radius"] = scale[0]
                args_dict["height"] = scale[2]
            elif issubclass(morph_class, gs.morphs.Plane):
                args_dict["plane_size"] = scale[:2]
            else:
                raise ValueError(f"Invalid morph type: {morph_class}")
        elif issubclass(morph_class, gs.morphs.FileMorph):
            if issubclass(morph_class, gs.morphs.Mesh):
                args_dict["scale"] = scale
            elif issubclass(morph_class, (gs.morphs.MJCF, gs.morphs.URDF)):
                args_dict["scale"] = scale[0]
            else:
                raise ValueError(f"Invalid morph type: {morph_class}")
        else:
            raise ValueError(f"Invalid morph type: {morph_class}")

        return args_dict

    def _generate_vgeom_scale(self, vgeom, entity_scale):
        vgeom_type = vgeom.type
        vgeom_data = vgeom.data
        if vgeom_type == gs.GEOM_TYPE.BOX:
            vgeom_scale = (vgeom_data[0], vgeom_data[1], vgeom_data[2])
        elif vgeom_type == gs.GEOM_TYPE.PLANE:
            vgeom_scale = (vgeom_data[3], vgeom_data[4], 1.0)
        elif vgeom_type == gs.GEOM_TYPE.CYLINDER:
            vgeom_scale = (vgeom_data[0], vgeom_data[1], vgeom_data[2])
        elif vgeom_type == gs.GEOM_TYPE.CAPSULE:
            vgeom_scale = (vgeom_data[0], vgeom_data[0], vgeom_data[1])
        elif vgeom_type == gs.GEOM_TYPE.SPHERE:
            vgeom_scale = (vgeom_data[0], vgeom_data[0], vgeom_data[0])
        else:
            vgeom_scale = entity_scale

        vgeom_scale = (vgeom_scale[0], vgeom_scale[2], vgeom_scale[1])
        return vgeom_scale

    def _generate_surface(self, surface: gs.surfaces.Surface, is_plane: bool) -> dict:
        surface_dict = {}
        surface_dict["surface_type"] = surface.__class__.__name__.lower()

        # Update all attributes
        for property_name in SURFACE_PROPERTIES:
            if hasattr(surface, property_name):
                property = getattr(surface, property_name)
                if property is not None:
                    surface_dict[property_name] = property

        for texture_name, color_name in SURFACE_TEXTURES:
            if hasattr(surface, texture_name):
                texture = getattr(surface, texture_name)
                if texture is not None:
                    if isinstance(texture, gs.textures.ImageTexture):
                        if texture.image_path is not None:
                            surface_dict[texture_name] = self._get_rel_asset_path(texture.image_path)
                            surface_dict[color_name] = texture.image_color
                        else:
                            surface_dict[color_name] = texture._mean_color.tolist()
                    elif isinstance(texture, gs.textures.ColorTexture):
                        surface_dict[color_name] = texture.color
                    if hasattr(surface_dict[color_name], "__len__") and len(surface_dict[color_name]) == 1:
                        surface_dict[color_name] = surface_dict[color_name][0]

        # Special case for plane
        if is_plane:
            if "diffuse_texture" not in surface_dict and "color" not in surface_dict:
                surface_dict["diffuse_texture"] = self._get_rel_asset_path(
                    os.path.join(gs.utils.get_assets_dir(), "textures/checker.png")
                )

        return surface_dict

    def _load_surface(self, surface_dict: dict) -> gs.surfaces.Surface:
        surface_class = SURFACE_TYPE_TO_CLASS[surface_dict.get("surface_type", "default")]

        surface_args = {}
        for property_name in SURFACE_PROPERTIES:
            if property_name in surface_dict:
                surface_args[property_name] = surface_dict[property_name]

        for texture_name, color_name in SURFACE_TEXTURES:
            if texture_name in surface_dict:
                surface_args[texture_name] = gs.textures.ImageTexture(
                    image_path=self._get_abs_asset_path(surface_dict[texture_name]),
                    image_color=surface_dict.get(color_name),
                )
            elif color_name in surface_dict:
                surface_args[texture_name] = gs.textures.ColorTexture(
                    color=surface_dict[color_name],
                )

        return surface_class(**surface_args)

    def _export_geoms_to_json(self, entity, entity_dict: dict, export_path: str):
        geoms_json_content = {
            "version": CURRENT_SCENE_DESCRIPTION_VERSION,
            "asset_root_path": self._asset_dir,
            ElementType.RIGID_ENTITY.value: [],
        }

        assert isinstance(entity.morph, gs.morphs.FileMorph), "Entity must be a FileMorph"
        entity_type = self._generate_entity_type(entity.morph)

        file_dir = None
        if isinstance(entity.morph, gs.morphs.MJCF):
            file_path = entity.morph.file
            xml_root = ET.parse(file_path).getroot()
            compiler = xml_root.find("compiler")
            meshdir = None
            assetdir = None
            if compiler is not None:
                meshdir = compiler.get("meshdir")
                assetdir = compiler.get("assetdir")
            file_dir = os.path.dirname(file_path)
            if meshdir is not None:
                file_dir = os.path.join(file_dir, meshdir)
            elif assetdir is not None:
                file_dir = os.path.join(file_dir, assetdir)

        is_mjcf = isinstance(entity.morph, gs.morphs.MJCF)
        vgeoms_state_pos = ti_to_torch(self._scene.rigid_solver.vgeoms_state.pos).squeeze(1)
        vgeoms_state_quat = ti_to_torch(self._scene.rigid_solver.vgeoms_state.quat).squeeze(1)
        links_state_pos = ti_to_torch(self._scene.rigid_solver.links_state.pos).squeeze(1)
        links_state_quat = ti_to_torch(self._scene.rigid_solver.links_state.quat).squeeze(1)

        for vgeom in entity.vgeoms:
            vgeom_dict = {}
            vgeom_type, vgeom_name = self._generate_vgeom_type_name(vgeom, entity_type)

            vgeom_dict["position"] = _pos_to_y_up(
                links_state_pos[vgeom.link.idx] if is_mjcf else vgeoms_state_pos[vgeom._idx]
            ).tolist()  # FIXME: Dynamic
            vgeom_dict["rotation"] = _quat_to_y_up(
                links_state_quat[vgeom.link.idx] if is_mjcf else vgeoms_state_quat[vgeom._idx], True
            ).tolist()  # FIXME: Dynamic
            vgeom_dict["scale"] = self._generate_vgeom_scale(vgeom, entity_dict.get("scale"))  # FIXME: Dynamic

            vgeom_surface = entity_dict.get("material_override").copy()
            if not "color" in vgeom_surface or not "opacity" in vgeom_surface:
                visual = vgeom.get_trimesh().visual
                if isinstance(visual, ColorVisuals):
                    geom_color = visual.main_color / 255.0
                    if not "color" in vgeom_surface:
                        vgeom_surface["color"] = geom_color[:3].tolist()
                    if not "opacity" in vgeom_surface:
                        vgeom_surface["opacity"] = geom_color[3]
            vgeom_dict["material_override"] = vgeom_surface  # FIXME: Dynamic

            vgeom_dict["entity_type"] = vgeom_type
            vgeom_dict["name"] = vgeom_name

            if "mesh_path" in vgeom.metadata:
                mesh_path = vgeom.metadata["mesh_path"]
                if file_dir is not None:
                    mesh_path = os.path.join(file_dir, mesh_path)
                vgeom_dict["uri"] = self._get_rel_asset_path(mesh_path)

            geoms_json_content["mesh_entities"].append(vgeom_dict)

        json.dump(geoms_json_content, open(export_path, "w"), indent=4)

    # Cameras
    def _generate_camera_desc(self, camera: gs.vis.camera.Camera, camera_name: str) -> dict:
        camera_pos = camera._initial_pos
        camera_lookat = camera._initial_lookat
        camera_transform = (
            camera._initial_transform
            if camera._initial_transform is not None
            else gu.pos_lookat_up_to_T(camera_pos, camera_lookat, camera._initial_up)
        )
        camera_quat = gu.R_to_quat(camera_transform[:3, :3])

        camera_dict = {
            "name": camera_name,
            "model": camera.model,
            "position": _pos_to_y_up(camera_pos).tolist(),
            "rotation": _quat_to_y_up(camera_quat, True).tolist(),
            "fov": camera.fov,
            "aperture": camera.aperture,
            "near_plane": camera.near,
            "far_plane": camera.far,
            "resolution": camera.res,
            "focal_length": camera.focal_len,
            "samples_per_pixel": camera.spp,
            "denoise": camera.denoise,
        }
        return camera_dict

    def _load_camera_desc(self, camera_dict: dict) -> dict:
        camera_pos = _pos_from_y_up(camera_dict.get("position"))
        camera_quat = _quat_from_y_up(camera_dict.get("rotation"))
        camera_transform = gu.trans_quat_to_T(camera_pos, camera_quat)
        camera_lookat = gu.T_to_pos_lookat_up(camera_transform)[1]

        args_dict = {
            "pos": camera_pos.tolist(),
            "lookat": camera_lookat.tolist(),
            "res": camera_dict.get("resolution"),
            "fov": camera_dict.get("fov", camera_dict.get("fov_y")),
            "aperture": camera_dict.get("aperture"),
            "near": camera_dict.get("near_plane"),
            "far": camera_dict.get("far_plane"),
            "spp": camera_dict.get("samples_per_pixel"),
            "denoise": camera_dict.get("denoise"),
        }
        return args_dict

    def _capture_camera_desc(self):
        cameras = [camera for camera in self._scene.visualizer.cameras if not camera.debug]
        if not cameras:
            return {}

        cameras_pos = torch.stack([camera.get_pos() for camera in cameras])
        cameras_quat = torch.stack([camera.get_quat() for camera in cameras])
        cameras_lookat = torch.stack([camera.get_lookat() for camera in cameras])

        # No need to convert to y-up space, since the quat returned by get_quat is already in y-up space
        # TODO: Consider storing the z-up quat in the camera objects, and only convert on demand
        convert_to_y_up_list = [True] * len(cameras)
        cameras_pos = _pos_to_y_up(cameras_pos)
        cameras_quat = _quat_to_y_up(cameras_quat, convert_to_y_up_list)
        cameras_lookat = _pos_to_y_up(cameras_lookat)

        transforms = {
            "pos": _serialize_transforms(cameras_pos),
            "quat": _serialize_transforms(cameras_quat),
            "lookat": _serialize_transforms(cameras_lookat),
        }
        return transforms

    def _restore_camera_desc(self, camera_transforms):
        cameras_dict = self._json_content.get(ElementType.CAMERA.value)
        if not cameras_dict:
            return {}

        cameras_pos = _pos_from_y_up(
            _deserialize_transforms(camera_transforms["pos"], shape=(len(cameras_dict), -1, 3)).squeeze(1)
        )
        cameras_lookat = _pos_from_y_up(
            _deserialize_transforms(camera_transforms["lookat"], shape=(len(cameras_dict), -1, 3)).squeeze(1)
        )

        cameras_capture = {}
        for i, camera_dict in enumerate(cameras_dict):
            camera_name = camera_dict["name"]
            cameras_capture[camera_name] = {
                "pos": cameras_pos[i].tolist(),
                "lookat": cameras_lookat[i].tolist(),
            }
        return cameras_capture

    def _generate_camera_name(self, camera: gs.vis.camera.Camera, index: int) -> str:
        return f"{camera.model}_{index}"

    def _load_camera_name(self, camera_dict: dict, index: int) -> str:
        return f"{camera_dict.get('model', 'camera')}_{index}"

    # Lights
    def _generate_light_desc(self, light) -> dict:
        if not isinstance(light, gs.vis.apollo_renderer.Light):
            return

        light_dict = {
            "color": light.color,
            "intensity": light.intensity,
            "cast_shadow": light.castshadow,
        }
        if light.directional:
            light_dict["type"] = "directional"
            light_dict["direction"] = _pos_to_y_up(light.dir).tolist()
        else:
            light_dict["type"] = "spot"
            light_dict["position"] = _pos_to_y_up(light.pos).tolist()
            light_dict["direction"] = _pos_to_y_up(light.dir).tolist()
            light_dict["radius"] = random.randint(10, 50)  # Temporary random radius
            light_dict["attenuation"] = light.attenuation
            light_dict["inner_cone_angle"] = light.cutoffDeg
            light_dict["outer_cone_angle"] = light.cutoffDeg
            light_dict["falloff"] = 1.0
        light_dict["name"] = light_dict["type"]
        return light_dict

    def _generate_environment_map_desc(self) -> dict:
        # TODO: Fix this to align with the new schema
        if self._scene.visualizer.raytracer is None:
            return

        light_dict = {}
        light_dict["type"] = "environment"
        light_dict["name"] = "environment"
        surface = self._scene.visualizer.raytracer.env_sphere.surface
        if surface is not None:
            full_texture_path = surface.emissive_texture.input_image_path
            light_dict["texture"] = self.get_rel_asset_path(full_texture_path)
        light_dict["rotation"] = self._scene.visualizer.raytracer.env_sphere.quat.tolist()
        light_dict["intensity"] = 1.0
        return light_dict

    @property
    def scene(self) -> gs.Scene:
        return self._scene

    @property
    def json_content(self) -> dict:
        return self._json_content
