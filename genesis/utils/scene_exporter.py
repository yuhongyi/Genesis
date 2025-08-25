import json
import os
import math
import random
import xml.etree.ElementTree as ET

import torch
import genesis as gs
import genesis.utils.geom as gu
from trimesh.visual.color import ColorVisuals

CURRENT_SCENE_DESCRIPTION_VERSION = 1
DEFAULT_ASSET_ROOT_PATH = gs.utils.get_assets_dir()


# Helper functions
def pos_to_v1_coordinate_system(pos):
    to_y_forward = torch.tensor([0.7071068, -0.7071068, 0, 0], dtype=gs.tc_float, device=gs.device)
    return gu.transform_by_quat(pos, to_y_forward)
    # return torch.tensor([pos[0], pos[2], -pos[1]])
    # return pos


def quat_to_v1_coordinate_system(quat):
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    # w, x, y, z = 1, 0, 0, 0
    return torch.tensor([x + w, x - w, y - z, y + z]) / math.sqrt(2.0)
    # return torch.tensor([w, x, z, -y])  # y-up
    # return quat


def T_to_quat_v1_coordinate_system(T):
    R = T[:3, :3]
    quat = gu.R_to_quat(R)
    return quat_to_v1_coordinate_system(quat)


def wxyz_to_xyzw(wxyz):
    return torch.tensor([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])


class SceneDescriptionFrame:
    def __init__(self):
        self._mesh_transforms = None
        self._camera_transforms = None
        self._light_transforms = None


class SceneDescriptionExporter:

    def __init__(self, export_path, scene):
        self._export_path = export_path
        self._scene = scene
        self._frame_time = scene.sim_options.dt
        self._json_content = dict()

    def generate_initial_scene_description(self, num_envs=1, asset_root_path=DEFAULT_ASSET_ROOT_PATH):
        # headers
        self._json_content = {
            "version": CURRENT_SCENE_DESCRIPTION_VERSION,
            "num_environments": num_envs,
            "asset_root_path": asset_root_path,
            "mesh_entities": [],
            "camera_entities": [],
            "light_entities": [],
            "frames": [],
        }

        # mesh entities
        for entity in self._scene.entities:
            self._add_entity_to_json(self._json_content["mesh_entities"], entity)

        # camera entities
        for camera in self._scene.visualizer.cameras:
            self._add_camera_to_json(self._json_content["camera_entities"], camera)

        # light entities
        for light in self._scene.visualizer.batch_renderer.lights:
            self._add_light_to_json(self._json_content["light_entities"], light)

    def capture_frame(self):
        frame = dict()

        frame["mesh_transforms"] = self._get_mesh_transforms()
        frame["camera_transforms"] = self._get_camera_transforms()
        frame["light_transforms"] = self._get_light_transforms()

        self._json_content["frames"].append(frame)

    def _get_mesh_transforms(self):
        transforms = dict()
        transforms["pos"] = self._scene.rigid_solver.vgeoms_state.pos.to_torch().tolist()
        transforms["quat"] = self._scene.rigid_solver.vgeoms_state.quat.to_torch().tolist()
        return transforms

    def _get_camera_transforms(self):
        transforms = dict()
        transforms["pos"] = [camera.get_pos() for camera in self._scene.visualizer.cameras]
        transforms["quat"] = [camera.get_quat() for camera in self._scene.visualizer.cameras]
        return transforms

    def _get_light_transforms(self):
        # Lights are not movable for now
        return None

    def export(self):
        with open(self._export_path, "w") as f:
            json.dump(self._json_content, f, indent=4)

    # Meshes
    def _add_entity_to_json(self, entities_array, entity):
        # Skip if entity is not a RigidEntity
        if not isinstance(entity, gs.engine.entities.RigidEntity):
            return

        entity_type = self._get_entity_type(entity)
        entity_scale = self._get_entity_scale(entity)
        entity_material_override = self._get_entity_material_override(entity)
        for vgeom in entity.vgeoms:
            vgeom_dict = {}
            vgeom_dict["entity_type"] = entity_type
            vgeom_dict["position"] = self._get_vgeom_position(vgeom)
            vgeom_dict["rotation"] = self._get_vgeom_rotation(vgeom)
            vgeom_dict["scale"] = entity_scale
            uri = self._get_vgeom_uri(vgeom)
            if uri is not None:
                vgeom_dict["uri"] = uri
            vgeom_dict["material_override"] = self._get_vgeom_material_override(
                entity_type, vgeom, entity_material_override
            )
            entities_array.append(vgeom_dict)

    def _get_entity_type(self, entity):
        if isinstance(entity.morph, gs.morphs.FileMorph):
            return "mesh"
        elif isinstance(entity.morph, gs.morphs.Box):
            return "box"
        elif isinstance(entity.morph, gs.morphs.Cylinder):
            return "cylinder"
        elif isinstance(entity.morph, gs.morphs.Sphere):
            return "sphere"
        elif isinstance(entity.morph, gs.morphs.Plane):
            return "plane"
        else:
            return "unknown"

    def _get_vgeom_position(self, vgeom):
        # if more than 1 dim, return the first dim
        # In Genesis, when n_envs == 0, dim is 1, otherwise dim is 2
        init_pos = vgeom.init_pos
        init_quat = vgeom.init_quat
        if vgeom.get_pos().dim() > 1:
            return pos_to_v1_coordinate_system(vgeom.get_pos()[0]).tolist()
        else:
            return pos_to_v1_coordinate_system(vgeom.get_pos()).tolist()

    def _get_vgeom_rotation(self, vgeom):
        # if more than 1 dim, return the first dim
        # In Genesis, when n_envs == 0, dim is 1, otherwise dim is 2
        if vgeom.get_quat().dim() > 1:
            quat = vgeom.get_quat()[0]
        else:
            quat = vgeom.get_quat()

        if not isinstance(vgeom.entity.morph, gs.morphs.Primitive):
            quat = quat_to_v1_coordinate_system(quat)
        return wxyz_to_xyzw(quat).tolist()

    def _get_entity_scale(self, entity):
        if hasattr(entity.morph, "scale"):
            if isinstance(entity.morph.scale, float):
                return (entity.morph.scale, entity.morph.scale, entity.morph.scale)
            elif isinstance(entity.morph.scale, tuple) and len(entity.morph.scale) == 3:
                return entity.morph.scale

        # Fall back to default scale (1.0, 1.0, 1.0)
        return (1.0, 1.0, 1.0)

    def _get_vgeom_uri(self, vgeom):
        if "mesh_path" in vgeom.metadata:
            mesh_path = vgeom.metadata["mesh_path"]
            if os.path.isabs(mesh_path):
                return os.path.relpath(mesh_path, DEFAULT_ASSET_ROOT_PATH)
            elif isinstance(vgeom.entity.morph, gs.morphs.MJCF):
                return self._get_vgeom_uri_mjcf(vgeom)
            else:
                return mesh_path
        else:
            return None

    def _get_vgeom_uri_mjcf(self, vgeom):
        # get meshdir and assetdir from mujoco file
        entity_file_path = vgeom.entity.morph.file
        xml_root = ET.parse(entity_file_path).getroot()
        compiler = xml_root.find("compiler")
        meshdir = None
        assetdir = None
        if compiler is not None:
            meshdir = compiler.get("meshdir")
            assetdir = compiler.get("assetdir")

        entity_file_abs_dir = os.path.dirname(entity_file_path)
        mesh_path = vgeom.metadata["mesh_path"]

        # Search for the mesh in the meshdir or assetdir
        if meshdir is not None:
            abs_mesh_path = os.path.join(entity_file_abs_dir, meshdir, mesh_path)
        elif assetdir is not None:
            abs_mesh_path = os.path.join(entity_file_abs_dir, assetdir, mesh_path)
        else:
            abs_mesh_path = os.path.join(entity_file_abs_dir, mesh_path)
        if os.path.exists(abs_mesh_path):
            return os.path.relpath(abs_mesh_path, DEFAULT_ASSET_ROOT_PATH)

        return None

    def _get_entity_material_override(self, entity):
        surface = entity.surface
        material_override = {}

        # Define attribute pairs as (attribute_name, property_name) tuples
        MATERIAL_PROPERTIES = [
            # Material properties
            "color",
            "opacity",
            "roughness",
            "metallic",
            "emissive",
            "ior",
            "doublesided",
            "subsurface",
            "thickness",
        ]

        MATERIAL_TEXTURES = [
            # Texture properties
            "diffuse_texture",
            "opacity_texture",
            "roughness_texture",
            "metallic_texture",
            "normal_texture",
            "emissive_texture",
            "specular_texture",
            "transmission_texture",
            "thickness_texture",
        ]

        # Update all attributes
        for property_name in MATERIAL_PROPERTIES:
            self._get_material_property_override(material_override, surface, property_name)

        for texture_name in MATERIAL_TEXTURES:
            self._get_material_texture_override(material_override, surface, texture_name)

        return material_override

    def _get_material_property_override(self, material_override, surface, property_name):
        if hasattr(surface, property_name) and getattr(surface, property_name) is not None:
            material_override[property_name] = getattr(surface, property_name)
            return

    def _get_material_texture_override(self, material_override, surface, texture_name):
        if hasattr(surface, texture_name) and getattr(surface, texture_name) is not None:
            texture = getattr(surface, texture_name)
            if isinstance(texture, gs.textures.ImageTexture) and texture.input_image_path is not None:
                material_override[texture_name] = texture.input_image_path
            return

    def _get_vgeom_material_override(self, entity_type, vgeom, entity_material_override):
        # If entity-level material override is not specified,
        # use the geometry-level material override for certain properties
        material_override = entity_material_override

        # Fall back to color from ColorVisuals if color override not specified on entity level
        if not "color" in entity_material_override or not "opacity" in entity_material_override:
            visual = vgeom.get_trimesh().visual
            if isinstance(visual, ColorVisuals):
                geom_color = visual.main_color / 255.0
                if not "color" in entity_material_override:
                    material_override["color"] = geom_color[:3].tolist()
                if not "opacity" in entity_material_override:
                    material_override["opacity"] = geom_color[3]

        # Special case for plane
        if entity_type == "plane":
            material_override["diffuse_texture"] = "textures/checker.png"

        return material_override

    # Cameras
    def _add_camera_to_json(self, cameras_array, camera):
        # Skip if camera is not a CameraEntity
        if not isinstance(camera, gs.vis.camera.Camera):
            return

        if camera.debug:
            return

        camera_dict = {}
        camera_dict["position"] = self._get_camera_position(camera)
        camera_dict["rotation"] = self._get_camera_rotation(camera)
        camera_dict["fov"] = camera.fov
        camera_dict["aperture"] = camera.aperture
        camera_dict["near_plane"] = camera.near
        camera_dict["far_plane"] = camera.far
        camera_dict["resolution"] = camera.res
        camera_dict["focal_length"] = camera.focal_len
        camera_dict["samples_per_pixel"] = camera.spp
        camera_dict["denoise"] = camera.denoise
        cameras_array.append(camera_dict)

    def _get_camera_position(self, camera):
        return pos_to_v1_coordinate_system(camera._initial_pos).cpu().tolist()

    def _get_camera_rotation(self, camera):
        if camera._initial_transform is not None:
            transform = camera._initial_transform
        else:
            transform = gu.pos_lookat_up_to_T(camera._initial_pos, camera._initial_lookat, camera._initial_up)

        return wxyz_to_xyzw(T_to_quat_v1_coordinate_system(transform)).tolist()

    # Lights
    def _add_light_to_json(self, lights_array, light):
        if not isinstance(light, gs.vis.batch_renderer.Light):
            return

        light_dict = {}
        if light.directional:
            light_dict["type"] = "directional"
            light_dict["direction"] = light.dir
        else:
            light_dict["type"] = "spot"
            light_dict["position"] = light.pos
            light_dict["direction"] = light.dir
            light_dict["radius"] = random.randint(10, 50)  # Temporary random radius
            light_dict["attenuation"] = random.randint(10, 20) * 0.1  # Temporary random attenuation
            light_dict["inner_cone_angle"] = light.cutoffDeg
            light_dict["outer_cone_angle"] = min(light.cutoffDeg * 1.2, 179.0)  # Temporary outer cone angle
            light_dict["falloff"] = random.randint(10, 20) * 0.1  # Temporary random falloff
        light_dict["color"] = self._get_light_color(light)
        light_dict["intensity"] = light.intensity
        light_dict["cast_shadow"] = light.castshadow
        lights_array.append(light_dict)

    def _get_light_color(self, light):
        # TODO: Implement conversion from light.color to hex color
        return random.choice(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
        )  # Temporary random color
