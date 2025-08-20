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


def _pos_to_v1_coordinate_system(pos):
    to_y_forward = torch.tensor([0.7071068, -0.7071068, 0, 0], dtype=gs.tc_float, device=gs.device)
    return gu.transform_by_quat(pos, to_y_forward)
    # return torch.tensor([pos[0], pos[2], -pos[1]])
    # return pos


def _quat_to_v1_coordinate_system(quat):
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    # w, x, y, z = 1, 0, 0, 0
    return torch.tensor([x + w, x - w, y - z, y + z]) / math.sqrt(2.0)
    # return torch.tensor([w, x, z, -y])  # y-up
    # return quat


def _T_to_quat_v1_coordinate_system(T):
    R = T[:3, :3]
    quat = gu.R_to_quat(R)
    return _quat_to_v1_coordinate_system(quat)


def wxyz_to_xyzw(wxyz):
    return torch.tensor([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])


def generate_scene_description(scene, num_envs=1, asset_root_path=DEFAULT_ASSET_ROOT_PATH):
    # headers
    json_content = {
        "version": CURRENT_SCENE_DESCRIPTION_VERSION,
        "num_environments": num_envs,
        "asset_root_path": asset_root_path,
        "mesh_entities": [],
        "camera_entities": [],
        "light_entities": [],
    }

    # mesh entities
    for entity in scene.entities:
        add_entity_to_json(json_content["mesh_entities"], entity)

    # camera entities
    for camera in scene.visualizer.cameras:
        add_camera_to_json(json_content["camera_entities"], camera)

    # light entities
    for light in scene.visualizer.batch_renderer.lights:
        add_light_to_json(json_content["light_entities"], light)

    return json_content


# Meshes
def add_entity_to_json(entities_array, entity):
    # Skip if entity is not a RigidEntity
    if not isinstance(entity, gs.engine.entities.RigidEntity):
        return

    entity_type = get_entity_type(entity)
    entity_scale = get_entity_scale(entity)
    entity_material_override = get_entity_material_override(entity)
    for vgeom in entity.vgeoms:
        vgeom_dict = {}
        vgeom_dict["entity_type"] = entity_type
        vgeom_dict["position"] = get_vgeom_position(vgeom)
        vgeom_dict["rotation"] = get_vgeom_rotation(vgeom)
        vgeom_dict["scale"] = entity_scale
        uri = get_vgeom_uri(vgeom)
        if uri is not None:
            vgeom_dict["uri"] = uri
        vgeom_dict["material_override"] = get_vgeom_material_override(entity_type, vgeom, entity_material_override)
        entities_array.append(vgeom_dict)


def get_entity_type(entity):
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


def get_vgeom_position(vgeom):
    # if more than 1 dim, return the first dim
    # In Genesis, when n_envs == 0, dim is 1, otherwise dim is 2
    init_pos = vgeom.init_pos
    init_quat = vgeom.init_quat
    if vgeom.get_pos().dim() > 1:
        return _pos_to_v1_coordinate_system(vgeom.get_pos()[0]).tolist()
    else:
        return _pos_to_v1_coordinate_system(vgeom.get_pos()).tolist()


def get_vgeom_rotation(vgeom):
    # if more than 1 dim, return the first dim
    # In Genesis, when n_envs == 0, dim is 1, otherwise dim is 2
    if vgeom.get_quat().dim() > 1:
        quat = vgeom.get_quat()[0]
    else:
        quat = vgeom.get_quat()

    if not isinstance(vgeom.entity.morph, gs.morphs.Primitive):
        quat = _quat_to_v1_coordinate_system(quat)
    return wxyz_to_xyzw(quat).tolist()


def get_entity_scale(entity):
    if hasattr(entity.morph, "scale"):
        if isinstance(entity.morph.scale, float):
            return (entity.morph.scale, entity.morph.scale, entity.morph.scale)
        elif isinstance(entity.morph.scale, tuple) and len(entity.morph.scale) == 3:
            return entity.morph.scale

    # Fall back to default scale (1.0, 1.0, 1.0)
    return (1.0, 1.0, 1.0)


def get_vgeom_uri(vgeom):
    if "mesh_path" in vgeom.metadata:
        mesh_path = vgeom.metadata["mesh_path"]
        if os.path.isabs(mesh_path):
            return os.path.relpath(mesh_path, DEFAULT_ASSET_ROOT_PATH)
        elif isinstance(vgeom.entity.morph, gs.morphs.MJCF):
            return get_vgeom_uri_mjcf(vgeom)
        else:
            return mesh_path
    else:
        return None


def get_vgeom_uri_mjcf(vgeom):
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


def get_entity_material_override(entity):
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
        get_material_property_override(material_override, surface, property_name)

    for texture_name in MATERIAL_TEXTURES:
        get_material_texture_override(material_override, surface, texture_name)

    return material_override


def get_material_property_override(material_override, surface, property_name):
    if hasattr(surface, property_name) and getattr(surface, property_name) is not None:
        material_override[property_name] = getattr(surface, property_name)
        return


def get_material_texture_override(material_override, surface, texture_name):
    if hasattr(surface, texture_name) and getattr(surface, texture_name) is not None:
        texture = getattr(surface, texture_name)
        if isinstance(texture, gs.textures.ImageTexture) and texture.input_image_path is not None:
            material_override[texture_name] = texture.input_image_path
        return


def get_vgeom_material_override(entity_type, vgeom, entity_material_override):
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
def add_camera_to_json(cameras_array, camera):
    # Skip if camera is not a CameraEntity
    if not isinstance(camera, gs.vis.camera.Camera):
        return

    if camera.debug:
        return

    camera_dict = {}
    camera_dict["position"] = get_camera_position(camera)
    camera_dict["rotation"] = get_camera_rotation(camera)
    camera_dict["fov"] = camera.fov
    camera_dict["aperture"] = camera.aperture
    camera_dict["near_plane"] = camera.near
    camera_dict["far_plane"] = camera.far
    camera_dict["resolution"] = camera.res
    camera_dict["focal_length"] = camera.focal_len
    camera_dict["samples_per_pixel"] = camera.spp
    camera_dict["denoise"] = camera.denoise
    cameras_array.append(camera_dict)


def get_camera_position(camera):
    return _pos_to_v1_coordinate_system(camera._initial_pos).cpu().tolist()


def get_camera_rotation(camera):
    if camera._initial_transform is not None:
        transform = camera._initial_transform
    else:
        transform = gu.pos_lookat_up_to_T(camera._initial_pos, camera._initial_lookat, camera._initial_up)

    return wxyz_to_xyzw(_T_to_quat_v1_coordinate_system(transform)).tolist()


# Lights
def add_light_to_json(lights_array, light):
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
    light_dict["color"] = get_light_color(light)
    light_dict["intensity"] = light.intensity
    light_dict["cast_shadow"] = light.castshadow
    lights_array.append(light_dict)


def get_light_color(light):
    # TODO: Implement conversion from light.color to hex color
    return random.choice([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])  # Temporary random color
