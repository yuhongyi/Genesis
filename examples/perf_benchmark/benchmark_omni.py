# Before running, convert the assets like:
# python examples/perf_benchmark/process_xml.py \
#   --file ./genesis/assets/xml/franka_emika_panda/panda.xml

######################## Parse arguments #######################
# Create a struct to store the arguments
import argparse
from batch_benchmark import BenchmarkArgs

import os
import math
import numpy as np
import torch
import psutil
import pynvml
from scipy.spatial.transform import Rotation as R

import carb
import isaaclab.sim as sim_utils
import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils
from isaaclab.sensors.camera import TiledCamera, TiledCameraCfg
from isaaclab.sim.converters import (
    MjcfConverter, MjcfConverterCfg,
    UrdfConverter, UrdfConverterCfg
)
from isaaclab.utils.math import (
    create_rotation_matrix_from_view,
    quat_from_matrix,
)
import omni.replicator.core as rep
from pxr import UsdLux, PhysxSchema

from isaacsim.core.utils.extensions import enable_extension

from isaaclab.app import AppLauncher

def load_mjcf(mjcf_path):
    return MjcfConverter(
        MjcfConverterCfg(
            asset_path=mjcf_path,
            fix_base=True,
            force_usd_conversion=True
        )
    ).usd_path

def load_urdf(urdf_path):
    return UrdfConverter(
        UrdfConverterCfg(
            asset_path=urdf_path,
            joint_drive=None,
            fix_base=True,
            force_usd_conversion=True
        )
    ).usd_path

def init_isaac(benchmark_args):
    ########################## init ##########################
    stage_utils.create_new_stage()
    stage = stage_utils.get_current_stage()
    scene = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(device="cuda:0", dt=0.01,)
    )
    cam_eye = (
        benchmark_args.camera_posX,
        benchmark_args.camera_posY,
        benchmark_args.camera_posZ
    )
    cam_target = (
        benchmark_args.camera_lookatX,
        benchmark_args.camera_lookatY,
        benchmark_args.camera_lookatZ
    )
    scene.set_camera_view(eye=cam_eye, target=cam_target)
    cam_eye = torch.Tensor(cam_eye).reshape(-1, 3)
    cam_target = torch.Tensor(cam_target).reshape(-1, 3)
    carb_settings = carb.settings.get_settings()

    physxSceneAPI = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/physicsScene"))
    physxSceneAPI.CreateGpuTempBufferCapacityAttr(16 * 1024 * 1024 * 2)
    physxSceneAPI.CreateGpuHeapCapacityAttr(64 * 1024 * 1024 * 2)
    physxSceneAPI.CreateGpuMaxRigidPatchCountAttr(8388608)
    physxSceneAPI.CreateGpuMaxRigidContactCountAttr(16777216)

    # Options: https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer_pt.html
    print("Before setting:")
    print("Render mode:", carb_settings.get("/rtx/rendermode"))
    print("Sample per pixel:", carb_settings.get("/rtx/pathtracing/spp"))
    print("Total spp:", carb_settings.get("/rtx/pathtracing/totalSpp"))
    print("Clamp spp:", carb_settings.get("/rtx/pathtracing/clampSpp"))
    print("Max bounce:", carb_settings.get("/rtx/pathtracing/maxBounces"))
    print("Optix Denoiser", carb_settings.get("/rtx/pathtracing/optixDenoiser/enabled"))
    print("Shadows", carb_settings.get("/rtx/shadows/enabled"))

    rep.settings.set_render_rtx_realtime()
    if benchmark_args.rasterizer:
        # carb_settings.set("/rtx/rendermode", "Hydra Storm")
        carb_settings.set("/rtx/rendermode", "RayTracedLighting")
    else:
        carb_settings.set("/rtx/rendermode", "PathTracing")
    carb_settings.set("/rtx/pathtracing/spp", benchmark_args.spp)
    carb_settings.set("/rtx/pathtracing/totalSpp", benchmark_args.spp)
    carb_settings.set("/rtx/pathtracing/clampSpp", benchmark_args.spp)
    carb_settings.set("/rtx/pathtracing/maxBounces", benchmark_args.max_bounce)
    carb_settings.set("/rtx/pathtracing/optixDenoiser/enabled", False)
    carb_settings.set("/rtx/shadows/enabled", False)

    print("After setting:")
    print("Render mode:", carb_settings.get("/rtx/rendermode"))
    print("Sample per pixel:", carb_settings.get("/rtx/pathtracing/spp"))
    print("Total spp:", carb_settings.get("/rtx/pathtracing/totalSpp"))
    print("Clamp spp:", carb_settings.get("/rtx/pathtracing/clampSpp"))
    print("Max bounce:", carb_settings.get("/rtx/pathtracing/maxBounces"))
    print("Optix Denoiser", carb_settings.get("/rtx/pathtracing/optixDenoiser/enabled"))
    print("Shadows", carb_settings.get("/rtx/shadows/enabled"))

    ########################## entities ##########################
    spacing_row = np.array((2.0, -6.0))
    spacing_col = np.array((-6.0, -2.0))
    n_cols = int(math.sqrt(benchmark_args.n_envs))
    offsets = []
    for i in range(benchmark_args.n_envs):
        col = i % n_cols
        row = i // n_cols
        offset_XY = (row * spacing_row + col * spacing_col)
        offset = np.array([*offset_XY, 0.0])
        offsets.append(offset)
        prim_utils.create_prim(
            f"/World/Origin{i:05d}", "Xform", translation=offset
        )
    offsets = np.array(offsets)

    # load objects
    # plane_path = load_urdf(os.path.join("genesis/assets", "urdf/plane/plane.urdf"))
    plane_path = os.path.abspath(os.path.join("genesis/assets", "urdf/plane_usd/plane.usd"))
    print(plane_path)
    plane_cfg = sim_utils.UsdFileCfg(usd_path=plane_path)
    plane_cfg.func("/World/Origin.*/plane", plane_cfg)

    robot_name = f"{os.path.splitext(benchmark_args.mjcf)[0]}_new.xml"
    robot_path = load_mjcf(os.path.join("genesis/assets", robot_name))
    print("Robot asset:", robot_path)
    robot_cfg = sim_utils.UsdFileCfg(usd_path=robot_path)
    robot_cfg.func("/World/Origin.*/robot", robot_cfg)
    # robot = Articulation(
    #     cfg = ArticulationCfg(
    #         prim_path="/World/Origin.*/robot",
    #         actuators={},))
    # for i in range(benchmark_args.n_envs):
    #     stage.RemovePrim(f"/World/Origin{i:05d}/robot/worldBody")
    # print(stage.GetPrimAtPath("World/Origin00000/robot/worldBody"))
    # print(stage.GetPrimAtPath("World/Origin00001/robot/worldBody"))

    cam_fov = math.radians(benchmark_args.camera_fov)
    cam_hapert = 20.955
    cam_fol = cam_hapert / (2 * math.tan(cam_fov / 2))
    cam_quat = quat_from_matrix(
        create_rotation_matrix_from_view(
            cam_target, cam_eye, stage_utils.get_stage_up_axis()
        ) @ R.from_euler('z', 180, degrees=True).as_matrix()   
    )
    cam_eye = tuple(cam_eye.detach().cpu().squeeze().numpy())
    cam_quat = tuple(cam_quat.detach().cpu().squeeze().numpy())
    # cam_quat = (-cam_quat[3], cam_quat[2], -cam_quat[1], cam_quat[0])
    # cam_quat = (-cam_quat[0], -cam_quat[1], -cam_quat[2], -cam_quat[3])

    print(cam_eye, cam_quat)
    print(type(cam_eye), type(cam_quat))

    cam_0 = TiledCamera(
        TiledCameraCfg(
            height=benchmark_args.resX,
            width=benchmark_args.resY,
            offset=TiledCameraCfg.OffsetCfg(
                pos=cam_eye,
                rot=cam_quat,
                convention="ros"
            ),
            prim_path="/World/Origin.*/camera",
            update_period=0,
            data_types=["rgb", "depth"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=cam_fol,
            ),
        )
    )

    ########################## cameras ##########################
    dir_light_pos = torch.Tensor([[0.0, 0.0, 1.5]])
    dir_light_quat = quat_from_matrix(
        create_rotation_matrix_from_view(
            dir_light_pos,
            torch.Tensor([[1.0, 1.0, -2.0]]),
            stage_utils.get_stage_up_axis()))
    dir_light_pos = tuple(dir_light_pos.detach().cpu().squeeze().numpy())
    dir_light_quat = tuple(dir_light_quat.detach().cpu().squeeze().numpy())
    dir_light_cfg = sim_utils.DistantLightCfg(intensity=500.0, angle=45.0)
    dir_light_prim = dir_light_cfg.func(
        "/World/DirectionalLight", dir_light_cfg,
        translation=dir_light_pos,
        orientation=dir_light_quat)

    cone_light_pos = torch.Tensor([[4, -4, 4]])
    cone_light_quat = quat_from_matrix(
        create_rotation_matrix_from_view(
            cone_light_pos,
            torch.Tensor([[-1, 1, -1]]),
            stage_utils.get_stage_up_axis()))
    cone_light_cfg = sim_utils.SphereLightCfg(intensity=1000.0, radius=0.1)
    cone_light_pos = tuple(cone_light_pos.detach().cpu().squeeze().numpy())
    cone_light_quat = tuple(cone_light_quat.detach().cpu().squeeze().numpy())
    cone_light_prim = cone_light_cfg.func(
        "/World/ConeLight", cone_light_cfg,
        translation=cone_light_pos,
        orientation=cone_light_quat)
    cone_light = UsdLux.LightAPI(cone_light_prim)
    UsdLux.ShapingAPI.Apply(cone_light_prim)
    cone_light_prim.SetTypeName("SphereLight")

    return scene, cam_0

def get_utilization_percentages(reset: bool = False, max_values: list[float] = [0.0, 0.0, 0.0, 0.0]) -> list[float]:
    """Get the maximum CPU, RAM, GPU utilization (processing), and
    GPU memory usage percentages since the last time reset was true."""
    if reset:
        max_values[:] = [0, 0, 0, 0]  # Reset the max values

    # CPU utilization
    cpu_usage = psutil.cpu_percent(interval=0.1)
    max_values[0] = max(max_values[0], cpu_usage)

    # RAM utilization
    memory_info = psutil.virtual_memory()
    ram_usage = memory_info.percent
    max_values[1] = max(max_values[1], ram_usage)

    # GPU utilization using pynvml
    if torch.cuda.is_available():
        pynvml.nvmlInit()  # Initialize NVML
        for i in range(torch.cuda.device_count()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)

            # GPU Utilization
            gpu_utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_processing_utilization_percent = gpu_utilization.gpu  # GPU core utilization
            max_values[2] = max(max_values[2], gpu_processing_utilization_percent)

            # GPU Memory Usage
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_memory_total = memory_info.total
            gpu_memory_used = memory_info.used
            gpu_memory_utilization_percent = (gpu_memory_used / gpu_memory_total) * 100
            max_values[3] = max(max_values[3], gpu_memory_utilization_percent)

        pynvml.nvmlShutdown()  # Shutdown NVML after usage
    else:
        gpu_processing_utilization_percent = None
        gpu_memory_utilization_percent = None
    return max_values

def fill_gpu_cache_with_random_data():
    # 100 MB of random data
    dummy_data = torch.rand(100, 1024, 1024, device="cuda")
    # Make some random data manipulation to the entire tensor
    dummy_data = dummy_data.sqrt()

def run_benchmark(scene, camera, benchmark_args):
    try:
        n_envs = benchmark_args.n_envs
        n_steps = benchmark_args.n_steps

        # warmup
        system_utilization_analytics = get_utilization_percentages()
        print(
            f"| CPU:{system_utilization_analytics[0]}% | "
            f"RAM:{system_utilization_analytics[1]}% | "
            f"GPU Compute:{system_utilization_analytics[2]}% | "
            f"GPU Memory: {system_utilization_analytics[3]:.2f}% |"
        )

        scene.reset()
        dt = scene.get_physics_dt()
        for i in range(3):
            scene.step()
            camera.update(dt)
            _ = camera.data
        print("Env and steps:", n_envs, n_steps)
        
        if benchmark_args.gui:
            while True:
               scene.step()

        # fill gpu cache with random data
        fill_gpu_cache_with_random_data()

        # timer
        image_dir = os.path.splitext(benchmark_args.benchmark_result_file_path)[0]
        os.makedirs(image_dir, exist_ok=True)
        from time import time
        start_time = time()

        for i in range(n_steps):
            camera.update(dt)
            # rgb_tiles = camera.data.output.get("rgb").detach().cpu().numpy()
            # depth_tiles = camera.data.output.get("depth").detach().cpu().numpy()
            # print(rgb_tiles.shape, depth_tiles.shape)
            # print(rgb_tiles.dtype, depth_tiles.dtype)

            # for j in range(n_envs):
            #     rgb_image = Image.fromarray(rgb_tiles[j])
            #     rgb_name = f"image_rgb_{i}_{j}_bounce{benchmark_args.max_bounce}_spp{benchmark_args.spp}_shadow{shadow}.png"
            #     rgb_path = os.path.join(image_dir, rgb_name)
            #     rgb_image.save(rgb_path)
            #     print("Image saved:", rgb_path)

            #     depth_tile = depth_tiles[j][:, :, 0]
            #     depth_tile = ((1.0 - (depth_tile / np.max(depth_tile))) * 255.0).astype(np.uint8)
            #     depth_image = Image.fromarray(depth_tile, mode="L")
            #     depth_path = os.path.join(image_dir, f"image_depth_{i}_{j}.png")
            #     depth_image.save(depth_path)
            #     print("Image saved:", depth_path)

        end_time = time()
        time_taken = end_time - start_time
        time_taken_per_env = time_taken / n_envs
        fps = n_envs * n_steps / time_taken
        fps_per_env = n_steps / time_taken
        system_utilization_analytics = get_utilization_percentages()
        print(f'Time taken: {time_taken} seconds')
        print(f'Time taken per env: {time_taken_per_env} seconds')
        print(f'FPS: {fps}')
        print(f'FPS per env: {fps_per_env}')
        print(
            f"| CPU:{system_utilization_analytics[0]}% | "
            f"RAM:{system_utilization_analytics[1]}% | "
            f"GPU Compute:{system_utilization_analytics[2]}% | "
            f" GPU Memory: {system_utilization_analytics[3]:.2f}% |"
        )

        # Append a line with all args and results in csv format
        with open(benchmark_args.benchmark_result_file_path, 'a') as f:
            f.write(f'succeeded,{benchmark_args.mjcf},{benchmark_args.rasterizer},{benchmark_args.n_envs},{benchmark_args.n_steps},{benchmark_args.resX},{benchmark_args.resY},{benchmark_args.camera_posX},{benchmark_args.camera_posY},{benchmark_args.camera_posZ},{benchmark_args.camera_lookatX},{benchmark_args.camera_lookatY},{benchmark_args.camera_lookatZ},{benchmark_args.camera_fov},{time_taken},{time_taken_per_env},{fps},{fps_per_env}\n')
        
        print("App closing..")
        # app.close()
        print("App closed!")
    
    except Exception as e:
        print(f"Error during benchmark: {e}")
        raise

def main():
    ######################## Parse arguments #######################
    benchmark_args = BenchmarkArgs.parse_args()

    ######################## Launch app #######################
    app = AppLauncher(
        headless=not benchmark_args.gui,
        enable_cameras=True,
        device="cuda:0",
        rendering_mode="performance",
    ).app
    enable_extension("isaacsim.asset.importer.mjcf")

    ######################## Initialize scene #######################
    scene, camera = init_isaac(benchmark_args)

    ######################## Run benchmark #######################
    run_benchmark(scene, camera, benchmark_args)

if __name__ == "__main__":
    main()
