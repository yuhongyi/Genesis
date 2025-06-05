import argparse
import os

import numpy as np
import genesis as gs
import torch

# Create a struct to store the arguments
class BenchmarkArgs:
    def __init__(self, rasterizer, n_envs, n_steps, resX, resY, camera_posX, camera_posY, camera_posZ, camera_lookatX, camera_lookatY, camera_lookatZ, camera_fov, mjcf, benchmark_result_file_path):
        self.rasterizer = rasterizer
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.resX = resX
        self.resY = resY
        self.camera_posX = camera_posX
        self.camera_posY = camera_posY
        self.camera_posZ = camera_posZ
        self.camera_lookatX = camera_lookatX
        self.camera_lookatY = camera_lookatY
        self.camera_lookatZ = camera_lookatZ
        self.camera_fov = camera_fov
        self.mjcf = mjcf
        self.benchmark_result_file_path = benchmark_result_file_path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--rasterizer", action="store_true", default=False)
    parser.add_argument("-n", "--n_envs", type=int, default=1024)
    parser.add_argument("-s", "--n_steps", type=int, default=1)
    parser.add_argument("-x", "--resX", type=int, default=1024)
    parser.add_argument("-y", "--resY", type=int, default=1024)
    parser.add_argument("-i", "--camera_posX", type=float, default=1.5)
    parser.add_argument("-j", "--camera_posY", type=float, default=0.5)
    parser.add_argument("-k", "--camera_posZ", type=float, default=1.5)
    parser.add_argument("-l", "--camera_lookatX", type=float, default=0.0)
    parser.add_argument("-m", "--camera_lookatY", type=float, default=0.0)
    parser.add_argument("-o", "--camera_lookatZ", type=float, default=0.5)
    parser.add_argument("-v", "--camera_fov", type=float, default=45)
    parser.add_argument("-f", "--mjcf", type=str, default="xml/franka_emika_panda/panda.xml")
    parser.add_argument("-g", "--benchmark_result_file_path", type=str, default="benchmark.csv")
    args = parser.parse_args()
    benchmark_args = BenchmarkArgs(
        rasterizer=args.rasterizer,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        resX=args.resX,
        resY=args.resY,
        camera_posX=args.camera_posX,
        camera_posY=args.camera_posY,
        camera_posZ=args.camera_posZ,
        camera_lookatX=args.camera_lookatX,
        camera_lookatY=args.camera_lookatY,
        camera_lookatZ=args.camera_lookatZ,
        camera_fov=args.camera_fov,
        mjcf=args.mjcf,
        benchmark_result_file_path=args.benchmark_result_file_path,
    )
    print(f"Benchmark with args:")
    print(f"  rasterizer: {benchmark_args.rasterizer}")
    print(f"  n_envs: {benchmark_args.n_envs}")
    print(f"  n_steps: {benchmark_args.n_steps}")
    print(f"  resolution: {benchmark_args.resX}x{benchmark_args.resY}")
    print(f"  camera_pos: ({benchmark_args.camera_posX}, {benchmark_args.camera_posY}, {benchmark_args.camera_posZ})")
    print(f"  camera_lookat: ({benchmark_args.camera_lookatX}, {benchmark_args.camera_lookatY}, {benchmark_args.camera_lookatZ})")
    print(f"  camera_fov: {benchmark_args.camera_fov}")
    print(f"  mjcf: {benchmark_args.mjcf}")
    print(f"  benchmark_result_file_path: {benchmark_args.benchmark_result_file_path}")
    return benchmark_args

def init_gs(benchmark_args):
    ########################## init ##########################
    try:
        gs.init(backend=gs.gpu)
    except Exception as e:
        print(f"Failed to initialize GPU backend: {e}")
        print("Falling back to CPU backend")
        gs.init(backend=gs.cpu)
    
    ########################## create a scene ##########################
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(benchmark_args.camera_posX, benchmark_args.camera_posY, benchmark_args.camera_posZ),
            camera_lookat=(benchmark_args.camera_lookatX, benchmark_args.camera_lookatY, benchmark_args.camera_lookatZ),
            camera_fov=benchmark_args.camera_fov,
        ),
        show_viewer=False,
        rigid_options=gs.options.RigidOptions(
            # constraint_solver=gs.constraint_solver.Newton,
            ),
        renderer = gs.options.renderers.BatchRenderer(
            use_rasterizer=benchmark_args.rasterizer,
            batch_render_res=(benchmark_args.resX, benchmark_args.resY),
        )            
    )

    ########################## entities ##########################
    #plane = scene.add_entity(
    #    gs.morphs.Plane(),
    #)
    franka = scene.add_entity(
        gs.morphs.MJCF(file=benchmark_args.mjcf),
        visualize_contact=True,
    )

    ########################## cameras ##########################
    cam_0 = scene.add_camera(
        pos=(benchmark_args.camera_posX, benchmark_args.camera_posY, benchmark_args.camera_posZ),
        lookat=(benchmark_args.camera_lookatX, benchmark_args.camera_lookatY, benchmark_args.camera_lookatZ),
        fov=benchmark_args.camera_fov,
    )
    scene.add_light(
        pos=[0.0, 0.0, 1.5],
        dir=[1.0, 1.0, -2.0],
        directional=1,
        castshadow=1,
        cutoff=45.0,
        intensity=0.5
    )
    scene.add_light(
        pos=[4, -4, 4],
        dir=[-1, 1, -1],
        directional=0,
        castshadow=1,
        cutoff=45.0,
        intensity=1
    )
    ########################## build ##########################
    n_envs = benchmark_args.n_envs
    n_steps = benchmark_args.n_steps
    scene.build(n_envs=n_envs)
    return scene, n_envs, n_steps

def add_noise_to_all_cameras(scene):
    for cam in scene.visualizer.cameras:
        cam.set_pose(
            pos=cam.pos_all_envs + torch.rand((cam.n_envs, 3), device=cam.pos_all_envs.device) * 0.002 - 0.001,
            lookat=cam.lookat_all_envs + torch.rand((cam.n_envs, 3), device=cam.lookat_all_envs.device) * 0.002 - 0.001,
            up=cam.up_all_envs + torch.rand((cam.n_envs, 3), device=cam.up_all_envs.device) * 0.002 - 0.001,
        )

def run_benchmark(scene, n_envs, n_steps, benchmark_args):
    try:
        # warmup
        scene.step()
        rgb, depth, _, _ = scene.batch_render()

        # timer
        from time import time
        start_time = time()

        for i in range(n_steps):
            add_noise_to_all_cameras(scene)
            rgb, depth, _, _ = scene.batch_render(force_render=True)
        
        end_time = time()
        time_taken = end_time - start_time
        time_taken_per_env = time_taken / n_envs
        fps = n_envs * n_steps / time_taken
        fps_per_env = n_steps / time_taken

        print(f'Time taken: {time_taken} seconds')
        print(f'Time taken per env: {time_taken_per_env} seconds')
        print(f'FPS: {fps}')
        print(f'FPS per env: {fps_per_env}')

        # Ensure the directory exists
        os.makedirs(os.path.dirname(benchmark_args.benchmark_result_file_path), exist_ok=True)

        # Append a line with all args and results in csv format
        with open(benchmark_args.benchmark_result_file_path, 'a') as f:
            f.write(f'succeeded,{benchmark_args.mjcf},{benchmark_args.rasterizer},{benchmark_args.n_envs},{benchmark_args.n_steps},{benchmark_args.resX},{benchmark_args.resY},{benchmark_args.camera_posX},{benchmark_args.camera_posY},{benchmark_args.camera_posZ},{benchmark_args.camera_lookatX},{benchmark_args.camera_lookatY},{benchmark_args.camera_lookatZ},{benchmark_args.camera_fov},{time_taken},{time_taken_per_env},{fps},{fps_per_env}\n')
    except Exception as e:
        print(f"Error during benchmark: {e}")
        raise

def main():
    ######################## Parse arguments #######################
    benchmark_args = parse_args()

    ######################## Initialize scene #######################
    scene, n_envs, n_steps = init_gs(benchmark_args)

    ######################## Run benchmark #######################
    run_benchmark(scene, n_envs, n_steps, benchmark_args)

if __name__ == "__main__":
    main()

