import argparse
import os

import numpy as np
import genesis as gs
import torch
from batch_benchmark import BenchmarkArgs

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
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(file=benchmark_args.mjcf),
        visualize_contact=False,
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
    scene.build(n_envs=benchmark_args.n_envs)
    return scene

def fill_gpu_cache_with_random_data():
    # 100 MB of random data
    dummy_data =torch.rand(100, 1024, 1024, device="cuda")
    # Make some random data manipulation to the entire tensor
    dummy_data = dummy_data.sqrt()

def run_benchmark(scene, benchmark_args):
    try:
        n_envs = benchmark_args.n_envs
        n_steps = benchmark_args.n_steps

        # warmup
        scene.step()
        rgb, depth, _, _ = scene.render_all_cams()

        # fill gpu cache with random data
        fill_gpu_cache_with_random_data()

        # timer
        from time import time
        start_time = time()

        for i in range(n_steps):
            rgb, depth, _, _ = scene.render_all_cams(force_render=True)
        
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
            f.write(f'succeeded,{benchmark_args.mjcf},{benchmark_args.renderer_name},{benchmark_args.rasterizer},{benchmark_args.n_envs},{benchmark_args.n_steps},{benchmark_args.resX},{benchmark_args.resY},{benchmark_args.camera_posX},{benchmark_args.camera_posY},{benchmark_args.camera_posZ},{benchmark_args.camera_lookatX},{benchmark_args.camera_lookatY},{benchmark_args.camera_lookatZ},{benchmark_args.camera_fov},{time_taken},{time_taken_per_env},{fps},{fps_per_env}\n')
    except Exception as e:
        print(f"Error during benchmark: {e}")
        raise

def main():
    ######################## Parse arguments #######################
    benchmark_args = BenchmarkArgs.parse_args()

    ######################## Initialize scene #######################
    scene = init_gs(benchmark_args)

    ######################## Run benchmark #######################
    run_benchmark(scene, benchmark_args)

if __name__ == "__main__":
    main()

