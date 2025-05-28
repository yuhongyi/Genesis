import argparse

import genesis as gs


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.cpu if args.cpu else gs.gpu)

    ########################## create a scene ##########################
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        show_viewer=args.vis,
        rigid_options=gs.options.RigidOptions(
            # constraint_solver=gs.constraint_solver.Newton,
        ),
        vis_options=gs.options.VisOptions(
            use_batch_renderer=True,
            use_rasterizer=True,
        ),
    )

    ########################## entities ##########################
    #plane = scene.add_entity(
    #    gs.morphs.Plane(),
    #)
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
        visualize_contact=True,
    )

    ########################## cameras ##########################
    cam0 = scene.add_camera(
        res=(512, 512),
        pos=(1.5, 0.5, 1.5),
        lookat=(0.0, 0.0, 0.5),
        fov=45,
    )
    cam1 = scene.add_camera(
        res=(512, 512),
        pos=(1.5, -0.5, 1.5),
        lookat=(0.0, 0.0, 0.5),
        fov=45,
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
        intensity=0.5
    )
    ########################## build ##########################
    n_envs = 2
    n_steps = 1
    scene.build(n_envs=n_envs)

    # warmup
    scene.step()
    rgb, depth, _, _ = scene.batch_render()

    # timer
    from time import time
    start_time = time()

    for i in range(n_steps):
        scene.step()
        rgb, depth, _, _ = scene.batch_render()
        output_rgb_and_depth('img_output/test', rgb, depth, i)
    
    end_time = time()
    print(f'n_envs: {n_envs}')
    print(f'Time taken: {end_time - start_time} seconds')
    print(f'Time taken per env: {(end_time - start_time) / n_envs} seconds')
    print(f'FPS: {n_envs * n_steps / (end_time - start_time)}')
    print(f'FPS per env: {n_steps / (end_time - start_time)}')


import os
import cv2
import numpy as np
# TODO: Dump image faster, e.g., asynchronously or generate a video instead of saving images.
def output_rgb(output_dir, rgb, i_env, i_cam, i_step):
    rgb = rgb.cpu().numpy()[i_env, i_cam]
    cv2.imwrite(f'{output_dir}/rgb_env{i_env}_cam{i_cam}_{i_step:03d}.png', rgb)

def output_depth(output_dir, depth, i_env, i_cam, i_step):
    depth = depth.cpu().numpy()[i_env, i_cam]
    depth = np.asarray(depth)
    depth = np.clip(depth, 0, 100)
    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_normalized.astype(np.uint8)
    cv2.imwrite(f'{output_dir}/depth_env{i_env}_cam{i_cam}_{i_step:03d}.png', depth_uint8)

def output_rgb_and_depth(output_dir, rgb, depth, i_step):
    # loop over the first and second dimension of rgb and depth
    for i_env in range(rgb.shape[0]):
        for i_cam in range(rgb.shape[1]):
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_rgb(output_dir, rgb, i_env, i_cam, i_step)
            output_depth(output_dir, depth, i_env, i_cam, i_step)

def output_rgb_single_cam(output_dir, rgb, i_env, i_step, cam_idx):
    rgb = rgb.cpu().numpy()[i_env]
    cv2.imwrite(f'{output_dir}/rgb_env{i_env}_cam{cam_idx}_{i_step:03d}.png', rgb)

def output_depth_single_cam(output_dir, depth, i_env, i_step, cam_idx):
    depth = depth.cpu().numpy()[i_env]
    depth = np.asarray(depth)
    depth = np.clip(depth, 0, 100)
    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_normalized.astype(np.uint8)
    cv2.imwrite(f'{output_dir}/depth_env{i_env}_cam{cam_idx}_{i_step:03d}.png', depth_uint8)

def output_rgb_and_depth_single_cam(output_dir, rgb, depth, i_step, cam_idx):
    # loop over the first and second dimension of rgb and depth
    for i_env in range(rgb.shape[0]):
        for i_cam in range(rgb.shape[1]):
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_rgb_single_cam(output_dir, rgb, i_env, i_step, i_cam)
            output_depth_single_cam(output_dir, depth, i_env, i_step, i_cam)


if __name__ == "__main__":
    main()
