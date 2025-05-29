import argparse
import os
import cv2

import genesis as gs
import numpy as np


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
            use_batch_renderer=False,
            use_rasterizer=True,
        ),
    )

    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
        visualize_contact=True,
    )

    ########################## cameras ##########################
    cam_0 = scene.add_camera(
        res=(1280, 960),
        pos=(3.5, 0.0, 2.5),
        lookat=(0, 0, 0.5),
        fov=30,
        GUI=True,
    )
    ########################## build ##########################
    if not os.path.exists('img_output'):
        os.makedirs('img_output')
    scene.build()
    for i in range(10):
        scene.step()
        rgb, depth, seg, normal = cam_0.render(rgb=True, depth=True)
        output_rgb_and_depth('img_output', rgb, depth, i, cam_0.idx)

def output_rgb(output_dir, rgb, i_step, cam_idx):
    rgb[..., [0, 2]] = rgb[..., [2, 0]]
    cv2.imwrite(f'{output_dir}/rgb_cam{cam_idx}_{i_step:03d}.png', rgb)

def output_depth(output_dir, depth, i_step, cam_idx):
    depth = np.clip(depth, 0, 100)
    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_normalized.astype(np.uint8)
    cv2.imwrite(f'{output_dir}/depth_cam{cam_idx}_{i_step:03d}.png', depth_uint8)

def output_rgb_and_depth(output_dir, rgb, depth, i_step, cam_idx):
    #swap r and b channels
    rgb[..., [0, 2]] = rgb[..., [2, 0]]
    # loop over the first and second dimension of rgb and depth
    for i_env in range(rgb.shape[0]):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_rgb(output_dir, rgb, i_step, cam_idx)
        output_depth(output_dir, depth, i_step, cam_idx)

if __name__ == "__main__":
    main()
