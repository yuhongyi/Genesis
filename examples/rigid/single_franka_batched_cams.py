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
    batch_renderer = scene.add_batch_cameras(
        cameras = [
            {
                "res": (256, 256),
                "pos": (3.5, 0.0, 2.5),
                "lookat": (0, 0, 0.5),
                "fov": 40,
            }
        ],
    )

    scene.add_batch_render_light(
        pos=[-0.13820243, 0.12866199, 2.0],
        dir=[0.0, 0.0, -1.0],
        directional=0,
        castshadow=1,
        cutoff=1.1808988
    )

    scene.add_batch_render_light(
        pos=[0.0, 0.0, 1.5],
        dir=[0.0, 0.0, -1.0],
        directional=1,
        castshadow=1,
        cutoff=45.0
    )
    ########################## build ##########################
    scene.build()
    for i in range(10):
        scene.step()
        rgb, depth = batch_renderer.render()
        output_rgb_and_depth('img_output', rgb, depth, i)


import os
import cv2
import numpy as np
# TODO: Dump image faster, e.g., asynchronously or generate a video instead of saving images.
def output_rgb(output_dir, rgb, i_env, i_cam, i_step):
    rgb = rgb.cpu().numpy()[i_env, i_cam]
    print(f'rgb_{i_env}_{i_cam}_{i_step}: {rgb.shape}, {rgb.mean()}')
    cv2.imwrite(f'{output_dir}/rgb_env{i_env}_cam{i_cam}_{i_step}.png', rgb)

def output_depth(output_dir, depth, i_env, i_cam, i_step):
    depth = depth.cpu().numpy()[i_env, i_cam]
    depth = np.asarray(depth)
    depth = np.clip(depth, 0, 100)
    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_normalized.astype(np.uint8)
    cv2.imwrite(f'{output_dir}/depth_env{i_env}_cam{i_cam}_{i_step}.png', depth_uint8)

def output_rgb_and_depth(output_dir, rgb, depth, i_step):
    # loop over the first and second dimension of rgb and depth
    for i_env in range(rgb.shape[0]):
        for i_cam in range(rgb.shape[1]):
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_rgb(output_dir, rgb, i_env, i_cam, i_step)
            output_depth(output_dir, depth, i_env, i_cam, i_step)

if __name__ == "__main__":
    main()
