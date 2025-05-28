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
    import cv2
    ########################## build ##########################
    import os
    if not os.path.exists('img_output'):
        os.makedirs('img_output')
    scene.build()
    for i in range(10):
        scene.step()
        rgb, depth, seg, normal = cam_0.render()
        cv2.imwrite(f'img_output/rgb_{i}.png', i)

if __name__ == "__main__":
    main()
