import argparse

import genesis as gs
from genesis.utils.image_exporter import FrameImageExporter


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
        visualize_contact=False,
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
    scene.build()

    # Create an image exporter
    output_dir = 'img_output/test'
    exporter = FrameImageExporter(output_dir)
    for i in range(2):
        scene.step()
        rgb, depth, _, _ = cam_0.render()
        exporter.export_frame_single_cam(i, cam_0.idx, rgb=rgb, depth=depth)


if __name__ == "__main__":
    main()
