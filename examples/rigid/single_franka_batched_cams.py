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
                "view_res": (256, 256),
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
        print(f"RGB shape: {rgb.shape}, Depth shape: {depth.shape}")
        print(f"RGB.mean(): {rgb.mean()}, Depth.mean(): {depth.mean()}")


if __name__ == "__main__":
    main()
