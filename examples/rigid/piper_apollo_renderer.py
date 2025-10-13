import numpy as np
import time
import genesis as gs
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.gpu, precision="32")
    ########################## create a scene ##########################
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3, -1, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=30,
            res=(960, 640),
            max_FPS=60,
        ),
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        rigid_options=gs.options.RigidOptions(),
        renderer=gs.options.renderers.ApolloRenderer(
            render_mode="forward",
            max_pt_depth=2,
            scene_description_export_path="demo_output/piper_scene_description.json",
        ),
        show_viewer=args.vis,
    )

    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    cam_0 = scene.add_camera(
        res=(512, 512),
        pos=(8.5, 0.0, 1.5),
        lookat=(3.0, 0.0, 0.7),
        fov=60,
        GUI=True,
        spp=16,
        near=0.1,
        far=100.0,
    )
    piper = scene.add_entity(
        gs.morphs.URDF(
            file="piper_description/urdf/piper_description.urdf",
            fixed=True,
            # merge_fixed_links=False,
            # pos=(0,0,0.12)
        ),
        # vis_mode="collision"
    )

    ########################## build ##########################
    scene.build()

    n_step = 1000
    for i in range(n_step):
        print("step", i)

        cam_0.set_pose(lookat=(3.0, 0.0, 0.5 + i * 0.1))
        scene.step()
        rgba0, _, _, _ = cam_0.render()


if __name__ == "__main__":
    main()
