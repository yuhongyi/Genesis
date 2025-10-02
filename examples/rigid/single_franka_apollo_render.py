import argparse
import json
import numpy as np

import genesis as gs
from genesis.utils.geom import trans_to_T
from genesis.utils.image_exporter import FrameImageExporter
from genesis.utils.scene_exporter import SceneDescriptionExporter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    parser.add_argument("-b", "--n_envs", type=int, default=0)
    parser.add_argument("-s", "--n_steps", type=int, default=200)
    parser.add_argument("-r", "--render_all_cameras", action="store_true", default=False)
    parser.add_argument("-o", "--output_dir", type=str, default="demo_output")
    parser.add_argument("-u", "--use_rasterizer", action="store_true", default=False)
    parser.add_argument("-d", "--debug", action="store_true", default=False)
    parser.add_argument("-l", "--seg_level", type=str, default="link")
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.cpu if args.cpu else gs.gpu)

    ########################## create a scene ##########################
    scene = gs.Scene(
        renderer=gs.options.renderers.ApolloRenderer(
            render_mode="forward",
            max_pt_depth=2,
        ),
    )

    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    franka_mjcf = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
            pos=(-0.5, -0.5, 0.0),
        ),
    )
    franka_urdf = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/panda_bullet/panda.urdf",
            pos=(-0.5, 0.5, 0.0),
        ),
    )

    ########################## cameras ##########################
    debug_cam = scene.add_camera(
        res=(720, 1280),
        pos=(1.5, -0.5, 1.0),
        lookat=(0.0, 0.0, 0.5),
        fov=60,
        GUI=args.vis,
        debug=True,
    )
    cam_0 = scene.add_camera(
        res=(512, 512),
        pos=(1.5, 0.5, 1.5),
        lookat=(0.0, 0.0, 0.5),
        fov=45,
        GUI=args.vis,
    )
    cam_0.attach(franka_mjcf.links[6], trans_to_T(np.array([0.0, 0.5, 0.0])))
    cam_1 = scene.add_camera(
        res=(512, 512),
        pos=(1.5, -0.5, 1.5),
        lookat=(0.0, 0.0, 0.5),
        fov=45,
        GUI=args.vis,
    )
    cam_2 = scene.add_camera(
        res=(512, 512),
        pos=(0.0, 0.1, 5.0),
        lookat=(0.0, 0.0, 0.0),
        fov=45,
        GUI=args.vis,
    )

    scene.add_light(
        pos=(0.0, 0.0, 1.5),
        dir=(1.0, 1.0, -2.0),
        color=(1.0, 1.0, 1.0),
        directional=True,
        castshadow=True,
        cutoff=45.0,
        intensity=5,
    )
    scene.add_light(
        pos=(4, -4, 4),
        dir=(0, 0, -1),
        directional=False,
        castshadow=True,
        cutoff=80.0,
        intensity=1.0,
        attenuation=0.1,
    )

    ########################## build ##########################
    scene.build(n_envs=args.n_envs)

    scene_description_exporter = SceneDescriptionExporter(scene)

    # Create an image exporter
    exporter = FrameImageExporter(args.output_dir)

    for i in range(args.n_steps):
        scene.step()
        scene_description_exporter.capture_frame()
        rgba0, _, _, _ = cam_0.render()
        rgba1, _, _, _ = cam_1.render()
        rgba2, _, _, _ = cam_2.render()
        exporter.export_frame_single_camera(i, cam_0.idx, rgb=rgba0)
        exporter.export_frame_single_camera(i, cam_1.idx, rgb=rgba1)
        exporter.export_frame_single_camera(i, cam_2.idx, rgb=rgba2)

    scene_description_exporter.export_to_file("demo_output/franka_scene_description.json")


if __name__ == "__main__":
    main()
