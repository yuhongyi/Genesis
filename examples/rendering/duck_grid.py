import json

import genesis as gs
from genesis.utils.image_exporter import FrameImageExporter
from genesis.utils.scene_exporter import SceneDescriptionExporter


def main():
    ########################## init ##########################
    gs.init(backend=gs.gpu, seed=0, precision="32", logging_level="debug")

    ########################## create a scene ##########################
    scene = gs.Scene(
        renderer=gs.options.renderers.BatchRenderer(
            use_rasterizer=True,
        ),
        show_viewer=True,
    )

    ########################## materials ##########################

    ########################## entities ##########################
    # floor
    plane = scene.add_entity(
        morph=gs.morphs.Plane(
            pos=(0.0, 0.0, -2.0),
        ),
        surface=gs.surfaces.Aluminium(
            ior=10.0,
        ),
    )
    # asset's own attributes
    for x in range(-2, 2, 2):
        for y in range(-2, 2, 2):
            for z in range(-2, 2, 2):
                scene.add_entity(
                    morph=gs.morphs.Mesh(
                        file="meshes/duck/duck.obj",
                        scale=0.005,
                        pos=(x, y, z),
                        euler=(90, 0, 0),
                    ),
                )
    ########################## cameras ##########################
    cam_0 = scene.add_camera(
        res=(1600, 900),
        pos=(8.5, 0.0, 1.5),
        lookat=(3.0, 0.0, 0.7),
        fov=60,
        GUI=True,
        spp=512,
    )
    scene.add_light(
        pos=(0.0, 0.0, 1.5),
        dir=(-1.0, -1.0, -1.0),
        color=(1.0, 0.5, 0.0),
        directional=True,
        castshadow=True,
        cutoff=45.0,
        intensity=1.0,
    )

    scene.build()

    scene_description_exporter = SceneDescriptionExporter("duck_grid_scene.json", scene)
    scene_description_exporter.generate_initial_scene_description()

    ########################## forward + backward twice ##########################
    scene.reset()
    horizon = 1000

    # Create an image exporter
    exporter = FrameImageExporter("duck_grid_output")
    for i in range(horizon):
        scene.step()
        scene_description_exporter.capture_frame()
        rgba, depth, _, _ = cam_0.render()
        exporter.export_frame_single_camera(i, cam_0.idx, rgb=rgba, depth=depth)

    scene_description_exporter.export()


if __name__ == "__main__":
    main()
