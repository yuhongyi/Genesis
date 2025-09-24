import json

import genesis as gs
from genesis.utils.image_exporter import FrameImageExporter


def main():
    ########################## init ##########################
    gs.init(seed=0, precision="32", logging_level="debug")

    ########################## create a scene ##########################
    scene = gs.Scene(
        renderer=gs.options.renderers.ApolloRenderer(
            render_mode="forward",
            max_pt_depth=2,
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

    # user specified external color texture
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/sphere.obj",
            scale=0.5,
            pos=(0.0, -3, 0.0),
        ),
        surface=gs.surfaces.Rough(
            diffuse_texture=gs.textures.ColorTexture(
                color=(1.0, 0.5, 0.5),
            ),
        ),
    )
    # user specified color (using color shortcut)
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/sphere.obj",
            scale=0.5,
            pos=(0.0, -1.8, 0.0),
        ),
        surface=gs.surfaces.Rough(
            color=(1.0, 1.0, 1.0),
        ),
    )
    # smooth shortcut
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/sphere.obj",
            scale=0.5,
            pos=(0.0, -0.6, 0.0),
        ),
        surface=gs.surfaces.Smooth(
            color=(0.6, 0.8, 1.0),
        ),
    )
    # Iron
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/sphere.obj",
            scale=0.5,
            pos=(0.0, 0.6, 0.0),
        ),
        surface=gs.surfaces.Iron(
            color=(1.0, 1.0, 1.0),
        ),
    )
    # Gold
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/sphere.obj",
            scale=0.5,
            pos=(0.0, 1.8, 0.0),
        ),
        surface=gs.surfaces.Gold(
            color=(1.0, 1.0, 1.0),
        ),
    )
    # Glass
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/sphere.obj",
            scale=0.5,
            pos=(0.0, 3.0, 0.0),
        ),
        surface=gs.surfaces.Glass(
            color=(1.0, 1.0, 1.0),
        ),
    )
    # Opacity
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/sphere.obj",
            scale=0.5,
            pos=(2.0, -3, 0.0),
        ),
        surface=gs.surfaces.Smooth(color=(1.0, 1.0, 1.0, 0.5)),
    )
    # asset's own attributes
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/wooden_sphere_OBJ/wooden_sphere.obj",
            scale=0.15,
            pos=(2.2, -2.3, 0.0),
        ),
    )
    # override asset's attributes
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/wooden_sphere_OBJ/wooden_sphere.obj",
            scale=0.15,
            pos=(2.2, -1.0, 0.0),
        ),
        surface=gs.surfaces.Rough(
            diffuse_texture=gs.textures.ImageTexture(
                image_path="textures/checker.png",
            )
        ),
    )
    ########################## cameras ##########################
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
    cam_1 = scene.add_camera(
        res=(1024, 1024),
        pos=(6.5, 0.0, 1.5),
        lookat=(3.0, 0.0, 0.5),
        fov=45,
        GUI=True,
        spp=16,
        near=0.1,
        far=100.0,
    )
    scene.add_light(
        pos=(0.0, 0.0, 1.5),
        dir=(-1.0, -1.0, -1.0),
        color=(1.0, 1.0, 1.0),
        directional=True,
        castshadow=True,
        cutoff=45.0,
        intensity=5.0,
    )

    scene.build()

    ########################## forward + backward twice ##########################
    STEPS = 10

    # Create an image exporter
    exporter = FrameImageExporter("demo_output")
    for i in range(STEPS):
        cam_1.set_pose(lookat=(3.0, 0.0, 0.5 + i * 0.1))
        scene.step()
        rgba0, _, _, _ = cam_0.render()
        rgba1, _, _, _ = cam_1.render()
        exporter.export_frame_single_camera(i, cam_0.idx, rgb=rgba0)
        exporter.export_frame_single_camera(i, cam_1.idx, rgb=rgba1)


if __name__ == "__main__":
    main()
