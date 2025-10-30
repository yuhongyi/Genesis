import genesis as gs
from genesis.utils.scene_description import SceneDescription
from genesis.utils.image_exporter import FrameImageExporter


def main():
    ########################## init ##########################
    gs.init(seed=0, precision="32", logging_level="info")

    ########################## create a scene ##########################
    scene_description = SceneDescription()
    scene_description.load_from_file("urdf/panda_bullet/panda.json")
    scene = scene_description.scene
    camera = scene.visualizer.cameras[0]
    entity = scene.entities[0]

    print(camera.get_pos(), camera.get_lookat())
    print(entity.get_pos(), entity.get_quat())

    rgb = camera.render()[0]
    exporter = FrameImageExporter("demo_output")
    exporter.export_frame_single_camera(0, camera.idx, rgb=rgb)


if __name__ == "__main__":
    main()
