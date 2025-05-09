import pyglet

import genesis as gs
from genesis.repr_base import RBC

from .camera import Camera
from .rasterizer import Rasterizer
from .madrona_rasterizer import MadronaRasterizer
import numpy as np


VIEWER_DEFAULT_HEIGHT_RATIO = 0.5
VIEWER_DEFAULT_ASPECT_RATIO = 0.75


class DummyViewerLock:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass

from genesis.utils.geom import T_to_trans_quat
import torch
class BatchRenderer:
    """
    This class is used to manage batch cameras.
    """

    def __init__(self, visualizer):
        self._visualizer = visualizer
        self._cameras = gs.List()

    def add_camera(self, res, pos, lookat, up, model, fov, aperture, focus_dist, GUI, spp, denoise):
        camera = Camera(
            self._visualizer,
            len(self._cameras),
            model,
            res,
            pos,
            lookat,
            up,
            fov,
            aperture,
            focus_dist,
            GUI,
            spp,
            denoise
        )
        self._cameras.append(camera)
        return camera
    
    def build(self):
        """
        Build all cameras in the batch and initialize Moderona renderer
        """
        for camera in self._cameras:
            camera._build(batch_camera = True)

        # Build madrona batch renderer
        gpu_id = 0
        self.num_worlds = 2
        batch_render_view_res = (256, 256)
        enabled_geom_groups = [0, 1, 2, 3, 4, 5]
        add_cam_debug_geo = False
        use_rasterizer = True

        self.rigid = self._visualizer.scene.rigid_solver
        mesh_vertices = self.rigid.vverts_info.init_pos.to_numpy()
        mesh_faces = self.rigid.vfaces_info.vverts_idx.to_numpy()
        mesh_vertex_offsets = self.rigid.vgeoms_info.vvert_start.to_numpy()
        mesh_face_offsets = self.rigid.vgeoms_info.vface_start.to_numpy()
        n_vgeom = self.rigid.n_vgeoms

        face_start = self.rigid.vgeoms_info.vface_start.to_numpy()
        face_end = self.rigid.vgeoms_info.vface_end.to_numpy()
        for i in range(n_vgeom):
            mesh_faces[face_start[i] : face_end[i]] -= mesh_vertex_offsets[i]

        mesh_texcoords = np.zeros([0, 2], dtype=np.float32)
        mesh_texcoord_offsets = np.full((n_vgeom,), -1, dtype=np.int32)
        mesh_texcoord_num = np.full((n_vgeom,), 0, dtype=np.int32)
        geom_types = np.full((n_vgeom,), 7, dtype=np.int32)
        geom_groups = np.full((n_vgeom,), 2, dtype=np.int32)
        geom_data_ids = np.arange(n_vgeom, dtype=np.int32)
        geom_sizes = np.ones((n_vgeom, 3), dtype=np.float32)
        geom_mat_ids = np.full((n_vgeom,), -1, dtype=np.int32)
        geom_rgba = np.array([[0.8,0.6,0.4,1]]*n_vgeom, dtype=np.float32)
        mat_rgba = np.array([[0.8,0.6,0.4,1]]*6, dtype=np.float32)
        num_lights = 1
        num_cams = len(self._cameras) if self._cameras is not None else 0
        assert num_cams > 0, "Must have at least one camera for Madrona to work!"

        mat_tex_ids = np.full((6, 10), -1, dtype=np.int32)
        mat_tex_ids[-1,1] = 1
        tex_data = np.full((4988592), 127, dtype=np.uint8)
        # add 255 every third element to create 4 channel rgba texture
        tex_data = np.insert(
            tex_data, np.arange(3, tex_data.shape[0], 3), 255, axis=0
        )
        tex_offsets = np.array([0, 4718592], dtype=np.int32)
        tex_widths = np.array([512, 300], dtype=np.int32)
        tex_heights = np.array([3072,  300], dtype=np.int32)
        tex_nchans = np.array([3, 3], dtype=np.int32)
        cam_fovy = np.array([45.0], dtype=np.float32)

        self.madrona = MadronaBatchRenderer(
            gpu_id=gpu_id,
            mesh_vertices=mesh_vertices,
            mesh_faces=mesh_faces,
            mesh_vertex_offsets=mesh_vertex_offsets,
            mesh_face_offsets=mesh_face_offsets,
            mesh_texcoords=mesh_texcoords,
            mesh_texcoord_offsets=mesh_texcoord_offsets,
            mesh_texcoord_num=mesh_texcoord_num,
            geom_types=geom_types,
            geom_groups=geom_groups,
            geom_data_ids=geom_data_ids,
            geom_sizes=geom_sizes,
            geom_mat_ids=geom_mat_ids,
            geom_rgba=geom_rgba,
            mat_rgba=mat_rgba,
            mat_tex_ids=mat_tex_ids,

            tex_data=tex_data,
            tex_offsets=tex_offsets,
            tex_widths=tex_widths,
            tex_heights=tex_heights,
            tex_nchans=tex_nchans,
            num_lights=num_lights,
            num_cams=num_cams,
            num_worlds=self.num_worlds,

            batch_render_view_width=batch_render_view_res[0],
            batch_render_view_height=batch_render_view_res[1],
            cam_fovy=cam_fovy,
            enabled_geom_groups=enabled_geom_groups,
            add_cam_debug_geo=add_cam_debug_geo,
            use_rt=not use_rasterizer,
        )
    
        cam_pos, cam_rot = self.get_camera_pos_rot()
        
        geom_pos = self.rigid.vgeoms_state.pos.to_numpy()
        geom_rot = self.rigid.vgeoms_state.quat.to_numpy()
        geom_pos = np.swapaxes(geom_pos, 0, 1)
        geom_rot = np.swapaxes(geom_rot, 0, 1)

        # TODO: Move to scene.add_light()
        light_pos = np.array([[-0.13820243,  0.12866199,  2. ],
        [0. , 0. , 1.5]], dtype=np.float32)
        light_dir = np.array([[ 0.,  0., -1.],
        [ 0.,  0., -1.]], dtype=np.float32)
        light_directional = np.array([0, 1], dtype=np.uint8)
        light_castshadow = np.array([1, 1], dtype=np.uint8)
        light_cutoff = np.array([1.1808988, 45.], dtype=np.float32)

        print("init")
        print("cam_pos", cam_pos)
        print("cam_rot", cam_rot)
        print("cam_fovy", cam_fovy)
        print("geom_pos", geom_pos)
        print("geom_rot", geom_rot)
        print("light_pos", light_pos)
        print("light_dir", light_dir)
        print("light_directional", light_directional)
        print("light_castshadow", light_castshadow)
        print("light_cutoff", light_cutoff)
        print("geom_mat_ids", geom_mat_ids)
        print("geom_rgba", geom_rgba)
        print("geom_sizes", geom_sizes)

        geom_rgba_uint = np.array(geom_rgba * 255, np.uint32) 
        geom_rgb = geom_rgba_uint[...,0] * (1 << 16) + geom_rgba_uint[...,1] * (1 << 8) + geom_rgba_uint[...,2]

        print("geom_rgb", geom_rgb)
        # Make a copy to actually shuffle the memory layout before passing to C++
        self.madrona.init(
            geom_pos.copy(),
            geom_rot.copy(),
            cam_pos.copy(),
            cam_rot.copy(),
            np.repeat(geom_mat_ids[np.newaxis], self.num_worlds, axis=0),
            np.repeat(geom_rgb[np.newaxis], self.num_worlds, axis=0),
            np.repeat(geom_sizes[np.newaxis], self.num_worlds, axis=0),
            light_pos,
            light_dir,
            light_directional,
            light_castshadow,
            light_cutoff,
        )

    def render(self, rgb=True, depth=False, segmentation=False, normal=False):
        """
        Render all cameras in the batch.
        """
        #TODO: implement this
        print("here is rendering")
        # Assume execution on GPU
        # TODO: Need to check if the device is GPU or CPU, or assert if not GPU
        cam_pos, cam_rot = self.get_camera_pos_rot_gpu()
        geom_pos, geom_rot = self.get_geom_pos_rot_gpu()
        #print("geom_pos", geom_pos.shape, geom_pos.dtype, geom_pos.is_cuda, geom_pos.device)
        #print("geom_rot", geom_rot.shape, geom_rot.dtype, geom_rot.is_cuda, geom_rot.device)
        #print("cam_pos", cam_pos.shape, cam_pos.dtype, cam_pos.is_cuda, cam_pos.device)
        #print("cam_rot", cam_rot.shape, cam_rot.dtype, cam_rot.is_cuda, cam_rot.device)

        self.madrona.render_torch(
            geom_pos,
            geom_rot,
            cam_pos,
            cam_rot,
        )
        #rgb_torch = self.madrona.rgb_tensor().to_torch()
        #depth_torch = self.madrona.depth_tensor().to_torch()    
        #return rgb_torch, depth_torch

    ########################## Utils ##########################
    def get_camera_pos_rot(self):
        trans_list, quat_list = zip(*[T_to_trans_quat(c.transform) for c in self._cameras])
        cam_pos = np.array(trans_list, dtype=np.float32)
        cam_rot = np.array(quat_list, dtype=np.float32)
        cam_pos = np.repeat(cam_pos[None], self.num_worlds, axis=0)
        cam_rot = np.repeat(cam_rot[None], self.num_worlds, axis=0)
        return cam_pos, cam_rot

    def get_camera_pos_rot_gpu(self):
        cam_pos, cam_rot = self.get_camera_pos_rot()
        cam_pos = torch.tensor(cam_pos).to("cuda")
        cam_rot = torch.tensor(cam_rot).to("cuda")
        return cam_pos, cam_rot

    def get_geom_pos_rot_gpu(self):
        geom_pos = self.rigid.vgeoms_state.pos.to_torch()
        geom_rot = self.rigid.vgeoms_state.quat.to_torch()
        geom_pos = geom_pos.transpose(0, 1).contiguous().to("cuda")
        geom_rot = geom_rot.transpose(0, 1).contiguous().to("cuda")
        return geom_pos, geom_rot

class Visualizer(RBC):
    """
    This abstraction layer manages viewer and renderers.
    """

    def __init__(self, scene, show_viewer, vis_options, viewer_options, renderer):
        self._t = -1
        self._scene = scene

        self._context = None
        self._viewer = None
        self._rasterizer = None
        self._raytracer = None
        self._batch_rasterizer = None

        # Rasterizer context is shared by viewer and rasterizer
        try:
            from .viewer import Viewer
            from .rasterizer_context import RasterizerContext
            #from .madrona_rasterizer_context import MadronaRasterizerContext

        except Exception as e:
            gs.raise_exception_from("Rendering not working on this machine.", e)
        self._context = RasterizerContext(vis_options)
        #self._context = MadronaRasterizerContext(vis_options)

        # try to connect to display
        try:
            if pyglet.version < "2.0":
                display = pyglet.canvas.Display()
                screen = display.get_default_screen()
                scale = 1.0
            else:
                display = pyglet.display.get_display()
                screen = display.get_default_screen()
                scale = screen.get_scale()
            self._connected_to_display = True
        except Exception as e:
            if show_viewer:
                gs.raise_exception_from("No display detected. Use `show_viewer=False` for headless mode.", e)
            self._connected_to_display = False

        if show_viewer:
            if viewer_options.res is None:
                viewer_height = (screen.height * scale) * VIEWER_DEFAULT_HEIGHT_RATIO
                viewer_width = viewer_height / VIEWER_DEFAULT_ASPECT_RATIO
                viewer_options.res = (int(viewer_width), int(viewer_height))
            if viewer_options.run_in_thread is None:
                if gs.platform == "Linux":
                    viewer_options.run_in_thread = True
                elif gs.platform == "macOS":
                    viewer_options.run_in_thread = False
                    gs.logger.warning(
                        "Mac OS detected. The interactive viewer will only be responsive if a simulation is running."
                    )
                elif gs.platform == "Windows":
                    viewer_options.run_in_thread = True
            if gs.platform == "macOS" and viewer_options.run_in_thread:
                gs.raise_exception("Running viewer in background thread is not supported on MacOS.")

            self._viewer = Viewer(viewer_options, self._context)

        # Rasterizer is always needed for depth and segmentation mask rendering.
        self._rasterizer = Rasterizer(self._viewer, self._context)
        self._batch_rasterizer = MadronaRasterizer(self._viewer, self._context)

        if isinstance(renderer, gs.renderers.RayTracer):
            from .raytracer import Raytracer

            self._renderer = self._raytracer = Raytracer(renderer, vis_options)

        else:
            #self._renderer = self._rasterizer
            self._renderer = self._batch_rasterizer
            self._raytracer = None

        self._cameras = gs.List()
        self._batch_renderer = BatchRenderer(self)

    def __del__(self):
        self.destroy()

    def destroy(self):
        if self._viewer is not None:
            self._viewer.stop()
            self._viewer = None
        if self._rasterizer is not None:
            self._rasterizer.destroy()
            self._rasterizer = None
        if self._batch_rasterizer is not None:
            self._batch_rasterizer.destroy()
            self._batch_rasterizer = None
        if self._raytracer is not None:
            self._raytracer.destroy()
            self._raytracer = None
        if self._context is not None:
            self._context.destroy()
            del self._context
            self._context = None
        self._renderer = None

    def add_camera(self, res, pos, lookat, up, model, fov, aperture, focus_dist, GUI, spp, denoise):
        camera = Camera(
            self, len(self._cameras), model, res, pos, lookat, up, fov, aperture, focus_dist, GUI, spp, denoise
        )
        self._cameras.append(camera)
        return camera
    
    def add_batch_camera(self, res, pos, lookat, up, model, fov, aperture, focus_dist, GUI, spp, denoise):
        camera = self._batch_renderer.add_camera(res, pos, lookat, up, model, fov, aperture, focus_dist, GUI, spp, denoise)
        return camera

    def reset(self):
        self._t = -1

        self._context.reset()

        # temp fix for cam.render() segfault
        if self._viewer is not None:
            # need to update viewer once here, because otherwise camera will update scene if render is called right
            # after build, which will lead to segfault.
            # TODO: this slows down visualizer.update(). Needs to remove this once the bug is fixed.
            try:
                self._viewer.update(auto_refresh=True)
            except:
                pass

        if self._raytracer is not None:
            self._raytracer.reset()

    def build(self):
        self._context.build(self._scene)

        if self._viewer is not None:
            self._viewer.build(self._scene)
            self.viewer_lock = self._viewer.lock
        else:
            self.viewer_lock = DummyViewerLock()

        self._rasterizer.build()
        self._batch_rasterizer.build()
        if self._raytracer is not None:
            self._raytracer.build(self._scene)

        self._batch_renderer.build()

        for camera in self._cameras:
            camera._build()

        if self._cameras:
            # need to update viewer once here, because otherwise camera will update scene if render is called right
            # after build, which will lead to segfault.
            if self._viewer is not None:
                self._viewer.update(auto_refresh=True)
            else:
                # viewer creation will compile rendering kernels if viewer is not created, render here once to compile
                # TODO: Is this still necessary with batch rasterizer?
                self._rasterizer.render_camera(self._cameras[0])

    def update(self, force=True, auto=None):
        if force:  # force update
            self.reset()
        elif self._viewer is not None:
            if self._viewer.is_alive():
                self._viewer.update(auto_refresh=auto)
            else:
                gs.raise_exception("Viewer closed.")

    def update_visual_states(self):
        """
        Update all visualization-only variables here.
        """
        if self._t < self._scene._t:
            self._t = self._scene._t

            for camera in self._cameras:
                if camera._attached_link is not None:
                    camera.move_to_attach()

            if self._scene.rigid_solver.is_active():
                self._scene.rigid_solver.update_geoms_render_T()
                self._scene.rigid_solver._kernel_update_vgeoms()

                # drone propellers
                for entity in self._scene.rigid_solver.entities:
                    if isinstance(entity, gs.engine.entities.DroneEntity):
                        entity.update_propeller_vgeoms()

                self._scene.rigid_solver.update_vgeoms_render_T()

            if self._scene.avatar_solver.is_active():
                self._scene.avatar_solver.update_geoms_render_T()
                self._scene.avatar_solver._kernel_update_vgeoms()
                self._scene.avatar_solver.update_vgeoms_render_T()

            if self._scene.mpm_solver.is_active():
                self._scene.mpm_solver.update_render_fields()

            if self._scene.sph_solver.is_active():
                self._scene.sph_solver.update_render_fields()

            if self._scene.pbd_solver.is_active():
                self._scene.pbd_solver.update_render_fields()

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def viewer(self):
        return self._viewer

    @property
    def rasterizer(self):
        return self._rasterizer
    
    @property
    def batch_rasterizer(self):
        return self._batch_rasterizer

    @property
    def context(self):
        return self._context

    @property
    def raytracer(self):
        return self._raytracer

    @property
    def renderer(self):
        return self._renderer

    @property
    def scene(self):
        return self._scene

    @property
    def connected_to_display(self):
        return self._connected_to_display

    @property
    def cameras(self):
        return self._cameras
