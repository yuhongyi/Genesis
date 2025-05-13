import os
import gc

import numpy as np
import genesis as gs
import torch
from genesis.repr_base import RBC
from .camera import Camera
from madrona_mjx.renderer_gs import BatchRendererGS

class Light:
    def __init__(self, pos, dir, directional, castshadow, cutoff):
        self._pos = pos
        self._dir = dir
        self._directional = directional
        self._castshadow = castshadow
        self._cutoff = cutoff

class BatchRenderer(RBC):
    """
    This class is used to manage batch rendering
    """

    def __init__(self, visualizer):
        self._visualizer = visualizer
        self._cameras = gs.List()
        self._lights = gs.List()

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
    
    def add_light(self, pos, dir, directional, castshadow, cutoff):
        self._lights.append(Light(pos, dir, directional, castshadow, cutoff))
    
    def build(self):
        """
        Build all cameras in the batch and initialize Moderona renderer
        """
        if(len(self._cameras) == 0):
            raise ValueError("No cameras to render")

        for camera in self._cameras:
            camera._build()

        self.renderer = BatchRendererGS(
            self._visualizer.scene.rigid_solver,
            torch.cuda.current_device(),
            self._visualizer.scene.n_envs,
            self._cameras,
            self._lights,
            self._cameras[0].res[0], # Use first camera's resolution until we support render from separate camera
            self._cameras[0].res[1],
            False, # add_cam_debug_geo
            False, # use_rasterizer
        )

    def update_scene(self):
        self._visualizer._context.update()

    def render(self, rgb=True, depth=False, segmentation=False, normal=False):
        """
        Render all cameras in the batch.
        """
        # TODO: Control whether to render rgb, depth, segmentation, normal separately
        self.update_scene()
        rgb_torch, depth_torch = self.renderer.render(self._visualizer.scene.rigid_solver)
        return rgb_torch, depth_torch
    
    def destroy(self):
        self._cameras.clear()
        self._lights.clear()
        self.renderer.destroy()

    @property
    def cameras(self):
        return self._cameras
    
    @property
    def lights(self):
        return self._lights
