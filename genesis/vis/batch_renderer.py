import os
import gc

import numpy as np
import genesis as gs
import torch
from genesis.repr_base import RBC
from .camera import Camera
from madrona_mjx.renderer_gs import BatchRendererGS

class Light:
    def __init__(self, pos, dir, intensity, directional, castshadow, cutoff):
        self._pos = pos
        self._dir = dir
        self._intensity = intensity
        self._directional = directional
        self._castshadow = castshadow
        self._cutoff = cutoff

    @property
    def pos(self):
        return self._pos
    
    @property
    def dir(self):
        return self._dir
    
    @property
    def intensity(self):
        return self._intensity
    
    @property
    def directional(self):
        return self._directional
    
    @property
    def castshadow(self):
        return self._castshadow
    
    @property
    def cutoffRad(self):
        return np.deg2rad(self._cutoff)
    
    @property
    def cutoffDeg(self):
        return self._cutoff
    
class BatchRenderer(RBC):
    """
    This class is used to manage batch rendering
    """

    def __init__(self, visualizer, vis_options):
        self._visualizer = visualizer
        self._lights = gs.List()
        self._use_rasterizer = vis_options.use_rasterizer
    
    def add_light(self, pos, dir, intensity, directional, castshadow, cutoff):
        self._lights.append(Light(pos, dir, intensity, directional, castshadow, cutoff))
    
    def build(self):
        """
        Build all cameras in the batch and initialize Moderona renderer
        """
        cameras = self._visualizer._cameras
        if(len(cameras) == 0):
            raise ValueError("No cameras to render")

        self.renderer = BatchRendererGS(
            self._visualizer.scene.rigid_solver,
            torch.cuda.current_device(),
            self._visualizer.scene.n_envs,
            cameras,
            self._lights,
            cameras[0].res[0], # Use first camera's resolution until we support render from separate camera
            cameras[0].res[1],
            False, # add_cam_debug_geo
            self._use_rasterizer, # use_rasterizer
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
        return rgb_torch, depth_torch, None, None
    
    def destroy(self):
        self._lights.clear()
        self.renderer.destroy()
    
    @property
    def lights(self):
        return self._lights
