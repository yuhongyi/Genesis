import os
import gc

import numpy as np
import genesis as gs
import torch
from genesis.repr_base import RBC
from .camera import Camera
from madrona_gs.renderer_gs import BatchRendererGS

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

    def __init__(self, visualizer, renderer_options):
        self._visualizer = visualizer
        self._lights = gs.List()
        self._renderer_options = renderer_options
        self._rgb_torch = None
        self._depth_torch = None
        self._last_t = -1
    
    def add_light(self, pos, dir, intensity, directional, castshadow, cutoff):
        self._lights.append(Light(pos, dir, intensity, directional, castshadow, cutoff))
    
    def build(self):
        """
        Build all cameras in the batch and initialize Moderona renderer
        """
        if(len(self._visualizer._cameras) == 0):
            raise ValueError("No cameras to render")
        cameras = np.array(self._visualizer._cameras)
        lights = np.array(self._lights)
        solver = self._visualizer.scene.rigid_solver
        device = torch.cuda.current_device()
        n_envs = self._visualizer.scene.n_envs if self._visualizer.scene.n_envs > 0 else 1
        res = self._renderer_options.batch_render_res
        use_rasterizer = self._renderer_options.use_rasterizer

        self._renderer = BatchRendererGS(
            solver,
            device,
            n_envs,
            cameras,
            lights,
            res[0],
            res[1],
            False, # add_cam_debug_geo
            use_rasterizer, # use_rasterizer
        )
        self._renderer.init()

    def update_scene(self):
        self._visualizer._context.update()

    def render(self, rgb=True, depth=False, segmentation=False, normal=False):
        """
        Render all cameras in the batch.
        """
        if(self._last_t == self._visualizer.scene.t):
            return self._rgb_torch, self._depth_torch, None, None
        self._last_t = self._visualizer.scene.t # Update last_t to current time to avoid re-rendering if the scene is not updated
        
        # TODO: Control whether to render rgb, depth, segmentation, normal separately
        self.update_scene()
        self._rgb_torch, self._depth_torch = self._renderer.render()
        return self._rgb_torch, self._depth_torch, None, None
    
    def destroy(self):
        self._lights.clear()
        self._renderer.destroy()
        self._rgb_torch = None
        self._depth_torch = None
    
    @property
    def lights(self):
        return self._lights
