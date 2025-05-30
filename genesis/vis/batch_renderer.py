import os
import gc

import numpy as np
import genesis as gs
import torch
from genesis.repr_base import RBC
from .camera import Camera
from madrona_gs.renderer_gs import BatchRendererGS
import taichi as ti

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
        cameras = self._visualizer._cameras
        lights = self._lights
        solver = self._visualizer.scene.rigid_solver
        device = torch.cuda.current_device()
        n_envs = self._visualizer.scene.n_envs if self._visualizer.scene.n_envs > 0 else 1
        res = self._renderer_options.batch_render_res
        use_rasterizer = self._renderer_options.use_rasterizer

        # Build taichi arrays to store light properties once.
        # If later we need to support dynamic lights, we should consider storing light properties as taichi fields in Genesis.
        n_lights = len(lights)
        light_pos = ti.ndarray(dtype=ti.f32, shape=(n_lights, 3))
        light_dir = ti.ndarray(dtype=ti.f32, shape=(n_lights, 3))
        light_intensity = ti.ndarray(dtype=ti.f32, shape=(n_lights,))
        light_directional = ti.ndarray(dtype=ti.u8, shape=(n_lights,))
        light_castshadow = ti.ndarray(dtype=ti.u8, shape=(n_lights,))
        light_cutoff = ti.ndarray(dtype=ti.f32, shape=(n_lights,))

        # Fill the arrays with light data
        # Convert all light properties to numpy arrays first
        pos_array = np.array([light.pos for light in lights], dtype=np.float32)
        dir_array = np.array([light.dir for light in lights], dtype=np.float32)
        intensity_array = np.array([light.intensity for light in lights], dtype=np.float32)
        directional_array = np.array([light.directional for light in lights], dtype=np.uint8)
        castshadow_array = np.array([light.castshadow for light in lights], dtype=np.uint8)
        cutoff_array = np.array([light.cutoffRad for light in lights], dtype=np.float32)

        # Fill the taichi arrays with the concatenated numpy arrays
        light_pos.from_numpy(pos_array)
        light_dir.from_numpy(dir_array)
        light_intensity.from_numpy(intensity_array)
        light_directional.from_numpy(directional_array)
        light_castshadow.from_numpy(castshadow_array)
        light_cutoff.from_numpy(cutoff_array)

        self._renderer = BatchRendererGS(
            solver,
            device,
            n_envs,
            cameras,
            light_pos,
            light_dir,
            light_intensity,
            light_directional,
            light_castshadow,
            light_cutoff,
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
