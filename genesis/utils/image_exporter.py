import os
import cv2
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor

class FrameImageExporter:
    @staticmethod
    def _export_frame_rgb_cam(export_dir, i_env, i_cam, camera_name, i_step, rgb):
        rgb = rgb[i_env, i_cam, ..., [2, 1, 0]].cpu().numpy()
        cv2.imwrite(f'{export_dir}/rgb_env{i_env}_{camera_name}_{i_step:03d}.png', rgb)

    @staticmethod
    def _export_frame_depth_cam(export_dir, i_env, i_cam, camera_name, i_step, depth_normalized):
        depth_normalized = depth_normalized[i_env, i_cam].cpu().numpy()
        cv2.imwrite(f'{export_dir}/depth_env{i_env}_{camera_name}_{i_step:03d}.png', depth_normalized)

    @staticmethod
    def _worker_export_frame_cam(args):
        export_dir, i_env, i_cam, camera_name, rgb, depth_normalized, i_step = args
        if rgb is not None:
            FrameImageExporter._export_frame_rgb_cam(export_dir, i_env, i_cam, camera_name, i_step, rgb)
        if depth_normalized is not None:
            FrameImageExporter._export_frame_depth_cam(export_dir, i_env, i_cam, camera_name, i_step, depth_normalized)

    def __init__(self, export_dir, depth_clip_max=100, depth_scale='log'):
        self.export_dir = export_dir
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        self.depth_clip_max = depth_clip_max
        self.depth_scale = depth_scale

    def _normalize_depth(self, depth):
        """Normalize depth values for visualization.
        
        Args:
            depth: Tensor of depth values
            
        Returns:
            Normalized depth tensor as uint8
        """
        # Clip depth values
        depth = depth.clamp(0, self.depth_clip_max)
        
        # Apply scaling if specified
        if self.depth_scale == 'log':
            depth = torch.log(depth + 1)
            
        # Calculate min/max for each image in the batch
        depth_min = depth.amin(dim=(-3, -2), keepdim=True)
        depth_max = depth.amax(dim=(-3, -2), keepdim=True)
        
        # Normalize to 0-255 range
        return ((depth - depth_min) / (depth_max - depth_min) * 255).to(torch.uint8)
    
    def _get_camera_name(self, i_cam):
        return 'cam' + str(i_cam)

    def export_frame_batch_cam(self, i_step, camera_idx=None, rgb=None, depth=None):
        """
        Export frames for a batch of environments and cameras.

        Args:
            i_step: The current step index.
            camera_idx: array of indices of cameras to export. If None, all cameras are exported.
            rgb: RGB image tensor of shape (n_envs, n_cams, H, W, 3).
            depth: Depth tensor of shape (n_envs, n_cams, H, W).
        """

        assert rgb.ndim == 5 and depth.ndim == 5, "rgb and depth must be of shape (n_envs, n_cams, H, W, 3)"
        
        if camera_idx is None:
            camera_idx = range(rgb.shape[1])
        
        depth_normalized = self._normalize_depth(depth) if depth is not None else None
        
        args_list = [(self.export_dir, i_env, i_cam, self._get_camera_name(i_cam), rgb, depth_normalized, i_step) 
                     for i_env in range(rgb.shape[0]) for i_cam in camera_idx]
        with ThreadPoolExecutor() as executor:
            executor.map(FrameImageExporter._worker_export_frame_cam, args_list)

    def export_frame_single_cam(self, i_step, i_cam, rgb=None, depth=None):
        """
        Export frames for a single camera.

        Args:
            i_step: The current step index.
            cam_idx: The index of the camera.
            rgb: RGB image tensor of shape (n_envs, H, W, 3).
            depth: Depth tensor of shape (n_envs, H, W).
        """
        # Move rgb and depth to torch tensor
        if isinstance(rgb, np.ndarray):
            rgb = torch.from_numpy(rgb.copy())
        if isinstance(depth, np.ndarray):
            depth = torch.from_numpy(depth.copy())
        
        # Unsqueeze rgb and depth to (n_envs, 1, H, W, 3) and (n_envs, 1, H, W, 1)
        if rgb.ndim == 4:
            rgb = rgb.unsqueeze(1)
        elif rgb.ndim == 3:
            rgb = rgb.unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError(f"Invalid rgb shape: {rgb.shape}")
        
        if depth.ndim == 4:
            depth = depth.unsqueeze(1)
        elif depth.ndim == 2:
            depth = depth.unsqueeze(0).unsqueeze(0).unsqueeze(4)
        else:
            raise ValueError(f"Invalid depth shape: {depth.shape}")

        depth_normalized = self._normalize_depth(depth) if depth is not None else None
        
        args_list = [(self.export_dir, i_env, 0, self._get_camera_name(i_cam), rgb, depth_normalized, i_step) 
                     for i_env in range(rgb.shape[0])]
        with ThreadPoolExecutor() as executor:
            executor.map(FrameImageExporter._worker_export_frame_cam, args_list) 
