import os
import cv2
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor


class FrameImageExporter:
    @staticmethod
    def _export_frame_rgb_cam(export_dir, i_cam, i_env, i_step, rgb):
        rgb = rgb[i_env, ..., [2, 1, 0]].cpu().numpy()
        cv2.imwrite(f"{export_dir}/rgb_cam{i_cam}_env{i_env}_{i_step:03d}.png", rgb)

    @staticmethod
    def _export_frame_depth_cam(export_dir, i_cam, i_env, i_step, depth):
        depth = depth[i_env].cpu().numpy()
        cv2.imwrite(f"{export_dir}/depth_cam{i_cam}_env{i_env}_{i_step:03d}.png", depth)

    @staticmethod
    def _worker_export_frame_cam(args):
        export_dir, i_cam, i_env, rgb, depth, i_step = args
        if rgb is not None:
            FrameImageExporter._export_frame_rgb_cam(export_dir, i_cam, i_env, i_step, rgb)
        if depth is not None:
            FrameImageExporter._export_frame_depth_cam(export_dir, i_cam, i_env, i_step, depth)

    def __init__(self, export_dir, depth_clip_max=100, depth_scale="log"):
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
        if self.depth_scale == "log":
            depth = torch.log(depth + 1)

        # Calculate min/max for each image in the batch
        depth_min = depth.amin(dim=(-3, -2), keepdim=True)
        depth_max = depth.amax(dim=(-3, -2), keepdim=True)

        # Normalize to 0-255 range
        return ((depth - depth_min) / (depth_max - depth_min) * 255).to(torch.uint8)

    def export_frame_all_cams(self, i_step, camera_idx=None, rgb=None, depth=None):
        """
        Export frames for all cameras.

        Args:
            i_step: The current step index.
            camera_idx: array of indices of cameras to export. If None, all cameras are exported.
            rgb: RGB image is a list of tensors of shape (n_envs, H, W, 3).
            depth: Depth image is a list of tensors of shape (n_envs, H, W).
        """
        if rgb is None and depth is None:
            print("No rgb or depth to export")
            return
        if rgb is not None:
            assert isinstance(rgb, list) and len(rgb) > 0, "rgb must be a list of tensors with length > 0"
        if depth is not None:
            assert isinstance(depth, list) and len(depth) > 0, "depth must be a list of tensors with length > 0"
        if camera_idx is None:
            camera_idx = range(len(rgb)) if rgb is not None else range(len(depth))
        for i_cam in camera_idx:
            rgb_cam = rgb[i_cam] if i_cam < len(rgb) else None
            depth_cam = depth[i_cam] if i_cam < len(depth) else None
            if rgb_cam is not None or depth_cam is not None:
                self.export_frame_single_cam(i_step, i_cam, rgb_cam, depth_cam)

    def export_frame_single_cam(self, i_step, i_cam, rgb=None, depth=None):
        """
        Export frames for a single camera.

        Args:
            i_step: The current step index.
            i_cam: The index of the camera.
            rgb: RGB image tensor of shape (n_envs, H, W, 3).
            depth: Depth tensor of shape (n_envs, H, W).
        """
        if rgb is not None:
            if isinstance(rgb, np.ndarray):
                rgb = torch.from_numpy(rgb.copy())

            # Unsqueeze rgb to (n_envs, H, W, 3) if n_envs > 0
            if rgb.ndim == 3:
                rgb = rgb.unsqueeze(0)
            assert rgb.ndim == 4, "rgb must be of shape (n_envs, H, W, 3)"

        if depth is not None:
            if isinstance(depth, np.ndarray):
                depth = torch.from_numpy(depth.copy())

            # Unsqueeze depth to (n_envs, H, W, 1) if n_envs > 0
            if depth.ndim == 3:
                depth = depth.unsqueeze(0)
            elif depth.ndim == 2:
                depth = depth.unsqueeze(0).unsqueeze(3)
            depth = self._normalize_depth(depth)
            assert depth.ndim == 4, "depth must be of shape (n_envs, H, W, 1)"

        env_idx = range(rgb.shape[0]) if rgb is not None else range(depth.shape[0])
        args_list = [(self.export_dir, i_cam, i_env, rgb, depth, i_step) for i_env in env_idx]
        with ThreadPoolExecutor() as executor:
            executor.map(FrameImageExporter._worker_export_frame_cam, args_list)
