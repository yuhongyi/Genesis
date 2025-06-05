import os
import cv2
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor

class FrameImageExporter:
    @staticmethod
    def _export_frame_rgb_batch_cam(export_dir, i_env, i_cam, i_step, rgb):
        rgb = rgb[i_env, i_cam, ..., [2, 1, 0]].cpu().numpy()
        cv2.imwrite(f'{export_dir}/rgb_env{i_env}_cam{i_cam}_{i_step:03d}.png', rgb)

    @staticmethod
    def _export_frame_depth_batch_cam(export_dir, i_env, i_cam, i_step, depth_normalized):
        depth_normalized = depth_normalized[i_env, i_cam].cpu().numpy()
        cv2.imwrite(f'{export_dir}/depth_env{i_env}_cam{i_cam}_{i_step:03d}.png', depth_normalized)

    @staticmethod
    def _export_frame_rgb_single_cam(export_dir, i_env, i_step, cam_idx, rgb):
        rgb = rgb[i_env, ..., [2, 1, 0]].cpu().numpy()
        cv2.imwrite(f'{export_dir}/rgb_env{i_env}_cam{cam_idx}_{i_step:03d}.png', rgb)

    @staticmethod
    def _export_frame_depth_single_cam(export_dir, i_env, i_step, cam_idx, depth_normalized):
        depth_normalized = depth_normalized[i_env].cpu().numpy()
        cv2.imwrite(f'{export_dir}/depth_env{i_env}_cam{cam_idx}_{i_step:03d}.png', depth_normalized)

    @staticmethod
    def _worker_export_single_env_cam(args):
        export_dir, i_env, i_cam, rgb, depth_normalized, i_step = args
        if rgb is not None:
            FrameImageExporter._export_frame_rgb_batch_cam(export_dir, i_env, i_cam, i_step, rgb)
        if depth_normalized is not None:
            FrameImageExporter._export_frame_depth_batch_cam(export_dir, i_env, i_cam, i_step, depth_normalized)

    @staticmethod
    def _worker_export_single_env(args):
        export_dir, i_env, rgb, depth_normalized, i_step, cam_idx = args
        if rgb is not None:
            FrameImageExporter._export_frame_rgb_single_cam(export_dir, i_env, i_step, cam_idx, rgb)
        if depth_normalized is not None:
            FrameImageExporter._export_frame_depth_single_cam(export_dir, i_env, i_step, cam_idx, depth_normalized)

    def __init__(self, export_dir, depth_clip_max=100, depth_scale='log'):
        self.export_dir = export_dir
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        self.depth_clip_max = depth_clip_max
        self.depth_scale = depth_scale

    def export_frame_batch_cam(self, i_step, rgb=None, depth=None):
        if depth is not None:
            # Batch process depth on GPU
            depth = depth.clamp(0, self.depth_clip_max)
            # Scale depth
            if self.depth_scale == 'log':
                depth = torch.log(depth + 1)
            # Calculate min/max for each image in the batch
            depth_min = depth.amin(dim=(-3, -2), keepdim=True)
            depth_max = depth.amax(dim=(-3, -2), keepdim=True)
            depth_normalized = ((depth - depth_min) / (depth_max - depth_min) * 255).to(torch.uint8)
            # Move to CPU before threading
            depth_normalized = depth_normalized
        else:
            depth_normalized = None
        
        if rgb is not None:
            # Move to CPU before threading
            rgb = rgb
        
        args_list = [(self.export_dir, i_env, i_cam, rgb, depth_normalized, i_step) 
                     for i_env in range(rgb.shape[0]) for i_cam in range(rgb.shape[1])]
        with ThreadPoolExecutor() as executor:
            executor.map(FrameImageExporter._worker_export_single_env_cam, args_list)

    def export_frame_single_cam(self, i_step, cam_idx, rgb=None, depth=None):
        if depth is not None:
            # Batch process depth on GPU
            depth = depth.clamp(0, self.depth_clip_max)
            # Scale depth
            if self.depth_scale == 'log':
                depth = torch.log(depth + 1)
            # Calculate min/max for each image in the batch
            depth_min = depth.amin(dim=(-3, -2), keepdim=True)
            depth_max = depth.amax(dim=(-3, -2), keepdim=True)
            depth_normalized = ((depth - depth_min) / (depth_max - depth_min) * 255).to(torch.uint8)
            # Move to CPU before threading
            depth_normalized = depth_normalized
        else:
            depth_normalized = None
        
        if rgb is not None:
            # Move to CPU before threading
            rgb = rgb
        
        args_list = [(self.export_dir, i_env, rgb, depth_normalized, i_step, cam_idx) 
                     for i_env in range(rgb.shape[0])]
        with ThreadPoolExecutor() as executor:
            executor.map(FrameImageExporter._worker_export_single_env, args_list) 