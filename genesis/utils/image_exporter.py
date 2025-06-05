import os
import cv2
import numpy as np
from multiprocessing import Pool

class FrameImageExporter:
    def __init__(self, export_dir, depth_clip_max=100):
        self.export_dir = export_dir
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        self.depth_clip_max = depth_clip_max
    
    def _export_frame_rgb_batch_cam(self, i_env, i_cam, i_step, rgb):
        rgb = rgb[i_env, i_cam, ..., [2, 1, 0]].cpu().numpy()
        cv2.imwrite(f'{self.export_dir}/rgb_env{i_env}_cam{i_cam}_{i_step:03d}.png', rgb)

    def _export_frame_depth_batch_cam(self, i_env, i_cam, i_step, depth):
        depth = depth[i_env, i_cam].cpu().numpy()
        depth = np.clip(depth, 0, self.depth_clip_max)
        depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(f'{self.export_dir}/depth_env{i_env}_cam{i_cam}_{i_step:03d}.png', depth_normalized.astype(np.uint8))

    def export_frame_batch_cam(self, i_step, rgb=None, depth=None):
        def export_single_env_cam(args):
            i_env, i_cam, rgb, depth, i_step = args
            if rgb is not None:
                self._export_frame_rgb_batch_cam(i_env, i_cam, i_step, rgb)
            if depth is not None:
                self._export_frame_depth_batch_cam(i_env, i_cam, i_step, depth)
        
        args_list = [(i_env, i_cam, rgb, depth, i_step) for i_env in range(rgb.shape[0]) for i_cam in range(rgb.shape[1])]
        with Pool() as pool:
            pool.map(export_single_env_cam, args_list)

    def _export_frame_rgb_single_cam(self, i_env, i_step, cam_idx, rgb):
        rgb = rgb[i_env, ..., [2, 1, 0]].cpu().numpy()
        cv2.imwrite(f'{self.export_dir}/rgb_env{i_env}_cam{cam_idx}_{i_step:03d}.png', rgb)

    def _export_frame_depth_single_cam(self, i_env, i_step, cam_idx, depth):
        depth = depth[i_env].cpu().numpy()
        depth = np.clip(depth, 0, self.depth_clip_max)
        depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(f'{self.export_dir}/depth_env{i_env}_cam{cam_idx}_{i_step:03d}.png', depth_normalized.astype(np.uint8))

    def export_frame_single_cam(self, i_step, cam_idx, rgb=None, depth=None):
        def export_single_env(args):
            i_env, rgb, depth, i_step, cam_idx = args
            if rgb is not None:
                self._export_frame_rgb_single_cam(i_env, i_step, cam_idx, rgb)
            if depth is not None:
                self._export_frame_depth_single_cam(i_env, i_step, cam_idx, depth)
        
        args_list = [(i_env, rgb, depth, i_step, cam_idx) for i_env in range(rgb.shape[0])]
        with Pool() as pool:
            pool.map(export_single_env, args_list) 