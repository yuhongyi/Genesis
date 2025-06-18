#!/usr/bin/env python3
# ppo_pick_and_place.py
#
# End-to-end GPU PPO training for a batched pick-and-place task
#   • Environment   : GraspingEnv (Genesis-Sim style API)
#   • Policy / Value: ActorCritic
#   • Algorithm     : PPO with GAE-λ
#
# Requirements
#   pip install torch numpy
#   # plus your own Genesis-Sim fork, e.g.
#   pip install git+https://github.com/GenesisRobotics/genesis-sim.git
# ---------------------------------------------------------------------

import math, random, time
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import cv2
import os
from datetime import datetime

# ---------------------------------------------------------------------
# 1. Environment
# ---------------------------------------------------------------------

import taichi as ti
import genesis as gs
from genesis.options.renderers import BatchRenderer
from genesis.utils.geom import trans_to_T


class GraspingEnv:
    """
    A vectorised GPU pick-and-place scene with two cameras and a Franka arm.
    `num_envs` independent copies run in one Scene for fast batched stepping.
    """

    def __init__(self, num_envs: int = 32, res: tuple[int, int] = (64, 64), max_steps=128, use_rgb=True, use_depth=True):
        gs.init(gs.gpu, logging_level="WARN")
        self.num_envs = num_envs
        self._res = res
        self.max_steps = max_steps
        self.use_rgb = use_rgb
        self.use_depth = use_depth
        self.step_cnt = torch.zeros(num_envs, device="cuda", dtype=torch.int32)

        # -------- build scene --------
        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3.5, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            show_viewer=False,
            show_FPS=False,
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=False,
            ),
            renderer=gs.options.renderers.BatchRenderer(
                use_rasterizer=False,
            ),
        )

        self.plane = self.scene.add_entity(gs.morphs.Plane())
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
            visualize_contact=False,
        )
        self.cube_init_pos = (0.3, 0.3, 0.05)
        self.cube = self.scene.add_entity(
            gs.morphs.Box(size=[0.05, 0.05, 0.05], pos=self.cube_init_pos),
        )

        # gripper-mounted cam
        # hand_cam = self.scene.add_camera(
        #     pos=(1.5, 0.5, 1.5),
        #     lookat=(0.0, 0.0, 0.5),
        #     fov=45,
        #     GUI=False,
        # )
        # hand_cam.attach(self.robot.links[6], trans_to_T(np.array([0.0, 0.5, 0.0])))

        # overview cam
        self.scene.add_camera(res=res, pos=(1.5, 0.0, 1.5), lookat=(0.0, 0.0, 0.5), fov=45, GUI=False)

        self.scene.build(n_envs=num_envs)

        # -------- spaces / bookkeeping --------
        self.action_dim = 9  # 9 joints on Franka
        self.n_cams = 1
        self.obs_dim = 9 + 7
        if self.use_rgb:
            self.obs_dim += res[0] * res[1] * 3 * self.n_cams
        if self.use_depth:
            self.obs_dim += res[0] * res[1] * 1 * self.n_cams

        self.target_pos = torch.tensor([0.3, 0.0, 0.05], device="cuda")

    # -------------------------------------------------------------
    def reset(self, mask: torch.Tensor | None = None):
        """
        Reset every env whose index is in `mask` (1D bool) – if mask is None,
        reset all envs.
        """
        if mask is None:
            mask = torch.ones(self.num_envs, dtype=torch.bool, device="cuda")
        idxs = mask.nonzero(as_tuple=False).squeeze(1)

        self.step_cnt[idxs] = 0

        self.scene.reset(envs_idx=idxs)
        # random XY positions in a 40 cm square
        # rand_xy = (torch.rand(len(idxs), 2, device='cuda') - 0.5) * 0.4
        # new_pos = torch.cat([rand_xy, torch.full((len(idxs),1),0.05,device='cuda')], 1)
        # self.cube.set_pos(new_pos, env_ids=idxs)
        # self.cube.set_pos(self.cube_init_pos)
        # # reopen gripper & home arm
        # self.robot.set_qpos(torch.zeros(self.action_dim, device='cuda'), env_ids=idxs)

        return self._obs()

    # -------------------------------------------------------------
    def _obs(self):
        list_obs = []
        joint = self.robot.get_qpos()
        cube_pos = self.cube.get_pos()
        cube_quat = self.cube.get_quat()
        list_obs.append(joint)
        list_obs.append(cube_pos)
        list_obs.append(cube_quat)
        if self.use_depth or self.use_rgb:
            rgb, depth, _, _ = self.scene.render_all_cams()  # (batch, n_camera, H, W, channel)
            if self.use_rgb:
                rgb = rgb[..., :3]
                rgb = rgb.float().reshape(self.num_envs, -1) / 255.0
                list_obs.append(rgb)
            if self.use_depth:
                depth = depth.float().reshape(self.num_envs, -1)
                list_obs.append(depth)
        obs = torch.cat(list_obs, 1)
        return obs

    # -------------------------------------------------------------
    def step(self, act: torch.Tensor):
        """
        act : (N, 9) desired joint targets
        """
        self.robot.control_dofs_position(act)
        self.scene.step()

        obs = self._obs()
        cube = self.cube.get_pos()
        dist = torch.norm(cube - self.target_pos, 2, -1)  # (N,)
        ee_link = self.robot.get_link("hand")
        dist_between_cube_and_hand = torch.norm(cube - ee_link.get_pos(), 2, -1) / 3
        rew = -dist - dist_between_cube_and_hand

        self.step_cnt += 1
        success = dist < 0.05
        timeout = self.step_cnt >= self.max_steps
        nan_err = torch.isnan(dist)
        done = success | timeout | nan_err

        info = {"success": success}
        return obs, rew, done, info


# ---------------------------------------------------------------------
# 2. Network
# ---------------------------------------------------------------------


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: tuple[int, int] = (256, 256)):
        super().__init__()

        # ---------- shared trunk ----------
        layers = []
        last = obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(inplace=True)]
            last = h
        self.trunk = nn.Sequential(*layers)

        # ---------- policy ----------
        self.mu_head = nn.Linear(last, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

        # ---------- value ----------
        self.v_head = nn.Linear(last, 1)

        # ---------- init ----------
        self.apply(self._ortho_init)

    # -------------------------------------------------------------
    @staticmethod
    def _ortho_init(m: nn.Module):
        """Orthogonal init (recommended for PPO)."""
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
            nn.init.zeros_(m.bias)

    # -------------------------------------------------------------
    def dist(self, x: torch.Tensor) -> Normal:
        """Return a Normal distribution π(a|s)."""
        feat = self.trunk(x)
        mu = self.mu_head(feat)
        std = self.log_std.exp()
        return Normal(mu, std)

    # -------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """
        Forward pass used by the trainer:
          returns (policy_mean, state_value)
        """
        feat = self.trunk(x)
        mu = self.mu_head(feat)
        v = self.v_head(feat).squeeze(-1)  # (B,) not (B,1)
        return mu, v


# ---------------------------------------------------------------------
# 3. PPO Trainer
# ---------------------------------------------------------------------


@torch.no_grad()
def compute_gae(rew, val, next_val, done, gamma, lam):
    """
    Generalised Advantage Estimation (vectorised over time and envs).
    """
    T, N = rew.shape
    adv = torch.zeros_like(rew)
    gae = torch.zeros(N, device=rew.device)
    for t in reversed(range(T)):
        mask = 1.0 - done[t].float()
        delta = rew[t] + gamma * next_val * mask - val[t]
        gae = delta + gamma * lam * mask * gae
        adv[t] = gae
        next_val = val[t]
    return adv, adv + val


def save_video(frames, filename):
    """Save frames as a video file."""
    if not frames:
        return

    # Get video properties from first frame
    height, width = frames[0].shape[:2]
    fps = 30

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    # Write frames
    for frame in frames:
        out.write(frame)

    out.release()


def main():
    device = torch.device("cuda")
    NUM_ENVS = 1024
    STEPS = 128
    UPDATES = 10_000
    EPOCHS = 4
    MB = 4096
    GAMMA = 0.99
    LAMBDA = 0.95
    CLIP = 0.2
    ENT = 0.01
    VCOEF = 0.5
    LR = 3e-4
    MAX_GRAD = 0.5
    res = (64, 64)
    log_freq = 10
    use_rgb = True
    use_depth = False
    max_sim_steps = 128

    # Create output directory for videos
    video_dir = "training_videos"
    os.makedirs(video_dir, exist_ok=True)

    env = GraspingEnv(NUM_ENVS, res=res, use_rgb=use_rgb, use_depth=use_depth, max_steps=max_sim_steps)

    # n_test_steps = 100
    # total_time = 0
    # without_cache = True
    # s_time = time.time()
    # for i in range(n_test_steps):
    #     if without_cache:
    #         env.scene.step()

    #     torch.cuda.synchronize()
    #     start_time = time.time()
    #     rgb, _, _, _ = env.scene.render_all_cams()
    #     torch.cuda.synchronize()
    #     total_time += time.time() - start_time
    # fps = n_test_steps / total_time * rgb.shape[0] / 1e3
    # print(f"fps: {fps:.2f}k")


    obs = env.reset()
    net = ActorCritic(env.obs_dim, env.action_dim).to(device)
    opt = optim.Adam(net.parameters(), lr=LR, eps=1e-5)

    # rollout buffers
    obs_buf = torch.zeros(STEPS, NUM_ENVS, env.obs_dim, device=device)
    act_buf = torch.zeros(STEPS, NUM_ENVS, env.action_dim, device=device)
    logp_buf = torch.zeros(STEPS, NUM_ENVS, device=device)
    val_buf = torch.zeros(STEPS, NUM_ENVS, device=device)
    rew_buf = torch.zeros(STEPS, NUM_ENVS, device=device)
    done_buf = torch.zeros(STEPS, NUM_ENVS, dtype=torch.bool, device=device)

    global_step = 0
    for upd in range(1, UPDATES + 1):
        # ────────── collect rollout ──────────
        frames = []  # Store frames for video
        t_start_sim = time.time()
        for t in range(STEPS):          
            start_time = time.time()
            obs_buf[t] = obs
            mean, val = net(obs)
            dist = Normal(mean, net.log_std.exp())
            act = dist.sample()
            logp = dist.log_prob(act).sum(-1)

            next_obs, rew, done, info = env.step(act)
            act_buf[t] = act
            logp_buf[t] = logp.detach()
            val_buf[t] = val.detach()
            rew_buf[t] = rew
            done_buf[t] = done
            # Store frames for first 4 environments
            if upd % log_freq == 0:
                rgb, depth, _, _ = env.scene.render_all_cams()
                # Convert to numpy and get first 4 environments  
                # frame0 = rgb[0, 0].cpu().numpy() 
                # frame1 = rgb[0, 1].cpu().numpy()
                # frame = np.concatenate((frame0, frame1), axis=1)


                # depth_frame = depth[0, 0].cpu().numpy()
                # depth_frame = np.concatenate((depth_frame, depth_frame, depth_frame, np.ones_like(depth_frame)), axis=-1)
                # rgb_frame = rgb[0, 0].cpu().numpy()
                # frame = np.concatenate((rgb_frame, depth_frame), axis=1)

                frame = rgb[0, 0].cpu().numpy()
                # Convert to BGR for OpenCV
                frame = frame[..., [2, 1, 0]] 
                frames.append(frame)
            # manual reset
            if done.any():
                next_obs[done] = env.reset(done)[done]
            obs = next_obs
            global_step += NUM_ENVS
        t_end_sim = time.time()
        total_time = t_end_sim - t_start_sim
        fps = NUM_ENVS * STEPS / total_time
        print(f"========================== \nIteration {upd} Simulation time: {total_time:.2f} seconds, FPS: {fps:.2f}")
        # Save video every 10 updates
        if upd % log_freq == 0 and frames:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = os.path.join(video_dir, f"training_update_{upd}_{timestamp}.mp4")
            print(f"Saving video to {video_path}")
            save_video(frames, video_path)
            print("done")

        t_start_train = time.time()
        with torch.no_grad():
            _, next_val = net(obs)

        adv, ret = compute_gae(rew_buf, val_buf, next_val, done_buf, GAMMA, LAMBDA)

        # ────────── optimise ──────────
        B = STEPS * NUM_ENVS
        flat_obs = obs_buf.reshape(B, -1)
        flat_act = act_buf.reshape(B, -1)
        flat_logp = logp_buf.reshape(B)
        flat_adv = adv.reshape(B)
        flat_ret = ret.reshape(B)
        flat_val = val_buf.reshape(B)

        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)

        for epoch in range(EPOCHS):
            idx = torch.randperm(B, device=device)
            for s in range(0, B, MB):
                mb = idx[s : s + MB]

                d = net.dist(flat_obs[mb])
                new_logp = d.log_prob(flat_act[mb]).sum(-1)
                entropy = d.entropy().sum(-1)

                ratio = torch.exp(new_logp - flat_logp[mb])
                surr1 = ratio * flat_adv[mb]
                surr2 = torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * flat_adv[mb]
                pi_loss = -torch.min(surr1, surr2).mean()

                _, new_val = net(flat_obs[mb])
                v_clipped = flat_val[mb] + (new_val - flat_val[mb]).clamp(-CLIP, CLIP)
                v_loss = 0.5 * torch.max((new_val - flat_ret[mb]).pow(2), (v_clipped - flat_ret[mb]).pow(2)).mean()

                loss = pi_loss + VCOEF * v_loss - ENT * entropy.mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), MAX_GRAD)
                opt.step()
        t_end_train = time.time()
        print(f"Training time: {t_end_train - t_start_train:.2f} seconds")
        if upd % log_freq == 0:
            with torch.no_grad():
                # Recompute new_logp over the entire flattened rollout:
                full_dist = net.dist(flat_obs)  # dist over B states
                new_logp_full = full_dist.log_prob(flat_act).sum(-1)  # shape (B,)
                approx_kl = (flat_logp - new_logp_full).mean().item()
                avg_rew = rew_buf.mean().item()
            print(
                f"upd {upd:5d} | steps {global_step:9d} | "
                f"loss {loss.item():.3f} | KL {approx_kl:.4f} | avg_rew {avg_rew:.4f}"
            )

    print("🎉 Training finished")


# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
