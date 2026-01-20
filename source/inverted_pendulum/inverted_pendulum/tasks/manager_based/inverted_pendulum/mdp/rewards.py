# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)


def joint_pos_target_l2_linear(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize linear joint position deviation from a target value (no wrapping)."""
    # extract the used quantities
    asset: Articulation = env.scene[asset_cfg.name]
    # get joint positions directly (no wrap to pi)
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    # compute the reward (L2 error)
    return torch.sum(torch.square(joint_pos - target), dim=1)


def base_xy_dist_penalty(env: ManagerBasedRLEnv, target_pos: tuple[float, float], asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize the L2 distance of the base from a target XY position."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    
    # get root position (N, 3)
    # 减去环境原点，转换为相对于环境中心的坐标
    root_pos = asset.data.root_pos_w - env.scene.env_origins
    
    # 输出z坐标，控制频率
    # if env.common_step_counter % 30 == 0:  # 每秒打印一次 (假设 30Hz)
    #     print(f"Root Z Position: {root_pos[0, 2].item():.4f}")
    
    # calculate xy distance to target (assumes target is in world frame)
    # usually target is (0, 0) for origin
    target_tensor = torch.tensor(target_pos, device=asset.device).repeat(root_pos.shape[0], 1)
    
    # compute squared distance in XY plane
    dist_sq = torch.sum(torch.square(root_pos[:, :2] - target_tensor), dim=1)
    
    # return as penalty (positive distance -> larger value, will be weighted negatively)
    return dist_sq


def base_orientation_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize deviation of base orientation from flat (roll=0, pitch=0) and forward (yaw=0?).
    
    Actually user asked for penalty on Roll, Pitch AND Yaw.
    So we penalize the magnitude of all three Euler angles relative to identity quaternion (0, 0, 0).
    """
    from isaaclab.utils.math import euler_xyz_from_quat

    # extract the used quantities
    asset: Articulation = env.scene[asset_cfg.name]
    
    # get root orientation quaternion (N, 4) -> (w, x, y, z)
    root_quat = asset.data.root_quat_w
    
    # convert to euler angles (roll, pitch, yaw)
    # Note: quat_to_euler_angles returns (roll, pitch, yaw)
    r, p, y = euler_xyz_from_quat(root_quat)
    
    # Sum of squares of all three angles (L2 norm of euler vector)
    # This penalizes any deviation from flat and zero-yaw.
    penalty = torch.square(r) + torch.square(p) + torch.square(y)
    
    return penalty


def base_orientation_roll_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize deviation of base roll from 0."""
    from isaaclab.utils.math import euler_xyz_from_quat

    asset: Articulation = env.scene[asset_cfg.name]
    r, _, _ = euler_xyz_from_quat(asset.data.root_quat_w)
    return torch.square(r)


def base_orientation_pitch_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize deviation of base pitch from 0."""
    from isaaclab.utils.math import euler_xyz_from_quat

    asset: Articulation = env.scene[asset_cfg.name]
    _, p, _ = euler_xyz_from_quat(asset.data.root_quat_w)
    return torch.square(p)


def base_orientation_yaw_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize deviation of base yaw from 0."""
    from isaaclab.utils.math import euler_xyz_from_quat

    asset: Articulation = env.scene[asset_cfg.name]
    _, _, y = euler_xyz_from_quat(asset.data.root_quat_w)
    return torch.square(y)



def base_height_penalty(env: ManagerBasedRLEnv, target_height: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize deviation of base height from target."""
    # extract the used quantities
    asset: Articulation = env.scene[asset_cfg.name]
    
    # get root position (N, 3)
    root_pos = asset.data.root_pos_w - env.scene.env_origins
    
    # Z coordinate penalty
    return torch.square(root_pos[:, 2] - target_height)


def root_pos_out_of_bounds(env: ManagerBasedRLEnv, bounds: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate when the root position is out of bounds in XY plane."""
    # extract the used quantities
    asset: Articulation = env.scene[asset_cfg.name]
    
    # get root position (N, 3)
    # 减去环境原点，转换为相对于环境中心的坐标
    root_pos = asset.data.root_pos_w - env.scene.env_origins
    
    # check if x or y is out of bounds
    out_of_bounds = (root_pos[:, 0] * root_pos[:, 0] + root_pos[:, 1] * root_pos[:, 1]) > (bounds * bounds)
    return out_of_bounds


def wheels_off_ground_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize if wheels are lifted off the ground (Z position)."""
    # extract the used quantities
    asset: Articulation = env.scene[asset_cfg.name]
    
    # get body positions (N, num_bodies, 3)
    body_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    
    # get Z height (N, num_wheels)
    # Penalize purely based on height. You might want to subtract wheel radius if known.
    # Assuming ground is at Z=0.
    wheel_heights = body_pos[..., 2]
    
    # Sum penalty for all matching wheels (left and right)
    return torch.sum(wheel_heights, dim=1)


def base_pitch_out_of_bounds(env: ManagerBasedRLEnv, max_pitch: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate when the base pitch exceeds the limit."""
    from isaaclab.utils.math import euler_xyz_from_quat

    asset: Articulation = env.scene[asset_cfg.name]
    _, p, _ = euler_xyz_from_quat(asset.data.root_quat_w)
    
    return torch.abs(p) > max_pitch


def root_height_below_threshold(env: ManagerBasedRLEnv, min_height: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate when the root height is below a threshold."""
    # extract the used quantities
    asset: Articulation = env.scene[asset_cfg.name]
    
    # get root position (N, 3)
    root_pos = asset.data.root_pos_w - env.scene.env_origins
    
    # Check Z height
    return root_pos[:, 2] < min_height