# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi, quat_apply_inverse

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


def root_ang_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize angular velocity Z (yaw rate) L2 norm."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_ang_vel_b[:, 2])


def base_gravity_projection_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize the magnitude of the gravity vector projected onto the base XY plane."""
    # extract the used quantities
    asset: Articulation = env.scene[asset_cfg.name]
    
    # get root orientation quaternion (N, 4) -> (w, x, y, z)
    root_quat = asset.data.root_quat_w
    
    # world gravity vector (assumed to be [0, 0, -1])
    gravity_vec_w = torch.tensor([0.0, 0.0, -1.0], device=asset.device).repeat(root_quat.shape[0], 1)
    
    # project gravity into base frame
    gravity_b = quat_apply_inverse(root_quat, gravity_vec_w)
    
    # compute squared magnitude of the projection onto the XY plane
    # Ideally, if upright, gravity_b should be [0, 0, -1], so X and Y are 0.
    penalty = torch.sum(torch.square(gravity_b[:, :2]), dim=1)
    
    return penalty


def root_ang_vel_z_out_of_bounds(env: ManagerBasedRLEnv, max_ang_vel_z: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate when the root angular velocity Z exceeds the limit."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.abs(asset.data.root_ang_vel_b[:, 2]) > max_ang_vel_z


def track_lin_vel_x_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (x-axis) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the error
    # asset.data.root_lin_vel_b implies velocity in Base Frame (Body Frame).
    # This checks the forward velocity relative to the robot itself, not world X.
    lin_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 0] - asset.data.root_lin_vel_b[:, 0])
    # 控制频率输出目标线速度和实际线速度
    if env.common_step_counter % 30 == 0:  # 每秒打印
        print(f"Target Lin Vel X: {env.command_manager.get_command(command_name)[:, 0][0].item():.4f}, Actual Lin Vel X: {asset.data.root_lin_vel_b[:, 0][0].item():.4f}")
        print(f"exp(-error/std^2): {torch.exp(-lin_vel_error / std**2)[0].item():.4f}")
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the error
    # angular velocity is in base frame
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 1] - asset.data.root_ang_vel_b[:, 2])

    # 控制频率输出目标角速度和实际角速度
    if env.common_step_counter % 30 == 0:  # 每秒打印
        print(f"Target Ang Vel Z: {env.command_manager.get_command(command_name)[:, 1][0].item():.4f}, Actual Ang Vel Z: {asset.data.root_ang_vel_b[:, 2][0].item():.4f}")
        print(f"exp(-error/std^2): {torch.exp(-ang_vel_error / std**2)[0].item():.4f}")
    return torch.exp(-ang_vel_error / std**2)


def track_leg_length_exp(
    env: ManagerBasedRLEnv, 
    std: float, 
    command_name: str, 
    asset_cfg: SceneEntityCfg,
    wheel_body_names: list[str],
    front_body_names: list[str],
) -> torch.Tensor:
    """Reward tracking of leg length commands using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 获取 body indices
    wheel_indices, _ = asset.find_bodies(wheel_body_names)
    front_indices, _ = asset.find_bodies(front_body_names)
    
    # 确保索引数量一致，通常是左右两条腿
    if len(wheel_indices) != len(front_indices):
         raise ValueError(f"Number of wheel bodies ({len(wheel_indices)}) and front bodies ({len(front_indices)}) must match.")
    
    # 获取 body 位置 (全局坐标) (num_envs, num_bodies, 3)
    wheel_pos_w = asset.data.body_pos_w[:, wheel_indices, :]
    front_pos_w = asset.data.body_pos_w[:, front_indices, :]
    
    # 获取 base 位置和姿态
    root_pos_w = asset.data.root_pos_w
    root_quat_w = asset.data.root_quat_w
    
    # 扩展 root 维度以匹配 body (num_envs, num_legs, 3/4)
    # 这里我们需要分别计算每条腿的长度
    num_legs = len(wheel_indices)
    
    # 计算相对位置
    diff_w = wheel_pos_w - front_pos_w
    
    # 转换到 base 坐标系
    # 由于 diff_w 是 (env, legs, 3), root_quat 是 (env, 4)
    # 我们先 reshape
    diff_w_flat = diff_w.view(-1, 3)
    root_quat_expanded = root_quat_w.unsqueeze(1).repeat(1, num_legs, 1).view(-1, 4)
    
    diff_b_flat = quat_apply_inverse(root_quat_expanded, diff_w_flat)
    diff_b = diff_b_flat.view(env.num_envs, num_legs, 3)
    
    # 计算 XZ 平面投影距离 (x, z) -> indices 0, 2
    # leg_length = sqrt(dx^2 + dz^2)
    current_leg_lengths = torch.norm(diff_b[:, :, [0, 2]], dim=-1) # (num_envs, num_legs)
    
    # 获取目标腿长 (num_envs,)
    target_leg_length = env.command_manager.get_command(command_name)[:, 2]
    
    # 计算误差 (平均每条腿的误差)
    # 扩展 target 以匹配 legs
    target_expanded = target_leg_length.unsqueeze(1).repeat(1, num_legs)
    
    error = torch.mean(torch.square(current_leg_lengths - target_expanded), dim=1)
    # 控制频率输出目标腿长和实际腿长
    if env.common_step_counter % 30 == 0:  # 每秒打印
        print(f"Target Leg Length: {target_leg_length[0].item():.4f}, Actual Leg Lengths: {current_leg_lengths[0].tolist()}")
        print(f"exp(-error/std^2): {torch.exp(-error / std**2)[0].item():.4f}")
    
    return torch.exp(-error / std**2)