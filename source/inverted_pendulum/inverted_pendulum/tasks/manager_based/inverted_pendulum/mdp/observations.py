# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.sensors import Imu
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def imu_orientation(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Observation of the IMU orientation (quaternion)."""
    # extract the sensor
    sensor: Imu = env.scene[sensor_cfg.name]
    # return the orientation in world frame (w, x, y, z)
    return sensor.data.quat_w

def imu_angular_velocity(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Observation of the IMU angular velocity (3D vector)."""
    # extract the sensor
    sensor: Imu = env.scene[sensor_cfg.name]
    # return the angular velocity in sensor frame (x, y, z)
    return sensor.data.ang_vel_b

def wheel_odometry(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, wheel_radius: float) -> torch.Tensor:
    """Observation of wheel odometry (linear displacement) based on joint positions.
    
    Args:
        env: The environment.
        asset_cfg: The configuration for the asset (robot).
        wheel_radius: The radius of the wheels in meters.
        
    Returns:
        The linear displacement of the wheels (joint_pos * radius).
    """
    # extract the asset
    asset: Articulation = env.scene[asset_cfg.name]
    # get joint positions (radians) for the specified joints (wheels)
    wheel_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    
    # convert to linear distance: pos * radius
    return wheel_pos * wheel_radius


def base_gravity_projection(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Observation of gravity projected到机体坐标系的XY分量，用于姿态感知。"""
    asset: Articulation = env.scene[asset_cfg.name]
    root_quat = asset.data.root_quat_w
    gravity_vec_w = torch.tensor([0.0, 0.0, -1.0], device=asset.device).repeat(root_quat.shape[0], 1)
    gravity_b = quat_apply_inverse(root_quat, gravity_vec_w)
    return gravity_b[:, :2]


def base_lin_vel_body(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Base 线速度（体坐标）。"""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b


def action_history(env: ManagerBasedRLEnv, history_length: int) -> torch.Tensor:
    """动作历史堆叠（长度=history_length），使用 ActionManager 的当前/上一步动作。"""
    # 使用环境上的缓存，避免每步重新分配
    action_manager = env.action_manager
    action_dim = action_manager.total_action_dim
    device = env.device if hasattr(env, "device") else action_manager.action.device

    if not hasattr(env, "_action_history_buf") or env._action_history_buf.shape[1] != history_length:
        env._action_history_buf = torch.zeros(env.num_envs, history_length, action_dim, device=device)

    buf = env._action_history_buf
    current_action = action_manager.action

    # 滚动历史并写入最新动作
    buf = torch.roll(buf, shifts=1, dims=1)
    buf[:, 0, :] = current_action
    env._action_history_buf = buf

    return buf.reshape(env.num_envs, history_length * action_dim)


def reset_action_history(env: ManagerBasedRLEnv, env_ids) -> dict:
    """在 reset 时清零动作历史缓存。"""
    if not hasattr(env, "_action_history_buf"):
        return {}
    if env_ids is None or (isinstance(env_ids, slice) and env_ids == slice(None)):
        env._action_history_buf.zero_()
    else:
        env._action_history_buf[env_ids] = 0.0
    return {}
