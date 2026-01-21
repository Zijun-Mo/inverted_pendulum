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
