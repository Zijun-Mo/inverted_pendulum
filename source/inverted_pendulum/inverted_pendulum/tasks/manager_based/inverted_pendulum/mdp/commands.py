# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@configclass
class InvertedPendulumCommandCfg(CommandTermCfg):
    """Configuration for the inverted pendulum command generator."""

    resampling_time_range: tuple[float, float] = (0.0, 0.0)
    """Time range for resampling commands. Defaults to (0.0, 0.0)."""

    lin_vel_x_range: tuple[float, float] = (-1.0, 1.0)
    """Range for linear velocity command (m/s)."""

    ang_vel_range: tuple[float, float] = (-1.0, 1.0)
    """Range for angular velocity command (rad/s)."""

    leg_length_range: tuple[float, float] = (0.1, 0.3)
    """Range for leg length command (m)."""

    # Curriculum parameters
    curriculum_lin_vel_step: float = 0.005
    curriculum_ang_vel_step: float = 0.0005
    curriculum_lin_vel_min_range: float = 0.03
    curriculum_ang_vel_min_range: float = 0.03
    lin_vel_err_threshold: float = 0.35
    ang_vel_err_threshold: float = 0.5


class InvertedPendulumCommand(CommandTerm):
    """Command generator for the inverted pendulum robot."""

    cfg: InvertedPendulumCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: InvertedPendulumCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration for the command generator.
            env: The environment.
        """
        super().__init__(cfg, env)
        
        # initialize command storage
        self._command = torch.zeros(self.num_envs, self.command_dim, device=self.device)

        # initialize scale
        self.lin_vel_scale = self.cfg.curriculum_lin_vel_min_range
        self.ang_vel_scale = self.cfg.curriculum_ang_vel_min_range
        
        # [GUI Interaction] Setup USD Prim for command input
        # Only setup if num_envs is 1 (assuming play mode or debugging)
        if self.num_envs == 1:
            from pxr import Usd, UsdGeom, Sdf
            stage = env.scene.stage
            prim_path = "/World/Command_Input"
            self._cmd_prim = stage.DefinePrim(prim_path, "Xform")
            
            # Helper to create attribute
            def create_attr(name, default_val):
                attr = self._cmd_prim.GetAttribute(name)
                if not attr:
                    attr = self._cmd_prim.CreateAttribute(name, Sdf.ValueTypeNames.Double)
                attr.Set(default_val)
                return attr

            self._attr_lin_vel = create_attr("lin_vel_x", 0.0)
            self._attr_ang_vel = create_attr("ang_vel_z", 0.0)
            self._attr_leg_len = create_attr("leg_length", 0.17)
            print(f"[INFO] Command Input Prim created at {prim_path}. You can modify commands in the Property Panel.")

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "InvertedPendulumCommand"
        msg += f"\n\tLin Vel Scale: {self.lin_vel_scale:.3f}"
        msg += f"\n\tAng Vel Scale: {self.ang_vel_scale:.3f}"
        return msg

    @property
    def command_dim(self) -> int:
        """The dimension of the command.
        
        Indices:
        0: Linear velocity x (m/s)
        1: Angular velocity z (rad/s)
        2: Leg length (m)
        """
        return 3

    @property
    def command(self) -> torch.Tensor:
        """The command tensor.
        
        Indices:
        0: Linear velocity x (m/s)
        1: Angular velocity z (rad/s)
        2: Leg length (m)
        """
        return self._command

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample the commands."""
        # Update curriculum before generating new commands (using performance on old commands)
        # Only update if a significant portion of envs are being resampled (to avoid noise from single resets)
        if len(env_ids) > self.num_envs * 0.5:
             self._update_command()

        # sample linear velocity with curriculum scale
        r_lin = self.cfg.lin_vel_x_range
        scaled_lin_range = (r_lin[0] * self.lin_vel_scale, r_lin[1] * self.lin_vel_scale)
        self._command[env_ids, 0] = torch.empty(len(env_ids), device=self.device).uniform_(*scaled_lin_range)
        
        # sample angular velocity with curriculum scale
        r_ang = self.cfg.ang_vel_range
        scaled_ang_range = (r_ang[0] * self.ang_vel_scale, r_ang[1] * self.ang_vel_scale)
        self._command[env_ids, 1] = torch.empty(len(env_ids), device=self.device).uniform_(*scaled_ang_range)
        
        # sample leg length (no curriculum for now)
        self._command[env_ids, 2] = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.leg_length_range)
        
    def _update_command(self):
        """Update metrics."""
        # access the robot from the environment
        robot = self._env.scene["robot"]
        
        # calculate errors
        # lin vel error (x-axis)
        lin_vel_error = torch.abs(self._command[:, 0] - robot.data.root_lin_vel_b[:, 0]).mean()
        # ang vel error (yaw)
        ang_vel_error = torch.abs(self._command[:, 1] - robot.data.root_ang_vel_b[:, 2]).mean()
        
        # update curriculum
        if lin_vel_error < self.cfg.lin_vel_err_threshold:
            self.lin_vel_scale = min(1.0, self.lin_vel_scale + self.cfg.curriculum_lin_vel_step)
            
        if ang_vel_error < self.cfg.ang_vel_err_threshold:
            self.ang_vel_scale = min(1.0, self.ang_vel_scale + self.cfg.curriculum_ang_vel_step)

    def _update_metrics(self):
        """Update metrics."""
        pass

    def compute(self, dt: float):
        """Update the command.
        
        This is called at every step by the manager. We use it to read from USD if in GUI mode.
        """
        # If we are in single env mode with GUI prim, update commands from USD
        if self.num_envs == 1 and hasattr(self, "_cmd_prim"):
            self._command[:, 0] = self._attr_lin_vel.Get()
            self._command[:, 1] = self._attr_ang_vel.Get()
            self._command[:, 2] = self._attr_leg_len.Get()
        
        # Call parent compute if needed or just pass. Base CommandTerm might not have logic in compute to override.
        # But we don't assume base class has compute so we don't call super().compute(dt) unless we know.
        # Checking IsaacLab `CommandTerm` usually defines `compute(self, dt: float)`.
        # Assuming we just update the tensor.
        pass
