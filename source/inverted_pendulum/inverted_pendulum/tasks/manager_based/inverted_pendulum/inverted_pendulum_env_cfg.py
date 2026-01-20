# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.sensors import ImuCfg
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from . import mdp

##
# Pre-defined configs
##

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG  # isort:skip


##
# Scene definition
##


@configclass
class InvertedPendulumSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground", # type: ignore
        spawn=sim_utils.GroundPlaneCfg(
            size=(100.0, 100.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.6,
                dynamic_friction=0.6,
                restitution=0.0,
            ),
        ), 
    )

    # robot
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            # 确保使用绝对路径或正确的相对路径
            usd_path="USD/COD-2026RoboMaster-Balance.usd", 
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0), # 初始高度 -0.3
            # 将所有关节初始化为0
            joint_pos={".*": 0.0}, 
        ),
        actuators={
            # 定义关节驱动器
            "joints": ImplicitActuatorCfg(
                # 匹配所有前、后关节和轮子
                joint_names_expr=[".*_front_joint", ".*_rear_joint", ".*_Wheel_joint"], 
                effort_limit=400.0,
                velocity_limit=100.0,
                stiffness=0.0,
                damping=2.0,
            ),
            # 如果腿部关节(Closure/Front/Rear)是被动的或弹簧阻尼的，
            # 不需要在这里添加 implicit actuator，或者添加一个只有刚度阻尼的被动 actuator
        },
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight", # type: ignore
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0), # type: ignore
    )

    # IMU sensor
    imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot", 
        # 匹配四个髋关节
        joint_names=[".*_front_joint", ".*_rear_joint"], 
        scale=50.0
    )

    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot", 
        # 匹配两个轮关节
        joint_names=[".*_Wheel_joint"], 
        scale=20.0
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        imu_quat = ObsTerm(
            func=mdp.imu_orientation,
            params={"sensor_cfg": SceneEntityCfg("imu")},
            noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.05),
        )
        wheel_odom = ObsTerm(
            func=mdp.wheel_odometry,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Wheel_joint"]),
                "wheel_radius": 0.09,
            },
            noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.01), # 假设有少量里程计噪声
        )
        # 仅保留前后髋关节的位置观测
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_front_joint", ".*_rear_joint"])},
        )
        # 保留前后髋关节和轮子的速度观测
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=[".*_front_joint", ".*_rear_joint", ".*_Wheel_joint"]
                )
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]), # 匹配所有关节
            "position_range": (0.0, 0.0), # 固定为0
            "velocity_range": (0.0, 0.0), # 固定为0
        },
    )

    reset_root = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "z": (0.3, 0.3), "yaw": (-0.0, 0.0), "pitch": (-0.0, 0.0), "roll": (-0.0, 0.0)},
            "velocity_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "z": (-0.0, 0.0), "roll": (-0.0, 0.0), "pitch": (-0.0, 0.0), "yaw": (-0.0, 0.0)},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0) # type: ignore
    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0) # type: ignore

    # (3) Base XY Position Penalty (Distance from origin)
    base_xy_pos = RewTerm(
        func=mdp.base_xy_dist_penalty,
        weight=-0.1, # 负权重表示惩罚 (距离越大，奖励越低)
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_pos": (0.0, 0.0),
        },
    )

    # (3.5) Base Height Penalty (Target 0.3m)
    base_height = RewTerm(
        func=mdp.base_height_penalty,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_height": 0.3,
        },
    )

    # (4) Base Orientation Penalties (Roll, Pitch, Yaw separate)
    base_orientation_roll = RewTerm(
        func=mdp.base_orientation_roll_penalty,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    base_orientation_pitch = RewTerm(
        func=mdp.base_orientation_pitch_penalty,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    base_orientation_yaw = RewTerm(
        func=mdp.base_orientation_yaw_penalty,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # (5) Joint position target penalty (for front and rear joints)
    joint_pos_penalty = RewTerm(
        func=mdp.joint_pos_target_l2,
        weight=-0.1,
        params={
            "target": 0.0,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_front_joint", ".*_rear_joint"] # 只针对前后关节
            ),
        },
    )

    # (6) Joint velocity penalty
    joint_vel_penalty = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])}, # 匹配所有关节
    )

    # (7) Wheels Off Ground Penalty
    wheels_off_ground = RewTerm(
        func=mdp.wheels_off_ground_penalty,
        weight=-1.0, # 惩罚轮子抬起
        params={
            # 使用 body_names 匹配 USD 中的 Rigid Body 名称
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_Wheel_link"]), 
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True) # type: ignore
    # (2) Cart out of bounds
    root_out_of_bounds = DoneTerm(
        func=mdp.root_pos_out_of_bounds,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "bounds": 1.5,
        },
    )

    # (3) Pitch out of bounds (9 degrees = ~0.157 rad)
    pitch_limit = DoneTerm(
        func=mdp.base_pitch_out_of_bounds,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "max_pitch": 0.157,
        },
    )

    # (4) Height too low (Fall down)
    low_height = DoneTerm(
        func=mdp.root_height_below_threshold,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "min_height": 0.25,
        },
    )


##
# Environment configuration
##


@configclass
class InvertedPendulumEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: InvertedPendulumSceneCfg = InvertedPendulumSceneCfg(
        num_envs=4096, env_spacing=4.0 # type: ignore
    )
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation