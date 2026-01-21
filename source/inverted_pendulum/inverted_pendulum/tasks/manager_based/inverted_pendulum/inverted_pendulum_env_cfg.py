# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg
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
            # 髋关节使用PD位置控制
            "hips": ImplicitActuatorCfg(
                joint_names_expr=[".*_front_joint", ".*_rear_joint"],
                effort_limit_sim=25.0,
                velocity_limit_sim=100.0,
                stiffness=30.0,
                damping=0.8,
            ),
            # 轮子使用速度控制
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=[".*_Wheel_joint"],
                effort_limit_sim=6.0,
                velocity_limit_sim=100.0,
                stiffness=0.0,
                damping=1.0,
            ),
            # 被动关节（闭链关节），不施加力矩，仅有由于摩擦产生的阻尼
            "passive": ImplicitActuatorCfg(
                joint_names_expr=[".*_child.*", ".*_joint3_joint"],
                effort_limit_sim=0.0,
                velocity_limit_sim=100.0,
                stiffness=0.0,
                damping=0.1, # Small damping for stability
            ),
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

    # Contact sensor
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        history_length=3,
        track_air_time=False,
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # 髋关节位置控制
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*_front_joint", ".*_rear_joint"],
        scale=0.5,
        use_default_offset=True,
        clip={".*": (-0.5, 0.5)},
    )

    # 轮子速度控制
    joint_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=[".*_Wheel_joint"],
        scale=10.0,
        use_default_offset=True,
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_commands = mdp.InvertedPendulumCommandCfg(
        class_type=mdp.InvertedPendulumCommand,
        resampling_time_range=(2.0, 3.0),
        lin_vel_x_range=(-2.0, 2.0),
        ang_vel_range=(-16.0, 16.0),
        leg_length_range=(0.14, 0.24),
        debug_vis=True,
        # Curriculum settings
        curriculum_lin_vel_step=0.005,
        curriculum_ang_vel_step=0.005,
        curriculum_lin_vel_min_range=0.3,
        curriculum_ang_vel_min_range=0.03,
        lin_vel_err_threshold=0.35,
        ang_vel_err_threshold=0.5,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_commands"},
        )
        imu_quat = ObsTerm(
            func=mdp.imu_orientation,
            params={"sensor_cfg": SceneEntityCfg("imu")},
            noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.05),
        )
        imu_ang_vel = ObsTerm(
            func=mdp.imu_angular_velocity,
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

    # (3) Base Orientation Penalties (Gravity Projection)
    # Penalize the projection of gravity onto the base XY plane to encourage upright posture.
    flat_orientation_l2 = RewTerm(
        func=mdp.base_gravity_projection_penalty,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # (4) Joint velocity penalty
    joint_vel_penalty = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.0005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_front_joint", ".*_rear_joint"])}, # 匹配髋关节
    )

    # (5) Wheels Off Ground Penalty
    wheels_off_ground = RewTerm(
        func=mdp.wheels_off_ground_penalty,
        weight=-1.0, # 惩罚轮子抬起
        params={
            # 使用 body_names 匹配 USD 中的 Rigid Body 名称
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_Wheel_link"]), 
        },
    )

    # (6) Linear Velocity Tracking
    track_lin_vel_x = RewTerm(
        func=mdp.track_lin_vel_x_exp,
        weight=5.0,
        params={
            "command_name": "base_commands",
            "std": 0.5,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # (7) Angular Velocity Tracking
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=5.0,
        params={
            "command_name": "base_commands",
            "std": 1.0, # Increased from 0.5 to 1.0 to widen the reward basin
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # (9) Leg Length Tracking
    track_leg_length = RewTerm(
        func=mdp.track_leg_length_exp,
        weight=1.0,
        params={
            "command_name": "base_commands",
            "std": 0.05,
            "asset_cfg": SceneEntityCfg("robot"),
            "wheel_body_names": [".*_Wheel_link"], # 匹配 Left_Wheel_link, Right_Wheel_link
            "front_body_names": [".*_front_link"], # 匹配 Left_front_link, Right_front_link
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True) # type: ignore

    # (2) Illegal Contact (Body touching ground)
    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"), "threshold": 1.0},
    )
    
    # (3) Excessive Angular Velocity (Spinning out of control)
    max_ang_vel_z = DoneTerm(
        func=mdp.root_ang_vel_z_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "max_ang_vel_z": 25.0},
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
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 10
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation