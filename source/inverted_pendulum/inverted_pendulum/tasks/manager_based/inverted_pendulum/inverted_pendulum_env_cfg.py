# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.sim.simulation_cfg import SimulationCfg, PhysxCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg
from isaaclab.terrains import (
    FlatPatchSamplingCfg,
    TerrainGeneratorCfg,
    TerrainImporterCfg,
)
from isaaclab.terrains.height_field import (
    HfPyramidSlopedTerrainCfg,
    HfPyramidStairsTerrainCfg,
    HfRandomUniformTerrainCfg,
)
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg

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

    # terrain importer (provides env origins + flat spawn patches)
    terrain = TerrainImporterCfg(
        prim_path="/World/Terrain",
        terrain_type="generator",
        use_terrain_origins=True,
        max_init_terrain_level=0,
        debug_vis=False,
        terrain_generator=TerrainGeneratorCfg(
            seed=123,
            curriculum=True,
            size=(10.0, 10.0),
            num_rows=8,
            num_cols=8,
            border_width=0.25,
            horizontal_scale=0.05,
            vertical_scale=0.003,
            slope_threshold=0.75,
            difficulty_range=(0.0, 1.0),
            # 1:1:1:1 mix = flat : random rugged : slope : stairs
            sub_terrains={
                "flat": HfPyramidSlopedTerrainCfg(
                    proportion=0.25,
                    slope_range=(0.0, 0.0),
                    platform_width=1.5,
                    flat_patch_sampling={
                        "init_pos": FlatPatchSamplingCfg(
                            num_patches=16,
                            patch_radius=0.35,
                            max_height_diff=0.02,
                            x_range=(-1.4, 1.4),
                            y_range=(-1.4, 1.4),
                        )
                    },
                ),
                "rugged": HfRandomUniformTerrainCfg(
                    proportion=0.25,
                    noise_range=(-0.02, 0.02),
                    noise_step=0.02,
                    downsampled_scale=0.1,
                    flat_patch_sampling={
                        "init_pos": FlatPatchSamplingCfg(
                            num_patches=16,
                            patch_radius=0.3,
                            max_height_diff=0.06,
                            x_range=(-0.9, 0.9),
                            y_range=(-0.9, 0.9),
                        )
                    },
                ),
                "slope": HfPyramidSlopedTerrainCfg(
                    proportion=0.25,
                    slope_range=(0.02, 0.3),
                    platform_width=1.25,
                    flat_patch_sampling={
                        "init_pos": FlatPatchSamplingCfg(
                            num_patches=16,
                            patch_radius=0.3,
                            max_height_diff=0.04,
                            x_range=(-1.0, 1.0),
                            y_range=(-1.0, 1.0),
                        )
                    },
                ),
                "stairs": HfPyramidStairsTerrainCfg(
                    proportion=0.25,
                    step_height_range=(0.01, 0.1),
                    step_width=0.35,
                    platform_width=1.25,
                    flat_patch_sampling={
                        "init_pos": FlatPatchSamplingCfg(
                            num_patches=16,
                            patch_radius=0.3,
                            max_height_diff=0.03,
                            x_range=(-1.0, 1.0),
                            y_range=(-1.0, 1.0),
                        )
                    },
                ),
            },
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
            # Lift spawn so terrain bumps don't bury the base; reset helper samples flat patches and adds this
            pos=(0.0, 0.0, 0.45),
            # 将所有关节初始化为0
            joint_pos={".*": 0.0}, 
        ),
        actuators={
            # 髋关节使用PD位置控制
            "hips": ImplicitActuatorCfg(
                joint_names_expr=[".*_front_joint", ".*_rear_joint"],
                effort_limit_sim=50.0,
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

    # Infinite plane outside the generated terrain for safety catch
    ground_plane = AssetBaseCfg(
        prim_path="/World/GroundPlane", # type: ignore
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="average",
                restitution_combine_mode="min",
                dynamic_friction=1.0,
                static_friction=1.0,
                restitution=0.0,
            ),
        ),
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
        lin_vel_x_range=(-3.0, 3.0),
        ang_vel_range=(-16.0, 16.0),
        leg_length_range=(0.14, 0.24),
        debug_vis=True,
        # Curriculum settings（对齐 wheel_legged_genesis）
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
        gravity_xy = ObsTerm(
            func=mdp.base_gravity_projection,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.01),
        )
        imu_quat = ObsTerm(
            func=mdp.imu_orientation,
            params={"sensor_cfg": SceneEntityCfg("imu")},
            noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.01),
        )
        imu_ang_vel = ObsTerm(
            func=mdp.imu_angular_velocity,
            params={"sensor_cfg": SceneEntityCfg("imu")},
            noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.01),
            scale=0.25,
        )
        base_lin_vel_b = ObsTerm(
            func=mdp.base_lin_vel_body,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.01),
            scale=2.0,
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
            noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.01),
            scale=1.0,
        )
        # 保留前后髋关节和轮子的速度观测
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=[".*_front_joint", ".*_rear_joint", ".*_Wheel_joint"]
                )
            },
            noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.01),
            scale=0.05,
        )
        # 动作历史（长度9）
        action_history = ObsTerm(
            func=mdp.action_history,
            params={"history_length": 9},
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
        func=mdp.reset_root_state_from_terrain,
        mode="reset",
        params={
            "pose_range": {"roll": (-0.02, 0.02), "pitch": (-0.02, 0.02), "yaw": (-0.1, 0.1)},
            "velocity_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.1, 0.1), "roll": (-0.2, 0.2), "pitch": (-0.2, 0.2), "yaw": (-0.2, 0.2)},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    reset_action_history = EventTerm(
        func=mdp.reset_action_history,
        mode="reset",
        params={},
    )

    # bump terrain difficulty when policy is consistently succeeding
    unlock_terrain_level = EventTerm(
        func=mdp.increment_terrain_level_if_success,
        mode="reset",
        params={
            "success_threshold": 0.8,
            "increment": 1,
            "max_level": None,  # defaults to num_rows - 1
            "log_key": "terrain_level",
        },
    )

    # domain randomization (aligns with wheel_legged_genesis style)
    randomize_friction = EventTerm(
        func=mdp.randomize_if_success_rate,
        mode="reset",
        params={
            "success_threshold": 0.6,
            "fallback_survival_fraction": 0.3,
            "randomize_func": mdp.randomize_rigid_body_material,
            "randomize_kwargs": {
                "asset_cfg": SceneEntityCfg("robot"),
                "static_friction_range": (0.6, 1.2),
                "dynamic_friction_range": (0.5, 1.0),
                "restitution_range": (0.0, 0.05),
                "num_buckets": 64,
                "make_consistent": True,
            },
        },
    )

    randomize_mass = EventTerm(
        func=mdp.randomize_if_success_rate,
        mode="reset",
        params={
            "success_threshold": 0.6,
            "fallback_survival_fraction": 0.3,
            "randomize_func": mdp.randomize_rigid_body_mass,
            "randomize_kwargs": {
                "asset_cfg": SceneEntityCfg("robot"),
                "mass_distribution_params": (0.8, 1.2),
                "operation": "scale",
                "distribution": "uniform",
                "recompute_inertia": True,
                "min_mass": 0.05,
            },
        },
    )

    randomize_com = EventTerm(
        func=mdp.randomize_if_success_rate,
        mode="reset",
        params={
            "success_threshold": 0.6,
            "fallback_survival_fraction": 0.3,
            "randomize_func": mdp.randomize_rigid_body_com,
            "randomize_kwargs": {
                "asset_cfg": SceneEntityCfg("robot"),
                "com_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.02, 0.02)},
            },
        },
    )

    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_if_success_rate,
        mode="reset",
        params={
            "success_threshold": 0.7,
            "fallback_survival_fraction": 0.5,
            "randomize_func": mdp.randomize_actuator_gains,
            "randomize_kwargs": {
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[".*_front_joint", ".*_rear_joint", ".*_Wheel_joint"],
                ),
                "stiffness_distribution_params": (0.8, 1.2),
                "damping_distribution_params": (0.6, 1.1),
                "operation": "scale",
                "distribution": "uniform",
            },
        },
    )

    randomize_actuator_gains_descent = EventTerm(
        func=mdp.randomize_if_success_rate,
        mode="reset",
        params={
            "success_threshold": 0.75,
            "fallback_survival_fraction": 0.6,
            "randomize_func": mdp.randomize_actuator_gains,
            "randomize_kwargs": {
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[".*_front_joint", ".*_rear_joint", ".*_Wheel_joint"],
                ),
                "stiffness_distribution_params": (0.6, 0.9),
                "damping_distribution_params": (0.6, 0.95),
                "operation": "scale",
                "distribution": "uniform",
            },
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

    # (10) 对称性奖励（左右髋）
    similar_hip = RewTerm(
        func=mdp.similar_hip,
        weight=0.5,
        params={
            "sigma": 0.05,
            "asset_cfg": SceneEntityCfg("robot"),
            "left_joint_names": ["Left_front_joint", "Left_rear_joint"],
            "right_joint_names": ["Right_front_joint", "Right_rear_joint"],
            "include_velocity": True,
        },
    )

    # (11) 动作变化率惩罚
    action_rate_hips = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    action_rate_wheels = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # (12) 力矩惩罚
    joint_torque_penalty = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_front_joint", ".*_rear_joint", ".*_Wheel_joint"])}
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

    # (2.1) Base link too low
    base_link_too_low = DoneTerm(
        func=mdp.root_height_below_threshold,
        params={"min_height": -10.0, "asset_cfg": SceneEntityCfg("robot")},
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
    # Terrain toggles
    terrain_scale: float = 1.0
    use_stairs: bool = True
    survival_fraction_for_randomization: float = 0.3

    # Scene settings
    scene: InvertedPendulumSceneCfg = InvertedPendulumSceneCfg(
        num_envs=16384, env_spacing=4.0 # type: ignore
    )
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # PhysX GPU buffer tuning to avoid collisionStackSize overflow
    sim: SimulationCfg = SimulationCfg(
        physx=PhysxCfg(
            gpu_collision_stack_size=2**28,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
        ),
    )

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

        # apply terrain scaling and optional stairs toggle
        tg = self.scene.terrain.terrain_generator
        if tg is not None:
            scale = self.terrain_scale
            tg.size = (tg.size[0] * scale, tg.size[1] * scale)
            tg.horizontal_scale *= scale
            tg.vertical_scale *= scale
            tg.border_width *= scale
            if not self.use_stairs and "stairs" in tg.sub_terrains:
                tg.sub_terrains.pop("stairs")