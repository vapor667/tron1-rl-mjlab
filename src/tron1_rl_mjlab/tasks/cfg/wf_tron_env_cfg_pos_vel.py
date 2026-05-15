import math
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.utils.noise import GaussianNoiseCfg
from mjlab.viewer import ViewerConfig

from ...assets.wf_tron.wf_tron import WF_TRON_ROBOT_CFG
from .terrain_cfg import TERRAINS_ENTITY_CFG, PLANE_ENTITY_CFG
from .. import mdp


CONTACT_FORCES_SENSOR_CFG = ContactSensorCfg(
    name="contact_forces",
    primary=ContactMatch(
        mode="body",
        pattern=("abad_.*", "hip_.*", "knee_.*", "base_Link"),
        entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=4,
)


SCENE_CFG = SceneCfg(
    num_envs=4096,
    extent=1.0,
    terrain=PLANE_ENTITY_CFG,
    sensors=(CONTACT_FORCES_SENSOR_CFG,),
    entities={"robot": WF_TRON_ROBOT_CFG},
)

VIEWER_CONFIG = ViewerConfig(
    origin_type=ViewerConfig.OriginType.ASSET_BODY,
    entity_name="robot",
    body_name="base_Link",
    distance=3.0,
    elevation=10.0,
    azimuth=90.0,
)


def make_commands() -> dict[str, CommandTermCfg]:
    """Create command configurations."""
    return {
        "base_pose": mdp.UniformWorldPoseCommandCfg(
            entity_name="robot",
            body_name="base_Link",
            resampling_time_range=(5.0, 10.0),
            resampling_time_scale=(0.5, 5.0),
            rel_standing_envs=0.02,
            rel_velocity_envs=0.25,
            debug_vis=True,
            ranges=mdp.UniformWorldPoseCommandCfg.Ranges(
                # pos lin
                pos_x=(-0.2, 0.2),  # min max [m]
                pos_y=(-0.2, 0.2),  # min max [m]
                # vel
                vel_x=(-0.7, 0.7),  # min max [m/s] in target frame
                vel_y=(-0.5, 0.5),  # min max [m/s] in target frame
                vel_yaw=(-math.pi / 2, math.pi / 2),  # min max [rad/s]
            ),
            se3_decrease_vel_range=(0.5, 1.4),
        )
    }


def make_actions() -> dict[str, ActionTermCfg]:
    """Create action configurations."""
    return {
        "joint_pos": mdp.JointPositionActionCfg(
            entity_name="robot",
            actuator_names=("abad_[RL]_Joint", "hip_[RL]_Joint", "knee_[RL]_Joint"),
            scale=0.5,
            use_default_offset=True
        ),
        "joint_vel": mdp.JointVelocityActionCfg(
            entity_name="robot",
            actuator_names=("wheel_[RL]_Joint",),
            scale=5.0,
            use_default_offset=True
        )
    }


def make_observations() -> dict[str, ObservationGroupCfg]:
    """Create observation configurations."""
    # Commands observation terms
    commands_terms = {
        "base_pose_commands": ObservationTermCfg(func=mdp.mixed_base_commands_b),
        "base_se3_decrease_rate": ObservationTermCfg(func=mdp.mixed_base_se3_decrease_rate),
        "base_commands_vel": ObservationTermCfg(func=mdp.mixed_base_commands_vel_c),
    }

    # Policy observation terms
    policy_terms = {
        # robot base measurements
        "base_ang_vel": ObservationTermCfg(
            func=mdp.base_ang_vel,
            noise=GaussianNoiseCfg(mean=0.0, std=0.05),
            scale=0.25,
        ),
        "proj_gravity": ObservationTermCfg(
            func=mdp.projected_gravity,
            noise=GaussianNoiseCfg(mean=0.0, std=0.025),
            scale=1.0,
        ),
        # robot joint measurements exclude wheel pos
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg(
                name="robot",
                joint_names=("abad_[RL]_Joint", "hip_[RL]_Joint", "knee_[RL]_Joint")
            )},
            noise=GaussianNoiseCfg(mean=0.0, std=0.01),
            scale=1.0,
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            noise=GaussianNoiseCfg(mean=0.0, std=0.01),
            scale=0.05,
        ),
        # last action
        "last_action": ObservationTermCfg(
            func=mdp.last_action,
            noise=GaussianNoiseCfg(mean=0.0, std=0.01),
            scale=1.0,
        ),
    }

    # Critic observation terms
    critic_terms = {
        "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel, scale=1.0),
        "base_ang_vel": ObservationTermCfg(func=mdp.base_ang_vel, scale=0.25),
        "proj_gravity": ObservationTermCfg(func=mdp.projected_gravity, scale=1.0),
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg(
                name="robot",
                joint_names=("abad_[RL]_Joint", "hip_[RL]_Joint", "knee_[RL]_Joint")
            )},
            scale=1.0,
        ),
        "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel, scale=0.05),
        "last_action": ObservationTermCfg(func=mdp.last_action, scale=1.0),
        "joint_torque": ObservationTermCfg(func=mdp.actuator_force, scale=0.01),
        "joint_acc": ObservationTermCfg(func=mdp.joint_acc, scale=0.1),
        "feet_lin_vel": ObservationTermCfg(
            func=mdp.body_lin_vel,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="wheel_.*")},
            scale=0.1
        ),
        "base_height_error": ObservationTermCfg(func=mdp.base_height_error, scale=3.0),
        "foot_rel_position_w": ObservationTermCfg(func=mdp.foot_rel_position_w, scale=1.5),
    }

    return {
        "actor": ObservationGroupCfg(
            terms=commands_terms | policy_terms,
            enable_corruption=True,
            concatenate_terms=True,
        ),
        "history": ObservationGroupCfg(
            terms=commands_terms | policy_terms,
            enable_corruption=True,
            concatenate_terms=True,
            history_length=20,
            flatten_history_dim=True,
        ),
        "critic": ObservationGroupCfg(
            terms=commands_terms | critic_terms,
            enable_corruption=False,
            concatenate_terms=True,
        ),
    }


def make_events() -> dict[str, EventTermCfg]:
    """Create event configurations."""
    return {
        # Startup events
        "prepare_quantities": EventTermCfg(
            func=mdp.prepare_quantities,
            mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "add_base_mass": EventTermCfg(
            func=mdp.dr.body_mass,
            mode="startup",
            params={
                "ranges": (-0.5, 2.0),
                "operation": "add",
                "distribution": "uniform",
                "asset_cfg": SceneEntityCfg("robot", body_names="base_Link"),
            },
        ),
        "add_link_mass": EventTermCfg(
            func=mdp.dr.body_mass,
            mode="startup",
            params={
                "ranges": (0.8, 1.2),
                "operation": "scale",
                "distribution": "uniform",
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_[LR]_Link"),
            },
        ),
        "robot_physics_material": EventTermCfg(
            func=mdp.dr.geom_friction,
            mode="startup",
            params={
                "ranges": {
                    0: (0.4, 1.2),  # Static friction
                    1: (0.2, 0.9),  # Dynamic friction (torsional)
                    2: (0.0, 1.0),  # Rolling friction
                },
                "operation": "abs",
                "distribution": "uniform",
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            },
        ),
        "robot_center_of_mass": EventTermCfg(
            func=mdp.dr.body_com_offset,
            mode="startup",
            params={
                "ranges": {
                    0: (-0.03, 0.03),  # X axis
                    1: (-0.03, 0.03),  # Y axis
                    2: (-0.03, 0.03),  # Z axis
                },
                "operation": "add",
                "distribution": "uniform",
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        # Reset events
        "reset_robot_base": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
                "velocity_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (-0.5, 0.5),
                    "roll": (-0.5, 0.5),
                    "pitch": (-0.5, 0.5),
                    "yaw": (-0.5, 0.5),
                },
            },
        ),
        "reset_robot_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.2, 0.2),
                "velocity_range": (-0.5, 0.5),
            },
        ),
        "randomize_joint_stiffness": EventTermCfg(
            func=mdp.dr.joint_stiffness,
            mode="startup",
            params={
                "ranges": (0.8, 1.2),
                "operation": "scale",
                "distribution": "log_uniform",
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            },
        ),
        "randomize_joint_damping": EventTermCfg(
            func=mdp.dr.joint_damping,
            mode="startup",
            params={
                "ranges": (0.8, 1.2),
                "operation": "scale",
                "distribution": "log_uniform",
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            },
        ),
        # Interval events
        "push_robot": EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(10.0, 15.0),
            params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}}
        ),
    }


def make_rewards() -> dict[str, RewardTermCfg]:
    """Create reward configurations."""
    return {
        # safety
        "safety_exp": RewardTermCfg(
            func=mdp.safety_reward_exp,
            weight=1.0,
            params={"base_height_target": 0.9, "std": math.sqrt(0.5), "command_name": "base_pose", "task_mode": "position"}
        ),
        # tasks
        "track_base_position_exp": RewardTermCfg(
            func=mdp.track_base_position_exp,
            weight=2.0,
            params={
                "command_name": "base_pose",
                "std": math.sqrt(0.5),
                "task_mode": "position",
            },
        ),
        "track_base_orientation_exp": RewardTermCfg(
            func=mdp.track_base_orientation_exp,
            weight=3.0,
            params={
                "command_name": "base_pose",
                "std": math.sqrt(0.5),
                "task_mode": "position",
            },
        ),
        "track_base_pb": RewardTermCfg(
            func=mdp.track_base_pb,
            weight=15.0,
            params={"command_name": "base_pose", "task_mode": "position"},
        ),
        "track_base_reference_exp": RewardTermCfg(
            func=mdp.track_base_reference_exp,
            weight=1.5,
            params={"command_name": "base_pose", "std": math.sqrt(0.5), "task_mode": "position"},
        ),
        # velocity tasks
        "vel_stand_still": RewardTermCfg(
            func=mdp.stand_still_velocity,
            weight=-5.0,
            params={"command_name": "base_pose", "task_mode": "velocity"},
        ),
        "keep_balance": RewardTermCfg(
            func=mdp.stay_alive_task,
            weight=1.0,
            params={"command_name": "base_pose", "task_mode": "velocity"},
        ),
        "track_lin_vel_xy": RewardTermCfg(
            func=mdp.track_lin_vel_xy_exp,
            weight=3.0,
            params={"command_name": "base_pose", "std": math.sqrt(0.2), "task_mode": "velocity"},
        ),
        "track_ang_vel_z": RewardTermCfg(
            func=mdp.track_ang_vel_z_exp,
            weight=1.0,
            params={"command_name": "base_pose", "std": math.sqrt(0.25), "task_mode": "velocity"},
        ),
        "leg_symmetry": RewardTermCfg(
            func=mdp.leg_symmetry_task,
            weight=0.5,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="wheel_.*"),
                "std": math.sqrt(0.5),
                "command_name": "base_pose",
                "task_mode": "velocity",
            },
        ),
        "same_foot_x_penalty": RewardTermCfg(
            func=mdp.same_feet_x_position_task,
            weight=-50.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="wheel_.*"),
                "command_name": "base_pose",
                "task_mode": "velocity",
            },
        ),
        "lin_vel_z_l2": RewardTermCfg(
            func=mdp.lin_vel_z_l2_task,
            weight=-0.3,
            params={"command_name": "base_pose", "task_mode": "velocity"},
        ),
        "ang_vel_xy_l2": RewardTermCfg(
            func=mdp.ang_vel_xy_l2_task,
            weight=-0.3,
            params={"command_name": "base_pose", "task_mode": "velocity"},
        ),
        "joint_torque_l2": RewardTermCfg(
            func=mdp.joint_torques_l2_task,
            weight=-1.6e-4,
            params={"command_name": "base_pose", "task_mode": "velocity"},
        ),
        "joint_acc_l2": RewardTermCfg(
            func=mdp.joint_acc_l2_task,
            weight=-1.5e-7,
            params={"command_name": "base_pose", "task_mode": "velocity"},
        ),
        "undesired_contacts": RewardTermCfg(
            func=mdp.undesired_contacts_task,
            weight=-0.25,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces"),
                "threshold": 10.0,
                "command_name": "base_pose",
                "task_mode": "velocity",
            },
        ),
        "flat_orientation_l2": RewardTermCfg(
            func=mdp.flat_orientation_l2_task,
            weight=-12.0,
            params={"command_name": "base_pose", "task_mode": "velocity"},
        ),
        "base_com_height": RewardTermCfg(
            func=mdp.base_com_height_task,
            params={"target_height": 0.80, "command_name": "base_pose", "task_mode": "velocity"},
            weight=-30.0,
        ),
        "feet_distance": RewardTermCfg(
            func=mdp.feet_distance_task,
            weight=-100.0,
            params={
                "min_feet_distance": 0.32,
                "max_feet_distance": 0.35,
                "feet_links_name": ["wheel_[RL]_Link"],
                "command_name": "base_pose",
                "task_mode": "velocity",
            },
        ),
        "joint_power_l1": RewardTermCfg(
            func=mdp.joint_powers_l1_task,
            weight=-2e-5,
            params={"command_name": "base_pose", "task_mode": "velocity"},
        ),
        "wheel_joint_vel_l2": RewardTermCfg(
            func=mdp.joint_vel_l2_task,
            weight=-5e-3,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names="wheel_.+"),
                "command_name": "base_pose",
                "task_mode": "velocity",
            },
        ),
        "non_wheel_joint_vel_l2_vel": RewardTermCfg(
            func=mdp.joint_vel_l2_task,
            weight=-0.03,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names="(?!wheel_).*"),
                "command_name": "base_pose",
                "task_mode": "velocity",
            },
        ),
        # penalties
        "dof_weighted_torques_l2": RewardTermCfg(
            func=mdp.weighted_joint_torques_l2_task,
            weight=-4.0e-5,
            params={
                "torque_weight": {
                    "abad_L_Joint": 0.2,
                    "hip_L_Joint": 0.2,
                    "knee_L_Joint": 0.2,
                    "abad_R_Joint": 0.2,
                    "hip_R_Joint": 0.2,
                    "knee_R_Joint": 0.2,
                    "wheel_L_Joint": 8.0,
                    "wheel_R_Joint": 8.0,
                },
                "command_name": "base_pose",
                "task_mode": "position",
            },
        ),
        "dof_weighted_power_l1": RewardTermCfg(
            func=mdp.weighted_joint_power_l1_task,
            weight=-2.5e-4,
            params={
                "power_weight": {
                    "abad_L_Joint": 1.0,
                    "hip_L_Joint": 1.0,
                    "knee_L_Joint": 1.0,
                    # "foot_L_Joint": 1.0,
                    "abad_R_Joint": 1.0,
                    "hip_R_Joint": 1.0,
                    "knee_R_Joint": 1.0,
                    # "foot_R_Joint": 1.0,
                    "wheel_L_Joint": 2.0,
                    "wheel_R_Joint": 2.0,
                },
                "command_name": "base_pose",
                "task_mode": "position",
            },
        ),
        "action_rate_l2": RewardTermCfg(
            func=mdp.action_rate_l2_task,
            weight=-0.002,
            params={"command_name": "base_pose", "task_mode": "position"},
        ),
        "action_smoothness": RewardTermCfg(
            func=mdp.action_smoothness_penalty_task,
            weight=-0.006,
            params={"command_name": "base_pose", "task_mode": "position"},
        ),
        "dof_vel_wheel_l2": RewardTermCfg(
            func=mdp.joint_vel_l2_task,
            weight=-0.0005,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names="wheel_.+"),
                "command_name": "base_pose",
                "task_mode": "position",
            }
        ),
        "dof_vel_non_wheel_l2": RewardTermCfg(
            func=mdp.joint_vel_l2_task,
            weight=-0.001,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names="(?!wheel_).*"),
                "command_name": "base_pose",
                "task_mode": "position",
            },
        ),
        "dof_non_wheel_pos_limits": RewardTermCfg(
            func=mdp.joint_pos_limits_task,
            weight=-5.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names="(?!wheel_).*"),
                "command_name": "base_pose",
                "task_mode": "position",
            },
        ),
    }


def make_terminations() -> dict[str, TerminationTermCfg]:
    """Create termination configurations."""
    return {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "bad_orientation": TerminationTermCfg(
            func=mdp.bad_orientation_stochastic,
            params={
                "limit_angle": math.pi * 0.4,
                "probability": 0.1,
            },  # Expect step = 1 / probability
        ),
        "bad_height": TerminationTermCfg(
            func=mdp.bad_height_stochastic,
            params={
                "limit_height": 0.5,
                "probability": 0.1,
            },  # Expect step = 1 / probability
        ),
    }


def make_curriculum() -> dict[str, CurriculumTermCfg]:
    """Create curriculum configurations."""
    return {
        "pos_commands_ranges_level": CurriculumTermCfg(
            func=mdp.pos_commands_ranges_level,
            params={
                "max_range": mdp.UniformWorldPoseCommandCfg.Ranges(
                    # pos lin
                    pos_x=(-2.0, 2.0),
                    pos_y=(-2.0, 2.0),
                    # vel
                    vel_x=(-1.0, 1.0),
                    vel_y=(-1.0, 1.0),
                    vel_yaw=(-2.0, 2.0),
                ),
                "update_interval": 80 * 24,  # 80 iterations * 24 steps per iteration
                "command_name": "base_pose",
                "update_velocity_ranges": False,
            },
        ),
        "vel_commands_ranges_level": CurriculumTermCfg(
            func=mdp.vel_commands_ranges_level,
            params={
                "max_range": {
                    "vel_x": (-1.0, 1.0),
                    "vel_y": (-0.75, 0.75),
                    "vel_yaw": (-math.pi, math.pi),
                },
                "update_interval": 80 * 24,
                "command_name": "base_pose",
            },
        )
    }


SIM_CFG = SimulationCfg(
    mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
    ),
    nconmax=256,
    njmax=512,
)


def make_wf_tron_env_cfg() -> ManagerBasedRlEnvCfg:
    """Factory function to create WF-TRON environment configuration for training."""
    return ManagerBasedRlEnvCfg(
        scene=SCENE_CFG,
        observations=make_observations(),
        actions=make_actions(),
        commands=make_commands(),
        rewards=make_rewards(),
        events=make_events(),
        terminations=make_terminations(),
        curriculum=make_curriculum(),
        sim=SIM_CFG,
        viewer=VIEWER_CONFIG,
        decimation=4,
        episode_length_s=20.0,
        seed=0,
    )


def make_wf_tron_play_env_cfg() -> ManagerBasedRlEnvCfg:
    """Factory function to create WF-TRON environment configuration for play."""
    env_cfg = deepcopy(make_wf_tron_env_cfg())
    env_cfg.scene.num_envs = 4
    env_cfg.commands["base_pose"].ranges = deepcopy(
        env_cfg.curriculum["pos_commands_ranges_level"].params["max_range"]
    )
    return env_cfg
