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
from mjlab.sensor import ContactMatch, ContactSensorCfg, GridPatternCfg, ObjRef, RayCastSensorCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.utils.noise import GaussianNoiseCfg
from mjlab.viewer import ViewerConfig

from ...assets.wf_tron.wf_tron import WF_TRON_ROBOT_CFG
from .terrain_cfg import PLANE_ENTITY_CFG, TERRAINS_ENTITY_CFG, TERRAINS_PLAY_ENTITY_CFG
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

TERRAIN_SCAN_SENSOR_CFG = RayCastSensorCfg(
	name="terrain_scan",
	frame=ObjRef(type="body", name="base_Link", entity="robot"),
	ray_alignment="yaw",
	pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
	max_distance=5.0,
	exclude_parent_body=True,
	debug_vis=False,
)


SCENE_CFG = SceneCfg(
	num_envs=4096,
	extent=1.0,
	terrain=TERRAINS_ENTITY_CFG,
	sensors=(CONTACT_FORCES_SENSOR_CFG, TERRAIN_SCAN_SENSOR_CFG),
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
	"""Create velocity command configurations."""
	return {
		"base_velocity": mdp.UniformVelocityCommandCfg(
			entity_name="robot",
			heading_command=True,
			heading_control_stiffness=1.0,
			rel_standing_envs=0.02,
			rel_heading_envs=1.0,
			debug_vis=True,
			resampling_time_range=(3.0, 15.0),
			ranges=mdp.UniformVelocityCommandCfg.Ranges(
				vel_x=(-1.2, 1.2),
				vel_y=(-1.0, 1.0),
				vel_yaw=(-math.pi, math.pi),
				heading=(-math.pi, math.pi),
			),
		)
	}


def make_actions() -> dict[str, ActionTermCfg]:
	"""Create action configurations."""
	return {
		"joint_pos": mdp.JointPositionActionCfg(
			entity_name="robot",
			actuator_names=("abad_[RL]_Joint", "hip_[RL]_Joint", "knee_[RL]_Joint"),
			scale=0.25,
			use_default_offset=True,
		),
		"joint_vel": mdp.JointVelocityActionCfg(
			entity_name="robot",
			actuator_names=("wheel_[RL]_Joint",),
			scale=1.0,
			use_default_offset=True,
		),
	}


def make_observations() -> dict[str, ObservationGroupCfg]:
	"""Create observation configurations."""
	command_terms = {
		"velocity_commands": ObservationTermCfg(
			func=mdp.generated_commands,
			params={"command_name": "base_velocity"},
			scale=1.0,
		),
	}

	policy_terms = {
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
		"joint_pos": ObservationTermCfg(
			func=mdp.joint_pos_rel,
			params={
				"asset_cfg": SceneEntityCfg(
					name="robot",
					joint_names=("abad_[RL]_Joint", "hip_[RL]_Joint", "knee_[RL]_Joint"),
				)
			},
			noise=GaussianNoiseCfg(mean=0.0, std=0.01),
			scale=1.0,
		),
		"joint_vel": ObservationTermCfg(
			func=mdp.joint_vel_rel,
			noise=GaussianNoiseCfg(mean=0.0, std=0.01),
			scale=0.05,
		),
		"last_action": ObservationTermCfg(
			func=mdp.last_action,
			noise=GaussianNoiseCfg(mean=0.0, std=0.01),
			scale=1.0,
		),
	}

	critic_terms = {
		"base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel, scale=1.0),
		"base_ang_vel": ObservationTermCfg(func=mdp.base_ang_vel, scale=0.25),
		"proj_gravity": ObservationTermCfg(func=mdp.projected_gravity, scale=1.0),
		"joint_pos": ObservationTermCfg(
			func=mdp.joint_pos_rel,
			params={
				"asset_cfg": SceneEntityCfg(
					name="robot",
					joint_names=("abad_[RL]_Joint", "hip_[RL]_Joint", "knee_[RL]_Joint"),
				)
			},
			scale=1.0,
		),
		"joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel, scale=0.05),
		"last_action": ObservationTermCfg(func=mdp.last_action, scale=1.0),
		"velocity_commands": ObservationTermCfg(
			func=mdp.generated_commands,
			params={"command_name": "base_velocity"},
			scale=1.0,
		),
		"joint_torque": ObservationTermCfg(func=mdp.actuator_force, scale=0.01),
		"joint_acc": ObservationTermCfg(func=mdp.joint_acc, scale=0.1),
		"feet_lin_vel": ObservationTermCfg(
			func=mdp.body_lin_vel,
			params={"asset_cfg": SceneEntityCfg("robot", body_names="wheel_.*")},
			scale=0.1,
		),
		"base_height_error": ObservationTermCfg(func=mdp.base_height_error, scale=3.0),
		"foot_rel_position_w": ObservationTermCfg(func=mdp.foot_rel_position_w, scale=1.5),
		"scandots": ObservationTermCfg(
			func=mdp.height_scan,
			params={"sensor_name": "terrain_scan"},
			scale=1 / TERRAIN_SCAN_SENSOR_CFG.max_distance,
		),
	}

	return {
		"actor": ObservationGroupCfg(
			terms=command_terms | policy_terms,
			enable_corruption=True,
			concatenate_terms=True,
		),
		"history": ObservationGroupCfg(
			terms=command_terms | policy_terms,
			enable_corruption=True,
			concatenate_terms=True,
			history_length=20,
			flatten_history_dim=True,
		),
		"critic": ObservationGroupCfg(
			terms=command_terms | critic_terms,
			enable_corruption=False,
			concatenate_terms=True,
			nan_policy="warn",
		),
	}


def make_events() -> dict[str, EventTermCfg]:
	"""Create event configurations."""
	return {
		"prepare_quantities": EventTermCfg(
			func=mdp.prepare_quantities,
			mode="startup",
			params={"asset_cfg": SceneEntityCfg("robot")},
		),
		# '''
		"add_base_mass": EventTermCfg(
			func=mdp.dr.body_mass,
			mode="startup",
			params={
				"ranges": (-5.0, 5.0),
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
		"randomize_rigid_body_mass_inertia": EventTermCfg(
			func=mdp.dr.pseudo_inertia,
			mode="startup",
			params={
				"asset_cfg": SceneEntityCfg("robot"),
				"d_range": (0.8, 1.2),
				"distribution": "uniform",
			},
		),
		"robot_physics_material": EventTermCfg(
			func=mdp.dr.geom_friction,
			mode="startup",
			params={
				"ranges": {
					0: (0.4, 1.2),
					1: (0.7, 0.9),
					2: (0.0, 1.0),
				},
				"operation": "abs",
				"distribution": "uniform",
				"asset_cfg": SceneEntityCfg("robot", body_names=".*"),
			},
		),
		"robot_joint_stiffness_and_damping": EventTermCfg(
			func=mdp.dr.pd_gains,
			mode="startup",
			params={
				"asset_cfg": SceneEntityCfg(
					"robot",
					actuator_ids=[0],
				),
				"kp_range": (32.0, 48.0),
				"kd_range": (2.0, 3.0),
				"operation": "abs",
				"distribution": "uniform",
			},
		),
		"robot_center_of_mass": EventTermCfg(
			func=mdp.dr.body_com_offset,
			mode="startup",
			params={
				"ranges": {
					0: (-0.075, 0.075),
					1: (-0.075, 0.075),
					2: (-0.075, 0.075),
				},
				"operation": "add",
				"distribution": "uniform",
				"asset_cfg": SceneEntityCfg("robot"),
			},
		),
		# '''
		"reset_robot_base": EventTermCfg(
			func=mdp.reset_root_state_from_flat_patches,
			mode="reset",
			params={
				"patch_name": "spawn",
				"pose_range": {"yaw": (-3.14, 3.14)},
				"velocity_range": {
					"x": (-0.5, 0.5),
					"y": (-0.5, 0.5),
					"z": (-0.0, 0.0),
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
				"position_range": (-0.5, 0.5),
				"velocity_range": (-1.0, 1.0),
			},
		),
		# '''
		"randomize_joint_stiffness": EventTermCfg(
			func=mdp.dr.joint_stiffness,
			mode="reset",
			params={
				"ranges": (0.5, 2.0),
				"operation": "scale",
				"distribution": "log_uniform",
				"asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
			},
		),
		"randomize_joint_damping": EventTermCfg(
			func=mdp.dr.joint_damping,
			mode="reset",
			params={
				"ranges": (0.5, 2.0),
				"operation": "scale",
				"distribution": "log_uniform",
				"asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
			},
		),
		# '''
		"push_robot": EventTermCfg(
			func=mdp.push_by_setting_velocity,
			mode="interval",
			interval_range_s=(5.0, 10.0),
			params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
		),
	}


def make_rewards() -> dict[str, RewardTermCfg]:
	"""Create reward configurations."""
	return {
		"keep_balance": RewardTermCfg(func=mdp.stay_alive, weight=1.0),
		"stand_still": RewardTermCfg(
			func=mdp.stand_still_velocity,
			weight=-5.0,
			params={"command_name": "base_velocity"},
		),
		"track_lin_vel_xy": RewardTermCfg(
			func=mdp.track_lin_vel_xy_exp,
			weight=3.0,
			params={"command_name": "base_velocity", "std": math.sqrt(0.2)},
		),
		"track_ang_vel_z": RewardTermCfg(
			func=mdp.track_ang_vel_z_exp,
			weight=1.0,
			params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
		),
		"leg_symmetry": RewardTermCfg(
			func=mdp.leg_symmetry,
			weight=0.25,
			params={"asset_cfg": SceneEntityCfg("robot", body_names="wheel_.*"), "std": math.sqrt(0.5)},
		),
		"same_foot_x_penalty": RewardTermCfg(
			func=mdp.same_feet_x_position,
			weight=-10.0,
			params={"asset_cfg": SceneEntityCfg("robot", body_names="wheel_.*")},
		),
		"lin_vel_z_l2": RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-0.3),
		"ang_vel_xy_l2": RewardTermCfg(func=mdp.ang_vel_xy_l2, weight=-0.3),
		"joint_torque_l2": RewardTermCfg(func=mdp.joint_torques_l2, weight=-8.0e-5),
		"joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-1.5e-7),
		"action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.02),
		"non_wheel_pos_limits": RewardTermCfg(
			func=mdp.joint_pos_limits,
			weight=-2.0,
			params={"asset_cfg": SceneEntityCfg("robot", joint_names="(?!wheel_).*")},
		),
		"undesired_contacts": RewardTermCfg(
			func=mdp.undesired_contacts,
			weight=-0.25,
			params={
				"sensor_cfg": SceneEntityCfg("contact_forces"),
				"threshold": 10.0,
			},
		),
		"action_smoothness": RewardTermCfg(func=mdp.action_smoothness_penalty, weight=-0.01),
		"flat_orientation_l2": RewardTermCfg(func=mdp.flat_orientation_l2, weight=-12.0),
		"base_com_height": RewardTermCfg(func=mdp.base_com_height, params={"target_height": 0.80}, weight=-30.0),
		"feet_distance": RewardTermCfg(
			func=mdp.feet_distance,
			weight=-100.0,
			params={
				"min_feet_distance": 0.25,
				"max_feet_distance": 0.55,
				"feet_links_name": ["wheel_[RL]_Link"],
			},
		),
		"joint_power_l1": RewardTermCfg(func=mdp.joint_powers_l1, weight=-2e-5),
		"wheel_joint_vel_l2": RewardTermCfg(
			func=mdp.joint_vel_l2,
			weight=-5e-4,
			params={"asset_cfg": SceneEntityCfg("robot", joint_names="wheel_.+")},
		),
		"non_wheel_joint_vel_l2": RewardTermCfg(
			func=mdp.joint_vel_l2,
			weight=-0.015,
			params={"asset_cfg": SceneEntityCfg("robot", joint_names="(?!wheel_).*")},
		),
	}


def make_terminations() -> dict[str, TerminationTermCfg]:
	"""Create termination configurations."""
	return {
		"time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
		"base_contact": TerminationTermCfg(
			func=mdp.bad_height_stochastic,
			params={"limit_height": 0.5, "probability": 1.0},
		),
		"nan_detection": TerminationTermCfg(func=mdp.nan_detection),
	}


def make_curriculum() -> dict[str, CurriculumTermCfg]:
	"""Create curriculum configurations."""
	return {
		"command_ranges_level": CurriculumTermCfg(
			func=mdp.vel_commands_ranges_level,
			params={
				"max_range": {
					"vel_x": (-1.5, 1.5),
					"vel_y": (-1.2, 1.2),
					"vel_yaw": (-math.pi, math.pi),
				},
				"update_interval": 80 * 24,
				"command_name": "base_velocity",
			},
		),
		"terrain_levels": CurriculumTermCfg(
			func=mdp.terrain_levels_velocity,
			params={
				"command_name": "base_velocity",
				"progress_distance_scale": 0.75,
				"min_command_fraction": 0.50,
			},
		),
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


def make_wf_tron_velocity_env_cfg() -> ManagerBasedRlEnvCfg:
	"""Factory function to create WF-TRON velocity environment configuration for training."""
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
		seed=42,
	)


def make_wf_tron_velocity_play_env_cfg() -> ManagerBasedRlEnvCfg:
	"""Factory function to create WF-TRON velocity environment configuration for play."""
	env_cfg = deepcopy(make_wf_tron_velocity_env_cfg())
	env_cfg.scene.num_envs = 8
	env_cfg.scene.terrain = TERRAINS_PLAY_ENTITY_CFG
	return env_cfg
