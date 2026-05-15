"""Curriculum functions for the task."""

from __future__ import annotations

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

from .commands import UniformWorldPoseCommandCfg

from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def pos_commands_ranges_level(
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | slice,
        max_range: UniformWorldPoseCommandCfg.Ranges,
        update_interval: int = 80 * 24,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
        command_name: str = "base_pose",
        update_velocity_ranges: bool = True,
) -> torch.Tensor:
    command_cfg: UniformWorldPoseCommandCfg = env.command_manager.get_term(command_name).cfg
    x = command_cfg.ranges.pos_x[1]
    if (env.common_step_counter + 1) % update_interval == 0:
        # Update position ranges
        x = command_cfg.ranges.pos_x[1] + 0.1
        y = command_cfg.ranges.pos_y[1] + 0.1
        x = min(x, max_range.pos_x[1])
        y = min(y, max_range.pos_y[1])
        command_cfg.ranges.pos_x = (-x, x)
        command_cfg.ranges.pos_y = (-y, y)

        # Update velocity ranges if they exist
        if update_velocity_ranges and hasattr(command_cfg.ranges, 'vel_x') and hasattr(max_range, 'vel_x'):
            vel_x = command_cfg.ranges.vel_x[1] + 0.05
            vel_y = command_cfg.ranges.vel_y[1] + 0.05
            vel_yaw = command_cfg.ranges.vel_yaw[1] + 0.1
            vel_x = min(vel_x, max_range.vel_x[1])
            vel_y = min(vel_y, max_range.vel_y[1])
            vel_yaw = min(vel_yaw, max_range.vel_yaw[1])
            command_cfg.ranges.vel_x = (-vel_x, vel_x)
            command_cfg.ranges.vel_y = (-vel_y, vel_y)
            command_cfg.ranges.vel_yaw = (-vel_yaw, vel_yaw)

    # return the mean terrain level
    return torch.ones(1, dtype=torch.float) * x


def vel_commands_ranges_level(
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | slice,
        max_range: dict[str, tuple[float, float]],
        update_interval: int = 80 * 24,
        command_name: str = "base_velocity",
) -> torch.Tensor:
    command_cfg = env.command_manager.get_term(command_name).cfg
    vel_x = command_cfg.ranges.vel_x[1]

    if (env.common_step_counter + 1) % update_interval == 0:
        vel_x = min(command_cfg.ranges.vel_x[1] + 0.05, max_range["vel_x"][1])
        vel_y = min(command_cfg.ranges.vel_y[1] + 0.05, max_range["vel_y"][1])
        vel_yaw = min(command_cfg.ranges.vel_yaw[1] + 0.1, max_range["vel_yaw"][1])

        command_cfg.ranges.vel_x = (-vel_x, vel_x)
        command_cfg.ranges.vel_y = (-vel_y, vel_y)
        command_cfg.ranges.vel_yaw = (-vel_yaw, vel_yaw)

        if getattr(command_cfg.ranges, "heading", None) is not None and "heading" in max_range:
            heading = min(command_cfg.ranges.heading[1] + 0.1, max_range["heading"][1])
            command_cfg.ranges.heading = (-heading, heading)

    return torch.ones(1, dtype=torch.float, device=env.device) * vel_x


def terrain_levels_velocity(
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | slice,
        command_name: str = "base_velocity",
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
        progress_distance_scale: float = 0.5,
        min_command_fraction: float = 0.3,
) -> torch.Tensor:
    """Update terrain levels from episode walking distance.

    This mirrors mjlab's velocity terrain curriculum, with configurable progress
    and regression thresholds for this robot.
    """
    asset = env.scene[asset_cfg.name]
    terrain = env.scene.terrain
    if terrain is None or terrain.terrain_origins is None:
        return torch.zeros(1, dtype=torch.float, device=env.device)

    terrain_generator = terrain.cfg.terrain_generator
    if terrain_generator is None:
        return torch.zeros(1, dtype=torch.float, device=env.device)

    command = env.command_manager.get_command(command_name)
    if command is None:
        return torch.zeros(1, dtype=torch.float, device=env.device)

    distance = torch.norm(
        asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
    )

    move_up = distance > terrain_generator.size[0] * progress_distance_scale

    move_down = (
        distance
        < torch.norm(command[env_ids, :2], dim=1)
        * env.max_episode_length_s
        * min_command_fraction
    )
    move_down *= ~move_up

    terrain.update_env_origins(env_ids, move_up, move_down)

    return torch.mean(terrain.terrain_levels.float())
