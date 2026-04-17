from mjlab.tasks.registry import register_mjlab_task
from mjlab.rl import MjlabOnPolicyRunner

from .tasks.cfg.wf_tron_env_cfg import make_wf_tron_env_cfg, make_wf_tron_play_env_cfg
from .tasks.cfg.wf_tron_velocity_env_cfg import make_wf_tron_velocity_env_cfg, make_wf_tron_velocity_play_env_cfg
from .tasks.cfg.wf_tron_rl_cfg import make_wf_tron_rl_cfg

register_mjlab_task(
    task_id="Mjlab-WF-Tron",
    env_cfg=make_wf_tron_env_cfg(),
    play_env_cfg=make_wf_tron_play_env_cfg(),
    rl_cfg=make_wf_tron_rl_cfg(),
    runner_cls=MjlabOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-WF-Tron-Velocity",
    env_cfg=make_wf_tron_velocity_env_cfg(),
    play_env_cfg=make_wf_tron_velocity_play_env_cfg(),
    rl_cfg=make_wf_tron_rl_cfg(),
    runner_cls=MjlabOnPolicyRunner,
)
