# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# Standard RSL-RL PPO (for reference/testing)
gym.register(
    id="Template-Himloco-Lab-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.himloco_lab_env_cfg:HimlocoLabEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

# HimLoco RSL-RL (history-informed model)
gym.register(
    id="Template-Himloco-Lab-HIM-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.himloco_lab_env_cfg:HimlocoLabEnvCfg",
        "himloco_rsl_rl_cfg_entry_point": f"{agents.__name__}.himloco_rsl_rl_cfg:HimlocoRunnerConfig",
    },
)