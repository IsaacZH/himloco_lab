# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from himloco_lab.rsl_rl.config import HIMOnPolicyRunnerCfg


def add_himloco_rsl_rl_args(parser: argparse.ArgumentParser):
    """Add HimLoco RSL-RL arguments to the parser.

    Args:
        parser: The parser to add the arguments to.
    """
    # create a new argument group
    arg_group = parser.add_argument_group("himloco_rsl_rl", description="Arguments for HimLoco RSL-RL agent.")
    # -- experiment arguments
    arg_group.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment folder where logs will be stored."
    )
    arg_group.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")
    # -- load arguments
    arg_group.add_argument("--resume", action="store_true", default=False, help="Whether to resume from a checkpoint.")
    arg_group.add_argument("--load_run", type=str, default=None, help="Name of the run folder to resume from.")
    arg_group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from.")
    # -- logger arguments
    arg_group.add_argument(
        "--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="Logger module to use."
    )
    arg_group.add_argument(
        "--log_project_name", type=str, default=None, help="Name of the logging project when using wandb or neptune."
    )


def parse_himloco_rsl_rl_cfg(task_name: str, args_cli: argparse.Namespace):
    """Parse configuration for HimLoco RSL-RL agent based on inputs.

    Args:
        task_name: The name of the environment.
        args_cli: The command line arguments.

    Returns:
        The parsed configuration for HimLoco RSL-RL agent based on inputs.
    """
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    # load the default configuration
    agent_cfg = load_cfg_from_registry(task_name, "himloco_rsl_rl_cfg")
    agent_cfg = update_himloco_rsl_rl_cfg(agent_cfg, args_cli)
    return agent_cfg


def update_himloco_rsl_rl_cfg(agent_cfg, args_cli: argparse.Namespace):
    """Update configuration for HimLoco RSL-RL agent based on inputs.

    Args:
        agent_cfg: The configuration for HimLoco RSL-RL agent.
        args_cli: The command line arguments.

    Returns:
        The updated configuration for HimLoco RSL-RL agent based on inputs.
    """
    # override the default configuration with CLI arguments
    if hasattr(args_cli, "seed") and args_cli.seed is not None:
        # randomly sample a seed if seed = -1
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
        agent_cfg.seed = args_cli.seed
    if hasattr(args_cli, "resume") and args_cli.resume is not None:
        agent_cfg.resume = args_cli.resume
    if hasattr(args_cli, "load_run") and args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if hasattr(args_cli, "checkpoint") and args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if hasattr(args_cli, "run_name") and args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if hasattr(args_cli, "experiment_name") and args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    if hasattr(args_cli, "logger") and args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    # set the project name for wandb and neptune
    if hasattr(args_cli, "log_project_name") and args_cli.log_project_name is not None:
        if agent_cfg.logger in {"wandb", "neptune"}:
            agent_cfg.wandb_project = args_cli.log_project_name
            agent_cfg.neptune_project = args_cli.log_project_name

    return agent_cfg
