# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from himloco_lab.rsl_rl.config import HIMOnPolicyRunnerCfg, HIMPPPOAlgorithmCfg, HIMPPOActorCriticCfg

@configclass
class PPORunnerCfg(HIMOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 50
    experiment_name = "go2_rough"
    history_length = 5
    policy = HIMPPOActorCriticCfg(
        actor_hidden_dims = [512, 256, 128],
        critic_hidden_dims = [512, 256, 128],
        init_noise_std = 1.0,
        activation = 'elu', # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
    )
    algorithm = HIMPPPOAlgorithmCfg(
        value_loss_coef = 1.0,
        use_clipped_value_loss = True,
        clip_param = 0.2,
        entropy_coef = 0.01,
        num_learning_epochs = 5,
        num_mini_batches = 4, # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3, #5.e-4
        schedule = 'adaptive', # could be adaptive, fixed
        gamma = 0.99,
        lam = 0.95,
        desired_kl = 0.01,
        max_grad_norm = 1.0,
    )
    