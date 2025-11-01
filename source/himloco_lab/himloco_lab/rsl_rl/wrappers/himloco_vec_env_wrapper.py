# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrapper to configure Isaac Lab environment for HimLoco RSL-RL."""

from __future__ import annotations

import gymnasium as gym
import torch

from isaaclab.envs import ManagerBasedRLEnv
from ..env.vec_env import VecEnv

class HimlocoVecEnvWrapper(VecEnv):
    """Wraps around Isaac Lab environment for HimLoco RSL-RL.
    
    This wrapper adapts Isaac Lab ManagerBasedRLEnv to the interface required by 
    HimLoco's custom RSL-RL implementation.
    
    Key features:
    - Converts TensorDict observations to plain tensors
    - Maintains history buffers for observations
    - Tracks terminated environments and their privileged observations
    - Provides 7-value return format required by HimLoco
    
    .. caution::
        This class must be the last wrapper in the wrapper chain.
    
    Args:
        env: The Isaac Lab ManagerBasedRLEnv environment.
        history_length: Number of historical time steps to stack with current observation.
            0 = only current step, 1 = current + 1 past step, etc.
        privileged_history_length: Number of historical time steps to stack with current privileged observation.
            0 = only current step, 1 = current + 1 past step, etc.
    
    Raises:
        ValueError: When the environment is not an instance of ManagerBasedRLEnv.
    """

    def __init__(
        self, 
        env: ManagerBasedRLEnv,
        history_length: int = 0,
        privileged_history_length: int = 0,
    ):
        """Initialize the wrapper.
        
        Args:
            env: Isaac Lab ManagerBasedRLEnv environment.
            history_length: Number of historical time steps to stack with current observation.
                0 = only current step (num_obs = num_one_step_obs)
                1 = current + 1 past step (num_obs = num_one_step_obs * 2)
                n = current + n past steps (num_obs = num_one_step_obs * (n+1))
            privileged_history_length: Number of historical time steps to stack with current privileged observation.
                Same interpretation as history_length.
        """
        # check that input is valid
        if not isinstance(env.unwrapped, ManagerBasedRLEnv):
            raise ValueError(
                "The environment must be inherited from ManagerBasedRLEnv. "
                f"Environment type: {type(env)}"
            )
        
        # check that observation manager exists
        if not hasattr(env.unwrapped, "observation_manager"):
            raise ValueError(
                "The environment must have an observation_manager. "
                "ManagerBasedRLEnv requires observation_manager to be configured."
            )
        # check that action manager exists
        if not hasattr(env.unwrapped, "action_manager"):
            raise ValueError(
                "The environment must have an action_manager. "
                "ManagerBasedRLEnv requires action_manager to be configured."
            )
        
        # initialize the wrapper
        self.env = env
        
        # store information required by HimLoco RSL-RL
        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length
        
        # obtain action dimension from action manager
        self.num_actions = self.unwrapped.action_manager.total_action_dim
        
        # obtain single-step observation dimensions from observation manager
        # These are the dimensions of observations returned by the environment at each step
        self.num_one_step_obs = self.unwrapped.observation_manager.group_obs_dim["policy"][0]
        
        # Set history lengths
        self.history_length = history_length
        
        # Calculate total observation dimensions
        # history_length=0 means only current step (num_obs = num_one_step_obs)
        # history_length=1 means current + 1 past step (num_obs = num_one_step_obs * 2)
        self.num_obs = self.num_one_step_obs * (self.history_length + 1)
        
        # History buffer for policy observations
        self.obs_history_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device)
        
        # Buffers for terminated environments (HimLoco-specific)
        self._termination_ids = torch.tensor([], dtype=torch.long, device=self.device)
        
        # Initialize privileged observation related attributes if critic observations exist
        if "critic" in self.unwrapped.observation_manager.group_obs_dim:
            self.num_one_step_privileged_obs = self.unwrapped.observation_manager.group_obs_dim["critic"][0]
            self.privileged_history_length = privileged_history_length
            self.num_privileged_obs = self.num_one_step_privileged_obs * (self.privileged_history_length + 1)
            self.privileged_obs_history_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device)
            self._termination_privileged_obs = torch.zeros(0, self.num_privileged_obs, device=self.device)
        else:
            self.num_one_step_privileged_obs = None
            self.privileged_history_length = None
            self.num_privileged_obs = None
            self.privileged_obs_history_buf = None
            self._termination_privileged_obs = None
        
        # reset at the start since the HimLoco runner does not call reset
        self.env.reset()

    def __str__(self):
        """Returns the wrapper name and the :attr:`env` representation string."""
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)

    """
    Properties -- Gym.Wrapper
    """

    @property
    def cfg(self) -> object:
        """Returns the configuration class instance of the environment."""
        return self.env.cfg

    @property
    def render_mode(self) -> str | None:
        """Returns the :attr:`Env` :attr:`render_mode`."""
        return self.env.render_mode

    @property
    def observation_space(self):
        """Returns the :attr:`Env` :attr:`observation_space`."""
        return self.env.observation_space

    @property
    def action_space(self):
        """Returns the :attr:`Env` :attr:`action_space`."""
        return self.env.action_space

    @classmethod
    def class_name(cls) -> str:
        """Returns the class name of the wrapper."""
        return cls.__name__

    @property
    def unwrapped(self):
        """Returns the base environment of the wrapper."""
        return self.env.unwrapped

    """
    Properties
    """

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """The episode length buffer."""
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        """Set the episode length buffer.
        
        Note:
            This is needed to perform random initialization of episode lengths in HimLoco RSL-RL.
        """
        self.unwrapped.episode_length_buf = value

    """
    Operations - MDP
    """

    def seed(self, seed: int = -1) -> int:
        """Set the seed for the environment."""
        return self.env.seed(seed)

    def reset(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Reset the environment.
        
        Returns:
            obs policy
        """
        # reset the environment
        obs_dict, _ = self.env.reset()
        return obs_dict["policy"], {"observations": obs_dict}

    def get_observations(self) -> torch.Tensor:
        """Get current policy observations.
        
        Returns:
            Policy observations (history-stacked) as a tensor.
        """
        return self.obs_history_buf

    def get_privileged_observations(self) -> torch.Tensor | None:
        """Get current privileged observations.
        
        Returns:
            Privileged observations (history-stacked) as a tensor, or None if not available.
        """
        return self.privileged_obs_history_buf

    def compute_termination_observations(self, env_ids: torch.Tensor, last_privileged_obs: torch.Tensor) -> torch.Tensor | None:
        """This method extracts the privileged observations for environments that just terminated.
        
        Args:
            env_ids: Indices of environments that terminated.
            last_privileged_obs: Last single-step privileged observations from environment.
        
        Returns:
            Privileged observations for terminated environments.
            Shape: (len(env_ids), num_one_step_privileged_obs).
        """
        if len(env_ids) == 0:
            return torch.zeros(0, self.num_one_step_privileged_obs, device=self.device)
        
        termination_obs = last_privileged_obs[env_ids]
        
        return termination_obs

    def step(self, actions: torch.Tensor) -> tuple[
        torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, dict,
        torch.Tensor, torch.Tensor
    ]:
        """Execute one time-step of the environment's dynamics.
        
        This extends the standard step to track terminated environments and their
        privileged observations, which is required by HimLoco for proper value
        bootstrapping at episode boundaries.
        
        Args:
            actions: Actions to apply in the environment.
        
        Returns:
            obs: Policy observations for all environments
            privileged_obs: Privileged/critic observations (None if not available)
            rewards: Rewards for all environments  
            dones: Done flags for all environments
            infos: Additional information dictionary
            termination_ids: Indices of environments that terminated this step
            termination_privileged_obs: Privileged obs for terminated environments
        """
        # get the observations from the observation manager before resetting
        last_obs_dict = self.env.unwrapped.observation_manager.compute(update_history=True)
        # execute step in Isaac Lab environment (managed by environment)
        obs_dict, rewards, terminated, truncated, infos = self.env.step(actions)
        
        # compute dones for compatibility with HimLoco RSL-RL
        dones = (terminated | truncated).to(dtype=torch.long)
        
        # move time out information to the extras dict (for infinite horizon tasks)
        if not self.unwrapped.cfg.is_finite_horizon:
            infos["time_outs"] = truncated
        
        # Extract policy observations (single-step from environment)
        if "policy" in obs_dict:
            current_obs = obs_dict["policy"]
        else:
            first_key = next(iter(obs_dict.keys()))
            current_obs = obs_dict[first_key]
        
        # Update policy observation history buffer
        # Environment returns single-step obs, wrapper does history stacking
        # [new_single_step_obs, old_history[:-single_step]]
        if self.history_length > 0:
            self.obs_history_buf = torch.cat(
                (
                    current_obs[:, :self.num_one_step_obs],
                    self.obs_history_buf[:, :-self.num_one_step_obs]
                ),
                dim=-1
            )
        else:
            self.obs_history_buf = current_obs
        
        # Track which environments terminated this step (HimLoco-specific)
        self._termination_ids = torch.nonzero(dones, as_tuple=False).squeeze(-1)
        
        # Update privileged observation history buffer if available
        if "critic" in obs_dict:
            current_privileged_obs = obs_dict["critic"]
            # Compute termination observations before updating buffer
            self._termination_privileged_obs = self.compute_termination_observations(
                self._termination_ids, last_obs_dict["critic"]
            )
            # Update history buffer
            if self.privileged_history_length > 0:
                self.privileged_obs_history_buf = torch.cat(
                    (
                        current_privileged_obs[:, :self.num_one_step_privileged_obs],
                        self.privileged_obs_history_buf[:, :-self.num_one_step_privileged_obs]
                    ),
                    dim=-1
                )
            else:
                self.privileged_obs_history_buf = current_privileged_obs
        
        return (
            self.obs_history_buf,
            self.privileged_obs_history_buf,
            rewards,
            dones,
            infos,
            self._termination_ids,
            self._termination_privileged_obs,
        )

    def close(self):
        """Close the environment."""
        return self.env.close()

    def __getattr__(self, name):
        """Forward all other attribute access to wrapped environment."""
        return getattr(self.env, name)
