from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.assets import Articulation, RigidObject


def base_external_force(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """observe external force applied on the base"""
    asset: Articulation = env.scene[asset_cfg.name]
    # shape: (num_envs, 3)
    return asset._external_force_b[:, asset_cfg.body_ids, :].squeeze(1).clone()