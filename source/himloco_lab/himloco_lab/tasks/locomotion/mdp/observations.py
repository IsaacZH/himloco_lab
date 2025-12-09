from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.sensors import RayCaster


def base_external_force(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """observe external force applied on the base"""
    asset: Articulation = env.scene[asset_cfg.name]
    # shape: (num_envs, 3)
    return asset._external_force_b[:, asset_cfg.body_ids, :].squeeze(1).clone()


def height_scan_clip(
    env: ManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg,
    clip: tuple[float, float] = (-1.0, 1.0), 
    offset: float = 0.5) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame.

    The provided offset (Defaults to 0.5) is subtracted from the returned values.
    """
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # height scan: height = sensor_height - hit_point_z - offset
    height = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset
    return torch.clip(height, clip[0], clip[1])

def base_height_scan(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Observe base height w.r.t. the terrain height.

    For flat terrain, this is the height in the world frame.
    For rough terrain, sensor readings are used to estimate the terrain height.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        ray_hits = sensor.data.ray_hits_w[..., 2]

        # Replace invalid values (NaN, Inf, too large) with NaN for masked mean
        valid_mask = ~torch.isnan(ray_hits) & ~torch.isinf(ray_hits) & (torch.abs(ray_hits) < 1e6)
        ray_hits_masked = torch.where(valid_mask, ray_hits, torch.tensor(float('nan'), device=ray_hits.device))

        # Compute mean ignoring NaN (i.e., ignoring invalid points)
        # nanmean computes mean per environment, automatically ignoring NaN values
        terrain_height = torch.nanmean(ray_hits_masked, dim=1)

        # The observation is the difference between the base height and the estimated terrain height
        height_obs = asset.data.root_pos_w[:, 2] - terrain_height
    else:
        # Use the base height in the world frame directly for flat terrain
        height_obs = asset.data.root_pos_w[:, 2]

    # shape: (num_envs, 1)
    return height_obs.unsqueeze(-1)