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


def height_scan_clip_first(
    env: ManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg, 
    offset: float = 0.5,
    clip_min: float = -1.0,
    clip_max: float = 1.0,
    scale: float = 5.0
) -> torch.Tensor:
    """Height scan with CLIP-FIRST-THEN-SCALE processing to match HIMLOCO_GO2.
    
    This function implements the same height processing order as HIMLOCO_GO2:
    1. Compute raw height: sensor_z - hit_point_z - offset
    2. Clip to [clip_min, clip_max] (default [-1, 1])
    3. Scale by scale factor (default 5.0)
    
    Final range: [clip_min * scale, clip_max * scale] = [-5, 5] by default
    
    This is CRITICAL to prevent NaN in complex terrain:
    - Without proper clipping FIRST, heights of ±3m become ±15 after scaling
    - This causes numerical instability in the estimator network
    
    Args:
        env: The environment instance
        sensor_cfg: The height scanner sensor configuration
        offset: Height offset to subtract (default 0.5, matches HIMLOCO_GO2's -0.5 offset)
        clip_min: Minimum clip value before scaling (default -1.0)
        clip_max: Maximum clip value before scaling (default 1.0)
        scale: Scale factor applied after clipping (default 5.0)
    
    Returns:
        Clipped and scaled height measurements, shape (num_envs, num_rays)
    """
    # Extract the height scanner sensor
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    
    # Compute raw height: sensor_height - hit_point_z - offset
    # This matches HIMLOCO_GO2: self.root_states[:, 2] - 0.5 - self.measured_heights
    raw_heights = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset
    
    # CLIP FIRST (to [-1, 1] by default)
    clipped_heights = torch.clip(raw_heights, clip_min, clip_max)
    
    # THEN SCALE (× 5.0 by default)
    scaled_heights = clipped_heights * scale
    
    return scaled_heights