import torch
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg


from isaaclab.envs import ManagerBasedEnv


def apply_periodic_external_force_torque(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    period_step: int,
    force_range: tuple[float, float],
    torque_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Apply periodic external forces and torques.
    
    This function applies random forces and torques sampled from given ranges, but only
    when the call count reaches a multiple of period_step. At other times, zero forces
    and torques are applied. The call count is tracked using an attribute on the environment.
    
    Args:
        env: The RL environment
        env_ids: IDs of environments to apply forces to. If None, applies to all environments.
        period_step: Period in simulation steps for applying random forces/torques
        force_range: (min, max) range for force magnitude
        torque_range: (min, max) range for torque magnitude
        asset_cfg: Configuration for the asset to apply forces to
    """
    # Initialize call count on first call
    if not hasattr(env, "external_force_call_count") or env.external_force_call_count is None:
        env.external_force_call_count = 0
    
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    
    # resolve number of bodies
    num_bodies = (
        len(asset_cfg.body_ids)
        if isinstance(asset_cfg.body_ids, list)
        else asset.num_bodies
    )
    
    # Create zero forces and torques
    size = (len(env_ids), num_bodies, 3)
    zero_forces = torch.zeros(size, device=asset.device, dtype=torch.float32)
    zero_torques = torch.zeros(size, device=asset.device, dtype=torch.float32)
    
    # Only apply random forces/torques when call count is multiple of period_step
    if env.external_force_call_count % period_step == 0:
        # sample random forces and torques
        forces = math_utils.sample_uniform(*force_range, size, asset.device)
        torques = math_utils.sample_uniform(*torque_range, size, asset.device)
    else:
        # apply zero forces and torques
        forces = zero_forces
        torques = zero_torques
    
    # set the forces and torques into the buffers
    asset.permanent_wrench_composer.set_forces_and_torques(
        forces=forces, torques=torques, env_ids=env_ids, body_ids=asset_cfg.body_ids
    )
    
    # increment call count
    env.external_force_call_count += 1