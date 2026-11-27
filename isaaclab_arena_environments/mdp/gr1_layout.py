# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import secrets
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def get_layout_rng(env: ManagerBasedRLEnv, randomization_seed: int | None) -> torch.Generator:
    """Get the environment's layout RNG without reseeding it on reset."""
    import torch

    if not hasattr(env, "_gr1_layout_rng"):
        seed = secrets.randbits(63) if randomization_seed is None else randomization_seed
        env._gr1_layout_rng = torch.Generator(device=env.device).manual_seed(seed)
        print(f"[INFO] GR1 layout randomization seed: {seed}")
    return env._gr1_layout_rng


def reset_object_pose(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pose_range: dict[str, tuple[float, float]],
    randomization_seed: int | None,
) -> None:
    """Restore the default root state with optional XY [m] and yaw [rad] offsets."""
    import torch

    from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if len(env_ids) == 0:
        return

    asset = env.scene[asset_cfg.name]
    pose = asset.data.default_root_pose.torch[env_ids]
    pose[:, :3] += env.scene.env_origins[env_ids]
    if pose_range:
        rng = get_layout_rng(env, randomization_seed)
        ranges = [pose_range.get(axis, (0.0, 0.0)) for axis in ("x", "y", "yaw")]
        low, high = pose.new_tensor(ranges).unbind(dim=1)
        samples = torch.rand((len(env_ids), 3), device=env.device, generator=rng)
        offsets = low + samples * (high - low)
        pose[:, :2] += offsets[:, :2]
        if "yaw" in pose_range:
            yaw = offsets[:, 2]
            zeros = torch.zeros_like(yaw)
            pose[:, 3:7] = quat_mul(pose[:, 3:7], quat_from_euler_xyz(zeros, zeros, yaw))

    asset.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
    asset.write_root_velocity_to_sim_index(root_velocity=asset.data.default_root_vel.torch[env_ids], env_ids=env_ids)
