# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory
from isaaclab_arena.utils.pose import Pose

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


@dataclass
class GR1PickPlaceEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the GR1T2 steering-wheel pick-and-place environment."""

    embodiment: str = "gr1_pink"
    teleop_device: str | None = None
    episode_length_s: float = 20.0


@register_environment
class GR1PickPlaceEnvironment(ArenaEnvironmentFactory[GR1PickPlaceEnvironmentCfg]):
    """GR1T2 task ported from Isaac Lab's contrib steering-wheel pick-place config."""

    name = "gr1_pick_place"
    _legacy_argparse_cfg_type = GR1PickPlaceEnvironmentCfg

    def build(self, cfg: GR1PickPlaceEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the GR1T2 steering-wheel pick-and-place task."""
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.relations.relations import AtPosition, On
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.goal_pose_task import GoalPoseTask

        assert cfg.episode_length_s > 0, "episode_length_s must be positive"
        assert cfg.embodiment == "gr1_pink", "The source task requires the GR1T2 Pink IK embodiment"

        embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(enable_cameras=cfg.enable_cameras)
        table = self.asset_registry.get_asset_by_name("packing_table")()
        steering_wheel = Object(
            name="steering_wheel",
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Mimic/pick_place_task/pick_place_assets/steering_wheel.usd",
            object_type=ObjectType.RIGID,
            scale=(0.75, 0.75, 0.75),
            initial_pose=Pose(position_xyz=(-0.45, 0.45, 0.9996)),
        )
        steering_wheel.add_relation(On(table))
        steering_wheel.add_relation(AtPosition(x=-0.45, y=0.45))

        embodiment.set_initial_pose(
            Pose(position_xyz=(0.0, 0.0, 0.93), rotation_xyzw=(0.0, 0.0, 0.7071, 0.7071))
        )
        if cfg.teleop_device is None:
            teleop_device = None
        else:
            teleop_device = self.device_registry.get_device_by_name(cfg.teleop_device)()

        scene = Scene(assets=[table, steering_wheel])
        task = GoalPoseTask(
            object=steering_wheel,
            episode_length_s=cfg.episode_length_s,
            target_x_range=(0.40, 0.85),
            target_y_range=(0.35, 0.60),
            target_z_range=(0.50, 1.10),
        )
        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=teleop_device,
        )
