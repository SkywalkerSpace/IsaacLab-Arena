# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import IsaacLabArenaManagerBasedRLEnvCfg


@dataclass
class GR1PickPlaceEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the GR1T2 steering-wheel pick-and-place environment."""

    embodiment: str = "gr1_pink"
    object: str = "steering_wheel"
    teleop_device: str | None = None
    episode_length_s: float = 20.0


def _configure_gr1_pick_place(
    env_cfg: IsaacLabArenaManagerBasedRLEnvCfg,
) -> IsaacLabArenaManagerBasedRLEnvCfg:
    """Apply 50 Hz control and preserve the source task's glove feedback."""
    from isaaclab_teleop.haptic_feedback import GloveHapticFeedbackCfg
    from isaaclab_teleop.isaac_teleop_cfg import XrCameraFeedCfg

    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import set_control_rate_50hz

    env_cfg = set_control_rate_50hz(env_cfg)
    env_cfg.sim.render_interval = 2
    env_cfg.num_rerenders_on_reset = 3
    env_cfg.haptic_feedback = GloveHapticFeedbackCfg(
        left_sensor_name="left_hand_contact",
        right_sensor_name="right_hand_contact",
    )
    if env_cfg.isaac_teleop is not None and hasattr(env_cfg.scene, "robot_pov_cam"):
        env_cfg.isaac_teleop.xr_camera_feeds = [
            XrCameraFeedCfg(
                camera_name="robot_pov_cam",
                enable_dlss_ray_reconstruction=True,
                dlss_exec_mode="quality",
                offset_m=(0.0, -0.15),
                max_update_hz=0.0,
            )
        ]
    return env_cfg


@register_environment
class GR1PickPlaceEnvironment(ArenaEnvironmentFactory[GR1PickPlaceEnvironmentCfg]):
    """Pick the steering wheel from the table's left side and place it on the right."""

    name = "gr1_pick_place"
    _legacy_argparse_cfg_type = GR1PickPlaceEnvironmentCfg

    def build(self, cfg: GR1PickPlaceEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the source scene with Arena task, teleoperation, and Mimic interfaces."""
        import isaaclab.envs.mdp as mdp
        from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
        from isaaclab.scene import InteractiveSceneCfg
        from isaaclab.utils.configclass import configclass
        from isaaclab_tasks.contrib.pick_place.pickplace_gr1t2_env_cfg import (
            EventCfg,
            PickPlaceGR1T2SceneCfg,
            TerminationsCfg,
        )

        from isaaclab_arena.assets.object import Object
        from isaaclab_arena.assets.object_type import ObjectType
        from isaaclab_arena.embodiments.common.arm_mode import ArmMode
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.goal_pose_task import GoalPoseTask
        from isaaclab_arena.tasks.pick_and_place_task import PickPlaceMimicEnvCfg
        from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg
        from isaaclab_arena.utils.configclass import combine_configclass_instances
        from isaaclab_arena.utils.pose import Pose

        assert math.isfinite(cfg.episode_length_s) and cfg.episode_length_s > 0, "episode_length_s must be positive"
        assert cfg.embodiment == "gr1_pink", "The source task requires the GR1T2 Pink IK embodiment"

        source_scene = PickPlaceGR1T2SceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)

        @configclass
        class PickPlaceTaskSceneCfg(InteractiveSceneCfg):
            # Preserve filtering against the wheel's nested rigid body, not its Xform root.
            left_hand_contact = source_scene.left_hand_contact.copy()
            right_hand_contact = source_scene.right_hand_contact.copy()

        @configclass
        class PickPlaceObjectPolicyCfg(ObservationGroupCfg):
            object_pos = ObservationTermCfg(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg(cfg.object)})
            object_rot = ObservationTermCfg(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg(cfg.object)})
            concatenate_terms = False
            enable_corruption = False

        class GR1PickPlaceTask(GoalPoseTask):
            def get_termination_cfg(self) -> TaskTerminationCfg:
                source_terminations = TerminationsCfg()
                source_terminations.success.params["object_cfg"] = SceneEntityCfg(self.object.name)
                source_terminations.object_dropping.params["asset_cfg"] = SceneEntityCfg(self.object.name)
                return TaskTerminationCfg(
                    timeout_s=self.episode_length_s,
                    success=[
                        CompletionCriteria(
                            name="pick_and_place",
                            predicate_sequence=[source_terminations.success],
                        )
                    ],
                    failures={"object_dropping": source_terminations.object_dropping},
                )

            def get_mimic_env_cfg(self, arm_mode: ArmMode):
                assert arm_mode == ArmMode.RIGHT, "GR1 pick-place demonstrations use the right arm"
                mimic_cfg = PickPlaceMimicEnvCfg(
                    arm_mode=arm_mode,
                    pick_up_object_name=self.object.name,
                )
                mimic_cfg.datagen_config.name = "gr1_pick_place_D0"
                for subtask_list in mimic_cfg.subtask_configs.values():
                    for subtask in subtask_list:
                        subtask.subtask_term_offset_range = (0, 0)
                        subtask.action_noise = 0.003
                # The table is static (AssetBase), so GR1's Mimic object poses do not contain it.
                # Place in the fixed environment frame rather than following the pickup object.
                place_subtask = mimic_cfg.subtask_configs[arm_mode.value][-1]
                place_subtask.object_ref = None
                place_subtask.selection_strategy = "random"
                place_subtask.selection_strategy_kwargs = {}
                # Keep the unused arm in its recorded pose, independent of pickup randomization.
                idle_subtasks = mimic_cfg.subtask_configs.get(arm_mode.get_other_arm().value)
                assert idle_subtasks, "GR1 pick-place requires an idle subtask for the unused arm"
                idle_subtask = idle_subtasks[0]
                idle_subtask.object_ref = None
                idle_subtask.selection_strategy = "random"
                idle_subtask.selection_strategy_kwargs = {}
                return mimic_cfg

            def get_viewer_cfg(self):
                from isaaclab.envs.common import ViewerCfg

                return ViewerCfg(eye=(7.5, 7.5, 7.5), lookat=(0.0, 0.0, 0.0))

        embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(enable_cameras=cfg.enable_cameras)
        # Use the source high-PD robot, including its initial joints, rather than Arena's default gains.
        embodiment.scene_config.robot = source_scene.robot.copy()
        embodiment.set_initial_pose(
            Pose(
                position_xyz=tuple(source_scene.robot.init_state.pos),
                rotation_xyzw=tuple(source_scene.robot.init_state.rot),
            )
        )
        embodiment.action_config.upper_body_ik.controller.usd_path = embodiment.scene_config.robot.spawn.usd_path
        if embodiment.camera_config is not None:
            embodiment.camera_config.robot_pov_cam = source_scene.robot_pov_cam.copy()
        # Merge terms within one policy group; top-level composition cannot merge different policy types.
        embodiment.observation_config.policy = combine_configclass_instances(
            "GR1PickPlacePolicyCfg",
            embodiment.observation_config.policy,
            PickPlaceObjectPolicyCfg(),
            bases=(ObservationGroupCfg,),
        )

        table = Object(
            name="packing_table",
            prim_path=source_scene.packing_table.prim_path,
            object_type=ObjectType.BASE,
            spawner_cfg=source_scene.packing_table.spawn.copy(),
            initial_pose=Pose(
                position_xyz=tuple(source_scene.packing_table.init_state.pos),
                rotation_xyzw=tuple(source_scene.packing_table.init_state.rot),
            ),
        )
        pickup_object = Object(
            name=cfg.object,
            prim_path=source_scene.object.prim_path,
            object_type=ObjectType.RIGID,
            spawner_cfg=source_scene.object.spawn.copy(),
            initial_pose=Pose(
                position_xyz=tuple(source_scene.object.init_state.pos),
                rotation_xyzw=tuple(source_scene.object.init_state.rot),
            ),
        )
        # The task's reset event below restores and randomizes the source default pose.
        pickup_object.disable_reset_pose()
        ground = Object(
            name="ground",
            prim_path=source_scene.ground.prim_path,
            object_type=ObjectType.BASE,
            spawner_cfg=source_scene.ground.spawn.copy(),
        )
        light = Object(
            name="light",
            prim_path=source_scene.light.prim_path,
            object_type=ObjectType.BASE,
            spawner_cfg=source_scene.light.spawn.copy(),
        )
        scene = Scene(assets=[table, pickup_object, ground, light])

        task = GR1PickPlaceTask(object=pickup_object, episode_length_s=cfg.episode_length_s)
        task.task_description = (
            "Pick up the steering wheel from the left side of the table and place it on the right side."
        )
        task.scene_config = PickPlaceTaskSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
        task.events_cfg = EventCfg()
        task.events_cfg.reset_object.params["asset_cfg"] = SceneEntityCfg(pickup_object.name)

        teleop_device = None
        if cfg.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(cfg.teleop_device)()

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=teleop_device,
            env_cfg_callback=_configure_gr1_pick_place,
        )
