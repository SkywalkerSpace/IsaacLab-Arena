# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

if TYPE_CHECKING:
    import torch

    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg

    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import IsaacLabArenaManagerBasedRLEnvCfg


_ASSET_DIR = Path(__file__).resolve().parents[1] / "examples" / "assets" / "mjcf"
LID_POSITION = (-0.05, 0.30, 1.01)
BIN_POSITION = (0.05, 0.55, 1.07)
LID_SCALE = (0.8, 0.8, 0.8)
BIN_SCALE = (0.5, 0.5, 0.5)


@dataclass
class GR1TwoArmBoxCleanupEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the GR1T2 two-arm box-cleanup environment."""

    embodiment: str = "gr1_pink"
    teleop_device: str | None = None
    episode_length_s: float = 20.0
    lid_usd_path: str = str(_ASSET_DIR / "box_cleanup_lid.usd")
    bin_usd_path: str = str(_ASSET_DIR / "box_cleanup_bin.usd")

    def __post_init__(self) -> None:
        assert math.isfinite(self.episode_length_s) and self.episode_length_s > 0, "episode_length_s must be positive"
        assert self.embodiment == "gr1_pink", "Two-arm box cleanup requires the GR1T2 Pink IK embodiment"


def _spawn_packing_table_without_container(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Spawn the table and deactivate its default container in every cloned environment."""
    import isaaclab.sim as sim_utils
    from pxr import Usd

    table_prim = sim_utils.spawn_from_usd(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )
    stage = sim_utils.get_current_stage()
    table_paths = sim_utils.find_matching_prim_paths(prim_path)
    found_container = False
    for table_path in table_paths:
        table = stage.GetPrimAtPath(table_path)
        for child in Usd.PrimRange(table):
            if child.GetName().casefold() == "container_h20":
                child.SetActive(False)
                found_container = True
    if not found_container:
        child_paths = []
        for table_path in table_paths:
            for child in Usd.PrimRange(stage.GetPrimAtPath(table_path)):
                child_paths.append(child.GetPath().pathString)
        raise RuntimeError(f"Could not find container_h20 below packing table(s). Descendants: {child_paths}")
    return table_prim


def _object_near_target(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    task_link_name: str = "right_hand_roll_link",
    max_xy_distance: float = 0.10,
    max_height_difference: float = 0.08,
    max_height: float = 1.5,
    right_wrist_max_x: float = 0.26,
    min_vel: float = 0.20,
) -> torch.Tensor:
    """Check proximity [m], componentwise speed [m/s, rad/s], and wrist retraction [m]."""
    import torch

    obj = env.scene[object_cfg.name]
    target = env.scene[target_cfg.name]
    obj_pos = obj.data.root_pos_w.torch - env.scene.env_origins
    target_pos = target.data.root_pos_w.torch - env.scene.env_origins
    xy_distance = torch.linalg.vector_norm(obj_pos[:, :2] - target_pos[:, :2], dim=1)
    height_difference = torch.abs(obj_pos[:, 2] - target_pos[:, 2])
    velocity = torch.abs(obj.data.root_vel_w.torch)
    robot = env.scene["robot"]
    wrist_idx = robot.data.body_names.index(task_link_name)
    wrist_x = robot.data.body_pos_w.torch[:, wrist_idx, 0] - env.scene.env_origins[:, 0]
    return (
        (xy_distance < max_xy_distance)
        & (height_difference < max_height_difference)
        & (obj_pos[:, 2] < max_height)
        & (wrist_x < right_wrist_max_x)
        & torch.all(velocity < min_vel, dim=1)
    )


def _configure_two_arm_box_cleanup(
    env_cfg: IsaacLabArenaManagerBasedRLEnvCfg,
) -> IsaacLabArenaManagerBasedRLEnvCfg:
    """Apply the original GR1 timing, glove feedback, and camera recording settings."""
    from isaaclab_arena_environments.gr1_pick_place_environment import _configure_gr1_pick_place

    env_cfg = _configure_gr1_pick_place(env_cfg)
    if hasattr(env_cfg.scene, "robot_pov_cam"):
        env_cfg.image_obs_list = ["robot_pov_cam", "left_wrist_cam", "right_wrist_cam"]
    return env_cfg


@register_environment
class GR1TwoArmBoxCleanupEnvironment(ArenaEnvironmentFactory[GR1TwoArmBoxCleanupEnvironmentCfg]):
    """Manipulate the lid and bin with both hands until the lid settles near the bin."""

    name = "gr1_two_arm_box_cleanup"
    _legacy_argparse_cfg_type = GR1TwoArmBoxCleanupEnvironmentCfg

    @staticmethod
    def _add_legacy_cli_only_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--object",
            type=str,
            default=None,
            help="Accepted for CLI compatibility only; use --lid_usd_path and --bin_usd_path to select assets.",
        )

    def build(self, cfg: GR1TwoArmBoxCleanupEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the custom task from stock GR1 configs without importing the custom Lab environment."""
        import isaaclab.envs.mdp as mdp
        import isaaclab.sim as sim_utils
        from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
        from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
        from isaaclab.scene import InteractiveSceneCfg
        from isaaclab.sensors import CameraCfg
        from isaaclab.utils.configclass import configclass
        from isaaclab_tasks.contrib.pick_place.pickplace_gr1t2_env_cfg import (
            EventCfg,
            PickPlaceGR1T2SceneCfg,
            TerminationsCfg,
        )

        from isaaclab_arena.assets.object import Object
        from isaaclab_arena.assets.object_type import ObjectType
        from isaaclab_arena.embodiments.common.arm_mode import ArmMode
        from isaaclab_arena.embodiments.gr1t2.gr1t2 import GR1T2CameraCfg
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.common.mimic_default_params import MIMIC_DATAGEN_CONFIG_DEFAULTS
        from isaaclab_arena.tasks.goal_pose_task import GoalPoseTask
        from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg
        from isaaclab_arena.utils.configclass import combine_configclass_instances
        from isaaclab_arena.utils.pose import Pose

        source_scene = PickPlaceGR1T2SceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
        source_events = EventCfg()

        @configclass
        class BoxCleanupSceneCfg(InteractiveSceneCfg):
            # The custom lid has a root rigid body, unlike the nested stock steering wheel.
            left_hand_contact = source_scene.left_hand_contact.replace(filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"])
            right_hand_contact = source_scene.right_hand_contact.replace(filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"])

        @configclass
        class BoxCleanupEventsCfg(EventCfg):
            reset_object = source_events.reset_object.replace(
                params={
                    "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "yaw": (-1.57079, 1.57079)},
                    "velocity_range": {},
                    "asset_cfg": SceneEntityCfg("object"),
                }
            )
            reset_object_2 = source_events.reset_object.replace(
                params={
                    "pose_range": {"x": (-0.05, 0.05), "y": (0.0, 0.05), "yaw": (-1.57079, 1.57079)},
                    "velocity_range": {},
                    "asset_cfg": SceneEntityCfg("object_2"),
                }
            )

        @configclass
        class BoxCleanupPolicyCfg(ObservationGroupCfg):
            object_pos = ObservationTermCfg(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("object")})
            object_rot = ObservationTermCfg(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("object")})
            object_2_pos = ObservationTermCfg(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("object_2")})
            object_2_rot = ObservationTermCfg(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("object_2")})
            concatenate_terms = False
            enable_corruption = False

        @configclass
        class BoxCleanupCameraCfg(GR1T2CameraCfg):
            left_wrist_cam = CameraCfg(
                prim_path="{ENV_REGEX_NS}/Robot/left_hand_pitch_link/LeftWristCam",
                update_period=0.0,
                height=256,
                width=256,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(focal_length=18.15, clipping_range=(0.1, 2.0)),
                offset=CameraCfg.OffsetCfg(
                    pos=(0.0, 0.15, 0.0), rot=(0.92387953, 0.0, 0.0, 0.38268343), convention="ros"
                ),
            )
            right_wrist_cam = CameraCfg(
                prim_path="{ENV_REGEX_NS}/Robot/right_hand_pitch_link/RightWristCam",
                update_period=0.0,
                height=256,
                width=256,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(focal_length=18.15, clipping_range=(0.1, 2.0)),
                offset=CameraCfg.OffsetCfg(
                    pos=(0.0, -0.15, 0.0), rot=(-0.92387953, 0.0, 0.0, 0.38268343), convention="ros"
                ),
            )

        class BoxCleanupCompletionCriteria(CompletionCriteria):
            """Keep derived predicate tuples out of the serialized environment config."""

            def __post_init__(self):
                # Retain stock validation, but store only constructor fields for Hydra.
                self.logical = CompletionCriteria(**vars(self)).logical

            @property
            def canonical_predicate_sequences(self):
                return CompletionCriteria(**vars(self)).canonical_predicate_sequences

        class TwoArmBoxCleanupTask(GoalPoseTask):
            def get_termination_cfg(self) -> TaskTerminationCfg:
                source_terminations = TerminationsCfg()
                success = source_terminations.success.replace(
                    func=_object_near_target,
                    params={
                        "task_link_name": "right_hand_roll_link",
                        "object_cfg": SceneEntityCfg("object"),
                        "target_cfg": SceneEntityCfg("object_2"),
                    },
                )
                return TaskTerminationCfg(
                    timeout_s=self.episode_length_s,
                    success=[BoxCleanupCompletionCriteria(name="box_cleanup", predicate_sequence=[success])],
                    failures={"object_dropping": source_terminations.object_dropping},
                )

            def get_mimic_env_cfg(self, arm_mode: ArmMode):
                assert arm_mode == ArmMode.DUAL_ARM, "Box cleanup requires both arms in Mimic demonstrations"
                mimic_cfg = MimicEnvCfg()
                mimic_cfg.datagen_config.name = "gr1_two_arm_box_cleanup_D0"
                for key, value in MIMIC_DATAGEN_CONFIG_DEFAULTS.items():
                    setattr(mimic_cfg.datagen_config, key, value)
                for arm, pickup_name, destination_name in (("left", "object", "object_2"), ("right", "object_2", None)):
                    mimic_cfg.subtask_configs[arm] = [
                        SubTaskConfig(
                            object_ref=pickup_name,
                            subtask_term_signal=f"grasp_{arm}",
                            subtask_term_offset_range=(0, 0),
                            selection_strategy="nearest_neighbor_object",
                            selection_strategy_kwargs={"nn_k": 3},
                            action_noise=0.003,
                        ),
                        SubTaskConfig(
                            object_ref=destination_name,
                            subtask_term_signal=None,
                            subtask_term_offset_range=(0, 0),
                            selection_strategy="nearest_neighbor_object" if destination_name is not None else "random",
                            selection_strategy_kwargs={"nn_k": 3} if destination_name is not None else {},
                            action_noise=0.003,
                        ),
                    ]
                return mimic_cfg

        embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(enable_cameras=cfg.enable_cameras)
        embodiment.arm_mode = ArmMode.DUAL_ARM
        embodiment.scene_config.robot = source_scene.robot.copy()
        embodiment.scene_config.robot.init_state.joint_pos.update(
            {"right_elbow_pitch_joint": -2.260, "left_elbow_pitch_joint": -2.260}
        )
        embodiment.set_initial_pose(
            Pose(
                position_xyz=tuple(source_scene.robot.init_state.pos),
                rotation_xyzw=tuple(source_scene.robot.init_state.rot),
            )
        )
        embodiment.action_config.upper_body_ik.controller.usd_path = embodiment.scene_config.robot.spawn.usd_path
        embodiment.camera_config = BoxCleanupCameraCfg(robot_pov_cam=source_scene.robot_pov_cam.copy())
        embodiment.observation_config.policy = combine_configclass_instances(
            "GR1TwoArmBoxCleanupPolicyCfg",
            embodiment.observation_config.policy,
            BoxCleanupPolicyCfg(),
            bases=(ObservationGroupCfg,),
        )

        table = Object(
            name="packing_table",
            prim_path=source_scene.packing_table.prim_path,
            object_type=ObjectType.BASE,
            spawner_cfg=source_scene.packing_table.spawn.replace(
                func=_spawn_packing_table_without_container,
                make_uninstanceable=True,
            ),
            initial_pose=Pose(
                position_xyz=tuple(source_scene.packing_table.init_state.pos),
                rotation_xyzw=tuple(source_scene.packing_table.init_state.rot),
            ),
        )
        lid = Object(
            name="object",
            prim_path=source_scene.object.prim_path,
            object_type=ObjectType.RIGID,
            spawner_cfg=source_scene.object.spawn.replace(
                usd_path=cfg.lid_usd_path, scale=LID_SCALE, activate_contact_sensors=True
            ),
            initial_pose=Pose(position_xyz=LID_POSITION),
        )
        bin_object = Object(
            name="object_2",
            prim_path="{ENV_REGEX_NS}/Object2",
            object_type=ObjectType.RIGID,
            spawner_cfg=source_scene.object.spawn.replace(
                usd_path=cfg.bin_usd_path, scale=BIN_SCALE, activate_contact_sensors=True
            ),
            initial_pose=Pose(position_xyz=BIN_POSITION),
        )
        # Source reset events restore and randomize both default poses without competing asset resets.
        lid.disable_reset_pose()
        bin_object.disable_reset_pose()
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
        scene = Scene(assets=[table, lid, bin_object, ground, light])
        task = TwoArmBoxCleanupTask(object=lid, episode_length_s=cfg.episode_length_s)
        task.task_description = "Use both hands to bring the box lid close to the bin, then retract the right hand."
        task.scene_config = BoxCleanupSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
        task.events_cfg = BoxCleanupEventsCfg()

        teleop_device = None
        if cfg.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(cfg.teleop_device)()
        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=teleop_device,
            env_cfg_callback=_configure_two_arm_box_cleanup,
        )
