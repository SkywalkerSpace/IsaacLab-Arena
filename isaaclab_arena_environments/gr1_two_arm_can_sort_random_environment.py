# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
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


_ASSET_DIR = Path(__file__).resolve().parents[1] / "examples" / "assets" / "mjcf2usd"
HANDOFF_MAX_DISTANCE = 0.10
BOX_LEFT_POSITION = (-0.25, 0.50, 1.07)
BOX_RIGHT_POSITION = (0.25, 0.50, 1.07)
CAN_POSITION = (0.0, 0.30, 1.01)
BOX_SCALE = (0.5, 0.5, 0.5)
CAN_SCALE = (0.8, 0.8, 0.8)


@dataclass
class GR1TwoArmCanSortRandomEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the GR1T2 two-arm can-sort-random environment."""

    embodiment: str = "gr1_pink"
    teleop_device: str | None = None
    episode_length_s: float = 20.0
    can_sort_blue_box_path: str = str(_ASSET_DIR / "can_sort_blue_box.usd")
    can_sort_red_box_path: str = str(_ASSET_DIR / "can_sort_red_box.usd")
    can_sort_blue_can_path: str = str(_ASSET_DIR / "can_sort_blue_can.usd")
    can_sort_red_can_path: str = str(_ASSET_DIR / "can_sort_red_can.usd")
    success_max_xy_distance: float = 0.10
    success_max_height_difference: float = 0.08
    success_max_speed: float = 0.20
    """Componentwise can velocity limits in m/s (linear) and rad/s (angular)."""
    can_reference_offset: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """Success reference point in the can's root frame, in meters."""
    box_reference_offset: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """Success reference point in the matching box's root frame, in meters."""

    def __post_init__(self) -> None:
        assert math.isfinite(self.episode_length_s) and self.episode_length_s > 0, "episode_length_s must be positive"
        if self.embodiment not in ("gr1_pink", "gr1_joint"):
            raise ValueError(
                f"Two-arm can sort random requires a GR1T2 embodiment ('gr1_pink' or 'gr1_joint'); "
                f"got {self.embodiment!r}."
            )
        for name in ("success_max_xy_distance", "success_max_height_difference", "success_max_speed"):
            value = getattr(self, name)
            assert math.isfinite(value) and value > 0, f"{name} must be finite and positive"
        for name in ("can_reference_offset", "box_reference_offset"):
            offset = getattr(self, name)
            assert len(offset) == 3 and all(math.isfinite(value) for value in offset), (
                f"{name} must contain three finite coordinates"
            )


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


def _object_lifted(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    min_height: float,
) -> torch.Tensor:
    """Use object lift above an environment-local height [m] as a grasp proxy."""
    obj = env.scene[object_cfg.name]
    return obj.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2] > min_height


def _object_near_left_palm(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    max_distance: float = HANDOFF_MAX_DISTANCE,
) -> torch.Tensor:
    """Signal the right-to-left handoff when the object is close to the left palm [m]."""
    import torch

    obj = env.scene[object_cfg.name]
    robot = env.scene["robot"]
    palm_index = robot.data.body_names.index("left_hand_pitch_link")
    object_pos = obj.data.root_pos_w.torch
    palm_pos = robot.data.body_pos_w.torch[:, palm_index]
    return torch.linalg.vector_norm(object_pos - palm_pos, dim=1) < max_distance


def _randomize_box_sides(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None) -> None:
    """Randomly swap the blue and red boxes between the two left-side positions on reset."""
    import torch

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device).reshape(-1)
    if len(env_ids) == 0:
        return
    swap = torch.rand(len(env_ids), device=env.device) < 0.5
    blue_x = torch.where(swap, BOX_RIGHT_POSITION[0], BOX_LEFT_POSITION[0])
    red_x = torch.where(swap, BOX_LEFT_POSITION[0], BOX_RIGHT_POSITION[0])
    identity = torch.tensor((0.0, 0.0, 0.0, 1.0), device=env.device)
    for name, x in (("blue_box", blue_x), ("red_box", red_x)):
        asset = env.scene[name]
        pose = torch.zeros((len(env_ids), 7), device=env.device)
        pose[:, 0] = x
        pose[:, 1] = BOX_LEFT_POSITION[1]
        pose[:, 2] = BOX_LEFT_POSITION[2]
        pose[:, 3:] = identity
        pose[:, :3] += env.scene.env_origins[env_ids]
        asset.write_root_pose_to_sim(pose, env_ids=env_ids)
        asset.write_root_velocity_to_sim(torch.zeros((len(env_ids), 6), device=env.device), env_ids=env_ids)


def _object_near_target(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    max_xy_distance: float = 0.10,
    max_height_difference: float = 0.08,
    max_height: float = 1.5,
    max_speed: float = 0.20,
    object_reference_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    target_reference_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    required_variant_index: int | None = None,
) -> torch.Tensor:
    """Check local reference-point proximity [m] and can velocity limits [m/s, rad/s]."""
    import torch

    from isaaclab.utils.math import quat_apply

    obj = env.scene[object_cfg.name]
    target = env.scene[target_cfg.name]
    obj_pos = obj.data.root_pos_w.torch - env.scene.env_origins
    target_pos = target.data.root_pos_w.torch - env.scene.env_origins
    object_offset_O = obj_pos.new_tensor(object_reference_offset).expand_as(obj_pos)
    target_offset_T = target_pos.new_tensor(target_reference_offset).expand_as(target_pos)
    # O/T are object/target root frames; E is the translated environment frame.
    object_point_E = obj_pos + quat_apply(obj.data.root_quat_w.torch, object_offset_O)
    target_point_E = target_pos + quat_apply(target.data.root_quat_w.torch, target_offset_T)
    xy_distance = torch.linalg.vector_norm(object_point_E[:, :2] - target_point_E[:, :2], dim=1)
    height_difference = torch.abs(object_point_E[:, 2] - target_point_E[:, 2])
    velocity = torch.abs(obj.data.root_vel_w.torch)
    is_matching_variant = True
    if required_variant_index is not None:
        can_set = getattr(env, "_arena_can_object_set", None)
        if can_set is None:
            can_set = env.unwrapped.task._can_object_set
        active_variant_index = can_set.variant_indices_by_env
        is_matching_variant = torch.tensor(
            [index == required_variant_index for index in active_variant_index], device=obj_pos.device
        )
    return (
        (xy_distance < max_xy_distance)
        & (height_difference < max_height_difference)
        & (obj_pos[:, 2] < max_height)
        & torch.all(velocity < max_speed, dim=1)
        & is_matching_variant
    )


def _configure_two_arm_can_sort_random(
    env_cfg: IsaacLabArenaManagerBasedRLEnvCfg,
) -> IsaacLabArenaManagerBasedRLEnvCfg:
    """Apply 50 Hz control, glove feedback, and camera recording settings."""
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import set_control_rate_50hz
    from isaaclab_arena_environments.gr1_pick_place_environment import _configure_gr1_pick_place

    env_cfg = _configure_gr1_pick_place(env_cfg)
    env_cfg = set_control_rate_50hz(env_cfg)
    if hasattr(env_cfg.scene, "robot_pov_cam"):
        env_cfg.image_obs_list = ["robot_pov_cam", "left_wrist_cam", "right_wrist_cam"]
    return env_cfg


@register_environment
class GR1TwoArmCanSortRandomEnvironment(ArenaEnvironmentFactory[GR1TwoArmCanSortRandomEnvironmentCfg]):
    """Hand a randomly colored can from right to left and place it into its matching box."""

    name = "gr1_two_arm_can_sort_random"
    _legacy_argparse_cfg_type = GR1TwoArmCanSortRandomEnvironmentCfg

    @staticmethod
    def _add_legacy_cli_only_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--object",
            type=str,
            default=None,
            help="Accepted for CLI compatibility only; use --can_sort_blue_box_path and --can_sort_red_box_path to select assets.",
        )

    def build(self, cfg: GR1TwoArmCanSortRandomEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the custom task from stock GR1 configs without importing the custom Lab environment."""
        import isaaclab.envs.mdp as mdp
        import isaaclab.sim as sim_utils
        import isaaclab.utils.math as pose_utils
        from isaaclab.envs.mimic_env_cfg import (
            MimicEnvCfg,
            SubTaskConfig,
            SubTaskConstraintConfig,
            SubTaskConstraintType,
        )
        from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
        from isaaclab.scene import InteractiveSceneCfg
        from isaaclab.sensors import CameraCfg
        from isaaclab.utils.configclass import configclass
        from isaaclab_tasks.contrib.pick_place.pickplace_gr1t2_env_cfg import PickPlaceGR1T2SceneCfg, TerminationsCfg

        from isaaclab_arena.assets.object import Object
        from isaaclab_arena.assets.object_type import ObjectType
        from isaaclab_arena.embodiments.common.arm_mode import ArmMode
        from isaaclab_arena.embodiments.gr1t2.gr1t2 import GR1T2CameraCfg, GR1T2MimicEnv
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.common.mimic_default_params import MIMIC_DATAGEN_CONFIG_DEFAULTS
        from isaaclab_arena.tasks.goal_pose_task import GoalPoseTask
        from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg
        from isaaclab_arena.utils.configclass import combine_configclass_instances
        from isaaclab_arena.utils.pose import Pose

        # Isaac Lab Mimic calls this helper for TRANSFORM coordination. Some
        # Isaac Lab versions do not expose it from isaaclab.utils.math.
        if not hasattr(pose_utils, "get_delta_object_pose"):
            pose_utils.get_delta_object_pose = lambda cur_obj_pose, src_obj_pose: pose_utils.pose_in_A_to_pose_in_B(
                pose_in_A=pose_utils.pose_inv(src_obj_pose), pose_A_in_B=cur_obj_pose
            )

        source_scene = PickPlaceGR1T2SceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)

        @configclass
        class CanSortRandomSceneCfg(InteractiveSceneCfg):
            left_hand_contact = source_scene.left_hand_contact.replace(filter_prim_paths_expr=["{ENV_REGEX_NS}/Can"])
            right_hand_contact = source_scene.right_hand_contact.replace(
                filter_prim_paths_expr=["{ENV_REGEX_NS}/Can"]
            )

        @configclass
        class CanSortRandomEventsCfg:
            reset_all = EventTermCfg(func=mdp.reset_scene_to_default, mode="reset")
            reset_can = EventTermCfg(
                func=mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {"x": (-0.04, 0.04), "y": (-0.04, 0.04), "yaw": (-1.57079, 1.57079)},
                    "velocity_range": {},
                    "asset_cfg": SceneEntityCfg("can"),
                },
            )
            randomize_box_sides = EventTermCfg(func=_randomize_box_sides, mode="reset")

        @configclass
        class CanSortRandomPolicyCfg(ObservationGroupCfg):
            can_pos = ObservationTermCfg(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("can")})
            can_rot = ObservationTermCfg(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("can")})
            blue_box_pos = ObservationTermCfg(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("blue_box")})
            blue_box_rot = ObservationTermCfg(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("blue_box")})
            red_box_pos = ObservationTermCfg(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("red_box")})
            red_box_rot = ObservationTermCfg(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("red_box")})
            concatenate_terms = False
            enable_corruption = False

        @configclass
        class CanSortRandomSubtaskTermsCfg(ObservationGroupCfg):
            # Both arms' first subtasks end at the handoff pose. The right arm
            # reaches it first; sequential constraints then let the left arm take over.
            handoff = ObservationTermCfg(func=_object_near_left_palm, params={"object_cfg": SceneEntityCfg("can")})
            concatenate_terms = False
            enable_corruption = False

        @configclass
        class CanSortRandomObservationsCfg:
            subtask_terms: CanSortRandomSubtaskTermsCfg = CanSortRandomSubtaskTermsCfg()

        class CanSortRandomMimicEnv(GR1T2MimicEnv):
            def get_subtask_term_signals(self, env_ids=None):
                if env_ids is None:
                    env_ids = slice(None)
                terms = self.obs_buf["subtask_terms"]
                return {"handoff": terms["handoff"][env_ids]}

        @configclass
        class CanSortRandomCameraCfg(GR1T2CameraCfg):
            left_wrist_cam = CameraCfg(
                prim_path="{ENV_REGEX_NS}/Robot/left_hand_pitch_link/LeftWristCam",
                update_period=0.0,
                height=512,
                width=512,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(focal_length=18.15, clipping_range=(0.1, 2.0)),
                offset=CameraCfg.OffsetCfg(
                    pos=(0.0, 0.15, 0.0), rot=(0.92387953, 0.0, 0.0, 0.38268343), convention="ros"
                ),
            )
            right_wrist_cam = CameraCfg(
                prim_path="{ENV_REGEX_NS}/Robot/right_hand_pitch_link/RightWristCam",
                update_period=0.0,
                height=512,
                width=512,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(focal_length=18.15, clipping_range=(0.1, 2.0)),
                offset=CameraCfg.OffsetCfg(
                    pos=(0.0, -0.15, 0.0), rot=(-0.92387953, 0.0, 0.0, 0.38268343), convention="ros"
                ),
            )

        class CanSortRandomCompletionCriteria(CompletionCriteria):
            """Keep derived predicate tuples out of the serialized environment config."""

            def __post_init__(self):
                # Retain stock validation, but store only constructor fields for Hydra.
                self.logical = CompletionCriteria(**vars(self)).logical

            @property
            def canonical_predicate_sequences(self):
                return CompletionCriteria(**vars(self)).canonical_predicate_sequences

        class TwoArmCanSortRandomTask(GoalPoseTask):
            def __init__(self, *args, target_box_name: str, **kwargs):
                super().__init__(*args, **kwargs)
                self.target_box_name = target_box_name

            def get_observation_cfg(self):
                return CanSortRandomObservationsCfg()

            def get_termination_cfg(self) -> TaskTerminationCfg:
                source_terminations = TerminationsCfg()
                success_blue = source_terminations.success.replace(
                    func=_object_near_target,
                    params={
                        "object_cfg": SceneEntityCfg("can"),
                        "target_cfg": SceneEntityCfg("blue_box"),
                        "max_xy_distance": cfg.success_max_xy_distance,
                        "max_height_difference": cfg.success_max_height_difference,
                        "max_speed": cfg.success_max_speed,
                        "object_reference_offset": tuple(cfg.can_reference_offset),
                        "target_reference_offset": tuple(cfg.box_reference_offset),
                        "required_variant_index": 0,
                    },
                )
                success_red = source_terminations.success.replace(
                    func=_object_near_target,
                    params={
                        "object_cfg": SceneEntityCfg("can"),
                        "target_cfg": SceneEntityCfg("red_box"),
                        "max_xy_distance": cfg.success_max_xy_distance,
                        "max_height_difference": cfg.success_max_height_difference,
                        "max_speed": cfg.success_max_speed,
                        "object_reference_offset": tuple(cfg.can_reference_offset),
                        "target_reference_offset": tuple(cfg.box_reference_offset),
                        "required_variant_index": 1,
                    },
                )
                return TaskTerminationCfg(
                    timeout_s=self.episode_length_s,
                    success=[
                        CanSortRandomCompletionCriteria(name="blue_can_in_blue_box", predicate_sequence=[success_blue]),
                        CanSortRandomCompletionCriteria(name="red_can_in_red_box", predicate_sequence=[success_red]),
                    ],
                failures={
                    "can_dropping": source_terminations.object_dropping.replace(
                        params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("can")}
                    )
                },
                )

            def get_mimic_env_cfg(self, arm_mode: ArmMode):
                assert arm_mode == ArmMode.DUAL_ARM, "Can sort random one arm in Mimic demonstrations"
                mimic_cfg = MimicEnvCfg()
                mimic_cfg.datagen_config.name = "gr1_two_arm_can_sort_random_D0"
                for key, value in MIMIC_DATAGEN_CONFIG_DEFAULTS.items():
                    setattr(mimic_cfg.datagen_config, key, value)
                mimic_cfg.subtask_configs["right"] = [
                    SubTaskConfig(
                        object_ref="can",
                        subtask_term_signal="handoff",
                        first_subtask_start_offset_range=(0, 0),
                        subtask_term_offset_range=(0, 0),
                        selection_strategy="nearest_neighbor_object",
                        selection_strategy_kwargs={"nn_k": 3},
                        action_noise=0.003,
                        num_interpolation_steps=0,
                        num_fixed_steps=0,
                        apply_noise_during_interpolation=False,
                    ),
                    SubTaskConfig(
                        object_ref=self.target_box_name,
                        subtask_term_signal=None,
                        subtask_term_offset_range=(0, 0),
                        selection_strategy="nearest_neighbor_object",
                        selection_strategy_kwargs={"nn_k": 3},
                        action_noise=0.003,
                        num_interpolation_steps=3,
                        num_fixed_steps=0,
                        apply_noise_during_interpolation=False,
                    ),
                ]
                mimic_cfg.subtask_configs["left"] = [
                    SubTaskConfig(
                        object_ref="can",
                        subtask_term_signal="handoff",
                        first_subtask_start_offset_range=(0, 0),
                        subtask_term_offset_range=(0, 0),
                        selection_strategy="nearest_neighbor_object",
                        selection_strategy_kwargs={"nn_k": 3},
                        action_noise=0.003,
                        num_interpolation_steps=0,
                        num_fixed_steps=0,
                        apply_noise_during_interpolation=False,
                    ),
                    SubTaskConfig(
                        object_ref=self.target_box_name,
                        subtask_term_signal=None,
                        subtask_term_offset_range=(0, 0),
                        selection_strategy="nearest_neighbor_object",
                        selection_strategy_kwargs={"nn_k": 3},
                        action_noise=0.003,
                        num_interpolation_steps=3,
                        num_fixed_steps=0,
                        apply_noise_during_interpolation=False,
                    ),
                ]
                mimic_cfg.task_constraint_configs = [
                    SubTaskConstraintConfig(
                        eef_subtask_constraint_tuple=[("right", 0), ("left", 0)],
                        constraint_type=SubTaskConstraintType.SEQUENTIAL,
                    )
                ]
                return mimic_cfg

        embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(enable_cameras=cfg.enable_cameras)
        embodiment.arm_mode = ArmMode.DUAL_ARM
        embodiment.mimic_env = CanSortRandomMimicEnv
        embodiment.scene_config.robot = source_scene.robot.copy()
        # The source robot faces +Y. XR forward maps to +Y with an identity anchor;
        # FOLLOW_PRIM_SMOOTHED follows only pelvis yaw changes, not its initial yaw.
        embodiment.xr.anchor_rot = (0.0, 0.0, 0.0, 1.0)
        # embodiment.scene_config.robot.init_state.joint_pos.update(
        #     {"right_elbow_pitch_joint": -2.260, "left_elbow_pitch_joint": -2.260}
        # )
        embodiment.set_initial_pose(
            Pose(
                position_xyz=tuple(source_scene.robot.init_state.pos),
                rotation_xyzw=tuple(source_scene.robot.init_state.rot),
            )
        )
        if cfg.embodiment == "gr1_pink":
            embodiment.action_config.upper_body_ik.controller.usd_path = embodiment.scene_config.robot.spawn.usd_path
        robot_pov_cam = source_scene.robot_pov_cam.copy()
        robot_pov_cam.height = 512
        robot_pov_cam.width = 512
        embodiment.camera_config = CanSortRandomCameraCfg(robot_pov_cam=robot_pov_cam)
        embodiment.observation_config.policy = combine_configclass_instances(
            "GR1TwoArmCanSortRandomPolicyCfg",
            embodiment.observation_config.policy,
            CanSortRandomPolicyCfg(),
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
        blue_box = Object(
            name="blue_box",
            prim_path="{ENV_REGEX_NS}/BlueBox",
            object_type=ObjectType.RIGID,
            spawner_cfg=source_scene.object.spawn.replace(
                usd_path=cfg.can_sort_blue_box_path, scale=BOX_SCALE, activate_contact_sensors=True
            ),
            initial_pose=Pose(position_xyz=BOX_LEFT_POSITION),
        )
        red_box = Object(
            name="red_box",
            prim_path="{ENV_REGEX_NS}/RedBox",
            object_type=ObjectType.RIGID,
            spawner_cfg=source_scene.object.spawn.replace(
                usd_path=cfg.can_sort_red_box_path, scale=BOX_SCALE, activate_contact_sensors=True
            ),
            initial_pose=Pose(position_xyz=BOX_RIGHT_POSITION),
        )
        from isaaclab_arena.assets.object_set import RigidObjectSet

        blue_can = Object(
            name="blue_can_variant",
            usd_path=cfg.can_sort_blue_can_path,
            scale=CAN_SCALE,
        )
        red_can = Object(
            name="red_can_variant",
            usd_path=cfg.can_sort_red_can_path,
            scale=CAN_SCALE,
        )
        can = RigidObjectSet(
            name="can",
            objects=[blue_can, red_can],
            random_choice=True,
            prim_path="{ENV_REGEX_NS}/Can",
            initial_pose=Pose(position_xyz=CAN_POSITION),
        )
        can.assign_variants(num_envs=1)
        can.disable_reset_pose()
        blue_box.disable_reset_pose()
        red_box.disable_reset_pose()
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
        scene = Scene(assets=[table, blue_box, red_box, can, ground, light])
        color_name = "blue" if can.variant_indices_by_env[0] == 0 else "red"
        target_box_name = f"{color_name}_box"
        task = TwoArmCanSortRandomTask(
            object=can,
            episode_length_s=cfg.episode_length_s,
            target_box_name=target_box_name,
        )
        task._can_object_set = can
        task.task_description = (
            f"Pick up the {color_name} can with the right hand, hand it to the left hand, "
            f"and place it inside the {color_name} box. The blue and red boxes are on the left; "
            "their left-right positions are randomized."
        )
        task.scene_config = CanSortRandomSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
        task.events_cfg = CanSortRandomEventsCfg()

        teleop_device = None
        if cfg.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(cfg.teleop_device)()
        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=teleop_device,
            env_cfg_callback=_configure_two_arm_can_sort_random,
        )
