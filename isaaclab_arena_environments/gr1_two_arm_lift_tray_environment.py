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


_ASSET_DIR = Path(__file__).resolve().parents[1] / "assets" / "usd"
TABLE_SURFACE_HEIGHT = 1.0
SPAWN_CLEARANCE = 0.002
OBJ0_POSITION_XY = (-0.18, 0.40)
OBJ1_POSITION_XY = (0.18, 0.40)
POT_POSITION_XY = (0.0, 0.2)
OBJ0_SCALE = (1.0, 1.0, 1.0)
OBJ1_SCALE = (1.0, 1.0, 1.0)
POT_SCALE = (1.0, 1.0, 0.2)


@dataclass
class GR1TwoArmLiftTrayEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the GR1T2 two-arm lift-tray environment."""

    embodiment: str = "gr1_pink"
    teleop_device: str | None = None
    episode_length_s: float = 20.0
    lift_tray_obj0_path: str = str(_ASSET_DIR / "lift_tray_obj0.usd")
    lift_tray_obj1_path: str = str(_ASSET_DIR / "lift_tray_obj1.usd")
    lift_tray_pot_path: str = str(_ASSET_DIR / "lift_tray_pot.usd")
    success_max_xy_distance: float = 0.10
    success_max_height_difference: float = 0.08
    success_max_speed: float = 0.20
    """Componentwise object velocity limits relative to the pot [m/s, rad/s]."""
    pickup_height: float = 0.05
    """Object lift above its initial root height for each Mimic pickup waypoint [m]."""
    pot_lift_height: float = 0.10
    """Required pot lift above its initial root height for task success [m]."""
    obj0_reference_offset: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """Containment reference point in the scaled obj0 root frame [m]."""
    obj1_reference_offset: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """Containment reference point in the scaled obj1 root frame [m]."""
    pot_reference_offset: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """Containment reference point in the scaled pot root frame [m]."""

    def __post_init__(self) -> None:
        assert math.isfinite(self.episode_length_s) and self.episode_length_s > 0, "episode_length_s must be positive"
        if self.embodiment not in ("gr1_pink", "gr1_joint"):
            raise ValueError(
                f"Two-arm lift tray requires a GR1T2 embodiment ('gr1_pink' or 'gr1_joint'); "
                f"got {self.embodiment!r}."
            )
        for name in (
            "success_max_xy_distance",
            "success_max_height_difference",
            "success_max_speed",
            "pickup_height",
            "pot_lift_height",
        ):
            value = getattr(self, name)
            assert math.isfinite(value) and value > 0, f"{name} must be finite and positive"
        for name in ("obj0_reference_offset", "obj1_reference_offset", "pot_reference_offset"):
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


def _object_in_pot(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    pot_cfg: SceneEntityCfg,
    max_xy_distance: float = 0.10,
    max_height_difference: float = 0.08,
    max_speed: float = 0.20,
    object_reference_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    pot_reference_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Approximate containment using pot-frame proximity [m] and relative velocity [m/s, rad/s]."""
    import torch

    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    obj = env.scene[object_cfg.name]
    pot = env.scene[pot_cfg.name]
    obj_pos = obj.data.root_pos_w.torch
    pot_pos = pot.data.root_pos_w.torch
    object_offset_O = obj_pos.new_tensor(object_reference_offset).expand_as(obj_pos)
    pot_offset_P = pot_pos.new_tensor(pot_reference_offset).expand_as(pot_pos)
    # O/P are object/pot root frames; W is the simulation world frame.
    object_point_W = obj_pos + quat_apply(obj.data.root_quat_w.torch, object_offset_O)
    pot_point_W = pot_pos + quat_apply(pot.data.root_quat_w.torch, pot_offset_P)
    displacement_P = quat_apply_inverse(pot.data.root_quat_w.torch, object_point_W - pot_point_W)
    xy_distance = torch.linalg.vector_norm(displacement_P[:, :2], dim=1)
    height_difference = torch.abs(displacement_P[:, 2])
    # A carried object may move with the pot; reject motion relative to that rigid motion instead.
    pot_velocity = pot.data.root_vel_w.torch
    expected_linear_velocity = pot_velocity[:, :3] + torch.cross(
        pot_velocity[:, 3:], obj_pos - pot_pos, dim=1
    )
    relative_linear_velocity = obj.data.root_vel_w.torch[:, :3] - expected_linear_velocity
    relative_angular_velocity = obj.data.root_vel_w.torch[:, 3:] - pot_velocity[:, 3:]
    return (
        (xy_distance < max_xy_distance)
        & (height_difference < max_height_difference)
        & torch.all(torch.abs(relative_linear_velocity) < max_speed, dim=1)
        & torch.all(torch.abs(relative_angular_velocity) < max_speed, dim=1)
    )


def _objects_in_pot(
    env: ManagerBasedRLEnv,
    obj0_cfg: SceneEntityCfg,
    obj1_cfg: SceneEntityCfg,
    pot_cfg: SceneEntityCfg,
    max_xy_distance: float,
    max_height_difference: float,
    max_speed: float,
    obj0_reference_offset: tuple[float, float, float],
    obj1_reference_offset: tuple[float, float, float],
    pot_reference_offset: tuple[float, float, float],
    min_pot_height: float | None = None,
) -> torch.Tensor:
    """Require both objects in the pot, optionally above an environment-local pot height [m]."""
    obj0_placed = _object_in_pot(
        env,
        object_cfg=obj0_cfg,
        pot_cfg=pot_cfg,
        max_xy_distance=max_xy_distance,
        max_height_difference=max_height_difference,
        max_speed=max_speed,
        object_reference_offset=obj0_reference_offset,
        pot_reference_offset=pot_reference_offset,
    )
    obj1_placed = _object_in_pot(
        env,
        object_cfg=obj1_cfg,
        pot_cfg=pot_cfg,
        max_xy_distance=max_xy_distance,
        max_height_difference=max_height_difference,
        max_speed=max_speed,
        object_reference_offset=obj1_reference_offset,
        pot_reference_offset=pot_reference_offset,
    )
    objects_placed = obj0_placed & obj1_placed
    if min_pot_height is not None:
        objects_placed &= _object_lifted(env, pot_cfg, min_pot_height)
    return objects_placed


def _configure_two_arm_lift_tray(
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
class GR1TwoArmLiftTrayEnvironment(ArenaEnvironmentFactory[GR1TwoArmLiftTrayEnvironmentCfg]):
    """Independently place obj0 with the left hand and obj1 with the right, then lift the pot together."""

    name = "gr1_two_arm_lift_tray"
    _legacy_argparse_cfg_type = GR1TwoArmLiftTrayEnvironmentCfg

    @staticmethod
    def _add_legacy_cli_only_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--object",
            type=str,
            default=None,
            help=(
                "Accepted for CLI compatibility only; use --lift_tray_obj0_path, --lift_tray_obj1_path, "
                "and --lift_tray_pot_path to select assets."
            ),
        )

    def build(self, cfg: GR1TwoArmLiftTrayEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the custom task from stock GR1 configs without importing the custom Lab environment."""
        import isaaclab.envs.mdp as mdp
        import isaaclab.sim as sim_utils
        import isaaclab.utils.math as pose_utils
        from isaaclab.envs.mimic_env_cfg import (
            MimicEnvCfg,
            SubTaskConfig,
            SubTaskConstraintConfig,
            SubTaskConstraintCoordinationScheme,
            SubTaskConstraintType,
        )
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
        from isaaclab_arena.embodiments.gr1t2.gr1t2 import GR1T2CameraCfg, GR1T2MimicEnv
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.common.mimic_default_params import MIMIC_DATAGEN_CONFIG_DEFAULTS
        from isaaclab_arena.tasks.goal_pose_task import GoalPoseTask
        from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg
        from isaaclab_arena.utils.configclass import combine_configclass_instances
        from isaaclab_arena.utils.pose import Pose
        from isaaclab_arena.utils.usd.helpers import compute_local_bounding_box_from_usd

        # Isaac Lab Mimic calls this helper for TRANSFORM coordination. Some
        # Isaac Lab versions do not expose it from isaaclab.utils.math.
        if not hasattr(pose_utils, "get_delta_object_pose"):
            pose_utils.get_delta_object_pose = lambda cur_obj_pose, src_obj_pose: pose_utils.pose_in_A_to_pose_in_B(
                pose_in_A=pose_utils.pose_inv(src_obj_pose), pose_A_in_B=cur_obj_pose
            )

        source_scene = PickPlaceGR1T2SceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
        source_events = EventCfg()
        # Use scaled USD extents so each object's bottom starts just above the table.
        positions = {}
        for name, path, scale, xy in (
            ("obj0", cfg.lift_tray_obj0_path, OBJ0_SCALE, OBJ0_POSITION_XY),
            ("obj1", cfg.lift_tray_obj1_path, OBJ1_SCALE, OBJ1_POSITION_XY),
            ("pot", cfg.lift_tray_pot_path, POT_SCALE, POT_POSITION_XY),
        ):
            box = compute_local_bounding_box_from_usd(path, scale)
            positions[name] = (*xy, TABLE_SURFACE_HEIGHT + SPAWN_CLEARANCE - box.min_point[0, 2].item())

        objects_in_pot_params = {
            "obj0_cfg": SceneEntityCfg("obj0"),
            "obj1_cfg": SceneEntityCfg("obj1"),
            "pot_cfg": SceneEntityCfg("pot"),
            "max_xy_distance": cfg.success_max_xy_distance,
            "max_height_difference": cfg.success_max_height_difference,
            "max_speed": cfg.success_max_speed,
            "obj0_reference_offset": tuple(cfg.obj0_reference_offset),
            "obj1_reference_offset": tuple(cfg.obj1_reference_offset),
            "pot_reference_offset": tuple(cfg.pot_reference_offset),
        }

        @configclass
        class LiftTraySceneCfg(InteractiveSceneCfg):
            left_hand_contact = source_scene.left_hand_contact.copy()
            right_hand_contact = source_scene.right_hand_contact.copy()

        @configclass
        class LiftTrayEventsCfg(EventCfg):
            # Disable the inherited reset for the stock asset named "object".
            reset_object = None
            reset_obj0 = source_events.reset_object.replace(
                params={
                    "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "yaw": (-1.57079, 1.57079)},
                    "velocity_range": {},
                    "asset_cfg": SceneEntityCfg("obj0"),
                }
            )
            reset_obj1 = source_events.reset_object.replace(
                params={
                    "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "yaw": (-1.57079, 1.57079)},
                    "velocity_range": {},
                    "asset_cfg": SceneEntityCfg("obj1"),
                }
            )
            reset_pot = source_events.reset_object.replace(
                params={
                    "pose_range": {"x": (-0.05, 0.05), "y": (0.0, 0.05), "yaw": (-1.57079, 1.57079)},
                    "velocity_range": {},
                    "asset_cfg": SceneEntityCfg("pot"),
                }
            )

        @configclass
        class LiftTrayPolicyCfg(ObservationGroupCfg):
            obj0_pos = ObservationTermCfg(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("obj0")})
            obj0_rot = ObservationTermCfg(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("obj0")})
            obj1_pos = ObservationTermCfg(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("obj1")})
            obj1_rot = ObservationTermCfg(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("obj1")})
            pot_pos = ObservationTermCfg(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("pot")})
            pot_rot = ObservationTermCfg(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("pot")})
            concatenate_terms = False
            enable_corruption = False

        @configclass
        class LiftTraySubtaskTermsCfg(ObservationGroupCfg):
            obj0_lifted = ObservationTermCfg(
                func=_object_lifted,
                params={"object_cfg": SceneEntityCfg("obj0"), "min_height": positions["obj0"][2] + cfg.pickup_height},
            )
            obj1_lifted = ObservationTermCfg(
                func=_object_lifted,
                params={"object_cfg": SceneEntityCfg("obj1"), "min_height": positions["obj1"][2] + cfg.pickup_height},
            )
            objects_placed = ObservationTermCfg(func=_objects_in_pot, params=objects_in_pot_params)
            concatenate_terms = False
            enable_corruption = False

        @configclass
        class LiftTrayObservationsCfg:
            subtask_terms: LiftTraySubtaskTermsCfg = LiftTraySubtaskTermsCfg()

        class LiftTrayMimicEnv(GR1T2MimicEnv):
            def get_subtask_term_signals(self, env_ids=None):
                if env_ids is None:
                    env_ids = slice(None)
                terms = self.obs_buf["subtask_terms"]
                return {name: terms[name][env_ids] for name in ("obj0_lifted", "obj1_lifted", "objects_placed")}

        @configclass
        class LiftTrayCameraCfg(GR1T2CameraCfg):
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

        class LiftTrayCompletionCriteria(CompletionCriteria):
            """Keep derived predicate tuples out of the serialized environment config."""

            def __post_init__(self):
                # Retain stock validation, but store only constructor fields for Hydra.
                self.logical = CompletionCriteria(**vars(self)).logical

            @property
            def canonical_predicate_sequences(self):
                return CompletionCriteria(**vars(self)).canonical_predicate_sequences

        class TwoArmLiftTrayTask(GoalPoseTask):
            def get_observation_cfg(self):
                return LiftTrayObservationsCfg()

            def get_termination_cfg(self) -> TaskTerminationCfg:
                source_terminations = TerminationsCfg()
                success = source_terminations.success.replace(
                    func=_objects_in_pot,
                    params={
                        **objects_in_pot_params,
                        "min_pot_height": positions["pot"][2] + cfg.pot_lift_height,
                    },
                )
                failures = {}
                for name in ("obj0", "obj1", "pot"):
                    failures[f"{name}_dropping"] = source_terminations.object_dropping.replace(
                        params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg(name)}
                    )
                return TaskTerminationCfg(
                    timeout_s=self.episode_length_s,
                    success=[LiftTrayCompletionCriteria(name="lift_tray", predicate_sequence=[success])],
                    failures=failures,
                )

            def get_mimic_env_cfg(self, arm_mode: ArmMode):
                assert arm_mode == ArmMode.DUAL_ARM, "Lift tray requires both arms in Mimic demonstrations"
                mimic_cfg = MimicEnvCfg()
                mimic_cfg.datagen_config.name = "gr1_two_arm_lift_tray_D0"
                for key, value in MIMIC_DATAGEN_CONFIG_DEFAULTS.items():
                    setattr(mimic_cfg.datagen_config, key, value)
                # Pickup waypoints are object-relative; placement and the shared lift are pot-relative.
                for arm, pickup_ref, term_signal in (
                    ("left", "obj0", "obj0_lifted"),
                    ("right", "obj1", "obj1_lifted"),
                ):
                    mimic_cfg.subtask_configs[arm] = [
                        SubTaskConfig(
                            object_ref=pickup_ref,
                            subtask_term_signal=term_signal,
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
                            object_ref="pot",
                            subtask_term_signal="objects_placed",
                            subtask_term_offset_range=(0, 0),
                            selection_strategy="nearest_neighbor_object",
                            selection_strategy_kwargs={"nn_k": 3},
                            action_noise=0.003,
                            num_interpolation_steps=3,
                            num_fixed_steps=0,
                            apply_noise_during_interpolation=False,
                        ),
                        SubTaskConfig(
                            object_ref="pot",
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
                # Pick and place independently in either order; coordinate only the final pot lift.
                mimic_cfg.task_constraint_configs = [
                    SubTaskConstraintConfig(
                        eef_subtask_constraint_tuple=[("left", 2), ("right", 2)],
                        constraint_type=SubTaskConstraintType.COORDINATION,
                        coordination_scheme=SubTaskConstraintCoordinationScheme.TRANSFORM,
                        coordination_synchronize_start=True,
                    ),
                ]
                return mimic_cfg

        embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(enable_cameras=cfg.enable_cameras)
        embodiment.arm_mode = ArmMode.DUAL_ARM
        embodiment.mimic_env = LiftTrayMimicEnv
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
        embodiment.camera_config = LiftTrayCameraCfg(robot_pov_cam=robot_pov_cam)
        embodiment.observation_config.policy = combine_configclass_instances(
            "GR1TwoArmLiftTrayPolicyCfg",
            embodiment.observation_config.policy,
            LiftTrayPolicyCfg(),
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
        obj0 = Object(
            name="obj0",
            prim_path="{ENV_REGEX_NS}/Obj0",
            object_type=ObjectType.RIGID,
            spawner_cfg=source_scene.object.spawn.replace(
                usd_path=cfg.lift_tray_obj0_path,
                scale=OBJ0_SCALE,
                activate_contact_sensors=True,
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
            ),
            initial_pose=Pose(position_xyz=positions["obj0"]),
        )
        obj1 = Object(
            name="obj1",
            prim_path="{ENV_REGEX_NS}/Obj1",
            object_type=ObjectType.RIGID,
            spawner_cfg=source_scene.object.spawn.replace(
                usd_path=cfg.lift_tray_obj1_path,
                scale=OBJ1_SCALE,
                activate_contact_sensors=True,
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
            ),
            initial_pose=Pose(position_xyz=positions["obj1"]),
        )
        pot = Object(
            name="pot",
            prim_path="{ENV_REGEX_NS}/Pot",
            object_type=ObjectType.RIGID,
            spawner_cfg=source_scene.object.spawn.replace(
                usd_path=cfg.lift_tray_pot_path, scale=POT_SCALE, activate_contact_sensors=True
            ),
            initial_pose=Pose(position_xyz=positions["pot"]),
        )
        # Task reset events randomize all three default poses without competing asset resets.
        for obj in (obj0, obj1, pot):
            obj.disable_reset_pose()
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
        scene = Scene(assets=[table, obj0, obj1, pot, ground, light])
        task = TwoArmLiftTrayTask(object=pot, episode_length_s=cfg.episode_length_s)
        task.task_description = (
            "Independently pick up and place obj0 with the left hand and obj1 with the right hand into the pot, "
            "in either order or simultaneously, then lift the pot with both hands."
        )
        task.scene_config = LiftTraySceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
        pot_contact_path = pot.get_contact_sensor_prim_path(cfg.lift_tray_pot_path)
        task.scene_config.left_hand_contact.filter_prim_paths_expr = [
            obj0.get_contact_sensor_prim_path(cfg.lift_tray_obj0_path),
            pot_contact_path,
        ]
        task.scene_config.right_hand_contact.filter_prim_paths_expr = [
            obj1.get_contact_sensor_prim_path(cfg.lift_tray_obj1_path),
            pot_contact_path,
        ]
        task.events_cfg = LiftTrayEventsCfg()

        teleop_device = None
        if cfg.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(cfg.teleop_device)()
        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=teleop_device,
            env_cfg_callback=_configure_two_arm_lift_tray,
        )
