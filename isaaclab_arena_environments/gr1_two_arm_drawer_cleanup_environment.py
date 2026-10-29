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
MUG_POSITION_XY = (-0.2, 0.30)
DRAWER_POSITION_XY = (0.10, 0.40)
MUG_SCALE = (0.1, 0.1, 0.1)
DRAWER_SCALE = (1.0, 1.0, 1.0)
DRAWER_JOINT_NAME = "goal_slidey"
DRAWER_BODY_NAME = "drawer_link_joint_frame_0"


@dataclass
class GR1TwoArmDrawerCleanupEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the GR1T2 two-arm drawer-cleanup environment."""

    embodiment: str = "gr1_pink"
    teleop_device: str | None = None
    episode_length_s: float = 20.0
    drawer_cabinet_path: str = str(_ASSET_DIR / "drawer_cabinet.usd")
    drawer_mug_path: str = str(_ASSET_DIR / "drawer_mug.usd")
    success_max_xy_distance: float = 0.10
    success_max_height_difference: float = 0.08
    success_max_speed: float = 0.20
    """Componentwise mug velocity limits in m/s (linear) and rad/s (angular)."""
    drawer_open_distance: float = 0.20
    """Minimum opening displacement of the drawer's negative-Y slide joint [m]."""
    mug_reference_offset: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """Offset from the mug's bottom reference point in its scaled root frame [m]."""
    drawer_reference_offset: list[float] = field(default_factory=lambda: [0.0, 0.0, -0.08])
    """Placement reference point on the drawer floor in the moving drawer body's frame [m]."""

    def __post_init__(self) -> None:
        assert math.isfinite(self.episode_length_s) and self.episode_length_s > 0, "episode_length_s must be positive"
        if self.embodiment not in ("gr1_pink", "gr1_joint"):
            raise ValueError(
                f"Two-arm drawer cleanup requires a GR1T2 embodiment ('gr1_pink' or 'gr1_joint'); "
                f"got {self.embodiment!r}."
            )
        for name in (
            "success_max_xy_distance",
            "success_max_height_difference",
            "success_max_speed",
            "drawer_open_distance",
        ):
            value = getattr(self, name)
            assert math.isfinite(value) and value > 0, f"{name} must be finite and positive"
        for name in ("mug_reference_offset", "drawer_reference_offset"):
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


def _spawn_drawer_mug(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Add convex mesh collision to the visual-only mug asset in every cloned environment."""
    import isaaclab.sim as sim_utils
    from pxr import Usd, UsdGeom, UsdPhysics

    mug_prim = sim_utils.spawn_from_usd(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    stage = sim_utils.get_current_stage()
    for mug_path in sim_utils.find_matching_prim_paths(prim_path):
        for child in Usd.PrimRange(stage.GetPrimAtPath(mug_path)):
            if child.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(child).CreateCollisionEnabledAttr(True)
                UsdPhysics.MeshCollisionAPI.Apply(child).CreateApproximationAttr("convexHull")
    return mug_prim


def _drawer_opened(
    env: ManagerBasedRLEnv,
    drawer_cfg: SceneEntityCfg,
    min_distance: float,
) -> torch.Tensor:
    """Check opening displacement of the negative-Y drawer slide [m]."""
    drawer = env.scene[drawer_cfg.name]
    joint_id = drawer.joint_names.index(DRAWER_JOINT_NAME)
    return drawer.data.joint_pos.torch[:, joint_id] < -min_distance


def _mug_in_drawer(
    env: ManagerBasedRLEnv,
    mug_cfg: SceneEntityCfg,
    drawer_cfg: SceneEntityCfg,
    drawer_open_distance: float,
    max_xy_distance: float,
    max_height_difference: float,
    max_speed: float,
    mug_reference_offset: tuple[float, float, float],
    drawer_reference_offset: tuple[float, float, float],
) -> torch.Tensor:
    """Check mug placement relative to the open, moving drawer and mug velocity limits."""
    import torch

    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    mug = env.scene[mug_cfg.name]
    drawer = env.scene[drawer_cfg.name]
    body_id = drawer.body_names.index(DRAWER_BODY_NAME)
    # W/M/D denote world, mug root, and moving drawer body frames.
    T_W_D = drawer.data.body_link_pose_w.torch[:, body_id]
    mug_offset_M = T_W_D.new_tensor(mug_reference_offset).expand_as(mug.data.root_pos_w.torch)
    mug_point_W = mug.data.root_pos_w.torch + quat_apply(mug.data.root_quat_w.torch, mug_offset_M)
    mug_point_D = quat_apply_inverse(T_W_D[:, 3:7], mug_point_W - T_W_D[:, :3])
    delta_D = mug_point_D - mug_point_D.new_tensor(drawer_reference_offset)
    return (
        _drawer_opened(env, drawer_cfg, drawer_open_distance)
        & (torch.linalg.vector_norm(delta_D[:, :2], dim=1) < max_xy_distance)
        & (torch.abs(delta_D[:, 2]) < max_height_difference)
        & torch.all(torch.abs(mug.data.root_vel_w.torch) < max_speed, dim=1)
    )


def _configure_two_arm_drawer_cleanup(
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
class GR1TwoArmDrawerCleanupEnvironment(ArenaEnvironmentFactory[GR1TwoArmDrawerCleanupEnvironmentCfg]):
    """Lift a mug with the left hand, open the drawer with the right, and place the mug inside."""

    name = "gr1_two_arm_drawer_cleanup"
    _legacy_argparse_cfg_type = GR1TwoArmDrawerCleanupEnvironmentCfg

    @staticmethod
    def _add_legacy_cli_only_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--object",
            type=str,
            default=None,
            help=(
                "Accepted for CLI compatibility only; use --drawer_mug_path and --drawer_cabinet_path to select assets."
            ),
        )

    def build(self, cfg: GR1TwoArmDrawerCleanupEnvironmentCfg) -> IsaacLabArenaEnvironment:
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

        source_scene = PickPlaceGR1T2SceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
        source_events = EventCfg()
        # Use scaled USD extents so each object's bottom starts just above the table.
        positions = {}
        for name, path, scale, xy in (
            ("mug", cfg.drawer_mug_path, MUG_SCALE, MUG_POSITION_XY),
            ("drawer", cfg.drawer_cabinet_path, DRAWER_SCALE, DRAWER_POSITION_XY),
        ):
            bounds = compute_local_bounding_box_from_usd(path, scale)
            positions[name] = (*xy, TABLE_SURFACE_HEIGHT + SPAWN_CLEARANCE - bounds.min_point[0, 2].item())
        mug_bounds = compute_local_bounding_box_from_usd(cfg.drawer_mug_path, MUG_SCALE)
        mug_reference_offset = list(cfg.mug_reference_offset)
        mug_reference_offset[2] += mug_bounds.min_point[0, 2].item()

        @configclass
        class DrawerCleanupSceneCfg(InteractiveSceneCfg):
            left_hand_contact = source_scene.left_hand_contact.replace(filter_prim_paths_expr=["{ENV_REGEX_NS}/Mug"])
            right_hand_contact = source_scene.right_hand_contact.replace(
                filter_prim_paths_expr=["{ENV_REGEX_NS}/Drawer/Geometry/body_0/object/base/drawer_link_joint_frame_0"]
            )

        @configclass
        class DrawerCleanupEventsCfg(EventCfg):
            reset_object = None
            reset_mug = source_events.reset_object.replace(
                params={
                    "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "yaw": (-1.57079, 1.57079)},
                    "velocity_range": {},
                    "asset_cfg": SceneEntityCfg("mug"),
                }
            )
            reset_drawer = source_events.reset_object.replace(
                params={
                    "pose_range": {"x": (-0.05, 0.05), "y": (0.0, 0.05), "yaw": (-1.57079, 1.57079)},
                    "velocity_range": {},
                    "asset_cfg": SceneEntityCfg("drawer"),
                }
            )

        @configclass
        class DrawerCleanupPolicyCfg(ObservationGroupCfg):
            mug_pos = ObservationTermCfg(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("mug")})
            mug_rot = ObservationTermCfg(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("mug")})
            drawer_pos = ObservationTermCfg(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("drawer")})
            drawer_rot = ObservationTermCfg(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("drawer")})
            drawer_joint_pos = ObservationTermCfg(
                func=mdp.joint_pos,
                params={"asset_cfg": SceneEntityCfg("drawer", joint_names=[DRAWER_JOINT_NAME])},
            )
            concatenate_terms = False
            enable_corruption = False

        @configclass
        class DrawerCleanupSubtaskTermsCfg(ObservationGroupCfg):
            mug_lifted = ObservationTermCfg(
                func=_object_lifted,
                params={"object_cfg": SceneEntityCfg("mug"), "min_height": positions["mug"][2] + 0.05},
            )
            drawer_opened = ObservationTermCfg(
                func=_drawer_opened,
                params={"drawer_cfg": SceneEntityCfg("drawer"), "min_distance": cfg.drawer_open_distance},
            )
            concatenate_terms = False
            enable_corruption = False

        @configclass
        class DrawerCleanupObservationsCfg:
            subtask_terms: DrawerCleanupSubtaskTermsCfg = DrawerCleanupSubtaskTermsCfg()

        class DrawerCleanupMimicEnv(GR1T2MimicEnv):
            def get_object_poses(self, env_ids=None):
                if env_ids is None:
                    env_ids = slice(None)
                poses = super().get_object_poses(env_ids)
                drawer = self.scene["drawer"]
                body_id = drawer.body_names.index(DRAWER_BODY_NAME)
                T_W_D = drawer.data.body_link_pose_w.torch[env_ids, body_id]
                drawer_pos = T_W_D[:, :3] - self.scene.env_origins[env_ids]
                poses["drawer_inside"] = pose_utils.make_pose(
                    drawer_pos, pose_utils.matrix_from_quat(T_W_D[:, 3:7])
                )
                return poses

            def get_subtask_term_signals(self, env_ids=None):
                if env_ids is None:
                    env_ids = slice(None)
                terms = self.obs_buf["subtask_terms"]
                return {name: terms[name][env_ids] for name in ("mug_lifted", "drawer_opened")}

        @configclass
        class DrawerCleanupCameraCfg(GR1T2CameraCfg):
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

        class DrawerCleanupCompletionCriteria(CompletionCriteria):
            """Keep derived predicate tuples out of the serialized environment config."""

            def __post_init__(self):
                # Retain stock validation, but store only constructor fields for Hydra.
                self.logical = CompletionCriteria(**vars(self)).logical

            @property
            def canonical_predicate_sequences(self):
                return CompletionCriteria(**vars(self)).canonical_predicate_sequences

        class TwoArmDrawerCleanupTask(GoalPoseTask):
            def get_observation_cfg(self):
                return DrawerCleanupObservationsCfg()

            def get_termination_cfg(self) -> TaskTerminationCfg:
                source_terminations = TerminationsCfg()
                success = source_terminations.success.replace(
                    func=_mug_in_drawer,
                    params={
                        "mug_cfg": SceneEntityCfg("mug"),
                        "drawer_cfg": SceneEntityCfg("drawer"),
                        "drawer_open_distance": cfg.drawer_open_distance,
                        "max_xy_distance": cfg.success_max_xy_distance,
                        "max_height_difference": cfg.success_max_height_difference,
                        "max_speed": cfg.success_max_speed,
                        "mug_reference_offset": tuple(mug_reference_offset),
                        "drawer_reference_offset": tuple(cfg.drawer_reference_offset),
                    },
                )
                return TaskTerminationCfg(
                    timeout_s=self.episode_length_s,
                    success=[DrawerCleanupCompletionCriteria(name="drawer_cleanup", predicate_sequence=[success])],
                    failures={
                        "mug_dropping": source_terminations.object_dropping.replace(
                            params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("mug")}
                        )
                    },
                )

            def get_mimic_env_cfg(self, arm_mode: ArmMode):
                assert arm_mode == ArmMode.DUAL_ARM, "Drawer cleanup requires both arms in Mimic demonstrations"
                mimic_cfg = MimicEnvCfg()
                mimic_cfg.datagen_config.name = "gr1_two_arm_drawer_cleanup_D0"
                for key, value in MIMIC_DATAGEN_CONFIG_DEFAULTS.items():
                    setattr(mimic_cfg.datagen_config, key, value)
                for arm, pickup_ref, term_signal in (
                    ("left", "mug", "mug_lifted"),
                    ("right", "drawer", "drawer_opened"),
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
                            object_ref="drawer_inside",
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
                # Keep the right hand idle until pickup; separate stages avoid overlapping constraints.
                mimic_cfg.subtask_configs["right"].insert(
                    0,
                    SubTaskConfig(
                        object_ref=None,
                        subtask_term_signal="mug_lifted",
                        subtask_term_offset_range=(0, 0),
                        selection_strategy="random",
                        action_noise=0.0,
                        num_interpolation_steps=0,
                        num_fixed_steps=0,
                        apply_noise_during_interpolation=False,
                    ),
                )
                mimic_cfg.task_constraint_configs = [
                    SubTaskConstraintConfig(
                        eef_subtask_constraint_tuple=[("left", 0), ("right", 0)],
                        constraint_type=SubTaskConstraintType.SEQUENTIAL,
                        sequential_min_time_diff=1,
                    ),
                    SubTaskConstraintConfig(
                        eef_subtask_constraint_tuple=[("right", 1), ("left", 1)],
                        constraint_type=SubTaskConstraintType.SEQUENTIAL,
                    ),
                ]
                return mimic_cfg

        embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(enable_cameras=cfg.enable_cameras)
        embodiment.arm_mode = ArmMode.DUAL_ARM
        embodiment.mimic_env = DrawerCleanupMimicEnv
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
        embodiment.camera_config = DrawerCleanupCameraCfg(robot_pov_cam=robot_pov_cam)
        embodiment.observation_config.policy = combine_configclass_instances(
            "GR1TwoArmDrawerCleanupPolicyCfg",
            embodiment.observation_config.policy,
            DrawerCleanupPolicyCfg(),
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
        mug_spawn_cfg = sim_utils.UsdFileCfg(
            func=_spawn_drawer_mug,
            usd_path=cfg.drawer_mug_path,
            scale=MUG_SCALE,
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=False),
            mass_props=sim_utils.MassCfg(mass=0.15),
            mass_props_create_if_missing=True,
            activate_contact_sensors=True,
        )
        mug = Object(
            name="mug",
            prim_path="{ENV_REGEX_NS}/Mug",
            object_type=ObjectType.RIGID,
            spawner_cfg=mug_spawn_cfg,
            initial_pose=Pose(position_xyz=positions["mug"]),
        )
        drawer = Object(
            name="drawer",
            prim_path="{ENV_REGEX_NS}/Drawer",
            object_type=ObjectType.ARTICULATION,
            spawner_cfg=sim_utils.UsdFileCfg(
                usd_path=cfg.drawer_cabinet_path,
                scale=DRAWER_SCALE,
                fix_root_link=True,
                activate_contact_sensors=True,
                joint_drive_props={
                    "/Physics/goal_slidey": [sim_utils.UsdPhysicsDriveCfg(stiffness=0.0, damping=0.0)]
                },
            ),
            initial_pose=Pose(position_xyz=positions["drawer"]),
        )
        # reset_all closes the drawer; the pose events independently randomize the mug and the cabinet.
        mug.disable_reset_pose()
        drawer.disable_reset_pose()
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
        scene = Scene(assets=[table, mug, drawer, ground, light])
        task = TwoArmDrawerCleanupTask(object=mug, episode_length_s=cfg.episode_length_s)
        task.task_description = (
            "Pick up the mug with the left hand, open the drawer with the right hand, "
            "then place the mug inside the drawer with the left hand."
        )
        task.scene_config = DrawerCleanupSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
        task.events_cfg = DrawerCleanupEventsCfg()

        teleop_device = None
        if cfg.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(cfg.teleop_device)()
        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=teleop_device,
            env_cfg_callback=_configure_two_arm_drawer_cleanup,
        )
