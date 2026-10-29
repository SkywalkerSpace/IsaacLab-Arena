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

    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import IsaacLabArenaManagerBasedRLEnvCfg


_ASSET_DIR = Path(__file__).resolve().parents[1] / "assets" / "usd"
POSITION_XY_RANDOM_MAX = 0.05
TABLE_SURFACE_HEIGHT = 1.0
SPAWN_CLEARANCE = 0.002
COFFEE_POD_POSITION_XY = (-0.2, 0.35)
COFFEE_MACHINE_POSITION_XY = (0.10, 0.40)
COFFEE_MACHINE_ROTATION_XYZW = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
COFFEE_POD_GRASP_MAX_DISTANCE = 0.05
COFFEE_POD_GRASP_MIN_LIFT = 0.03


@dataclass
class GR1TwoArmCoffeeEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the GR1T2 two-arm coffee environment."""

    embodiment: str = "gr1_pink"
    teleop_device: str | None = None
    episode_length_s: float = 20.0
    randomization_seed: int | None = None
    """Layout seed; None samples a fresh seed on each launch, independently of Arena's seed."""
    coffee_machine_path: str = str(_ASSET_DIR / "coffee_machine.usd")
    coffee_pod_path: str = str(_ASSET_DIR / "coffee_pod.usd")
    success_max_distance: float = 0.05
    """Maximum distance from the pod center to the lid's local bounding box, in meters."""

    def __post_init__(self) -> None:
        assert math.isfinite(self.episode_length_s) and self.episode_length_s > 0, "episode_length_s must be positive"
        assert self.randomization_seed is None or (
            isinstance(self.randomization_seed, int) and 0 <= self.randomization_seed < 2**63
        ), "randomization_seed must be None or an integer in [0, 2**63)"
        if self.embodiment not in ("gr1_pink", "gr1_joint"):
            raise ValueError(
                f"Two-arm coffee requires a GR1T2 embodiment ('gr1_pink' or 'gr1_joint'); "
                f"got {self.embodiment!r}."
            )
        assert math.isfinite(self.success_max_distance) and self.success_max_distance > 0, (
            "success_max_distance must be finite and positive"
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


def _normalize_coffee_object(root):
    """Merge the coffee pod's fixed links into one root rigid body."""
    from pxr import Usd, UsdPhysics

    prims = list(Usd.PrimRange(root))
    joints = [prim for prim in prims if prim.IsA(UsdPhysics.Joint)]
    for prim in joints:
        assert prim.IsA(UsdPhysics.FixedJoint), f"Cannot merge a movable coffee joint: {prim.GetPath()}"
        prim.SetActive(False)
    for prim in prims:
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        if prim != root and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
        if prim.HasAPI(UsdPhysics.MassAPI):
            prim.RemoveAPI(UsdPhysics.MassAPI)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateRigidBodyEnabledAttr(True)
    return root


def _spawn_coffee_object(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Spawn the converted MJCF pod as a single rigid body without changing its source USD."""
    import isaaclab.sim as sim_utils

    prim = sim_utils.spawn_from_usd(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    stage = sim_utils.get_current_stage()
    for path in sim_utils.find_matching_prim_paths(prim_path):
        _normalize_coffee_object(stage.GetPrimAtPath(path))
        # Normalization moves physics ownership to the root, so reapply its settings there.
        if cfg.rigid_props is not None:
            sim_utils.modify_rigid_body_properties(path, cfg.rigid_props)
        if cfg.activate_contact_sensors:
            sim_utils.activate_contact_sensors(path)
    return prim


def _randomize_coffee_layout(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    positions: dict[str, tuple[float, float, float]],
    randomization_seed: int | None = None,
) -> None:
    """Jitter the coffee pod and machine independently by ±5 cm along each XY axis."""
    import secrets
    import torch

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device).reshape(-1)
    if len(env_ids) == 0:
        return
    if not hasattr(env, "_coffee_rng"):
        seed = secrets.randbits(63) if randomization_seed is None else randomization_seed
        env._coffee_rng = torch.Generator(device=env.device).manual_seed(seed)
        env._coffee_randomization_seed = seed
        print(f"[INFO] Two-arm coffee layout randomization seed: {seed}")
    # Do not reseed on reset: a fixed seed reproduces a sequence, not a single layout.
    offsets = (
        torch.rand((len(env_ids), 2, 2), device=env.device, generator=env._coffee_rng) * 2.0 - 1.0
    ) * POSITION_XY_RANDOM_MAX
    for offset_index, name in enumerate(("coffee_pod", "coffee_machine")):
        pose = torch.zeros((len(env_ids), 7), device=env.device)
        pose[:, :3] = torch.tensor(positions[name], device=env.device) + env.scene.env_origins[env_ids]
        pose[:, :2] += offsets[:, offset_index]
        pose[:, 6] = 1.0  # Isaac Lab 3.0 uses XYZW quaternions.
        if name == "coffee_machine":
            pose[:, 3:7] = pose.new_tensor(COFFEE_MACHINE_ROTATION_XYZW)
        asset = env.scene[name]
        asset.write_root_pose_to_sim(pose, env_ids=env_ids)
        if name == "coffee_machine":
            # The fixed articulation base has no velocity; reset its movable lid instead.
            asset.write_joint_state_to_sim(
                asset.data.default_joint_pos.torch[env_ids].clone(),
                asset.data.default_joint_vel.torch[env_ids].clone(),
                env_ids=env_ids,
            )
        else:
            asset.write_root_velocity_to_sim(torch.zeros((len(env_ids), 6), device=env.device), env_ids=env_ids)


def _coffee_object_dropped(env: ManagerBasedRLEnv, object_name: str, minimum_height: float = 0.5) -> torch.Tensor:
    """Detect a coffee object falling below an environment-local height [m]."""
    object_height = env.scene[object_name].data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    return object_height < minimum_height


def _coffee_pod_near_left_palm(env: ManagerBasedRLEnv, min_height: float) -> torch.Tensor:
    """Use coffee-pod lift and left-palm proximity as a grasp proxy."""
    import torch

    robot = env.scene["robot"]
    palm_index = robot.data.body_names.index("left_hand_pitch_link")
    coffee_pod_pos = env.scene["coffee_pod"].data.root_pos_w.torch
    palm_pos = robot.data.body_pos_w.torch[:, palm_index]
    lifted = coffee_pod_pos[:, 2] - env.scene.env_origins[:, 2] > min_height
    return lifted & (torch.linalg.vector_norm(coffee_pod_pos - palm_pos, dim=1) < COFFEE_POD_GRASP_MAX_DISTANCE)


def _coffee_pod_near_lid(
    env: ManagerBasedRLEnv,
    pod_center_offset: tuple[float, float, float],
    lid_body_name: str,
    lid_min: tuple[float, float, float],
    lid_max: tuple[float, float, float],
    max_distance: float,
) -> torch.Tensor:
    """Detect the pod center near the moving lid's local bounding box."""
    import torch

    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    coffee_pod = env.scene["coffee_pod"]
    coffee_machine = env.scene["coffee_machine"]
    lid_index = coffee_machine.data.body_names.index(lid_body_name)
    # W is world; L/P are the lid-link/pod root frames.
    q_W_L = coffee_machine.data.body_quat_w.torch[:, lid_index]
    p_W_L = coffee_machine.data.body_pos_w.torch[:, lid_index]
    q_W_P = coffee_pod.data.root_quat_w.torch
    p_W_P = coffee_pod.data.root_pos_w.torch
    center_P = p_W_P.new_tensor(pod_center_offset).expand_as(p_W_P)
    p_W_center = p_W_P + quat_apply(q_W_P, center_P)
    p_L_center = quat_apply_inverse(q_W_L, p_W_center - p_W_L)
    closest_L = torch.clamp(p_L_center, min=p_L_center.new_tensor(lid_min), max=p_L_center.new_tensor(lid_max))
    return torch.linalg.vector_norm(p_L_center - closest_L, dim=1) < max_distance


def _configure_two_arm_coffee(
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
class GR1TwoArmCoffeeEnvironment(ArenaEnvironmentFactory[GR1TwoArmCoffeeEnvironmentCfg]):
    """Pick up the coffee pod with the left hand and bring it near the coffee machine's lid."""

    name = "gr1_two_arm_coffee"
    _legacy_argparse_cfg_type = GR1TwoArmCoffeeEnvironmentCfg

    @staticmethod
    def _add_legacy_cli_only_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--object",
            type=str,
            default=None,
            help=(
                "Accepted for CLI compatibility only; use --coffee_pod_path and --coffee_machine_path to select assets."
            ),
        )

    def build(self, cfg: GR1TwoArmCoffeeEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the coffee-pod approach task from stock GR1 configs and two coffee USD assets."""
        import isaaclab.envs.mdp as mdp
        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
        from isaaclab.managers import (
            EventTermCfg,
            ObservationGroupCfg,
            ObservationTermCfg,
            SceneEntityCfg,
            TerminationTermCfg,
        )
        from isaaclab.scene import InteractiveSceneCfg
        from isaaclab.sensors import CameraCfg
        from isaaclab.utils.configclass import configclass
        from isaaclab_tasks.contrib.pick_place.pickplace_gr1t2_env_cfg import PickPlaceGR1T2SceneCfg
        from pxr import Usd, UsdGeom, UsdPhysics

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
        asset_paths = {
            "coffee_pod": cfg.coffee_pod_path,
            "coffee_machine": cfg.coffee_machine_path,
        }
        bounds = {}
        contact_paths = {}
        for name, path in asset_paths.items():
            contact_paths[name] = f"{{ENV_REGEX_NS}}/{name.capitalize()}"
            box = compute_local_bounding_box_from_usd(path, (1.0, 1.0, 1.0))
            bounds[name] = (box.min_point[0].tolist(), box.max_point[0].tolist())
        # Read actual USD extents rather than assuming the roots lie on the table surface.
        positions = {}
        for name, xy in (
            ("coffee_pod", COFFEE_POD_POSITION_XY),
            ("coffee_machine", COFFEE_MACHINE_POSITION_XY),
        ):
            positions[name] = (*xy, TABLE_SURFACE_HEIGHT + SPAWN_CLEARANCE - bounds[name][0][2])
        machine_stage = Usd.Stage.Open(cfg.coffee_machine_path)
        machine_root = machine_stage.GetDefaultPrim()
        lid_joints = [
            UsdPhysics.RevoluteJoint(prim)
            for prim in Usd.PrimRange(machine_root)
            if prim.IsA(UsdPhysics.RevoluteJoint)
        ]
        assert len(lid_joints) == 1, "Coffee machine must contain one lid hinge"
        lid_joint = lid_joints[0]
        lid_targets = lid_joint.GetBody1Rel().GetTargets()
        assert len(lid_targets) == 1 and "lid" in str(lid_targets[0]).casefold(), "Expected a lid hinge"
        lid_open_angle = math.radians(min(120.0, lid_joint.GetUpperLimitAttr().Get()))
        machine_contact_paths = []
        for prim in Usd.PrimRange(machine_root):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                relative_path = prim.GetPath().MakeRelativePath(machine_root.GetPath())
                machine_contact_paths.append(f"{contact_paths['coffee_machine']}/{relative_path}")
        lid_prim = machine_stage.GetPrimAtPath(lid_targets[0])
        assert lid_prim.HasAPI(UsdPhysics.RigidBodyAPI), "Lid hinge must target a rigid body"
        # Exclude the link's authored pose: the predicate uses its live articulation pose.
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
        lid_box = bbox_cache.ComputeUntransformedBound(lid_prim).ComputeAlignedBox()
        assert not lid_box.IsEmpty(), "Coffee machine lid must have nonempty geometry bounds"
        pod_min, pod_max = bounds["coffee_pod"]
        coffee_pod_near_lid_params = {
            "pod_center_offset": tuple((low + high) / 2.0 for low, high in zip(pod_min, pod_max)),
            "lid_body_name": lid_prim.GetName(),
            "lid_min": tuple(lid_box.GetMin()),
            "lid_max": tuple(lid_box.GetMax()),
            "max_distance": cfg.success_max_distance,
        }

        @configclass
        class CoffeeSceneCfg(InteractiveSceneCfg):
            left_hand_contact = source_scene.left_hand_contact.replace(
                filter_prim_paths_expr=["{ENV_REGEX_NS}/Coffee_pod", "{ENV_REGEX_NS}/Coffee_machine"]
            )
            right_hand_contact = source_scene.right_hand_contact.replace(
                filter_prim_paths_expr=["{ENV_REGEX_NS}/Coffee_pod", "{ENV_REGEX_NS}/Coffee_machine"]
            )

        @configclass
        class CoffeeEventsCfg:
            # The layout event resets the dynamic pod and the fixed-base machine articulation.
            reset_all = EventTermCfg(
                func=mdp.reset_root_state_uniform,
                mode="reset",
                params={"asset_cfg": SceneEntityCfg("robot"), "pose_range": {}, "velocity_range": {}},
            )
            reset_robot_joints = EventTermCfg(
                func=mdp.reset_joints_by_offset,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg("robot"),
                    "position_range": (0.0, 0.0),
                    "velocity_range": (0.0, 0.0),
                },
            )
            randomize_coffee_layout = EventTermCfg(
                func=_randomize_coffee_layout,
                mode="reset",
                params={"positions": positions, "randomization_seed": cfg.randomization_seed},
            )

        @configclass
        class CoffeePolicyCfg(ObservationGroupCfg):
            coffee_pod_pos = ObservationTermCfg(
                func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("coffee_pod")}
            )
            coffee_pod_rot = ObservationTermCfg(
                func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("coffee_pod")}
            )
            coffee_machine_pos = ObservationTermCfg(
                func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("coffee_machine")}
            )
            coffee_machine_rot = ObservationTermCfg(
                func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("coffee_machine")}
            )
            concatenate_terms = False
            enable_corruption = False

        @configclass
        class CoffeeSubtaskTermsCfg(ObservationGroupCfg):
            coffee_pod_grasped = ObservationTermCfg(
                func=_coffee_pod_near_left_palm,
                params={"min_height": positions["coffee_pod"][2] + COFFEE_POD_GRASP_MIN_LIFT},
            )
            concatenate_terms = False
            enable_corruption = False

        @configclass
        class CoffeeObservationsCfg:
            subtask_terms: CoffeeSubtaskTermsCfg = CoffeeSubtaskTermsCfg()

        class CoffeeMimicEnv(GR1T2MimicEnv):
            def get_subtask_term_signals(self, env_ids=None):
                if env_ids is None:
                    env_ids = slice(None)
                terms = self.obs_buf["subtask_terms"]
                return {"coffee_pod_grasped": terms["coffee_pod_grasped"][env_ids]}

        @configclass
        class CoffeeCameraCfg(GR1T2CameraCfg):
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

        class CoffeeCompletionCriteria(CompletionCriteria):
            """Keep derived predicate tuples out of the serialized environment config."""

            def __post_init__(self):
                self.logical = CompletionCriteria(**vars(self)).logical

            @property
            def canonical_predicate_sequences(self):
                return CompletionCriteria(**vars(self)).canonical_predicate_sequences

        class TwoArmCoffeeTask(GoalPoseTask):
            def get_metrics(self):
                from isaaclab_arena.metrics.success_rate import SuccessRateMetric

                return [SuccessRateMetric()]

            def get_observation_cfg(self):
                return CoffeeObservationsCfg()

            def get_termination_cfg(self) -> TaskTerminationCfg:
                return TaskTerminationCfg(
                    timeout_s=self.episode_length_s,
                    success=[
                        CoffeeCompletionCriteria(
                            name="coffee_pod_near_lid",
                            predicate_sequence=[
                                TerminationTermCfg(
                                    func=_coffee_pod_near_lid,
                                    params=coffee_pod_near_lid_params,
                                )
                            ],
                        )
                    ],
                    failures={
                        f"{name}_dropping": TerminationTermCfg(
                            func=_coffee_object_dropped, params={"object_name": name, "minimum_height": 0.5}
                        )
                        for name in ("coffee_pod", "coffee_machine")
                    },
                )

            def get_mimic_env_cfg(self, arm_mode: ArmMode):
                assert arm_mode == ArmMode.DUAL_ARM, "Two-arm coffee requires dual-arm Mimic demonstrations"
                mimic_cfg = MimicEnvCfg()
                mimic_cfg.datagen_config.name = "gr1_two_arm_coffee_D0"
                for key, value in MIMIC_DATAGEN_CONFIG_DEFAULTS.items():
                    setattr(mimic_cfg.datagen_config, key, value)
                # Split at the left-hand grasp, then adapt the approach to the randomized machine pose.
                object_refs = {
                    "left": ("coffee_pod", "coffee_machine"),
                    "right": ("coffee_machine", "coffee_machine"),
                }
                term_signals = ("coffee_pod_grasped", None)
                for arm, refs in object_refs.items():
                    mimic_cfg.subtask_configs[arm] = [
                        SubTaskConfig(
                            object_ref=object_ref,
                            subtask_term_signal=term_signals[index],
                            first_subtask_start_offset_range=(0, 0),
                            subtask_term_offset_range=(0, 0),
                            selection_strategy="nearest_neighbor_object",
                            selection_strategy_kwargs={"nn_k": 3},
                            action_noise=0.003,
                            num_interpolation_steps=0 if index == 0 else 3,
                            num_fixed_steps=0,
                            apply_noise_during_interpolation=False,
                        )
                        for index, object_ref in enumerate(refs)
                    ]
                return mimic_cfg

        embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(enable_cameras=cfg.enable_cameras)
        embodiment.arm_mode = ArmMode.DUAL_ARM
        embodiment.mimic_env = CoffeeMimicEnv
        embodiment.scene_config.robot = source_scene.robot.copy()
        # The source robot faces +Y. XR forward maps to +Y with an identity anchor;
        # FOLLOW_PRIM_SMOOTHED follows only pelvis yaw changes, not its initial yaw.
        embodiment.xr.anchor_rot = (0.0, 0.0, 0.0, 1.0)
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
        embodiment.camera_config = CoffeeCameraCfg(robot_pov_cam=robot_pov_cam)
        embodiment.observation_config.policy = combine_configclass_instances(
            "GR1TwoArmCoffeePolicyCfg",
            embodiment.observation_config.policy,
            CoffeePolicyCfg(),
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
        coffee_assets = {}
        for name, path in asset_paths.items():
            is_machine = name == "coffee_machine"
            addons = {"make_uninstanceable": True}
            if is_machine:
                # Preserve the imported hinge; only the articulation base is fixed to the world.
                addons["fix_root_link"] = True
                addons["rigid_props"] = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False)
            else:
                addons["func"] = _spawn_coffee_object
                addons["physics_material"] = sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0, dynamic_friction=1.0
                )
            asset = Object(
                name=name,
                prim_path=f"{{ENV_REGEX_NS}}/{name.capitalize()}",
                object_type=ObjectType.ARTICULATION if is_machine else ObjectType.RIGID,
                usd_path=path,
                spawn_cfg_addon=addons,
                initial_pose=Pose(
                    position_xyz=positions[name],
                    rotation_xyzw=COFFEE_MACHINE_ROTATION_XYZW if is_machine else (0.0, 0.0, 0.0, 1.0),
                ),
            )
            if is_machine:
                asset.object_cfg.init_state.joint_pos = {lid_joint.GetPrim().GetName(): lid_open_angle}
                asset.object_cfg.actuators = {
                    "lid": ImplicitActuatorCfg(
                        joint_names_expr=[lid_joint.GetPrim().GetName()], stiffness=0.0, damping=0.005
                    )
                }
            asset.disable_reset_pose()
            coffee_assets[name] = asset
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
        scene = Scene(assets=[table, *coffee_assets.values(), ground, light])
        task = TwoArmCoffeeTask(object=coffee_assets["coffee_pod"], episode_length_s=cfg.episode_length_s)
        task.task_description = (
            "Pick up the coffee_pod with the left hand and bring it near the coffee_machine's upper lid. "
            "The task succeeds when the pod center is within the configured distance of the moving lid's bounds. "
            "The coffee_pod and coffee_machine independently move by up to 5 cm along each XY axis at reset; "
            "the machine base remains fixed during the episode, with a passive hinged lid initially open."
        )
        task.scene_config = CoffeeSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
        coffee_contact_paths = [contact_paths["coffee_pod"], *machine_contact_paths]
        task.scene_config.left_hand_contact.filter_prim_paths_expr = coffee_contact_paths
        task.scene_config.right_hand_contact.filter_prim_paths_expr = list(coffee_contact_paths)
        task.events_cfg = CoffeeEventsCfg()

        teleop_device = None
        if cfg.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(cfg.teleop_device)()
        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=teleop_device,
            env_cfg_callback=_configure_two_arm_coffee,
        )
