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


_ASSET_DIR = Path(__file__).resolve().parents[1] / "assets" / "mjcf2usd"
POSITION_XY_RANDOM_MAX = 0.05
TABLE_SURFACE_HEIGHT = 1.0
SPAWN_CLEARANCE = 0.002
CUP_POSITION_XY = (-0.15, 0.40)
BOWL_POSITION_XY = (0.10, 0.30)
PAD_POSITION_XY = (0.30, 0.30)
CUP_GRASP_MAX_DISTANCE = 0.10
CUP_GRASP_MIN_LIFT = 0.03


@dataclass
class GR1TwoArmPouringEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the GR1T2 two-arm pouring environment."""

    embodiment: str = "gr1_pink"
    teleop_device: str | None = None
    episode_length_s: float = 20.0
    randomization_seed: int | None = None
    """Layout seed; None samples a fresh seed on each launch, independently of Arena's seed."""
    pouring_ball_path: str = str(_ASSET_DIR / "pouring_ball.usd")
    pouring_bowl_path: str = str(_ASSET_DIR / "pouring_bowl.usd")
    pouring_cup_path: str = str(_ASSET_DIR / "pouring_cup.usd")
    pouring_pad_path: str = str(_ASSET_DIR / "pouring_pad.usd")
    ball_in_bowl_max_xy_distance: float = 0.035
    """Conservative ball-center radius around the bowl's local center, in meters."""
    success_max_xy_distance: float = 0.04
    """Maximum bowl-to-pad center distance in the pad's local XY plane, in meters."""
    success_max_height_difference: float = 0.015
    """Tolerance between the bowl bottom and pad top, in meters."""
    success_max_speed: float = 0.20
    """Componentwise ball and bowl velocity limits in m/s (linear) and rad/s (angular)."""

    def __post_init__(self) -> None:
        assert math.isfinite(self.episode_length_s) and self.episode_length_s > 0, "episode_length_s must be positive"
        assert self.randomization_seed is None or (
            isinstance(self.randomization_seed, int) and 0 <= self.randomization_seed < 2**63
        ), "randomization_seed must be None or an integer in [0, 2**63)"
        if self.embodiment not in ("gr1_pink", "gr1_joint"):
            raise ValueError(
                f"Two-arm pouring requires a GR1T2 embodiment ('gr1_pink' or 'gr1_joint'); "
                f"got {self.embodiment!r}."
            )
        for name in (
            "ball_in_bowl_max_xy_distance",
            "success_max_xy_distance",
            "success_max_height_difference",
            "success_max_speed",
        ):
            value = getattr(self, name)
            assert math.isfinite(value) and value > 0, f"{name} must be finite and positive"


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


def _pouring_collision_body(root):
    """Find the sole rigid body owning collision geometry, ignoring empty imported links."""
    from pxr import Usd, UsdPhysics

    bodies = {}
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        owner = prim
        while owner and owner.GetPath().HasPrefix(root.GetPath()):
            if owner.HasAPI(UsdPhysics.RigidBodyAPI):
                bodies[str(owner.GetPath())] = owner
                break
            owner = owner.GetParent()
    assert len(bodies) == 1, (
        f"Pouring assets require one collision-bearing rigid body under {root.GetPath()}; found {list(bodies)}"
    )
    return next(iter(bodies.values()))


def _normalize_pouring_object(root):
    """Remove empty URDF links and fixed joints from a single-body pouring prop."""
    from pxr import Usd, UsdPhysics

    body = _pouring_collision_body(root)
    prims = list(Usd.PrimRange(root))
    joints = [prim for prim in prims if prim.IsA(UsdPhysics.Joint)]
    assert all(prim.IsA(UsdPhysics.FixedJoint) for prim in joints), (
        f"Pouring props must not contain movable joints: {root.GetPath()}"
    )
    for prim in joints:
        prim.SetActive(False)
    for prim in prims:
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        if prim != body and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
            prim.RemoveAPI(UsdPhysics.MassAPI)
    return body


def _spawn_pouring_object(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Spawn converted MJCF props as single rigid bodies without modifying the source USD."""
    import isaaclab.sim as sim_utils

    prim = sim_utils.spawn_from_usd(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    stage = sim_utils.get_current_stage()
    for path in sim_utils.find_matching_prim_paths(prim_path):
        _normalize_pouring_object(stage.GetPrimAtPath(path))
    return prim


def _randomize_pouring_layout(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    positions: dict[str, tuple[float, float, float]],
    randomization_seed: int | None = None,
) -> None:
    """Jitter cup, bowl, and pad independently by ±5 cm in XY, keeping the ball inside the cup."""
    import secrets
    import torch

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device).reshape(-1)
    if len(env_ids) == 0:
        return
    if not hasattr(env, "_pouring_rng"):
        seed = secrets.randbits(63) if randomization_seed is None else randomization_seed
        env._pouring_rng = torch.Generator(device=env.device).manual_seed(seed)
        env._pouring_randomization_seed = seed
        print(f"[INFO] Two-arm pouring layout randomization seed: {seed}")
    # Do not reseed on reset: a fixed seed reproduces a sequence, not a single layout.
    offsets = (
        torch.rand((len(env_ids), 3, 2), device=env.device, generator=env._pouring_rng) * 2.0 - 1.0
    ) * POSITION_XY_RANDOM_MAX
    for name, offset_index in (("cup", 0), ("ball", 0), ("bowl", 1), ("pad", 2)):
        pose = torch.zeros((len(env_ids), 7), device=env.device)
        pose[:, :3] = torch.tensor(positions[name], device=env.device) + env.scene.env_origins[env_ids]
        pose[:, :2] += offsets[:, offset_index]
        pose[:, 6] = 1.0  # Isaac Lab 3.0 uses XYZW quaternions.
        asset = env.scene[name]
        asset.write_root_pose_to_sim(pose, env_ids=env_ids)
        asset.write_root_velocity_to_sim(torch.zeros((len(env_ids), 6), device=env.device), env_ids=env_ids)


def _pouring_object_dropped(env: ManagerBasedRLEnv, object_name: str, minimum_height: float = 0.5) -> torch.Tensor:
    """Detect a pouring object falling below an environment-local height [m]."""
    object_height = env.scene[object_name].data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    return object_height < minimum_height


def _cup_near_left_palm(env: ManagerBasedRLEnv, min_height: float) -> torch.Tensor:
    """Use cup lift and left-palm proximity as a grasp proxy."""
    import torch

    robot = env.scene["robot"]
    palm_index = robot.data.body_names.index("left_hand_pitch_link")
    cup_pos = env.scene["cup"].data.root_pos_w.torch
    palm_pos = robot.data.body_pos_w.torch[:, palm_index]
    lifted = cup_pos[:, 2] - env.scene.env_origins[:, 2] > min_height
    return lifted & (torch.linalg.vector_norm(cup_pos - palm_pos, dim=1) < CUP_GRASP_MAX_DISTANCE)


def _ball_in_bowl(
    env: ManagerBasedRLEnv,
    max_xy_distance: float,
    min_height: float,
    max_height: float,
    bowl_center_xy: tuple[float, float],
) -> torch.Tensor:
    """Approximate containment using a conservative cylinder in the upright bowl's local frame."""
    import torch

    from isaaclab.managers import SceneEntityCfg
    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    from isaaclab_arena.tasks.predicates.spatial import object_in_contact_with_target

    touching_bowl = object_in_contact_with_target(env, SceneEntityCfg("ball_bowl_contact"), force_threshold=0.001)
    ball = env.scene["ball"]
    bowl = env.scene["bowl"]
    q_W_B = bowl.data.root_quat_w.torch
    p_B_ball = quat_apply_inverse(q_W_B, ball.data.root_pos_w.torch - bowl.data.root_pos_w.torch)
    center_xy = p_B_ball.new_tensor(bowl_center_xy)
    up_B = p_B_ball.new_tensor((0.0, 0.0, 1.0)).expand_as(p_B_ball)
    upright = quat_apply(q_W_B, up_B)[:, 2] > 0.95
    return (
        upright
        & touching_bowl
        & (torch.linalg.vector_norm(p_B_ball[:, :2] - center_xy, dim=1) < max_xy_distance)
        & (p_B_ball[:, 2] > min_height)
        & (p_B_ball[:, 2] < max_height)
    )


def _pouring_complete(
    env: ManagerBasedRLEnv,
    ball_in_bowl_params: dict,
    bowl_bottom_offset: tuple[float, float, float],
    pad_top_offset: tuple[float, float, float],
    max_xy_distance: float,
    max_height_difference: float,
    max_speed: float,
) -> torch.Tensor:
    """Require the ball inside the bowl and the upright, slow bowl resting near the pad top."""
    import torch

    from isaaclab.managers import SceneEntityCfg
    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    from isaaclab_arena.tasks.predicates.spatial import object_in_contact_with_target

    touching_pad = object_in_contact_with_target(env, SceneEntityCfg("bowl_pad_contact"), force_threshold=0.001)
    bowl = env.scene["bowl"]
    pad = env.scene["pad"]
    p_W_bowl = bowl.data.root_pos_w.torch
    p_W_pad = pad.data.root_pos_w.torch
    bottom_B = p_W_bowl.new_tensor(bowl_bottom_offset).expand_as(p_W_bowl)
    top_P = p_W_pad.new_tensor(pad_top_offset).expand_as(p_W_pad)
    p_W_bottom = p_W_bowl + quat_apply(bowl.data.root_quat_w.torch, bottom_B)
    p_W_top = p_W_pad + quat_apply(pad.data.root_quat_w.torch, top_P)
    # W is world; B/P are the bowl/pad root frames.
    difference_P = quat_apply_inverse(pad.data.root_quat_w.torch, p_W_bottom - p_W_top)
    bowl_on_pad = (
        (torch.linalg.vector_norm(difference_P[:, :2], dim=1) < max_xy_distance)
        & (torch.abs(difference_P[:, 2]) < max_height_difference)
        & torch.all(torch.abs(bowl.data.root_vel_w.torch) < max_speed, dim=1)
    )
    ball_still = torch.all(torch.abs(env.scene["ball"].data.root_vel_w.torch) < max_speed, dim=1)
    return _ball_in_bowl(env, **ball_in_bowl_params) & bowl_on_pad & touching_pad & ball_still


def _configure_two_arm_pouring(
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
class GR1TwoArmPouringEnvironment(ArenaEnvironmentFactory[GR1TwoArmPouringEnvironmentCfg]):
    """Grasp the cup with the left hand, pour the ball into the bowl, and place the bowl on the pad."""

    name = "gr1_two_arm_pouring"
    _legacy_argparse_cfg_type = GR1TwoArmPouringEnvironmentCfg

    @staticmethod
    def _add_legacy_cli_only_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--object",
            type=str,
            default=None,
            help=(
                "Accepted for CLI compatibility only; use --pouring_cup_path and --pouring_bowl_path to select assets."
            ),
        )

    def build(self, cfg: GR1TwoArmPouringEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the pouring task from stock GR1 configs and the four pouring USD assets."""
        import isaaclab.envs.mdp as mdp
        import isaaclab.sim as sim_utils
        from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
        from isaaclab.managers import (
            EventTermCfg,
            ObservationGroupCfg,
            ObservationTermCfg,
            SceneEntityCfg,
            TerminationTermCfg,
        )
        from isaaclab.scene import InteractiveSceneCfg
        from isaaclab.sensors import CameraCfg, ContactSensorCfg
        from isaaclab.utils.configclass import configclass
        from isaaclab_tasks.contrib.pick_place.pickplace_gr1t2_env_cfg import PickPlaceGR1T2SceneCfg
        from pxr import Usd

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
            "ball": cfg.pouring_ball_path,
            "cup": cfg.pouring_cup_path,
            "bowl": cfg.pouring_bowl_path,
            "pad": cfg.pouring_pad_path,
        }
        bounds = {}
        contact_paths = {}
        for name, path in asset_paths.items():
            stage = Usd.Stage.Open(path)
            root = stage.GetDefaultPrim()
            body = _pouring_collision_body(root)
            body_suffix = str(body.GetPath())[len(str(root.GetPath())) :]
            contact_paths[name] = f"{{ENV_REGEX_NS}}/{name.capitalize()}{body_suffix}"
            box = compute_local_bounding_box_from_usd(path, (1.0, 1.0, 1.0))
            bounds[name] = (box.min_point[0].tolist(), box.max_point[0].tolist())
        # Read actual USD extents rather than assuming the roots lie on the table surface.
        positions = {}
        for name, xy in (("cup", CUP_POSITION_XY), ("bowl", BOWL_POSITION_XY), ("pad", PAD_POSITION_XY)):
            positions[name] = (*xy, TABLE_SURFACE_HEIGHT + SPAWN_CLEARANCE - bounds[name][0][2])
        cup_min, cup_max = bounds["cup"]
        ball_min, ball_max = bounds["ball"]
        ball_radius = max(high - low for low, high in zip(ball_min, ball_max)) / 2.0
        cup_center = tuple((low + high) / 2.0 for low, high in zip(cup_min, cup_max))
        assert cup_max[2] - cup_min[2] > 2.0 * ball_radius, "The ball must fit below the cup rim"
        positions["ball"] = tuple(value + offset for value, offset in zip(positions["cup"], cup_center))
        bowl_min, bowl_max = bounds["bowl"]
        pad_min, pad_max = bounds["pad"]
        bowl_center_xy = tuple((bowl_min[index] + bowl_max[index]) / 2.0 for index in range(2))
        ball_in_bowl_params = {
            "max_xy_distance": cfg.ball_in_bowl_max_xy_distance,
            "min_height": bowl_min[2] + ball_radius - SPAWN_CLEARANCE,
            "max_height": bowl_max[2] + SPAWN_CLEARANCE,
            "bowl_center_xy": bowl_center_xy,
        }

        @configclass
        class PouringSceneCfg(InteractiveSceneCfg):
            ball_bowl_contact: ContactSensorCfg | None = None
            bowl_pad_contact: ContactSensorCfg | None = None
            left_hand_contact = source_scene.left_hand_contact.replace(
                filter_prim_paths_expr=["{ENV_REGEX_NS}/Cup", "{ENV_REGEX_NS}/Bowl"]
            )
            right_hand_contact = source_scene.right_hand_contact.replace(
                filter_prim_paths_expr=["{ENV_REGEX_NS}/Cup", "{ENV_REGEX_NS}/Bowl"]
            )

        @configclass
        class PouringEventsCfg:
            reset_all = EventTermCfg(func=mdp.reset_scene_to_default, mode="reset")
            randomize_pouring_layout = EventTermCfg(
                func=_randomize_pouring_layout,
                mode="reset",
                params={"positions": positions, "randomization_seed": cfg.randomization_seed},
            )

        @configclass
        class PouringPolicyCfg(ObservationGroupCfg):
            ball_pos = ObservationTermCfg(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("ball")})
            ball_rot = ObservationTermCfg(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("ball")})
            cup_pos = ObservationTermCfg(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("cup")})
            cup_rot = ObservationTermCfg(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("cup")})
            bowl_pos = ObservationTermCfg(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("bowl")})
            bowl_rot = ObservationTermCfg(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("bowl")})
            pad_pos = ObservationTermCfg(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("pad")})
            pad_rot = ObservationTermCfg(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("pad")})
            concatenate_terms = False
            enable_corruption = False

        @configclass
        class PouringSubtaskTermsCfg(ObservationGroupCfg):
            cup_grasped = ObservationTermCfg(
                func=_cup_near_left_palm, params={"min_height": positions["cup"][2] + CUP_GRASP_MIN_LIFT}
            )
            ball_in_bowl = ObservationTermCfg(func=_ball_in_bowl, params=ball_in_bowl_params)
            concatenate_terms = False
            enable_corruption = False

        @configclass
        class PouringObservationsCfg:
            subtask_terms: PouringSubtaskTermsCfg = PouringSubtaskTermsCfg()

        class PouringMimicEnv(GR1T2MimicEnv):
            def get_subtask_term_signals(self, env_ids=None):
                if env_ids is None:
                    env_ids = slice(None)
                terms = self.obs_buf["subtask_terms"]
                return {name: terms[name][env_ids] for name in ("cup_grasped", "ball_in_bowl")}

        @configclass
        class PouringCameraCfg(GR1T2CameraCfg):
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

        class PouringCompletionCriteria(CompletionCriteria):
            """Keep derived predicate tuples out of the serialized environment config."""

            def __post_init__(self):
                self.logical = CompletionCriteria(**vars(self)).logical

            @property
            def canonical_predicate_sequences(self):
                return CompletionCriteria(**vars(self)).canonical_predicate_sequences

        class TwoArmPouringTask(GoalPoseTask):
            def get_metrics(self):
                from isaaclab_arena.metrics.success_rate import SuccessRateMetric

                return [SuccessRateMetric()]

            def get_observation_cfg(self):
                return PouringObservationsCfg()

            def get_termination_cfg(self) -> TaskTerminationCfg:
                return TaskTerminationCfg(
                    timeout_s=self.episode_length_s,
                    success=[
                        PouringCompletionCriteria(
                            name="ball_in_bowl_on_pad",
                            predicate_sequence=[
                                TerminationTermCfg(
                                    func=_pouring_complete,
                                    params={
                                        "ball_in_bowl_params": ball_in_bowl_params,
                                        "bowl_bottom_offset": (*bowl_center_xy, bowl_min[2]),
                                        "pad_top_offset": (
                                            (pad_min[0] + pad_max[0]) / 2.0,
                                            (pad_min[1] + pad_max[1]) / 2.0,
                                            pad_max[2],
                                        ),
                                        "max_xy_distance": cfg.success_max_xy_distance,
                                        "max_height_difference": cfg.success_max_height_difference,
                                        "max_speed": cfg.success_max_speed,
                                    },
                                )
                            ],
                        )
                    ],
                    failures={
                        f"{name}_dropping": TerminationTermCfg(
                            func=_pouring_object_dropped, params={"object_name": name, "minimum_height": 0.5}
                        )
                        for name in ("ball", "cup", "bowl")
                    },
                )

            def get_mimic_env_cfg(self, arm_mode: ArmMode):
                assert arm_mode == ArmMode.DUAL_ARM, "Two-arm pouring requires dual-arm Mimic demonstrations"
                mimic_cfg = MimicEnvCfg()
                mimic_cfg.datagen_config.name = "gr1_two_arm_pouring_D0"
                for key, value in MIMIC_DATAGEN_CONFIG_DEFAULTS.items():
                    setattr(mimic_cfg.datagen_config, key, value)
                # The left hand pours the cup; the right hand carries the receiving bowl to the pad.
                object_refs = {"left": ("cup", "bowl", "cup"), "right": ("bowl", "bowl", "pad")}
                term_signals = ("cup_grasped", "ball_in_bowl", None)
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
        embodiment.mimic_env = PouringMimicEnv
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
        embodiment.camera_config = PouringCameraCfg(robot_pov_cam=robot_pov_cam)
        embodiment.observation_config.policy = combine_configclass_instances(
            "GR1TwoArmPouringPolicyCfg",
            embodiment.observation_config.policy,
            PouringPolicyCfg(),
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
        pouring_assets = {}
        for name, path in asset_paths.items():
            addons = {"func": _spawn_pouring_object, "make_uninstanceable": True}
            if name == "pad":
                addons["rigid_props"] = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
            asset = Object(
                name=name,
                prim_path=f"{{ENV_REGEX_NS}}/{name.capitalize()}",
                object_type=ObjectType.RIGID,
                usd_path=path,
                spawn_cfg_addon=addons,
                initial_pose=Pose(position_xyz=positions[name]),
            )
            asset.disable_reset_pose()
            pouring_assets[name] = asset
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
        scene = Scene(assets=[table, *pouring_assets.values(), ground, light])
        task = TwoArmPouringTask(object=pouring_assets["bowl"], episode_length_s=cfg.episode_length_s)
        task.task_description = (
            "Grasp the cup containing the ball with the left hand. Pour the ball into the bowl, "
            "then place the bowl on the pad with the ball still inside. "
            "The cup, bowl, and pad independently move by up to 5 cm along each XY axis at reset; "
            "the ball moves with the cup so that it starts inside."
        )
        task.scene_config = PouringSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
        task.scene_config.ball_bowl_contact = ContactSensorCfg(
            prim_path=contact_paths["ball"], filter_prim_paths_expr=[contact_paths["bowl"]]
        )
        task.scene_config.bowl_pad_contact = ContactSensorCfg(
            prim_path=contact_paths["bowl"], filter_prim_paths_expr=[contact_paths["pad"]]
        )
        pouring_contact_paths = [contact_paths[name] for name in ("cup", "bowl")]
        task.scene_config.left_hand_contact.filter_prim_paths_expr = pouring_contact_paths
        task.scene_config.right_hand_contact.filter_prim_paths_expr = list(pouring_contact_paths)
        task.events_cfg = PouringEventsCfg()

        teleop_device = None
        if cfg.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(cfg.teleop_device)()
        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=teleop_device,
            env_cfg_callback=_configure_two_arm_pouring,
        )
