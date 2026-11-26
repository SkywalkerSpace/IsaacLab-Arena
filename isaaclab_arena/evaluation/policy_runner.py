# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import os
import tqdm
from contextlib import closing, nullcontext
from functools import partial
from importlib import import_module
from math import exp
from typing import TYPE_CHECKING

from isaaclab_arena.assets.registries import PolicyRegistry
from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.evaluation.policy_runner_cli import (
    DEFAULT_ACTION_SMOOTHING,
    DEFAULT_FINGER_ACTION_SMOOTHING,
    add_policy_cli_args,
    add_policy_runner_arguments,
    build_policy_from_cli,
)
from isaaclab_arena.metrics.metrics_logger import metrics_to_plain_python_types
from isaaclab_arena.utils.hydra_overrides import assert_hydra_overrides
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext
from isaaclab_arena.utils.multiprocess import get_local_rank, get_world_size
from isaaclab_arena.utils.timer import Timer
from isaaclab_arena.video.video_recording import VideoRecordingCfg, timestamped_run_dir, wrap_env_for_video
from isaaclab_arena.visualization.report import build_report, serve_until_ctrl_c
from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser

if TYPE_CHECKING:
    import torch

    from isaaclab_arena.evaluation.action_parquet_recorder import ActionParquetRecorder
    from isaaclab_arena.metrics.metric_data import MetricsDataCollection
    from isaaclab_arena.policy.policy_base import PolicyBase


_ARM_ACTION_DIM = 14

# Sigma in steps (must be positive). Larger values give nearby steps more weight;
# smaller values preserve more of the original action.
_ACTION_SMOOTHING_GAUSSIAN_SIGMA = 1.0

# Execution cap in steps (must be positive). Larger values can reduce replanning frequency
# and raise the blend cap; smaller values replan more often and cap the blend at fewer steps.
_ACTION_SMOOTHING_CHUNK_STEP_LIMIT = 8

# Current action weight (0 to 1). Larger values respond faster;
# smaller values smooth more but add lag.
_ACTION_SMOOTHING_EMA_ALPHA = 0.75


def get_policy_cls(policy_type: str) -> type[PolicyBase]:
    """Get the policy class for the given policy type name.

    Note that this function:
    - first: checks for a registered policy type in the PolicyRegistry
    - if not found, it tries to dynamically import the policy class, treating
      the policy_type argument as a string representing the module path and class name.

    """
    policy_registry = PolicyRegistry()
    if policy_registry.is_registered(policy_type):
        return policy_registry.get_policy(policy_type)
    else:
        print(f"Policy {policy_type} is not registered. Dynamically importing from path: {policy_type}")
        assert "." in policy_type, (
            "policy_type must be a dotted Python import path of the form 'module.submodule.ClassName', got:"
            f" {policy_type}"
        )
        # Dynamically import the class from the string path
        module_path, class_name = policy_type.rsplit(".", 1)
        module = import_module(module_path)
        policy_cls = getattr(module, class_name)
        return policy_cls


def is_distributed(args_cli: argparse.Namespace) -> bool:
    return (
        "cuda" in args_cli.device and hasattr(args_cli, "distributed") and args_cli.distributed and get_world_size() > 1
    )


def _gaussian_smooth_action_chunk(chunk: torch.Tensor, *, action_dim_limit: int | None = None) -> torch.Tensor:
    """Smooth the leading action dimensions along time in a single or batched chunk."""
    from torch import cat
    from torch.nn.functional import pad

    length = chunk.shape[-2]
    if length <= 1:
        return chunk

    # Five-tap Gaussian. Repeat endpoints instead of looking across chunks.
    sigma_squared = _ACTION_SMOOTHING_GAUSSIAN_SIGMA**2
    neighbor_weight, outer_weight = exp(-0.5 / sigma_squared), exp(-2.0 / sigma_squared)
    normalizer = 1.0 + 2.0 * (neighbor_weight + outer_weight)
    prefix = chunk[..., :action_dim_limit]
    padded = pad(prefix.transpose(-1, -2), (2, 2), mode="replicate").transpose(-1, -2)
    smoothed = (
        prefix
        + neighbor_weight * (padded[..., 1 : length + 1, :] + padded[..., 3 : length + 3, :])
        + outer_weight * (padded[..., :length, :] + padded[..., 4:, :])
    ) / normalizer
    if action_dim_limit is not None and action_dim_limit < chunk.shape[-1]:
        return cat((smoothed, chunk[..., action_dim_limit:]), dim=-1)
    return smoothed


def rollout_policy(
    env,
    policy: PolicyBase,
    num_steps: int | None,
    num_episodes: int | None,
    *,
    action_recorder: ActionParquetRecorder | None = None,
    action_smoothing: bool = DEFAULT_ACTION_SMOOTHING,
    finger_action_smoothing: bool = DEFAULT_FINGER_ACTION_SMOOTHING,
) -> MetricsDataCollection | None:
    import torch

    assert num_steps is not None or num_episodes is not None, "Either num_steps or num_episodes must be provided"
    assert num_steps is None or num_episodes is None, "Only one of num_steps or num_episodes must be provided"

    pbar = None
    steps_pbar = None
    smoothing_dim_limit = None if finger_action_smoothing else _ARM_ACTION_DIM
    # Keep direct policy use unchanged outside this rollout.
    has_chunk_transform = hasattr(policy, "action_chunk_transform")
    previous_chunk_transform = getattr(policy, "action_chunk_transform", None)
    has_chunk_step_limit = hasattr(policy, "action_chunk_step_limit")
    previous_chunk_step_limit = getattr(policy, "action_chunk_step_limit", None)
    has_crossfade_dim_limit = hasattr(policy, "action_chunk_crossfade_dim_limit")
    previous_crossfade_dim_limit = getattr(policy, "action_chunk_crossfade_dim_limit", None)
    try:
        if has_chunk_transform:
            policy.action_chunk_transform = (
                partial(_gaussian_smooth_action_chunk, action_dim_limit=smoothing_dim_limit) if action_smoothing else None
            )
        if has_chunk_step_limit:
            # A step cap enables crossfade with the unexecuted tail of the previous chunk.
            policy.action_chunk_step_limit = _ACTION_SMOOTHING_CHUNK_STEP_LIMIT if action_smoothing else None
        if has_crossfade_dim_limit:
            policy.action_chunk_crossfade_dim_limit = smoothing_dim_limit if action_smoothing else None
        if num_episodes is not None:
            env.unwrapped.configure_episode_limit(num_episodes)
        obs, _ = env.reset()
        if action_recorder is not None:
            action_recorder.sync_episodes(env.unwrapped)
        policy.reset()
        policy.set_task_description(env.unwrapped.get_language_instruction())

        # Setup progress bar based on num_steps or num_episodes
        if num_steps is not None:
            pbar = tqdm.tqdm(total=num_steps, desc="Steps", unit="step")
        else:
            pbar = tqdm.tqdm(total=num_episodes, desc="Episodes", unit="episode", position=0)
            steps_pbar = tqdm.tqdm(desc="Steps", unit="step", position=1, mininterval=0, miniters=1)

        num_steps_completed = 0
        previous_prefix = None
        reset_env_ids = None

        while True:
            with torch.inference_mode(), Timer("step"):
                with Timer("policy_inference"):
                    actions = policy.get_action(env, obs)

                if action_smoothing:
                    prefix = actions[..., :smoothing_dim_limit]
                    if previous_prefix is not None:
                        smoothed = torch.lerp(previous_prefix, prefix, _ACTION_SMOOTHING_EMA_ALPHA)
                        if reset_env_ids is not None and reset_env_ids.numel() > 0:
                            smoothed[reset_env_ids] = prefix[reset_env_ids]
                        if smoothing_dim_limit is not None and smoothing_dim_limit < actions.shape[-1]:
                            actions = torch.cat((smoothed, actions[..., smoothing_dim_limit:]), dim=-1)
                        else:
                            actions = smoothed
                        prefix = smoothed
                    # Keep the EMA state independent of any in-place changes by env.step().
                    previous_prefix = prefix.clone()

                if action_recorder is not None:
                    action_recorder.record(actions)
                with Timer("env_step"):
                    obs, _, terminated, truncated, _ = env.step(actions)

                num_steps_completed += 1
                if steps_pbar is not None:
                    steps_pbar.update(1)

                reset_env_ids = env.unwrapped.reset_env_ids
                if reset_env_ids.numel() > 0:
                    tqdm.tqdm.write(f"Resetting policy state for env_ids: {reset_env_ids}")
                    policy.reset(env_ids=reset_env_ids)

                if terminated.any() or truncated.any():
                    if action_recorder is not None:
                        action_recorder.sync_episodes(env.unwrapped)
                    if hasattr(env.unwrapped.cfg, "metrics") and env.unwrapped.cfg.metrics is not None:
                        metrics = env.unwrapped.compute_metrics()
                        tqdm.tqdm.write(
                            f"[Rank {get_local_rank()}/{get_world_size()}] Metrics:"
                            f" {metrics_to_plain_python_types(metrics)}"
                        )
                if num_episodes is not None:
                    completed_episode_count = env.unwrapped.completed_episode_count
                    pbar.update(completed_episode_count - pbar.n)
                    if completed_episode_count == num_episodes:
                        break
                # Break if number of steps is reached
                if num_steps is not None:
                    pbar.update(1)
                    if num_steps_completed >= num_steps:
                        break

        if steps_pbar is not None:
            steps_pbar.close()
        pbar.close()

    except Exception as e:
        if steps_pbar is not None:
            steps_pbar.close()
        if pbar is not None:
            pbar.close()
        raise RuntimeError(f"Error rolling out policy: {e}")

    else:

        # Only compute metrics if env has non-None metrics.
        # Use unwrapped to reach the base env through any gym wrappers (e.g. OrderEnforcing)
        if hasattr(env.unwrapped.cfg, "metrics") and env.unwrapped.cfg.metrics is not None:
            return env.unwrapped.compute_metrics()
        return None

    finally:
        if has_chunk_transform:
            policy.action_chunk_transform = previous_chunk_transform
        if has_chunk_step_limit:
            policy.action_chunk_step_limit = previous_chunk_step_limit
        if has_crossfade_dim_limit:
            policy.action_chunk_crossfade_dim_limit = previous_crossfade_dim_limit


def list_variations(args_parser: argparse.ArgumentParser) -> None:
    """Print the Hydra-configurable variations for the selected environment."""
    args_parser = get_isaaclab_arena_environments_cli_parser(args_parser)
    args_cli, hydra_overrides = args_parser.parse_known_args()
    assert_hydra_overrides(hydra_overrides, args_parser)
    arena_builder = get_arena_builder_from_cli(args_cli, hydra_overrides=hydra_overrides)
    print(arena_builder.get_variations_catalogue_as_string())


def main():
    """Run an IsaacLab Arena environment with a policy.
    Use --distributed with torchrun command for one process per GPU on multi-GPU machines. AppLauncher uses LOCAL_RANK for device.
    """
    args_parser = get_isaaclab_arena_cli_parser()
    # We do this as the parser is shared between the example environment and policy runner
    args_cli, unknown = args_parser.parse_known_args()

    local_rank = get_local_rank()
    world_size = get_world_size()
    # Setting device to local rank before SimulationAppContext
    if is_distributed(args_cli):
        args_cli.device = f"cuda:{local_rank}"
        print(f"[Rank {local_rank}/{world_size}] One Isaac Lab instance per process on cuda:{local_rank}")

    # --record_camera_video requires cameras to be enabled at sim startup, before SimulationAppContext.
    if "--record_camera_video" in unknown:
        args_cli.enable_cameras = True

    with SimulationAppContext(args_cli):

        # Configure RTX before creating the environment so rendering uses settings
        # that do not depend on NGX/DLSS.

        # import carb.settings
        # s = carb.settings.get_settings()
        # s.set("/rtx/dldenoiser/enabled", False)
        # s.set("/rtx/post/dlss/execMode", 0)
        # s.set("/rtx/post/dlssg/enabled", False)
        # s.set("/rtx/post/aa/op", 1)

        # s.set("/rtx/rendermode", "PathTracing")
        # s.set("/rtx/pathtracing/optixDenoiser/enabled", False)
        # s.set("/rtx/pathtracing/spp", 64)
        # s.set("/rtx/pathtracing/totalSpp", 64)
        # s.set("/rtx/pathtracing/maxBounces", 4)

        # Get the policy-type flag before proceeding to other arguments
        add_policy_runner_arguments(args_parser)
        args_cli, _ = args_parser.parse_known_args()

        # --list_variations only inspects the environment, so short-circuit reading other args.
        if args_cli.list_variations:
            list_variations(args_parser)
            return

        # Get the policy class from the policy type
        assert args_cli.policy_type is not None, "--policy_type is required."
        policy_cls = get_policy_cls(args_cli.policy_type)
        print(
            f"[Rank {local_rank}/{world_size}] Requested policy type: {args_cli.policy_type} -> Policy class:"
            f" {policy_cls}"
        )

        # Add the example environment arguments and config-derived policy arguments.
        args_parser = get_isaaclab_arena_environments_cli_parser(args_parser)
        args_parser = add_policy_cli_args(args_parser, policy_cls)
        args_cli, hydra_overrides = args_parser.parse_known_args()
        assert_hydra_overrides(hydra_overrides, args_parser)
        # Re-apply per-rank device after parse preventing device got overwritten by the default value
        if is_distributed(args_cli):
            args_cli.distributed = True
            args_cli.device = f"cuda:{local_rank}"
            # Per-rank seed when distributed so each process has a different seed
            if args_cli.seed is not None:
                args_cli.seed += local_rank

        # Re-apply enable_cameras: the full parse resets it to default False.
        if args_cli.record_camera_video:
            args_cli.enable_cameras = True

        # Build scene. Use rgb_array render mode when recording so RecordVideo can grab frames.
        arena_builder = get_arena_builder_from_cli(args_cli, hydra_overrides=hydra_overrides)

        output_dir = timestamped_run_dir(args_cli.output_base_dir)
        video_cfg = VideoRecordingCfg(
            record_viewport_video=args_cli.record_viewport_video,
            record_camera_video=args_cli.record_camera_video,
            video_base_dir=output_dir,
        )
        env = arena_builder.make_registered(render_mode=video_cfg.render_mode)

        # Write per-episode results to disk.
        results_path = os.path.join(output_dir, f"episode_results_rank{local_rank}.jsonl")
        env.unwrapped.episode_recorder.set_job_name("policy_runner")
        env.unwrapped.episode_recorder.set_output_path(results_path)

        # Create the policy through the typed config compatibility adapter.
        policy = build_policy_from_cli(policy_cls, args_cli)

        # Simulation length.
        if policy.has_length():
            num_steps = policy.length()
            num_episodes = None
        else:
            if args_cli.num_steps is not None:
                num_steps = args_cli.num_steps
                num_episodes = None
                print(f"[Rank {local_rank}/{world_size}] Simulation length: {num_steps} steps")
            elif args_cli.num_episodes is not None:
                num_steps = None
                num_episodes = args_cli.num_episodes
                print(f"[Rank {local_rank}/{world_size}] Simulation length: {num_episodes} episodes")
            else:
                raise ValueError(f"[Rank {local_rank}/{world_size}] Either num_steps or num_episodes must be provided")

        # Optionally wrap with the viewport/camera video recorders (both independent).
        env = wrap_env_for_video(env, video_cfg, num_steps, num_episodes)

        steps_str = f"{num_steps} steps" if num_steps is not None else f"{num_episodes} episodes"
        print(f"[Rank {local_rank}/{world_size}] Starting rollout ({steps_str})")
        try:
            if args_cli.record_actions:
                from isaaclab_arena.evaluation.action_parquet_recorder import ActionParquetRecorder

                rank = int(os.environ.get("RANK", local_rank))
                action_recording = closing(ActionParquetRecorder(output_dir, rank=rank))
                print(f"[Rank {rank}/{world_size}] Recording per-episode actions in {output_dir}")
            else:
                action_recording = nullcontext()

            with action_recording as action_recorder:
                metrics = rollout_policy(
                    env,
                    policy,
                    num_steps,
                    num_episodes,
                    action_recorder=action_recorder,
                    action_smoothing=args_cli.action_smoothing,
                    finger_action_smoothing=args_cli.finger_action_smoothing,
                )

            if metrics is not None:
                print(f"[Rank {local_rank}/{world_size}] Metrics: {metrics_to_plain_python_types(metrics)}")
        finally:
            try:
                # Explicitly clean up the remote client/server rather than relying on a destructor.
                if policy.is_remote:
                    policy.shutdown_remote(kill_server=args_cli.remote_kill_on_exit)
            finally:
                env.close()

        # Write and serve the evaluation report.
        # Only the local rank 0 writes/serves it, to avoid races on a shared output dir.
        if get_local_rank() == 0:
            report_path = build_report(output_dir)
            if args_cli.serve_evaluation_report:
                serve_until_ctrl_c(report_path.parent, args_cli.evaluation_report_port, report_path.name)


if __name__ == "__main__":
    main()
