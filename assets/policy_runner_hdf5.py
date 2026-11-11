# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Evaluate a policy from recorded HDF5 initial states, not recorded actions.

Pass the registered task as a subcommand, as in policy_runner.py. It must match
data.attrs['env_args']['env_name']. Recorded dt, decimation and render_interval
override the task defaults. Assets, joint order,
cameras and physics settings must still match the recording's environment code;
the HDF5 metadata is not a complete environment specification.

Use --inspect_dataset to validate the input without starting Isaac Sim. Evaluation
supports one environment, numerical demo ordering, and optional --episode_names.
--num_episodes selects the first N entries rather than repeating demonstrations.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class RecordedDataset:
    """Validated metadata and initial states loaded without images or actions."""

    path: Path
    env_args: dict[str, Any]
    states: dict[str, dict]

    @property
    def environment_name(self) -> str:
        return self.env_args["env_name"]

    def summary(self) -> dict[str, Any]:
        return {
            "dataset_file": str(self.path),
            "env_args": self.env_args,
            "num_episodes": len(self.states),
            "episode_names": list(self.states),
        }


def load_recorded_dataset(path: str) -> RecordedDataset:
    """Read and validate all initial states in a version-1 (XYZW) HDF5 recording."""
    import h5py
    import numpy as np

    dataset_path = Path(path).expanduser().resolve()
    states = {}
    with h5py.File(dataset_path, "r") as source:
        assert int(source.attrs.get("format_version", 0)) == 1, (
            "Expected format_version=1 (XYZW quaternions); legacy datasets require conversion."
        )
        data = source["data"]
        env_args = json.loads(data.attrs["env_args"])
        assert isinstance(env_args.get("env_name"), str) and env_args["env_name"], "Missing env_name"
        sim_args = env_args.get("sim_args", {})
        for key in ("dt", "decimation", "render_interval"):
            if key not in sim_args:
                continue
            value = sim_args[key]
            assert isinstance(value, (int, float)) and not isinstance(value, bool), f"Invalid {key}: {value}"
            assert math.isfinite(value) and value > 0, f"Invalid {key}: {value}"
            if key != "dt":
                assert int(value) == value, f"{key} must be an integer"

        names = list(data.keys())
        assert names, "The HDF5 contains no demonstrations"
        assert all(name.startswith("demo_") and name[5:].isdigit() for name in names), (
            "Expected episode names of the form demo_N"
        )
        for name in sorted(names, key=lambda item: int(item[5:])):
            assert "initial_state" in data[name], f"{name} has no initial_state"
            state = {}
            initial = data[name]["initial_state"]
            for category, fields in (
                ("articulation", ("root_pose", "root_velocity", "joint_position", "joint_velocity")),
                ("rigid_object", ("root_pose", "root_velocity")),
            ):
                state[category] = {}
                for asset_name, asset in initial.get(category, {}).items():
                    values = {}
                    for field in fields:
                        assert field in asset, f"{name}/{category}/{asset_name} is missing {field}"
                        value = np.asarray(asset[field])
                        assert value.ndim == 2 and value.shape[0] == 1, (
                            f"{name}/{asset_name}/{field}: expected (1, N), got {value.shape}"
                        )
                        assert np.issubdtype(value.dtype, np.floating) and np.isfinite(value).all(), (
                            f"{name}/{asset_name}/{field}: expected finite floating-point values"
                        )
                        if field in ("root_pose", "root_velocity"):
                            width = 7 if field == "root_pose" else 6
                            assert value.shape[1] == width, f"{name}/{asset_name}/{field}: wrong width"
                        if field == "root_pose":
                            assert np.allclose(np.linalg.norm(value[:, 3:], axis=1), 1.0, atol=1e-3), (
                                f"{name}/{asset_name}: quaternion is not normalized"
                            )
                        values[field] = value
                    if category == "articulation":
                        assert values["joint_position"].shape == values["joint_velocity"].shape
                    state[category][asset_name] = values
            assert state["articulation"] or state["rigid_object"], f"{name}: empty initial_state"
            unsupported = set(initial) - {"articulation", "rigid_object"}
            assert not unsupported, f"{name}: unsupported state categories {unsupported}"
            states[name] = state
    return RecordedDataset(dataset_path, env_args, states)


def select_episodes(dataset: RecordedDataset, names: list[str] | None, limit: int | None) -> list[str]:
    """Select unique source demos in explicit order, or numerical order by default."""
    selected = list(dataset.states) if names is None else names
    assert selected and len(set(selected)) == len(selected), "Select at least one demo, without duplicates"
    missing = set(selected) - dataset.states.keys()
    assert not missing, f"Unknown demonstrations: {sorted(missing)}"
    if limit is not None:
        assert 0 < limit <= len(selected), "num_episodes must be positive and no larger than the selection"
        selected = selected[:limit]
    return selected


def add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    """Add options shared by inspection and simulation entry points."""
    parser.add_argument("--dataset_file", required=True, help="Recorded HDF5 with env_args and initial_state")
    parser.add_argument("--inspect_dataset", action="store_true", help="Validate and print metadata without Isaac Sim")
    parser.add_argument("--episode_names", nargs="+", help="Source demos, e.g. demo_0 demo_10; defaults to all demos")


def configure_recorded_resets(env_cfg, dataset: RecordedDataset, episode_names: list[str]) -> None:
    """Restore a source initial state at the end of every reset, including automatic resets."""
    import torch

    from isaaclab.managers import EventTermCfg

    from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTermCfg

    sim_args = dataset.env_args.get("sim_args", {})
    if "dt" in sim_args:
        env_cfg.sim.dt = float(sim_args["dt"])
    if "decimation" in sim_args:
        env_cfg.decimation = int(sim_args["decimation"])
    if "render_interval" in sim_args:
        env_cfg.sim.render_interval = int(sim_args["render_interval"])

    def restore_initial_state(env, env_ids):
        # This is already inside _reset_idx: calling env.reset_to here would recurse.
        # scene.reset_to is the state-writing portion of that API. Lab forwards the
        # simulation and refreshes cameras/observations after reset events finish.
        assert len(env_ids) == 1 and int(env_ids[0]) == 0, "Only num_envs=1 is supported"
        episode_index = env.get_episode_index(0)
        assert 0 <= episode_index < len(episode_names), "Unexpected extra episode reset"
        name = episode_names[episode_index]
        state = dataset.states[name]
        assert set(state["articulation"]) == set(env.scene.articulations), "Recorded articulations do not match scene"
        assert set(state["rigid_object"]) == set(env.scene.rigid_objects), "Recorded rigid objects do not match scene"
        tensor_state = {}
        for category, assets in state.items():
            tensor_state[category] = {}
            for asset_name, values in assets.items():
                if category == "articulation":
                    assert values["joint_position"].shape[1] == env.scene.articulations[asset_name].num_joints, (
                        f"Joint count mismatch for {asset_name}; use the recording's robot model and joint order"
                    )
                tensor_state[category][asset_name] = {
                    key: torch.as_tensor(value, device=env.device) for key, value in values.items()
                }
        env.scene.reset_to(tensor_state, env_ids=env_ids, is_relative=True)
        print(f"[HDF5] Episode {episode_index}: restored {name}")

    def record_source(env, env_id):
        return {
            "source_dataset": str(dataset.path),
            "source_demo": episode_names[env.get_episode_index(env_id)],
            "initial_state_source": "hdf5",
        }

    # Insertion order matters: recorded states must override all ordinary reset events.
    event_name = "restore_hdf5_initial_state"
    assert not hasattr(env_cfg.events, event_name), f"Event {event_name} already exists"
    setattr(env_cfg.events, event_name, EventTermCfg(func=restore_initial_state, mode="reset"))
    env_cfg.episode_recorders.hdf5_source = EpisodeRecorderTermCfg(func=record_source)


def main() -> None:
    """Run inspection, or the standard policy rollout with recorded-state reset events."""
    bootstrap = argparse.ArgumentParser(description=__doc__)
    add_dataset_arguments(bootstrap)
    bootstrap_args, _ = bootstrap.parse_known_args()
    dataset = load_recorded_dataset(bootstrap_args.dataset_file)
    if bootstrap_args.inspect_dataset:
        selected = select_episodes(dataset, bootstrap_args.episode_names, None)
        summary = dataset.summary()
        summary["episode_names"] = selected
        summary["num_episodes"] = len(selected)
        print(json.dumps(summary, indent=2))
        return

    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.evaluation.policy_runner_cli import add_policy_runner_arguments
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

    parser = get_isaaclab_arena_cli_parser()
    add_dataset_arguments(parser)
    add_policy_runner_arguments(parser)
    args, _ = parser.parse_known_args()
    assert args.num_envs == 1 and not args.distributed, "Use one environment without --distributed"
    assert args.num_steps is None, "Use --num_episodes or --episode_names, not --num_steps"
    assert not args.mimic and args.env_spec is None, "Use a registered task subcommand, not --mimic or --env_spec"
    assert args.policy_type is not None, "--policy_type is required"
    if args.record_camera_video:
        args.enable_cameras = True

    with SimulationAppContext(args):

        # Configure RTX before creating the environment so rendering uses settings
        # that do not depend on NGX/DLSS.

        import carb.settings
        s = carb.settings.get_settings()
        s.set("/rtx/dldenoiser/enabled", False)
        s.set("/rtx/post/dlss/execMode", 0)
        s.set("/rtx/post/dlssg/enabled", False)
        s.set("/rtx/post/aa/op", 1)

        s.set("/rtx/rendermode", "PathTracing")
        s.set("/rtx/pathtracing/optixDenoiser/enabled", False)
        s.set("/rtx/pathtracing/spp", 64)
        s.set("/rtx/pathtracing/totalSpp", 64)
        s.set("/rtx/pathtracing/maxBounces", 4)

        from isaaclab_arena.evaluation.policy_runner import get_policy_cls, rollout_policy
        from isaaclab_arena.evaluation.policy_runner_cli import add_policy_cli_args, build_policy_from_cli
        from isaaclab_arena.metrics.metrics_logger import metrics_to_plain_python_types
        from isaaclab_arena.utils.hydra_overrides import assert_hydra_overrides
        from isaaclab_arena.video.video_recording import VideoRecordingCfg, timestamped_run_dir, wrap_env_for_video
        from isaaclab_arena.visualization.report import build_report, serve_until_ctrl_c
        from isaaclab_arena_environments.cli import (
            get_arena_builder_from_cli,
            get_isaaclab_arena_environments_cli_parser,
        )

        policy_cls = get_policy_cls(args.policy_type)
        parser = get_isaaclab_arena_environments_cli_parser(parser)
        add_policy_cli_args(parser, policy_cls)
        args, overrides = parser.parse_known_args()
        assert_hydra_overrides(overrides, parser)
        assert args.example_environment is not None, "Pass the task name before its environment flags, e.g. ${TSAK_NAME}"
        assert args.example_environment == dataset.environment_name, (
            f"Task {args.example_environment!r} does not match HDF5 task {dataset.environment_name!r}"
        )
        args.enable_cameras = args.enable_cameras or args.record_camera_video
        if hasattr(args, "randomize_object_poses"):
            args.randomize_object_poses = False
        selected = select_episodes(dataset, args.episode_names, args.num_episodes)
        print(f"[HDF5] Task: {dataset.environment_name}; evaluating {len(selected)} recorded layouts")
        print("[HDF5] Assets, joint order, cameras and physics must match the recording; metadata is not a full scene.")

        builder = get_arena_builder_from_cli(args, hydra_overrides=overrides)
        env_cfg, env_kwargs = builder.compose_manager_cfg()
        configure_recorded_resets(env_cfg, dataset, selected)
        output_dir = Path(timestamped_run_dir(args.output_base_dir))
        output_dir.mkdir(parents=True, exist_ok=False)
        manifest = dataset.summary()
        manifest.update(
            episode_names=selected,
            num_episodes=len(selected),
            policy_type=args.policy_type,
            policy_config_yaml_path=getattr(args, "policy_config_yaml_path", None),
            language_instruction=args.language_instruction,
            step_dt=env_cfg.sim.dt * env_cfg.decimation,
        )
        (output_dir / "hdf5_evaluation.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        video_cfg = VideoRecordingCfg(
            record_viewport_video=args.record_viewport_video,
            record_camera_video=args.record_camera_video,
            video_base_dir=str(output_dir),
        )
        env = builder.make_registered(env_cfg, env_kwargs, render_mode=video_cfg.render_mode)
        policy = None
        try:
            env.unwrapped.episode_recorder.set_job_name("policy_runner_hdf5")
            env.unwrapped.episode_recorder.set_output_path(output_dir / "episode_results_rank0.jsonl")
            policy = build_policy_from_cli(policy_cls, args)
            assert not policy.has_length(), "Use a checkpoint policy, not a fixed-length replay policy"
            env = wrap_env_for_video(env, video_cfg, num_steps=None, num_episodes=len(selected))
            # Reuse Arena's episode budget and auto-reset handling. The final reset
            # event restores the next demo before the policy receives its observations.
            metrics = rollout_policy(env, policy, num_steps=None, num_episodes=len(selected))
            if metrics is not None:
                print(f"Metrics: {metrics_to_plain_python_types(metrics)}")
        finally:
            try:
                if policy is not None and policy.is_remote:
                    policy.shutdown_remote(kill_server=args.remote_kill_on_exit)
            finally:
                env.close()
        report_path = build_report(str(output_dir))
        print(f"[HDF5] Results: {output_dir}")
        if args.serve_evaluation_report:
            serve_until_ctrl_c(report_path.parent, args.evaluation_report_port, report_path.name)


if __name__ == "__main__":
    main()
