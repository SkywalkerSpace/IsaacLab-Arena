# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Replay demonstrations and save synchronized RGB, observations, actions, and states to HDF5.

Example::

    uv run python scripts/tools/replay_record_demos.py --dataset_file datasets/dataset.hdf5 \
        --output_file datasets/dataset_replay.hdf5 --camera_names table_cam wrist_cam

The task is read from the input dataset unless overridden with ``--task``. Cameras must already
be configured in the task's scene with RGB or RGBA output. All RGB cameras are recorded by default.
Episodes are replayed sequentially in one environment; the input dataset is never modified and an
existing output file is never overwritten. No LeRobot conversion or MP4 encoding is performed.

Output layout (T is the number of source actions)::

    data/demo_N/initial_state/...          Initial scene state, relative to the environment origin
    data/demo_N/actions                    (T, action_dim), original environment commands
    data/demo_N/processed_actions          (T, processed_action_dim), controller targets
    data/demo_N/obs/<policy_term>         Pre-action policy terms, e.g. robot_joint_pos, as in Mimic
    data/demo_N/camera_obs/<obs_term>     (T, H, W, 3), uint8 RGB; e.g. robot_pov_cam_rgb
    data/demo_N/action/<action_term>     Post-action observation terms, if the task defines this group
    data/demo_N/robot_state/joint_pos/... Joint positions [m or rad, depending on joint type]
    data/demo_N/robot_state/joint_vel/... Joint velocities [m/s or rad/s, depending on joint type]
    data/demo_N/camera_info/<obs_term>/... Pre-action world pose (XYZW quaternion) and intrinsics
    data/demo_N/states/...                 Post-action scene states, as in record_demos.py
    data/demo_N/frame_index                (T,), zero-based episode frame indices
    data/demo_N/timestamp                  (T,), simulation time [s], starting at zero

Each row pairs obs[t] with actions[t]; states[t] and processed_actions[t] are captured after that
command. FPS is 1 / env.step_dt, not wall-clock replay speed. Action terms/dimensions, joint names,
camera names, source episode, and task are stored as metadata. Only complete episodes are exported;
an interrupted episode is discarded while already completed episodes remain available.

Schema version 2 follows the default Mimic/Arena recorder layout. Version 1 files using obs/policy
and obs/images must be re-recorded before using the existing Arena LeRobot converter. The converter
currently drops the final row of observations, actions, and images; this script does not invent an
extra action or duplicate a terminal frame. Match the conversion YAML's FPS, camera observation
names, and joint-space configurations to this dataset.
"""

from __future__ import annotations

import argparse
import contextlib
import json
from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np


def _snapshot(value, env_index: int | None = None) -> dict | np.ndarray:
    """Copy tensors, ProxyArrays, or nested dictionaries to host memory before stepping."""
    if isinstance(value, dict):
        return {key: _snapshot(item, env_index) for key, item in value.items()}
    value = getattr(value, "torch", value)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if env_index is not None:
        value = value[env_index]
    return value.copy()


def _flatten(data: dict, prefix: str = "") -> dict[str, np.ndarray]:
    """Flatten nested data into relative HDF5 dataset paths."""
    result = {}
    for key, value in data.items():
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            result.update(_flatten(value, path))
        else:
            result[path] = np.asarray(value)
    return result


class ReplayDatasetWriter:
    """Stream synchronized steps to a new HDF5 file without buffering an episode's images."""

    def __init__(self, output_file: str, env_name: str, step_dt: float, metadata: dict):
        """Create a dataset exclusively; ``step_dt`` is the control step duration [s]."""
        if step_dt <= 0:
            raise ValueError("step_dt must be positive.")
        path = Path(output_file).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(path, "x")
        self._file.attrs["format_version"] = 1  # Isaac Lab XYZW quaternion convention.
        self._file.attrs["replay_record_schema_version"] = 2
        self._data = self._file.create_group("data")
        self._data.attrs["env_args"] = json.dumps({"env_name": env_name, "type": 2})
        self._data.attrs["total"] = 0
        self._data.attrs["fps"] = 1.0 / step_dt
        self._data.attrs["step_dt"] = step_dt
        self._data.attrs["metadata"] = json.dumps(metadata)
        self._step_dt = step_dt
        self._episode = None
        self._step_paths: set[str] | None = None
        self._episode_count = 0

    def begin_episode(self, source_episode: str, initial_state: dict, seed: int | None = None):
        """Start an episode outside ``data`` so incomplete episodes are not advertised."""
        if self._episode is not None:
            raise RuntimeError("Finish the current episode before starting another.")
        self._episode = self._file.create_group("_pending_episode")
        self._episode.attrs["source_episode"] = source_episode
        self._episode.attrs["num_samples"] = 0
        if seed is not None:
            self._episode.attrs["seed"] = seed
        for path, value in _flatten({"initial_state": initial_state}).items():
            self._episode.create_dataset(path, data=value)
        self._step_paths = None

    def append_step(self, data: dict):
        """Append one transition; all temporal datasets must have the same leading dimension."""
        if self._episode is None:
            raise RuntimeError("Start an episode before recording steps.")
        index = int(self._episode.attrs["num_samples"])
        values = _flatten(data)
        values["frame_index"] = np.asarray(index, dtype=np.int64)
        values["timestamp"] = np.asarray(index * self._step_dt, dtype=np.float64)
        if self._step_paths is not None and values.keys() != self._step_paths:
            raise ValueError("The recorded fields changed during the episode.")
        for path, value in values.items():
            if path not in self._episode:
                self._episode.create_dataset(
                    path,
                    shape=(0, *value.shape),
                    maxshape=(None, *value.shape),
                    dtype=value.dtype,
                    chunks=(1, *value.shape),
                    compression="gzip",
                    compression_opts=2,
                )
            dataset = self._episode[path]
            if dataset.shape[1:] != value.shape or dataset.dtype != value.dtype:
                raise ValueError(f"Shape or dtype changed for {path} during the episode.")
            dataset.resize(index + 1, axis=0)
            dataset[index] = value
        self._step_paths = set(values)
        self._episode.attrs["num_samples"] = index + 1

    def finish_episode(self, success: bool | None = None, success_source: str = "unknown"):
        """Publish a completed episode and flush it to disk."""
        if self._episode is None:
            raise RuntimeError("No episode is being recorded.")
        num_samples = int(self._episode.attrs["num_samples"])
        if num_samples == 0:
            raise ValueError("Cannot export an episode without actions.")
        if success is not None:
            self._episode.attrs["success"] = bool(success)
        self._episode.attrs["success_source"] = success_source
        self._file.move("_pending_episode", f"data/demo_{self._episode_count}")
        self._data.attrs["total"] += num_samples
        self._episode_count += 1
        self._episode = None
        self._file.flush()

    def close(self):
        """Discard any incomplete episode and close the file, preserving completed episodes."""
        if self._file:
            if "_pending_episode" in self._file:
                del self._file["_pending_episode"]
            self._file.close()


def _camera_observation_names(env, cameras: dict) -> dict[str, str]:
    """Map camera observation term names to selected scene sensors, preserving task-specific names."""
    observations_cfg = getattr(getattr(env, "cfg", None), "observations", None)
    camera_cfg = getattr(observations_cfg, "camera_obs", None)
    names = {}
    if camera_cfg is not None:
        for name, term in vars(camera_cfg).items():
            params = getattr(term, "params", {}) or {}
            sensor_name = getattr(params.get("sensor_cfg"), "name", None)
            if sensor_name in cameras and params.get("data_type") in {"rgb", "rgba"}:
                names[name] = sensor_name
    for sensor_name in cameras:
        if sensor_name not in names.values():
            names[f"{sensor_name}_rgb"] = sensor_name
    return names


def _record_episode(env, episode, writer: ReplayDatasetWriter, cameras: dict, is_running: Callable[[], bool]) -> bool:
    """Stream the same obs, camera_obs, action, and processed_actions fields as Mimic/Arena recorders."""
    camera_names = None
    while (action := episode.get_next_action()) is not None:
        if not is_running():
            return False
        if camera_names is None:
            camera_names = _camera_observation_names(env, cameras)
        step_data = {
            # PreStepFlatPolicyObservationsRecorder writes policy terms directly under obs.
            "obs": _snapshot(env.obs_buf["policy"], 0),
            "robot_state": {
                "joint_pos": {name: _snapshot(robot.data.joint_pos, 0) for name, robot in env.scene.articulations.items()},
                "joint_vel": {name: _snapshot(robot.data.joint_vel, 0) for name, robot in env.scene.articulations.items()},
            },
            "camera_obs": {},
            "camera_info": {},
            "actions": _snapshot(action),
        }
        camera_observations = env.obs_buf.get("camera_obs", {})
        for name, sensor_name in camera_names.items():
            camera = cameras[sensor_name]
            camera.update(0.0, force_recompute=True)
            camera_data = camera.data
            if name in camera_observations:
                # Use the task observation, including any observation-side image processing.
                image = _snapshot(camera_observations[name], 0)[..., :3]
            else:
                output = camera_data.output
                image = _snapshot(output["rgb"] if "rgb" in output else output["rgba"], 0)[..., :3]
            if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                raise ValueError(f"Camera observation {name} must provide uint8 RGB/RGBA images in HWC layout.")
            step_data["camera_obs"][name] = image
            step_data["camera_info"][name] = {
                "pos_w": _snapshot(camera_data.pos_w, 0),
                "quat_w_world": _snapshot(camera_data.quat_w_world, 0),
                "intrinsic_matrix": _snapshot(camera_data.intrinsic_matrices, 0),
            }
        env.step(action[None])
        step_data["processed_actions"] = np.concatenate(
            [
                _snapshot(env.action_manager.get_term(name).processed_actions, 0)
                for name in env.action_manager.active_terms
            ],
            axis=-1,
        )
        step_data["states"] = _snapshot(env.scene.get_state(is_relative=True), 0)
        # Match Arena's PostStepFlatPolicyActionObservationRecorder, including eef/teleop commands.
        if "action" in env.obs_buf:
            step_data["action"] = _snapshot(env.obs_buf["action"], 0)
        writer.append_step(step_data)
    return True


def main():  # noqa: C901
    """Launch Isaac Sim, replay selected episodes, and save a synchronized HDF5 dataset."""
    # Simulator-dependent imports must follow AppLauncher startup.
    import warp as wp

    wp.config.enable_backward = False

    from isaaclab.app import AppLauncher
    from isaaclab.utils.string import list_intersection, string_to_callable

    from isaaclab_tasks.utils import setup_preset_cli

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_file", default="datasets/dataset.hdf5", help="Input demonstration HDF5 file.")
    parser.add_argument("--output_file", default=None, help="Output HDF5 file (default: <input_stem>_replay.hdf5).")
    parser.add_argument("--task", default=None, help="Override the task stored in the input dataset.")
    parser.add_argument("--select_episodes", type=int, nargs="+", default=[], help="Episode indices; default: all.")
    parser.add_argument("--camera_names", nargs="+", default=None, help="Scene camera names; default: all RGB cameras.")
    parser.add_argument(
        "--num_envs", type=int, default=1, choices=[1], help="Sequential recording uses one environment."
    )
    parser.add_argument("--validate_success_rate", action="store_true", help="Evaluate task success after replay.")
    parser.add_argument(
        "--reset_sim_buffer_each_episode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reset simulation buffers before each episode, as in record_demos.py.",
    )
    parser.add_argument("--external_callback", default=None, help="Fully qualified environment registration callback.")
    AppLauncher.add_app_launcher_args(parser)
    args_cli, hydra_args = setup_preset_cli(parser)
    input_path = Path(args_cli.dataset_file).expanduser().resolve()
    output_path = (
        Path(args_cli.output_file).expanduser().resolve()
        if args_cli.output_file
        else input_path.with_name(f"{input_path.stem}_replay.hdf5")
    )
    if not input_path.is_file():
        parser.error(f"Input dataset does not exist: {input_path}")
    if output_path == input_path or output_path.exists():
        parser.error(f"Output must be a new file, different from the input: {output_path}")

    app_launcher = AppLauncher(args_cli, enable_cameras=True)
    simulation_app = app_launcher.app
    try:
        import sys

        import gymnasium as gym
        import torch

        from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg
        from isaaclab.sensors import Camera
        from isaaclab.utils.datasets import HDF5DatasetFileHandler

        import isaaclab_tasks  # noqa: F401
        from isaaclab_tasks.utils import resolve_task_config

        remaining_args = None
        if args_cli.external_callback:
            remaining_args = string_to_callable(args_cli.external_callback, separator=".")()
        sys.argv = [sys.argv[0]] + list_intersection(hydra_args, remaining_args)

        with contextlib.ExitStack() as stack:
            source = HDF5DatasetFileHandler()
            stack.callback(source.close)
            source.open(str(input_path))
            task = args_cli.task or source.get_env_name()
            if not task:
                raise ValueError("Specify --task or store an environment name in the input dataset.")
            episode_names = list(source.get_episode_names())
            indices = args_cli.select_episodes or list(range(len(episode_names)))
            if not indices:
                raise ValueError("No episodes found in the input dataset.")
            if any(index < 0 or index >= len(episode_names) for index in indices):
                raise ValueError(f"Episode indices must be between 0 and {len(episode_names) - 1}.")

            env_cfg, _ = resolve_task_config(task, "")
            env_cfg.sim.device = args_cli.device
            env_cfg.scene.num_envs = 1
            env_cfg.sim.render_interval = env_cfg.decimation
            env_cfg.num_rerenders_on_reset = max(1, env_cfg.num_rerenders_on_reset)
            env_cfg.observations.policy.concatenate_terms = False
            success_term = getattr(env_cfg.terminations, "success", None) if args_cli.validate_success_rate else None
            if args_cli.validate_success_rate and success_term is None:
                raise ValueError("The task has no success termination term to validate.")
            env_cfg.recorders = {}
            env_cfg.terminations = {}
            env = gym.make(task, cfg=env_cfg).unwrapped
            stack.callback(env.close)

            available_cameras = {
                name: sensor
                for name, sensor in env.scene.sensors.items()
                if isinstance(sensor, Camera) and {"rgb", "rgba"}.intersection(sensor.cfg.data_types)
            }
            camera_names = args_cli.camera_names or list(available_cameras)
            if not camera_names or any(name not in available_cameras for name in camera_names):
                raise ValueError(
                    f"Select at least one configured RGB/RGBA camera. Available cameras: {list(available_cameras)}"
                )
            cameras = {name: available_cameras[name] for name in camera_names}
            for camera in cameras.values():
                camera.cfg.update_period = 0.0
                camera.cfg.update_latest_camera_pose = True
            metadata = {
                "source_file": str(input_path),
                "observation_alignment": "pre_action",
                "state_alignment": "post_action",
                "image_layout": "HWC",
                "image_color_space": "RGB",
                "quaternion_order": "xyzw",
                "camera_pose_convention": "world (+X forward, +Z up)",
                "action_terms": list(env.action_manager.active_terms),
                "action_term_dims": list(env.action_manager.action_term_dim),
                "joint_names": {name: list(robot.joint_names) for name, robot in env.scene.articulations.items()},
                "camera_names": camera_names,
                "camera_observation_names": _camera_observation_names(env, cameras),
            }
            writer = ReplayDatasetWriter(str(output_path), task.split(":")[-1], env.step_dt, metadata)
            stack.callback(writer.close)
            paused = False

            def set_paused(value: bool):
                nonlocal paused
                paused = value

            keyboard = None
            if app_launcher.has_window:
                keyboard = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.1, rot_sensitivity=0.1))
                keyboard.add_callback("B", lambda: set_paused(True))
                keyboard.add_callback("N", lambda: set_paused(False))
                print('Press "B" to pause and "N" to resume replay recording.')

            def is_running() -> bool:
                while paused and simulation_app.is_running() and not simulation_app.is_exiting():
                    env.sim.render()
                return simulation_app.is_running() and not simulation_app.is_exiting() and not env.sim.is_stopped()

            completed = 0
            successes = 0
            with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
                env.reset()
                for index in indices:
                    if not is_running():
                        break
                    name = episode_names[index]
                    episode = source.load_episode(name, env.device)
                    if episode.get_initial_state() is None or episode.get_action(0) is None:
                        raise ValueError(f"Episode {name} must contain initial_state and at least one action.")
                    if args_cli.reset_sim_buffer_each_episode:
                        env.sim.reset()
                    seed = int(episode.seed) if episode.seed is not None else None
                    env.reset_to(episode.get_initial_state(), env_ids=None, seed=seed, is_relative=True)
                    writer.begin_episode(name, _snapshot(env.scene.get_state(is_relative=True), 0), seed)
                    print(f"Recording episode #{index}: {name}")
                    if not _record_episode(env, episode, writer, cameras, is_running):
                        break
                    success = (
                        bool(success_term.func(env, **success_term.params)[0]) if success_term else episode.success
                    )
                    writer.finish_episode(success, "replay" if success_term else "source_dataset")
                    completed += 1
                    successes += int(bool(success))
            print(f"Saved {completed} complete episodes to: {output_path}")
            if success_term:
                print(f"Successfully replayed: {successes}/{completed}")
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
