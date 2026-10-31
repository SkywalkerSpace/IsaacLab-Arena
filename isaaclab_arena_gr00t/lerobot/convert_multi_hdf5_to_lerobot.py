# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import h5py
import json
import multiprocessing as mp
import numpy as np
import shutil
import subprocess
import time
import traceback
from copy import copy
from dataclasses import fields
from pathlib import Path
from queue import Empty
from tqdm import tqdm
from typing import Any

import imageio.v2 as imageio
import pandas as pd

from isaaclab_arena_gr00t.lerobot.config.dataset_config import Gr00tDatasetConfig
from isaaclab_arena_gr00t.utils.image_conversion import resize_frames_with_padding
from isaaclab_arena_gr00t.utils.io_utils import (
    create_config_from_yaml,
    dump_json,
    dump_jsonl,
    load_config_from_yaml,
    load_dict_from_yaml,
    load_json,
    load_robot_joints_config_from_yaml,
)
from isaaclab_arena_gr00t.utils.joints_conversion import remap_sim_joints_to_policy_joints
from isaaclab_arena_gr00t.utils.robot_eef_pose import EefPose
from isaaclab_arena_gr00t.utils.robot_joints import JointsAbsPosition


def _get_video_pairs(config: Gr00tDatasetConfig) -> list[tuple[str, str]]:
    """Return (simulation camera key, LeRobot video key) pairs for one or more cameras."""
    camera_names = config.pov_cam_name_sim
    video_names = config.video_name_lerobot
    camera_names = list(camera_names) if isinstance(camera_names, (list, tuple)) else [camera_names]
    video_names = list(video_names) if isinstance(video_names, (list, tuple)) else [video_names]

    if len(camera_names) != len(video_names):
        raise ValueError(
            "pov_cam_name_sim and video_name_lerobot must contain the same number of entries "
            f"(got {len(camera_names)} and {len(video_names)})."
        )
    if not camera_names or any(not name for name in camera_names + video_names):
        raise ValueError("Camera and video key lists must be non-empty and contain no empty values.")
    if len(set(camera_names)) != len(camera_names):
        raise ValueError("pov_cam_name_sim contains duplicate camera keys.")
    if len(set(video_names)) != len(video_names):
        raise ValueError("video_name_lerobot contains duplicate video keys.")

    return list(zip(camera_names, video_names))


def wait_for_video_completion(video_path: str, max_wait_time: int = 60, check_interval: float = 0.5) -> bool:
    """
    Wait for a video file to be completely written and accessible.

    Args:
        video_path: Path to the video file.
        max_wait_time: Maximum time to wait in seconds.
        check_interval: How often to check the file status in seconds.

    Returns:
        True if video is complete and accessible, False if timeout.
    """
    video_path = Path(video_path)
    start_time = time.time()

    while time.time() - start_time < max_wait_time:
        # Check if file exists
        if not video_path.exists():
            time.sleep(check_interval)
            continue

        # Check if file is still being written by checking if size is stable
        try:
            size1 = video_path.stat().st_size
            time.sleep(check_interval)
            size2 = video_path.stat().st_size

            # If size is stable and > 0, file is likely complete
            if size1 == size2 and size1 > 0:
                # Try to open file to ensure it's not locked
                try:
                    with open(video_path, "rb") as f:
                        # Try to read first few bytes to ensure file is accessible
                        f.read(1024)
                    return True
                except OSError:
                    # File still locked, continue waiting
                    pass
        except OSError:
            # File might be in process of being created
            pass

        time.sleep(check_interval)

    return False


def get_video_metadata(video_path: str) -> dict[str, Any] | None:
    """
    Get video metadata in the specified format with robust file completion checking.

    Args:
        video_path: Path to the video file.

    Returns:
        Video metadata including shape, names, and video_info, or None if an error occurs.
    """
    # Wait for video file to be completely written
    if not wait_for_video_completion(video_path, max_wait_time=60):
        print(f"Timeout waiting for video completion: {video_path}")
        return None

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=height,width,codec_name,pix_fmt,r_frame_rate",
        "-of",
        "json",
        video_path,
    ]

    try:
        output = subprocess.check_output(cmd).decode("utf-8")
        probe_data = json.loads(output)
        stream = probe_data["streams"][0]

        # Parse frame rate (comes as fraction like "15/1")
        num, den = map(int, stream["r_frame_rate"].split("/"))
        fps = num / den

        # Check for audio streams
        audio_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "json",
            video_path,
        ]
        audio_output = subprocess.check_output(audio_cmd).decode("utf-8")
        audio_data = json.loads(audio_output)
        has_audio = len(audio_data.get("streams", [])) > 0

        metadata = {
            "dtype": "video",
            "shape": [stream["height"], stream["width"], 3],  # Assuming 3 channels
            "names": ["height", "width", "channel"],
            "video_info": {
                "video.width": stream["width"],
                "video.height": stream["height"],
                "video.fps": fps,
                "video.codec": stream["codec_name"],
                "video.pix_fmt": stream["pix_fmt"],
                "video.channels": 3,
                "video.is_depth_map": False,
                "has_audio": has_audio,
            },
        }

        print(f"Successfully extracted metadata for {video_path}")
        return metadata

    except subprocess.CalledProcessError as e:
        print(f"Error running ffprobe: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing ffprobe output: {e}")
        return None


def get_feature_info(
    step_data: pd.DataFrame, video_paths: dict[str, str], config: Gr00tDatasetConfig
) -> dict[str, Any]:
    """
    Get feature info from each frame of the video.

    Args:
        step_data: DataFrame containing data of an episode.
        video_paths: Dictionary mapping video keys to their file paths.
        config: Configuration object containing dataset and path information.
    Returns:
        Dictionary containing feature information for each column and video.
    """
    policy_joints_config = load_robot_joints_config_from_yaml(config.policy_joints_config_path)
    # flatten dict of dict into a single dict, perseving the order of the keys
    policy_joints_names = []
    for joint_group in policy_joints_config.keys():
        for joint_name in policy_joints_config[joint_group]:
            policy_joints_names.append(joint_name)
    features = {}
    for video_key, video_path in video_paths.items():
        video_metadata = get_video_metadata(video_path)
        features[video_key] = video_metadata
    assert isinstance(step_data, pd.DataFrame)
    for column in step_data.columns:
        column_data = np.stack(step_data[column], axis=0)
        shape = column_data.shape
        if len(shape) == 1:
            shape = (1,)
        else:
            shape = shape[1:]
        features[column] = {
            "dtype": column_data.dtype.name,
            "shape": shape,
        }
        # State & action
        if column in [config.lerobot_keys["state"], config.lerobot_keys["action"]]:
            dof = column_data.shape[1]
            assert dof == len(policy_joints_names)
            features[column]["names"] = [f"{policy_joints_names[i]}" for i in range(dof)]

    return features


def extract_teleop_command(trajectory: h5py.Dataset, teleop_key: str, config: Gr00tDatasetConfig) -> dict[str, Any]:
    """
    Extract the teleop command from the trajectory.
    """
    assert "action" in trajectory.keys()
    assert teleop_key in config.hdf5_keys
    teleop_command = trajectory["action"][config.hdf5_keys[teleop_key]][:-1]
    return [row for row in teleop_command]


def generate_info(
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
    total_videos: int,
    total_chunks: int,
    config: Gr00tDatasetConfig,
    step_data: pd.DataFrame,
    video_paths: dict[str, str],
) -> dict[str, Any]:
    """
    Generate the info.json data field.

    Args:
        total_episodes: Total number of episodes in the dataset.
        total_frames: Total number of frames across all episodes.
        total_tasks: Total number of tasks in the dataset.
        total_videos: Total number of videos in the dataset.
        total_chunks: Total number of data chunks.
        config: Configuration object containing dataset and path information.
        step_data: DataFrame containing step data for an example episode.
        video_paths: Dictionary mapping video keys to their file paths.

    Returns:
        Dictionary containing the info.json data field.
    """

    info_template = load_json(config.info_template_path)

    info_template["robot_type"] = config.robot_type
    info_template["total_episodes"] = total_episodes
    info_template["total_frames"] = total_frames
    info_template["total_tasks"] = total_tasks
    info_template["total_videos"] = total_videos
    info_template["total_chunks"] = total_chunks
    info_template["chunks_size"] = config.chunks_size
    info_template["fps"] = config.fps

    info_template["data_path"] = config.data_path
    info_template["video_path"] = config.video_path

    features = get_feature_info(step_data, video_paths, config)

    info_template["features"] = features
    return info_template


def write_video_job(queue: mp.Queue, error_queue: mp.Queue, config: Gr00tDatasetConfig) -> None:
    """
    Write frames to videos in mp4 format.

    Args:
        queue: Multiprocessing queue containing video frame data to be written.
        error_queue: Multiprocessing queue for reporting errors from worker processes.
        config: Configuration object containing dataset and path information.

    Returns:
        None
    """
    while True:
        job = queue.get()
        if job is None:
            break
        try:
            video_path, frames, fps, video_type = job
            if video_type == "image":
                # Create parent directory if it doesn't exist
                video_path.parent.mkdir(parents=True, exist_ok=True)
                assert (
                    frames.shape[1:] == config.original_image_size
                ), f"frames.shape[1:] {frames.shape[1:]} != config.original_image_size {config.original_image_size}"
                if config.target_image_size != config.original_image_size:
                    frames = resize_frames_with_padding(
                        frames, target_image_size=config.target_image_size, bgr_conversion=False, pad_img=True
                    )
                # h264 codec encoding
                imageio.mimwrite(video_path, frames, fps=fps, codec="libx264")

        except Exception as e:
            # Get the traceback and put in error queue
            error_msg = f"Error creating video {video_path}: {e}\n{traceback.format_exc()}"
            print(error_msg)
            error_queue.put(error_msg)


def convert_trajectory_to_df(
    trajectory: h5py.Dataset,
    episode_index: int,
    index_start: int,
    config: Gr00tDatasetConfig,
) -> dict[str, Any]:
    """
    Convert a single trajectory from HDF5 to a pandas DataFrame.

    Args:
        trajectory: HDF5 dataset containing trajectory data.
        episode_index: Index of the current episode.
        index_start: Starting index for the episode.
        config: Configuration object containing dataset and path information.

    Returns:
        Dictionary containing the DataFrame, annotation set, and episode length.
    """

    return_dict = {}
    data = {}

    policy_modality_config = load_json(config.modality_template_path)

    policy_joints_config = load_robot_joints_config_from_yaml(config.policy_joints_config_path)
    action_joints_config = load_robot_joints_config_from_yaml(config.action_joints_config_path)
    state_joints_config = load_robot_joints_config_from_yaml(config.state_joints_config_path)

    """Get joints state/action/timestamp from HDF5 file"""
    length = None
    assert "obs" in trajectory.keys()
    for key, hdf5_key_name in config.hdf5_keys.items():
        if key not in ["state", "action"]:
            continue
        lerobot_key_name = config.lerobot_keys[key]
        # state
        if key == "state":
            assert hdf5_key_name in trajectory["obs"].keys()
            joints = trajectory["obs"][hdf5_key_name]
        # action target
        else:
            assert hdf5_key_name in trajectory.keys()
            joints = trajectory[hdf5_key_name]
        # state
        if key == "state":
            # NOTE(xinjieyao, 2025-09-25): remove the last obs due to Lab reports observations
            joints = joints[:-1]
            input_joints_config = state_joints_config
        # action target
        elif key == "action":
            # NOTE(xinjieyao, 2025-09-25): remove the last idle action due to Lab reports actions
            joints = joints[:-1]
            input_joints_config = action_joints_config
        else:
            raise ValueError(f"Unknown key: {key}")
        assert joints.ndim == 2
        assert joints.shape[1] == len(input_joints_config)

        # 1.1. Remap the joints from Lab order to the LeRobot-GR00T order
        joints = JointsAbsPosition.from_array(joints, input_joints_config, device="cpu")
        remapped_joints = remap_sim_joints_to_policy_joints(joints, policy_joints_config)

        # 1.2. Fill in the missing joints with zeros
        ordered_joints = []
        for joint_group in policy_modality_config[key].keys():
            # NOTE(xinjieyao, 2025-09-25): Those are not joint position commands, which do not need remapping orders
            if (
                joint_group == "left_wrist_pose"
                or joint_group == "right_wrist_pose"
                or joint_group == "base_height_command"
                or joint_group == "navigate_command"
                or joint_group == "torso_orientation_rpy_command"
            ):
                continue
            num_joints = (
                policy_modality_config[key][joint_group]["end"] - policy_modality_config[key][joint_group]["start"]
            )

            if joint_group not in remapped_joints.keys():
                remapped_joints[joint_group] = np.zeros(
                    (joints.get_joints_pos().shape[0], num_joints), dtype=np.float64
                )
            else:
                assert remapped_joints[joint_group].shape[1] == num_joints
            ordered_joints.append(remapped_joints[joint_group])

        # 1.3. Concatenate the arrays for parquets
        concatenated = np.concatenate(ordered_joints, axis=1)
        data[lerobot_key_name] = [row for row in concatenated]

    assert len(data[config.lerobot_keys["action"]]) == len(data[config.lerobot_keys["state"]])
    length = len(data[config.lerobot_keys["action"]])
    data["timestamp"] = np.arange(length).astype(np.float64) * (1.0 / config.fps)

    """Get eef pose for state/action from HDF5 file(optional)"""

    for key in ["obs", "action"]:
        # break for loop if key not in trajectory.keys()
        if key not in trajectory.keys():
            continue
        eef_pose = {}
        for side in ["left", "right"]:
            if f"{side}_eef_pos" in config.hdf5_keys and f"{side}_eef_quat" in config.hdf5_keys:
                side_eef_pos = trajectory[key][config.hdf5_keys[f"{side}_eef_pos"]]
                side_eef_quat = trajectory[key][config.hdf5_keys[f"{side}_eef_quat"]]
                side_eef_pose = EefPose.from_array(side_eef_pos[:-1], side_eef_quat[:-1], device="cpu")
                eef_pose[side] = side_eef_pose.get_eef_pose()
        if "left" in eef_pose and "right" in eef_pose:
            eef_pose = np.concatenate([eef_pose["left"].numpy(), eef_pose["right"].numpy()], axis=1).astype(np.float64)

            assert eef_pose.shape == (length, 14), f"{eef_pose.shape} != ({length}, 14)"
            assert f"{key}_eef_pose" in config.lerobot_keys, f"{key}_eef_pose not in config.lerobot_keys"
            lerobot_key_name = config.lerobot_keys[f"{key}_eef_pose"]
            data[lerobot_key_name] = [row for row in eef_pose]

    """Get teleop command for action from HDF5 file(optional)"""

    teleop_command_keys = [
        "teleop_base_height_command",
        "teleop_navigate_command",
        "teleop_torso_orientation_rpy_command",
    ]
    for teleop_key in teleop_command_keys:
        if teleop_key in config.hdf5_keys:
            assert teleop_key in config.lerobot_keys
            data[config.lerobot_keys[teleop_key]] = extract_teleop_command(trajectory, teleop_key, config)

    # 2. Get the annotation
    assert "annotation" in config.lerobot_keys
    annotation_keys = config.lerobot_keys["annotation"]
    # task selection
    data[annotation_keys[0]] = np.ones(length, dtype=int) * config.task_index

    # 3. Other data
    data["episode_index"] = np.ones(length, dtype=int) * episode_index
    data["task_index"] = np.ones(length, dtype=int) * config.task_index
    data["index"] = np.arange(length, dtype=int) + index_start
    data["frame_index"] = np.arange(length, dtype=int)
    # last frame is successful
    reward = np.zeros(length, dtype=np.float64)
    reward[-1] = 1
    done = np.zeros(length, dtype=bool)
    done[-1] = True
    data["next.reward"] = reward
    data["next.done"] = done
    data["observation.img_state_delta"] = np.zeros(length, dtype=np.float64)

    dataframe = pd.DataFrame(data)

    return_dict["data"] = dataframe
    return_dict["length"] = length
    return_dict["annotation"] = set(data[annotation_keys[0]])
    return return_dict


def load_multi_hdf5_configs(yaml_path: str | Path) -> list[Gr00tDatasetConfig]:
    """Load shared conversion settings and per-file paths and instructions from a multi-HDF5 YAML file."""
    yaml_path = Path(yaml_path).expanduser().resolve()
    settings = load_dict_from_yaml(yaml_path)
    datasets = settings["datasets"]
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("datasets must be a non-empty list of HDF5 paths and language instructions.")

    def resolve_path(value: str) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else (yaml_path.parent / path).resolve()

    base_yaml = resolve_path(settings["base_config"])
    output_dir = resolve_path(settings["output_dir"])
    # Refuse to mix new episodes with an existing dataset or remove source data.
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}. Choose a new output_dir.")

    source_paths = []
    for source in datasets:
        path = resolve_path(source["hdf5_file_path"])
        if not path.is_file():
            raise FileNotFoundError(f"HDF5 file not found: {path}")
        instruction = source["language_instruction"]
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"language_instruction must be a non-empty string for {path}.")
        source_paths.append(path)

    config_data = load_config_from_yaml(base_yaml, Gr00tDatasetConfig)
    config_data.update(hdf5_name=str(source_paths[0]), output_dir=output_dir)
    base_config = Gr00tDatasetConfig(**config_data)
    configs = []
    task_indices = {}
    for source, path in zip(datasets, source_paths):
        instruction = source["language_instruction"]
        task_index = task_indices.setdefault(instruction, len(task_indices))
        # Copy without re-running __post_init__, which can remove an existing output directory.
        config = copy(base_config)
        config.hdf5_name = str(path)
        config.hdf5_file_path = path
        config.language_instruction = instruction
        config.task_index = task_index
        configs.append(config)
    return configs


def convert_hdf5_to_lerobot(configs: Gr00tDatasetConfig | list[Gr00tDatasetConfig]):
    """Convert one or more HDF5 files with shared settings into a single GR00T-LeRobot dataset."""
    if isinstance(configs, Gr00tDatasetConfig):
        configs = [configs]
    assert configs, "At least one HDF5 source is required."
    config = configs[0]
    assert config.chunks_size > 0, "chunks_size must be positive."
    video_pairs = _get_video_pairs(config)
    tasks = {}
    for source_config in configs:
        assert source_config.hdf5_file_path.is_file(), f"{source_config.hdf5_file_path} does not exist"
        assert source_config.lerobot_data_dir == config.lerobot_data_dir, "All sources must share one output directory."
        if source_config.task_index in tasks:
            assert tasks[source_config.task_index] == source_config.language_instruction, "Conflicting task instructions."
        tasks[source_config.task_index] = source_config.language_instruction

    lerobot_meta_dir = config.lerobot_data_dir / "meta"
    lerobot_meta_dir.mkdir(parents=True, exist_ok=True)
    queue = mp.Queue(maxsize=10)
    error_queue = mp.Queue()
    workers = []
    total_length = 0
    example_data = None
    video_paths = {}
    episodes_info = []

    try:
        for _ in range(4):
            worker = mp.Process(target=write_video_job, args=(queue, error_queue, config))
            worker.start()
            workers.append(worker)

        for source_config in configs:
            print(f"Converting {source_config.hdf5_file_path} (task_index={source_config.task_index})")
            source_episode_start = len(episodes_info)
            with h5py.File(source_config.hdf5_file_path, "r") as hdf5_handler:
                hdf5_data = hdf5_handler["data"]
                for trajectory_id in tqdm(list(hdf5_data.keys())):
                    # Keep episode and frame indices continuous across all source files.
                    episode_index = len(episodes_info)
                    trajectory = hdf5_data[trajectory_id]
                    try:
                        df_ret_dict = convert_trajectory_to_df(
                            trajectory=trajectory,
                            episode_index=episode_index,
                            index_start=total_length,
                            config=source_config,
                        )
                    except Exception as e:
                        print(f"Error loading trajectory {source_config.hdf5_file_path}:{trajectory_id}: {e}")
                        continue

                    dataframe = df_ret_dict["data"]
                    length = df_ret_dict["length"]
                    if example_data is not None:
                        assert dataframe.columns.equals(example_data["data"].columns), "Source feature columns differ."
                        for column in dataframe.columns:
                            assert np.asarray(dataframe[column].iloc[0]).shape == np.asarray(
                                example_data["data"][column].iloc[0]
                            ).shape, f"Source feature shape differs for {column}."
                    episode_chunk = episode_index // config.chunks_size
                    save_relpath = config.data_path.format(episode_chunk=episode_chunk, episode_index=episode_index)
                    save_path = config.lerobot_data_dir / save_relpath
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    dataframe.to_parquet(save_path)

                    camera_obs = trajectory["camera_obs"]
                    for camera_name, video_name in video_pairs:
                        if camera_name not in camera_obs:
                            raise KeyError(
                                f"Camera '{camera_name}' configured for video '{video_name}' "
                                f"is missing from trajectory '{trajectory_id}'. Available cameras: {list(camera_obs.keys())}"
                            )
                        new_video_relpath = config.video_path.format(
                            episode_chunk=episode_chunk, video_key=video_name, episode_index=episode_index
                        )
                        new_video_path = config.lerobot_data_dir / new_video_relpath
                        video_paths.setdefault(video_name, new_video_path)
                        # Remove the last frame due to how Lab reports observations.
                        frames = np.array(camera_obs[camera_name])[:-1]
                        assert len(frames) == length, (
                            f"Camera '{camera_name}' has {len(frames)} frames, expected {length} "
                            f"for trajectory '{trajectory_id}'."
                        )
                        queue.put((new_video_path, frames, config.fps, "image"))

                    total_length += length
                    episodes_info.append({
                        "episode_index": episode_index,
                        "tasks": [tasks[task_index] for task_index in df_ret_dict["annotation"]],
                        "length": length,
                    })
                    if example_data is None:
                        example_data = df_ret_dict
            assert len(episodes_info) > source_episode_start, (
                f"No valid episodes were converted from {source_config.hdf5_file_path}."
            )

        assert example_data is not None, "No valid episodes were converted from the configured HDF5 files."
        for _ in workers:
            queue.put(None)
        # Drain errors while workers exit so their queue feeder threads cannot block join().
        video_errors = []
        while any(worker.is_alive() for worker in workers):
            try:
                video_errors.append(error_queue.get(timeout=0.1))
            except Empty:
                pass
        for worker in workers:
            worker.join()
            assert worker.exitcode == 0, f"Video worker exited with code {worker.exitcode}."
        while True:
            try:
                video_errors.append(error_queue.get_nowait())
            except Empty:
                break
        if video_errors:
            raise RuntimeError("Video creation failed:\n" + "\n".join(video_errors))

        task_jsonlines = [{"task_index": task_index, "task": task} for task_index, task in tasks.items()]
        dump_jsonl(task_jsonlines, lerobot_meta_dir / config.tasks_fname)
        dump_jsonl(episodes_info, lerobot_meta_dir / config.episodes_fname)
        shutil.copy(config.modality_template_path, lerobot_meta_dir / config.modality_fname)
        info_json = generate_info(
            total_episodes=len(episodes_info),
            total_frames=total_length,
            total_tasks=len(tasks),
            total_videos=len(episodes_info) * len(video_pairs),
            total_chunks=(len(episodes_info) + config.chunks_size - 1) // config.chunks_size,
            step_data=example_data["data"],
            video_paths=video_paths,
            config=config,
        )
        info_json["splits"] = {"train": f"0:{len(episodes_info)}"}
        dump_json(info_json, lerobot_meta_dir / config.info_fname, indent=4)
        print(f"Successfully converted {len(configs)} HDF5 files to {config.lerobot_data_dir}")
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
            worker.join()
        queue.cancel_join_thread()
        queue.close()
        error_queue.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert HDF5 datasets to one GR00T LeRobot dataset")
    parser.add_argument("--yaml_file", help="Path to multi_hdf5.yaml or a single-file configuration", required=True)
    parser.add_argument("--hdf5_name", help="Input HDF5 filename (single-file mode only)")
    parser.add_argument("--language_instruction", help="Task instruction (single-file mode only)")
    args = parser.parse_args()

    settings = load_dict_from_yaml(args.yaml_file)
    if "datasets" in settings:
        if args.hdf5_name is not None or args.language_instruction is not None:
            parser.error("Set per-file paths and language_instruction in multi_hdf5.yaml, not CLI overrides.")
        configs = load_multi_hdf5_configs(args.yaml_file)
    else:
        configs = [
            create_config_from_yaml(args.yaml_file, Gr00tDatasetConfig, args.hdf5_name, args.language_instruction)
        ]
    print("\n" + "=" * 50)
    print("GR00T LEROBOT DATASET CONFIGURATION:")
    print("=" * 50)
    for field in fields(Gr00tDatasetConfig):
        if field.init:
            print(f"  {field.name}: {getattr(configs[0], field.name)}")
    for source_config in configs:
        print(f"  source: {source_config.hdf5_file_path}")
        print(f"    task_index: {source_config.task_index}")
        print(f"    language_instruction: {source_config.language_instruction}")
    print("=" * 50 + "\n")
    convert_hdf5_to_lerobot(configs)
