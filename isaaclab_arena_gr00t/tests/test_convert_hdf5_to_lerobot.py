# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import json
import shutil
from pathlib import Path

import h5py
import pandas as pd
import pytest
import yaml

from isaaclab_arena_gr00t.lerobot.config.dataset_config import Gr00tDatasetConfig
from isaaclab_arena_gr00t.lerobot.convert_hdf5_to_lerobot import (
    _resolve_video_pairs,
    convert_hdf5_to_lerobot,
    generate_modality,
)
from isaaclab_arena_gr00t.tests.utils.constants import TestConstants
from isaaclab_arena_gr00t.utils.io_utils import create_config_from_yaml

pytestmark = pytest.mark.gr00t_policy


@pytest.mark.parametrize("num_episodes_with_front_cam", [0, 1, 2])
def test_optional_front_camera(tmp_path, num_episodes_with_front_cam):
    hdf5_path = tmp_path / "demos.hdf5"
    with h5py.File(hdf5_path, "w") as hdf5_file:
        hdf5_data = hdf5_file.create_group("data")
        for index in range(2):
            cameras = hdf5_data.create_group(f"demo_{index}/camera_obs")
            cameras.create_dataset("robot_pov_cam_rgb", data=[0])
            cameras.create_dataset("left_wrist_cam_rgb", data=[0])
            cameras.create_dataset("right_wrist_cam_rgb", data=[0])
            if index < num_episodes_with_front_cam:
                cameras.create_dataset("front_cam_rgb", data=[0])

        gr1_dir = Path(TestConstants.repo_root) / "isaaclab_arena_gr00t" / "embodiments" / "gr1"
        config = Gr00tDatasetConfig(
            data_root=tmp_path,
            modality_template_path=gr1_dir / "modality_two_arm.json",
            info_template_path=gr1_dir / "info.json",
            policy_joints_config_path=gr1_dir / "gr00t_26dof_joint_space.yaml",
            action_joints_config_path=gr1_dir / "36dof_joint_space.yaml",
            state_joints_config_path=gr1_dir / "54dof_joint_space.yaml",
            hdf5_name=hdf5_path.name,
            pov_cam_name_sim=["robot_pov_cam_rgb", "left_wrist_cam_rgb", "right_wrist_cam_rgb"],
            video_name_lerobot=[
                "observation.images.ego_view",
                "observation.images.left_wrist",
                "observation.images.right_wrist",
            ],
            optional_video_cameras={"front_cam_rgb": "observation.images.front_view"},
        )
        video_pairs = _resolve_video_pairs(config, hdf5_data, list(hdf5_data))

    expected_pairs = list(zip(config.pov_cam_name_sim, config.video_name_lerobot))
    if num_episodes_with_front_cam == 2:
        expected_pairs.append(("front_cam_rgb", "observation.images.front_view"))
    assert video_pairs == expected_pairs
    assert generate_modality(config, video_pairs)["video"] == {
        video_key.rsplit(".", 1)[-1]: {"original_key": video_key} for _, video_key in expected_pairs
    }


def test_gr1_two_arm_requires_front_camera(tmp_path, monkeypatch):
    monkeypatch.chdir(TestConstants.repo_root)
    config_path = Path("isaaclab_arena_gr00t/lerobot/config/gr1_two_arm_config.yaml")
    config_values = yaml.safe_load(config_path.read_text())
    hdf5_path = tmp_path / "demos.hdf5"
    with h5py.File(hdf5_path, "w") as hdf5_file:
        camera_obs = hdf5_file.create_group("data/demo_0/camera_obs")
        for camera_name in ("robot_pov_cam_rgb", "left_wrist_cam_rgb", "right_wrist_cam_rgb"):
            camera_obs.create_dataset(camera_name, data=[0])

    config_values.update(data_root=tmp_path, hdf5_name=hdf5_path.name)
    config = Gr00tDatasetConfig(**config_values)
    with pytest.raises(KeyError, match="front_cam_rgb.*demo_0"):
        convert_hdf5_to_lerobot(config)
    assert not config.lerobot_data_dir.exists()

    with h5py.File(hdf5_path, "r+") as hdf5_file:
        hdf5_data = hdf5_file["data"]
        hdf5_data["demo_0/camera_obs"].create_dataset("front_cam_rgb", data=[0])
        video_pairs = _resolve_video_pairs(config, hdf5_data, ["demo_0"])

    assert video_pairs[-1] == ("front_cam_rgb", "observation.images.front_view")
    template = json.loads(Path(config.modality_template_path).read_text())
    assert generate_modality(config, video_pairs)["video"] == template["video"]


def test_g1_convert_hdf5_to_lerobot():
    # Load expected data for comparison
    expected_g1_parquet = pd.read_parquet(
        TestConstants.test_data_dir + "/test_g1_locomanip_lerobot/data/chunk-000/episode_000000.parquet"
    )
    g1_ds_config = create_config_from_yaml(
        TestConstants.test_data_dir + "/test_g1_locomanip_lerobot/test_g1_locomanip_config.yaml", Gr00tDatasetConfig
    )

    # Clean up any existing output directory
    if g1_ds_config.lerobot_data_dir.exists():

        shutil.rmtree(g1_ds_config.lerobot_data_dir)

    # Run conversion
    convert_hdf5_to_lerobot(g1_ds_config)

    # assert it has episodes.jsonl file
    assert (g1_ds_config.lerobot_data_dir / "meta" / "episodes.jsonl").exists()

    # assert it has tasks.jsonl file
    assert (g1_ds_config.lerobot_data_dir / "meta" / "tasks.jsonl").exists()

    # assert it has info.json file
    assert (g1_ds_config.lerobot_data_dir / "meta" / "info.json").exists()

    # assert it has modality.json file
    assert (g1_ds_config.lerobot_data_dir / "meta" / "modality.json").exists()

    # assert it has data/ folder has parquet files
    parquet_files = list((g1_ds_config.lerobot_data_dir / "data").glob("**/*.parquet"))
    assert len(parquet_files) == 1

    # assert it has videos/ folder has mp4 files
    mp4_files = list((g1_ds_config.lerobot_data_dir / "videos").glob("**/*.mp4"))
    assert len(mp4_files) == 1
    # check parquet file contains expected columns
    actual_df = pd.read_parquet(parquet_files[0])
    expected_columns = set(expected_g1_parquet.columns)
    actual_columns = set(actual_df.columns)
    assert expected_columns.issubset(actual_columns), f"Missing columns: {expected_columns - actual_columns}"
    # check parquet file data is the same as expected
    assert actual_df.equals(expected_g1_parquet)

    # remove lerobot_data_dir
    shutil.rmtree(g1_ds_config.lerobot_data_dir.parent)


if __name__ == "__main__":
    test_g1_convert_hdf5_to_lerobot()
