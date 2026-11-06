# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Merge already-converted, same-task LeRobot v2.0/v2.1 datasets.

Examples (run from the repository root):
    python assets/merge_lerobot_datasets.py \
        --datasets /data/run1/lerobot /data/run2/lerobot --output /data/merged
    python assets/merge_lerobot_datasets.py \
        --datasets /data/run1/lerobot /data/run2/lerobot --output /data/merged --apply

Each meta/episodes.jsonl is the KEEP list: omitted failed episodes are ignored,
even if their files remain. Inputs are concatenated in command-line order, with
retained episodes sorted by original episode_index within each input. No prior
cleanup or --original-count is needed. Episode IDs and global frame indices are
rebuilt from zero; frame_index/timestamp and video bytes are preserved. Input
chunk sizes may differ; output chunks contain 1000 episodes by default.

All retained data becomes the train split (source split memberships are NOT
preserved). Task descriptions must match exactly; different source task_index
values are mapped to output task_index=0. Features, FPS, robot type and modality
must agree. Only per-episode parquet plus optional external video is supported.

Requires numpy, pandas and pyarrow, plus clean_lerobot_dataset.py beside this
script; no Isaac Sim or Arena imports. Numeric statistics, including exact
q01/q99, are recomputed (RAM includes one full numeric feature). Existing video
statistics require complete per-episode coverage across all inputs if present
anywhere, and are aggregated without decoding videos. Video integrity/frame
counts are not checked. Old relative statistics and initial_actions.npz caches
are never copied. If relative_stats.json exists, explicitly choose
--drop-relative-stats or --gr00t-root to regenerate it before training.

Default: read-only metadata/file preview. --apply builds a staged copy and only
publishes it after success. Output must be new and separate from every input;
sources are never modified. Root extras/README/caches are not copied. The output
merge_report.json records source-to-output episode mappings.
"""

import argparse
import copy
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from clean_lerobot_dataset import (
    KNOWN_META,
    aggregate_video_stats,
    episode_path,
    numeric_array,
    read_json,
    read_jsonl,
    summarize,
    write_json,
    write_jsonl,
)


@dataclass
class Source:
    """Validated metadata and retained episodes of one input dataset."""

    root: Path
    info: dict
    episodes: list
    task_index: int
    episode_stats: dict
    global_stats: dict
    modality: dict | None


def load_source(root):
    """Read a source keep list and check metadata and retained episode files."""
    info = read_json(root / "meta/info.json")
    if info.get("codebase_version") not in {"v2.0", "v2.1"}:
        raise ValueError(f"{root}: only LeRobot v2.0/v2.1 per-episode datasets are supported.")
    unknown = {path.name for path in (root / "meta").iterdir()} - KNOWN_META
    if unknown:
        raise ValueError(f"{root}: unsupported metadata may contain stale references: {sorted(unknown)}")
    chunk_size = info["chunks_size"]
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError(f"{root}: chunks_size must be a positive integer.")
    episodes = read_jsonl(root / "meta/episodes.jsonl")
    ids = [ep["episode_index"] for ep in episodes]
    if any(type(index) is not int or index < 0 for index in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"{root}: episode IDs must be unique non-negative integers.")
    if not episodes:
        raise ValueError(f"{root}: episodes.jsonl keep list is empty.")
    episodes.sort(key=lambda ep: ep["episode_index"])
    tasks = read_jsonl(root / "meta/tasks.jsonl")
    if len(tasks) != 1 or set(tasks[0]) != {"task_index", "task"}:
        raise ValueError(f"{root}: expected exactly one task with task_index and task fields.")
    task = tasks[0]
    if type(task["task_index"]) is not int or task["task_index"] < 0 or not isinstance(task["task"], str):
        raise ValueError(f"{root}: invalid task ID or description.")
    cameras = []
    for key, feature in info["features"].items():
        dtype = feature["dtype"]
        if dtype == "video":
            cameras.append(key)
        elif not (dtype.startswith(("float", "int", "uint")) or dtype in {"bool", "string"}):
            raise ValueError(f"{root}: unsupported feature {key}={dtype} (embedded images are not supported).")
    required = {"episode_index", "index", "frame_index", "task_index", "timestamp"}
    if not required <= info["features"].keys():
        raise ValueError(f"{root}: missing standard features: {required - info['features'].keys()}")
    paths = set()
    for ep in episodes:
        if set(ep) != {"episode_index", "tasks", "length"}:
            raise ValueError(f"{root}: expected episode_index, tasks and length only: {ep}")
        if ep["tasks"] != [task["task"]]:
            raise ValueError(f"{root}: episode {ep['episode_index']} has a different task description.")
        old = ep["episode_index"]
        data = episode_path(root, info["data_path"], old, chunk_size, "data")
        files = [data]
        for camera in cameras:
            files.append(episode_path(root, info["video_path"], old, chunk_size, "videos", camera))
        for path in files:
            if not path.is_file() or path.stat().st_size == 0 or path in paths:
                raise ValueError(f"Missing, empty or shared retained file: {path}")
            paths.add(path)
        if type(ep["length"]) is not int or ep["length"] <= 0 or pq.ParquetFile(data).metadata.num_rows != ep["length"]:
            raise ValueError(f"{data}: episode length must be positive and match parquet row count.")
    episode_stats = {}
    if (root / "meta/episodes_stats.jsonl").exists():
        for entry in read_jsonl(root / "meta/episodes_stats.jsonl"):
            old = entry["episode_index"]
            if old in episode_stats:
                raise ValueError(f"{root}: duplicate episode statistics for {old}.")
            episode_stats[old] = entry["stats"]
    global_stats = read_json(root / "meta/stats.json") if (root / "meta/stats.json").exists() else {}
    modality = read_json(root / "meta/modality.json") if (root / "meta/modality.json").exists() else None
    return Source(root, info, episodes, task["task_index"], episode_stats, global_stats, modality), task["task"]


def prepare(args):
    """Validate sources, compatibility, statistics policy and output without writing."""
    if len(args.datasets) < 2:
        raise ValueError("Provide at least two --datasets.")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive.")
    roots = [path.expanduser().resolve() for path in args.datasets]
    if len(set(roots)) != len(roots):
        raise ValueError("Duplicate input datasets are not allowed.")
    if args.output.expanduser().is_symlink():
        raise ValueError("Output must not be a symlink.")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if not output.parent.is_dir():
        raise ValueError(f"Output parent must already exist: {output.parent}")
    for root in roots:
        if output.is_relative_to(root) or root.is_relative_to(output):
            raise ValueError(f"Output must be separate from input: {root}")
    if args.drop_relative_stats and args.gr00t_root:
        raise ValueError("Choose --drop-relative-stats OR --gr00t-root, not both.")
    if args.modality_config_path and not args.gr00t_root:
        raise ValueError("--modality-config-path requires --gr00t-root.")
    if args.gr00t_root and not (args.gr00t_root.expanduser().resolve() / "gr00t/data/stats.py").is_file():
        raise ValueError("--gr00t-root must contain gr00t/data/stats.py.")
    if args.modality_config_path and not args.modality_config_path.expanduser().is_file():
        raise ValueError("--modality-config-path does not exist.")
    sources = []
    description = None
    for root in roots:
        source, task = load_source(root)
        if sources:
            reference = sources[0]
            if task != description:
                raise ValueError(f"{root}: task description differs from the first dataset.")
            for key in ("features", "fps", "robot_type"):
                if source.info.get(key) != reference.info.get(key):
                    raise ValueError(f"{root}: incompatible info.json field {key!r}.")
            if source.modality != reference.modality:
                raise ValueError(f"{root}: modality.json differs (or is missing in only some inputs).")
        if (root / "meta/relative_stats.json").exists() and not (args.drop_relative_stats or args.gr00t_root):
            raise ValueError(f"{root}: relative_stats.json exists; choose --drop-relative-stats or --gr00t-root.")
        sources.append(source)
        description = task
    info = copy.deepcopy(sources[0].info)
    info.update(
        codebase_version="v2.1", chunks_size=args.chunk_size,
        data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        video_path="videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    )
    cameras = [key for key, feature in info["features"].items() if feature["dtype"] == "video"]
    numeric = []
    for key, feature in info["features"].items():
        if feature["dtype"].startswith(("float", "int", "uint")) or feature["dtype"] == "bool":
            numeric.append(key)
    video_stats_keys = []
    for camera in cameras:
        any_stats = False
        for source in sources:
            if camera in source.global_stats or any(
                camera in source.episode_stats.get(ep["episode_index"], {}) for ep in source.episodes
            ):
                any_stats = True
        if any_stats:
            for source in sources:
                for ep in source.episodes:
                    if camera not in source.episode_stats.get(ep["episode_index"], {}):
                        raise ValueError(
                            f"{source.root}: missing {camera} statistics for episode {ep['episode_index']}."
                        )
            video_stats_keys.append(camera)
    return sources, output, info, description, numeric, cameras, video_stats_keys


def assign_indices(frame, key, values):
    """Assign integer indices without silently overflowing the existing column dtype."""
    dtype = frame[key].dtype
    if not pd.api.types.is_integer_dtype(dtype) or pd.api.types.is_bool_dtype(dtype):
        raise ValueError(f"Index column {key} must have integer dtype, got {dtype}.")
    bounds = np.iinfo(dtype)
    if values.min() < bounds.min or values.max() > bounds.max:
        raise ValueError(f"Merged {key} exceeds source dtype {dtype}; convert source indices to int64 first.")
    frame[key] = values.astype(dtype)


def regenerate_relative_stats(stage, args):
    """Regenerate GR00T statistics with its actual embodiment/modality logic."""
    code = (
        "import sys, runpy; from pathlib import Path; sys.path.insert(0, sys.argv[1]); "
        "sys.path.insert(0, str(Path(sys.argv[4]).parent)) if sys.argv[4] else None; "
        "runpy.run_path(sys.argv[4]) if sys.argv[4] else None; "
        "from gr00t.data.stats import main; from gr00t.data.types import EmbodimentTag; "
        "tag = EmbodimentTag[sys.argv[3]] if sys.argv[3] in EmbodimentTag.__members__ "
        "else EmbodimentTag(sys.argv[3]); main(sys.argv[2], tag)"
    )
    root = args.gr00t_root.expanduser().resolve()
    modality = str(args.modality_config_path.expanduser().resolve()) if args.modality_config_path else ""
    subprocess.run(
        [sys.executable, "-I", "-c", code, str(root), str(stage), args.embodiment_tag, modality], cwd=root, check=True,
    )
    if not (stage / "meta/relative_stats.json").is_file():
        raise ValueError("GR00T did not generate relative_stats.json; check embodiment/modality configuration.")


def rebuild(stage, plan, args):
    """Build merged episode files, metadata, statistics and provenance in staging."""
    sources, _, info, task, numeric, cameras, video_stats_keys = plan
    meta = stage / "meta"
    meta.mkdir()
    if sources[0].modality is not None:
        write_json(meta / "modality.json", sources[0].modality)
    episodes, episode_stats, data_paths, reports = [], [], [], []
    offset = 0
    schema = None
    for source in sources:
        mapping = {}
        for ep in source.episodes:
            old, new = ep["episode_index"], len(episodes)
            print(f"  {source.root}: episode {old} -> {new}", flush=True)
            old_path = episode_path(source.root, source.info["data_path"], old, source.info["chunks_size"], "data")
            current_schema = pq.read_schema(old_path).remove_metadata()
            if schema is None:
                schema = current_schema
            elif not schema.equals(current_schema):
                raise ValueError(f"Incompatible parquet schema: {old_path}")
            frame = pd.read_parquet(old_path)
            if len(frame) != ep["length"] or not (frame["episode_index"] == old).all():
                raise ValueError(f"Parquet length/episode_index does not match metadata: {old_path}")
            if not (frame["task_index"] == source.task_index).all():
                raise ValueError(f"Unknown task_index in {old_path}")
            if not np.array_equal(frame["frame_index"].to_numpy(), np.arange(len(frame))):
                raise ValueError(f"Non-contiguous frame_index would break video alignment: {old_path}")
            expected_columns = set(info["features"]) - set(cameras)
            if set(frame.columns) != expected_columns:
                raise ValueError(f"Parquet columns do not match declared non-video features: {old_path}")
            assign_indices(frame, "episode_index", np.full(len(frame), new, dtype=np.int64))
            assign_indices(frame, "index", np.arange(offset, offset + len(frame), dtype=np.int64))
            assign_indices(frame, "task_index", np.zeros(len(frame), dtype=np.int64))
            path = episode_path(stage, info["data_path"], new, info["chunks_size"], "data")
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False)
            data_paths.append(path)
            for camera in cameras:
                src = episode_path(
                    source.root, source.info["video_path"], old, source.info["chunks_size"], "videos", camera,
                )
                dst = episode_path(stage, info["video_path"], new, info["chunks_size"], "videos", camera)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            stats = {key: summarize(numeric_array(frame[key])) for key in numeric}
            for camera in video_stats_keys:
                stats[camera] = source.episode_stats[old][camera]
            episodes.append({"episode_index": new, "tasks": [task], "length": len(frame)})
            episode_stats.append({"episode_index": new, "stats": stats})
            mapping[old] = new
            offset += len(frame)
        reports.append({
            "source": str(source.root), "old_to_new_episode_index": mapping,
            "initial_actions_dropped": (source.root / "meta/initial_actions.npz").exists(),
            "relative_stats_dropped": (source.root / "meta/relative_stats.json").exists() and args.drop_relative_stats,
        })
    global_stats = {}
    for key in numeric:
        print(f"  recomputing stats: {key}", flush=True)
        arrays = [numeric_array(pd.read_parquet(path, columns=[key])[key]) for path in data_paths]
        global_stats[key] = summarize(np.concatenate(arrays, axis=0))
    for camera in video_stats_keys:
        global_stats[camera] = aggregate_video_stats([entry["stats"][camera] for entry in episode_stats])
    info.update(
        total_episodes=len(episodes), total_frames=offset, total_tasks=1,
        total_videos=len(episodes) * len(cameras),
        total_chunks=(len(episodes) + info["chunks_size"] - 1) // info["chunks_size"],
        splits={"train": f"0:{len(episodes)}"},
    )
    write_json(meta / "info.json", info)
    write_jsonl(meta / "tasks.jsonl", [{"task_index": 0, "task": task}])
    write_jsonl(meta / "episodes.jsonl", episodes)
    write_jsonl(meta / "episodes_stats.jsonl", episode_stats)
    write_json(meta / "stats.json", global_stats)
    if args.gr00t_root:
        regenerate_relative_stats(stage, args)
    write_json(stage / "merge_report.json", {
        "sources": reports, "total_episodes": len(episodes), "total_frames": offset,
        "chunks_size": info["chunks_size"], "split_policy": "all retained episodes become train",
    })


def main():
    """Preview or apply a non-destructive merge of same-task datasets."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", type=Path, required=True, help="Input LeRobot roots, in merge order")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; parent must exist")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Output episodes per chunk (default: 1000)")
    parser.add_argument("--apply", action="store_true", help="Write output; otherwise read-only preview")
    parser.add_argument("--gr00t-root", type=Path, help="GR00T checkout for relative-stat regeneration")
    parser.add_argument("--embodiment-tag", default="GR1", help="GR00T embodiment tag (default: GR1)")
    parser.add_argument("--modality-config-path", type=Path, help="Optional custom GR00T modality registration .py")
    parser.add_argument(
        "--drop-relative-stats", action="store_true",
        help="Explicitly omit old relative stats; regenerate before training",
    )
    args = parser.parse_args()
    stage = None
    try:
        plan = prepare(args)
        sources, output, *_ = plan
        total = sum(len(source.episodes) for source in sources)
        for source in sources:
            print(f"Source: {source.root} — keep {len(source.episodes)} episodes from episodes.jsonl")
        print(f"Destination: {output}\nEpisodes: {total}, new IDs: 0..{total - 1}")
        print(f"Chunk size: {args.chunk_size}; chunks: {(total + args.chunk_size - 1) // args.chunk_size}")
        print(f"WARNING: all retained episodes become train (0:{total}); source split memberships are not preserved.")
        print("Source files stay untouched. Root extras and initial_actions.npz caches are not copied.")
        if args.drop_relative_stats:
            print("WARNING: relative_stats.json is omitted; regenerate before relative-action training.")
        if not args.apply:
            print("DRY RUN: no files changed. Add --apply to execute (parquet values/schema validated during rebuild).")
            return
        stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.merging-", dir=output.parent))
        rebuild(stage, plan, args)
        # Reserve exclusively; never replace an already-existing output directory.
        output.mkdir()
        try:
            stage.rename(output)
        except BaseException:
            output.rmdir()
            raise
        stage = None
        print(f"DONE: {output}\nID mapping: {output / 'merge_report.json'}")
    except (ValueError, KeyError, TypeError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    finally:
        if stage is not None and stage.exists():
            shutil.rmtree(stage)


if __name__ == "__main__":
    main()
