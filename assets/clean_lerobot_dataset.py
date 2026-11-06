# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Keep episodes listed in meta/episodes.jsonl and compact a LeRobot v2.x dataset.

Examples (original episode IDs are zero-based, in 0..original-count-1):
    python assets/clean_lerobot_dataset.py --original-count 50
    python assets/clean_lerobot_dataset.py --original-count 50 --apply
    python assets/clean_lerobot_dataset.py --dataset /data/lerobot --original-count 100 \
        --output /data/lerobot_clean --apply

Edit meta/episodes.jsonl first: its episode_index values are the KEEP list. IDs
missing from that list are removed, including trailing IDs inferred from
--original-count. Their parquet/videos may already be deleted. Retained files
must exist. Rebuilding excludes removed files rather than unlinking source files
before success; --output leaves the source untouched. No HDF5 demos are modified.

The default is a read-only preview. --apply without --output replaces the dataset
only after building a complete staged copy, retaining the original in a sibling
backup directory. No backup is automatically deleted. Stop recording/training
before applying, and allow disk space for another complete dataset.

Requires numpy, pandas and pyarrow, but not Isaac Sim or Arena. Numeric statistics
(including exact q01/q99) are recomputed; peak RAM includes one full numeric feature.
Existing per-episode video statistics are retained and aggregated without decoding
or re-encoding video. Video integrity/frame counts are not checked.

Existing GR00T relative statistics are NEVER copied. Supply --gr00t-root to
regenerate them using the active Python environment (--embodiment-tag defaults
to GR1), optionally with --modality-config-path. Alternatively, use --drop-relative-stats
and regenerate them before training. Tasks keep their original IDs and mapping.
Only known metadata is accepted; caches, README/model cards and other root extras
are not copied into the rebuilt dataset. They remain in the source or backup.
The derived meta/initial_actions.npz cache is also omitted; regenerate it from
cleaned training data if needed. info.json totals/splits and numeric statistics
are rebuilt; modality.json and task IDs/mapping remain unchanged.
"""

import argparse
import copy
import json
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from string import Formatter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


KNOWN_META = {
    "info.json", "episodes.jsonl", "tasks.jsonl", "modality.json", "stats.json",
    "episodes_stats.jsonl", "relative_stats.json", "initial_actions.npz",
}


def read_json(path):
    """Read a JSON document."""
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    """Read non-empty JSON lines."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, value):
    """Write finite, human-readable JSON."""
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    """Write one finite JSON object per line."""
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def removed_episode_ids(original_count, mapping):
    """Return original IDs absent from the episodes.jsonl keep list."""
    return sorted(set(range(original_count)) - set(mapping))


def episode_path(root, pattern, episode_id, chunk_size, folder, video_key=None):
    """Format an episode path and keep it within its dataset data/videos subtree."""
    fields = {field for _, field, _, _ in Formatter().parse(pattern) if field is not None}
    allowed = {"episode_index", "episode_chunk", "video_key"}
    if "episode_index" not in fields or not fields <= allowed:
        raise ValueError(f"Not a supported per-episode path pattern: {pattern}")
    relative = Path(pattern.format(
        episode_index=episode_id, episode_chunk=episode_id // chunk_size, video_key=video_key,
    ))
    if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != folder:
        raise ValueError(f"Unsafe {folder} path: {relative}")
    path = root / relative
    if not path.resolve().is_relative_to((root / folder).resolve()):
        raise ValueError(f"Path escapes dataset: {path}")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes dataset: {path}")
    return path


def compact_splits(splits, mapping):
    """Preserve membership of numeric half-open episode ranges after compaction."""
    result = {}
    for name, bounds in splits.items():
        if not isinstance(bounds, str) or not re.fullmatch(r"\d+:\d+", bounds):
            raise ValueError(f"Unsupported split {name!r}: {bounds!r}; expected 'start:end'.")
        start, end = map(int, bounds.split(":"))
        if end < start:
            raise ValueError(f"Reversed split: {name}={bounds}")
        members = [new for old, new in mapping.items() if start <= old < end]
        if members:
            assert members == list(range(members[0], members[-1] + 1)), "Non-contiguous split mapping"
            result[name] = f"{members[0]}:{members[-1] + 1}"
        else:
            insertion = sum(old < start for old in mapping)
            result[name] = f"{insertion}:{insertion}"
    return result


def numeric_array(series):
    """Convert a scalar/vector column to a finite float64 array with a feature axis."""
    values = np.stack(series.to_numpy()).astype(np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if not np.isfinite(values).all():
        raise ValueError(f"Retained data contains NaN/Inf in {series.name}; remove/fix that episode first.")
    return values


def summarize(values):
    """Compute population statistics and exact quantiles along the frame axis."""
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
        "count": [len(values)],
    }


def aggregate_video_stats(entries):
    """Combine existing video population statistics using their observation counts."""
    counts = np.array([np.asarray(entry["count"]).item() for entry in entries], dtype=np.float64)
    if not np.isfinite(counts).all() or (counts <= 0).any():
        raise ValueError("Invalid per-episode video statistics counts.")
    weights = counts / counts.sum()
    means = np.array([entry["mean"] for entry in entries], dtype=np.float64)
    stds = np.array([entry["std"] for entry in entries], dtype=np.float64)
    weights = weights.reshape((-1,) + (1,) * (means.ndim - 1))
    mean = (weights * means).sum(axis=0)
    variance = (weights * (stds ** 2 + (means - mean) ** 2)).sum(axis=0)
    return {
        "min": np.min([entry["min"] for entry in entries], axis=0).tolist(),
        "max": np.max([entry["max"] for entry in entries], axis=0).tolist(),
        "mean": mean.tolist(), "std": np.sqrt(np.maximum(variance, 0)).tolist(),
        "count": [int(counts.sum())],
    }


def prepare(args):
    """Validate metadata, retained files and destination without writing anything."""
    source = args.dataset.expanduser().resolve()
    info = read_json(source / "meta/info.json")
    if info.get("codebase_version") not in {"v2.0", "v2.1"}:
        raise ValueError("Only LeRobot v2.0/v2.1 per-episode datasets are supported (not v3).")
    unknown_meta = {path.name for path in (source / "meta").iterdir()} - KNOWN_META
    if unknown_meta:
        raise ValueError(f"Unknown metadata may contain episode references; handle it explicitly first: {unknown_meta}")
    chunk_size = info["chunks_size"]
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunks_size must be a positive integer.")
    episodes = sorted(read_jsonl(source / "meta/episodes.jsonl"), key=lambda ep: ep["episode_index"])
    ids = [ep["episode_index"] for ep in episodes]
    if any(type(ep_id) is not int or ep_id < 0 for ep_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("Episode IDs must be unique non-negative integers.")
    if type(args.original_count) is not int or args.original_count <= 0:
        raise ValueError("--original-count must be a positive integer (original IDs are 0..N-1).")
    outside = [ep_id for ep_id in ids if ep_id >= args.original_count]
    if outside:
        raise ValueError(f"Episode IDs outside --original-count {args.original_count}: {outside}")
    kept = episodes
    if not kept:
        raise ValueError("episodes.jsonl keep list is empty; refusing to remove every episode.")
    mapping = {ep["episode_index"]: new for new, ep in enumerate(kept)}
    info = copy.deepcopy(info)
    info["splits"] = compact_splits(info.get("splits", {"train": f"0:{max(ids) + 1}"}), mapping)
    cameras = [key for key, feature in info["features"].items() if feature["dtype"] == "video"]
    numeric = []
    for key, feature in info["features"].items():
        dtype = feature["dtype"]
        if dtype.startswith(("float", "int", "uint")) or dtype == "bool":
            numeric.append(key)
        elif dtype not in {"video", "string"}:
            raise ValueError(f"Unsupported feature dtype: {key}={dtype}; embedded images are not supported.")
    tasks = read_jsonl(source / "meta/tasks.jsonl")
    task_ids = [task["task_index"] for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Duplicate task IDs in tasks.jsonl.")
    paths = set()
    for ep in kept:
        old = ep["episode_index"]
        data_path = episode_path(source, info["data_path"], old, chunk_size, "data")
        metadata = pq.ParquetFile(data_path).metadata
        if ep["length"] <= 0 or metadata.num_rows != ep["length"]:
            raise ValueError(f"Episode {old}: metadata length {ep['length']} != parquet rows {metadata.num_rows} (or empty).")
        for path in [data_path] + [
            episode_path(source, info["video_path"], old, chunk_size, "videos", camera) for camera in cameras
        ]:
            if not path.is_file() or path.stat().st_size == 0 or path in paths:
                raise ValueError(f"Missing, empty or shared episode file: {path}")
            paths.add(path)
    if args.output and args.output.expanduser().is_symlink():
        raise ValueError("Output must not be a symlink.")
    output = args.output.expanduser().resolve() if args.output else source
    if args.output and (output.is_relative_to(source) or source.is_relative_to(output)):
        raise ValueError("Output must be separate from the source, not its ancestor or descendant.")
    if args.output and output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if not output.parent.is_dir():
        raise ValueError(f"Output parent must already exist: {output.parent}")
    if args.modality_config_path and not args.gr00t_root:
        raise ValueError("--modality-config-path requires --gr00t-root and --embodiment-tag.")
    if args.drop_relative_stats and args.gr00t_root:
        raise ValueError("Choose regeneration OR --drop-relative-stats, not both.")
    if args.gr00t_root and not (args.gr00t_root.expanduser().resolve() / "gr00t/data/stats.py").is_file():
        raise ValueError("--gr00t-root must contain gr00t/data/stats.py.")
    if args.modality_config_path and not args.modality_config_path.expanduser().is_file():
        raise ValueError("--modality-config-path does not exist.")
    if (source / "meta/relative_stats.json").exists() and not (args.gr00t_root or args.drop_relative_stats):
        raise ValueError(
            "relative_stats.json exists: supply --gr00t-root ROOT --embodiment-tag TAG to regenerate, "
            "or explicitly --drop-relative-stats (regenerate before training)."
        )
    old_episode_stats = {}
    if (source / "meta/episodes_stats.jsonl").exists():
        for entry in read_jsonl(source / "meta/episodes_stats.jsonl"):
            ep_id = entry["episode_index"]
            if ep_id in old_episode_stats:
                raise ValueError(f"Duplicate episode statistics for {ep_id}.")
            old_episode_stats[ep_id] = entry["stats"]
    old_stats = read_json(source / "meta/stats.json") if (source / "meta/stats.json").exists() else {}
    video_stats_keys = []
    for camera in cameras:
        any_stats = camera in old_stats or any(camera in old_episode_stats.get(old, {}) for old in mapping)
        if any_stats:
            for old in mapping:
                if camera not in old_episode_stats.get(old, {}):
                    raise ValueError(f"Cannot update {camera} global stats: missing per-episode stats for {old}.")
            video_stats_keys.append(camera)
    return source, output, info, kept, mapping, tasks, numeric, cameras, old_episode_stats, video_stats_keys


def rebuild(stage, plan, args):
    """Build a compact dataset and recompute statistics before any source replacement."""
    source, _, info, kept, mapping, tasks, numeric, cameras, old_episode_stats, video_stats_keys = plan
    meta = stage / "meta"
    meta.mkdir()
    # Modality describes feature slices, not episode IDs. Initial actions are a
    # derived cache with no supported schema here; omit rather than copy stale data.
    if (source / "meta/modality.json").is_file():
        shutil.copy2(source / "meta/modality.json", meta / "modality.json")
    episodes = []
    episode_stats = []
    data_paths = []
    offset = 0
    task_ids = {task["task_index"] for task in tasks}
    for ep in kept:
        old = ep["episode_index"]
        new = mapping[old]
        print(f"  episode {old} -> {new}", flush=True)
        old_path = episode_path(source, info["data_path"], old, info["chunks_size"], "data")
        frame = pd.read_parquet(old_path)
        if not (frame["episode_index"] == old).all():
            raise ValueError(f"Parquet episode_index does not match metadata: {old_path}")
        if not set(frame["task_index"].unique()) <= task_ids:
            raise ValueError(f"Unknown task_index in {old_path}")
        if not np.array_equal(frame["frame_index"].to_numpy(), np.arange(len(frame))):
            raise ValueError(f"Non-contiguous frame_index in {old_path}; refusing to break video alignment.")
        frame["episode_index"] = np.full(len(frame), new, dtype=frame["episode_index"].dtype)
        frame["index"] = np.arange(offset, offset + len(frame), dtype=frame["index"].dtype)
        new_path = episode_path(stage, info["data_path"], new, info["chunks_size"], "data")
        new_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(new_path, index=False)
        data_paths.append(new_path)
        for camera in cameras:
            old_video = episode_path(source, info["video_path"], old, info["chunks_size"], "videos", camera)
            new_video = episode_path(stage, info["video_path"], new, info["chunks_size"], "videos", camera)
            if new_video.exists():
                raise ValueError(f"Colliding video paths: {new_video}")
            new_video.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old_video, new_video)
        # Reject custom episode fields instead of silently keeping stale indices/paths.
        unknown_fields = set(ep) - {"episode_index", "length", "tasks"}
        if unknown_fields:
            raise ValueError(f"Unknown episode metadata fields require explicit handling: {unknown_fields}")
        episodes.append({**ep, "episode_index": new, "length": len(frame)})
        stats = {key: summarize(numeric_array(frame[key])) for key in numeric}
        for camera in video_stats_keys:
            stats[camera] = old_episode_stats[old][camera]
        episode_stats.append({"episode_index": new, "stats": stats})
        offset += len(frame)
    global_stats = {}
    # Read a single feature at a time to avoid holding all state/action features in RAM.
    for key in numeric:
        print(f"  recomputing stats: {key}", flush=True)
        arrays = [numeric_array(pd.read_parquet(path, columns=[key])[key]) for path in data_paths]
        global_stats[key] = summarize(np.concatenate(arrays, axis=0))
    for camera in video_stats_keys:
        global_stats[camera] = aggregate_video_stats([entry["stats"][camera] for entry in episode_stats])
    info.update(
        total_episodes=len(episodes), total_frames=offset, total_tasks=len(tasks),
        total_videos=len(episodes) * len(cameras),
        total_chunks=(len(episodes) + info["chunks_size"] - 1) // info["chunks_size"],
    )
    write_json(meta / "info.json", info)
    write_jsonl(meta / "episodes.jsonl", episodes)
    write_jsonl(meta / "tasks.jsonl", tasks)
    write_jsonl(meta / "episodes_stats.jsonl", episode_stats)
    write_json(meta / "stats.json", global_stats)
    if args.gr00t_root:
        # Use GR00T's action representation/horizon logic, never a generic action-state subtraction.
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
            [sys.executable, "-I", "-c", code, str(root), str(stage), args.embodiment_tag, modality],
            cwd=root, check=True,
        )
        if (source / "meta/relative_stats.json").exists() and not (meta / "relative_stats.json").is_file():
            raise ValueError("GR00T did not generate relative_stats.json; check the embodiment/modality configuration.")
    write_json(stage / "cleanup_report.json", {
        "source": str(source), "original_count": args.original_count,
        "removed_episode_ids": removed_episode_ids(args.original_count, mapping),
        "old_to_new_episode_index": mapping, "total_frames": offset,
        "initial_actions_dropped": (source / "meta/initial_actions.npz").exists(),
        "relative_stats_dropped": (source / "meta/relative_stats.json").exists() and args.drop_relative_stats,
    })


def main():
    """Preview or apply episode removal with continuous episode/frame indexing."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset", type=Path,
        default=Path("/Users/skywalker/Documents/exe/gr1_two_arm_can_sort_random_recorded_50_replay/lerobot"),
        help="LeRobot root; meta/episodes.jsonl supplies the keep list (default: recorded_50_replay/lerobot)",
    )
    parser.add_argument(
        "--original-count", type=int, required=True,
        help="Original episode count N before filtering; IDs absent from episodes.jsonl in 0..N-1 are removed",
    )
    parser.add_argument("--output", type=Path, help="Write a separate NEW directory instead of replacing the source")
    parser.add_argument("--apply", action="store_true", help="Apply changes; otherwise read-only preview")
    parser.add_argument("--gr00t-root", type=Path, help="GR00T checkout for optional relative-stat regeneration")
    parser.add_argument("--embodiment-tag", default="GR1", help="GR00T embodiment tag (default: GR1)")
    parser.add_argument("--modality-config-path", type=Path, help="Optional custom GR00T modality registration .py")
    parser.add_argument("--drop-relative-stats", action="store_true", help="Explicitly omit old relative stats; regenerate before training")
    args = parser.parse_args()
    stage = None
    try:
        plan = prepare(args)
        source, output, _, kept, mapping, *_ = plan
        print(f"Source: {source}\nDestination: {output}")
        print(f"Original episode count: {args.original_count}")
        print(f"Remove original IDs absent from episodes.jsonl: {removed_episode_ids(args.original_count, mapping)}")
        print(f"Keep: {len(kept)} episodes, {sum(ep['length'] for ep in kept)} frames")
        print(f"New IDs: 0..{len(mapping) - 1}\nSplits: {plan[2]['splits']}")
        print("Root extras/caches are not copied; all original content remains in source/backup.")
        if (source / "meta/initial_actions.npz").exists():
            print("WARNING: initial_actions.npz cache will be omitted. Regenerate it from the cleaned dataset.")
        if args.drop_relative_stats:
            print("WARNING: relative_stats.json will be omitted. Regenerate before relative-action training.")
        if not args.apply:
            print("DRY RUN: no files changed. Add --apply to execute (parquet values validated during rebuild).")
            return
        stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.cleaning-", dir=output.parent))
        rebuild(stage, plan, args)
        if args.output:
            # Reserve destination exclusively so an existing directory is never overwritten.
            output.mkdir()
            try:
                stage.rename(output)
            except BaseException:
                output.rmdir()
                raise
        else:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            backup = source.with_name(f"{source.name}.backup-{stamp}-{uuid.uuid4().hex[:8]}")
            source.rename(backup)
            try:
                stage.rename(source)
            except BaseException:
                backup.rename(source)
                raise
            print(f"Original dataset backup (not deleted): {backup}")
        stage = None
        print(f"DONE: {output}\nID mapping: {output / 'cleanup_report.json'}")
    except (ValueError, KeyError, TypeError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    finally:
        if stage is not None and stage.exists():
            shutil.rmtree(stage)


if __name__ == "__main__":
    main()
