# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Back up and filter meta/episodes.jsonl using retained ego-view video filenames.

Usage:
    python assets/filter_episodes_by_videos.py /path/to/lerobot
    python assets/filter_episodes_by_videos.py /path/to/lerobot --dry-run

Only episodes.jsonl is modified; episode IDs, videos, parquet files, statistics,
and other metadata are left unchanged. No third-party dependencies are required.
"""

import argparse
import json
import os
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path


def filter_episodes(root: Path, dry_run: bool = False) -> None:
    """Keep JSONL entries whose episode_index matches an ego-view video filename."""
    root = root.expanduser().resolve()
    episodes_path = root / "meta" / "episodes.jsonl"
    if not episodes_path.is_file():
        raise ValueError(f"找不到文件：{episodes_path}")

    video_ids = set()
    for video in (root / "videos").glob("chunk-*/observation.images.ego_view/episode_*.mp4"):
        match = re.fullmatch(r"episode_(\d+)\.mp4", video.name)
        if match and video.is_file():
            video_ids.add(int(match.group(1)))
    if not video_ids:
        raise ValueError("没有找到符合命名规则的 ego_view 视频；为避免误删，未修改 JSONL。")

    # Keep the original bytes, field order, and line endings of retained records.
    kept_lines = []
    total = 0
    for line_number, line in enumerate(episodes_path.read_bytes().splitlines(keepends=True), start=1):
        if not line.strip():
            continue
        try:
            episode = json.loads(line)
        except (ValueError, UnicodeError) as exc:
            raise ValueError(f"JSONL 第 {line_number} 行解析失败：{exc}") from exc
        index = episode.get("episode_index") if isinstance(episode, dict) else None
        if type(index) is not int or index < 0:
            raise ValueError(f"JSONL 第 {line_number} 行缺少有效的非负整数 episode_index。")
        total += 1
        if index in video_ids:
            kept_lines.append(line)

    removed = total - len(kept_lines)
    print(f"匹配到视频 episode 数：{len(video_ids)}")
    print(f"原始条目：{total}，保留：{len(kept_lines)}，{'预计删除' if dry_run else '待删除'}：{removed}")
    if total and not kept_lines:
        raise ValueError("视频文件名与 JSONL 条目完全不匹配；为避免清空文件，未修改 JSONL。")
    if dry_run:
        print("预览模式：未修改文件，也未创建备份。")
        return
    if not removed:
        print("删除了 0 条；无需修改，未创建备份。")
        return

    # Exclusive creation prevents overwriting an existing backup.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup_path = episodes_path.with_name(f"episodes.jsonl.bak.{timestamp}")
    with backup_path.open("xb") as backup, episodes_path.open("rb") as source:
        shutil.copyfileobj(source, backup)
    shutil.copystat(episodes_path, backup_path)
    print(f"备份：{backup_path}")

    # Stage next to the original so replacement is atomic on the same filesystem.
    staged_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=episodes_path.parent, prefix=".episodes_", delete=False) as staged:
            staged_path = Path(staged.name)
            staged.writelines(kept_lines)
            staged.flush()
            os.fsync(staged.fileno())
        shutil.copymode(episodes_path, staged_path)
        os.replace(staged_path, episodes_path)
    finally:
        if staged_path is not None:
            staged_path.unlink(missing_ok=True)
    print(f"删除了 {removed} 条，保留 {len(kept_lines)} 条。")
    print(f"已更新：{episodes_path}")


def main() -> None:
    """Parse the dataset root and optionally preview the filtering."""
    parser = argparse.ArgumentParser(description="根据保留的 ego_view 视频备份并过滤 meta/episodes.jsonl。")
    parser.add_argument("dataset_root", type=Path, help="LeRobot 数据集根目录（包含 videos/ 和 meta/）")
    parser.add_argument("--dry-run", action="store_true", help="仅预览删除数量，不修改文件")
    args = parser.parse_args()
    try:
        filter_episodes(args.dataset_root, args.dry_run)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"错误：{exc}\n")


if __name__ == "__main__":
    main()
