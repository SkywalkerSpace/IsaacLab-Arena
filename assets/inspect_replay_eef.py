# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Inspect EEF fields in the first two demos of a replay HDF5 file.

python3 assets/inspect_replay_eef.py "datasets/${TSAK_NAME}_recorded_${NUM_RECORDED_DEMOS}_replay.hdf5"
"""

import argparse
import json
import re
from pathlib import Path

import h5py


def inspect_demo(name: str, demo: h5py.Group, frames: int) -> None:
    print(f"\n/data/{name} (num_samples={demo.attrs.get('num_samples', 'unknown')})")
    eef_fields = []

    def collect_eef(path: str, obj: h5py.Dataset | h5py.Group) -> None:
        if isinstance(obj, h5py.Dataset) and "eef" in path.lower():
            eef_fields.append(path)

    demo.visititems(collect_eef)
    if eef_fields:
        print("EEF datasets: found")
    else:
        print("EEF datasets: no explicit EEF fields found (raw actions may still contain EEF targets).")

    for path in [*eef_fields, "actions", "processed_actions"]:
        dataset = demo.get(path)
        if not isinstance(dataset, h5py.Dataset):
            continue
        preview = dataset[()] if dataset.ndim == 0 else dataset[:frames]
        print(f"  {path}: shape={dataset.shape}, dtype={dataset.dtype}")
        print(preview)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hdf5_file", type=Path, help="Replay HDF5 file to inspect (read-only).")
    parser.add_argument("--frames", type=int, default=2, help="Number of rows to print per field (default: 2).")
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be at least 1")

    with h5py.File(args.hdf5_file, "r") as hdf5:
        data = hdf5.get("data")
        if not isinstance(data, h5py.Group):
            parser.error("HDF5 file has no /data group")
        demo_names = [name for name in data if re.fullmatch(r"demo_\d+", name) and isinstance(data[name], h5py.Group)]
        demo_names.sort(key=lambda name: int(name.split("_")[1]))
        print(f"{args.hdf5_file}: {len(demo_names)} demo(s), showing the first {min(2, len(demo_names))}")

        metadata = json.loads(data.attrs.get("metadata", "{}"))
        for key in ("action_terms", "action_term_dims"):
            if key in metadata:
                print(f"{key}: {metadata[key]}")

        for name in demo_names[:2]:
            inspect_demo(name, data[name], args.frames)


if __name__ == "__main__":
    main()
