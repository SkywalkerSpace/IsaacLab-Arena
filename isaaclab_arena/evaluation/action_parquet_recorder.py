# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import torch
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnv


class _EpisodeParquetWriter:
    def __init__(self, path: Path, env_id: int, steps_per_batch: int):
        self._path = path
        self._env_id = env_id
        self._steps_per_batch = steps_per_batch
        self._steps_written = 0
        self._pending: list[torch.Tensor] = []
        self._writer: pq.ParquetWriter | None = None

    def record(self, action: torch.Tensor) -> None:
        self._pending.append(action)
        if len(self._pending) >= self._steps_per_batch:
            self._flush()

    def _flush(self) -> None:
        if not self._pending:
            return

        actions = torch.stack(self._pending).numpy()
        num_steps, action_dim = actions.shape
        table = pa.table({
            "step": np.arange(self._steps_written, self._steps_written + num_steps, dtype=np.int64),
            "env_id": np.full(num_steps, self._env_id, dtype=np.int32),
            "action": pa.FixedSizeListArray.from_arrays(pa.array(actions.reshape(-1)), action_dim),
        })
        if self._writer is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = pq.ParquetWriter(self._path, table.schema)
        self._writer.write_table(table)
        self._steps_written += num_steps
        self._pending.clear()

    def close(self) -> None:
        try:
            self._flush()
        finally:
            if self._writer is not None:
                self._writer.close()


class ActionParquetRecorder:
    """Record a separate Parquet file with episode-local steps for each environment episode."""

    def __init__(self, output_dir: str | Path, *, rank: int, steps_per_batch: int = 256):
        assert steps_per_batch > 0, "steps_per_batch must be positive"
        self._output_dir = Path(output_dir)
        self._rank = rank
        self._steps_per_batch = steps_per_batch
        self._writers: dict[tuple[int, int], _EpisodeParquetWriter] = {}

    def sync_episodes(self, env: IsaacLabArenaManagerBasedRLEnv) -> None:
        """Finish replaced or retired episodes and track newly assigned episodes."""
        active_ids = env.active_episode_mask.nonzero(as_tuple=True)[0].tolist()
        active_episodes = {(env_id, env.get_episode_index(env_id)) for env_id in active_ids}
        for key in self._writers.keys() - active_episodes:
            self._writers.pop(key).close()
        for env_id, episode in active_episodes - self._writers.keys():
            path = self._output_dir / f"actions_rank{self._rank}_env{env_id}_episode{episode}.parquet"
            self._writers[env_id, episode] = _EpisodeParquetWriter(path, env_id, self._steps_per_batch)

    def record(self, actions: torch.Tensor) -> None:
        """Snapshot the actions for active episodes before env.step can mutate them."""
        if not self._writers:
            return
        dtype = torch.float32 if actions.dtype in (torch.float16, torch.bfloat16) else actions.dtype
        snapshot = actions.detach().to(device="cpu", dtype=dtype, copy=True)
        for (env_id, _), writer in self._writers.items():
            writer.record(snapshot[env_id])

    def close(self) -> None:
        """Finalize any episodes with actions that were still active when rollout stopped."""
        with ExitStack() as stack:
            for writer in self._writers.values():
                stack.callback(writer.close)
            self._writers.clear()
