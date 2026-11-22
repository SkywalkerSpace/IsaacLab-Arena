# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
from pathlib import Path


class ActionParquetRecorder:
    """Write raw rollout actions with one row per step and environment slot."""

    def __init__(self, path: str | Path, *, steps_per_batch: int = 256):
        assert steps_per_batch > 0, "steps_per_batch must be positive"
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("--record_actions requires pyarrow; install it in the evaluation environment.") from exc

        self._pa = pa
        self._pq = pq
        self._path = Path(path)
        self._steps_per_batch = steps_per_batch
        self._steps_written = 0
        self._pending: list[torch.Tensor] = []
        self._writer = None

    def record(self, actions: torch.Tensor) -> None:
        """Snapshot a batch of shape (num_envs, action_dim) before env.step."""
        dtype = torch.float32 if actions.dtype in (torch.float16, torch.bfloat16) else actions.dtype
        self._pending.append(actions.detach().to(device="cpu", dtype=dtype, copy=True))
        if len(self._pending) >= self._steps_per_batch:
            self._flush()

    def _flush(self) -> None:
        if not self._pending:
            return

        actions = torch.stack(self._pending).numpy()
        num_steps, num_envs, action_dim = actions.shape
        steps = np.arange(self._steps_written, self._steps_written + num_steps, dtype=np.int64)
        columns = {
            "step": np.repeat(steps, num_envs),
            "env_id": np.tile(np.arange(num_envs, dtype=np.int32), num_steps),
            "action": self._pa.FixedSizeListArray.from_arrays(self._pa.array(actions.reshape(-1)), action_dim),
        }
        table = self._pa.table(columns)
        if self._writer is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = self._pq.ParquetWriter(self._path, table.schema)
        self._writer.write_table(table)
        self._steps_written += num_steps
        self._pending.clear()

    def close(self) -> None:
        """Flush pending actions and finalize the file."""
        try:
            self._flush()
        finally:
            if self._writer is not None:
                self._writer.close()
