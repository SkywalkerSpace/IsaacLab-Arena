# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""ActionChunkScheduler: buffer an action chunk and step through it sequentially."""

from __future__ import annotations

import torch
from collections.abc import Callable

from isaaclab_arena.policy.action_scheduling.action_scheduler import ActionScheduler
from isaaclab_arena.policy.action_scheduling.chunk_blending import crossfade_action_chunks


class ActionChunkScheduler(ActionScheduler):
    """Buffers one action chunk and replays it one step at a time.

    Fetches a new action tensor from the policy only when the current one is exhausted.
    Per-env tracking allows environments to refetch independently.
    """

    def __init__(
        self,
        num_envs: int,
        action_chunk_length: int,
        action_horizon: int,
        action_dim: int,
        device: str | torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.num_envs = num_envs
        self.action_chunk_length = action_chunk_length
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.device = device

        self.current_action_chunk = torch.zeros(
            (num_envs, action_horizon, action_dim),
            dtype=dtype,
            device=device,
        )
        # Use a bool list to indicate that the action chunk is not yet computed for each env
        # True means the action chunk is not yet computed, False means the action
        self.current_action_index = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self.env_requires_new_chunk = torch.ones(num_envs, dtype=torch.bool, device=device)

        # Fetch-efficiency tracking: how many times each env triggered a chunk fetch,
        # and how many envs actually needed the fetch vs. total (wasted compute detection).
        self._n_fetch_calls: int = 0
        self._total_envs_needed: int = 0
        self._per_env_fetch_count = torch.zeros(num_envs, dtype=torch.int64, device=device)

    def get_action(
        self,
        fetch_action_tensor_fn: Callable[[], torch.Tensor],
        hold_action: torch.Tensor | None = None,
        *,
        step_limit: int | None = None,
        crossfade_dim_limit: int | None = None,
    ) -> torch.Tensor:
        """Return one action per env, refilling the chunk when needed.

        fetch_action_tensor_fn() must return a tensor of shape (num_envs, horizon, action_dim)
        with horizon >= action_horizon.

        ``hold_action`` is part of the base ``ActionScheduler`` API for schedulers that
        need a fallback for waiting envs; this scheduler doesn't have a waiting state
        so the argument is accepted and ignored.
        """
        del hold_action
        steps_per_chunk, keep_tail = self._execution_window(step_limit)
        if self.env_requires_new_chunk.any():
            new_chunk = fetch_action_tensor_fn()
            mask = self.env_requires_new_chunk
            self._n_fetch_calls += 1
            n_needed = int(mask.sum().item())
            self._total_envs_needed += n_needed
            self._per_env_fetch_count[mask] += 1
            self._install_chunk(new_chunk, mask, steps_per_chunk, keep_tail, crossfade_dim_limit)

        assert self.current_action_index.min() >= 0, "At least one env's action index is less than 0"
        assert (
            self.current_action_index.max() < steps_per_chunk
        ), "At least one env's action index is beyond the execution window"

        # Take one action per env at the current index (before incrementing)
        batch_idx = torch.arange(self.num_envs, device=self.device)
        action = self.current_action_chunk[batch_idx, self.current_action_index]
        assert action.shape == (
            self.num_envs,
            self.action_dim,
        ), f"{action.shape=} != ({self.num_envs}, {self.action_dim})"

        self._advance_chunks(~self.env_requires_new_chunk, steps_per_chunk, keep_tail)
        return action

    def _execution_window(self, step_limit: int | None) -> tuple[int, bool]:
        if step_limit is None:
            return self.action_chunk_length, False
        assert step_limit > 0, "step_limit must be positive"
        steps = min(self.action_chunk_length, step_limit)
        return steps, steps < self.action_horizon

    def _install_chunk(
        self, new_chunk: torch.Tensor, mask: torch.Tensor, steps: int, keep_tail: bool, crossfade_dim_limit: int | None
    ) -> None:
        incoming = new_chunk[mask, : self.action_horizon]
        if keep_tail:
            # Natural exhaustion keeps the index at `steps`; reset/hold invalidates the tail.
            has_tail = self.current_action_index[mask] == steps
            blended = crossfade_action_chunks(
                self.current_action_chunk[mask], incoming, steps, steps, action_dim_limit=crossfade_dim_limit
            )
            incoming = torch.where(has_tail[:, None, None], blended, incoming)
        self.current_action_chunk[mask] = incoming
        self.current_action_index[mask] = 0
        self.env_requires_new_chunk[mask] = False

    def _advance_chunks(self, active: torch.Tensor, steps: int, keep_tail: bool) -> None:
        self.current_action_index[active] += 1
        exhausted = active & (self.current_action_index >= steps)
        self.env_requires_new_chunk[exhausted] = True
        if not keep_tail:
            self._discard_chunks(exhausted)

    def _discard_chunks(self, env_ids: torch.Tensor | slice) -> None:
        self.current_action_chunk[env_ids] = 0.0
        self.current_action_index[env_ids] = -1

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Reset chunking state for the given envs (all if None)."""
        if env_ids is None:
            env_ids = slice(None)
        self._discard_chunks(env_ids)
        self.env_requires_new_chunk[env_ids] = True
