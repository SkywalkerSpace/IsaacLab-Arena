# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the ActionChunkScheduler and SyncedBatchActionScheduler.

Pure torch — no Isaac Sim, no remote server. Verifies the per-env stepping,
reset semantics, and (for the synced variant) the hold-on-wait behavior.
"""

from __future__ import annotations

import torch

import pytest

NUM_ENVS = 2
ACTION_CHUNK_LENGTH = 4
ACTION_HORIZON = 4
ACTION_DIM = 3


def _make_chunk() -> torch.Tensor:
    """Deterministic (NUM_ENVS, ACTION_HORIZON, ACTION_DIM) tensor with unique values per slot."""
    return torch.arange(NUM_ENVS * ACTION_HORIZON * ACTION_DIM, dtype=torch.float).reshape(
        NUM_ENVS, ACTION_HORIZON, ACTION_DIM
    )


@pytest.mark.parametrize("scheduler_name", ["ActionChunkScheduler", "SyncedBatchActionScheduler"])
def test_chunk_crossfade_is_time_aligned_and_skips_reset_envs(scheduler_name):
    from isaaclab_arena.policy import action_scheduling

    scheduler_cls = getattr(action_scheduling, scheduler_name)
    scheduler = scheduler_cls(num_envs=2, action_chunk_length=16, action_horizon=16, action_dim=3, device="cpu")
    previous = torch.arange(96, dtype=torch.float32).reshape(2, 16, 3)
    current = previous + 100
    following = previous + 200
    chunks = iter((previous, current, following))
    hold = torch.zeros(2, 3)

    def step():
        return scheduler.get_action(lambda: next(chunks), hold, step_limit=8)

    for k in range(8):
        torch.testing.assert_close(step(), previous[:, k])
    scheduler.reset(torch.tensor([1]))

    for k in range(8):
        new_fraction = k / 7
        expected = current[:, k].clone()
        expected[0] = (1 - new_fraction) * previous[0, 8 + k] + new_fraction * current[0, k]
        torch.testing.assert_close(step(), expected)

    # The blended prefix must not overwrite the tail for the next replan.
    torch.testing.assert_close(step(), current[:, 8])


# ----------------------------- ActionChunkScheduler ------------------------------


def test_action_chunk_scheduler_steps_through_chunk_and_refetches_when_exhausted():
    from isaaclab_arena.policy.action_scheduling import ActionChunkScheduler

    scheduler = ActionChunkScheduler(NUM_ENVS, ACTION_CHUNK_LENGTH, ACTION_HORIZON, ACTION_DIM, device="cpu")
    chunk = _make_chunk()

    fetch_calls = 0

    def fetch() -> torch.Tensor:
        nonlocal fetch_calls
        fetch_calls += 1
        return chunk

    # First call: every env needs a chunk → exactly one fetch, action is slot 0.
    a0 = scheduler.get_action(fetch)
    assert fetch_calls == 1
    assert a0.shape == (NUM_ENVS, ACTION_DIM)
    torch.testing.assert_close(a0, chunk[:, 0])

    # Subsequent calls within the chunk step without re-fetching.
    for k in range(1, ACTION_CHUNK_LENGTH):
        action = scheduler.get_action(fetch)
        torch.testing.assert_close(action, chunk[:, k])
    assert fetch_calls == 1

    # After draining the chunk, every env should be flagged as needing a new one.
    assert scheduler.env_requires_new_chunk.all()

    # Next call triggers a new fetch.
    scheduler.get_action(fetch)
    assert fetch_calls == 2


def test_action_chunk_scheduler_reset_marks_all_envs_for_refetch():
    from isaaclab_arena.policy.action_scheduling import ActionChunkScheduler

    scheduler = ActionChunkScheduler(NUM_ENVS, ACTION_CHUNK_LENGTH, ACTION_HORIZON, ACTION_DIM, device="cpu")
    chunk = _make_chunk()

    fetch_calls = 0

    def fetch() -> torch.Tensor:
        nonlocal fetch_calls
        fetch_calls += 1
        return chunk

    # Drive one step so the buffer is populated and indices have advanced.
    scheduler.get_action(fetch)
    assert fetch_calls == 1
    assert not scheduler.env_requires_new_chunk.any()

    scheduler.reset()

    assert scheduler.env_requires_new_chunk.all()
    assert (scheduler.current_action_index == -1).all()
    assert (scheduler.current_action_chunk == 0.0).all()

    # Next get_action must trigger a refetch.
    scheduler.get_action(fetch)
    assert fetch_calls == 2


def test_action_chunk_scheduler_reset_per_env_only_touches_selected_envs():
    from isaaclab_arena.policy.action_scheduling import ActionChunkScheduler

    scheduler = ActionChunkScheduler(NUM_ENVS, ACTION_CHUNK_LENGTH, ACTION_HORIZON, ACTION_DIM, device="cpu")
    chunk = _make_chunk()

    fetch_calls = 0

    def fetch() -> torch.Tensor:
        nonlocal fetch_calls
        fetch_calls += 1
        return chunk

    # Populate both envs through a normal fetch+step.
    scheduler.get_action(fetch)
    assert fetch_calls == 1

    scheduler.reset(torch.tensor([1]))

    # env 0 keeps its (advanced) state; env 1 is flagged for refetch.
    assert scheduler.env_requires_new_chunk.tolist() == [False, True]
    assert scheduler.current_action_index.tolist() == [1, -1]

    # Next call refetches because env 1 needs a chunk; env 0 must not be overwritten.
    env0_buffer_before = scheduler.current_action_chunk[0].clone()
    scheduler.get_action(fetch)
    assert fetch_calls == 2
    torch.testing.assert_close(scheduler.current_action_chunk[0], env0_buffer_before)


# --------------------------- SyncedBatchActionScheduler --------------------------


def test_synced_batch_scheduler_fetches_only_when_all_envs_need_a_chunk():
    from isaaclab_arena.policy.action_scheduling import SyncedBatchActionScheduler

    scheduler = SyncedBatchActionScheduler(NUM_ENVS, ACTION_CHUNK_LENGTH, ACTION_HORIZON, ACTION_DIM, device="cpu")
    chunk = _make_chunk()
    hold = torch.full((NUM_ENVS, ACTION_DIM), -1.0)

    fetch_calls = 0

    def fetch() -> torch.Tensor:
        nonlocal fetch_calls
        fetch_calls += 1
        return chunk

    # First call: every env needs a chunk → exactly one fetch.
    a0 = scheduler.get_action(fetch, hold)
    assert fetch_calls == 1
    torch.testing.assert_close(a0, chunk[:, 0])

    # Subsequent calls within the chunk: no further fetches, just stepping.
    for k in range(1, ACTION_CHUNK_LENGTH):
        a = scheduler.get_action(fetch, hold)
        torch.testing.assert_close(a, chunk[:, k])
    assert fetch_calls == 1

    # Chunk exhausted → next call triggers a new fetch.
    scheduler.get_action(fetch, hold)
    assert fetch_calls == 2


def test_synced_batch_scheduler_holds_waiting_envs_after_partial_reset():
    from isaaclab_arena.policy.action_scheduling import SyncedBatchActionScheduler

    scheduler = SyncedBatchActionScheduler(NUM_ENVS, ACTION_CHUNK_LENGTH, ACTION_HORIZON, ACTION_DIM, device="cpu")
    chunk = _make_chunk()
    hold = torch.tensor([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]])

    fetch_calls = 0

    def fetch() -> torch.Tensor:
        nonlocal fetch_calls
        fetch_calls += 1
        return chunk

    # Bring everyone in sync, take one step.
    scheduler.get_action(fetch, hold)  # fetch happens here
    assert fetch_calls == 1

    # Reset env 1; env 0 still has chunk to play.
    scheduler.reset(torch.tensor([1]))

    a = scheduler.get_action(fetch, hold)
    # No new fetch — env 0 is not yet exhausted, so .all() is False.
    assert fetch_calls == 1
    # env 0 advances to chunk[0, 1]; env 1 is waiting and gets the hold action.
    torch.testing.assert_close(a[0], chunk[0, 1])
    torch.testing.assert_close(a[1], hold[1])
