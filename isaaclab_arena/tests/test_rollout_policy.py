# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import io
import torch
from contextlib import closing
from types import SimpleNamespace

import pyarrow.parquet as pq

from isaaclab_arena.evaluation import policy_runner
from isaaclab_arena.evaluation.action_parquet_recorder import ActionParquetRecorder


class _RolloutEnvironment:
    """Return scheduled completion events without simulating episode scheduling."""

    def __init__(self, step_results):
        self.cfg = SimpleNamespace(metrics=None)
        self.step_results = iter(step_results)
        self.configured_episode_limits = []
        self.reset_called = False
        self.steps_completed = 0
        self.completed_episode_count = 0
        self.reset_env_ids = torch.empty(0, dtype=torch.long)
        self.active_episode_mask = torch.ones(3, dtype=torch.bool)
        self.episode_indices = [0, 0, 0]

    def get_episode_index(self, env_id):
        return self.episode_indices[env_id]

    def configure_episode_limit(self, num_episodes):
        assert not self.reset_called
        self.configured_episode_limits.append(num_episodes)

    def reset(self):
        self.reset_called = True
        return {"step": 0}, {}

    def get_language_instruction(self):
        return "Complete the task."

    def step(self, actions):
        completed_count, reset_env_ids, terminated_env_ids, truncated_env_ids = next(self.step_results)
        self.steps_completed += 1
        self.completed_episode_count = completed_count
        self.reset_env_ids = torch.tensor(reset_env_ids, dtype=torch.long)
        terminated = torch.zeros(3, dtype=torch.bool)
        truncated = torch.zeros(3, dtype=torch.bool)
        terminated[terminated_env_ids] = True
        truncated[truncated_env_ids] = True
        self.active_episode_mask[terminated | truncated] = False
        self.active_episode_mask[self.reset_env_ids] = True
        for env_id in reset_env_ids:
            self.episode_indices[env_id] += 1
        return {"step": self.steps_completed}, None, terminated, truncated, {}


class _Policy:
    def __init__(self):
        self.reset_calls = []
        self.observed_steps = []

    def reset(self, env_ids=None):
        self.reset_calls.append(None if env_ids is None else env_ids.tolist())

    def set_task_description(self, task_description):
        self.task_description = task_description

    def get_action(self, env, observation):
        self.observed_steps.append(observation["step"])
        return torch.zeros(3, 1)


def _wrap_environment(environment):
    return SimpleNamespace(unwrapped=environment, reset=environment.reset, step=environment.step)


def _capture_progress(monkeypatch, total):
    progress_bar = policy_runner.tqdm.tqdm(total=total, file=io.StringIO())
    monkeypatch.setattr(policy_runner.tqdm, "tqdm", lambda **kwargs: progress_bar)
    return progress_bar


def test_episode_rollout_waits_for_every_completion_and_resets_only_replacements(monkeypatch):
    environment = _RolloutEnvironment([
        (2, [0], [0, 1], []),
        (2, [], [], []),
        (3, [], [0], []),
        (4, [], [], [2]),
    ])
    policy = _Policy()
    progress_bar = _capture_progress(monkeypatch, total=4)

    metrics = policy_runner.rollout_policy(_wrap_environment(environment), policy, num_steps=None, num_episodes=4)

    assert metrics is None
    assert environment.configured_episode_limits == [4]
    assert environment.steps_completed == 4
    assert policy.reset_calls == [None, [0]]
    assert policy.observed_steps == [0, 1, 2, 3]
    assert policy.task_description == "Complete the task."
    assert progress_bar.n == 4


def test_step_rollout_keeps_its_step_limit_and_resets_actual_reset_environments(monkeypatch):
    environment = _RolloutEnvironment([
        (1, [1], [1], []),
        (1, [], [], []),
    ])
    policy = _Policy()
    progress_bar = _capture_progress(monkeypatch, total=2)

    policy_runner.rollout_policy(_wrap_environment(environment), policy, num_steps=2, num_episodes=None)

    assert environment.configured_episode_limits == []
    assert environment.steps_completed == 2
    assert policy.reset_calls == [None, [1]]
    assert policy.observed_steps == [0, 1]
    assert progress_bar.n == 2


def test_rollout_smooths_all_action_dimensions_and_resets_ema_history_per_environment(monkeypatch):
    environment = _RolloutEnvironment([
        (0, [], [], []),
        (1, [1], [1], []),
        (1, [], [], []),
    ])
    policy = _Policy()
    raw_actions = iter([torch.full((3, 16), value) for value in (2.0, 12.0, 22.0)])
    monkeypatch.setattr(policy, "get_action", lambda env, obs: next(raw_actions))
    env = _wrap_environment(environment)
    received_actions = []

    def step_and_mutate_actions(actions):
        received_actions.append(actions.clone())
        actions.zero_()
        return environment.step(actions)

    monkeypatch.setattr(env, "step", step_and_mutate_actions)
    policy_runner.rollout_policy(env, policy, num_steps=3, num_episodes=None)

    received = torch.stack(received_actions)
    expected = torch.tensor([[2.0, 2.0, 2.0], [8.0, 8.0, 8.0], [16.4, 22.0, 16.4]])
    torch.testing.assert_close(received, expected.unsqueeze(-1).expand_as(received))


def test_rollout_filters_chunks_before_ema_and_keeps_ema_across_chunks(monkeypatch):
    environment = _RolloutEnvironment([(0, [], [], [])] * 3)
    env = _wrap_environment(environment)
    policy = _Policy()
    policy.action_chunk_transform = None
    policy.action_chunk_step_limit = None
    impulse = torch.arange(1.0, 7.0).reshape(3, 2)
    chunks = iter([torch.stack((torch.zeros_like(impulse), impulse), dim=1), torch.zeros(3, 1, 2)])
    pending_actions = iter(())
    received = []

    def get_action(env, obs):
        nonlocal pending_actions
        assert policy.action_chunk_step_limit == 8
        action = next(pending_actions, None)
        if action is None:
            chunk = policy.action_chunk_transform(next(chunks))
            pending_actions = iter(chunk.unbind(dim=1))
            action = next(pending_actions)
        return action

    def record_step(actions):
        received.append(actions.clone())
        return environment.step(actions)

    monkeypatch.setattr(policy, "get_action", get_action)
    monkeypatch.setattr(env, "step", record_step)
    policy_runner.rollout_policy(env, policy, num_steps=3, num_episodes=None)

    # For a two-step chunk [0, x], endpoint padding puts 0.29869*x at the first step.
    first = 0.29869003 * impulse
    second = 0.4 * first + 0.6 * (impulse - first)
    torch.testing.assert_close(torch.stack(received), torch.stack((first, second, 0.4 * second)))
    assert policy.action_chunk_transform is None
    assert policy.action_chunk_step_limit is None


def test_rollout_cli_can_disable_all_action_smoothing(monkeypatch):
    parser = argparse.ArgumentParser()
    policy_runner.add_policy_runner_arguments(parser)
    assert parser.parse_args([]).action_smoothing is True
    assert parser.parse_args(["--action_smoothing"]).action_smoothing is True
    args = parser.parse_args(["--no-action_smoothing"])
    assert args.action_smoothing is False

    environment = _RolloutEnvironment([(0, [], [], [])] * 2)
    env = _wrap_environment(environment)
    policy = _Policy()

    def original_transform(chunk):
        return chunk + 1

    policy.action_chunk_transform = original_transform
    policy.action_chunk_step_limit = 4
    received = []

    def get_action(env, obs):
        assert policy.action_chunk_transform is None
        assert policy.action_chunk_step_limit is None
        return torch.full((3, 1), 10.0 * obs["step"])

    def record_step(actions):
        received.append(actions.clone())
        return environment.step(actions)

    monkeypatch.setattr(policy, "get_action", get_action)
    monkeypatch.setattr(env, "step", record_step)
    policy_runner.rollout_policy(env, policy, num_steps=2, num_episodes=None, action_smoothing=args.action_smoothing)

    torch.testing.assert_close(torch.stack(received), torch.tensor([0.0, 10.0]).reshape(2, 1, 1).expand(2, 3, 1))
    assert policy.action_chunk_transform is original_transform
    assert policy.action_chunk_step_limit == 4


def test_rollout_records_smoothed_actions_in_separate_episode_files(tmp_path, monkeypatch):
    environment = _RolloutEnvironment([
        (2, [0], [0, 1], []),
        (2, [], [], []),
        (3, [], [], [2]),
        (4, [], [0], []),
    ])
    policy = _Policy()
    actions = torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    monkeypatch.setattr(policy, "get_action", lambda env, obs: actions + obs["step"])
    env = _wrap_environment(environment)

    def step_and_mutate_actions(actions):
        actions.zero_()
        return environment.step(actions)

    monkeypatch.setattr(env, "step", step_and_mutate_actions)
    output_dir = tmp_path / "outputs"
    with closing(ActionParquetRecorder(output_dir, rank=2, steps_per_batch=2)) as recorder:
        policy_runner.rollout_policy(env, policy, num_steps=None, num_episodes=4, action_recorder=recorder)
        # Completed files are readable before the recorder closes.
        expected = {
            "actions_rank2_env0_episode0.parquet": (0, [[0.0, 1.0]]),
            "actions_rank2_env0_episode1.parquet": (0, [[1.0, 2.0], [1.6, 2.6], [2.44, 3.44]]),
            "actions_rank2_env1_episode0.parquet": (1, [[2.0, 3.0]]),
            "actions_rank2_env2_episode0.parquet": (2, [[4.0, 5.0], [4.6, 5.6], [5.44, 6.44]]),
        }
        assert {path.name for path in output_dir.glob("*.parquet")} == expected.keys()
        for filename, (env_id, actions) in expected.items():
            with pq.ParquetFile(output_dir / filename) as parquet:
                assert parquet.num_row_groups == (len(actions) + 1) // 2
                table = parquet.read().to_pydict()
                assert table["step"] == list(range(len(actions)))
                assert table["env_id"] == [env_id] * len(actions)
                torch.testing.assert_close(torch.tensor(table["action"]), torch.tensor(actions))
