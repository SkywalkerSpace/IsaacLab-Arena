# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gymnasium as gym
import torch
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from gymnasium.spaces.dict import Dict as GymSpacesDict
from typing import Generic, TypeVar


@dataclass
class PolicyCfg:
    """Mark a typed Arena policy configuration."""


PolicyCfgT = TypeVar("PolicyCfgT", bound=PolicyCfg)


class PolicyBase(ABC, Generic[PolicyCfgT]):
    """Define runtime behavior shared by Arena policies."""

    action_chunk_transform: Callable[[torch.Tensor], torch.Tensor] | None = None
    """Optional rollout transform that chunked policies apply once per fetched chunk."""

    action_chunk_step_limit: int | None = None
    """Optional cap on executed steps per prediction; blend with any unexecuted tail."""

    action_chunk_crossfade_dim_limit: int | None = None
    """Only crossfade this many leading action dimensions; None blends all dimensions."""

    def __init__(self, config: PolicyCfgT):
        self.config = config

    @abstractmethod
    def get_action(self, env: gym.Env, observation: GymSpacesDict) -> torch.Tensor:
        """
        Compute an action given the environment and observation.

        Args:
            env: The environment instance.
            observation: Observation dictionary from the environment.

        Returns:
            torch.Tensor: The action to take.
        """
        raise NotImplementedError("Function not implemented yet.")

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """
        Reset the policy.
        """
        pass

    def close(self) -> None:
        """Release resources held by the policy."""
        pass

    def set_task_description(self, task_description: str | None) -> str:
        """Set the task description of the task being evaluated."""
        self.task_description = task_description
        return self.task_description

    def has_length(self) -> bool:
        """Check if the policy is based on a recording (i.e. is a dataset-driven policy)."""
        return False

    def length(self) -> int | None:
        """Get the length of the policy (for dataset-driven policies)."""
        pass

    @property
    def is_remote(self) -> bool:
        """Check if policy is run remotely."""
        return False
