# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch


def crossfade_action_chunks(
    previous_chunk: torch.Tensor, new_chunk: torch.Tensor, executed_steps: int, steps_per_chunk: int
) -> torch.Tensor:
    """Blend aligned predictions over the next execution window, leaving the new tail intact."""
    old_tail = previous_chunk[..., executed_steps:, :]
    overlap = min(old_tail.shape[-2], new_chunk.shape[-2], steps_per_chunk)
    if overlap == 0:
        return new_chunk

    weights: torch.Tensor | float
    if overlap == 1:
        weights = 0.5
    else:
        weights = torch.linspace(0.0, 1.0, overlap, dtype=new_chunk.dtype, device=new_chunk.device).unsqueeze(-1)
    blended = torch.lerp(old_tail[..., :overlap, :], new_chunk[..., :overlap, :], weights)
    return torch.cat((blended, new_chunk[..., overlap:, :]), dim=-2)
