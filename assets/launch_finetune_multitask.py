# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Fine-tune one GR00T N1.6 policy on multiple same-embodiment LeRobot datasets.

Run with the Isaac-GR00T Python environment and make its checkout importable
(e.g. PYTHONPATH="$GROOT_ROOT${PYTHONPATH:+:$PYTHONPATH}"). Relative paths are
resolved from the current working directory, not from this script's directory. No GR00T source files
are modified. Model/training defaults mirror gr00t/experiment/launch_finetune.py
in the Isaac-GR00T-e29d8f version; CLI training options reuse FinetuneConfig.

Each path becomes a separate mixture entry. Equal weights (the default) give
entries equal target sampling weight, independently of dataset length. Weights
are relative, not percentages, and describe the shard sampler rather than an
exact per-batch quota. Keep task-specific language annotations in each dataset.
Datasets must share the selected embodiment and compatible camera/state/action
schemas. Do not use merge_lerobot_datasets.py to merge different tasks.
"""

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import tyro

from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.experiment.launch_finetune import load_modality_config


@dataclass(kw_only=True)
class MultiTaskFinetuneConfig(FinetuneConfig):
    """Original fine-tuning options with multiple dataset paths and weights."""

    dataset_path: tyro.conf.Suppress[str] = field(default="", kw_only=True)
    """Unused single-dataset option; use --dataset-paths instead."""

    dataset_paths: list[str]
    """LeRobot root directories, in the same order as --dataset-weights."""

    dataset_weights: list[float] | None = None
    """Positive relative sampling weights; defaults to one per dataset."""

    dry_run: bool = False
    """Print resolved fine-tuning options and mixture without starting training."""


def build_dataset_specs(dataset_paths, dataset_weights, embodiment_tag):
    """Build one validated mixture entry per dataset, preserving input order."""
    if len(dataset_paths) < 2:
        raise ValueError("Multi-task training requires at least two --dataset-paths.")
    weights = [1.0] * len(dataset_paths) if dataset_weights is None else dataset_weights
    if len(weights) != len(dataset_paths):
        raise ValueError("--dataset-weights must have one value per --dataset-paths entry.")
    if any(not math.isfinite(weight) or weight <= 0 for weight in weights):
        raise ValueError("Dataset weights must be finite and strictly positive.")

    datasets = []
    seen = set()
    for raw_path, weight in zip(dataset_paths, weights):
        path = Path(raw_path).expanduser().resolve()
        if path in seen:
            raise ValueError(f"Duplicate dataset path: {path}")
        seen.add(path)
        if not path.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {path}")
        for filename in ("info.json", "tasks.jsonl", "modality.json"):
            if not (path / "meta" / filename).is_file():
                raise FileNotFoundError(f"Missing LeRobot metadata: {path / 'meta' / filename}")
        datasets.append(
            {
                "dataset_paths": [str(path)],
                "mix_ratio": weight,
                "embodiment_tag": embodiment_tag,
            }
        )
    return datasets


def build_training_config(ft_config, datasets):
    """Apply the upstream N1.6 fine-tuning settings to a multi-dataset config."""
    from gr00t.configs.base_config import get_default_config

    # Register the external schema before constructing the default data config.
    if ft_config.modality_config_path is not None:
        load_modality_config(str(Path(ft_config.modality_config_path).expanduser().resolve()))
    config = get_default_config().load_dict(
        {"data": {"download_cache": False, "datasets": datasets}}
    )
    config.load_config_path = None

    for name in (
        "tune_llm",
        "tune_visual",
        "tune_projector",
        "tune_diffusion_model",
        "state_dropout_prob",
        "random_rotation_angle",
        "color_jitter_params",
    ):
        setattr(config.model, name, getattr(ft_config, name))
    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.eagle_collator = True
    config.model.model_name = "nvidia/Eagle-Block2A-2B-v2"
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True

    config.training.start_from_checkpoint = ft_config.base_model_path
    config.training.optim = "adamw_torch"
    for name in (
        "global_batch_size",
        "dataloader_num_workers",
        "learning_rate",
        "gradient_accumulation_steps",
        "output_dir",
        "save_steps",
        "save_total_limit",
        "num_gpus",
        "use_wandb",
        "max_steps",
        "weight_decay",
        "warmup_ratio",
    ):
        setattr(config.training, name, getattr(ft_config, name))
    config.training.wandb_project = "finetune-gr00t-n1d6"
    for name in ("shard_size", "episode_sampling_rate", "num_shards_per_epoch"):
        setattr(config.data, name, getattr(ft_config, name))
    return config


def main():
    """Parse fine-tuning options and launch a single mixed-task training run."""
    os.environ.setdefault("LOGURU_LEVEL", "INFO")
    ft_config = tyro.cli(MultiTaskFinetuneConfig, description=__doc__)
    datasets = build_dataset_specs(
        ft_config.dataset_paths, ft_config.dataset_weights, ft_config.embodiment_tag.value
    )
    config = build_training_config(ft_config, datasets)
    if ft_config.dry_run:
        # No dataset loading/statistics generation, model loading, or output writes.
        from dataclasses import asdict

        print(json.dumps({"finetune": asdict(ft_config), "datasets": datasets}, indent=2, default=str))
        print("Dry run only: dataset contents and model compatibility have not been validated.")
        return

    from gr00t.experiment.experiment import run

    run(config)


if __name__ == "__main__":
    main()
