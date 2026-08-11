"""End-to-end training startup tests for DPO, KTO, and GRPO.

Validates the full pipeline for each method: config -> data load -> model load ->
a few steps -> checkpoint saved. Gated behind the `e2e` marker so they do not
run in normal CI (deselect with `-m "not e2e"`).
"""

from __future__ import annotations

import pytest

from src.config import RunConfig
from src.train_dpo import build_dpo_config, train as train_dpo
from src.train_grpo import build_grpo_config, train as train_grpo
from src.train_kto import build_kto_config, train as train_kto
from tools.convert_data import to_dpo, to_grpo, to_kto

pytestmark = pytest.mark.e2e


def test_dpo_training_startup(tmp_path) -> None:
    """A tiny DPO run should complete a few steps and save a checkpoint."""
    rows = [
        {"prompt": "Read this.", "chosen": "Good answer.", "rejected": "Bad answer."},
        {"prompt": "Read this.", "chosen": "Better answer.", "rejected": "Worse answer."},
    ]
    dataset = to_dpo(rows)
    dataset.save_to_disk(str(tmp_path / "train"))

    config = RunConfig.from_dict(
        {
            "method": "dpo",
            "model": {"model_id": "Qwen/Qwen3.5-0.8B", "lora_r": 4},
            "data": {"train_path": str(tmp_path / "train")},
            "training": {
                "output_dir": str(tmp_path / "out"),
                "dry_run": True,
                "dry_run_steps": 2,
                "save_steps": 1,
                "save_total_limit": 2,
                "report_to": [],
            },
        }
    )

    dpo_config = build_dpo_config(config)
    assert dpo_config.max_steps == 2

    train_dpo(config)
    assert (tmp_path / "out").exists()


def test_kto_training_startup(tmp_path) -> None:
    """A tiny KTO run should complete a few steps and save a checkpoint."""
    rows = [
        {"prompt": "Read this.", "completion": "Good answer.", "label": True},
        {"prompt": "Read this.", "completion": "Bad answer.", "label": False},
    ]
    dataset = to_kto(rows)
    dataset.save_to_disk(str(tmp_path / "train"))

    config = RunConfig.from_dict(
        {
            "method": "kto",
            "model": {"model_id": "Qwen/Qwen3.5-0.8B", "lora_r": 4},
            "data": {"train_path": str(tmp_path / "train")},
            "training": {
                "output_dir": str(tmp_path / "out"),
                "per_device_train_batch_size": 2,
                "dry_run": True,
                "dry_run_steps": 2,
                "save_steps": 1,
                "save_total_limit": 2,
                "report_to": [],
            },
        }
    )

    kto_config = build_kto_config(config)
    assert kto_config.max_steps == 2

    train_kto(config)
    assert (tmp_path / "out").exists()


def test_grpo_training_startup(tmp_path) -> None:
    """A tiny GRPO run should complete a few steps and save a checkpoint."""
    rows = [
        {"prompt": "Read this."},
        {"prompt": "Read this."},
    ]
    dataset = to_grpo(rows)
    dataset.save_to_disk(str(tmp_path / "train"))

    config = RunConfig.from_dict(
        {
            "method": "grpo",
            "model": {"model_id": "Qwen/Qwen3.5-0.8B", "lora_r": 4},
            "data": {"train_path": str(tmp_path / "train")},
            "training": {
                "output_dir": str(tmp_path / "out"),
                "per_device_train_batch_size": 2,
                "dry_run": True,
                "dry_run_steps": 2,
                "save_steps": 1,
                "save_total_limit": 2,
                "report_to": [],
            },
            "grpo": {"num_generations": 2, "max_completion_length": 32},
        }
    )

    grpo_config = build_grpo_config(config)
    assert grpo_config.max_steps == 2

    train_grpo(config)
    assert (tmp_path / "out").exists()
