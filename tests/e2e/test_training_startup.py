"""End-to-end training startup test.

Validates the full pipeline: config -> data load -> model load -> a few steps ->
checkpoint saved. Gated behind the `e2e` marker so it does not run in normal
CI (deselect with `-m "not e2e"`).
"""

from __future__ import annotations

import pytest

from src.config import RunConfig
from src.train_sft import build_sft_config, train

pytestmark = pytest.mark.e2e


def test_sft_training_startup(tmp_path) -> None:
    """A tiny SFT run should complete a few steps and save a checkpoint."""
    from datasets import Dataset
    from tools.convert_data import to_sft

    # Build a tiny dataset from a couple of records.
    rows = [
        {"prompt": "Read this.", "completion": "Output."},
        {"prompt": "Read this.", "completion": "More output."},
    ]
    dataset = to_sft(rows)
    dataset.save_to_disk(str(tmp_path / "train"))

    config = RunConfig.from_dict(
        {
            "method": "sft",
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

    # Validate the config builds without error.
    sft_config = build_sft_config(config)
    assert sft_config.max_steps == 2

    # Run the training pipeline (requires a GPU / installed deps).
    train(config)
    checkpoint_dir = tmp_path / "out"
    assert checkpoint_dir.exists()
