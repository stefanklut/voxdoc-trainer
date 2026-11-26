"""Tests for the checks that run before GRPO loads a model."""

from __future__ import annotations

import pytest
from datasets import Dataset

from src.config import RunConfig
from src.train_grpo import check_generation_batch, check_reward_columns


def _config(**training: object) -> RunConfig:
    """Build a GRPO RunConfig with the given training overrides.

    Args:
        **training: Keys placed in the ``training`` section.

    Returns:
        RunConfig: A config that would otherwise pass the batch check.
    """
    raw: dict = {"method": "grpo", "training": {"per_device_train_batch_size": 2, **training}}
    return RunConfig.from_dict(raw)


def test_generation_batch_accepts_whole_groups() -> None:
    """2 * 4 == 8 completions is exactly one group of num_generations=8."""
    check_generation_batch(_config(gradient_accumulation_steps=4))


def test_generation_batch_rejects_partial_groups() -> None:
    """A generation batch that cannot hold whole groups is rejected, with the fix spelled out."""
    with pytest.raises(ValueError, match="not divisible by num_generations"):
        check_generation_batch(_config(gradient_accumulation_steps=1))


def test_generation_batch_suggests_the_smallest_fix() -> None:
    """The message names the smallest gradient_accumulation_steps that works."""
    with pytest.raises(ValueError, match="multiple of 4"):
        check_generation_batch(_config(gradient_accumulation_steps=1))


def test_reward_columns_reject_a_missing_ground_truth() -> None:
    """A dataset without the column the CER reward needs fails before the model loads."""
    from src.rewards_pretrain import cer_reward

    dataset = Dataset.from_list([{"prompt": "Read this.", "solution": "text"}])
    with pytest.raises(ValueError, match="cer_reward\\(\\) needs \\['ground_truth'\\]"):
        check_reward_columns(dataset, [cer_reward], "data/grpo")


def test_reward_columns_accept_ground_truth() -> None:
    """The documented GRPO row shape passes."""
    from src.rewards import accuracy_reward, format_reward
    from src.rewards_pretrain import cer_reward

    dataset = Dataset.from_list([{"prompt": "Read this.", "ground_truth": "text", "images": ["p.jpg"]}])
    check_reward_columns(dataset, [cer_reward], "data/grpo")
    check_reward_columns(dataset, [format_reward, accuracy_reward], "data/grpo")


def test_reward_columns_tolerate_a_dataset_without_column_names() -> None:
    """An object that does not advertise columns is skipped rather than rejected."""
    from src.rewards_pretrain import cer_reward

    check_reward_columns(object(), [cer_reward], "data/grpo")
