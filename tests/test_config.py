"""Unit tests for the config system."""

from __future__ import annotations

from src.config import ModelConfig, RunConfig, load_config


def test_default_config() -> None:
    """A default RunConfig should use the target model and SFT method."""
    config = RunConfig()
    assert config.method == "sft"
    assert config.model.model_id == "Qwen/Qwen3.5-9B"


def test_from_dict_ignores_unknown_keys() -> None:
    """Unknown keys in a dict should be ignored."""
    config = RunConfig.from_dict({"method": "dpo", "unknown_key": 123})
    assert config.method == "dpo"


def test_from_dict_nested() -> None:
    """Nested dicts should populate sub-configs."""
    config = RunConfig.from_dict({"model": {"model_id": "Qwen/Qwen3.8-9B", "lora_r": 8}})
    assert config.model.model_id == "Qwen/Qwen3.8-9B"
    assert config.model.lora_r == 8


def test_model_swap_via_config() -> None:
    """Changing the model id in config should not require code changes."""
    config = RunConfig.from_dict({"model": {"model_id": "Qwen/Qwen3.8-9B"}})
    assert config.model.model_id == "Qwen/Qwen3.8-9B"
    assert isinstance(config.model, ModelConfig)


def test_load_config_none_returns_defaults() -> None:
    """load_config(None) should return a default RunConfig."""
    config = load_config(None)
    assert isinstance(config, RunConfig)
