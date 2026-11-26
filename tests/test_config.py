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


def test_default_lora_targets_cover_hybrid_models() -> None:
    """The default LoRA targets must reach the linear-attention and MLP layers too.

    Hybrid models such as Qwen3.5-27B put only a quarter of their layers behind
    ``*_proj`` attention, so an attention-only target list adapts ~10% of the
    model's linear layers.
    """
    targets = ModelConfig().lora_target_modules
    assert {"in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"} <= set(targets)
    assert {"gate_proj", "up_proj", "down_proj"} <= set(targets)
    assert {"q_proj", "k_proj", "v_proj", "o_proj"} <= set(targets)


def test_data_config_has_pixel_cap() -> None:
    """Datasets default to a bounded image size so a 20 megapixel scan cannot explode the sequence."""
    from utils.image_loading import DEFAULT_MAX_PIXELS

    assert RunConfig().data.max_pixels == DEFAULT_MAX_PIXELS
    assert RunConfig.from_dict({"data": {"max_pixels": 250_000}}).data.max_pixels == 250_000
    assert RunConfig.from_dict({"data": {"max_pixels": None}}).data.max_pixels is None


def test_from_dict_coerces_scientific_notation_strings() -> None:
    """YAML 1.1 parses ``1e-4`` as a string; the config must still read it as a number."""
    config = RunConfig.from_dict({"training": {"learning_rate": "1e-4", "max_steps": "-1"}, "data": {"max_pixels": "1003520"}})
    assert config.training.learning_rate == 1e-4
    assert isinstance(config.training.learning_rate, float)
    assert config.training.max_steps == -1
    assert isinstance(config.training.max_steps, int)
    assert config.data.max_pixels == 1_003_520


def test_from_dict_rejects_non_numeric_string() -> None:
    """A value that cannot be read as the field's numeric type is rejected at load time."""
    import pytest

    with pytest.raises(ValueError, match="training.learning_rate"):
        RunConfig.from_dict({"training": {"learning_rate": "fast"}})


def test_from_dict_warns_on_unknown_keys(caplog) -> None:
    """A typo'd config key is reported instead of being silently dropped."""
    import logging

    with caplog.at_level(logging.WARNING, logger="voxdoc-trainer.src.config"):
        RunConfig.from_dict({"method": "sft", "tarining": {"learning_rate": 1e-4}})
        RunConfig.from_dict({"training": {"leanring_rate": 1e-4}})
    messages = " ".join(record.message for record in caplog.records)
    assert "tarining" in messages
    assert "leanring_rate" in messages
