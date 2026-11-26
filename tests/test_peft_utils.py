"""Unit tests for PEFT adapter helpers."""

from __future__ import annotations

import logging

import torch.nn as nn

from src.config import ModelConfig
from src.peft_utils import count_lora_targets, log_lora_coverage


class _Hybrid(nn.Module):
    """A miniature of the Qwen3.5 block layout: delta-net, sparse attention, MLP, vision."""

    def __init__(self) -> None:
        super().__init__()
        for layer in range(4):
            block = nn.ModuleDict(
                {
                    "linear_attn": nn.ModuleDict(
                        {name: nn.Linear(8, 8) for name in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj")}
                    ),
                    "mlp": nn.ModuleDict({name: nn.Linear(8, 8) for name in ("gate_proj", "up_proj", "down_proj")}),
                }
            )
            if layer % 4 == 3:
                block["self_attn"] = nn.ModuleDict({name: nn.Linear(8, 8) for name in ("q_proj", "k_proj", "v_proj", "o_proj")})
            self.add_module(f"layer{layer}", block)
        self.vision = nn.ModuleDict(
            {
                "blocks0": nn.ModuleDict(
                    {
                        "attn": nn.ModuleDict({"qkv": nn.Linear(8, 8), "proj": nn.Linear(8, 8)}),
                        "mlp": nn.ModuleDict({"linear_fc1": nn.Linear(8, 8), "linear_fc2": nn.Linear(8, 8)}),
                    }
                ),
                "merger": nn.ModuleDict({"linear_fc1": nn.Linear(8, 8), "linear_fc2": nn.Linear(8, 8)}),
            }
        )
        self.lm_head = nn.Linear(8, 8)


def _linear_names(model: nn.Module) -> list[str]:
    """Return the dotted names of every linear layer in ``model``.

    Args:
        model (nn.Module): The module to walk.

    Returns:
        list[str]: One name per linear layer.
    """
    return [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]


def test_default_targets_reach_every_linear_but_the_head() -> None:
    """The shipped target list adapts the delta-net, attention, MLP and vision projections."""
    matched, total, unmatched_targets, unmatched_names = count_lora_targets(
        _linear_names(_Hybrid()), ModelConfig().lora_target_modules
    )
    assert total == 43
    assert matched == total - 1  # only the language-modelling head is left alone
    assert unmatched_names == ["lm_head"]
    assert unmatched_targets == []


def test_attention_only_targets_leave_most_layers_unadapted(caplog) -> None:
    """The historical attention-only list adapts a tenth of the model and must be reported."""
    model = _Hybrid()
    config = ModelConfig(lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    with caplog.at_level(logging.INFO, logger="voxdoc-trainer.src.peft_utils"):
        log_lora_coverage(model, config)
    matched, total, _, _ = count_lora_targets(_linear_names(model), config.lora_target_modules)
    assert (matched, total) == (4, 43)
    message = " ".join(record.message for record in caplog.records)
    assert "4/43" in message
    assert "in_proj_qkv" in message  # an unadapted module is named so the gap is actionable


def test_no_matching_targets_is_reported(caplog) -> None:
    """A target list that matches nothing is surfaced loudly, not silently trained."""
    with caplog.at_level(logging.INFO, logger="voxdoc-trainer.src.peft_utils"):
        log_lora_coverage(_Hybrid(), ModelConfig(lora_target_modules=["not_a_real_module"]))
    message = " ".join(record.message for record in caplog.records)
    assert "0/43" in message
    assert "not_a_real_module" in message


def test_targets_are_deduplicated() -> None:
    """A repeated pattern does not count a module twice."""
    names = ["model.layers.0.mlp.gate_proj", "model.layers.1.mlp.gate_proj"]
    once = count_lora_targets(names, ["gate_proj"])
    twice = count_lora_targets(names, ["gate_proj", "gate_proj"])
    assert once == twice
    assert once[0] == 2


def test_log_lora_coverage_ignores_lora_wrapper_linears(caplog) -> None:
    """LoRA A/B matrices added by PEFT are not themselves counted as targets."""
    model = _Hybrid()
    model.layer0.mlp.gate_proj = nn.ModuleDict(
        {"base_layer": nn.Linear(8, 8), "lora_A": nn.Linear(8, 2), "lora_B": nn.Linear(2, 8)}
    )
    with caplog.at_level(logging.INFO, logger="voxdoc-trainer.src.peft_utils"):
        log_lora_coverage(model, ModelConfig())
    message = " ".join(record.message for record in caplog.records)
    # The wrapped linear is still one target, and its lora_A/lora_B are not counted.
    assert "42/43" in message


def test_log_lora_coverage_tolerates_a_model_without_linears(caplog) -> None:
    """A model with nothing to adapt is skipped rather than dividing by zero."""
    with caplog.at_level(logging.WARNING, logger="voxdoc-trainer.src.peft_utils"):
        log_lora_coverage(nn.ReLU(), ModelConfig())
    assert caplog.records == []


def test_log_lora_coverage_skips_objects_without_named_modules(caplog) -> None:
    """A non-module object (e.g. a config) is ignored rather than crashing the run."""
    with caplog.at_level(logging.WARNING, logger="voxdoc-trainer.src.peft_utils"):
        log_lora_coverage(object(), ModelConfig())
    assert caplog.records == []
