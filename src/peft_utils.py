"""Shared PEFT (LoRA/QLoRA) helpers for voxdoc-trainer."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig

from src.config import ModelConfig
from utils.logging_utils import get_logger

logger = get_logger()


def _matches(name: str, target: str) -> bool:
    """Report whether a module name is selected by a LoRA target pattern.

    Mirrors PEFT's own rule: a target matches a module when it equals the module's
    last component, or the module name ends with that component as a whole one.

    Args:
        name (str): Dotted module name, as returned by ``named_modules``.
        target (str): One entry of ``lora_target_modules``.

    Returns:
        bool: True when PEFT would attach an adapter to this module.
    """
    return name == target or name.endswith(f".{target}")


def count_lora_targets(module_names: Iterable[str], target_modules: Iterable[str]) -> tuple[int, int, list[str], list[str]]:
    """Measure how much of a model a set of LoRA target patterns actually reaches.

    Target lists copied from text-only LLM recipes silently miss most of a
    vision-language or hybrid-attention model, because those models put their
    capacity in modules a q/k/v/o_proj list never names. Counting before training
    turns that silent failure into a number.

    Args:
        module_names (Iterable[str]): Linear module names to test.
        target_modules (Iterable[str]): The configured target patterns.

    Returns:
        tuple[int, int, list[str], list[str]]: How many names matched, how many
            there were in total, the patterns that matched nothing, and a sample
            of up to five unmatched module names.
    """
    targets = list(target_modules)
    names = list(module_names)
    matched = [name for name in names if any(_matches(name, target) for target in targets)]
    unmatched_targets = [target for target in targets if not any(_matches(name, target) for name in names)]
    matched_set = set(matched)
    unmatched_names = [name for name in names if name not in matched_set][:5]
    return len(matched), len(names), unmatched_targets, unmatched_names


def build_lora_config(model_config: ModelConfig) -> LoraConfig:
    """Build a LoraConfig from a ModelConfig.

    Args:
        model_config (ModelConfig): The model configuration.

    Returns:
        LoraConfig: The PEFT LoRA configuration.
    """
    return LoraConfig(
        r=model_config.lora_r,
        lora_alpha=model_config.lora_alpha,
        lora_dropout=model_config.lora_dropout,
        target_modules=model_config.lora_target_modules,
        bias=model_config.lora_bias,
    )


def log_lora_coverage(model: object, model_config: ModelConfig) -> None:
    """Log which fraction of a model's linear layers the LoRA targets reach.

    A target list that matches nothing leaves the adapter attached to no layer at
    all, and one that matches only the attention projections of a hybrid model
    adapts a small minority of its weights; both look like a successful run. This
    warns on the first and reports the share on the second.

    Args:
        model (object): The unwrapped model to inspect.
        model_config (ModelConfig): The configuration whose targets were used.
    """
    named_modules = getattr(model, "named_modules", None)
    if named_modules is None:
        return
    # PEFT keeps the original linear at <name>.base_layer inside a wrapper that is
    # itself an nn.Linear, and adds lora_A/lora_B linears. Reduce to the underlying
    # module names, since those are what the target list was matched against.
    names: list[str] = []
    for name, module in named_modules():
        if not isinstance(module, torch.nn.Linear) or ".lora_" in name:
            continue
        if name.endswith(".base_layer"):
            name = name[: -len(".base_layer")]
        if name not in names:
            names.append(name)
    linear_names = names
    if not linear_names:
        return
    matched, total, unmatched_targets, unmatched_names = count_lora_targets(linear_names, model_config.lora_target_modules)
    logger.info(
        "LoRA targets cover %d/%d linear modules (%.1f%%)",
        matched,
        total,
        100.0 * matched / total,
    )
    if matched == 0:
        logger.warning(
            "No linear module matched lora_target_modules=%s; the adapter trains nothing. " "First linear modules: %s",
            model_config.lora_target_modules,
            linear_names[:5],
        )
        return
    if unmatched_targets:
        logger.warning(
            "lora_target_modules entries matching no linear module: %s",
            ", ".join(unmatched_targets),
        )
    if matched == total:
        return
    logger.info(
        "Unadapted linear modules (showing up to 5 of %d): %s",
        total - matched,
        ", ".join(unmatched_names),
    )


def build_quantization_config(model_config: ModelConfig) -> BitsAndBytesConfig | None:
    """Build a BitsAndBytesConfig for QLoRA, or None if QLoRA is disabled.

    Args:
        model_config (ModelConfig): The model configuration.

    Returns:
        BitsAndBytesConfig | None: The quantization config, or None.
    """
    if not model_config.use_qlora:
        return None
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="bfloat16")
