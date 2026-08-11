"""Shared PEFT (LoRA/QLoRA) helpers for voxdoc-trainer."""

from __future__ import annotations

from transformers import BitsAndBytesConfig
from peft import LoraConfig

from src.config import ModelConfig


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
