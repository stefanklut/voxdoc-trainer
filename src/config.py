"""Configuration system for voxdoc-trainer.

Model-agnostic, YAML-driven configuration. The model id lives in config so new
models (e.g. Qwen3.8+) can be swapped in without code changes.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    """Configuration for the base model and PEFT adapters."""

    model_id: str = "Qwen/Qwen3.5-9B"
    trust_remote_code: bool = False
    dtype: str = "bfloat16"
    use_qlora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    lora_bias: str = "none"
    router_aux_loss_coef: float = 0.001

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ModelConfig":
        """Build a ModelConfig from a dict, ignoring unknown keys."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})


@dataclass
class DataConfig:
    """Configuration for dataset paths and preprocessing."""

    train_path: str = "data/train.jsonl"
    eval_path: str | None = None
    max_length: int | None = None
    packing: bool = False
    dataset_num_proc: int = 4
    shuffle: bool = True
    pad_to_multiple_of: int | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DataConfig":
        """Build a DataConfig from a dict, ignoring unknown keys."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})


@dataclass
class TrainingConfig:
    """Configuration for the training run (TRL/transformers native args)."""

    output_dir: str = "outputs/run"
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    num_train_epochs: float = 3.0
    max_steps: int = -1
    learning_rate: float = 1e-4
    lr_scheduler_type: str = "linear"
    warmup_steps: int = 0
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    bf16: bool = True
    fp16: bool = False
    logging_steps: int = 10
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int | None = 3
    eval_strategy: str = "no"
    eval_steps: int | None = None
    report_to: list[str] = field(default_factory=lambda: ["tensorboard"])
    seed: int = 42
    dry_run: bool = False
    dry_run_steps: int = 2
    # Assistant-only loss is not supported for vision-language models in TRL.
    # Keep False for VLMs (the default target); set True only for text-only SFT.
    assistant_only_loss: bool = False
    # Precomputing reference log probs is not supported for vision datasets in TRL.
    # Keep False for VLMs; set True only for text-only DPO.
    precompute_ref_log_probs: bool = False

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "TrainingConfig":
        """Build a TrainingConfig from a dict, ignoring unknown keys."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})


@dataclass
class GRPOConfig:
    """Configuration specific to GRPO (online RL)."""

    num_generations: int = 8
    max_completion_length: int = 512
    temperature: float = 1.0
    use_vllm: bool = False
    vllm_mode: str = "colocate"
    beta: float = 0.0
    loss_type: str = "dapo"
    log_completions: bool = False

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "GRPOConfig":
        """Build a GRPOConfig from a dict, ignoring unknown keys."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})


@dataclass
class RunConfig:
    """Top-level configuration bundling all sub-configs."""

    method: str = "sft"  # one of: sft, dpo, kto, grpo
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunConfig":
        """Load a RunConfig from a YAML file."""
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RunConfig":
        """Build a RunConfig from a nested dict, ignoring unknown keys."""
        known = {f.name: f for f in fields(cls)}
        kwargs: dict[str, Any] = {}
        for key, value in raw.items():
            if key not in known:
                continue
            if isinstance(value, dict):
                type_name = known[key].type
                sub = getattr(sys.modules[cls.__module__], type_name)
                kwargs[key] = sub.from_dict(value)
            else:
                kwargs[key] = value
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the config to a nested dict."""
        return asdict(self)


def load_config(path: str | Path | None) -> RunConfig:
    """Load a RunConfig from a YAML path, or return defaults if None."""
    if path is None:
        return RunConfig()
    return RunConfig.from_yaml(path)
