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

from utils.image_loading import DEFAULT_MAX_PIXELS
from utils.logging_utils import get_logger

logger = get_logger()

#: Numeric field types that :func:`_build_kwargs` coerces a YAML string back into.
_NUMERIC_TYPES = {"float": float, "int": int}


def _numeric_type(field_type: str) -> type | None:
    """Return the numeric type a dataclass field annotation asks for.

    Annotations are read as text (the module uses ``from __future__ import
    annotations``), so both ``"float"`` and the optional spelling
    ``"float | None"`` have to be recognised.

    Args:
        field_type (str): The field's annotation as written in the source.

    Returns:
        type | None: :class:`float` or :class:`int`, or None when the field is not
            numeric.
    """
    for part in field_type.replace("Optional[", "").replace("]", "").split("|"):
        numeric = _NUMERIC_TYPES.get(part.strip())
        if numeric is not None:
            return numeric
    return None


def _build_kwargs(cls: type, raw: dict[str, Any], context: str) -> dict[str, Any]:
    """Select the known fields of a dataclass from a raw mapping, coercing numbers.

    YAML 1.1 only recognises scientific notation when it carries a decimal point,
    so a learning rate written as ``1e-4`` arrives as the *string* ``"1e-4"`` and
    would otherwise reach the optimizer as a string. Such values are converted here
    so a config reads the way it looks, and a numeric field holding something that
    is not a number is rejected at load time rather than deep inside a trainer.

    Args:
        cls (type): The dataclass being built.
        raw (dict[str, Any]): The raw mapping parsed from YAML.
        context (str): Section name used when reporting unknown keys.

    Returns:
        dict[str, Any]: Keyword arguments accepted by ``cls``.

    Raises:
        ValueError: If a numeric field is given a value that is not a number.
    """
    known = {f.name: f for f in fields(cls)}
    kwargs: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in known:
            logger.warning("Ignoring unknown key '%s' in config section '%s'", key, context)
            continue
        numeric = _numeric_type(str(known[key].type))
        if numeric is not None and isinstance(value, str):
            try:
                value = numeric(value)
            except ValueError:
                raise ValueError(f"Config '{context}.{key}' must be a {numeric.__name__}, got {value!r}") from None
        kwargs[key] = value
    return kwargs


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
    # Every projection a Qwen3.5 block owns: the gated-delta-net linear attention
    # (in_proj_*/out_proj), the sparse full attention (q/k/v/o_proj), the MLP, and
    # the vision tower (qkv/proj/linear_fc*). The usual q/k/v/o_proj quartet alone
    # reaches only the handful of full-attention layers -- 10% of the linear
    # modules on a 27B hybrid model -- which leaves most of the network frozen.
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "in_proj_qkv",
            "in_proj_z",
            "in_proj_a",
            "in_proj_b",
            "out_proj",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "qkv",
            "proj",
            "linear_fc1",
            "linear_fc2",
        ]
    )
    lora_bias: str = "none"
    router_aux_loss_coef: float = 0.001

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ModelConfig":
        """Build a ModelConfig from a dict, ignoring unknown keys."""
        return cls(**_build_kwargs(cls, raw, "model"))


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
    image_root: str | None = None
    """Directory that recorded image paths are remapped onto.

    Only used by datasets built with ``--image-mode reference``. Leave it unset to
    use the corpus roots recorded in the dataset's image manifest; set it when the
    corpus has moved or is mounted somewhere else.
    """
    max_pixels: int | None = DEFAULT_MAX_PIXELS
    """Largest number of pixels a page may contribute to the vision encoder.

    Document scans are far larger than the encoder can use, and every patch it
    receives becomes a token in the language model's sequence. The ceiling is
    applied both to the processor and while decoding reference-mode images. Set
    it to None to feed images at their native resolution, which costs memory and
    step time roughly in proportion to the extra pixels.
    """

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DataConfig":
        """Build a DataConfig from a dict, ignoring unknown keys."""
        return cls(**_build_kwargs(cls, raw, "data"))


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
        return cls(**_build_kwargs(cls, raw, "training"))


@dataclass
class GRPOConfig:
    """Configuration specific to GRPO (online RL)."""

    # Which reward set the trainer uses:
    #   "finetune" (default) -> [format_reward, accuracy_reward] (src.rewards)
    #   "pretrain"           -> [cer_reward] (src.rewards_pretrain)
    # Pretraining (e.g. learning Old Dutch) optimizes CER directly; finetuning
    # refines an already-capable model with format + accuracy rewards.
    stage: str = "finetune"
    # Which CER metric the pretrain stage optimizes:
    #   "raw" (default) -> case-sensitive CER, whitespace counts (matches the
    #                      primary unnormalized CER reported by the eval)
    #   "normalized"    -> casefolded, whitespace-collapsed CER
    cer_metric: str = "raw"
    num_generations: int = 8
    # Generation cap for the online rollout. Must comfortably exceed the
    # thinking + answer length (see docs/usage.md); 512 truncates reasoning
    # models. None is not supported by TRL (it is used in loss math and as
    # max_new_tokens), so keep a finite value and monitor reward/truncated.
    max_completion_length: int = 2048
    temperature: float = 1.0
    use_vllm: bool = False
    vllm_mode: str = "colocate"
    beta: float = 0.0
    loss_type: str = "dapo"
    log_completions: bool = False
    # Per-reward-function weights, aligned with the order of the reward funcs
    # selected by ``stage``. For "finetune" the order is (format, accuracy):
    # format is a scaffold, accuracy carries the signal. For "pretrain" there
    # is a single reward (CER), so use [1.0].
    reward_weights: list[float] = field(default_factory=lambda: [0.2, 0.8])

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "GRPOConfig":
        """Build a GRPOConfig from a dict, ignoring unknown keys."""
        return cls(**_build_kwargs(cls, raw, "grpo"))


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
                logger.warning("Ignoring unknown key '%s' in config", key)
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
