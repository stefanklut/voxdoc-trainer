"""GRPO training entry point using TRL's GRPOTrainer.

Maps feedback data to online reinforcement learning via reward functions.
"""

from __future__ import annotations

import argparse
import inspect
import math
from typing import Any

from datasets import load_from_disk
from trl import GRPOConfig, GRPOTrainer

from src.config import RunConfig, load_config
from src.peft_utils import build_lora_config, build_quantization_config, log_lora_coverage
from src.rewards import accuracy_reward, format_reward
from src.rewards_pretrain import cer_reward, cer_reward_normalized
from utils.image_loading import load_processor, wrap_trainer_images
from utils.logging_utils import get_logger, setup_logger, write_run_metadata
from utils.paths import run_output_dir

logger = get_logger()


def check_generation_batch(config: RunConfig) -> None:
    """Reject a batch layout TRL cannot sample completions with.

    GRPO draws whole groups of ``num_generations`` completions per prompt, so the
    generation batch (per-device batch times gradient accumulation times the
    number of processes) has to be a multiple of the group size. TRL only reports
    this after loading the model, which on a large checkpoint costs a long wait
    before the failure.

    Args:
        config (RunConfig): The run configuration to validate.

    Raises:
        ValueError: If the generation batch is not divisible by ``num_generations``.
    """
    training = config.training
    num_generations = config.grpo.num_generations
    generation_batch = training.per_device_train_batch_size * training.gradient_accumulation_steps
    if generation_batch % num_generations:
        raise ValueError(
            f"per_device_train_batch_size ({training.per_device_train_batch_size}) * "
            f"gradient_accumulation_steps ({training.gradient_accumulation_steps}) = {generation_batch} "
            f"is not divisible by num_generations ({num_generations}). "
            f"Set gradient_accumulation_steps to a multiple of "
            f"{num_generations // math.gcd(generation_batch, num_generations)}."
        )


def check_reward_columns(dataset: Any, reward_funcs: list, dataset_path: str) -> None:
    """Reject a dataset that cannot feed its reward functions, before the model loads.

    TRL passes each reward function the dataset columns whose names match its
    parameters, and it only discovers a missing one while scoring the first batch
    of completions — after the checkpoint has loaded and the rollouts have been
    generated. On a 27B model that is many minutes and a full GPU allocation spent
    to reach ``TypeError: cer_reward() missing 1 required positional argument``.

    Args:
        dataset (Any): The training dataset, whose ``column_names`` are checked.
        reward_funcs (list): The reward functions selected for this run.
        dataset_path (str): Where the dataset was loaded from, for the error text.

    Raises:
        ValueError: If a reward function needs a column the dataset does not have.
    """
    columns = set(getattr(dataset, "column_names", None) or [])
    if not columns:
        return
    # TRL always supplies these three by keyword; everything else a reward asks
    # for has to come from a dataset column.
    provided = {"completions", "prompts", "completion_ids"}
    missing: dict[str, list[str]] = {}
    for func in reward_funcs:
        required = [
            name
            for name, param in inspect.signature(func).parameters.items()
            if param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD and param.default is inspect.Parameter.empty
        ]
        absent = [name for name in required if name not in provided and name not in columns]
        if absent:
            missing[func.__name__] = absent
    if missing:
        detail = "; ".join(f"{name}() needs {cols}" for name, cols in missing.items())
        raise ValueError(
            f"The GRPO dataset at {dataset_path} has columns {sorted(columns)} but {detail}. "
            "GRPO rows need a 'ground_truth' column holding the answer the rewards score "
            "the completion against (see docs/data_formats.md)."
        )


def build_grpo_config(config: RunConfig) -> GRPOConfig:
    """Build a GRPOConfig from a RunConfig.

    Args:
        config (RunConfig): The run configuration.

    Returns:
        GRPOConfig: The TRL GRPO configuration.

    Raises:
        ValueError: If the batch layout is incompatible with ``num_generations``.
    """
    check_generation_batch(config)
    training = config.training
    grpo = config.grpo
    return GRPOConfig(
        output_dir=training.output_dir,
        per_device_train_batch_size=training.per_device_train_batch_size,
        gradient_accumulation_steps=training.gradient_accumulation_steps,
        num_train_epochs=training.num_train_epochs,
        max_steps=training.dry_run_steps if training.dry_run else training.max_steps,
        learning_rate=training.learning_rate,
        lr_scheduler_type=training.lr_scheduler_type,
        warmup_steps=training.warmup_steps,
        weight_decay=training.weight_decay,
        max_grad_norm=training.max_grad_norm,
        gradient_checkpointing=training.gradient_checkpointing,
        bf16=training.bf16,
        fp16=training.fp16,
        logging_steps=training.logging_steps,
        save_strategy=training.save_strategy,
        save_steps=training.save_steps,
        save_total_limit=training.save_total_limit,
        report_to=training.report_to,
        seed=training.seed,
        num_generations=grpo.num_generations,
        max_completion_length=grpo.max_completion_length,
        temperature=grpo.temperature,
        use_vllm=grpo.use_vllm,
        vllm_mode=grpo.vllm_mode,
        beta=grpo.beta,
        loss_type=grpo.loss_type,
        log_completions=grpo.log_completions,
        # Keep the `ground_truth` column so the reward functions can score
        # completions against it (TRL drops non-model columns otherwise).
        remove_unused_columns=False,
        reward_weights=grpo.reward_weights,
        model_init_kwargs={"dtype": config.model.dtype},
        trust_remote_code=config.model.trust_remote_code,
    )


def select_reward_funcs(config: RunConfig) -> list:
    """Select the GRPO reward functions for the configured stage.

    Args:
        config (RunConfig): The run configuration.

    Returns:
        list: The reward functions, in the order their weights appear in
        ``grpo.reward_weights``.

    Raises:
        ValueError: If ``grpo.stage`` is unknown, ``grpo.cer_metric`` is
        unknown, or ``grpo.reward_weights`` does not have one weight per
        selected reward function.
    """
    stage = config.grpo.stage
    if stage == "pretrain":
        # Pretraining (e.g. learning Old Dutch): optimize CER directly.
        # "raw" (default) is case-sensitive and counts whitespace; "normalized"
        # casefolds and collapses whitespace on both sides.
        if config.grpo.cer_metric == "normalized":
            reward_funcs = [cer_reward_normalized]
        elif config.grpo.cer_metric == "raw":
            reward_funcs = [cer_reward]
        else:
            raise ValueError(f"Unknown grpo.cer_metric {config.grpo.cer_metric!r}; expected 'raw' or 'normalized'.")
    elif stage == "finetune":
        # Finetuning: format (well-formed answer) + accuracy (closeness to
        # ground truth). Weights come from the config and are aligned to this
        # order.
        reward_funcs = [format_reward, accuracy_reward]
    else:
        raise ValueError(f"Unknown grpo.stage {stage!r}; expected 'finetune' or 'pretrain'.")

    weights = config.grpo.reward_weights
    if len(weights) != len(reward_funcs):
        names = ", ".join(func.__name__ for func in reward_funcs)
        raise ValueError(
            f"grpo.reward_weights has {len(weights)} entries but stage {stage!r} "
            f"selects {len(reward_funcs)} reward function(s) ({names}). "
            "Provide one weight per reward function."
        )
    return reward_funcs


def train(config: RunConfig) -> None:
    """Run GRPO training.

    Args:
        config (RunConfig): The run configuration.
    """
    run_dir = run_output_dir(config)
    write_run_metadata(run_dir, config)
    logger.info("Starting GRPO training with model %s", config.model.model_id)

    train_dataset = load_from_disk(config.data.train_path)
    reward_funcs = select_reward_funcs(config)
    check_reward_columns(train_dataset, reward_funcs, config.data.train_path)
    processor = load_processor(config.model.model_id, config.model.trust_remote_code, config.data.max_pixels)

    trainer = GRPOTrainer(
        model=config.model.model_id,
        args=build_grpo_config(config),
        train_dataset=train_dataset,
        reward_funcs=reward_funcs,
        processing_class=processor,
        peft_config=build_lora_config(config.model),
        quantization_config=build_quantization_config(config.model),
    )
    if wrap_trainer_images(trainer, train_dataset, None, config.data.image_root, config.data.max_pixels):
        logger.info("Decoding images lazily from %s", config.data.image_root or "the recorded corpus roots")
    log_lora_coverage(trainer.model, config.model)
    trainer.train()
    logger.info("GRPO training complete. Output in %s", config.training.output_dir)


def main() -> None:
    """CLI entry point for GRPO training."""
    setup_logger()
    parser = argparse.ArgumentParser(description="Run GRPO training.")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file.")
    args = parser.parse_args()
    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
