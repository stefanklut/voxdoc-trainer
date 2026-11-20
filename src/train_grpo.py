"""GRPO training entry point using TRL's GRPOTrainer.

Maps feedback data to online reinforcement learning via reward functions.
"""

from __future__ import annotations

import argparse

from datasets import load_from_disk
from trl import GRPOConfig, GRPOTrainer

from src.config import RunConfig, load_config
from src.peft_utils import build_lora_config, build_quantization_config
from src.rewards import accuracy_reward, format_reward
from src.rewards_pretrain import cer_reward, cer_reward_normalized
from utils.logging_utils import get_logger, setup_logger, write_run_metadata
from utils.paths import run_output_dir

logger = get_logger()


def build_grpo_config(config: RunConfig) -> GRPOConfig:
    """Build a GRPOConfig from a RunConfig.

    Args:
        config (RunConfig): The run configuration.

    Returns:
        GRPOConfig: The TRL GRPO configuration.
    """
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
            raise ValueError(
                f"Unknown grpo.cer_metric {config.grpo.cer_metric!r}; expected 'raw' or 'normalized'."
            )
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

    trainer = GRPOTrainer(
        model=config.model.model_id,
        args=build_grpo_config(config),
        train_dataset=train_dataset,
        reward_funcs=select_reward_funcs(config),
        peft_config=build_lora_config(config.model),
        quantization_config=build_quantization_config(config.model),
    )
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
