"""GRPO training entry point using TRL's GRPOTrainer.

Maps feedback data to online reinforcement learning via reward functions.
"""

from __future__ import annotations

import argparse

from datasets import load_from_disk
from trl import GRPOConfig, GRPOTrainer

from src.config import RunConfig, load_config
from src.peft_utils import build_lora_config, build_quantization_config
from utils.logging import get_logger, write_run_metadata
from utils.paths import run_output_dir

logger = get_logger(__name__)


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
        model_init_kwargs={"dtype": config.model.dtype},
        trust_remote_code=config.model.trust_remote_code,
    )


def _default_reward_func(completions, **kwargs):
    """Default reward function: reward non-empty completions.

    This is a minimal placeholder so GRPO has a reward source. Replace with a
    task-specific reward function derived from your feedback data.
    """
    return [1.0 if completion else 0.0 for completion in completions]


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
        reward_funcs=_default_reward_func,
        peft_config=build_lora_config(config.model),
        quantization_config=build_quantization_config(config.model),
    )
    trainer.train()
    logger.info("GRPO training complete. Output in %s", config.training.output_dir)


def main() -> None:
    """CLI entry point for GRPO training."""
    parser = argparse.ArgumentParser(description="Run GRPO training.")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file.")
    args = parser.parse_args()
    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
