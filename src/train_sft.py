"""SFT training entry point using TRL's SFTTrainer.

Maps corrections data to supervised fine-tuning. Uses TRL/transformers native
features (checkpointing, report_to, eval, etc.) and PEFT LoRA/QLoRA.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_from_disk
from trl import SFTConfig, SFTTrainer

from src.config import RunConfig, load_config
from src.peft_utils import build_lora_config, build_quantization_config
from utils.logging import get_logger, write_run_metadata
from utils.paths import run_output_dir

logger = get_logger(__name__)


def build_sft_config(config: RunConfig) -> SFTConfig:
    """Build an SFTConfig from a RunConfig.

    Args:
        config (RunConfig): The run configuration.

    Returns:
        SFTConfig: The TRL SFT configuration.
    """
    training = config.training
    return SFTConfig(
        output_dir=training.output_dir,
        per_device_train_batch_size=training.per_device_train_batch_size,
        per_device_eval_batch_size=training.per_device_eval_batch_size,
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
        eval_strategy=training.eval_strategy,
        eval_steps=training.eval_steps,
        report_to=training.report_to,
        seed=training.seed,
        max_length=config.data.max_length,
        packing=config.data.packing,
        dataset_num_proc=config.data.dataset_num_proc,
        assistant_only_loss=training.assistant_only_loss,
        model_init_kwargs={"dtype": config.model.dtype},
        trust_remote_code=config.model.trust_remote_code,
    )


def train(config: RunConfig) -> None:
    """Run SFT training.

    Args:
        config (RunConfig): The run configuration.
    """
    run_dir = run_output_dir(config)
    write_run_metadata(run_dir, config)
    logger.info("Starting SFT training with model %s", config.model.model_id)

    train_dataset = load_from_disk(config.data.train_path)
    eval_dataset = load_from_disk(config.data.eval_path) if config.data.eval_path else None

    trainer = SFTTrainer(
        model=config.model.model_id,
        args=build_sft_config(config),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=build_lora_config(config.model),
        quantization_config=build_quantization_config(config.model),
    )
    trainer.train()
    logger.info("SFT training complete. Output in %s", config.training.output_dir)


def main() -> None:
    """CLI entry point for SFT training."""
    parser = argparse.ArgumentParser(description="Run SFT training.")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file.")
    args = parser.parse_args()
    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
