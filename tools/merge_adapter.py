"""Merge a trained LoRA adapter into the base model.

Wraps PEFT's `merge_and_unload` to produce a standalone merged model.
"""

from __future__ import annotations

import argparse

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor


def merge(base_model: str, adapter_path: str, output_dir: str) -> None:
    """Merge a LoRA adapter into the base model and save the result.

    Args:
        base_model (str): The base model id or path.
        adapter_path (str): Path to the trained adapter.
        output_dir (str): Where to save the merged model.
    """
    model = AutoModelForCausalLM.from_pretrained(base_model)
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    model.save_pretrained(output_dir)
    processor = AutoProcessor.from_pretrained(base_model)
    processor.save_pretrained(output_dir)
    print(f"Merged model saved to {output_dir}")


def main() -> None:
    """CLI entry point for merging an adapter."""
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into the base model.")
    parser.add_argument("--base_model", required=True, help="Base model id or path.")
    parser.add_argument("--adapter_path", required=True, help="Path to the trained adapter.")
    parser.add_argument("--output_dir", required=True, help="Where to save the merged model.")
    args = parser.parse_args()
    merge(args.base_model, args.adapter_path, args.output_dir)


if __name__ == "__main__":
    main()
