"""Merge a trained LoRA adapter into the base model.

Wraps PEFT's `merge_and_unload` to produce a standalone merged model.
"""

from __future__ import annotations

import argparse
import re
import warnings

from peft import PeftModel
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    PreTrainedModel,
)

#: PEFT warns, rather than raises, when adapter tensors have nowhere to land, and the
#: key list it prints is long enough to scroll straight past in a training log.
_MISSING_KEYS_RE = re.compile(r"Found missing adapter keys while loading the checkpoint: \[(.*)\]", re.DOTALL)


def resolve_model_class(base_model: str) -> type:
    """Pick the model class an adapter was trained against.

    A vision-language adapter carries tensors for the vision tower as well as the
    language model. Loading it onto the text-only class discards every tensor whose
    target module does not exist there -- PEFT only warns -- so the merged model keeps
    a fraction of the adapter and still reports success. On Qwen3.5 the text-only class
    has no vision tower at all, which drops 372 of the 474 tensors a full-corpus run
    produces.

    Args:
        base_model (str): Model id or local path.

    Returns:
        type: ``AutoModelForImageTextToText`` for a vision-language config, else
            ``AutoModelForCausalLM``.
    """
    config = AutoConfig.from_pretrained(base_model)
    if getattr(config, "vision_config", None) is not None or getattr(config, "image_token_id", None) is not None:
        return AutoModelForImageTextToText
    return AutoModelForCausalLM


def load_adapter(model: PreTrainedModel, adapter_path: str) -> PeftModel:
    """Load an adapter, turning PEFT's silent partial-load warning into an error.

    Args:
        model (PreTrainedModel): The base model to attach the adapter to.
        adapter_path (str): Path to the trained adapter.

    Returns:
        PeftModel: The model with the adapter applied.

    Raises:
        RuntimeError: If adapter tensors had no matching module, which means the base
            model is not the architecture the adapter was trained against.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = PeftModel.from_pretrained(model, adapter_path)
    for entry in caught:
        match = _MISSING_KEYS_RE.search(str(entry.message))
        if match:
            keys = [k.strip().strip("'\"") for k in match.group(1).split(",") if k.strip()]
            raise RuntimeError(
                f"Adapter '{adapter_path}' did not apply to {type(model).__name__}: "
                f"{len(keys)} of its tensors had no matching module "
                f"(e.g. {keys[0] if keys else 'unknown'}). "
                "Merge against the architecture the adapter was trained on."
            )
    return loaded


def merge(base_model: str, adapter_path: str, output_dir: str) -> None:
    """Merge a LoRA adapter into the base model and save the result.

    Args:
        base_model (str): The base model id or path.
        adapter_path (str): Path to the trained adapter.
        output_dir (str): Where to save the merged model.

    Raises:
        RuntimeError: If the adapter did not fully apply to the base model.
    """
    model_class = resolve_model_class(base_model)
    print(f"Loading {base_model} as {model_class.__name__}")
    model = model_class.from_pretrained(base_model)
    model = load_adapter(model, adapter_path)
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
