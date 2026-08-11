"""Serve a trained model with a lightweight inference script.

For production serving at scale, prefer vLLM or SGLang. This provides a
simple programmatic inference helper for quick checks.
"""

from __future__ import annotations

import argparse

from transformers import AutoModelForCausalLM, AutoProcessor


def generate(model_id: str, prompt: str, max_new_tokens: int = 256) -> str:
    """Generate a response from a model.

    Args:
        model_id (str): The model id or path.
        prompt (str): The user prompt.
        max_new_tokens (int): Maximum number of tokens to generate.

    Returns:
        str: The generated text.
    """
    model = AutoModelForCausalLM.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)
    messages = [{"role": "user", "content": prompt}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.decode(outputs[0], skip_special_tokens=True)


def main() -> None:
    """CLI entry point for inference."""
    parser = argparse.ArgumentParser(description="Run inference with a trained model.")
    parser.add_argument("--model", required=True, help="Model id or path.")
    parser.add_argument("--prompt", required=True, help="The user prompt.")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()
    print(generate(args.model, args.prompt, args.max_new_tokens))


if __name__ == "__main__":
    main()
