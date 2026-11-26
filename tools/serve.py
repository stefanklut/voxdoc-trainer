"""Serve a trained model with a lightweight inference script.

This helper loads model weights locally with ``transformers`` and generates a
single completion, optionally conditioned on one or more document images::

    python tools/serve.py --model outputs/merged --prompt "Read this document."
    python tools/serve.py --model outputs/merged --image scan.png \\
        --prompt "Transcribe this document."

For production serving at scale, prefer vLLM (``vllm serve``), which exposes an
OpenAI-compatible endpoint. The repo's other clients target such an endpoint
rather than loading weights: ``tools/generate_doclang.py`` for validated
DocLang, ``eval/`` for document evaluation, and the optional external
``vllm_cli`` client (see docs/usage.md).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
from PIL import Image as PILImage
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor

logger = logging.getLogger(__name__)

URL_SCHEMES = ("http://", "https://", "file://")


def load_images(image_paths: list[str]) -> list[PILImage.Image]:
    """
    Load document images for a vision-language model.

    Images are converted to RGB so that grayscale, palette, and RGBA scans are
    accepted by processors that expect three channels.

    Args:
        image_paths (list[str]): Paths to the source document images.
    Returns:
        list[PILImage.Image]: The opened images, in the order given.
    Raises:
        ValueError: If a reference is a URL rather than a local file.
        FileNotFoundError: If a path does not exist.
    """
    images: list[PILImage.Image] = []
    for image_path_str in image_paths:
        if image_path_str.startswith(URL_SCHEMES):
            raise ValueError(
                f"{image_path_str}: local inference reads images from disk; serve the model with "
                "vLLM (tools/generate_doclang.py or vllm_cli) to send image URLs."
            )
        image_path = Path(image_path_str).expanduser()
        if not image_path.is_file():
            raise FileNotFoundError(f"No such image: {image_path}")
        logger.debug("Loading image %s", image_path)
        images.append(PILImage.open(image_path).convert("RGB"))
    return images


def resolve_device(device: Optional[str] = None) -> str:
    """
    Return the device to run on, defaulting to CUDA when available.

    Args:
        device (Optional[str]): Explicit device (e.g. 'cuda', 'cuda:1', 'cpu'), or None to detect.
    Returns:
        str: The device string to place the model and inputs on.
    """
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model_and_processor(model_id: str, has_images: bool, dtype: str, device: str):
    """
    Load the model and its processor onto the requested device.

    A vision-language checkpoint is loaded with ``AutoModelForImageTextToText`` when
    images are part of the request; text-only requests keep using ``AutoModelForCausalLM``.

    Args:
        model_id (str): The model id or path.
        has_images (bool): Whether the request carries images.
        dtype (str): Weight dtype, e.g. 'auto', 'bfloat16', or 'float16'.
        device (str): Device to load onto.
    Returns:
        tuple: The model and the processor.
    """
    model_class = AutoModelForImageTextToText if has_images else AutoModelForCausalLM
    logger.info("Loading %s with %s on %s (%s)", model_id, model_class.__name__, device, dtype)
    model = model_class.from_pretrained(model_id, dtype=dtype)
    model.to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def generate(
    model_id: str,
    prompt: str,
    max_new_tokens: int = 256,
    images: Optional[list[str]] = None,
    dtype: str = "auto",
    device: Optional[str] = None,
) -> str:
    """
    Generate a response from a model.

    Args:
        model_id (str): The model id or path.
        prompt (str): The user prompt.
        max_new_tokens (int): Maximum number of tokens to generate.
        images (Optional[list[str]]): Document image paths to condition on.
        dtype (str): Weight dtype passed to ``from_pretrained``.
        device (Optional[str]): Device override; None auto-detects CUDA.
    Returns:
        str: The generated text, without the prompt.
    """
    loaded_images = load_images(images) if images else None
    model, processor = load_model_and_processor(model_id, bool(loaded_images), dtype, resolve_device(device))

    messages = [{"role": "user", "content": prompt}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Text-only processors (plain tokenizers) do not accept an images keyword, so omit it.
    encode_kwargs: dict = {"return_tensors": "pt"}
    if loaded_images is not None:
        encode_kwargs["images"] = loaded_images
    inputs = processor(text=[text], **encode_kwargs).to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # generate() echoes the prompt for decoder-only models, so decode only the continuation.
    prompt_length = inputs["input_ids"].shape[1]
    return processor.decode(outputs[0][prompt_length:], skip_special_tokens=True).strip()


def main() -> None:
    """CLI entry point for inference."""
    parser = argparse.ArgumentParser(description="Run inference with a trained model.")
    parser.add_argument("--model", required=True, help="Model id or path.")
    parser.add_argument("--prompt", required=True, help="The user prompt.")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument(
        "--image",
        action="append",
        default=None,
        metavar="IMAGE",
        help="Document image to condition on; repeatable. Local file paths only.",
    )
    parser.add_argument("--dtype", default="auto", help="Weight dtype, e.g. auto, bfloat16, float16.")
    parser.add_argument("--device", default=None, help="Device, e.g. cuda, cuda:1, cpu (default: auto-detect).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Log model loading details.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    response = generate(
        args.model,
        args.prompt,
        args.max_new_tokens,
        images=args.image,
        dtype=args.dtype,
        device=args.device,
    )
    print(response)


if __name__ == "__main__":
    main()
