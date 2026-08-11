"""Convert raw JSONL data into TRL-format Hugging Face datasets.

Mirrors TRL's `examples/datasets/` conversion pattern. Converts BEFORE applying
the chat template, per TRL recommendation.

Supported source formats (JSONL):
- SFT:  {"image": "path.png", "prompt": "...", "completion": "..."}
- DPO:  {"image": "path.png", "prompt": "...", "chosen": "...", "rejected": "..."}
- KTO:  {"image": "path.png", "prompt": "...", "completion": "...", "label": true}
- GRPO: {"image": "path.png", "prompt": "..."}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset
from PIL import Image as PILImage


def _load_rows(path: Path) -> list[dict]:
    """Load JSONL rows from a file."""
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _content_with_image(image_path: str | None, text: str) -> list[dict]:
    """Build a vision content block list for a message."""
    content: list[dict] = []
    if image_path:
        content.append({"type": "image"})
    content.append({"type": "text", "text": text})
    return content


def to_sft(rows: list[dict]) -> Dataset:
    """Convert correction rows to SFT prompt-completion format."""
    records = []
    for row in rows:
        prompt = [{"role": "user", "content": _content_with_image(row.get("image"), row["prompt"])}]
        completion = [{"role": "assistant", "content": [{"type": "text", "text": row["completion"]}]}]
        images = [PILImage.open(row["image"])] if row.get("image") else []
        records.append({"prompt": prompt, "completion": completion, "images": images})
    return Dataset.from_list(records)


def to_dpo(rows: list[dict]) -> Dataset:
    """Convert A/B comparison rows to DPO preference format."""
    records = []
    for row in rows:
        prompt = [{"role": "user", "content": _content_with_image(row.get("image"), row["prompt"])}]
        chosen = [{"role": "assistant", "content": [{"type": "text", "text": row["chosen"]}]}]
        rejected = [{"role": "assistant", "content": [{"type": "text", "text": row["rejected"]}]}]
        images = [PILImage.open(row["image"])] if row.get("image") else []
        records.append({"prompt": prompt, "chosen": chosen, "rejected": rejected, "images": images})
    return Dataset.from_list(records)


def to_kto(rows: list[dict]) -> Dataset:
    """Convert good/bad feedback rows to KTO unpaired preference format."""
    records = []
    for row in rows:
        prompt = [{"role": "user", "content": _content_with_image(row.get("image"), row["prompt"])}]
        completion = [{"role": "assistant", "content": [{"type": "text", "text": row["completion"]}]}]
        images = [PILImage.open(row["image"])] if row.get("image") else []
        records.append({"prompt": prompt, "completion": completion, "label": bool(row["label"]), "images": images})
    return Dataset.from_list(records)


def to_grpo(rows: list[dict]) -> Dataset:
    """Convert feedback rows to GRPO prompt format."""
    records = []
    for row in rows:
        prompt = [{"role": "user", "content": _content_with_image(row.get("image"), row["prompt"])}]
        images = [PILImage.open(row["image"])] if row.get("image") else []
        records.append({"prompt": prompt, "images": images})
    return Dataset.from_list(records)


CONVERTERS = {
    "sft": to_sft,
    "dpo": to_dpo,
    "kto": to_kto,
    "grpo": to_grpo,
}


def main() -> None:
    """Convert a raw JSONL file to a TRL-format HF dataset."""
    parser = argparse.ArgumentParser(description="Convert raw JSONL to a TRL-format HF dataset.")
    parser.add_argument("--input", required=True, help="Path to the raw JSONL file.")
    parser.add_argument("--output", required=True, help="Path to save the HF dataset.")
    parser.add_argument("--method", required=True, choices=list(CONVERTERS), help="Training method.")
    args = parser.parse_args()

    rows = _load_rows(Path(args.input))
    dataset = CONVERTERS[args.method](rows)
    dataset.save_to_disk(args.output)
    print(f"Converted {len(dataset)} rows to {args.method} format -> {args.output}")


if __name__ == "__main__":
    main()
