"""Convert raw JSONL data into TRL-format Hugging Face datasets.

Mirrors TRL's `examples/datasets/` conversion pattern. Converts BEFORE applying
the chat template, per TRL recommendation.

Supported source formats (JSONL). Each row may use either a structured
`messages` history (multi-turn) or the legacy `prompt` field (single-turn):

- SFT:  {"messages": [...], "completion": "..."}
- DPO:  {"messages": [...], "chosen": "...", "rejected": "..."}
- KTO:  {"messages": [...], "completion": "...", "label": true}
- GRPO: {"messages": [...]}

Legacy single-turn rows remain supported:

- SFT:  {"image": "path.png", "prompt": "...", "completion": "..."}
- DPO:  {"image": "path.png", "prompt": "...", "chosen": "...", "rejected": "..."}
- KTO:  {"image": "path.png", "prompt": "...", "completion": "...", "label": true}
- GRPO: {"image": "path.png", "prompt": "..."}

A message `content` may be a plain string or a list of content blocks. Image
blocks use `{"type": "image", "image": "path.png"}`; the actual PIL image is
collected into the dataset's `images` column and the block is replaced with a
`{"type": "image"}` placeholder, matching TRL's vision convention.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset
from PIL import Image as PILImage

VALID_ROLES = {"system", "user", "assistant"}


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


def _validate_messages(messages: list[dict]) -> None:
    """Validate a `messages` history, raising ValueError on malformed input."""
    if not messages:
        raise ValueError("Row has an empty `messages` history.")
    for index, message in enumerate(messages):
        role = message.get("role")
        if role not in VALID_ROLES:
            raise ValueError(f"Message {index} has invalid role {role!r}; expected one of {sorted(VALID_ROLES)}.")
        content = message.get("content")
        if isinstance(content, str):
            continue
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict) or "type" not in block:
                    raise ValueError(f"Message {index} has an invalid content block: {block!r}.")
            continue
        raise ValueError(f"Message {index} has invalid content: {content!r}.")
    if messages[-1].get("role") != "user":
        raise ValueError("The final message in `messages` must have role 'user'.")


def _validate_row(row: dict, method: str) -> None:
    """Validate a raw row for a given method, raising ValueError on bad input."""
    if "messages" in row:
        _validate_messages(row["messages"])
    elif "prompt" not in row:
        raise ValueError("Row must contain either a `messages` history or a `prompt` field.")

    if method == "sft" and "completion" not in row:
        raise ValueError("SFT rows require a `completion` field.")
    if method == "dpo" and ("chosen" not in row or "rejected" not in row):
        raise ValueError("DPO rows require `chosen` and `rejected` fields.")
    if method == "kto":
        if "completion" not in row or "label" not in row:
            raise ValueError("KTO rows require `completion` and `label` fields.")
        if not isinstance(row["label"], bool):
            raise ValueError("KTO `label` must be a boolean.")


def _build_prompt(row: dict) -> tuple[list[dict], list]:
    """Build a TRL conversational prompt and its image list from a raw row.

    Supports either a structured `messages` history or the legacy `prompt`
    (plus optional `image`) form. Image references are converted to
    `{"type": "image"}` placeholders and the corresponding PIL images are
    returned in the second element of the tuple.
    """
    images: list = []

    if "messages" in row:
        normalized: list[dict] = []
        for message in row["messages"]:
            role = message["role"]
            content = message["content"]
            if isinstance(content, str):
                normalized.append({"role": role, "content": [{"type": "text", "text": content}]})
                continue
            blocks: list[dict] = []
            for block in content:
                if block.get("type") == "image":
                    images.append(PILImage.open(block["image"]))
                    blocks.append({"type": "image"})
                else:
                    blocks.append(block)
            normalized.append({"role": role, "content": blocks})
        return normalized, images

    content = _content_with_image(row.get("image"), row["prompt"])
    if row.get("image"):
        images.append(PILImage.open(row["image"]))
    return [{"role": "user", "content": content}], images


def to_sft(rows: list[dict]) -> Dataset:
    """Convert correction rows to SFT prompt-completion format."""
    records = []
    for row in rows:
        _validate_row(row, "sft")
        prompt, images = _build_prompt(row)
        completion = [{"role": "assistant", "content": [{"type": "text", "text": row["completion"]}]}]
        records.append({"prompt": prompt, "completion": completion, "images": images})
    return Dataset.from_list(records)


def to_dpo(rows: list[dict]) -> Dataset:
    """Convert A/B comparison rows to DPO preference format."""
    records = []
    for row in rows:
        _validate_row(row, "dpo")
        prompt, images = _build_prompt(row)
        chosen = [{"role": "assistant", "content": [{"type": "text", "text": row["chosen"]}]}]
        rejected = [{"role": "assistant", "content": [{"type": "text", "text": row["rejected"]}]}]
        records.append({"prompt": prompt, "chosen": chosen, "rejected": rejected, "images": images})
    return Dataset.from_list(records)


def to_kto(rows: list[dict]) -> Dataset:
    """Convert good/bad feedback rows to KTO unpaired preference format."""
    records = []
    for row in rows:
        _validate_row(row, "kto")
        prompt, images = _build_prompt(row)
        completion = [{"role": "assistant", "content": [{"type": "text", "text": row["completion"]}]}]
        records.append({"prompt": prompt, "completion": completion, "label": bool(row["label"]), "images": images})
    return Dataset.from_list(records)


def to_grpo(rows: list[dict]) -> Dataset:
    """Convert feedback rows to GRPO prompt format."""
    records = []
    for row in rows:
        _validate_row(row, "grpo")
        prompt, images = _build_prompt(row)
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
