"""Convert raw JSONL data into TRL-format Hugging Face datasets.

Mirrors TRL's `examples/datasets/` conversion pattern. Converts BEFORE applying
the chat template, per TRL recommendation.

Supported source formats (JSONL). Each row may use either a structured
`messages` history (multi-turn) or the legacy `prompt` field (single-turn):

- SFT:  {"messages": [...], "completion": "..."}
- DPO:  {"messages": [...], "chosen": "...", "rejected": "..."}
- KTO:  {"messages": [...], "completion": "...", "label": true}
- GRPO: {"messages": [...], "ground_truth": "..."}

Legacy single-turn rows remain supported:

- SFT:  {"image": "path.png", "prompt": "...", "completion": "..."}
- DPO:  {"image": "path.png", "prompt": "...", "chosen": "...", "rejected": "..."}
- KTO:  {"image": "path.png", "prompt": "...", "completion": "...", "label": true}
- GRPO: {"image": "path.png", "prompt": "...", "ground_truth": "..."}

For GRPO, ``ground_truth`` is the answer the reward functions score generated
completions against (see :mod:`src.rewards`). It is kept as a dataset column so
the rewards can read it at training time.

A message `content` may be a plain string or a list of content blocks. Image
blocks use `{"type": "image", "image": "path.png"}`; the actual PIL image is
collected into the dataset's `images` column and the block is replaced with a
`{"type": "image"}` placeholder, matching TRL's vision convention.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from datasets import Dataset
from PIL import Image as PILImage

# Allow running as a plain script from the repo root (``python tools/...``).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.discovery import DiscoveryError, discover_jsonl  # noqa: E402
from utils.image_loading import (  # noqa: E402
    add_image_mode_argument,
    image_roots,
    relative_image_path,
    resolve_image_path,
    save_image_dataset,
)

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


def load_rows(inputs: Path | str | Sequence[Path | str]) -> list[dict]:
    """Load JSONL rows from files and/or directories.

    Directories contribute the ``.jsonl`` files directly inside them, in sorted
    order; individual file paths are read as-is, so a single-file invocation
    behaves exactly as before.

    Args:
        inputs (Path | str | Sequence[Path | str]): One path or several.

    Returns:
        list[dict]: The concatenated rows, in input order.

    Raises:
        FileNotFoundError: If a named input does not exist.
        DiscoveryError: If the inputs resolve to no JSONL file at all.
    """
    candidates: Sequence[Path | str] = [inputs] if isinstance(inputs, (str, Path)) else list(inputs)
    rows: list[dict] = []
    for path in discover_jsonl(candidates):
        rows.extend(_load_rows(path))
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
    if method == "grpo" and "ground_truth" not in row:
        raise ValueError("GRPO rows require a `ground_truth` field (the answer the reward scores against).")


def _build_prompt(row: dict, image_mode: str = "embed", roots: Sequence[Path] | None = None) -> tuple[list[dict], list]:
    """Build a TRL conversational prompt and its image list from a raw row.

    Supports either a structured `messages` history or the legacy `prompt`
    (plus optional `image`) form. Image references are converted to
    `{"type": "image"}` placeholders and the corresponding images are returned
    in the second element of the tuple.

    Args:
        row (dict): One raw JSONL row.
        image_mode (str): One of ``utils.image_loading.IMAGE_MODES``; "embed" opens
            the image so its bytes are inlined, "reference" stores the path instead.
        roots (Sequence[Path] | None): Directories that stored paths are made relative to.

    Returns:
        tuple[list[dict], list]: The conversational prompt and its image list.
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
                    images.append(_image_value(block["image"], image_mode, roots))
                    blocks.append({"type": "image"})
                else:
                    blocks.append(block)
            normalized.append({"role": role, "content": blocks})
        return normalized, images

    content = _content_with_image(row.get("image"), row["prompt"])
    if row.get("image"):
        images.append(_image_value(row["image"], image_mode, roots))
    return [{"role": "user", "content": content}], images


def _image_value(image_ref: str | Path, image_mode: str, roots: Sequence[Path] | None) -> object:
    """Turn a JSONL image reference into a dataset image entry.

    Args:
        image_ref (str | Path): The image path recorded in the row.
        image_mode (str): One of ``utils.image_loading.IMAGE_MODES``.
        roots (Sequence[Path] | None): Directories that stored paths are made relative to.

    Returns:
        object: A ``PIL.Image.Image`` in embed mode, else a path string.
    """
    if image_mode == "reference":
        return relative_image_path(image_ref, roots)
    # Resolve against the JSONL directories first: rows routinely record paths
    # relative to the corpus rather than to the working directory.
    resolved = resolve_image_path(image_ref, roots)
    with PILImage.open(resolved) as opened:
        opened.load()
        return opened


def _row_roots(inputs: Path | str | Sequence[Path | str]) -> list[Path]:
    """Collect the directories JSONL rows' relative image paths resolve against.

    Args:
        inputs (Path | str | Sequence[Path | str]): One path or several, as given to
            :func:`load_rows`.

    Returns:
        list[Path]: The parent directory of every discovered JSONL file.
    """
    candidates: Sequence[Path | str] = [inputs] if isinstance(inputs, (str, Path)) else list(inputs)
    return image_roots([path.parent for path in discover_jsonl(candidates)])


def to_sft_records(rows: list[dict], image_mode: str = "embed", roots: Sequence[Path] | None = None) -> list[dict]:
    """Convert correction rows to SFT prompt-completion records.

    Args:
        rows (list[dict]): Raw JSONL rows.
        image_mode (str): One of ``utils.image_loading.IMAGE_MODES``.
        roots (Sequence[Path] | None): Directories that stored paths are made relative to.

    Returns:
        list[dict]: The converted records.
    """
    records = []
    for row in rows:
        _validate_row(row, "sft")
        prompt, images = _build_prompt(row, image_mode, roots)
        completion = [{"role": "assistant", "content": [{"type": "text", "text": row["completion"]}]}]
        records.append({"prompt": prompt, "completion": completion, "images": images})
    return records


def to_dpo_records(rows: list[dict], image_mode: str = "embed", roots: Sequence[Path] | None = None) -> list[dict]:
    """Convert A/B comparison rows to DPO preference records.

    Args:
        rows (list[dict]): Raw JSONL rows.
        image_mode (str): One of ``utils.image_loading.IMAGE_MODES``.
        roots (Sequence[Path] | None): Directories that stored paths are made relative to.

    Returns:
        list[dict]: The converted records.
    """
    records = []
    for row in rows:
        _validate_row(row, "dpo")
        prompt, images = _build_prompt(row, image_mode, roots)
        chosen = [{"role": "assistant", "content": [{"type": "text", "text": row["chosen"]}]}]
        rejected = [{"role": "assistant", "content": [{"type": "text", "text": row["rejected"]}]}]
        records.append({"prompt": prompt, "chosen": chosen, "rejected": rejected, "images": images})
    return records


def to_kto_records(rows: list[dict], image_mode: str = "embed", roots: Sequence[Path] | None = None) -> list[dict]:
    """Convert good/bad feedback rows to KTO unpaired preference records.

    Args:
        rows (list[dict]): Raw JSONL rows.
        image_mode (str): One of ``utils.image_loading.IMAGE_MODES``.
        roots (Sequence[Path] | None): Directories that stored paths are made relative to.

    Returns:
        list[dict]: The converted records.
    """
    records = []
    for row in rows:
        _validate_row(row, "kto")
        prompt, images = _build_prompt(row, image_mode, roots)
        completion = [{"role": "assistant", "content": [{"type": "text", "text": row["completion"]}]}]
        records.append({"prompt": prompt, "completion": completion, "label": bool(row["label"]), "images": images})
    return records


def to_grpo_records(rows: list[dict], image_mode: str = "embed", roots: Sequence[Path] | None = None) -> list[dict]:
    """Convert feedback rows to GRPO prompt records.

    The ``ground_truth`` column is carried through so the reward functions can
    score generated completions against it (see :mod:`src.rewards`).

    Args:
        rows (list[dict]): Raw JSONL rows.
        image_mode (str): One of ``utils.image_loading.IMAGE_MODES``.
        roots (Sequence[Path] | None): Directories that stored paths are made relative to.

    Returns:
        list[dict]: The converted records.
    """
    records = []
    for row in rows:
        _validate_row(row, "grpo")
        prompt, images = _build_prompt(row, image_mode, roots)
        records.append({"prompt": prompt, "images": images, "ground_truth": row["ground_truth"]})
    return records


def to_sft(rows: list[dict], image_mode: str = "embed", roots: Sequence[Path] | None = None) -> Dataset:
    """Convert correction rows to SFT prompt-completion format."""
    return Dataset.from_list(to_sft_records(rows, image_mode, roots))


def to_dpo(rows: list[dict], image_mode: str = "embed", roots: Sequence[Path] | None = None) -> Dataset:
    """Convert A/B comparison rows to DPO preference format."""
    return Dataset.from_list(to_dpo_records(rows, image_mode, roots))


def to_kto(rows: list[dict], image_mode: str = "embed", roots: Sequence[Path] | None = None) -> Dataset:
    """Convert good/bad feedback rows to KTO unpaired preference format."""
    return Dataset.from_list(to_kto_records(rows, image_mode, roots))


def to_grpo(rows: list[dict], image_mode: str = "embed", roots: Sequence[Path] | None = None) -> Dataset:
    """Convert feedback rows to GRPO prompt format."""
    return Dataset.from_list(to_grpo_records(rows, image_mode, roots))


CONVERTERS = {
    "sft": to_sft,
    "dpo": to_dpo,
    "kto": to_kto,
    "grpo": to_grpo,
}

#: Record-level converters, used by :func:`main` so images can be stored lazily.
CONVERTER_RECORDS = {
    "sft": to_sft_records,
    "dpo": to_dpo_records,
    "kto": to_kto_records,
    "grpo": to_grpo_records,
}


def main() -> None:
    """Convert raw JSONL files or directories to a TRL-format HF dataset."""
    parser = argparse.ArgumentParser(description="Convert raw JSONL to a TRL-format HF dataset.")
    parser.add_argument(
        "--input",
        required=True,
        nargs="+",
        type=Path,
        help="Raw JSONL file(s) and/or director(ies); each directory contributes its *.jsonl files.",
    )
    parser.add_argument("--output", required=True, help="Path to save the HF dataset.")
    parser.add_argument("--method", required=True, choices=list(CONVERTERS), help="Training method.")
    add_image_mode_argument(parser)
    args = parser.parse_args()

    try:
        rows = load_rows(args.input)
        roots = _row_roots(args.input)
    except (FileNotFoundError, DiscoveryError) as exc:
        parser.error(str(exc))
    records = CONVERTER_RECORDS[args.method](rows, args.image_mode, roots)
    dataset = save_image_dataset(records, args.output, args.image_mode, roots)
    print(f"Converted {len(dataset)} rows to {args.method} format ({args.image_mode} images) -> {args.output}")


if __name__ == "__main__":
    main()
