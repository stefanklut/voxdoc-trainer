"""Build SFT training data that teaches a model to emit DocLang natively.

DocLang (https://doclang.ai/) is an AI-native document format designed for LLMs
to consume directly. If a model does not support it natively, we teach it to by
giving it many examples of existing OCR already in DocLang format.

This tool turns a directory of existing DocLang documents and their source
images into SFT examples: the model sees a document image and must produce the
DocLang representation as its completion. The completion is the raw DocLang
markup, so the model learns the format itself.

Every document is validated with the official `doclang` package before it is
added as an SFT record. Documents that fail validation, or that have no matching
source image, are skipped with a warning.

Note: DocLang is an open standard (Linux Foundation). Documents use the `.dclg`
extension.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from datasets import Dataset
from doclang import ValidationError, validate
from PIL import Image as PILImage


def find_image(image_dir: Path, stem: str) -> Path | None:
    """Find a source image for a document in any format PIL can open.

    Args:
        image_dir (Path): Directory to search for the source document image.
        stem (str): Document stem to match against image filenames.

    Returns:
        Path | None: The first matching image path, or None if no supported image exists.
    """
    for extension in sorted(PILImage.registered_extensions()):
        candidate = image_dir / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    return None


def build_sft_record(doclang_path: Path, image_path: Path) -> dict:
    """Build a single SFT record teaching the model to emit DocLang.

    Args:
        doclang_path (Path): Path to an existing DocLang document.
        image_path (Path): Path to the source document image.

    Returns:
        dict: An SFT record with prompt, completion (the DocLang markup), and images.
    """
    doclang_text = doclang_path.read_text(encoding="utf-8").strip()

    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Transcribe this historical document into DocLang format."},
            ],
        }
    ]
    completion = [{"role": "assistant", "content": [{"type": "text", "text": doclang_text}]}]
    images = [PILImage.open(image_path)]
    return {"prompt": prompt, "completion": completion, "images": images}


def build_dataset(
    input_dir: Path,
    image_dir: Path,
    validator: Callable[[Path], None] = validate,
) -> tuple[list[dict], int]:
    """Build SFT records from a directory of DocLang documents.

    Each document is validated before it is added as an SFT record; documents
    that fail validation or have no matching source image are skipped with a
    warning.

    Args:
        input_dir (Path): Directory of DocLang (.dclg) files.
        image_dir (Path): Directory of source document images.
        validator (Callable[[Path], None]): Validation callable invoked per document;
            should raise on invalid input. Defaults to the official `doclang.validate`.

    Returns:
        tuple[list[dict], int]: The list of SFT records and the number of skipped files.
    """
    records: list[dict] = []
    skipped = 0
    for doclang_file in sorted(input_dir.glob("*.dclg")):
        try:
            validator(doclang_file)
        except ValidationError as exc:
            print(f"Skipping {doclang_file.name}: invalid DocLang ({exc})")
            skipped += 1
            continue
        image_path = find_image(image_dir, doclang_file.stem)
        if image_path is None:
            print(f"Skipping {doclang_file.name}: no source image found in {image_dir}")
            skipped += 1
            continue
        records.append(build_sft_record(doclang_file, image_path))
    return records, skipped


def main() -> None:
    """Build an SFT dataset from a directory of existing DocLang documents."""
    parser = argparse.ArgumentParser(description="Build SFT training data from existing DocLang documents.")
    parser.add_argument("--input", required=True, help="Directory of DocLang (.dclg) files.")
    parser.add_argument("--output", required=True, help="Path to save the HF dataset.")
    parser.add_argument("--images", required=True, help="Directory of source document images.")
    args = parser.parse_args()

    input_dir = Path(args.input)
    image_dir = Path(args.images)
    records, skipped = build_dataset(input_dir, image_dir)

    dataset = Dataset.from_list(records)
    dataset.save_to_disk(args.output)
    print(f"Built {len(dataset)} DocLang SFT examples -> {args.output}")
    if skipped:
        print(f"Skipped {skipped} DocLang file(s)")


if __name__ == "__main__":
    main()
