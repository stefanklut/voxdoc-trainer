"""Build SFT training data that teaches a model to emit DocLang natively.

DocLang (https://doclang.ai/) is an AI-native XML document format designed for
LLMs to consume directly. If a model does not support it natively, we teach it to
by giving it many examples of existing OCR already in DocLang format.

This tool turns a directory of existing DocLang XML documents (plus their source
images) into SFT examples: the model sees a document image and must produce the
DocLang representation as its completion. The completion is the raw DocLang XML, so
the model learns the format itself.

Note: DocLang is an open standard (Linux Foundation). The reference
implementation may be early-stage; this tool reads the XML directly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset
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


def build_sft_record(xml_path: Path, image_path: Path | None) -> dict:
    """Build a single SFT record teaching the model to emit DocLang.

    Args:
        xml_path (Path): Path to an existing DocLang XML document.
        image_path (Path | None): Optional path to the source document image.

    Returns:
        dict: An SFT record with prompt, completion (the DocLang XML), and images.
    """
    doclang_xml = xml_path.read_text(encoding="utf-8").strip()

    prompt = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Transcribe this historical document into DocLang format."}],
        }
    ]
    completion = [{"role": "assistant", "content": [{"type": "text", "text": doclang_xml}]}]
    images = [PILImage.open(image_path)] if image_path else []
    return {"prompt": prompt, "completion": completion, "images": images}


def main() -> None:
    """Build an SFT dataset from a directory of existing DocLang XML documents."""
    parser = argparse.ArgumentParser(description="Build SFT training data from existing DocLang XML documents.")
    parser.add_argument("--input", required=True, help="Directory of DocLang XML files.")
    parser.add_argument("--output", required=True, help="Path to save the HF dataset.")
    parser.add_argument("--images", default=None, help="Optional directory of source document images.")
    args = parser.parse_args()

    input_dir = Path(args.input)
    image_dir = Path(args.images) if args.images else None
    records = []
    for xml_file in sorted(input_dir.glob("*.xml")):
        image_path = find_image(image_dir, xml_file.stem) if image_dir else None
        records.append(build_sft_record(xml_file, image_path))

    dataset = Dataset.from_list(records)
    dataset.save_to_disk(args.output)
    print(f"Built {len(dataset)} DocLang SFT examples -> {args.output}")


if __name__ == "__main__":
    main()
