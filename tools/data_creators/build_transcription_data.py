"""Build SFT training data that teaches a model to transcribe documents to plain unicode.

This tool turns a directory of PAGE XML files (plus their source images) into SFT
examples: the model sees a document image and must produce the document's text as
plain unicode. The completion is the transcription extracted from the PAGE XML
ground truth, with lines ordered by their reading order when available.

Each PAGE XML file is parsed with ``utils.page_xml_editor.PageXMLEditor``. Files
that cannot be parsed, that contain no text, or that have no matching source image
are skipped with a warning.

Note: PAGE XML is the PAGE (Page Analysis and Grounding) ground-truth format.
Documents use the ``.xml`` extension.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from datasets import Dataset
from PIL import Image as PILImage

sys.path.append(str(Path(__file__).resolve().parent.joinpath("..", "..")))
from utils.logging_utils import get_logger, setup_logger
from utils.page_xml_editor import PageXMLEditor

logger = get_logger()

PROMPT_TEXT = "Transcribe the text in this document."


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


def extract_transcription(page_xml_path: Path) -> str:
    """Extract the unicode transcription from a PAGE XML file.

    Args:
        page_xml_path (Path): Path to a PAGE XML file.

    Returns:
        str: The transcription text (may be empty if the file has no text lines).

    Raises:
        FileNotFoundError: If the file does not exist.
        ET.ParseError: If the file is not well-formed XML.
        ValueError: If the file is not a valid PAGE XML document.
    """
    editor = PageXMLEditor(page_xml_path)
    return editor.get_transcription_text()


def build_sft_record(transcription: str, image_path: Path) -> dict:
    """Build a single SFT record teaching the model to transcribe a document.

    Args:
        transcription (str): The ground-truth unicode transcription.
        image_path (Path): Path to the source document image.

    Returns:
        dict: An SFT record with prompt, completion (the transcription), and images.
    """
    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": PROMPT_TEXT},
            ],
        }
    ]
    completion = [{"role": "assistant", "content": [{"type": "text", "text": transcription}]}]
    images = [PILImage.open(image_path)]
    return {"prompt": prompt, "completion": completion, "images": images}


def build_dataset(input_dir: Path, image_dir: Path) -> tuple[list[dict], int]:
    """Build SFT records from a directory of PAGE XML files.

    Each file is parsed before it is added as an SFT record. Files that cannot be
    parsed, that contain no text, or that have no matching source image are skipped
    with a warning.

    Args:
        input_dir (Path): Directory of PAGE XML (.xml) files.
        image_dir (Path): Directory of source document images.

    Returns:
        tuple[list[dict], int]: The list of SFT records and the number of skipped files.
    """
    records: list[dict] = []
    skipped = 0
    for page_xml_file in sorted(input_dir.glob("*.xml")):
        try:
            transcription = extract_transcription(page_xml_file)
        except (FileNotFoundError, ET.ParseError, ValueError) as exc:
            logger.warning("Skipping %s: could not parse PAGE XML (%s)", page_xml_file.name, exc)
            skipped += 1
            continue
        if not transcription:
            logger.warning("Skipping %s: no transcription text found", page_xml_file.name)
            skipped += 1
            continue
        image_path = find_image(image_dir, page_xml_file.stem)
        if image_path is None:
            logger.warning("Skipping %s: no source image found in %s", page_xml_file.name, image_dir)
            skipped += 1
            continue
        records.append(build_sft_record(transcription, image_path))
    return records, skipped


def main() -> None:
    """Build an SFT dataset from a directory of PAGE XML files."""
    setup_logger()
    parser = argparse.ArgumentParser(description="Build SFT transcription training data from PAGE XML files.")
    parser.add_argument("--input", required=True, help="Directory of PAGE XML (.xml) files.")
    parser.add_argument("--output", required=True, help="Path to save the HF dataset.")
    parser.add_argument("--images", required=True, help="Directory of source document images.")
    args = parser.parse_args()

    input_dir = Path(args.input)
    image_dir = Path(args.images)
    records, skipped = build_dataset(input_dir, image_dir)

    dataset = Dataset.from_list(records)
    dataset.save_to_disk(args.output)
    logger.info("Built %d transcription SFT examples -> %s", len(dataset), args.output)
    if skipped:
        logger.info("Skipped %d PAGE XML file(s)", skipped)


if __name__ == "__main__":
    main()
