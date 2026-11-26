"""Build SFT training data that teaches a model to transcribe documents to plain unicode.

This tool turns PAGE XML files (plus their source images) into SFT examples: the
model sees a document image and must produce the document's text as plain unicode.
The completion is the transcription extracted from the PAGE XML ground truth, with
lines ordered by their reading order when available.

``--input`` accepts one or more corpus roots and is searched recursively: PAGE XML
files are discovered inside directories named ``page`` at any depth, so a corpus
laid out as ``corpus/document/page/*.xml`` works without pointing at each
directory. ``--images`` is optional and also accepts multiple roots; when it is
omitted, images are found from the corpus layout itself (the PAGE XML
``imageFilename`` attribute, or the directory above the ``page`` directory).

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
from collections.abc import Sequence
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.joinpath("..", "..")))
from utils.discovery import (  # noqa: E402, F401  (find_image is re-exported for callers)
    DiscoveryError,
    discover_page_xml,
    find_image,
    resolve_image,
)
from utils.image_loading import (  # noqa: E402
    add_image_mode_argument,
    image_field,
    image_roots,
    save_image_dataset,
)
from utils.logging_utils import get_logger, setup_logger
from utils.page_xml_editor import PageXMLEditor

logger = get_logger()

PROMPT_TEXT = "Transcribe the text in this document."


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


def build_sft_record(
    transcription: str,
    image_path: Path,
    image_mode: str = "embed",
    roots: Sequence[Path] | None = None,
) -> dict:
    """Build a single SFT record teaching the model to transcribe a document.

    Args:
        transcription (str): The ground-truth unicode transcription.
        image_path (Path): Path to the source document image.
        image_mode (str): One of ``utils.image_loading.IMAGE_MODES``; "embed" inlines
            the image bytes, "reference" stores the image path instead.
        roots (Sequence[Path] | None): Corpus roots that stored paths are made relative to.

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
    images = image_field(image_path, image_mode, roots)
    return {"prompt": prompt, "completion": completion, "images": images}


def build_dataset(
    input_dir: Path | Sequence[Path],
    image_dir: Path | Sequence[Path] | None = None,
    *,
    image_mode: str = "embed",
) -> tuple[list[dict], int]:
    """Build SFT records from the PAGE XML files discovered under the input roots.

    Each file is parsed before it is added as an SFT record. Files that cannot be
    parsed, that contain no text, or that have no matching source image are skipped
    with a warning.

    Args:
        input_dir (Path | Sequence[Path]): Corpus root(s); PAGE XML is discovered in
            ``page`` directories at any depth.
        image_dir (Path | Sequence[Path] | None): Explicit source image directories.
            When omitted, images are located from the corpus layout.
        image_mode (str): One of ``utils.image_loading.IMAGE_MODES``. "embed" (the
            default) stores image bytes in the dataset; "reference" stores paths.

    Returns:
        tuple[list[dict], int]: The list of SFT records and the number of skipped files.

    Raises:
        DiscoveryError: If no PAGE XML file is found under the input roots.
    """
    records: list[dict] = []
    skipped = 0
    roots = image_roots(input_dir, image_dir)
    for page_xml_file in discover_page_xml(input_dir):
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
        image_path = resolve_image(page_xml_file, image_dir)
        if image_path is None:
            logger.warning("Skipping %s: no source image found", page_xml_file)
            skipped += 1
            continue
        records.append(build_sft_record(transcription, image_path, image_mode, roots))
    return records, skipped


def main() -> None:
    """Build an SFT dataset from the PAGE XML files of one or more corpus roots."""
    setup_logger()
    parser = argparse.ArgumentParser(description="Build SFT transcription training data from PAGE XML files.")
    parser.add_argument(
        "--input",
        required=True,
        nargs="+",
        type=Path,
        help="Corpus root(s); PAGE XML files are discovered in 'page' directories at any depth.",
    )
    parser.add_argument("--output", required=True, help="Path to save the HF dataset.")
    parser.add_argument(
        "--images",
        nargs="+",
        type=Path,
        default=None,
        help=(
            "Optional source image director(ies). When omitted, images are found from the corpus "
            "layout (the PAGE XML imageFilename attribute, or the directory above the page directory)."
        ),
    )
    add_image_mode_argument(parser)
    args = parser.parse_args()

    try:
        records, skipped = build_dataset(args.input, args.images, image_mode=args.image_mode)
    except (DiscoveryError, NotADirectoryError) as exc:
        parser.error(str(exc))

    dataset = save_image_dataset(records, args.output, args.image_mode, image_roots(args.input, args.images))
    logger.info("Built %d transcription SFT examples (%s images) -> %s", len(dataset), args.image_mode, args.output)
    if skipped:
        logger.info("Skipped %d PAGE XML file(s)", skipped)


if __name__ == "__main__":
    main()
