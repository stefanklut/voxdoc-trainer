"""Build SFT training data for bounding-box <-> text grounding tasks.

This tool turns a directory of PAGE XML (``.xml``) and DocLang (``.dclg``)
files (plus their source images) into SFT examples in two modes:

- ``bbox_to_text``: the model sees the image and a bounding box and must
  produce the text inside the box.
- ``text_to_bbox``: the model sees the image and a line of text and must
  produce the bounding box for that text.

PAGE XML ``TextLine`` polygons are fit to axis-aligned bounding boxes. DocLang
``<text>`` elements use their four ``<location>`` values, interpreted as
``x_min, y_min, x_max, y_max`` (alternating axis order) in the document's
declared resolution space, relative to the top-left corner of the page (per
the DocLang spec).

Bboxes are serialized in one of two formats:

- ``qwen``: ``[x1, y1, x2, y2]`` integers normalized to the 0-1000 range
  (Qwen-VL grounding convention: ``round(pixel / dim * 1000)``).
- ``doclang``: the four ``<location value="N"/>`` elements in the document's
  native (declared) coordinate space — teaches the model to emit correct
  DocLang.

Normalization uses the source's declared size (PAGE XML
``imageWidth``/``imageHeight``; DocLang ``<default_resolution>``); when no
declared size is available, the actual image size is used.

When the total number of candidate (text, bbox) pairs across all input files
exceeds ``--max-lines``, a random subset of that size is sampled **globally**
(``--seed`` makes the selection reproducible).

Files that cannot be parsed, that contain no usable pairs, or that have no
matching source image are skipped with a warning.
"""

from __future__ import annotations

import argparse
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from datasets import Dataset
from PIL import Image as PILImage
from pydantic import BaseModel

sys.path.append(str(Path(__file__).resolve().parent.joinpath("..", "..")))
from src.doclang_structured.models import DefaultResolution, Text
from src.doclang_structured.parse import doclang_xml_to_model
from utils.logging_utils import get_logger, setup_logger
from utils.page_xml_editor import PageXMLEditor

logger = get_logger()

MODE_BBOX_TO_TEXT = "bbox_to_text"
MODE_TEXT_TO_BBOX = "text_to_bbox"
FORMAT_QWEN = "qwen"
FORMAT_DOCLANG = "doclang"

# Exception types that mark an input file as unparseable / invalid.
# ``doclang_xml_to_model`` raises ``ValueError`` for structural problems and
# ``pydantic.ValidationError`` (a ``ValueError`` subclass) for bad enum values,
# but ``TypeError`` (from ``int(None)``) when a required attribute such as
# ``thread_id`` or ``location@value`` is missing — so ``TypeError`` must be
# included or a single malformed file would crash the whole build.
_PARSE_ERRORS = (FileNotFoundError, ET.ParseError, ValueError, TypeError)


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


def polygon_to_bbox(points: np.ndarray) -> tuple[int, int, int, int]:
    """Fit an axis-aligned bounding box to a polygon.

    Args:
        points (np.ndarray): Polygon points with shape (N, 2) in (x, y) order.

    Returns:
        tuple[int, int, int, int]: (x1, y1, x2, y2) — top-left and bottom-right corners.
    """
    x1, y1 = int(points[:, 0].min()), int(points[:, 1].min())
    x2, y2 = int(points[:, 0].max()), int(points[:, 1].max())
    return (x1, y1, x2, y2)


def extract_page_xml_pairs(page_xml_path: Path) -> list[tuple[str, tuple[int, int, int, int]]]:
    """Extract (text, bbox) pairs from the TextLines of a PAGE XML file.

    Lines with empty text or without a ``Coords`` polygon are skipped.

    Args:
        page_xml_path (Path): Path to a PAGE XML file.

    Returns:
        list[tuple[str, tuple[int, int, int, int]]]: (text, (x1, y1, x2, y2)) pairs in document order.

    Raises:
        FileNotFoundError: If the file does not exist.
        ET.ParseError: If the file is not well-formed XML.
        ValueError: If the file is not a valid PAGE XML document.
    """
    editor = PageXMLEditor(page_xml_path)
    pairs: list[tuple[str, tuple[int, int, int, int]]] = []
    for line in editor.iterfind(".//TextLine"):
        text = editor.get_text(line).strip()
        if not text:
            continue
        try:
            bbox = polygon_to_bbox(editor.get_coords(line))
        except ValueError as exc:
            logger.warning("Skipping line %s in %s: %s", editor.get_id(line), page_xml_path.name, exc)
            continue
        pairs.append((text, bbox))
    return pairs


def _node_text(node: BaseModel) -> str:
    """Recursively concatenate all text content of a DocLang node."""
    text = getattr(node, "text", None)
    if isinstance(text, str):
        return text
    body = getattr(node, "body", None)
    if isinstance(body, list):
        return "".join(_node_text(child) for child in body)
    return ""


def _iter_text_nodes(node: BaseModel):
    """Yield every ``Text`` node in the subtree rooted at ``node`` (including ``node``)."""
    if isinstance(node, Text):
        yield node
    for name in type(node).model_fields:
        value = getattr(node, name)
        if isinstance(value, BaseModel):
            yield from _iter_text_nodes(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, BaseModel):
                    yield from _iter_text_nodes(item)


def _locations_to_corners(locations: list) -> tuple[int, int, int, int]:
    """Convert four DocLang ``<location>`` values to (x1, y1, x2, y2) corners.

    Per the DocLang spec, the four values are ``x_min, y_min, x_max, y_max``
    (alternating axis order) in the document's declared resolution space,
    relative to the top-left corner of the page.
    """
    x1, y1, x2, y2 = (location.value for location in locations)
    return (x1, y1, x2, y2)


def extract_doclang_pairs(doclang_path: Path) -> list[tuple[str, tuple[int, int, int, int]]]:
    """Extract (text, bbox) pairs from the ``<text>`` elements of a DocLang file.

    Only ``<text>`` elements with exactly four ``<location>`` values and
    non-empty text are included, at any nesting depth.

    Args:
        doclang_path (Path): Path to a DocLang (.dclg) file.

    Returns:
        list[tuple[str, tuple[int, int, int, int]]]: (text, (x1, y1, x2, y2)) pairs in document order.

    Raises:
        FileNotFoundError: If the file does not exist.
        ET.ParseError: If the file is not well-formed XML.
        ValueError: If the file is not a valid DocLang document.
        TypeError: If a required attribute (e.g. ``thread_id``, ``location@value``) is missing.
    """
    model = doclang_xml_to_model(doclang_path.read_text(encoding="utf-8"))
    pairs: list[tuple[str, tuple[int, int, int, int]]] = []
    for node in _iter_text_nodes(model):
        if len(node.head.locations) != 4:
            continue
        text = _node_text(node).strip()
        if not text:
            continue
        pairs.append((text, _locations_to_corners(node.head.locations)))
    return pairs


def declared_size(path: Path) -> tuple[int, int] | None:
    """Get the (width, height) declared by a PAGE XML or DocLang file.

    PAGE XML: the ``Page`` element's ``imageWidth``/``imageHeight``. DocLang:
    the ``<default_resolution width height/>`` item in ``<head>``.

    Args:
        path (Path): Path to a ``.xml`` or ``.dclg`` file.

    Returns:
        tuple[int, int] | None: (width, height), or None when no declared size is available.
    """
    if path.suffix == ".xml":
        try:
            height, width = PageXMLEditor(path).get_size()
        except _PARSE_ERRORS:
            return None
        return (width, height)
    try:
        model = doclang_xml_to_model(path.read_text(encoding="utf-8"))
    except _PARSE_ERRORS:
        return None
    if model.head is None:
        return None
    for item in model.head.items:
        if isinstance(item, DefaultResolution) and item.width is not None and item.height is not None:
            return (item.width, item.height)
    return None


def to_qwen_bbox(corners: tuple[int, int, int, int], size_wh: tuple[int, int]) -> list[int]:
    """Convert native-space corners to Qwen's 0-1000 normalized [x1, y1, x2, y2].

    Args:
        corners (tuple[int, int, int, int]): (x1, y1, x2, y2) in the source's native pixel space.
        size_wh (tuple[int, int]): (width, height) of the normalization space.

    Returns:
        list[int]: [x1, y1, x2, y2], each value clamped to [0, 1000].
    """
    x1, y1, x2, y2 = corners
    width, height = size_wh

    def normalize(value: int, dim: int) -> int:
        return max(0, min(1000, int(round(value / dim * 1000))))

    return [normalize(x1, width), normalize(y1, height), normalize(x2, width), normalize(y2, height)]


def format_qwen_bbox(bbox: list[int]) -> str:
    """Format a Qwen bbox as the ``[x1, y1, x2, y2]`` text the model should emit."""
    return "[" + ", ".join(str(value) for value in bbox) + "]"


def to_doclang_locations(corners: tuple[int, int, int, int]) -> str:
    """Serialize native-space corners as four DocLang ``<location>`` elements.

    The corners (x1, y1, x2, y2) map directly to the spec's
    ``x_min, y_min, x_max, y_max`` values.

    Args:
        corners (tuple[int, int, int, int]): (x1, y1, x2, y2) in the source's native pixel space.

    Returns:
        str: The four ``<location value="N"/>`` elements, concatenated.
    """
    x1, y1, x2, y2 = corners
    return (
        f'<location value="{x1}"/>'
        f'<location value="{y1}"/>'
        f'<location value="{x2}"/>'
        f'<location value="{y2}"/>'
    )


def _prompt_and_completion(mode: str, bbox_format: str, text: str, bbox_str: str) -> tuple[str, str]:
    """Build the prompt text and completion text for one example.

    Args:
        mode (str): "bbox_to_text" or "text_to_bbox".
        bbox_format (str): "qwen" or "doclang".
        text (str): The ground-truth text of the line.
        bbox_str (str): The serialized bbox (format-specific).

    Returns:
        tuple[str, str]: (prompt_text, completion_text).
    """
    if mode == MODE_BBOX_TO_TEXT:
        if bbox_format == FORMAT_QWEN:
            prompt = f"Transcribe the text inside the bounding box {bbox_str}."
        else:
            prompt = f"Transcribe the text inside the element with these DocLang locations: {bbox_str}."
        return prompt, text
    if bbox_format == FORMAT_QWEN:
        return f'Predict the bounding box for the text "{text}".', bbox_str
    prompt = f'Emit the DocLang <text> element with its <location> coordinates for the text: "{text}".'
    return prompt, f"<text>{bbox_str}{text}</text>"


def build_sft_record(
    mode: str,
    bbox_format: str,
    text: str,
    corners: tuple[int, int, int, int],
    size_wh: tuple[int, int],
    image_path: Path,
) -> dict:
    """Build a single SFT record for a bbox <-> text grounding example.

    Args:
        mode (str): "bbox_to_text" or "text_to_bbox".
        bbox_format (str): "qwen" or "doclang".
        text (str): The ground-truth text of the line.
        corners (tuple[int, int, int, int]): (x1, y1, x2, y2) in the source's native space.
        size_wh (tuple[int, int]): (width, height) of the normalization space.
        image_path (Path): Path to the source document image.

    Returns:
        dict: An SFT record with prompt, completion, and images.
    """
    if bbox_format == FORMAT_QWEN:
        bbox_str = format_qwen_bbox(to_qwen_bbox(corners, size_wh))
    else:
        bbox_str = to_doclang_locations(corners)
    prompt_text, completion_text = _prompt_and_completion(mode, bbox_format, text, bbox_str)
    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    completion = [{"role": "assistant", "content": [{"type": "text", "text": completion_text}]}]
    images = [PILImage.open(image_path)]
    return {"prompt": prompt, "completion": completion, "images": images}


def build_dataset(
    input_dir: Path,
    image_dir: Path,
    mode: str,
    bbox_format: str,
    max_lines: int | None = None,
    seed: int | None = None,
) -> tuple[list[dict], int]:
    """Build SFT records from a directory of PAGE XML and DocLang files.

    All candidate (text, bbox) pairs are collected across every input file
    first; when ``max_lines`` is given and the pool is larger, a random subset
    of that size is sampled globally (``seed`` makes the selection
    reproducible).

    Args:
        input_dir (Path): Directory of PAGE XML (.xml) and DocLang (.dclg) files.
        image_dir (Path): Directory of source document images.
        mode (str): "bbox_to_text" or "text_to_bbox".
        bbox_format (str): "qwen" or "doclang".
        max_lines (int | None): If set, keep at most this many (text, bbox) pairs, sampled randomly from the global pool.
        seed (int | None): Random seed for reproducible sampling.

    Returns:
        tuple[list[dict], int]: The list of SFT records and the number of skipped files.
    """
    candidates: list[tuple[str, tuple[int, int, int, int], Path, tuple[int, int]]] = []
    skipped = 0
    files = sorted(input_dir.glob("*.xml")) + sorted(input_dir.glob("*.dclg"))
    for path in files:
        try:
            if path.suffix == ".xml":
                pairs = extract_page_xml_pairs(path)
            else:
                pairs = extract_doclang_pairs(path)
        except _PARSE_ERRORS as exc:
            logger.warning("Skipping %s: could not parse (%s)", path.name, exc)
            skipped += 1
            continue
        if not pairs:
            logger.warning("Skipping %s: no usable (text, bbox) pairs found", path.name)
            skipped += 1
            continue
        image_path = find_image(image_dir, path.stem)
        if image_path is None:
            logger.warning("Skipping %s: no source image found in %s", path.name, image_dir)
            skipped += 1
            continue
        size_wh = declared_size(path)
        if size_wh is None:
            with PILImage.open(image_path) as image:
                size_wh = (image.width, image.height)
        for text, corners in pairs:
            candidates.append((text, corners, image_path, size_wh))

    if max_lines is not None and len(candidates) > max_lines:
        rng = random.Random(seed)
        candidates = rng.sample(candidates, max_lines)
        rng.shuffle(candidates)

    records = [
        build_sft_record(mode, bbox_format, text, corners, size_wh, image_path)
        for text, corners, image_path, size_wh in candidates
    ]
    return records, skipped


def main() -> None:
    """Build an SFT dataset of bbox <-> text grounding examples."""
    setup_logger()
    parser = argparse.ArgumentParser(
        description="Build SFT bbox <-> text grounding training data from PAGE XML and DocLang files."
    )
    parser.add_argument("--input", required=True, help="Directory of PAGE XML (.xml) and DocLang (.dclg) files.")
    parser.add_argument("--output", required=True, help="Path to save the HF dataset.")
    parser.add_argument("--images", required=True, help="Directory of source document images.")
    parser.add_argument(
        "--mode",
        required=True,
        choices=[MODE_BBOX_TO_TEXT, MODE_TEXT_TO_BBOX],
        help="bbox_to_text: predict the text inside a given box. text_to_bbox: predict the box for a given text.",
    )
    parser.add_argument(
        "--bbox-format",
        required=True,
        choices=[FORMAT_QWEN, FORMAT_DOCLANG],
        help="qwen: [x1, y1, x2, y2] normalized to 0-1000. doclang: four <location> elements in native space.",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=None,
        help="If set, randomly sample at most this many (text, bbox) pairs from the global pool of all input files.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible --max-lines sampling.")
    args = parser.parse_args()

    if args.max_lines is not None and args.max_lines < 1:
        parser.error("--max-lines must be >= 1")

    input_dir = Path(args.input)
    image_dir = Path(args.images)
    records, skipped = build_dataset(input_dir, image_dir, args.mode, args.bbox_format, args.max_lines, args.seed)

    dataset = Dataset.from_list(records)
    dataset.save_to_disk(args.output)
    logger.info("Built %d %s SFT examples (%s format) -> %s", len(dataset), args.mode, args.bbox_format, args.output)
    if args.max_lines is not None:
        logger.info("Sampled at most %d lines globally (seed=%s)", args.max_lines, args.seed)
    if skipped:
        logger.info("Skipped %d file(s)", skipped)


if __name__ == "__main__":
    main()
