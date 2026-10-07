"""Build SFT training data for bounding-box <-> text grounding tasks.

This tool turns a directory of PAGE XML (``.xml``) and DocLang (``.dclg``)
files (plus their source images) into SFT examples.

Line-level modes (PAGE XML and DocLang):

- ``bbox_to_text``: the model sees the image and a bounding box and must
  produce the text inside the box.
- ``text_to_bbox``: the model sees the image and a line of text and must
  produce the bounding box for that text.

Region-based modes (PAGE XML only — DocLang has no line structure). These
teach spatial/structural reasoning over the ``TextRegion`` -> ``TextLine``
hierarchy; the reading order is the document order of the lines within a
region (the only order that is guaranteed):

- ``line_neighbor``: given a line (by text or bbox) and an offset, produce
  the text or bbox of the Nth line above/below it (``--max-n`` bounds the
  offset).
- ``region_to_lines``: given a paragraph's bbox, list all its line bboxes in
  reading order.
- ``lines_to_region``: given a paragraph's line bboxes, produce the
  paragraph's bbox.
- ``region_to_transcription``: given a paragraph's bbox, transcribe all its
  lines in reading order.
- ``line_index``: given a line (by text or bbox), state its position in the
  paragraph, e.g. "line 3 of 7".
- ``line_ordering``: given a paragraph's line bboxes in shuffled order, give
  the reading order as 1-based indices.
- ``line_count``: given a paragraph's bbox, state how many lines it contains.

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
from dataclasses import dataclass
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
MODE_LINE_NEIGHBOR = "line_neighbor"
MODE_REGION_TO_LINES = "region_to_lines"
MODE_LINES_TO_REGION = "lines_to_region"
MODE_REGION_TO_TRANSCRIPTION = "region_to_transcription"
MODE_LINE_INDEX = "line_index"
MODE_LINE_ORDERING = "line_ordering"
MODE_LINE_COUNT = "line_count"
FORMAT_QWEN = "qwen"
FORMAT_DOCLANG = "doclang"

# Region-based modes operate on the PAGE XML TextRegion -> TextLine hierarchy
# and therefore only process .xml files (DocLang has no line structure).
REGION_MODES = frozenset(
    {
        MODE_LINE_NEIGHBOR,
        MODE_REGION_TO_LINES,
        MODE_LINES_TO_REGION,
        MODE_REGION_TO_TRANSCRIPTION,
        MODE_LINE_INDEX,
        MODE_LINE_ORDERING,
        MODE_LINE_COUNT,
    }
)

# Exception types that mark an input file as unparseable / invalid.
# ``doclang_xml_to_model`` raises ``ValueError`` for structural problems and
# ``pydantic.ValidationError`` (a ``ValueError`` subclass) for bad enum values,
# but ``TypeError`` (from ``int(None)``) when a required attribute such as
# ``thread_id`` or ``location@value`` is missing — so ``TypeError`` must be
# included or a single malformed file would crash the whole build.
_PARSE_ERRORS = (FileNotFoundError, ET.ParseError, ValueError, TypeError)


@dataclass
class TextLineData:
    """A text line with its axis-aligned bounding box."""

    text: str
    bbox: tuple[int, int, int, int]


@dataclass
class RegionData:
    """A text region (paragraph) with its text lines in reading order."""

    bbox: tuple[int, int, int, int]
    lines: list[TextLineData]


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


def extract_page_xml_regions(page_xml_path: Path) -> list[RegionData]:
    """Extract text regions with their ordered text lines from a PAGE XML file.

    Each ``TextRegion`` yields a :class:`RegionData` whose ``lines`` are the
    region's ``TextLine`` elements in document order (the reading order within
    a paragraph). Lines with empty text or without a ``Coords`` polygon are
    skipped; regions without a ``Coords`` polygon or without any usable line
    are skipped.

    Args:
        page_xml_path (Path): Path to a PAGE XML file.

    Returns:
        list[RegionData]: Regions in document order, each with its lines in reading order.

    Raises:
        FileNotFoundError: If the file does not exist.
        ET.ParseError: If the file is not well-formed XML.
        ValueError: If the file is not a valid PAGE XML document.
    """
    editor = PageXMLEditor(page_xml_path)
    regions: list[RegionData] = []
    for region in editor.iterfind(".//TextRegion"):
        try:
            region_bbox = polygon_to_bbox(editor.get_coords(region))
        except ValueError as exc:
            logger.warning("Skipping region %s in %s: %s", editor.get_id(region), page_xml_path.name, exc)
            continue
        lines: list[TextLineData] = []
        for line in region.iterfind(".//TextLine"):
            text = editor.get_text(line).strip()
            if not text:
                continue
            try:
                line_bbox = polygon_to_bbox(editor.get_coords(line))
            except ValueError as exc:
                logger.warning("Skipping line %s in %s: %s", editor.get_id(line), page_xml_path.name, exc)
                continue
            lines.append(TextLineData(text=text, bbox=line_bbox))
        if lines:
            regions.append(RegionData(bbox=region_bbox, lines=lines))
    return regions


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
    return f'<location value="{x1}"/>' f'<location value="{y1}"/>' f'<location value="{x2}"/>' f'<location value="{y2}"/>'


def format_bbox(corners: tuple[int, int, int, int], bbox_format: str, size_wh: tuple[int, int]) -> str:
    """Format a single bbox in the given serialization.

    Args:
        corners (tuple[int, int, int, int]): (x1, y1, x2, y2) in the source's native pixel space.
        bbox_format (str): "qwen" or "doclang".
        size_wh (tuple[int, int]): (width, height) of the normalization space.

    Returns:
        str: The serialized bbox.
    """
    if bbox_format == FORMAT_QWEN:
        return format_qwen_bbox(to_qwen_bbox(corners, size_wh))
    return to_doclang_locations(corners)


def format_bbox_list(
    corners_list: list[tuple[int, int, int, int]],
    bbox_format: str,
    size_wh: tuple[int, int],
) -> str:
    """Format a list of bboxes as a ``[[x1, y1, x2, y2], ...]`` array.

    In ``qwen`` format each box is normalized to the 0-1000 range; in
    ``doclang`` format the native-space integers are used.

    Args:
        corners_list (list[tuple[int, int, int, int]]): (x1, y1, x2, y2) corners in native space.
        bbox_format (str): "qwen" or "doclang".
        size_wh (tuple[int, int]): (width, height) of the normalization space.

    Returns:
        str: The serialized list of bboxes.
    """
    if bbox_format == FORMAT_QWEN:
        elements = [format_qwen_bbox(to_qwen_bbox(corners, size_wh)) for corners in corners_list]
    else:
        elements = [f"[{x1}, {y1}, {x2}, {y2}]" for x1, y1, x2, y2 in corners_list]
    return "[" + ", ".join(elements) + "]"


def _ordinal(n: int) -> str:
    """Format an integer as an English ordinal (1st, 2nd, 3rd, 4th, ...)."""
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


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


def _make_record(prompt_text: str, completion_text: str, image_path: Path) -> dict:
    """Build an SFT record with an image + text prompt and a text completion.

    Args:
        prompt_text (str): The user prompt text.
        completion_text (str): The assistant completion text.
        image_path (Path): Path to the source document image.

    Returns:
        dict: An SFT record with prompt, completion, and images.
    """
    return {
        "prompt": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ],
        "completion": [{"role": "assistant", "content": [{"type": "text", "text": completion_text}]}],
        "images": [PILImage.open(image_path)],
    }


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
    return _make_record(prompt_text, completion_text, image_path)


def build_line_neighbor_candidates(
    region: RegionData,
    bbox_format: str,
    size_wh: tuple[int, int],
    max_n: int,
) -> list[tuple[str, str]]:
    """Build (prompt, completion) pairs for the ``line_neighbor`` mode.

    For every line and every offset ``n`` in ``1..max_n`` in both directions
    (where the target line exists), four variants are emitted: the reference
    line is given by its text or its bounding box, and the answer is the
    target line's text or its bounding box.

    Args:
        region (RegionData): The text region with its lines in reading order.
        bbox_format (str): "qwen" or "doclang".
        size_wh (tuple[int, int]): (width, height) of the normalization space.
        max_n (int): Maximum line offset to consider.

    Returns:
        list[tuple[str, str]]: (prompt, completion) pairs.
    """
    candidates: list[tuple[str, str]] = []
    n_lines = len(region.lines)
    for i, line in enumerate(region.lines):
        for direction in ("above", "below"):
            for n in range(1, max_n + 1):
                j = i - n if direction == "above" else i + n
                if not 0 <= j < n_lines:
                    continue
                target = region.lines[j]
                references = (
                    f'the line "{line.text}"',
                    f"the line with bounding box {format_bbox(line.bbox, bbox_format, size_wh)}",
                )
                for reference in references:
                    candidates.append(
                        (
                            f"What is the text of the {_ordinal(n)} line {direction} {reference}?",
                            target.text,
                        )
                    )
                    candidates.append(
                        (
                            f"What is the bounding box of the {_ordinal(n)} line {direction} {reference}?",
                            format_bbox(target.bbox, bbox_format, size_wh),
                        )
                    )
    return candidates


def build_region_to_lines_candidates(
    region: RegionData,
    bbox_format: str,
    size_wh: tuple[int, int],
) -> list[tuple[str, str]]:
    """Build (prompt, completion) pairs for the ``region_to_lines`` mode.

    The prompt names the paragraph's bounding box; the completion lists the
    bounding boxes of all its text lines in reading order. Regions with fewer
    than two lines yield no candidates.

    Args:
        region (RegionData): The text region with its lines in reading order.
        bbox_format (str): "qwen" or "doclang".
        size_wh (tuple[int, int]): (width, height) of the normalization space.

    Returns:
        list[tuple[str, str]]: (prompt, completion) pairs.
    """
    if len(region.lines) < 2:
        return []
    prompt = (
        f"List the bounding boxes of all text lines in the paragraph "
        f"{format_bbox(region.bbox, bbox_format, size_wh)}, in reading order."
    )
    completion = format_bbox_list([line.bbox for line in region.lines], bbox_format, size_wh)
    return [(prompt, completion)]


def build_lines_to_region_candidates(
    region: RegionData,
    bbox_format: str,
    size_wh: tuple[int, int],
) -> list[tuple[str, str]]:
    """Build (prompt, completion) pairs for the ``lines_to_region`` mode.

    The prompt lists the bounding boxes of all the paragraph's text lines; the
    completion is the paragraph's bounding box. Regions with fewer than two
    lines yield no candidates.

    Args:
        region (RegionData): The text region with its lines in reading order.
        bbox_format (str): "qwen" or "doclang".
        size_wh (tuple[int, int]): (width, height) of the normalization space.

    Returns:
        list[tuple[str, str]]: (prompt, completion) pairs.
    """
    if len(region.lines) < 2:
        return []
    prompt = (
        f"Given the text lines "
        f"{format_bbox_list([line.bbox for line in region.lines], bbox_format, size_wh)}, "
        f"what is the bounding box of the paragraph they belong to?"
    )
    completion = format_bbox(region.bbox, bbox_format, size_wh)
    return [(prompt, completion)]


def build_region_to_transcription_candidates(
    region: RegionData,
    bbox_format: str,
    size_wh: tuple[int, int],
) -> list[tuple[str, str]]:
    """Build (prompt, completion) pairs for the ``region_to_transcription`` mode.

    The prompt names the paragraph's bounding box; the completion is all its
    lines' text in reading order, joined by newlines. Regions with fewer than
    two lines yield no candidates.

    Args:
        region (RegionData): The text region with its lines in reading order.
        bbox_format (str): "qwen" or "doclang".
        size_wh (tuple[int, int]): (width, height) of the normalization space.

    Returns:
        list[tuple[str, str]]: (prompt, completion) pairs.
    """
    if len(region.lines) < 2:
        return []
    prompt = (
        f"Transcribe all text lines in the paragraph " f"{format_bbox(region.bbox, bbox_format, size_wh)}, in reading order."
    )
    completion = "\n".join(line.text for line in region.lines)
    return [(prompt, completion)]


def build_line_index_candidates(
    region: RegionData,
    bbox_format: str,
    size_wh: tuple[int, int],
) -> list[tuple[str, str]]:
    """Build (prompt, completion) pairs for the ``line_index`` mode.

    For every line, two variants are emitted (reference by text and by
    bounding box); the answer is the line's 1-based position in the paragraph,
    e.g. "line 3 of 7". Regions with fewer than two lines yield no candidates.

    Args:
        region (RegionData): The text region with its lines in reading order.
        bbox_format (str): "qwen" or "doclang".
        size_wh (tuple[int, int]): (width, height) of the normalization space.

    Returns:
        list[tuple[str, str]]: (prompt, completion) pairs.
    """
    n_lines = len(region.lines)
    if n_lines < 2:
        return []
    candidates: list[tuple[str, str]] = []
    for i, line in enumerate(region.lines):
        position = f"line {i + 1} of {n_lines}"
        candidates.append((f'What is the position of the line "{line.text}" in its paragraph?', position))
        candidates.append(
            (
                f"What is the position of the line with bounding box "
                f"{format_bbox(line.bbox, bbox_format, size_wh)} in its paragraph?",
                position,
            )
        )
    return candidates


def build_line_ordering_candidates(
    region: RegionData,
    bbox_format: str,
    size_wh: tuple[int, int],
    rng: random.Random,
) -> list[tuple[str, str]]:
    """Build (prompt, completion) pairs for the ``line_ordering`` mode.

    The prompt lists the paragraph's line bboxes in a shuffled order; the
    completion is the reading order expressed as 1-based positions in the
    presented list. Regions with fewer than three lines yield no candidates.

    Args:
        region (RegionData): The text region with its lines in reading order.
        bbox_format (str): "qwen" or "doclang".
        size_wh (tuple[int, int]): (width, height) of the normalization space.
        rng (random.Random): RNG used to shuffle the presented order.

    Returns:
        list[tuple[str, str]]: (prompt, completion) pairs.
    """
    n_lines = len(region.lines)
    if n_lines < 3:
        return []
    line_bboxes = [line.bbox for line in region.lines]
    order = list(range(n_lines))
    shuffled = order[:]
    rng.shuffle(shuffled)
    if shuffled == order:
        shuffled = order[::-1]
    presented = [line_bboxes[k] for k in shuffled]
    prompt = (
        f"These are the text lines of a paragraph in random order: "
        f"{format_bbox_list(presented, bbox_format, size_wh)}. "
        f"Give their reading order as 1-based indices."
    )
    completion = ", ".join(str(shuffled.index(position) + 1) for position in order)
    return [(prompt, completion)]


def build_line_count_candidates(
    region: RegionData,
    bbox_format: str,
    size_wh: tuple[int, int],
) -> list[tuple[str, str]]:
    """Build (prompt, completion) pairs for the ``line_count`` mode.

    The prompt names the paragraph's bounding box; the completion is the
    number of text lines it contains.

    Args:
        region (RegionData): The text region with its lines in reading order.
        bbox_format (str): "qwen" or "doclang".
        size_wh (tuple[int, int]): (width, height) of the normalization space.

    Returns:
        list[tuple[str, str]]: (prompt, completion) pairs.
    """
    prompt = f"How many text lines are in the paragraph {format_bbox(region.bbox, bbox_format, size_wh)}?"
    completion = str(len(region.lines))
    return [(prompt, completion)]


def _region_candidates(
    region: RegionData,
    mode: str,
    bbox_format: str,
    size_wh: tuple[int, int],
    max_n: int,
    rng: random.Random,
) -> list[tuple[str, str]]:
    """Dispatch to the candidate builder for a region-based mode.

    Args:
        region (RegionData): The text region with its lines in reading order.
        mode (str): One of the region-based modes.
        bbox_format (str): "qwen" or "doclang".
        size_wh (tuple[int, int]): (width, height) of the normalization space.
        max_n (int): Maximum line offset for the line_neighbor mode.
        rng (random.Random): RNG used by the line_ordering mode.

    Returns:
        list[tuple[str, str]]: (prompt, completion) pairs.

    Raises:
        ValueError: If the mode is not a region-based mode.
    """
    if mode == MODE_LINE_NEIGHBOR:
        return build_line_neighbor_candidates(region, bbox_format, size_wh, max_n)
    if mode == MODE_REGION_TO_LINES:
        return build_region_to_lines_candidates(region, bbox_format, size_wh)
    if mode == MODE_LINES_TO_REGION:
        return build_lines_to_region_candidates(region, bbox_format, size_wh)
    if mode == MODE_REGION_TO_TRANSCRIPTION:
        return build_region_to_transcription_candidates(region, bbox_format, size_wh)
    if mode == MODE_LINE_INDEX:
        return build_line_index_candidates(region, bbox_format, size_wh)
    if mode == MODE_LINE_ORDERING:
        return build_line_ordering_candidates(region, bbox_format, size_wh, rng)
    if mode == MODE_LINE_COUNT:
        return build_line_count_candidates(region, bbox_format, size_wh)
    raise ValueError(f"Unknown region mode: {mode}")


def _build_flat_dataset(
    input_dir: Path,
    image_dir: Path,
    mode: str,
    bbox_format: str,
    max_lines: int | None = None,
    seed: int | None = None,
) -> tuple[list[dict], int]:
    """Build SFT records for a line-level mode from PAGE XML and DocLang files.

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


def _build_region_dataset(
    input_dir: Path,
    image_dir: Path,
    mode: str,
    bbox_format: str,
    max_lines: int | None = None,
    seed: int | None = None,
    max_n: int = 3,
) -> tuple[list[dict], int]:
    """Build SFT records for a region-based mode from PAGE XML files.

    Only ``.xml`` files are considered (DocLang has no line structure). All
    (prompt, completion) candidates are collected across every input file
    first; when ``max_lines`` is given and the pool is larger, a random subset
    of that size is sampled globally (``seed`` makes the selection
    reproducible).

    Args:
        input_dir (Path): Directory of PAGE XML (.xml) files.
        image_dir (Path): Directory of source document images.
        mode (str): One of the region-based modes.
        bbox_format (str): "qwen" or "doclang".
        max_lines (int | None): If set, keep at most this many examples, sampled randomly from the global pool.
        seed (int | None): Random seed for reproducible sampling and shuffling.
        max_n (int): Maximum line offset for the line_neighbor mode.

    Returns:
        tuple[list[dict], int]: The list of SFT records and the number of skipped files.
    """
    rng = random.Random(seed)
    candidates: list[tuple[str, str, Path]] = []
    skipped = 0
    for path in sorted(input_dir.glob("*.xml")):
        try:
            regions = extract_page_xml_regions(path)
        except _PARSE_ERRORS as exc:
            logger.warning("Skipping %s: could not parse (%s)", path.name, exc)
            skipped += 1
            continue
        if not regions:
            logger.warning("Skipping %s: no usable text regions found", path.name)
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
        for region in regions:
            for prompt, completion in _region_candidates(region, mode, bbox_format, size_wh, max_n, rng):
                candidates.append((prompt, completion, image_path))

    if max_lines is not None and len(candidates) > max_lines:
        candidates = rng.sample(candidates, max_lines)
        rng.shuffle(candidates)

    records = [_make_record(prompt, completion, image_path) for prompt, completion, image_path in candidates]
    return records, skipped


def build_dataset(
    input_dir: Path,
    image_dir: Path,
    mode: str,
    bbox_format: str,
    max_lines: int | None = None,
    seed: int | None = None,
    max_n: int = 3,
) -> tuple[list[dict], int]:
    """Build SFT records from a directory of PAGE XML and DocLang files.

    Line-level modes (``bbox_to_text``, ``text_to_bbox``) use both ``.xml``
    and ``.dclg`` files. Region-based modes (``line_neighbor``,
    ``region_to_lines``, ``lines_to_region``, ``region_to_transcription``,
    ``line_index``, ``line_ordering``, ``line_count``) use ``.xml`` files
    only. In all modes, when ``max_lines`` is given and the candidate pool is
    larger, a random subset of that size is sampled globally (``seed`` makes
    the selection reproducible).

    Args:
        input_dir (Path): Directory of PAGE XML (.xml) and DocLang (.dclg) files.
        image_dir (Path): Directory of source document images.
        mode (str): The task mode.
        bbox_format (str): "qwen" or "doclang".
        max_lines (int | None): If set, keep at most this many examples, sampled randomly from the global pool.
        seed (int | None): Random seed for reproducible sampling.
        max_n (int): Maximum line offset for the line_neighbor mode.

    Returns:
        tuple[list[dict], int]: The list of SFT records and the number of skipped files.
    """
    if mode in REGION_MODES:
        return _build_region_dataset(input_dir, image_dir, mode, bbox_format, max_lines, seed, max_n)
    return _build_flat_dataset(input_dir, image_dir, mode, bbox_format, max_lines, seed)


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
        choices=[MODE_BBOX_TO_TEXT, MODE_TEXT_TO_BBOX, *sorted(REGION_MODES)],
        help=(
            "bbox_to_text: predict the text inside a given box. text_to_bbox: predict the box for a given text. "
            "Region-based modes (PAGE XML only): line_neighbor, region_to_lines, lines_to_region, "
            "region_to_transcription, line_index, line_ordering, line_count."
        ),
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
    parser.add_argument(
        "--max-n",
        type=int,
        default=3,
        help="line_neighbor only: maximum line offset (1..max-n) to consider above/below the reference line.",
    )
    args = parser.parse_args()

    if args.max_lines is not None and args.max_lines < 1:
        parser.error("--max-lines must be >= 1")
    if args.max_n < 1:
        parser.error("--max-n must be >= 1")

    input_dir = Path(args.input)
    image_dir = Path(args.images)
    records, skipped = build_dataset(input_dir, image_dir, args.mode, args.bbox_format, args.max_lines, args.seed, args.max_n)

    dataset = Dataset.from_list(records)
    dataset.save_to_disk(args.output)
    logger.info("Built %d %s SFT examples (%s format) -> %s", len(dataset), args.mode, args.bbox_format, args.output)
    if args.max_lines is not None:
        logger.info("Sampled at most %d lines globally (seed=%s)", args.max_lines, args.seed)
    if skipped:
        logger.info("Skipped %d file(s)", skipped)


if __name__ == "__main__":
    main()
