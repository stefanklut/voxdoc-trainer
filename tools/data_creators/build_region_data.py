"""Build SFT training data for region detection and labeling tasks.

This tool turns PAGE XML (``.xml``) files (plus their source images) into SFT
examples: the model sees a document image and must emit the page's regions as
labeled bounding boxes.

``--input`` accepts one or more corpus roots and is searched recursively: PAGE
XML files are discovered inside directories named ``page`` at any depth, so a
corpus laid out as ``corpus/document/page/*.xml`` works without pointing at
each directory. ``--images`` is optional and also accepts multiple roots; when
it is omitted, images are found from the corpus layout itself (the PAGE XML
``imageFilename`` attribute, or the directory above the ``page`` directory).

Two task modes:

- ``detect_all``: one example per page. The prompt asks the model to detect
  and label all regions in the document and lists the label vocabulary
  present on that page. The completion is a compact JSON list of
  ``{"bbox": [x1, y1, x2, y2], "label": ...}`` objects covering every region
  on the page, in reading order.
- ``find_region``: two example styles per page. For every label present on
  the page, one example asks for all regions of that label ("Find all
  <label> regions in this document."). Additionally, for every region, one
  example asks for the region by its 1-based position within its label group
  ("Find the Nth <label> region in reading order."). Completions are compact
  JSON lists of ``{"bbox", "label"}`` objects (a single-element list for the
  ordinal style).

Regions are all PAGE XML elements whose tag ends in ``Region``
(``TextRegion``, ``ImageRegion``, ``TableRegion``, ``GraphicRegion``,
``LineRegion``, ``SeparatorRegion``, ``FormulaRegion``, ...), at any nesting
depth. Each region's polygon is fit to an axis-aligned bounding box.

Labels come from the region's ``custom`` attribute (``structure {type:X;}``)
when present, otherwise from the element tag lowercased
(``TextRegion`` -> ``textregion``).

Reading order is the ``readingOrder {index:N;}`` value from the ``custom``
attribute when present; regions without a reading order keep their document
order and sort last (the same convention as
``PageXMLEditor.get_transcription_text``).

Bboxes are serialized in one of two formats:

- ``qwen``: ``[x1, y1, x2, y2]`` integers normalized to the 0-1000 range
  (Qwen-VL grounding convention: ``round(pixel / dim * 1000)``).
- ``native``: ``[x1, y1, x2, y2]`` integers in the document's declared
  coordinate space (PAGE XML ``imageWidth``/``imageHeight``).

Prompts state the coordinate space explicitly so the model knows the
convention to use in its answer.

Normalization uses the source's declared size (PAGE XML
``imageWidth``/``imageHeight``); when no declared size is available, the
actual image size is used.

When the total number of candidates across all input files exceeds
``--max-lines``, a random subset of that size is sampled **globally**
(``--seed`` makes the selection reproducible).

Files that cannot be parsed, that contain no regions, or that have no
matching source image are skipped with a warning.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image as PILImage

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

MODE_DETECT_ALL = "detect_all"
MODE_FIND_REGION = "find_region"
FORMAT_QWEN = "qwen"
FORMAT_NATIVE = "native"

# Exception types that mark an input file as unparseable / invalid.
# ``PageXMLEditor`` raises ``FileNotFoundError`` for missing files,
# ``ET.ParseError`` for malformed XML, and ``ValueError`` for structural
# problems (missing Page/Coords); ``TypeError`` can arise from malformed
# attribute values (e.g. ``int()`` on a bad ``imageWidth``) — so it must be
# included or a single malformed file would crash the whole build.
_PARSE_ERRORS = (FileNotFoundError, ET.ParseError, ValueError, TypeError)

# Matches the ``structure {type:X;}`` entry of a region's ``custom``
# attribute. Kept local so this creator stays self-contained. Unlike
# ``PageXMLEditor.get_region_type`` (whose capture is greedy), the capture
# stops at the first ``;`` so a ``custom`` attribute holding several entries
# (e.g. ``structure {type:paragraph;} readingOrder {index:2;}``) still yields
# the bare type.
_STRUCTURE_TYPE_RE = re.compile(r".*structure {.*type:([^;]*);.*}")

# Mirrors the regex in ``PageXMLEditor._parse_reading_order``; kept local so
# this creator stays self-contained. Must stay in sync with the editor's
# method.
_READING_ORDER_RE = re.compile(r".*readingOrder {index:(\d+);.*}")


@dataclass
class RegionData:
    """A labeled region with its axis-aligned bounding box."""

    label: str
    bbox: tuple[int, int, int, int]
    reading_order: Optional[int]


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


def declared_size(path: Path) -> tuple[int, int] | None:
    """Get the (width, height) declared by a PAGE XML file.

    The ``Page`` element's ``imageWidth``/``imageHeight``.

    Args:
        path (Path): Path to a ``.xml`` file.

    Returns:
        tuple[int, int] | None: (width, height), or None when no declared size is available.
    """
    try:
        height, width = PageXMLEditor(path).get_size()
    except _PARSE_ERRORS:
        return None
    return (width, height)


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


def format_bbox(corners: tuple[int, int, int, int], bbox_format: str, size_wh: tuple[int, int]) -> list[int]:
    """Format a single bbox as a ``[x1, y1, x2, y2]`` list in the given coordinate space.

    Args:
        corners (tuple[int, int, int, int]): (x1, y1, x2, y2) in the source's native pixel space.
        bbox_format (str): "qwen" or "native".
        size_wh (tuple[int, int]): (width, height) of the normalization space.

    Returns:
        list[int]: [x1, y1, x2, y2] in the requested coordinate space.
    """
    if bbox_format == FORMAT_QWEN:
        return to_qwen_bbox(corners, size_wh)
    return list(corners)


def _coord_space_note(bbox_format: str) -> str:
    """Return a sentence stating the coordinate space for the given bbox format.

    Appended to prompts so the model knows the coordinate convention to use
    in its answer.

    Args:
        bbox_format (str): "qwen" or "native".

    Returns:
        str: A sentence (with a leading space) describing the coordinate space.
    """
    if bbox_format == FORMAT_QWEN:
        return " All bounding-box coordinates are integers normalized to the 0-1000 range."
    return " All bounding-box coordinates are pixel values relative to the top-left corner of the image."


def _ordinal(n: int) -> str:
    """Format an integer as an English ordinal (1st, 2nd, 3rd, ...)."""
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _parse_reading_order(element: ET.Element) -> Optional[int]:
    """Parse the reading order index from an element's ``custom`` attribute.

    Mirrors ``PageXMLEditor._parse_reading_order``; kept local so this
    creator stays self-contained. Must stay in sync with the editor's method.

    Args:
        element (ET.Element): The element to parse.

    Returns:
        Optional[int]: The reading order index, or None if not defined.
    """
    custom = element.attrib.get("custom")
    if custom is None:
        return None
    match = _READING_ORDER_RE.match(custom)
    if match is None:
        return None
    return int(match.group(1))


def _region_label(element: ET.Element) -> str:
    """Return the label for a region element.

    The label is the ``structure {type:X;}`` value from the element's
    ``custom`` attribute when present, otherwise the element tag lowercased
    (``TextRegion`` -> ``textregion``).

    Args:
        element (ET.Element): The region element.

    Returns:
        str: The region's label.
    """
    custom = element.attrib.get("custom")
    if custom is not None:
        match = _STRUCTURE_TYPE_RE.match(custom)
        if match is not None:
            return match.group(1)
    return element.tag.lower()


def extract_regions(page_xml_path: Path) -> list[RegionData]:
    """Extract all labeled regions from a PAGE XML file.

    Every element whose tag ends in ``Region`` is extracted, at any nesting
    depth. Regions without a ``Coords`` polygon are skipped with a warning.
    The result is sorted by reading order: regions with a ``readingOrder``
    custom value come first (by index), regions without one keep their
    document order and sort last.

    Args:
        page_xml_path (Path): Path to a PAGE XML file.

    Returns:
        list[RegionData]: Regions in reading order.

    Raises:
        FileNotFoundError: If the file does not exist.
        ET.ParseError: If the file is not well-formed XML.
        ValueError: If the file is not a valid PAGE XML document.
    """
    editor = PageXMLEditor(page_xml_path)
    indexed: list[tuple[int, RegionData]] = []
    for index, element in enumerate(editor.iter()):
        if not element.tag.endswith("Region"):
            continue
        try:
            bbox = polygon_to_bbox(editor.get_coords(element))
        except ValueError as exc:
            logger.warning("Skipping region %s in %s: %s", editor.get_id(element), page_xml_path.name, exc)
            continue
        indexed.append(
            (
                index,
                RegionData(
                    label=_region_label(element),
                    bbox=bbox,
                    reading_order=_parse_reading_order(element),
                ),
            )
        )
    indexed.sort(key=lambda item: (item[1].reading_order is None, item[1].reading_order or 0, item[0]))
    return [region for _, region in indexed]


def _format_boxes(regions: list[RegionData], bbox_format: str, size_wh: tuple[int, int]) -> str:
    """Serialize regions as a compact JSON list of ``{"bbox", "label"}`` objects.

    The shape matches the evaluation ``labeled_bbox`` contract (see
    ``docs/data_formats.md``): a JSON list of objects with exactly a
    ``bbox`` (``[x1, y1, x2, y2]``) and a ``label``.

    Args:
        regions (list[RegionData]): The regions to serialize, in the order to present them.
        bbox_format (str): "qwen" or "native".
        size_wh (tuple[int, int]): (width, height) of the normalization space.

    Returns:
        str: The compact JSON list.
    """
    boxes = [{"bbox": format_bbox(region.bbox, bbox_format, size_wh), "label": region.label} for region in regions]
    return json.dumps(boxes, separators=(",", ":"))


def build_detect_all_candidates(
    regions: list[RegionData],
    bbox_format: str,
    size_wh: tuple[int, int],
) -> list[tuple[str, str]]:
    """Build the (prompt, completion) pair for the ``detect_all`` mode.

    The prompt asks the model to detect and label all regions in the
    document and lists the label vocabulary present on the page; the
    completion is a compact JSON list of ``{"bbox", "label"}`` objects for
    every region, in reading order. Pages without regions yield no
    candidates.

    Args:
        regions (list[RegionData]): The page's regions in reading order.
        bbox_format (str): "qwen" or "native".
        size_wh (tuple[int, int]): (width, height) of the normalization space.

    Returns:
        list[tuple[str, str]]: A single (prompt, completion) pair, or an empty list.
    """
    if not regions:
        return []
    vocabulary = sorted({region.label for region in regions})
    prompt = (
        f"Detect and label all regions in this document. "
        f"Use these labels: {', '.join(vocabulary)}."
        f"{_coord_space_note(bbox_format)}"
    )
    return [(prompt, _format_boxes(regions, bbox_format, size_wh))]


def build_find_region_candidates(
    regions: list[RegionData],
    bbox_format: str,
    size_wh: tuple[int, int],
) -> list[tuple[str, str]]:
    """Build (prompt, completion) pairs for the ``find_region`` mode.

    Two styles are emitted per page. First, for every label present on the
    page, one example asks for all regions of that label; the completion
    lists every box carrying that label. Then, for every region, one example
    asks for the region by its 1-based position in its label group ("the Nth
    <label> region in reading order"); the completion is a single-element
    JSON list.

    Args:
        regions (list[RegionData]): The page's regions in reading order.
        bbox_format (str): "qwen" or "native".
        size_wh (tuple[int, int]): (width, height) of the normalization space.

    Returns:
        list[tuple[str, str]]: (prompt, completion) pairs.
    """
    if not regions:
        return []
    note = _coord_space_note(bbox_format)
    by_label: dict[str, list[RegionData]] = {}
    for region in regions:
        by_label.setdefault(region.label, []).append(region)

    candidates: list[tuple[str, str]] = []
    for label in sorted(by_label):
        group = by_label[label]
        candidates.append(
            (
                f"Find all {label} regions in this document.{note}",
                _format_boxes(group, bbox_format, size_wh),
            )
        )
    for label in sorted(by_label):
        group = by_label[label]
        for position, region in enumerate(group, start=1):
            candidates.append(
                (
                    f"Find the {_ordinal(position)} {label} region in reading order.{note}",
                    _format_boxes([region], bbox_format, size_wh),
                )
            )
    return candidates


def _make_record(
    prompt_text: str,
    completion_text: str,
    image_path: Path,
    image_mode: str = "embed",
    roots: Sequence[Path] | None = None,
) -> dict:
    """Build an SFT record with an image + text prompt and a text completion.

    Args:
        prompt_text (str): The user prompt text.
        completion_text (str): The assistant completion text.
        image_path (Path): Path to the source document image.
        image_mode (str): One of ``utils.image_loading.IMAGE_MODES``; "embed" inlines
            the image bytes, "reference" stores the image path instead.
        roots (Sequence[Path] | None): Corpus roots that stored paths are made relative to.

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
        "images": image_field(image_path, image_mode, roots),
    }


def build_dataset(
    input_dir: Path | Sequence[Path],
    image_dir: Path | Sequence[Path] | None,
    mode: str,
    bbox_format: str,
    max_lines: int | None = None,
    seed: int | None = None,
    *,
    image_mode: str = "embed",
) -> tuple[list[dict], int]:
    """Build SFT records from the PAGE XML files discovered under the input roots.

    Only PAGE XML is considered (DocLang has no region structure). All
    (prompt, completion) candidates are collected across every input file
    first; when ``max_lines`` is given and the pool is larger, a random subset
    of that size is sampled globally (``seed`` makes the selection
    reproducible).

    Args:
        input_dir (Path | Sequence[Path]): Corpus root(s); PAGE XML is discovered in
            ``page`` directories at any depth.
        image_dir (Path | Sequence[Path] | None): Explicit source image directories.
            When omitted, images are located from the corpus layout.
        mode (str): "detect_all" or "find_region".
        bbox_format (str): "qwen" or "native".
        max_lines (int | None): If set, keep at most this many examples, sampled randomly from the global pool.
        seed (int | None): Random seed for reproducible sampling.
        image_mode (str): One of ``utils.image_loading.IMAGE_MODES``. "embed" (the
            default) stores image bytes in the dataset; "reference" stores paths.

    Returns:
        tuple[list[dict], int]: The list of SFT records and the number of skipped files.

    Raises:
        ValueError: If the mode is not a region mode.
        DiscoveryError: If no PAGE XML file is found under the input roots.
    """
    if mode not in (MODE_DETECT_ALL, MODE_FIND_REGION):
        raise ValueError(f"Unknown mode: {mode}")
    rng = random.Random(seed)
    candidates: list[tuple[str, str, Path]] = []
    skipped = 0
    roots = image_roots(input_dir, image_dir)
    for path in discover_page_xml(input_dir):
        try:
            regions = extract_regions(path)
        except _PARSE_ERRORS as exc:
            logger.warning("Skipping %s: could not parse (%s)", path.name, exc)
            skipped += 1
            continue
        if not regions:
            logger.warning("Skipping %s: no regions found", path.name)
            skipped += 1
            continue
        image_path = resolve_image(path, image_dir)
        if image_path is None:
            logger.warning("Skipping %s: no source image found", path)
            skipped += 1
            continue
        size_wh = declared_size(path)
        if size_wh is None:
            with PILImage.open(image_path) as image:
                size_wh = (image.width, image.height)
        if mode == MODE_DETECT_ALL:
            pairs = build_detect_all_candidates(regions, bbox_format, size_wh)
        else:
            pairs = build_find_region_candidates(regions, bbox_format, size_wh)
        for prompt, completion in pairs:
            candidates.append((prompt, completion, image_path))

    if max_lines is not None and len(candidates) > max_lines:
        candidates = rng.sample(candidates, max_lines)
        rng.shuffle(candidates)

    records = [
        _make_record(prompt, completion, image_path, image_mode, roots) for prompt, completion, image_path in candidates
    ]
    return records, skipped


def main() -> None:
    """Build an SFT dataset of region detection and labeling examples."""
    setup_logger()
    parser = argparse.ArgumentParser(
        description="Build SFT region detection and labeling training data from PAGE XML files."
    )
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
    parser.add_argument(
        "--mode",
        required=True,
        choices=[MODE_DETECT_ALL, MODE_FIND_REGION],
        help=(
            "detect_all: one example per page — detect and label every region. "
            "find_region: per-label and per-region examples — find all regions of a label, "
            "or the Nth region of a label in reading order."
        ),
    )
    parser.add_argument(
        "--bbox-format",
        required=True,
        choices=[FORMAT_QWEN, FORMAT_NATIVE],
        help="qwen: [x1, y1, x2, y2] normalized to 0-1000. native: [x1, y1, x2, y2] in the declared pixel space.",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=None,
        help="If set, randomly sample at most this many examples from the global pool of all input files.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible --max-lines sampling.")
    add_image_mode_argument(parser)
    args = parser.parse_args()

    if args.max_lines is not None and args.max_lines < 1:
        parser.error("--max-lines must be >= 1")

    try:
        records, skipped = build_dataset(
            args.input, args.images, args.mode, args.bbox_format, args.max_lines, args.seed, image_mode=args.image_mode
        )
    except (DiscoveryError, NotADirectoryError) as exc:
        parser.error(str(exc))

    dataset = save_image_dataset(records, args.output, args.image_mode, image_roots(args.input, args.images))
    logger.info(
        "Built %d %s SFT examples (%s format, %s images) -> %s",
        len(dataset),
        args.mode,
        args.bbox_format,
        args.image_mode,
        args.output,
    )
    if args.max_lines is not None:
        logger.info("Sampled at most %d examples globally (seed=%s)", args.max_lines, args.seed)
    if skipped:
        logger.info("Skipped %d file(s)", skipped)


if __name__ == "__main__":
    main()
