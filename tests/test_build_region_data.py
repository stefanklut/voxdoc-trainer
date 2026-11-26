"""Unit tests for the region detection and labeling data builder."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tools.data_creators.build_region_data import (
    MODE_DETECT_ALL,
    MODE_FIND_REGION,
    RegionData,
    _coord_space_note,
    _make_record,
    _ordinal,
    _parse_reading_order,
    _region_label,
    build_dataset,
    build_detect_all_candidates,
    build_find_region_candidates,
    declared_size,
    extract_regions,
    find_image,
    format_bbox,
    polygon_to_bbox,
    to_qwen_bbox,
)
from utils.discovery import DiscoveryError

# The coordinate-space sentences appended to prompts (see _coord_space_note).
QWEN_NOTE = " All bounding-box coordinates are integers normalized to the 0-1000 range."
NATIVE_NOTE = " All bounding-box coordinates are pixel values relative to the top-left corner of the image."


def _make_image(path: Path) -> None:
    """Create a tiny test image at the given path."""
    Image.new("RGB", (8, 8), color="white").save(path)


def _make_page_xml_regions(
    path: Path,
    regions: list[tuple[str, str, str | None, list[tuple[str, str, str | None]]]],
    width: int = 100,
    height: int = 100,
) -> None:
    """Write a minimal PAGE XML file with the given regions.

    Args:
        path (Path): Where to write the XML file.
        regions (list[tuple[str, str, str | None, list[tuple[str, str, str | None]]]]):
            (tag, coords_points, custom, children) tuples in document order, where
            children are (tag, coords_points, custom) tuples nested inside the region.
        width (int): The Page imageWidth.
        height (int): The Page imageHeight.
    """
    region_blocks = []
    for tag, points, custom, children in regions:
        custom_attr = f' custom="{custom}"' if custom is not None else ""
        child_blocks = []
        for child_tag, child_points, child_custom in children:
            child_custom_attr = f' custom="{child_custom}"' if child_custom is not None else ""
            child_blocks.append(
                f'        <{child_tag} id="{child_tag.lower()}_{child_points.replace(" ", "_").replace(",", "")}"{child_custom_attr}>\n'
                f'          <Coords points="{child_points}"/>\n'
                f"        </{child_tag}>"
            )
        children_block = ("\n".join(child_blocks) + "\n") if child_blocks else ""
        region_blocks.append(
            f'    <{tag} id="{tag.lower()}_{points.replace(" ", "_").replace(",", "")}"{custom_attr}>\n'
            f'      <Coords points="{points}"/>\n'
            f"{children_block}"
            f"    </{tag}>"
        )
    regions_block = "\n".join(region_blocks)
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">\n'
        "  <Metadata>\n"
        "    <Creator>test</Creator>\n"
        "    <Created>2024-01-01T00:00:00</Created>\n"
        "    <LastChange>2024-01-01T00:00:00</LastChange>\n"
        "  </Metadata>\n"
        f'  <Page imageFilename="doc.png" imageWidth="{width}" imageHeight="{height}">\n'
        f"{regions_block}\n"
        "  </Page>\n"
        "</PcGts>\n"
    )
    path.write_text(xml, encoding="utf-8")


def _make_mixed_page(path: Path) -> None:
    """Write a PAGE XML page with a mix of region types, labels, and reading orders.

    The page (100x100) contains, in document order:
    - TextRegion, custom structure type "paragraph" + reading order 2, (10, 10, 50, 20)
    - TextRegion, no structure type, reading order 1, (10, 30, 50, 40)
    - ImageRegion, no custom, no reading order, (60, 10, 90, 40)
    - TableRegion, no custom, no reading order, (10, 50, 90, 90), with a nested
      TextRegion (custom structure type "cell", no reading order, (20, 60, 40, 70))
    - LineRegion, no custom, no reading order, 2-point coords (10, 95, 90, 95)
    """
    _make_page_xml_regions(
        path,
        [
            ("TextRegion", "10,10 50,10 50,20 10,20", "structure {type:paragraph;} readingOrder {index:2;}", []),
            ("TextRegion", "10,30 50,30 50,40 10,40", "readingOrder {index:1;}", []),
            ("ImageRegion", "60,10 90,10 90,40 60,40", None, []),
            (
                "TableRegion",
                "10,50 90,50 90,90 10,90",
                None,
                [("TextRegion", "20,60 40,60 40,70 20,70", "structure {type:cell;}")],
            ),
            ("LineRegion", "10,95 90,95", None, []),
        ],
    )


def _completion_text(record: dict) -> str:
    return record["completion"][0]["content"][0]["text"]


def _prompt_text(record: dict) -> str:
    return record["prompt"][0]["content"][1]["text"]


def _page_dir(tmp_path: Path) -> Path:
    """Return (creating if needed) the ``page`` directory of a corpus root.

    Args:
        tmp_path (Path): Corpus root.

    Returns:
        Path: The ``page`` directory.
    """
    page_dir = tmp_path / "page"
    page_dir.mkdir(exist_ok=True)
    return page_dir


def _make_pool(tmp_path: Path, n_files: int) -> Path:
    """Create ``n_files`` mixed PAGE XML pages plus matching images."""
    images = tmp_path / "images"
    images.mkdir()
    page_dir = _page_dir(tmp_path)
    for i in range(n_files):
        stem = f"doc{i}"
        _make_mixed_page(page_dir / f"{stem}.xml")
        _make_image(images / f"{stem}.png")
    return images


# --------------------------------------------------------------------------- #
# find_image
# --------------------------------------------------------------------------- #


def test_find_image_matches_non_png_format(tmp_path: Path) -> None:
    """find_image should locate a source image in any PIL-supported format."""
    _make_image(tmp_path / "doc.jpg")
    assert find_image(tmp_path, "doc") == tmp_path / "doc.jpg"


def test_find_image_returns_none_when_missing(tmp_path: Path) -> None:
    """find_image should return None when no supported image exists."""
    assert find_image(tmp_path, "missing") is None


# --------------------------------------------------------------------------- #
# polygon_to_bbox
# --------------------------------------------------------------------------- #


def test_polygon_to_bbox_axis_aligned() -> None:
    """An axis-aligned rectangle maps to its own corners."""
    points = np.array([[10, 20], [50, 20], [50, 80], [10, 80]], dtype=np.int32)
    assert polygon_to_bbox(points) == (10, 20, 50, 80)


def test_polygon_to_bbox_non_rectangular() -> None:
    """A rotated quadrilateral maps to its axis-aligned extent."""
    points = np.array([[0, 0], [10, 5], [5, 10], [-5, 5]], dtype=np.int32)
    assert polygon_to_bbox(points) == (-5, 0, 10, 10)


def test_polygon_to_bbox_two_points() -> None:
    """A 2-point line maps to a zero-height (or zero-width) box."""
    points = np.array([[10, 95], [90, 95]], dtype=np.int32)
    assert polygon_to_bbox(points) == (10, 95, 90, 95)


# --------------------------------------------------------------------------- #
# to_qwen_bbox / format_bbox
# --------------------------------------------------------------------------- #


def test_to_qwen_bbox_normalization() -> None:
    """Native corners are scaled to the 0-1000 range per axis."""
    assert to_qwen_bbox((10, 20, 60, 100), (100, 200)) == [100, 100, 600, 500]


def test_to_qwen_bbox_full_image() -> None:
    """A box covering the whole image maps to the full 0-1000 range."""
    assert to_qwen_bbox((0, 0, 100, 100), (100, 100)) == [0, 0, 1000, 1000]


def test_to_qwen_bbox_clamps_to_1000() -> None:
    """Values beyond the declared size clamp to 1000."""
    assert to_qwen_bbox((0, 0, 150, 250), (100, 200)) == [0, 0, 1000, 1000]


def test_format_bbox_qwen() -> None:
    """qwen format returns normalized coordinates."""
    assert format_bbox((10, 20, 60, 100), "qwen", (100, 200)) == [100, 100, 600, 500]


def test_format_bbox_native() -> None:
    """native format returns the native corners as a list."""
    assert format_bbox((10, 20, 60, 100), "native", (100, 200)) == [10, 20, 60, 100]


def test_coord_space_note() -> None:
    """Each format has its own coordinate-space sentence."""
    assert _coord_space_note("qwen") == QWEN_NOTE
    assert _coord_space_note("native") == NATIVE_NOTE


# --------------------------------------------------------------------------- #
# _ordinal
# --------------------------------------------------------------------------- #


def test_ordinal() -> None:
    """Ordinals follow English conventions, including the 11-13 'th' rule."""
    assert _ordinal(1) == "1st"
    assert _ordinal(2) == "2nd"
    assert _ordinal(3) == "3rd"
    assert _ordinal(4) == "4th"
    assert _ordinal(11) == "11th"
    assert _ordinal(12) == "12th"
    assert _ordinal(13) == "13th"
    assert _ordinal(21) == "21st"
    assert _ordinal(22) == "22nd"


# --------------------------------------------------------------------------- #
# _region_label / _parse_reading_order
# --------------------------------------------------------------------------- #


def _element(tag: str, custom: str | None = None):
    element = ET.Element(tag)
    if custom is not None:
        element.set("custom", custom)
    return element


def test_region_label_prefers_custom_structure_type() -> None:
    """A structure {type:X;} custom value wins over the tag."""
    assert _region_label(_element("TextRegion", "structure {type:paragraph;}")) == "paragraph"


def test_region_label_falls_back_to_lowercased_tag() -> None:
    """Without a custom structure type, the tag lowercased is used."""
    assert _region_label(_element("TextRegion")) == "textregion"
    assert _region_label(_element("ImageRegion")) == "imageregion"
    assert _region_label(_element("TableRegion")) == "tableregion"


def test_region_label_ignores_unrelated_custom() -> None:
    """A custom attribute without a structure type falls back to the tag."""
    assert _region_label(_element("TextRegion", "readingOrder {index:1;}")) == "textregion"


def test_region_label_multi_entry_custom() -> None:
    """A custom attribute with several entries yields the bare structure type."""
    assert _region_label(_element("TextRegion", "structure {type:paragraph;} readingOrder {index:2;}")) == "paragraph"


def test_parse_reading_order_present() -> None:
    """A readingOrder {index:N;} custom value is parsed to N."""
    assert _parse_reading_order(_element("TextRegion", "readingOrder {index:7;}")) == 7


def test_parse_reading_order_absent() -> None:
    """No custom attribute yields None."""
    assert _parse_reading_order(_element("TextRegion")) is None


def test_parse_reading_order_malformed() -> None:
    """A custom attribute without a reading order yields None."""
    assert _parse_reading_order(_element("TextRegion", "structure {type:paragraph;}")) is None


# --------------------------------------------------------------------------- #
# extract_regions
# --------------------------------------------------------------------------- #


def test_extract_regions_all_types_any_depth(tmp_path: Path) -> None:
    """Every *Region element is extracted, at any nesting depth, in reading order."""
    path = tmp_path / "doc.xml"
    _make_mixed_page(path)
    regions = extract_regions(path)
    assert [(region.label, region.bbox, region.reading_order) for region in regions] == [
        ("textregion", (10, 30, 50, 40), 1),
        ("paragraph", (10, 10, 50, 20), 2),
        ("imageregion", (60, 10, 90, 40), None),
        ("tableregion", (10, 50, 90, 90), None),
        ("cell", (20, 60, 40, 70), None),
        ("lineregion", (10, 95, 90, 95), None),
    ]


def test_extract_regions_sorts_reading_ordered_first(tmp_path: Path) -> None:
    """Regions with a reading order sort first (by index); the rest keep document order."""
    path = tmp_path / "doc.xml"
    _make_page_xml_regions(
        path,
        [
            ("TextRegion", "0,0 10,0 10,5 0,5", "readingOrder {index:5;}", []),
            ("ImageRegion", "20,0 30,0 30,5 20,5", None, []),
            ("TextRegion", "40,0 50,0 50,5 40,5", "readingOrder {index:2;}", []),
            ("GraphicRegion", "60,0 70,0 70,5 60,5", None, []),
        ],
    )
    regions = extract_regions(path)
    assert [region.reading_order for region in regions] == [2, 5, None, None]
    assert [region.label for region in regions] == ["textregion", "textregion", "imageregion", "graphicregion"]


def test_extract_regions_skips_region_without_coords(tmp_path: Path) -> None:
    """A region without a Coords element is skipped (the rest are kept)."""
    path = tmp_path / "doc.xml"
    _make_page_xml_regions(
        path,
        [
            ("TextRegion", "0,0 10,0 10,5 0,5", None, []),
            ("ImageRegion", "20,0 30,0 30,5 20,5", None, []),
        ],
    )
    # Remove the Coords element from the ImageRegion.
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        '<ImageRegion id="imageregion_200_300_305_205">\n      <Coords points="20,0 30,0 30,5 20,5"/>\n    </ImageRegion>',
        '<ImageRegion id="imageregion_200_300_305_205">\n    </ImageRegion>',
    )
    path.write_text(text, encoding="utf-8")
    regions = extract_regions(path)
    assert [region.label for region in regions] == ["textregion"]


def test_extract_regions_empty_page(tmp_path: Path) -> None:
    """A page with no regions yields an empty list."""
    path = tmp_path / "doc.xml"
    _make_page_xml_regions(path, [])
    assert extract_regions(path) == []


def test_extract_regions_missing_file(tmp_path: Path) -> None:
    """A missing file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        extract_regions(tmp_path / "missing.xml")


def test_extract_regions_malformed_xml(tmp_path: Path) -> None:
    """Malformed XML raises ET.ParseError."""
    path = tmp_path / "bad.xml"
    path.write_text("<PcGts><Page></PcGts>", encoding="utf-8")
    with pytest.raises(ET.ParseError):
        extract_regions(path)


def test_extract_regions_missing_page(tmp_path: Path) -> None:
    """A file without a Page element has no regions (consistent with extract_page_xml_regions)."""
    path = tmp_path / "bad.xml"
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">\n'
        "  <Metadata><Creator>test</Creator></Metadata>\n"
        "</PcGts>\n",
        encoding="utf-8",
    )
    assert extract_regions(path) == []


# --------------------------------------------------------------------------- #
# declared_size
# --------------------------------------------------------------------------- #


def test_declared_size_page_xml(tmp_path: Path) -> None:
    """declared_size returns the Page imageWidth/imageHeight as (width, height)."""
    path = tmp_path / "doc.xml"
    _make_mixed_page(path)
    assert declared_size(path) == (100, 100)


def test_declared_size_malformed_returns_none(tmp_path: Path) -> None:
    """A file without a Page element yields None."""
    path = tmp_path / "bad.xml"
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">\n'
        "  <Metadata><Creator>test</Creator></Metadata>\n"
        "</PcGts>\n",
        encoding="utf-8",
    )
    assert declared_size(path) is None


# --------------------------------------------------------------------------- #
# build_detect_all_candidates
# --------------------------------------------------------------------------- #


def test_build_detect_all_candidates_qwen(tmp_path: Path) -> None:
    """detect_all: one candidate; prompt lists the vocabulary; completion is the full JSON list."""
    path = tmp_path / "doc.xml"
    _make_mixed_page(path)
    regions = extract_regions(path)
    candidates = build_detect_all_candidates(regions, "qwen", (100, 100))
    assert len(candidates) == 1
    prompt, completion = candidates[0]
    assert prompt == (
        "Detect and label all regions in this document. "
        "Use these labels: cell, imageregion, lineregion, paragraph, tableregion, textregion."
        + QWEN_NOTE
    )
    assert json.loads(completion) == [
        {"bbox": [100, 300, 500, 400], "label": "textregion"},
        {"bbox": [100, 100, 500, 200], "label": "paragraph"},
        {"bbox": [600, 100, 900, 400], "label": "imageregion"},
        {"bbox": [100, 500, 900, 900], "label": "tableregion"},
        {"bbox": [200, 600, 400, 700], "label": "cell"},
        {"bbox": [100, 950, 900, 950], "label": "lineregion"},
    ]


def test_build_detect_all_candidates_native(tmp_path: Path) -> None:
    """detect_all + native: completion uses native coordinates and the native note."""
    path = tmp_path / "doc.xml"
    _make_mixed_page(path)
    regions = extract_regions(path)
    candidates = build_detect_all_candidates(regions, "native", (100, 100))
    prompt, completion = candidates[0]
    assert prompt.endswith(NATIVE_NOTE)
    assert json.loads(completion) == [
        {"bbox": [10, 30, 50, 40], "label": "textregion"},
        {"bbox": [10, 10, 50, 20], "label": "paragraph"},
        {"bbox": [60, 10, 90, 40], "label": "imageregion"},
        {"bbox": [10, 50, 90, 90], "label": "tableregion"},
        {"bbox": [20, 60, 40, 70], "label": "cell"},
        {"bbox": [10, 95, 90, 95], "label": "lineregion"},
    ]


def test_build_detect_all_candidates_empty(tmp_path: Path) -> None:
    """A page without regions yields no candidates."""
    assert build_detect_all_candidates([], "qwen", (100, 100)) == []


# --------------------------------------------------------------------------- #
# build_find_region_candidates
# --------------------------------------------------------------------------- #


def test_build_find_region_candidates(tmp_path: Path) -> None:
    """find_region: one per-label candidate plus one per-region ordinal candidate."""
    path = tmp_path / "doc.xml"
    _make_mixed_page(path)
    regions = extract_regions(path)
    candidates = build_find_region_candidates(regions, "native", (100, 100))
    # 6 labels + 6 regions = 12 candidates.
    assert len(candidates) == 12
    # Style A: per-label, sorted by label.
    assert candidates[0] == (
        "Find all cell regions in this document." + NATIVE_NOTE,
        '[{"bbox":[20,60,40,70],"label":"cell"}]',
    )
    assert candidates[1] == (
        "Find all imageregion regions in this document." + NATIVE_NOTE,
        '[{"bbox":[60,10,90,40],"label":"imageregion"}]',
    )
    assert candidates[2] == (
        "Find all lineregion regions in this document." + NATIVE_NOTE,
        '[{"bbox":[10,95,90,95],"label":"lineregion"}]',
    )
    assert candidates[3] == (
        "Find all paragraph regions in this document." + NATIVE_NOTE,
        '[{"bbox":[10,10,50,20],"label":"paragraph"}]',
    )
    assert candidates[4] == (
        "Find all tableregion regions in this document." + NATIVE_NOTE,
        '[{"bbox":[10,50,90,90],"label":"tableregion"}]',
    )
    # The textregion group has a single region (the paragraph region is labeled "paragraph").
    assert candidates[5] == (
        "Find all textregion regions in this document." + NATIVE_NOTE,
        '[{"bbox":[10,30,50,40],"label":"textregion"}]',
    )
    # Style B: per-region ordinal, sorted by label then position.
    assert candidates[6] == (
        "Find the 1st cell region in reading order." + NATIVE_NOTE,
        '[{"bbox":[20,60,40,70],"label":"cell"}]',
    )
    assert candidates[7] == (
        "Find the 1st imageregion region in reading order." + NATIVE_NOTE,
        '[{"bbox":[60,10,90,40],"label":"imageregion"}]',
    )
    assert candidates[8] == (
        "Find the 1st lineregion region in reading order." + NATIVE_NOTE,
        '[{"bbox":[10,95,90,95],"label":"lineregion"}]',
    )
    assert candidates[9] == (
        "Find the 1st paragraph region in reading order." + NATIVE_NOTE,
        '[{"bbox":[10,10,50,20],"label":"paragraph"}]',
    )
    assert candidates[10] == (
        "Find the 1st tableregion region in reading order." + NATIVE_NOTE,
        '[{"bbox":[10,50,90,90],"label":"tableregion"}]',
    )
    assert candidates[11] == (
        "Find the 1st textregion region in reading order." + NATIVE_NOTE,
        '[{"bbox":[10,30,50,40],"label":"textregion"}]',
    )


def test_build_find_region_candidates_ordinal_positions(tmp_path: Path) -> None:
    """Ordinal positions are 1-based within each label group, in reading order."""
    path = tmp_path / "doc.xml"
    _make_page_xml_regions(
        path,
        [
            ("TextRegion", "0,0 10,0 10,5 0,5", "readingOrder {index:2;}", []),
            ("TextRegion", "20,0 30,0 30,5 20,5", "readingOrder {index:1;}", []),
            ("TextRegion", "40,0 50,0 50,5 40,5", None, []),
        ],
    )
    regions = extract_regions(path)
    candidates = build_find_region_candidates(regions, "native", (100, 100))
    # 1 label + 3 regions = 4 candidates.
    assert len(candidates) == 4
    # Style A: all three textregion boxes, reading order (index 1, index 2, unordered).
    assert json.loads(candidates[0][1]) == [
        {"bbox": [20, 0, 30, 5], "label": "textregion"},
        {"bbox": [0, 0, 10, 5], "label": "textregion"},
        {"bbox": [40, 0, 50, 5], "label": "textregion"},
    ]
    # Style B: ordinal positions follow the reading order.
    assert candidates[1][0] == "Find the 1st textregion region in reading order." + NATIVE_NOTE
    assert json.loads(candidates[1][1]) == [{"bbox": [20, 0, 30, 5], "label": "textregion"}]
    assert candidates[2][0] == "Find the 2nd textregion region in reading order." + NATIVE_NOTE
    assert json.loads(candidates[2][1]) == [{"bbox": [0, 0, 10, 5], "label": "textregion"}]
    assert candidates[3][0] == "Find the 3rd textregion region in reading order." + NATIVE_NOTE
    assert json.loads(candidates[3][1]) == [{"bbox": [40, 0, 50, 5], "label": "textregion"}]


def test_build_find_region_candidates_empty(tmp_path: Path) -> None:
    """A page without regions yields no candidates."""
    assert build_find_region_candidates([], "qwen", (100, 100)) == []


# --------------------------------------------------------------------------- #
# _make_record
# --------------------------------------------------------------------------- #


def test_make_record_shape(tmp_path: Path) -> None:
    """Records follow the SFT shape: image + text prompt, text completion, images."""
    image_path = tmp_path / "doc.png"
    _make_image(image_path)
    record = _make_record("prompt text", '{"bbox":[0,0,1,1],"label":"text"}', image_path)
    content = record["prompt"][0]["content"]
    assert content[0] == {"type": "image"}
    assert content[1] == {"type": "text", "text": "prompt text"}
    assert record["completion"] == [{"role": "assistant", "content": [{"type": "text", "text": '{"bbox":[0,0,1,1],"label":"text"}'}]}]
    assert len(record["images"]) == 1


# --------------------------------------------------------------------------- #
# build_dataset
# --------------------------------------------------------------------------- #


def test_build_dataset_detect_all(tmp_path: Path) -> None:
    """detect_all: one record per page."""
    images = _make_pool(tmp_path, 2)
    records, skipped = build_dataset(tmp_path, images, MODE_DETECT_ALL, "qwen")
    assert skipped == 0
    assert len(records) == 2
    for record in records:
        assert _prompt_text(record).startswith("Detect and label all regions in this document.")
        assert _prompt_text(record).endswith(QWEN_NOTE)
        boxes = json.loads(_completion_text(record))
        assert len(boxes) == 6
        assert all(set(box) == {"bbox", "label"} for box in boxes)


def test_build_dataset_find_region(tmp_path: Path) -> None:
    """find_region: 12 records per page (6 labels + 6 regions)."""
    images = _make_pool(tmp_path, 1)
    records, skipped = build_dataset(tmp_path, images, MODE_FIND_REGION, "native")
    assert skipped == 0
    assert len(records) == 12
    prompts = [_prompt_text(record) for record in records]
    assert prompts[0] == "Find all cell regions in this document." + NATIVE_NOTE
    assert prompts[5] == "Find all textregion regions in this document." + NATIVE_NOTE
    assert prompts[6] == "Find the 1st cell region in reading order." + NATIVE_NOTE


def test_build_dataset_skips_file_without_image(tmp_path: Path) -> None:
    """A file with regions but no matching image is skipped."""
    page_dir = _page_dir(tmp_path)
    _make_mixed_page(page_dir / "a.xml")
    _make_mixed_page(page_dir / "b.xml")
    images = tmp_path / "images"
    images.mkdir()
    _make_image(images / "b.png")

    records, skipped = build_dataset(tmp_path, images, MODE_DETECT_ALL, "qwen")
    assert skipped == 1
    assert len(records) == 1


def test_build_dataset_skips_file_without_regions(tmp_path: Path) -> None:
    """A file with no regions is skipped."""
    page_dir = _page_dir(tmp_path)
    _make_page_xml_regions(page_dir / "a.xml", [])
    _make_mixed_page(page_dir / "b.xml")
    images = tmp_path / "images"
    images.mkdir()
    _make_image(images / "a.png")
    _make_image(images / "b.png")

    records, skipped = build_dataset(tmp_path, images, MODE_DETECT_ALL, "qwen")
    assert skipped == 1
    assert len(records) == 1


def test_build_dataset_skips_malformed_xml(tmp_path: Path) -> None:
    """A malformed .xml file is skipped, not fatal."""
    page_dir = _page_dir(tmp_path)
    (page_dir / "bad.xml").write_text("<PcGts><Page></PcGts>", encoding="utf-8")
    _make_mixed_page(page_dir / "good.xml")
    images = tmp_path / "images"
    images.mkdir()
    _make_image(images / "bad.png")
    _make_image(images / "good.png")

    records, skipped = build_dataset(tmp_path, images, MODE_DETECT_ALL, "qwen")
    assert skipped == 1
    assert len(records) == 1


def test_build_dataset_ignores_other_extensions(tmp_path: Path) -> None:
    """Files that are not .xml are ignored (not counted as skipped)."""
    page_dir = _page_dir(tmp_path)
    _make_mixed_page(page_dir / "doc.xml")
    (page_dir / "notes.txt").write_text("not a doc", encoding="utf-8")
    images = tmp_path / "images"
    images.mkdir()
    _make_image(images / "doc.png")

    records, skipped = build_dataset(tmp_path, images, MODE_DETECT_ALL, "qwen")
    assert len(records) == 1
    assert skipped == 0


def test_build_dataset_ignores_xml_outside_page_dirs(tmp_path: Path) -> None:
    """.xml files outside a 'page' directory are not discovered."""
    page_dir = _page_dir(tmp_path)
    _make_mixed_page(page_dir / "doc.xml")
    _make_mixed_page(tmp_path / "stray.xml")
    images = tmp_path / "images"
    images.mkdir()
    _make_image(images / "doc.png")
    _make_image(images / "stray.png")

    records, skipped = build_dataset(tmp_path, images, MODE_DETECT_ALL, "qwen")
    assert len(records) == 1
    assert skipped == 0


def test_build_dataset_discovers_nested_page_dirs(tmp_path: Path) -> None:
    """PAGE XML nested below the corpus root is discovered."""
    page_dir = tmp_path / "corpus" / "document_001" / "page"
    page_dir.mkdir(parents=True)
    _make_mixed_page(page_dir / "doc.xml")
    _make_image(page_dir.parent / "doc.png")

    records, skipped = build_dataset(tmp_path / "corpus", None, MODE_DETECT_ALL, "qwen")
    assert skipped == 0
    assert len(records) == 1


def test_build_dataset_raises_when_no_page_dir_exists(tmp_path: Path) -> None:
    """A corpus without any page directory is reported rather than silently empty."""
    (tmp_path / "flat.xml").write_text("<PcGts><Page/></PcGts>", encoding="utf-8")
    with pytest.raises(DiscoveryError):
        build_dataset(tmp_path, None, MODE_DETECT_ALL, "qwen")


def test_build_dataset_falls_back_to_image_size(tmp_path: Path) -> None:
    """A page without imageWidth/imageHeight normalizes by the actual image size."""
    path = _page_dir(tmp_path) / "doc.xml"
    _make_page_xml_regions(
        path,
        [("TextRegion", "0,0 4,0 4,4 0,4", None, [])],
        width=0,
        height=0,
    )
    # Remove the (zero) imageWidth/imageHeight attributes so declared_size returns None.
    text = path.read_text(encoding="utf-8")
    text = text.replace(' imageWidth="0" imageHeight="0"', "")
    path.write_text(text, encoding="utf-8")
    images = tmp_path / "images"
    images.mkdir()
    _make_image(images / "doc.png")  # 8x8

    records, skipped = build_dataset(tmp_path, images, MODE_DETECT_ALL, "qwen")
    assert skipped == 0
    # (0, 0, 4, 4) in an 8x8 image normalizes to (0, 0, 500, 500).
    assert json.loads(_completion_text(records[0])) == [{"bbox": [0, 0, 500, 500], "label": "textregion"}]


def test_build_dataset_no_sampling_preserves_order(tmp_path: Path) -> None:
    """Without max_lines, records follow deterministic file/candidate order."""
    images = _make_pool(tmp_path, 2)
    records, _ = build_dataset(tmp_path, images, MODE_DETECT_ALL, "qwen")
    assert len(records) == 2
    # Both pages are identical, so both completions are identical.
    assert _completion_text(records[0]) == _completion_text(records[1])


def test_build_dataset_sampling_caps_pool(tmp_path: Path) -> None:
    """When the pool exceeds max_lines, exactly max_lines records are kept."""
    images = _make_pool(tmp_path, 3)  # 3 pages * 12 find_region candidates = 36
    records, skipped = build_dataset(tmp_path, images, MODE_FIND_REGION, "qwen", max_lines=5)
    assert skipped == 0
    assert len(records) == 5


def test_build_dataset_sampling_keeps_all_when_pool_smaller(tmp_path: Path) -> None:
    """When the pool is at most max_lines, all candidates are kept."""
    images = _make_pool(tmp_path, 1)  # 12 find_region candidates
    records, skipped = build_dataset(tmp_path, images, MODE_FIND_REGION, "qwen", max_lines=50)
    assert skipped == 0
    assert len(records) == 12


def test_build_dataset_sampling_subset_of_global_pool(tmp_path: Path) -> None:
    """Sampled records are a subset of the global pool (across files)."""
    images = _make_pool(tmp_path, 3)  # 36 candidates across 3 files
    records, _ = build_dataset(tmp_path, images, MODE_FIND_REGION, "qwen", max_lines=5, seed=7)
    completions = [_completion_text(record) for record in records]
    assert len(completions) == 5
    # The pool is the union of every page's candidates (pages are identical here).
    regions = extract_regions(_page_dir(tmp_path) / "doc0.xml")
    pool = {completion for _, completion in build_find_region_candidates(regions, "qwen", (100, 100))}
    assert set(completions) <= pool


def test_build_dataset_sampling_is_reproducible_with_seed(tmp_path: Path) -> None:
    """The same seed yields the identical sampled selection and order."""
    images = _make_pool(tmp_path, 3)
    records_a, _ = build_dataset(tmp_path, images, MODE_FIND_REGION, "qwen", max_lines=5, seed=42)
    records_b, _ = build_dataset(tmp_path, images, MODE_FIND_REGION, "qwen", max_lines=5, seed=42)
    assert [_prompt_text(r) for r in records_a] == [_prompt_text(r) for r in records_b]


def test_build_dataset_unknown_mode_raises(tmp_path: Path) -> None:
    """An unknown mode raises ValueError."""
    images = tmp_path / "images"
    images.mkdir()
    with pytest.raises(ValueError, match="Unknown mode"):
        build_dataset(tmp_path, images, "unknown_mode", "qwen")


def test_make_record_reference_mode_stores_path(tmp_path: Path) -> None:
    """Reference mode stores the image path relative to the corpus root."""
    image = tmp_path / "images" / "doc.png"
    image.parent.mkdir()
    _make_image(image)
    record = _make_record("Find the regions.", "[[0,0,100,100]]", image, "reference", [tmp_path / "images"])
    assert record["images"] == ["doc.png"]


def test_make_record_embed_mode_unchanged(tmp_path: Path) -> None:
    """The default mode still embeds a decoded image."""
    image = tmp_path / "images" / "doc.png"
    image.parent.mkdir()
    _make_image(image)
    record = _make_record("Find the regions.", "[[0,0,100,100]]", image)
    assert isinstance(record["images"][0], Image.Image)


def test_build_dataset_reference_mode_stores_relative_paths(tmp_path: Path) -> None:
    """Reference mode yields string image entries for every record."""
    images = _make_pool(tmp_path, 2)
    records, skipped = build_dataset(tmp_path, images, MODE_DETECT_ALL, "qwen", image_mode="reference")
    assert skipped == 0
    assert records
    assert [record["images"] for record in records] == [["doc0.png"], ["doc1.png"]]


def test_build_dataset_embed_mode_unchanged(tmp_path: Path) -> None:
    """The default mode still embeds decoded images, so existing datasets are unaffected."""
    images = _make_pool(tmp_path, 1)
    records, skipped = build_dataset(tmp_path, images, MODE_DETECT_ALL, "qwen")
    assert skipped == 0
    assert all(isinstance(record["images"][0], Image.Image) for record in records)


def test_build_dataset_reference_mode_sampling_unchanged(tmp_path: Path) -> None:
    """Sampling is independent of the image mode, so both modes select the same rows."""
    images = _make_pool(tmp_path, 3)
    embedded, _ = build_dataset(tmp_path, images, MODE_FIND_REGION, "qwen", max_lines=5, seed=42)
    referenced, _ = build_dataset(
        tmp_path, images, MODE_FIND_REGION, "qwen", max_lines=5, seed=42, image_mode="reference"
    )
    assert [_prompt_text(r) for r in embedded] == [_prompt_text(r) for r in referenced]
    assert all(isinstance(r["images"][0], str) for r in referenced)
