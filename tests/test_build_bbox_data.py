"""Unit tests for the bbox <-> text grounding data builder."""

from __future__ import annotations

import random
import re
from pathlib import Path

import numpy as np
from PIL import Image

from tools.data_creators.build_bbox_data import (
    RegionData,
    TextLineData,
    _ordinal,
    build_dataset,
    build_line_count_candidates,
    build_line_index_candidates,
    build_line_neighbor_candidates,
    build_line_ordering_candidates,
    build_lines_to_region_candidates,
    build_region_to_lines_candidates,
    build_region_to_transcription_candidates,
    build_sft_record,
    declared_size,
    extract_doclang_pairs,
    extract_page_xml_pairs,
    extract_page_xml_regions,
    find_image,
    format_bbox,
    format_bbox_list,
    format_qwen_bbox,
    polygon_to_bbox,
    to_doclang_locations,
    to_qwen_bbox,
)

# The coordinate-space sentence appended to qwen prompts (see _coord_space_note).
QWEN_NOTE = " All bounding-box coordinates are integers normalized to the 0-1000 range."


def _make_image(path: Path) -> None:
    """Create a tiny test image at the given path."""
    Image.new("RGB", (8, 8), color="white").save(path)


def _make_page_xml(path: Path, lines: list[tuple[str, str | None, str]]) -> None:
    """Write a minimal PAGE XML file with the given text lines.

    Args:
        path (Path): Where to write the XML file.
        lines (list[tuple[str, str | None, str]]): (line_id, coords_points, text) tuples in
            document order. ``coords_points=None`` omits the ``Coords`` element.
    """
    line_blocks = []
    for line_id, points, text in lines:
        coords = f'\n        <Coords points="{points}"/>' if points is not None else ""
        line_blocks.append(
            f'      <TextLine id="{line_id}">{coords}\n'
            f"        <TextEquiv><Unicode>{text}</Unicode></TextEquiv>\n"
            f"      </TextLine>"
        )
    lines_block = "\n".join(line_blocks)
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">\n'
        "  <Metadata>\n"
        "    <Creator>test</Creator>\n"
        "    <Created>2024-01-01T00:00:00</Created>\n"
        "    <LastChange>2024-01-01T00:00:00</LastChange>\n"
        "  </Metadata>\n"
        '  <Page imageFilename="doc.png" imageWidth="100" imageHeight="100">\n'
        '    <TextRegion id="r1">\n'
        '      <Coords points="0,0 100,0 100,100 0,100"/>\n'
        f"{lines_block}\n"
        "    </TextRegion>\n"
        "  </Page>\n"
        "</PcGts>\n"
    )
    path.write_text(xml, encoding="utf-8")


def _make_page_xml_regions(
    path: Path,
    regions: list[tuple[str, str, list[tuple[str, str | None, str]]]],
) -> None:
    """Write a minimal PAGE XML file with the given text regions.

    Args:
        path (Path): Where to write the XML file.
        regions (list[tuple[str, str, list[tuple[str, str | None, str]]]]): (region_id, region_points, lines)
            tuples in document order, where lines are (line_id, coords_points, text) tuples and
            ``coords_points=None`` omits the ``Coords`` element.
    """
    region_blocks = []
    for region_id, region_points, lines in regions:
        line_blocks = []
        for line_id, points, text in lines:
            coords = f'\n        <Coords points="{points}"/>' if points is not None else ""
            line_blocks.append(
                f'      <TextLine id="{line_id}">{coords}\n'
                f"        <TextEquiv><Unicode>{text}</Unicode></TextEquiv>\n"
                f"      </TextLine>"
            )
        lines_block = "\n".join(line_blocks)
        region_blocks.append(
            f'    <TextRegion id="{region_id}">\n'
            f'      <Coords points="{region_points}"/>\n'
            f"{lines_block}\n"
            f"    </TextRegion>"
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
        '  <Page imageFilename="doc.png" imageWidth="100" imageHeight="100">\n'
        f"{regions_block}\n"
        "  </Page>\n"
        "</PcGts>\n"
    )
    path.write_text(xml, encoding="utf-8")


def _make_doclang(path: Path, body: str, default_resolution: tuple[int, int] | None = None) -> None:
    """Write a minimal DocLang file.

    Args:
        path (Path): Where to write the file.
        body (str): The document body (inner XML of ``<doclang>``).
        default_resolution (tuple[int, int] | None): If set, emit a
            ``<head><default_resolution width height/></head>``.
    """
    head = ""
    if default_resolution is not None:
        width, height = default_resolution
        head = f'<head><default_resolution width="{width}" height="{height}"/></head>'
    path.write_text(f'<doclang version="0.7">{head}{body}</doclang>', encoding="utf-8")


def _make_pool(tmp_path: Path, n_files: int, lines_per_file: int) -> Path:
    """Create ``n_files`` PAGE XML files (each with ``lines_per_file`` lines) plus images."""
    images = tmp_path / "images"
    images.mkdir()
    for i in range(n_files):
        stem = f"doc{i}"
        lines = [
            (f"l{j}", f"0,{j * 10} 100,{j * 10} 100,{j * 10 + 5} 0,{j * 10 + 5}", f"{stem} line {j}")
            for j in range(lines_per_file)
        ]
        _make_page_xml(tmp_path / f"{stem}.xml", lines)
        _make_image(images / f"{stem}.png")
    return images


def _completion_text(record: dict) -> str:
    return record["completion"][0]["content"][0]["text"]


def _prompt_text(record: dict) -> str:
    return record["prompt"][0]["content"][1]["text"]


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
    points = np.array([[10, 20], [110, 20], [110, 45], [10, 45]], dtype=np.int32)
    assert polygon_to_bbox(points) == (10, 20, 110, 45)


def test_polygon_to_bbox_non_rectangular() -> None:
    """An irregular polygon maps to its enclosing axis-aligned rectangle."""
    points = np.array([[0, 0], [50, 10], [40, 60], [5, 55]], dtype=np.int32)
    assert polygon_to_bbox(points) == (0, 0, 50, 60)


# --------------------------------------------------------------------------- #
# to_qwen_bbox / format_qwen_bbox
# --------------------------------------------------------------------------- #


def test_to_qwen_bbox_normalization() -> None:
    """Pixel corners are normalized by the declared size into 0-1000 space."""
    assert to_qwen_bbox((10, 20, 60, 100), (100, 200)) == [100, 100, 600, 500]


def test_to_qwen_bbox_full_image() -> None:
    """A box covering the whole image maps to [0, 0, 1000, 1000]."""
    assert to_qwen_bbox((0, 0, 100, 200), (100, 200)) == [0, 0, 1000, 1000]


def test_to_qwen_bbox_clamps_to_1000() -> None:
    """Coordinates beyond the declared size clamp to 1000."""
    assert to_qwen_bbox((0, 0, 150, 250), (100, 200)) == [0, 0, 1000, 1000]


def test_format_qwen_bbox() -> None:
    """The bbox is formatted as '[x1, y1, x2, y2]'."""
    assert format_qwen_bbox([100, 100, 600, 500]) == "[100, 100, 600, 500]"


# --------------------------------------------------------------------------- #
# to_doclang_locations
# --------------------------------------------------------------------------- #


def test_to_doclang_locations() -> None:
    """Corners (x1, y1, x2, y2) serialize directly as x_min, y_min, x_max, y_max."""
    assert to_doclang_locations((10, 20, 110, 45)) == (
        '<location value="10"/><location value="20"/><location value="110"/><location value="45"/>'
    )


# --------------------------------------------------------------------------- #
# extract_page_xml_pairs
# --------------------------------------------------------------------------- #


def test_extract_page_xml_pairs(tmp_path: Path) -> None:
    """Each TextLine yields a (text, bbox) pair in document order."""
    xml_path = tmp_path / "doc.xml"
    _make_page_xml(
        xml_path,
        [
            ("l1", "0,0 100,0 100,25 0,25", "first line"),
            ("l2", "0,30 80,30 80,55 0,55", "second line"),
        ],
    )
    assert extract_page_xml_pairs(xml_path) == [
        ("first line", (0, 0, 100, 25)),
        ("second line", (0, 30, 80, 55)),
    ]


def test_extract_page_xml_pairs_skips_empty_text(tmp_path: Path) -> None:
    """Lines with only whitespace are skipped."""
    xml_path = tmp_path / "doc.xml"
    _make_page_xml(
        xml_path,
        [
            ("l1", "0,0 100,0 100,25 0,25", "   "),
            ("l2", "0,30 80,30 80,55 0,55", "kept"),
        ],
    )
    assert extract_page_xml_pairs(xml_path) == [("kept", (0, 30, 80, 55))]


def test_extract_page_xml_pairs_skips_line_without_coords(tmp_path: Path) -> None:
    """A line without a Coords polygon is skipped; the rest of the file is kept."""
    xml_path = tmp_path / "doc.xml"
    _make_page_xml(
        xml_path,
        [
            ("l1", None, "no coords"),
            ("l2", "0,30 80,30 80,55 0,55", "kept"),
        ],
    )
    assert extract_page_xml_pairs(xml_path) == [("kept", (0, 30, 80, 55))]


# --------------------------------------------------------------------------- #
# extract_doclang_pairs
# --------------------------------------------------------------------------- #


def test_extract_doclang_pairs(tmp_path: Path) -> None:
    """Each <text> with four locations yields a (text, bbox) pair."""
    dclg_path = tmp_path / "doc.dclg"
    _make_doclang(
        dclg_path,
        '<text><location value="0"/><location value="0"/><location value="100"/><location value="50"/>Hello</text>'
        '<text><location value="10"/><location value="60"/><location value="80"/><location value="30"/>World</text>',
        default_resolution=(100, 100),
    )
    assert extract_doclang_pairs(dclg_path) == [
        ("Hello", (0, 0, 100, 50)),
        ("World", (10, 60, 80, 30)),
    ]


def test_extract_doclang_pairs_skips_without_locations(tmp_path: Path) -> None:
    """A <text> without four locations is skipped."""
    dclg_path = tmp_path / "doc.dclg"
    _make_doclang(
        dclg_path,
        "<text>No locations</text>"
        '<text><location value="0"/><location value="0"/><location value="100"/><location value="50"/>Kept</text>',
    )
    assert extract_doclang_pairs(dclg_path) == [("Kept", (0, 0, 100, 50))]


def test_extract_doclang_pairs_skips_empty_text(tmp_path: Path) -> None:
    """A <text> with four locations but only whitespace is skipped."""
    dclg_path = tmp_path / "doc.dclg"
    _make_doclang(
        dclg_path,
        '<text><location value="0"/><location value="0"/><location value="100"/><location value="50"/>   </text>'
        '<text><location value="0"/><location value="60"/><location value="100"/><location value="30"/>Kept</text>',
    )
    assert extract_doclang_pairs(dclg_path) == [("Kept", (0, 60, 100, 30))]


def test_extract_doclang_pairs_finds_nested_text(tmp_path: Path) -> None:
    """<text> elements nested in groups and list items are found at any depth."""
    dclg_path = tmp_path / "doc.dclg"
    _make_doclang(
        dclg_path,
        '<group><text><location value="1"/><location value="2"/><location value="3"/><location value="4"/>Nested</text></group>'
        '<list class="unordered"><ldiv/><text><location value="5"/><location value="6"/>'
        '<location value="7"/><location value="8"/>Item</text></list>',
    )
    assert extract_doclang_pairs(dclg_path) == [
        ("Nested", (1, 2, 3, 4)),
        ("Item", (5, 6, 7, 8)),
    ]


# --------------------------------------------------------------------------- #
# declared_size
# --------------------------------------------------------------------------- #


def test_declared_size_page_xml(tmp_path: Path) -> None:
    """PAGE XML declared size comes from the Page imageWidth/imageHeight."""
    xml_path = tmp_path / "doc.xml"
    _make_page_xml(xml_path, [("l1", "0,0 10,0 10,10 0,10", "x")])
    assert declared_size(xml_path) == (100, 100)


def test_declared_size_doclang(tmp_path: Path) -> None:
    """DocLang declared size comes from <default_resolution>."""
    dclg_path = tmp_path / "doc.dclg"
    _make_doclang(dclg_path, "<text>hi</text>", default_resolution=(200, 300))
    assert declared_size(dclg_path) == (200, 300)


def test_declared_size_doclang_without_resolution(tmp_path: Path) -> None:
    """A DocLang file without <default_resolution> has no declared size."""
    dclg_path = tmp_path / "doc.dclg"
    _make_doclang(dclg_path, "<text>hi</text>")
    assert declared_size(dclg_path) is None


def test_declared_size_malformed_doclang_returns_none(tmp_path: Path) -> None:
    """A .dclg missing a required attribute (TypeError) yields None, not a crash."""
    dclg_path = tmp_path / "bad.dclg"
    _make_doclang(dclg_path, "<text><thread/>x</text>")
    assert declared_size(dclg_path) is None


# --------------------------------------------------------------------------- #
# build_sft_record (all mode x format combos)
# --------------------------------------------------------------------------- #


def test_build_sft_record_bbox_to_text_qwen(tmp_path: Path) -> None:
    """bbox_to_text + qwen: prompt names the normalized box, completion is the text."""
    image_path = tmp_path / "doc.png"
    _make_image(image_path)
    record = build_sft_record("bbox_to_text", "qwen", "hello", (10, 20, 60, 100), (100, 200), image_path)
    content = record["prompt"][0]["content"]
    assert content[0] == {"type": "image"}
    assert content[1]["text"] == "Transcribe the text inside the bounding box [100, 100, 600, 500]." + QWEN_NOTE
    assert _completion_text(record) == "hello"
    assert len(record["images"]) == 1


def test_build_sft_record_bbox_to_text_doclang(tmp_path: Path) -> None:
    """bbox_to_text + doclang: prompt shows bare locations (no <text> wrapper)."""
    image_path = tmp_path / "doc.png"
    _make_image(image_path)
    record = build_sft_record("bbox_to_text", "doclang", "hello", (10, 20, 110, 45), (100, 200), image_path)
    assert _prompt_text(record) == (
        "Transcribe the text inside the element with these DocLang locations: "
        '<location value="10"/><location value="20"/><location value="110"/><location value="45"/>.'
    )
    assert _completion_text(record) == "hello"


def test_build_sft_record_text_to_bbox_qwen(tmp_path: Path) -> None:
    """text_to_bbox + qwen: prompt names the text, completion is the normalized box."""
    image_path = tmp_path / "doc.png"
    _make_image(image_path)
    record = build_sft_record("text_to_bbox", "qwen", "hello", (10, 20, 60, 100), (100, 200), image_path)
    assert _prompt_text(record) == 'Predict the bounding box for the text "hello".' + QWEN_NOTE
    assert _completion_text(record) == "[100, 100, 600, 500]"


def test_build_sft_record_text_to_bbox_doclang(tmp_path: Path) -> None:
    """text_to_bbox + doclang: completion is the full <text> element with locations."""
    image_path = tmp_path / "doc.png"
    _make_image(image_path)
    record = build_sft_record("text_to_bbox", "doclang", "hello", (10, 20, 110, 45), (100, 200), image_path)
    assert _prompt_text(record) == ('Emit the DocLang <text> element with its <location> coordinates for the text: "hello".')
    assert _completion_text(record) == (
        '<text><location value="10"/><location value="20"/><location value="110"/><location value="45"/>hello</text>'
    )


# --------------------------------------------------------------------------- #
# build_dataset
# --------------------------------------------------------------------------- #


def test_build_dataset_mixed_input(tmp_path: Path) -> None:
    """build_dataset handles a mix of .xml and .dclg files."""
    _make_page_xml(tmp_path / "a.xml", [("l1", "0,0 100,0 100,25 0,25", "from xml")])
    _make_doclang(
        tmp_path / "b.dclg",
        '<text><location value="0"/><location value="0"/><location value="100"/><location value="50"/>from dclg</text>',
        default_resolution=(100, 100),
    )
    images = tmp_path / "images"
    images.mkdir()
    _make_image(images / "a.png")
    _make_image(images / "b.png")

    records, skipped = build_dataset(tmp_path, images, "bbox_to_text", "qwen")
    assert skipped == 0
    assert len(records) == 2


def test_build_dataset_skips_file_without_image(tmp_path: Path) -> None:
    """A file with pairs but no matching image is skipped."""
    _make_page_xml(tmp_path / "a.xml", [("l1", "0,0 100,0 100,25 0,25", "text")])
    images = tmp_path / "images"
    images.mkdir()

    records, skipped = build_dataset(tmp_path, images, "bbox_to_text", "qwen")
    assert skipped == 1
    assert records == []


def test_build_dataset_skips_file_without_pairs(tmp_path: Path) -> None:
    """A file with no usable (text, bbox) pairs is skipped."""
    _make_doclang(tmp_path / "a.dclg", "<text>no locations</text>")
    images = tmp_path / "images"
    images.mkdir()
    _make_image(images / "a.png")

    records, skipped = build_dataset(tmp_path, images, "bbox_to_text", "qwen")
    assert skipped == 1
    assert records == []


def test_build_dataset_skips_malformed_doclang(tmp_path: Path) -> None:
    """A .dclg missing a required attribute (TypeError) is skipped, not fatal."""
    # <thread/> without thread_id makes doclang_xml_to_model raise TypeError.
    _make_doclang(tmp_path / "bad.dclg", "<text><thread/>x</text>")
    _make_page_xml(tmp_path / "good.xml", [("l1", "0,0 100,0 100,25 0,25", "ok")])
    images = tmp_path / "images"
    images.mkdir()
    _make_image(images / "bad.png")
    _make_image(images / "good.png")

    records, skipped = build_dataset(tmp_path, images, "bbox_to_text", "qwen")
    assert skipped == 1
    assert [_completion_text(r) for r in records] == ["ok"]


def test_build_dataset_ignores_other_extensions(tmp_path: Path) -> None:
    """Files that are neither .xml nor .dclg are ignored (not counted as skipped)."""
    (tmp_path / "notes.txt").write_text("not a doc", encoding="utf-8")
    images = tmp_path / "images"
    images.mkdir()

    records, skipped = build_dataset(tmp_path, images, "bbox_to_text", "qwen")
    assert records == []
    assert skipped == 0


def test_build_dataset_falls_back_to_image_size(tmp_path: Path) -> None:
    """A DocLang file without <default_resolution> normalizes by the actual image size."""
    dclg_path = tmp_path / "doc.dclg"
    _make_doclang(
        dclg_path,
        '<text><location value="0"/><location value="0"/><location value="4"/><location value="4"/>half</text>',
    )
    images = tmp_path / "images"
    images.mkdir()
    _make_image(images / "doc.png")  # 8x8

    records, skipped = build_dataset(tmp_path, images, "text_to_bbox", "qwen")
    assert skipped == 0
    assert _completion_text(records[0]) == "[0, 0, 500, 500]"


def test_build_dataset_no_sampling_preserves_order(tmp_path: Path) -> None:
    """Without --max-lines, records follow deterministic file/line order."""
    images = _make_pool(tmp_path, 2, 2)
    records, _ = build_dataset(tmp_path, images, "bbox_to_text", "qwen")
    assert [_completion_text(r) for r in records] == [
        "doc0 line 0",
        "doc0 line 1",
        "doc1 line 0",
        "doc1 line 1",
    ]


# --------------------------------------------------------------------------- #
# build_dataset — global random sampling
# --------------------------------------------------------------------------- #


def test_build_dataset_sampling_caps_pool(tmp_path: Path) -> None:
    """When the pool exceeds max_lines, exactly max_lines records are kept."""
    images = _make_pool(tmp_path, 3, 4)  # 12 candidates
    records, skipped = build_dataset(tmp_path, images, "bbox_to_text", "qwen", max_lines=5)
    assert skipped == 0
    assert len(records) == 5


def test_build_dataset_sampling_keeps_all_when_pool_smaller(tmp_path: Path) -> None:
    """When the pool is at most max_lines, all candidates are kept."""
    images = _make_pool(tmp_path, 1, 3)  # 3 candidates
    records, skipped = build_dataset(tmp_path, images, "bbox_to_text", "qwen", max_lines=10)
    assert skipped == 0
    assert len(records) == 3


def test_build_dataset_sampling_subset_of_global_pool(tmp_path: Path) -> None:
    """Sampled records are a duplicate-free subset of the global pool (across files)."""
    images = _make_pool(tmp_path, 3, 4)  # 12 candidates across 3 files
    records, _ = build_dataset(tmp_path, images, "bbox_to_text", "qwen", max_lines=5, seed=7)
    texts = [_completion_text(r) for r in records]
    full_pool = {f"doc{i} line {j}" for i in range(3) for j in range(4)}
    assert len(texts) == 5
    assert len(set(texts)) == 5
    assert set(texts) <= full_pool


def test_build_dataset_sampling_is_reproducible_with_seed(tmp_path: Path) -> None:
    """The same seed yields the identical sampled selection and order."""
    images = _make_pool(tmp_path, 3, 4)
    records_a, _ = build_dataset(tmp_path, images, "bbox_to_text", "qwen", max_lines=5, seed=42)
    records_b, _ = build_dataset(tmp_path, images, "bbox_to_text", "qwen", max_lines=5, seed=42)
    assert [_completion_text(r) for r in records_a] == [_completion_text(r) for r in records_b]


# --------------------------------------------------------------------------- #
# extract_page_xml_regions
# --------------------------------------------------------------------------- #


def test_extract_page_xml_regions(tmp_path: Path) -> None:
    """Each TextRegion yields a RegionData with its lines in document order."""
    xml_path = tmp_path / "doc.xml"
    _make_page_xml_regions(
        xml_path,
        [
            (
                "r1",
                "0,0 100,0 100,50 0,50",
                [
                    ("l1", "0,0 100,0 100,25 0,25", "first line"),
                    ("l2", "0,30 80,30 80,50 0,50", "second line"),
                ],
            ),
            (
                "r2",
                "0,60 100,60 100,100 0,100",
                [("l3", "0,60 100,60 100,85 0,85", "third line")],
            ),
        ],
    )
    regions = extract_page_xml_regions(xml_path)
    assert len(regions) == 2
    assert regions[0].bbox == (0, 0, 100, 50)
    assert [(line.text, line.bbox) for line in regions[0].lines] == [
        ("first line", (0, 0, 100, 25)),
        ("second line", (0, 30, 80, 50)),
    ]
    assert regions[1].bbox == (0, 60, 100, 100)
    assert [(line.text, line.bbox) for line in regions[1].lines] == [
        ("third line", (0, 60, 100, 85)),
    ]


def test_extract_page_xml_regions_skips_line_without_coords(tmp_path: Path) -> None:
    """A line without a Coords polygon is skipped; the region is kept."""
    xml_path = tmp_path / "doc.xml"
    _make_page_xml_regions(
        xml_path,
        [
            (
                "r1",
                "0,0 100,0 100,100 0,100",
                [
                    ("l1", None, "no coords"),
                    ("l2", "0,30 80,30 80,55 0,55", "kept"),
                ],
            ),
        ],
    )
    regions = extract_page_xml_regions(xml_path)
    assert len(regions) == 1
    assert [(line.text, line.bbox) for line in regions[0].lines] == [("kept", (0, 30, 80, 55))]


def test_extract_page_xml_regions_skips_region_without_usable_lines(tmp_path: Path) -> None:
    """A region whose lines all lack text or coords is skipped."""
    xml_path = tmp_path / "doc.xml"
    _make_page_xml_regions(
        xml_path,
        [
            (
                "r1",
                "0,0 100,0 100,50 0,50",
                [
                    ("l1", None, "no coords"),
                    ("l2", "0,30 80,30 80,50 0,50", "   "),
                ],
            ),
            (
                "r2",
                "0,60 100,60 100,100 0,100",
                [("l3", "0,60 100,60 100,85 0,85", "kept")],
            ),
        ],
    )
    regions = extract_page_xml_regions(xml_path)
    assert len(regions) == 1
    assert regions[0].bbox == (0, 60, 100, 100)


# --------------------------------------------------------------------------- #
# format_bbox / format_bbox_list / _ordinal
# --------------------------------------------------------------------------- #


def test_format_bbox_qwen() -> None:
    """qwen format normalizes to the 0-1000 range."""
    assert format_bbox((10, 20, 60, 100), "qwen", (100, 200)) == "[100, 100, 600, 500]"


def test_format_bbox_doclang() -> None:
    """doclang format emits the four <location> elements in native space."""
    assert format_bbox((10, 20, 110, 45), "doclang", (100, 200)) == (
        '<location value="10"/><location value="20"/><location value="110"/><location value="45"/>'
    )


def test_format_bbox_list_qwen() -> None:
    """A list of boxes serializes as an array of normalized boxes."""
    assert format_bbox_list([(10, 20, 60, 100), (0, 0, 100, 200)], "qwen", (100, 200)) == (
        "[[100, 100, 600, 500], [0, 0, 1000, 1000]]"
    )


def test_format_bbox_list_doclang() -> None:
    """In doclang format the list uses native-space integers."""
    assert format_bbox_list([(10, 20, 60, 100), (0, 0, 100, 200)], "doclang", (100, 200)) == (
        "[[10, 20, 60, 100], [0, 0, 100, 200]]"
    )


def test_ordinal() -> None:
    """Ordinals handle the st/nd/rd/th cases including the teens."""
    assert _ordinal(1) == "1st"
    assert _ordinal(2) == "2nd"
    assert _ordinal(3) == "3rd"
    assert _ordinal(4) == "4th"
    assert _ordinal(11) == "11th"
    assert _ordinal(22) == "22nd"


# --------------------------------------------------------------------------- #
# region-mode candidate builders
# --------------------------------------------------------------------------- #


def _three_line_region() -> RegionData:
    return RegionData(
        bbox=(0, 0, 100, 100),
        lines=[
            TextLineData(text="first", bbox=(0, 0, 100, 25)),
            TextLineData(text="second", bbox=(0, 30, 100, 55)),
            TextLineData(text="third", bbox=(0, 60, 100, 85)),
        ],
    )


def _single_line_region() -> RegionData:
    return RegionData(bbox=(0, 0, 100, 100), lines=[TextLineData(text="only", bbox=(0, 0, 100, 25))])


def test_build_line_neighbor_candidates_targets() -> None:
    """Each candidate targets the correct offset line in the correct direction."""
    region = _three_line_region()
    candidates = build_line_neighbor_candidates(region, "qwen", (100, 100), max_n=1)
    # 4 valid (line, direction, n) combos x 2 references x 2 outputs = 16
    assert len(candidates) == 16
    by_prompt = {prompt: completion for prompt, completion in candidates}
    assert by_prompt['What is the text of the 1st line below the line "first"?'] == "second"
    assert by_prompt['What is the text of the 1st line above the line "second"?'] == "first"
    assert by_prompt['What is the text of the 1st line below the line "second"?'] == "third"
    assert by_prompt['What is the text of the 1st line above the line "third"?'] == "second"
    assert by_prompt['What is the bounding box of the 1st line below the line "first"?' + QWEN_NOTE] == "[0, 300, 1000, 550]"
    assert (
        by_prompt["What is the text of the 1st line below the line with bounding box [0, 0, 1000, 250]?" + QWEN_NOTE]
        == "second"
    )


def test_build_line_neighbor_candidates_max_n() -> None:
    """max_n=2 adds the second-neighbor candidates with the right ordinal."""
    region = _three_line_region()
    candidates = build_line_neighbor_candidates(region, "qwen", (100, 100), max_n=2)
    # 6 valid (line, direction, n) combos x 4 variants = 24
    assert len(candidates) == 24
    by_prompt = {prompt: completion for prompt, completion in candidates}
    assert by_prompt['What is the text of the 2nd line below the line "first"?'] == "third"
    assert by_prompt['What is the text of the 2nd line above the line "third"?'] == "first"
    assert not any("3rd line" in prompt for prompt, _ in candidates)


def test_build_region_to_lines_candidates() -> None:
    """The completion lists all line bboxes in reading order."""
    region = _three_line_region()
    prompt, completion = build_region_to_lines_candidates(region, "qwen", (100, 100))[0]
    assert prompt == (
        "List the bounding boxes of all text lines in the paragraph [0, 0, 1000, 1000], in reading order." + QWEN_NOTE
    )
    assert completion == "[[0, 0, 1000, 250], [0, 300, 1000, 550], [0, 600, 1000, 850]]"


def test_build_region_to_lines_candidates_doclang() -> None:
    """doclang format uses the region locations and native-space list."""
    region = _three_line_region()
    prompt, completion = build_region_to_lines_candidates(region, "doclang", (100, 100))[0]
    assert prompt == (
        "List the bounding boxes of all text lines in the paragraph "
        '<location value="0"/><location value="0"/><location value="100"/><location value="100"/>, '
        "in reading order."
    )
    assert completion == "[[0, 0, 100, 25], [0, 30, 100, 55], [0, 60, 100, 85]]"


def test_build_region_to_lines_candidates_skips_single_line() -> None:
    """A region with a single line yields no candidates."""
    assert build_region_to_lines_candidates(_single_line_region(), "qwen", (100, 100)) == []


def test_build_lines_to_region_candidates() -> None:
    """The completion is the region bbox; the prompt lists the line bboxes."""
    region = _three_line_region()
    prompt, completion = build_lines_to_region_candidates(region, "qwen", (100, 100))[0]
    assert prompt == (
        "Given the text lines [[0, 0, 1000, 250], [0, 300, 1000, 550], [0, 600, 1000, 850]], "
        "what is the bounding box of the paragraph they belong to?" + QWEN_NOTE
    )
    assert completion == "[0, 0, 1000, 1000]"


def test_build_lines_to_region_candidates_skips_single_line() -> None:
    """A region with a single line yields no candidates."""
    assert build_lines_to_region_candidates(_single_line_region(), "qwen", (100, 100)) == []


def test_build_region_to_transcription_candidates() -> None:
    """The completion joins all lines' text with newlines in reading order."""
    region = _three_line_region()
    prompt, completion = build_region_to_transcription_candidates(region, "qwen", (100, 100))[0]
    assert prompt == "Transcribe all text lines in the paragraph [0, 0, 1000, 1000], in reading order." + QWEN_NOTE
    assert completion == "first\nsecond\nthird"


def test_build_region_to_transcription_candidates_skips_single_line() -> None:
    """A region with a single line yields no candidates."""
    assert build_region_to_transcription_candidates(_single_line_region(), "qwen", (100, 100)) == []


def test_build_line_index_candidates() -> None:
    """Each line yields a text-reference and a bbox-reference candidate."""
    region = _three_line_region()
    candidates = build_line_index_candidates(region, "qwen", (100, 100))
    assert len(candidates) == 6
    by_prompt = {prompt: completion for prompt, completion in candidates}
    assert by_prompt['What is the position of the line "first" in its paragraph?'] == "line 1 of 3"
    assert by_prompt['What is the position of the line "second" in its paragraph?'] == "line 2 of 3"
    assert by_prompt['What is the position of the line "third" in its paragraph?'] == "line 3 of 3"
    assert (
        by_prompt["What is the position of the line with bounding box [0, 300, 1000, 550] in its paragraph?" + QWEN_NOTE]
        == "line 2 of 3"
    )


def test_build_line_index_candidates_skips_single_line() -> None:
    """A region with a single line yields no candidates."""
    assert build_line_index_candidates(_single_line_region(), "qwen", (100, 100)) == []


def test_build_line_ordering_candidates_roundtrip() -> None:
    """The completion maps the presented (shuffled) boxes back to reading order."""
    region = _three_line_region()
    rng = random.Random(0)
    prompt, completion = build_line_ordering_candidates(region, "qwen", (100, 100), rng)[0]
    presented = [tuple(map(int, box)) for box in re.findall(r"\[(\d+), (\d+), (\d+), (\d+)\]", prompt)]
    assert len(presented) == 3
    # the presented order must actually be shuffled
    assert presented != [tuple(to_qwen_bbox(line.bbox, (100, 100))) for line in region.lines]
    answer = [int(index) for index in completion.split(",")]
    assert sorted(answer) == [1, 2, 3]
    for position, line in enumerate(region.lines):
        assert presented[answer[position] - 1] == tuple(to_qwen_bbox(line.bbox, (100, 100)))


def test_build_line_ordering_candidates_requires_three_lines() -> None:
    """Regions with fewer than three lines yield no candidates."""
    region = RegionData(
        bbox=(0, 0, 100, 100),
        lines=[
            TextLineData(text="a", bbox=(0, 0, 100, 25)),
            TextLineData(text="b", bbox=(0, 30, 100, 55)),
        ],
    )
    assert build_line_ordering_candidates(region, "qwen", (100, 100), random.Random(0)) == []


def test_build_line_count_candidates() -> None:
    """The completion is the number of lines in the region."""
    region = _three_line_region()
    prompt, completion = build_line_count_candidates(region, "qwen", (100, 100))[0]
    assert prompt == "How many text lines are in the paragraph [0, 0, 1000, 1000]?" + QWEN_NOTE
    assert completion == "3"


# --------------------------------------------------------------------------- #
# build_dataset — region-based modes
# --------------------------------------------------------------------------- #


def _make_region_pool(tmp_path: Path) -> Path:
    """Create a PAGE XML file with two regions (3 and 2 lines) plus an image."""
    images = tmp_path / "images"
    images.mkdir()
    _make_page_xml_regions(
        tmp_path / "doc.xml",
        [
            (
                "r1",
                "0,0 100,0 100,50 0,50",
                [
                    ("l1", "0,0 100,0 100,25 0,25", "alpha one"),
                    ("l2", "0,30 100,30 100,50 0,50", "alpha two"),
                    ("l3", "10,10 50,10 50,20 10,20", "alpha three"),
                ],
            ),
            (
                "r2",
                "0,60 100,60 100,100 0,100",
                [
                    ("l4", "0,60 100,60 100,80 0,80", "beta one"),
                    ("l5", "0,85 100,85 100,100 0,100", "beta two"),
                ],
            ),
        ],
    )
    _make_image(images / "doc.png")
    return images


def test_build_dataset_region_mode_region_to_lines(tmp_path: Path) -> None:
    """region_to_lines emits one record per region with >=2 lines."""
    images = _make_region_pool(tmp_path)
    records, skipped = build_dataset(tmp_path, images, "region_to_lines", "qwen")
    assert skipped == 0
    assert len(records) == 2
    assert sorted(_completion_text(record) for record in records) == [
        "[[0, 0, 1000, 250], [0, 300, 1000, 500], [100, 100, 500, 200]]",
        "[[0, 600, 1000, 800], [0, 850, 1000, 1000]]",
    ]


def test_build_dataset_region_mode_line_neighbor(tmp_path: Path) -> None:
    """line_neighbor emits prompt/completion pairs with the correct targets."""
    images = _make_region_pool(tmp_path)
    records, skipped = build_dataset(tmp_path, images, "line_neighbor", "qwen", max_n=1)
    assert skipped == 0
    # r1: 4 combos x 4 variants; r2: 2 combos x 4 variants
    assert len(records) == 24
    by_prompt = {_prompt_text(r): _completion_text(r) for r in records}
    assert by_prompt['What is the text of the 1st line below the line "alpha one"?'] == "alpha two"
    assert by_prompt['What is the text of the 1st line above the line "alpha two"?'] == "alpha one"
    assert by_prompt['What is the text of the 1st line below the line "beta one"?'] == "beta two"
    assert (
        by_prompt['What is the bounding box of the 1st line below the line "alpha one"?' + QWEN_NOTE] == "[0, 300, 1000, 500]"
    )


def test_build_dataset_region_mode_ignores_dclg(tmp_path: Path) -> None:
    """Region-based modes only process .xml files."""
    images = _make_region_pool(tmp_path)
    _make_doclang(
        tmp_path / "other.dclg",
        '<text><location value="0"/><location value="0"/><location value="100"/><location value="50"/>ignored</text>',
        default_resolution=(100, 100),
    )
    records, skipped = build_dataset(tmp_path, images, "line_count", "qwen")
    assert skipped == 0
    assert len(records) == 2
    assert sorted(_completion_text(r) for r in records) == ["2", "3"]


def test_build_dataset_region_mode_skips_file_without_image(tmp_path: Path) -> None:
    """A .xml file with regions but no matching image is skipped."""
    _make_page_xml_regions(
        tmp_path / "doc.xml",
        [
            (
                "r1",
                "0,0 100,0 100,100 0,100",
                [
                    ("l1", "0,0 100,0 100,25 0,25", "a"),
                    ("l2", "0,30 100,30 100,55 0,55", "b"),
                ],
            ),
        ],
    )
    images = tmp_path / "images"
    images.mkdir()
    records, skipped = build_dataset(tmp_path, images, "line_count", "qwen")
    assert skipped == 1
    assert records == []


def test_build_dataset_region_mode_skips_malformed_xml(tmp_path: Path) -> None:
    """A .xml file that is not well-formed is skipped, not fatal."""
    images = _make_region_pool(tmp_path)
    (tmp_path / "bad.xml").write_text("<not-closed", encoding="utf-8")
    records, skipped = build_dataset(tmp_path, images, "line_count", "qwen")
    assert skipped == 1
    assert len(records) == 2


def test_build_dataset_region_mode_skips_single_line_regions(tmp_path: Path) -> None:
    """region_to_lines emits no records when all regions have a single line."""
    images = tmp_path / "images"
    images.mkdir()
    _make_page_xml_regions(
        tmp_path / "doc.xml",
        [("r1", "0,0 100,0 100,100 0,100", [("l1", "0,0 100,0 100,25 0,25", "only")])],
    )
    _make_image(images / "doc.png")
    records, skipped = build_dataset(tmp_path, images, "region_to_lines", "qwen")
    assert skipped == 0
    assert records == []


def test_build_dataset_region_mode_sampling_is_reproducible(tmp_path: Path) -> None:
    """line_neighbor with max_lines + seed samples a reproducible subset."""
    images = _make_region_pool(tmp_path)
    records_a, _ = build_dataset(tmp_path, images, "line_neighbor", "qwen", max_lines=5, seed=42)
    records_b, _ = build_dataset(tmp_path, images, "line_neighbor", "qwen", max_lines=5, seed=42)
    assert len(records_a) == 5
    assert [_prompt_text(r) for r in records_a] == [_prompt_text(r) for r in records_b]
    assert [_completion_text(r) for r in records_a] == [_completion_text(r) for r in records_b]
