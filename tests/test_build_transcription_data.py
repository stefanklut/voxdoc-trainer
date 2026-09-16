"""Unit tests for the PAGE XML transcription data builder."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from tools.data_creators.build_transcription_data import (
    build_dataset,
    build_sft_record,
    extract_transcription,
    find_image,
)


def _make_image(path: Path) -> None:
    """Create a tiny test image at the given path."""
    Image.new("RGB", (8, 8), color="white").save(path)


def _make_page_xml(path: Path, lines: list[tuple[str, int | None, str]]) -> None:
    """Write a minimal PAGE XML file with the given text lines.

    Args:
        path (Path): Where to write the XML file.
        lines (list[tuple[str, int | None, str]]): (line_id, reading_order, text) tuples
            in document order.
    """
    line_blocks = []
    for line_id, reading_order, text in lines:
        custom = f' custom="readingOrder {{index:{reading_order};}}"' if reading_order is not None else ""
        line_blocks.append(
            f'      <TextLine id="{line_id}"{custom}>\n'
            f'        <Coords points="0,0 100,0 100,25 0,25"/>\n'
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


def test_find_image_matches_non_png_format(tmp_path: Path) -> None:
    """find_image should locate a source image in any PIL-supported format."""
    _make_image(tmp_path / "doc.jpg")
    assert find_image(tmp_path, "doc") == tmp_path / "doc.jpg"


def test_find_image_returns_none_when_missing(tmp_path: Path) -> None:
    """find_image should return None when no supported image exists."""
    assert find_image(tmp_path, "missing") is None


def test_extract_transcription_orders_by_reading_order(tmp_path: Path) -> None:
    """Lines should be ordered by reading order, not document order."""
    xml_path = tmp_path / "doc.xml"
    # Document order: index 2 first, then index 1.
    _make_page_xml(xml_path, [("l2", 2, "second line"), ("l1", 1, "first line")])
    assert extract_transcription(xml_path) == "first line\nsecond line"


def test_extract_transcription_document_order_without_reading_order(tmp_path: Path) -> None:
    """Lines without a reading order keep their document order."""
    xml_path = tmp_path / "doc.xml"
    _make_page_xml(xml_path, [("l1", None, "line one"), ("l2", None, "line two")])
    assert extract_transcription(xml_path) == "line one\nline two"


def test_extract_transcription_empty_when_no_text(tmp_path: Path) -> None:
    """A PAGE XML file with no text lines yields an empty transcription."""
    xml_path = tmp_path / "doc.xml"
    _make_page_xml(xml_path, [])
    assert extract_transcription(xml_path) == ""


def test_extract_transcription_prefers_unicode_over_plaintext(tmp_path: Path) -> None:
    """When TextEquiv has both PlainText and Unicode, the Unicode value is used."""
    xml_path = tmp_path / "doc.xml"
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
        '      <TextLine id="l1" custom="readingOrder {index:1;}">\n'
        '        <Coords points="0,0 100,0 100,25 0,25"/>\n'
        "        <TextEquiv>\n"
        "          <PlainText>plain text version</PlainText>\n"
        "          <Unicode>unicode version</Unicode>\n"
        "        </TextEquiv>\n"
        "      </TextLine>\n"
        "    </TextRegion>\n"
        "  </Page>\n"
        "</PcGts>\n"
    )
    xml_path.write_text(xml, encoding="utf-8")
    assert extract_transcription(xml_path) == "unicode version"


def test_build_sft_record_attaches_image(tmp_path: Path) -> None:
    """build_sft_record should attach the image and set the transcription as completion."""
    image_path = tmp_path / "doc.png"
    _make_image(image_path)

    record = build_sft_record("hello world", image_path)
    assert len(record["images"]) == 1
    assert record["completion"][0]["role"] == "assistant"
    assert record["completion"][0]["content"][0]["text"] == "hello world"
    content = record["prompt"][0]["content"]
    assert content[0] == {"type": "image"}
    assert content[1]["type"] == "text"


def test_build_dataset_includes_valid_documents(tmp_path: Path) -> None:
    """build_dataset should include valid PAGE XML files that have a matching image."""
    xml_path = tmp_path / "doc.xml"
    _make_page_xml(xml_path, [("l1", 1, "hello world")])
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    _make_image(images_dir / "doc.png")

    records, skipped = build_dataset(tmp_path, images_dir)
    assert skipped == 0
    assert len(records) == 1
    assert records[0]["completion"][0]["content"][0]["text"] == "hello world"
    assert len(records[0]["images"]) == 1


def test_build_dataset_skips_malformed_xml(tmp_path: Path) -> None:
    """build_dataset should skip files that are not well-formed XML."""
    (tmp_path / "bad.xml").write_text("<PcGts><Page>", encoding="utf-8")
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    records, skipped = build_dataset(tmp_path, images_dir)
    assert skipped == 1
    assert records == []


def test_build_dataset_skips_files_without_text(tmp_path: Path) -> None:
    """build_dataset should skip PAGE XML files that contain no text."""
    xml_path = tmp_path / "doc.xml"
    _make_page_xml(xml_path, [])
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    _make_image(images_dir / "doc.png")

    records, skipped = build_dataset(tmp_path, images_dir)
    assert skipped == 1
    assert records == []


def test_build_dataset_skips_files_without_image(tmp_path: Path) -> None:
    """build_dataset should skip valid PAGE XML files with no matching image."""
    xml_path = tmp_path / "doc.xml"
    _make_page_xml(xml_path, [("l1", 1, "hello world")])
    images_dir = tmp_path / "images"
    images_dir.mkdir()  # empty: no doc.png

    records, skipped = build_dataset(tmp_path, images_dir)
    assert skipped == 1
    assert records == []


def test_build_dataset_ignores_non_xml_files(tmp_path: Path) -> None:
    """build_dataset should only consider .xml files."""
    (tmp_path / "notes.txt").write_text("not xml", encoding="utf-8")
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    records, skipped = build_dataset(tmp_path, images_dir)
    assert records == []
    assert skipped == 0
