"""Unit tests for the DocLang data builder."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from tools.build_doclang_data import build_sft_record, find_image


def _make_image(path: Path) -> None:
    """Create a tiny test image at the given path."""
    Image.new("RGB", (8, 8), color="white").save(path)


def test_find_image_matches_non_png_format(tmp_path: Path) -> None:
    """find_image should locate a source image in any PIL-supported format."""
    _make_image(tmp_path / "doc.jpg")
    assert find_image(tmp_path, "doc") == tmp_path / "doc.jpg"


def test_find_image_returns_none_when_missing(tmp_path: Path) -> None:
    """find_image should return None when no supported image exists."""
    assert find_image(tmp_path, "missing") is None


def test_build_sft_record_attaches_image(tmp_path: Path) -> None:
    """build_sft_record should attach a non-png image to the record."""
    xml_path = tmp_path / "doc.xml"
    xml_path.write_text("<doc>content</doc>", encoding="utf-8")
    image_path = tmp_path / "doc.webp"
    _make_image(image_path)

    record = build_sft_record(xml_path, image_path)
    assert len(record["images"]) == 1
    assert record["completion"][0]["role"] == "assistant"
