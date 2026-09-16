"""Unit tests for the DocLang data builder."""

from __future__ import annotations

from pathlib import Path

from doclang import ValidationError
from PIL import Image

from tools.data_creators.build_doclang_data import (
    build_dataset,
    build_sft_record,
    find_image,
)


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
    doclang_path = tmp_path / "doc.dclg"
    doclang_path.write_text("<doclang>content</doclang>", encoding="utf-8")
    image_path = tmp_path / "doc.webp"
    _make_image(image_path)

    record = build_sft_record(doclang_path, image_path)
    assert len(record["images"]) == 1
    assert record["completion"][0]["role"] == "assistant"
    content = record["prompt"][0]["content"]
    assert content[0] == {"type": "image"}
    assert content[1]["type"] == "text"


def test_build_dataset_includes_valid_documents(tmp_path: Path) -> None:
    """build_dataset should include documents that pass validation and have an image."""
    doclang_path = tmp_path / "doc.dclg"
    doclang_path.write_text("<doclang>content</doclang>", encoding="utf-8")
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    _make_image(images_dir / "doc.png")

    records, skipped = build_dataset(tmp_path, images_dir, validator=lambda _path: None)
    assert skipped == 0
    assert len(records) == 1
    assert records[0]["completion"][0]["content"][0]["text"] == "<doclang>content</doclang>"
    assert len(records[0]["images"]) == 1


def test_build_dataset_skips_invalid_documents(tmp_path: Path) -> None:
    """build_dataset should skip documents that fail validation."""
    doclang_path = tmp_path / "doc.dclg"
    doclang_path.write_text("<doclang>content</doclang>", encoding="utf-8")
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    def failing_validator(_path: Path) -> None:
        raise ValidationError(xsd_errors=[{"message": "not valid DocLang"}], schematron_errors=[])

    records, skipped = build_dataset(tmp_path, images_dir, validator=failing_validator)
    assert skipped == 1
    assert records == []


def test_build_dataset_ignores_non_dclg_files(tmp_path: Path) -> None:
    """build_dataset should only consider .dclg files."""
    (tmp_path / "notes.xml").write_text("<doclang>content</doclang>", encoding="utf-8")
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    records, skipped = build_dataset(tmp_path, images_dir, validator=lambda _path: None)
    assert records == []
    assert skipped == 0


def test_build_dataset_skips_documents_without_image(tmp_path: Path) -> None:
    """build_dataset should skip valid documents that have no matching image."""
    doclang_path = tmp_path / "doc.dclg"
    doclang_path.write_text("<doclang>content</doclang>", encoding="utf-8")
    images_dir = tmp_path / "images"
    images_dir.mkdir()  # empty: no doc.png

    records, skipped = build_dataset(tmp_path, images_dir, validator=lambda _path: None)
    assert skipped == 1
    assert records == []
