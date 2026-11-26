"""Unit tests for the DocLang data builder."""

from __future__ import annotations

from pathlib import Path

from doclang import ValidationError
from PIL import Image
import pytest

from tools.data_creators.build_doclang_data import (
    build_dataset,
    build_sft_record,
    find_image,
)
from utils.discovery import DiscoveryError


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
    (tmp_path / "doc.dclg").write_text("<doclang>content</doclang>", encoding="utf-8")
    (tmp_path / "notes.xml").write_text("<doclang>content</doclang>", encoding="utf-8")
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    _make_image(images_dir / "doc.png")

    records, skipped = build_dataset(tmp_path, images_dir, validator=lambda _path: None)
    assert len(records) == 1
    assert skipped == 0


def test_build_dataset_discovers_nested_documents(tmp_path: Path) -> None:
    """build_dataset finds .dclg files nested below the corpus root."""
    nested = tmp_path / "corpus" / "document_001"
    nested.mkdir(parents=True)
    (nested / "doc.dclg").write_text("<doclang>content</doclang>", encoding="utf-8")
    _make_image(nested / "doc.png")

    records, skipped = build_dataset(tmp_path / "corpus", None, validator=lambda _path: None)
    assert skipped == 0
    assert len(records) == 1


def test_build_dataset_raises_when_no_documents_found(tmp_path: Path) -> None:
    """A corpus without any DocLang document is reported rather than silently empty."""
    (tmp_path / "notes.txt").write_text("not a doc", encoding="utf-8")
    with pytest.raises(DiscoveryError):
        build_dataset(tmp_path, None, validator=lambda _path: None)


def test_build_dataset_skips_documents_without_image(tmp_path: Path) -> None:
    """build_dataset should skip valid documents that have no matching image."""
    doclang_path = tmp_path / "doc.dclg"
    doclang_path.write_text("<doclang>content</doclang>", encoding="utf-8")
    images_dir = tmp_path / "images"
    images_dir.mkdir()  # empty: no doc.png

    records, skipped = build_dataset(tmp_path, images_dir, validator=lambda _path: None)
    assert skipped == 1
    assert records == []


def test_build_sft_record_embeds_image_by_default(tmp_path: Path) -> None:
    """Without an explicit mode the record carries a decoded image, as before."""
    doclang_path = tmp_path / "doc.dclg"
    doclang_path.write_text("<doclang>content</doclang>", encoding="utf-8")
    image_path = tmp_path / "doc.png"
    _make_image(image_path)
    record = build_sft_record(doclang_path, image_path)
    assert isinstance(record["images"][0], Image.Image)


def test_build_sft_record_reference_mode_stores_path(tmp_path: Path) -> None:
    """Reference mode stores the image path relative to the corpus root."""
    doclang_path = tmp_path / "doc.dclg"
    doclang_path.write_text("<doclang>content</doclang>", encoding="utf-8")
    image_path = tmp_path / "corpus" / "doc.png"
    image_path.parent.mkdir()
    _make_image(image_path)
    record = build_sft_record(doclang_path, image_path, "reference", [tmp_path / "corpus"])
    assert record["images"] == ["doc.png"]


def test_build_dataset_reference_mode_stores_relative_paths(tmp_path: Path) -> None:
    """build_dataset in reference mode yields string image entries for every record."""
    corpus = tmp_path / "corpus"
    (corpus / "document").mkdir(parents=True)
    (corpus / "document" / "doc.dclg").write_text("<doclang>content</doclang>", encoding="utf-8")
    _make_image(corpus / "document" / "doc.png")

    records, skipped = build_dataset(corpus, None, validator=lambda _path: None, image_mode="reference")
    assert skipped == 0
    assert [record["images"] for record in records] == [["document/doc.png"]]


def test_build_dataset_embed_mode_unchanged(tmp_path: Path) -> None:
    """The default mode still embeds decoded images, so existing datasets are unaffected."""
    doclang_path = tmp_path / "doc.dclg"
    doclang_path.write_text("<doclang>content</doclang>", encoding="utf-8")
    _make_image(tmp_path / "doc.png")

    records, skipped = build_dataset(tmp_path, None, validator=lambda _path: None)
    assert skipped == 0
    assert isinstance(records[0]["images"][0], Image.Image)
