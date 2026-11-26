"""Unit tests for corpus autodiscovery and source-image resolution."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from utils.discovery import (
    DiscoveryError,
    discover_doclang,
    discover_jsonl,
    discover_page_and_doclang,
    discover_page_xml,
    find_image,
    iter_page_dirs,
    page_image_filename,
    resolve_image,
)


def _make_image(path: Path) -> None:
    """Create a tiny test image at the given path."""
    Image.new("RGB", (8, 8), color="white").save(path)


def _make_page_xml(path: Path, image_filename: str | None = "doc.png") -> None:
    """Write a minimal PAGE XML file, optionally declaring an imageFilename.

    Args:
        path (Path): Where to write the XML file.
        image_filename (str | None): Value for the Page element's imageFilename attribute.
    """
    declared = f' imageFilename="{image_filename}"' if image_filename else ""
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">\n'
        f'  <Page{declared} imageWidth="100" imageHeight="100">\n'
        '    <TextRegion id="r1">\n'
        '      <Coords points="0,0 100,0 100,100 0,100"/>\n'
        '      <TextLine id="l1">\n'
        '        <Coords points="0,0 100,0 100,25 0,25"/>\n'
        "        <TextEquiv><Unicode>hello</Unicode></TextEquiv>\n"
        "      </TextLine>\n"
        "    </TextRegion>\n"
        "  </Page>\n"
        "</PcGts>\n"
    )
    path.write_text(xml, encoding="utf-8")


def _make_corpus(tmp_path: Path, n_docs: int = 2) -> Path:
    """Build a nested ``corpus/document/page/*.xml`` corpus with images one level up.

    Args:
        tmp_path (Path): Base directory.
        n_docs (int): Number of documents to create.

    Returns:
        Path: The corpus root.
    """
    corpus = tmp_path / "corpus"
    for doc in range(n_docs):
        doc_dir = corpus / f"document_{doc:03d}"
        page_dir = doc_dir / "page"
        page_dir.mkdir(parents=True)
        stem = f"document_{doc:03d}_0001"
        _make_page_xml(page_dir / f"{stem}.xml", image_filename=None)
        _make_image(doc_dir / f"{stem}.png")
    return corpus


# ---------------------------------------------------------------------------
# discover_page_xml
# ---------------------------------------------------------------------------


def test_discover_page_xml_finds_nested_page_dirs(tmp_path: Path) -> None:
    """PAGE XML inside document/page/ directories is discovered at any depth."""
    corpus = _make_corpus(tmp_path, n_docs=2)
    found = discover_page_xml(corpus)
    assert [path.stem for path in found] == ["document_000_0001", "document_001_0001"]
    assert all(path.parent.name == "page" for path in found)


def test_discover_page_xml_ignores_xml_outside_page_dirs(tmp_path: Path) -> None:
    """.xml files outside a 'page' directory are never ingested."""
    corpus = tmp_path / "corpus"
    (corpus / "document").mkdir(parents=True)
    _make_page_xml(corpus / "document" / "stray.xml")
    with pytest.raises(DiscoveryError):
        discover_page_xml(corpus)


def test_discover_page_xml_matches_dir_name_case_insensitively(tmp_path: Path) -> None:
    """The 'page' directory name is matched regardless of case."""
    page_dir = tmp_path / "doc" / "PAGE"
    page_dir.mkdir(parents=True)
    _make_page_xml(page_dir / "doc.xml")
    assert [path.name for path in discover_page_xml(tmp_path)] == ["doc.xml"]


def test_discover_page_xml_accepts_root_named_page(tmp_path: Path) -> None:
    """A root that is itself a 'page' directory is used directly."""
    page_dir = tmp_path / "page"
    page_dir.mkdir()
    _make_page_xml(page_dir / "doc.xml")
    assert [path.name for path in discover_page_xml(page_dir)] == ["doc.xml"]


def test_discover_page_xml_merges_multiple_roots(tmp_path: Path) -> None:
    """Several corpus roots are combined into one sorted file list."""
    first = _make_corpus(tmp_path / "a", n_docs=1)
    second = _make_corpus(tmp_path / "b", n_docs=1)
    found = discover_page_xml([first, second])
    assert len(found) == 2


def test_discover_page_xml_raises_when_nothing_found(tmp_path: Path) -> None:
    """An empty corpus raises a DiscoveryError naming the page/ convention."""
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(DiscoveryError, match="named 'page'"):
        discover_page_xml(empty)


def test_discover_page_xml_rejects_missing_root(tmp_path: Path) -> None:
    """A root that is not a directory raises."""
    with pytest.raises(NotADirectoryError):
        discover_page_xml(tmp_path / "nope")


def test_iter_page_dirs_reports_each_page_dir_once(tmp_path: Path) -> None:
    """A page directory nested inside another page dir is not reported twice."""
    outer = tmp_path / "doc" / "page"
    (outer / "page").mkdir(parents=True)
    assert list(iter_page_dirs(tmp_path)) == [outer]


# ---------------------------------------------------------------------------
# discover_doclang
# ---------------------------------------------------------------------------


def test_discover_doclang_recurses_without_dir_convention(tmp_path: Path) -> None:
    """.dclg files are found at any depth, whatever the directory is called."""
    deep = tmp_path / "anything" / "at" / "all"
    deep.mkdir(parents=True)
    (deep / "doc.dclg").write_text("<doclang/>", encoding="utf-8")
    assert [path.name for path in discover_doclang(tmp_path)] == ["doc.dclg"]


def test_discover_doclang_raises_when_nothing_found(tmp_path: Path) -> None:
    """An empty corpus raises DiscoveryError."""
    (tmp_path / "empty").mkdir()
    with pytest.raises(DiscoveryError):
        discover_doclang(tmp_path)


# ---------------------------------------------------------------------------
# discover_page_and_doclang
# ---------------------------------------------------------------------------


def test_discover_page_and_doclang_merges_both_kinds(tmp_path: Path) -> None:
    """Page-dir XML and nested DocLang are returned together."""
    page_dir = tmp_path / "document" / "page"
    page_dir.mkdir(parents=True)
    (page_dir / "a.xml").write_text("<PcGts/>", encoding="utf-8")
    (tmp_path / "b.dclg").write_text("<doclang/>", encoding="utf-8")
    assert [path.name for path in discover_page_and_doclang(tmp_path)] == ["b.dclg", "a.xml"]


def test_discover_page_and_doclang_accepts_doclang_only_corpus(tmp_path: Path) -> None:
    """A DocLang-only corpus without page directories is accepted."""
    (tmp_path / "doc.dclg").write_text("<doclang/>", encoding="utf-8")
    assert [path.name for path in discover_page_and_doclang(tmp_path)] == ["doc.dclg"]


def test_discover_page_and_doclang_accepts_page_xml_only_corpus(tmp_path: Path) -> None:
    """A PAGE-XML-only corpus is accepted."""
    page_dir = tmp_path / "page"
    page_dir.mkdir()
    (page_dir / "doc.xml").write_text("<PcGts/>", encoding="utf-8")
    assert [path.name for path in discover_page_and_doclang(tmp_path)] == ["doc.xml"]


def test_discover_page_and_doclang_ignores_stray_xml(tmp_path: Path) -> None:
    """XML outside a page directory does not satisfy the requirement."""
    (tmp_path / "stray.xml").write_text("<PcGts/>", encoding="utf-8")
    with pytest.raises(DiscoveryError):
        discover_page_and_doclang(tmp_path)


def test_discover_page_and_doclang_raises_when_nothing_found(tmp_path: Path) -> None:
    """An empty corpus raises DiscoveryError."""
    (tmp_path / "empty").mkdir()
    with pytest.raises(DiscoveryError):
        discover_page_and_doclang(tmp_path)


# ---------------------------------------------------------------------------
# discover_jsonl
# ---------------------------------------------------------------------------


def test_discover_jsonl_expands_directories(tmp_path: Path) -> None:
    """A directory contributes its top-level .jsonl files, sorted."""
    (tmp_path / "b.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "a.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")
    assert [path.name for path in discover_jsonl(tmp_path)] == ["a.jsonl", "b.jsonl"]


def test_discover_jsonl_passes_single_files_through(tmp_path: Path) -> None:
    """A single file path behaves exactly as before."""
    single = tmp_path / "train.jsonl"
    single.write_text("{}\n", encoding="utf-8")
    assert discover_jsonl(single) == [single]


def test_discover_jsonl_raises_for_missing_path(tmp_path: Path) -> None:
    """A path that is neither file nor directory raises."""
    with pytest.raises(FileNotFoundError):
        discover_jsonl(tmp_path / "missing.jsonl")


def test_discover_jsonl_raises_when_directory_is_empty(tmp_path: Path) -> None:
    """A directory with no .jsonl files raises DiscoveryError."""
    with pytest.raises(DiscoveryError):
        discover_jsonl(tmp_path)


# ---------------------------------------------------------------------------
# find_image
# ---------------------------------------------------------------------------


def test_find_image_matches_non_png_format(tmp_path: Path) -> None:
    """find_image should locate a source image in any PIL-supported format."""
    _make_image(tmp_path / "doc.jpg")
    assert find_image(tmp_path, "doc") == tmp_path / "doc.jpg"


def test_find_image_returns_none_when_missing(tmp_path: Path) -> None:
    """find_image should return None when no supported image exists."""
    assert find_image(tmp_path, "missing") is None


def test_find_image_ignores_directories(tmp_path: Path) -> None:
    """A directory named like an image is not returned."""
    (tmp_path / "doc.png").mkdir()
    assert find_image(tmp_path, "doc") is None


# ---------------------------------------------------------------------------
# page_image_filename
# ---------------------------------------------------------------------------


def test_page_image_filename_reads_attribute(tmp_path: Path) -> None:
    """The declared imageFilename is returned."""
    path = tmp_path / "doc.xml"
    _make_page_xml(path, image_filename="scan.tif")
    assert page_image_filename(path) == "scan.tif"


def test_page_image_filename_returns_none_when_absent(tmp_path: Path) -> None:
    """A PAGE XML without imageFilename yields None."""
    path = tmp_path / "doc.xml"
    _make_page_xml(path, image_filename=None)
    assert page_image_filename(path) is None


def test_page_image_filename_tolerates_malformed_xml(tmp_path: Path) -> None:
    """Unparseable XML yields None rather than raising."""
    (tmp_path / "bad.xml").write_text("<PcGts><Page>", encoding="utf-8")
    assert page_image_filename(tmp_path / "bad.xml") is None


# ---------------------------------------------------------------------------
# resolve_image
# ---------------------------------------------------------------------------


def test_resolve_image_prefers_explicit_images_root(tmp_path: Path) -> None:
    """An explicit image root wins over the corpus layout."""
    corpus = _make_corpus(tmp_path, n_docs=1)
    doc = discover_page_xml(corpus)[0]
    images = tmp_path / "images"
    images.mkdir()
    _make_image(images / f"{doc.stem}.jpg")
    assert resolve_image(doc, images) == images / f"{doc.stem}.jpg"


def test_resolve_image_falls_back_to_sibling_of_page_dir(tmp_path: Path) -> None:
    """Without --images, the image one level above the page dir is used."""
    corpus = _make_corpus(tmp_path, n_docs=1)
    doc = discover_page_xml(corpus)[0]
    resolved = resolve_image(doc, None)
    assert resolved == doc.parent.parent / f"{doc.stem}.png"


def test_resolve_image_uses_image_filename_attribute(tmp_path: Path) -> None:
    """The PAGE XML imageFilename attribute links a differently named image."""
    page_dir = tmp_path / "doc" / "page"
    page_dir.mkdir(parents=True)
    doc = page_dir / "page0001.xml"
    _make_page_xml(doc, image_filename="scan-0001.tif")
    _make_image(doc.parent.parent / "scan-0001.tif")
    assert resolve_image(doc, None) == doc.parent.parent / "scan-0001.tif"


def test_resolve_image_resolves_image_filename_relative_path(tmp_path: Path) -> None:
    """A relative imageFilename is resolved against the candidate directories."""
    page_dir = tmp_path / "doc" / "page"
    page_dir.mkdir(parents=True)
    doc = page_dir / "page0001.xml"
    _make_page_xml(doc, image_filename="images/scan.png")
    target = doc.parent.parent / "images"
    target.mkdir()
    _make_image(target / "scan.png")
    assert resolve_image(doc, None) == target / "scan.png"


def test_resolve_image_finds_image_next_to_doclang_file(tmp_path: Path) -> None:
    """A .dclg document finds its image in its own directory."""
    doc = tmp_path / "doc.dclg"
    doc.write_text("<doclang/>", encoding="utf-8")
    _make_image(tmp_path / "doc.png")
    assert resolve_image(doc, None) == tmp_path / "doc.png"


def test_resolve_image_returns_none_when_no_strategy_matches(tmp_path: Path) -> None:
    """An orphan document resolves to None."""
    page_dir = tmp_path / "doc" / "page"
    page_dir.mkdir(parents=True)
    doc = page_dir / "orphan.xml"
    _make_page_xml(doc, image_filename=None)
    assert resolve_image(doc, None) is None


def test_resolve_image_accepts_multiple_image_roots(tmp_path: Path) -> None:
    """The first image root containing the stem wins."""
    page_dir = tmp_path / "doc" / "page"
    page_dir.mkdir(parents=True)
    doc = page_dir / "doc.xml"
    _make_page_xml(doc, image_filename=None)
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    _make_image(second / "doc.png")
    assert resolve_image(doc, [first, second]) == second / "doc.png"
