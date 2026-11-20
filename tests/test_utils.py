"""Unit tests for shared utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config import RunConfig
from utils.logging_utils import config_hash, git_commit
from utils.page_xml_editor import PageXMLEditor, Region


def test_config_hash_is_stable() -> None:
    """The config hash should be deterministic for the same config."""
    config_a = RunConfig()
    config_b = RunConfig()
    assert config_hash(config_a) == config_hash(config_b)
    assert len(config_hash(config_a)) == 8


def test_git_commit_returns_string() -> None:
    """git_commit should return a non-empty string."""
    assert isinstance(git_commit(), str)


def _make_page_xml_with_region(path: Path, custom: str | None) -> None:
    """Write a minimal PAGE XML file with a single TextRegion.

    Args:
        path (Path): Where to write the XML file.
        custom (str | None): The value of the region's ``custom`` attribute, or None to omit it.
    """
    custom_attr = f' custom="{custom}"' if custom is not None else ""
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">\n'
        "  <Metadata>\n"
        "    <Creator>test</Creator>\n"
        "    <Created>2024-01-01T00:00:00</Created>\n"
        "    <LastChange>2024-01-01T00:00:00</LastChange>\n"
        "  </Metadata>\n"
        '  <Page imageFilename="doc.png" imageWidth="100" imageHeight="100">\n'
        f'    <TextRegion id="r1"{custom_attr}>\n'
        '      <Coords points="0,0 100,0 100,100 0,100"/>\n'
        "    </TextRegion>\n"
        "  </Page>\n"
        "</PcGts>\n"
    )
    path.write_text(xml, encoding="utf-8")


def test_get_region_type_single_entry(tmp_path: Path) -> None:
    """A structure {type:X;} custom value yields X."""
    path = tmp_path / "doc.xml"
    _make_page_xml_with_region(path, "structure {type:paragraph;}")
    editor = PageXMLEditor(path)
    region = next(editor.iterfind(".//TextRegion"))
    assert editor.get_region_type(region) == "paragraph"


def test_get_region_type_multi_entry_custom(tmp_path: Path) -> None:
    """A custom attribute with several entries yields the bare structure type."""
    path = tmp_path / "doc.xml"
    _make_page_xml_with_region(path, "structure {type:paragraph;} readingOrder {index:2;}")
    editor = PageXMLEditor(path)
    region = next(editor.iterfind(".//TextRegion"))
    assert editor.get_region_type(region) == "paragraph"


def test_get_region_type_without_custom(tmp_path: Path) -> None:
    """A region without a custom attribute yields None."""
    path = tmp_path / "doc.xml"
    _make_page_xml_with_region(path, None)
    editor = PageXMLEditor(path)
    region = next(editor.iterfind(".//TextRegion"))
    assert editor.get_region_type(region) is None


def test_region_region_type_multi_entry() -> None:
    """The Region.region_type property yields the bare type for a multi-entry custom."""
    points = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.int32)
    region = Region(points)
    region.attrib["custom"] = "structure {type:cell;} readingOrder {index:3;}"
    assert region.region_type == "cell"
