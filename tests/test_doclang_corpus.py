"""Behavioural tests for the DocLang test corpora.

Two corpora live under ``tests/data``:

* ``doclang_corpus/``  — 34 hand-authored **valid** ``.dclg`` documents plus a
  ``manifest.json`` recording which elements/features each exercises.
* ``doclang_invalid/`` — 20 hand-authored **invalid** ``.dclg`` documents plus a
  ``manifest.json`` recording the violated rule and which mechanism rejects it.

Guarantees checked:

* every valid document passes the **full** official validator (XSD + Schematron);
* every valid document round-trips through the parser
  (``.dclg`` → model → ``.dclg``) and the result is still valid;
* every invalid document is **rejected** by the full official validator;
* the valid corpus collectively **covers** every XSD element (the root
  ``doclang`` is implicit in every document).

These tests use the full validator (``xsd_only=False``), which requires a JRE,
because several invalid documents violate only Schematron (context-sensitive)
rules that XSD validation alone would not catch.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from doclang_xsd import xsd_inventory
from src.doclang_structured import (
    doclang_xml_to_model,
    model_to_doclang_xml,
    validate_doclang,
)
from src.doclang_structured.validate import DocLangValidationError

DATA = Path(__file__).parent / "data"
CORPUS_DIR = DATA / "doclang_corpus"
INVALID_DIR = DATA / "doclang_invalid"


def _corpus_files() -> list[Path]:
    return sorted(CORPUS_DIR.glob("*.dclg"))


def _invalid_files() -> list[Path]:
    return sorted(INVALID_DIR.glob("*.dclg"))


def _corpus_manifest() -> dict:
    return json.loads((CORPUS_DIR / "manifest.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Valid corpus
# --------------------------------------------------------------------------- #
def test_valid_corpus_not_empty() -> None:
    assert _corpus_files(), f"valid corpus is empty: {CORPUS_DIR}"


def test_valid_corpus_passes_full_validation() -> None:
    """Every valid document must pass XSD + Schematron validation."""
    for path in _corpus_files():
        validate_doclang(path.read_text(encoding="utf-8"))  # raises on failure


def test_valid_corpus_round_trips_through_parser() -> None:
    """Every valid document must parse to a model and re-serialize to valid XML."""
    for path in _corpus_files():
        xml = path.read_text(encoding="utf-8")
        model = doclang_xml_to_model(xml)
        reserialized = model_to_doclang_xml(model)
        validate_doclang(reserialized)  # the round-trip must stay valid


def test_corpus_covers_all_xsd_elements() -> None:
    """The corpus must collectively exercise every XSD element.

    The root element (``doclang``) is implicit in every document, so it is
    excluded; every other XSD element must appear in at least one document's
    manifest.
    """
    xsd = xsd_inventory()
    manifest = _corpus_manifest()
    covered: set[str] = set()
    for info in manifest.values():
        covered.update(info.get("elements", []))
    missing = (xsd.elements - {xsd.root_element}) - covered
    assert not missing, (
        f"valid corpus does not cover XSD elements: {sorted(missing)}. "
        "Add a document exercising each, then regenerate the manifest."
    )


# --------------------------------------------------------------------------- #
# Invalid corpus
# --------------------------------------------------------------------------- #
def test_invalid_corpus_not_empty() -> None:
    assert _invalid_files(), f"invalid corpus is empty: {INVALID_DIR}"


def test_invalid_corpus_fails_full_validation() -> None:
    """Every invalid document must be rejected by the full validator."""
    for path in _invalid_files():
        with pytest.raises(DocLangValidationError):
            validate_doclang(path.read_text(encoding="utf-8"))


def test_invalid_manifest_mechanisms_are_consistent() -> None:
    """The manifest must only use the two known mechanism labels."""
    manifest = json.loads((INVALID_DIR / "manifest.json").read_text(encoding="utf-8"))
    allowed = {"grammar+validator", "validator"}
    for fn, info in manifest.items():
        assert info["mechanism"] in allowed, (
            f"{fn} has unknown mechanism {info['mechanism']!r} (expected one of {sorted(allowed)})"
        )
    # Both categories must be represented (the CFG-expressible subset and the
    # context-sensitive subset).
    mechanisms = {info["mechanism"] for info in manifest.values()}
    assert mechanisms == allowed, (
        f"invalid manifest should cover both mechanisms, found {sorted(mechanisms)}"
    )
