"""Behavioural tests for the model → ``.dclg`` serializer and its inverse.

Guarantees checked:

* serializing a valid model yields XML that passes the official validator;
* the round-trip (model → XML → model → XML) is **stable** (idempotent);
* XML special characters are escaped and survive the round-trip;
* text runs with leading/trailing whitespace are wrapped in ``<content>``
  (``xml:space="preserve"``) and survive the round-trip;
* the element head is emitted in the canonical XSD order.
"""

from __future__ import annotations

from src.doclang_structured import (
    DocLang,
    doclang_xml_to_model,
    model_to_doclang_xml,
    validate_doclang,
)


def _doc(body: list) -> DocLang:
    return DocLang.model_validate({"body": body})


def _text(text: str) -> dict:
    return {"type": "text", "body": [{"type": "text_chunk", "text": text}]}


def test_serialize_produces_valid_xml() -> None:
    doc = _doc([_text("hello")])
    xml = model_to_doclang_xml(doc)
    validate_doclang(xml, xsd_only=True)  # fast XSD-only check


def test_round_trip_is_stable() -> None:
    doc = _doc([_text("hello world")])
    xml1 = model_to_doclang_xml(doc)
    doc2 = doclang_xml_to_model(xml1)
    xml2 = model_to_doclang_xml(doc2)
    assert xml1 == xml2


def test_special_characters_escape_and_round_trip() -> None:
    raw = "a < b & c > d"
    doc = _doc([_text(raw)])
    xml = model_to_doclang_xml(doc)
    assert "&lt;" in xml and "&amp;" in xml and "&gt;" in xml
    doc2 = doclang_xml_to_model(xml)
    # The special characters must survive the round-trip (lossless re-serialize).
    assert model_to_doclang_xml(doc2) == model_to_doclang_xml(doc)


def test_whitespace_run_wrapped_in_content() -> None:
    raw = "  leading and trailing  "
    doc = _doc([_text(raw)])
    xml = model_to_doclang_xml(doc)
    assert "<content" in xml
    doc2 = doclang_xml_to_model(xml)
    # The whitespace run must survive the round-trip (lossless re-serialize).
    assert model_to_doclang_xml(doc2) == model_to_doclang_xml(doc)


def test_element_head_emitted_in_canonical_order() -> None:
    doc = _doc(
        [
            {
                "type": "text",
                "head": {
                    "layer": {"type": "layer", "value": "body"},
                    "thread": {"type": "thread", "thread_id": 1},
                    "label": {"type": "label", "value": "L"},
                },
                "body": [{"type": "text_chunk", "text": "x"}],
            }
        ]
    )
    xml = model_to_doclang_xml(doc)
    # Even though the input dict listed them out of order, the output must be
    # label? thread? (xref|href)? layer? ...
    assert xml.index("<label") < xml.index("<thread") < xml.index("<layer")


def test_root_carries_namespace_and_version() -> None:
    xml = model_to_doclang_xml(_doc([_text("x")]))
    assert 'xmlns="https://www.doclang.ai/ns/v0"' in xml
    assert 'version="0.7"' in xml
