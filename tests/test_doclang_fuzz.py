"""Hypothesis fuzz tests for the structured-output pipeline.

The central *soundness* property: **anything the grammar accepts must pass the
official validator.** We exercise it by generating random ``DocLang`` models,
serializing them to ``.dclg`` (the serializer is designed to emit
grammar-conforming output), and asserting that the result is (a) accepted by the
grammar and (b) accepted by the XSD validator.

A fixed seed (``derandomize=True``) keeps failures reproducible. The fast
``xsd_only=True`` validation path is used to keep the fuzz loop quick; the
full XSD+Schematron path is covered deterministically by the corpus tests.

A second, *differential* soundness test mutates valid corpus documents (tag
swaps, bogus enum values, dropped tags, reordered heads, broken tables, and
xref+href collisions) and checks that the grammar and the XSD validator agree:
anything the grammar accepts must be XSD-valid, while grammar rejections of
XSD-valid mutants are reported as completeness gaps (without failing).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import lark
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.doclang_structured import (
    DocLang,
    model_to_doclang_xml,
    validate_doclang,
)
from src.doclang_structured.grammar import to_lark_grammar
from src.doclang_structured.validate import DocLangValidationError

# --------------------------------------------------------------------------- #
# Parser (built once)
# --------------------------------------------------------------------------- #
_PARSER: lark.Lark | None = None


def _parser() -> lark.Lark:
    global _PARSER
    if _PARSER is None:
        _PARSER = lark.Lark(to_lark_grammar(), start="root", parser="earley")
    return _PARSER


# --------------------------------------------------------------------------- #
# Strategies
# --------------------------------------------------------------------------- #
# A safe alphabet: letters, digits, spaces and a few punctuation marks that are
# legal raw text characters (never ``<``, ``>`` or ``&`` — the serializer would
# escape those, which is fine, but keeping them out keeps the corpus simple).
_SAFE_ALPHABET = st.sampled_from(list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:!?-"))
safe_text = st.text(alphabet=_SAFE_ALPHABET, min_size=0, max_size=16)


def text_chunk() -> st.SearchStrategy:
    return st.fixed_dictionaries({"type": st.just("text_chunk"), "text": safe_text})


def _textlike(type_: str) -> st.SearchStrategy:
    return st.fixed_dictionaries(
        {
            "type": st.just(type_),
            "body": st.lists(text_chunk(), min_size=0, max_size=3),
        }
    )


def list_item() -> st.SearchStrategy:
    return st.fixed_dictionaries(
        {
            "ldiv": st.fixed_dictionaries({"type": st.just("ldiv")}),
            "body": st.lists(text_chunk(), min_size=0, max_size=2),
        }
    )


def list_node() -> st.SearchStrategy:
    return st.fixed_dictionaries(
        {
            "type": st.just("list"),
            "class": st.sampled_from(["ordered", "unordered"]),
            "items": st.lists(list_item(), min_size=1, max_size=3),
        }
    )


# top_level: the elements a group / document body may directly contain. A group
# may nest further top-level elements, so this is a *recursive* strategy.
_top_level_base = st.one_of(
    _textlike("text"),
    _textlike("heading"),
    _textlike("code"),
    _textlike("formula"),
    list_node(),
)
top_level = st.recursive(
    _top_level_base,
    lambda element: st.fixed_dictionaries(
        {
            "type": st.just("group"),
            "body": st.lists(element, min_size=1, max_size=2),
        }
    ),
    max_leaves=6,
)

document = st.fixed_dictionaries({"body": st.lists(top_level, min_size=1, max_size=4)})


# --------------------------------------------------------------------------- #
# Differential mutation machinery
# --------------------------------------------------------------------------- #
# The *differential soundness* property: anything the grammar accepts must be
# XSD-valid. We mutate valid corpus documents and check that the grammar and
# the XSD validator agree on each mutant. Grammar rejections of XSD-valid
# mutants are *completeness gaps* (grammar too strict) — reported, not failing.

_CORPUS_DIR = Path(__file__).resolve().parent / "data" / "doclang_corpus"


def _load_corpus() -> list[tuple[str, str]]:
    """Load all valid corpus documents as ``(name, xml)`` pairs."""
    return [(path.name, path.read_text(encoding="utf-8").strip()) for path in sorted(_CORPUS_DIR.glob("*.dclg"))]


_CORPUS: list[tuple[str, str]] = _load_corpus()


def _mutate_tag_swap(xml: str) -> str | None:
    """Swap a whole element pair (footnote→page_header, else text→footnote)."""
    if "<footnote>" in xml:
        return xml.replace("<footnote>", "<page_header>", 1).replace("</footnote>", "</page_header>", 1)
    if "<text>" in xml:
        return xml.replace("<text>", "<footnote>", 1).replace("</text>", "</footnote>", 1)
    return None


_ENUM_PATTERNS = (
    'class="ordered"',
    'class="unordered"',
    'class="selected"',
    'class="unselected"',
    'class="read_only"',
    'class="fillable"',
    'class="undefined"',
    'class="chart"',
    'value="body"',
    'value="background"',
    'value="furniture"',
)


def _mutate_enum_bogus(xml: str) -> str | None:
    """Set the first present enum attribute to the invalid value ``bogus``."""
    for pattern in _ENUM_PATTERNS:
        if pattern in xml:
            attr = pattern.split("=", 1)[0]
            return xml.replace(pattern, f'{attr}="bogus"', 1)
    return None


_DROP_TAGS = ("<nl/>", "<ldiv/>", "<fcel/>", "<ecel/>")


def _mutate_drop_tag(xml: str) -> str | None:
    """Delete the first occurrence of the first present structural tag."""
    for tag in _DROP_TAGS:
        if tag in xml:
            return xml.replace(tag, "", 1)
    return None


def _mutate_reorder_head(xml: str) -> str | None:
    """Swap an adjacent out-of-order head pair (layer before thread)."""
    new_xml, count = re.subn(r'(<thread thread_id="\d+"/>)(<layer value="\w+"/>)', r"\2\1", xml, count=1)
    return new_xml if count else None


def _mutate_break_table(xml: str) -> str | None:
    """Remove the first ``<ecel/>`` to break table rectangularity."""
    if "<ecel/>" in xml:
        return xml.replace("<ecel/>", "", 1)
    return None


def _mutate_add_xref_href(xml: str) -> str | None:
    """Insert an ``<xref/>`` before the first ``<href`` (mutual exclusion)."""
    idx = xml.find("<href")
    if idx == -1:
        return None
    return xml[:idx] + '<xref thread_id="1"/>' + xml[idx:]


_MUTATIONS: dict[str, Callable[[str], str | None]] = {
    "tag_swap": _mutate_tag_swap,
    "enum_bogus": _mutate_enum_bogus,
    "drop_tag": _mutate_drop_tag,
    "reorder_head": _mutate_reorder_head,
    "break_table": _mutate_break_table,
    "add_xref_href": _mutate_add_xref_href,
}


def _build_mutants() -> list[tuple[str, str, str]]:
    """Precompute all applicable ``(doc_name, mutant, mutation)`` triples."""
    mutants: list[tuple[str, str, str]] = []
    for doc_name, xml in _CORPUS:
        for mut_name, mut_fn in _MUTATIONS.items():
            mutant = mut_fn(xml)
            if mutant is not None and mutant != xml:
                mutants.append((doc_name, mutant, mut_name))
    return mutants


_MUTANTS: list[tuple[str, str, str]] = _build_mutants()

# Completeness gaps (grammar rejected but XSD-valid), collected for reporting.
_COMPLETENESS_GAPS: list[tuple[str, str, str]] = []
_REPORTED_GAPS: set[tuple[str, str, str]] = set()


def _note_completeness_gap(doc_name: str, mut_name: str, xml: str) -> None:
    key = (doc_name, mut_name, xml)
    if key not in _REPORTED_GAPS:
        _REPORTED_GAPS.add(key)
        _COMPLETENESS_GAPS.append(key)
        print(f"[completeness gap] {doc_name} ({mut_name}): {xml}")


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
@settings(
    max_examples=150,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(doc_data=document)
def test_grammar_soundness(doc_data: dict) -> None:
    """model → serialize → (grammar accepts) AND (XSD-valid)."""
    model = DocLang.model_validate(doc_data)
    xml = model_to_doclang_xml(model)
    # (a) the grammar must accept the serialized output
    _parser().parse(xml)
    # (b) the official XSD validator must accept it
    validate_doclang(xml, xsd_only=True)


@settings(
    max_examples=150,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(doc_data=document)
def test_round_trip_preserves_model(doc_data: dict) -> None:
    """model → XML → model is stable (re-serializing is idempotent)."""
    from src.doclang_structured import doclang_xml_to_model

    model = DocLang.model_validate(doc_data)
    xml1 = model_to_doclang_xml(model)
    model2 = doclang_xml_to_model(xml1)
    xml2 = model_to_doclang_xml(model2)
    assert xml1 == xml2


@settings(
    max_examples=100,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(mutant=st.sampled_from(_MUTANTS))
def test_grammar_soundness_under_mutation(mutant: tuple[str, str, str]) -> None:
    """Differential soundness: grammar-accepted mutants must be XSD-valid."""
    doc_name, xml, mut_name = mutant
    try:
        _parser().parse(xml)
        grammar_accepted = True
    except lark.exceptions.LarkError:
        grammar_accepted = False
    try:
        validate_doclang(xml, xsd_only=True)
        xsd_valid = True
    except DocLangValidationError:
        xsd_valid = False
    if grammar_accepted and not xsd_valid:
        pytest.fail(
            "Grammar soundness violation: the grammar accepted a mutant that "
            "the XSD validator rejects.\n  doc: "
            f"{doc_name}\n  mutation: {mut_name}\n  mutant: {xml}"
        )
    if not grammar_accepted and xsd_valid:
        _note_completeness_gap(doc_name, mut_name, xml)


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(pytest.main([__file__, "-v"]))
