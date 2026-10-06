"""Sync tests: grammar + pydantic models + JSON schema vs the bundled XSD.

These are the *sync guarantee* for the structured-output package. The XSD
bundled in the installed ``doclang`` package is the single source of truth
(the *sync anchor*); the EBNF grammar, the pydantic models, and the JSON
schema are all hand-maintained projections of it. If the ``doclang`` package
is upgraded in a way that changes the element/enum inventory or the spec
version, these tests fail **loudly** and point at exactly what to re-sync.

The inventories are extracted in :mod:`tests.doclang_xsd`; this module only
asserts that they agree.
"""

from __future__ import annotations

import pytest

from doclang_xsd import (
    find_xsd,
    grammar_inventory,
    json_schema_inventory,
    pydantic_inventory,
    xsd_inventory,
)

# Pinned to the spec version declared in the bundled XSD (``<xs:schema
# version="...">``). Bump this — after re-syncing grammar.py / models.py /
# schema.py against the new XSD — when the doclang package is upgraded.
EXPECTED_DOCLANG_SPEC_VERSION = "0.7.3"

_RESYNC = (
    "The doclang package appears to have changed. Re-sync "
    "src/doclang_structured/grammar.py, models.py, and schema.py against the "
    "bundled XSD (see docs/structured_output.md), then bump "
    "EXPECTED_DOCLANG_SPEC_VERSION."
)


# --------------------------------------------------------------------------- #
# Fixtures (module-scoped: parse the XSD / grammar / models once per module)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def xsd():
    return xsd_inventory()


@pytest.fixture(scope="module")
def grammar():
    return grammar_inventory()


@pytest.fixture(scope="module")
def pydantic():
    return pydantic_inventory()


@pytest.fixture(scope="module")
def json_schema():
    return json_schema_inventory()


# --------------------------------------------------------------------------- #
# Sanity
# --------------------------------------------------------------------------- #
def test_xsd_anchor_found() -> None:
    """The sync anchor (the XSD bundled in the doclang package) must exist."""
    path = find_xsd()
    assert path.exists(), f"bundled XSD not found at {path}"


def test_spec_version_pinned(xsd) -> None:
    """The XSD spec version must match the pinned value (loud upgrade guard)."""
    assert xsd.spec_version == EXPECTED_DOCLANG_SPEC_VERSION, (
        f"XSD spec version is {xsd.spec_version!r}, expected "
        f"{EXPECTED_DOCLANG_SPEC_VERSION!r}. {_RESYNC}"
    )


# --------------------------------------------------------------------------- #
# Element inventory
# --------------------------------------------------------------------------- #
def test_grammar_covers_xsd_elements(xsd, grammar) -> None:
    """Every XSD element must be expressible in the grammar."""
    missing = xsd.elements - grammar.elements
    assert not missing, f"grammar is missing XSD elements: {sorted(missing)}. {_RESYNC}"


def test_pydantic_covers_xsd_elements(xsd, pydantic) -> None:
    """Every XSD element must have a pydantic model."""
    missing = xsd.elements - pydantic.elements
    assert not missing, (
        f"pydantic models are missing XSD elements: {sorted(missing)}. {_RESYNC}"
    )


def test_json_schema_covers_xsd(xsd, json_schema) -> None:
    """Every XSD element must be expressible in the JSON schema (superset)."""
    missing = xsd.elements - json_schema
    assert not missing, (
        f"JSON schema cannot express XSD elements: {sorted(missing)}. {_RESYNC}"
    )


def test_no_stale_elements(xsd, grammar, pydantic) -> None:
    """The grammar and models must not contain elements absent from the XSD."""
    stale_grammar = grammar.elements - xsd.elements
    stale_pydantic = pydantic.elements - xsd.elements
    assert not stale_grammar, (
        f"grammar has elements not in the XSD: {sorted(stale_grammar)}. {_RESYNC}"
    )
    assert not stale_pydantic, (
        f"pydantic models have elements not in the XSD: {sorted(stale_pydantic)}. {_RESYNC}"
    )


def test_root_element_matches(xsd, grammar) -> None:
    """The grammar's root element must be the XSD's root element."""
    assert grammar.root_element == xsd.root_element, (
        f"grammar root {grammar.root_element!r} != XSD root {xsd.root_element!r}. "
        f"{_RESYNC}"
    )


# --------------------------------------------------------------------------- #
# Enum inventory
# --------------------------------------------------------------------------- #
def test_grammar_enum_values_match_xsd(xsd, grammar) -> None:
    _assert_enums_match("grammar", xsd.enums, grammar.enums)


def test_pydantic_enum_values_match_xsd(xsd, pydantic) -> None:
    _assert_enums_match("pydantic", xsd.enums, pydantic.enums)


def _assert_enums_match(label: str, expected: dict, actual: dict) -> None:
    missing_keys = set(expected) - set(actual)
    stale_keys = set(actual) - set(expected)
    assert not missing_keys, (
        f"{label} is missing XSD enum attributes: {sorted(missing_keys)}. {_RESYNC}"
    )
    assert not stale_keys, (
        f"{label} has enum attributes not in the XSD: {sorted(stale_keys)}. {_RESYNC}"
    )
    for key in set(expected) & set(actual):
        assert expected[key] == actual[key], (
            f"{label} enum {key} has values {sorted(actual[key])}, "
            f"XSD has {sorted(expected[key])}. {_RESYNC}"
        )
