"""Inventory extraction for the DocLang XSD, grammar, and pydantic models.

This module provides the inventories that the sync tests
(``tests/test_doclang_sync.py``) compare against one another to guarantee that
the EBNF grammar, the pydantic models, and the JSON schema stay in lock-step
with the XSD bundled in the installed ``doclang`` package (the *sync anchor*).

Inventories
-----------
* :func:`xsd_inventory`         — parsed from the bundled ``doclang.xsd``.
* :func:`grammar_inventory`     — parsed from the GBNF grammar in
  ``src.doclang_structured.grammar``.
* :func:`pydantic_inventory`    — parsed from the models in
  ``src.doclang_structured.models``.
* :func:`json_schema_inventory` — the element names reachable in the JSON
  schema produced by ``src.doclang_structured.schema``.

Each element/enum inventory exposes:

* ``elements`` — the set of DocLang element names.
* ``enums``    — a mapping of ``(element, attribute) -> frozenset(values)`` for
  every enumerated attribute.

The XSD inventory additionally exposes ``root_element`` and ``spec_version``.

The sync anchor is discovered at runtime (``find_xsd``), so the suite tracks
``doclang`` package upgrades automatically; a pinned
``EXPECTED_DOCLANG_SPEC_VERSION`` in the sync tests fails loudly on upgrade.
"""

from __future__ import annotations

import re
import typing
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import FrozenSet

import doclang

__all__ = [
    "XsdInventory",
    "GrammarInventory",
    "PydanticInventory",
    "find_xsd",
    "xsd_inventory",
    "grammar_inventory",
    "pydantic_inventory",
    "json_schema_inventory",
]

_XS = "{http://www.w3.org/2001/XMLSchema}"


# --------------------------------------------------------------------------- #
# Inventory containers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class XsdInventory:
    """Element/enum inventory parsed from the bundled XSD."""

    elements: FrozenSet[str]
    enums: dict[tuple[str, str], FrozenSet[str]]
    root_element: str
    spec_version: str


@dataclass(frozen=True)
class GrammarInventory:
    """Element/enum inventory parsed from the GBNF grammar."""

    elements: FrozenSet[str]
    enums: dict[tuple[str, str], FrozenSet[str]]
    root_element: str


@dataclass(frozen=True)
class PydanticInventory:
    """Element/enum inventory parsed from the pydantic models."""

    elements: FrozenSet[str]
    enums: dict[tuple[str, str], FrozenSet[str]]


# --------------------------------------------------------------------------- #
# XSD
# --------------------------------------------------------------------------- #
def find_xsd() -> Path:
    """Locate the ``doclang.xsd`` bundled in the installed ``doclang`` package."""
    xsd = Path(doclang.__file__).parent / "doclang.xsd"
    if not xsd.exists():
        raise FileNotFoundError(f"doclang.xsd not found at {xsd!s}")
    return xsd


def _named_types(root: ET.Element) -> dict[str, ET.Element]:
    return {
        ct.get("name"): ct
        for ct in list(root.iter(f"{_XS}complexType")) + list(root.iter(f"{_XS}simpleType"))
        if ct.get("name")
    }


def _type_node(el: ET.Element, named: dict[str, ET.Element]) -> ET.Element | None:
    """Return the complexType/simpleType node governing ``el`` (inline or named)."""
    t = el.get("type")
    if t:
        return named.get(t.split(":")[-1])
    for c in el:
        if c.tag in (f"{_XS}complexType", f"{_XS}simpleType"):
            return c
    return None


def _find_root_element(
    root: ET.Element, defined: dict[str, ET.Element], elements: FrozenSet[str]
) -> str:
    """Best-effort determination of the document root element.

    The root is a *defined* element that is neither referenced by any
    ``xs:element ref`` nor defined inside an ``xs:group``. When several
    candidates remain (the XSD's ``<head>`` uses ``xs:any``, so its children
    are not explicitly ref'd), the true root is the candidate with the most
    outgoing element/group references — it is the element that *contains* the
    rest of the document.
    """
    refd: set[str] = set()
    for el in root.iter(f"{_XS}element"):
        r = el.get("ref")
        if r:
            refd.add(r.split(":")[-1])

    in_group: set[str] = set()
    groups: dict[str, ET.Element] = {}
    for g in root.iter(f"{_XS}group"):
        gn = g.get("name")
        if gn:
            groups[gn] = g
        for e in g.iter(f"{_XS}element"):
            n = e.get("name")
            if n:
                in_group.add(n)

    candidates = elements - refd - in_group
    if len(candidates) == 1:
        return next(iter(candidates))

    def outgoing(name: str) -> int:
        count = 0
        for c in defined[name].iter():
            if c.tag == f"{_XS}element" and c.get("ref"):
                count += 1
            elif c.tag == f"{_XS}group" and c.get("ref"):
                g = groups.get(c.get("ref").split(":")[-1])
                if g is not None:
                    count += sum(1 for e in g.iter(f"{_XS}element") if e.get("name") or e.get("ref"))
        return count

    return max(candidates, key=lambda n: outgoing(n))


def xsd_inventory(xsd_path: Path | None = None) -> XsdInventory:
    """Parse the bundled XSD into an :class:`XsdInventory`."""
    path = xsd_path or find_xsd()
    root = ET.parse(path).getroot()
    spec_version = root.get("version", "")
    named = _named_types(root)

    defined: dict[str, ET.Element] = {}
    for el in root.iter(f"{_XS}element"):
        n = el.get("name")
        if n is not None:
            defined[n] = el
    elements = frozenset(defined)

    enums: dict[tuple[str, str], set[str]] = {}
    for name, el in defined.items():
        tn = _type_node(el, named)
        if tn is None:
            continue
        for attr in tn.iter(f"{_XS}attribute"):
            an = attr.get("name")
            if an is None:
                continue
            vals = [e.get("value") for e in attr.iter(f"{_XS}enumeration") if e.get("value") is not None]
            if vals:
                enums.setdefault((name, an), set()).update(vals)

    return XsdInventory(
        elements=elements,
        enums={k: frozenset(v) for k, v in enums.items()},
        root_element=_find_root_element(root, defined, elements),
        spec_version=spec_version,
    )


# --------------------------------------------------------------------------- #
# Grammar (GBNF)
# --------------------------------------------------------------------------- #
def grammar_inventory() -> GrammarInventory:
    """Parse the GBNF grammar into a :class:`GrammarInventory`.

    Element rules are those of the form ``name_el ::= "<tag" ...`` (a rule that
    begins with an opening-tag literal); category rules (``top_level_el``,
    ``table_sep_el``, ...) begin with a rule reference and are excluded.

    Enum rules are ``name_enum ::= "a" | "b" | ...``; the ``(element, attr)``
    key is recovered from the attribute rule that references the enum (e.g.
    ``layer_value ::= " value=\\"" layer_enum \\""`` → ``(layer, value)``) and
    the element rule that references that attribute rule.
    """
    from src.doclang_structured.grammar import DOCLANG_GRAMMAR

    lines = [ln.strip() for ln in DOCLANG_GRAMMAR.splitlines()]

    elements: set[str] = set()
    root_element = ""
    for line in lines:
        m = re.match(r'(\w+)_el\s*::=\s*"<([a-z_]+)', line)
        if m:
            elements.add(m.group(2))
        r = re.match(r'root\s*::=\s*(\w+)_el', line)
        if r:
            root_element = r.group(1)

    # 1. enum rule -> values
    enum_rules: dict[str, tuple[str, ...]] = {}
    for line in lines:
        m = re.match(r'(\w+)_enum\s*::=\s*(.+)$', line)
        if m:
            enum_rules[m.group(1)] = tuple(re.findall(r'"([^"]*)"', m.group(2)))

    # 2. attribute rule -> (attr_name, enum_rule)
    #    The GBNF attribute literal is ``" <attr>=\\""`` (an escaped quote
    #    followed by the closing quote), hence the ``=\\"\s*"`` tail.
    attr_rules: dict[str, tuple[str, str]] = {}
    for line in lines:
        m = re.match(r'(\w+)\s*::=\s*"\s*(\w+)=\\"\s*"\s*(\w+)_enum', line)
        if m and m.group(3) in enum_rules:
            attr_rules[m.group(1)] = (m.group(2), m.group(3))

    # 3. element rule referencing an attribute rule -> (element, attr) -> values
    enums: dict[tuple[str, str], FrozenSet[str]] = {}
    for line in lines:
        m = re.match(r'(\w+)_el\s*::=\s*"<([a-z_]+)(.*)$', line)
        if not m:
            continue
        tag, rest = m.group(2), m.group(3)
        for attr_rule, (attr_name, enum_rule) in attr_rules.items():
            if re.search(r"\b" + re.escape(attr_rule) + r"\b", rest):
                enums[(tag, attr_name)] = frozenset(enum_rules[enum_rule])

    return GrammarInventory(
        elements=frozenset(elements),
        enums=enums,
        root_element=root_element,
    )


# --------------------------------------------------------------------------- #
# Pydantic models
# --------------------------------------------------------------------------- #
def _literal_values(annotation: object) -> set[str] | None:
    """Return the set of ``Literal`` values in ``annotation``, or ``None``.

    Handles bare ``Literal[...]`` and ``Optional[Literal[...]]`` (i.e.
    ``Union[Literal[...], None]``).
    """
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin is typing.Literal:
        return set(args)
    if origin is typing.Union:
        for a in args:
            if a is type(None):  # noqa: E721
                continue
            vals = _literal_values(a)
            if vals is not None:
                return vals
    return None


def pydantic_inventory() -> PydanticInventory:
    """Parse the pydantic models into a :class:`PydanticInventory`.

    The element inventory is the keys of ``ELEMENT_MODELS``. The enum inventory
    is recovered from every non-``type`` field whose annotation is a
    (possibly optional) ``Literal``; the attribute name is the field's alias
    (so ``class_`` → ``class``) or its Python name.
    """
    from src.doclang_structured.models import ELEMENT_MODELS

    elements = frozenset(ELEMENT_MODELS)
    enums: dict[tuple[str, str], set[str]] = {}
    for element_name, model in ELEMENT_MODELS.items():
        for field_name, field in model.model_fields.items():
            if field_name == "type":
                continue  # discriminator, not an attribute enum
            attr_name = field.alias or field_name
            values = _literal_values(field.annotation)
            if values is not None:
                enums.setdefault((element_name, attr_name), set()).update(values)

    return PydanticInventory(
        elements=elements,
        enums={k: frozenset(v) for k, v in enums.items()},
    )


# --------------------------------------------------------------------------- #
# JSON schema
# --------------------------------------------------------------------------- #
def json_schema_inventory() -> FrozenSet[str]:
    """Return the set of DocLang element names the JSON schema can express.

    This is the set of all ``type`` discriminator values found in ``$defs``
    (each element model carries a ``type`` ``const`` for a single element or an
    ``enum`` for a shared model, e.g. the formatting / table-separator
    elements), plus the two structural elements that have **no** ``type``
    discriminator: the document root (``doclang``, the top-level schema) and
    the head (``head``, a ``$def`` with no ``type``).

    Note: the set may also include non-element body-node types (e.g.
    ``text_chunk``). The sync test uses this as a *superset* check
    (XSD elements ⊆ this set), so such extra entries are harmless.
    """
    from src.doclang_structured.schema import get_doclang_json_schema

    schema = get_doclang_json_schema()
    elements: set[str] = set()
    for def_schema in schema.get("$defs", {}).values():
        type_prop = def_schema.get("properties", {}).get("type")
        if not isinstance(type_prop, dict):
            continue
        if "const" in type_prop:
            elements.add(type_prop["const"])
        elif "enum" in type_prop:
            elements.update(type_prop["enum"])
    # Structural elements without a ``type`` discriminator.
    elements.add("doclang")  # the document root (top-level schema)
    elements.add("head")     # the head (a $def with no ``type``)
    return frozenset(elements)
