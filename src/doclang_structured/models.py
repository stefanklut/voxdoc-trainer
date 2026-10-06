"""Pydantic v2 models mirroring the DocLang grammar (fallback path).

These models mirror :mod:`src.doclang_structured.grammar` exactly — the same
element set, the same body compositions, and the same enum domains as the
XSD bundled in the ``doclang`` package (v0.7.3).

Design
------
Every *node* model carries a ``type: Literal[...]`` field. The body fields are
recursive lists of **discriminated unions** (``Annotated[Union[...],
Field(discriminator="type")]``). This makes the generated JSON schema an
unambiguous set of discriminated unions, which is exactly what structured
decoders (vLLM ``response_format={"type": "json_schema"}``) need to constrain
generation.

The models are the *fallback* representation: the primary path emits raw
``.dclg`` XML guided by the EBNF grammar. The fallback path has the model emit
JSON conforming to :func:`src.doclang_structured.schema.get_doclang_json_schema`,
which is parsed into these models and serialized back to ``.dclg`` by
:mod:`src.doclang_structured.serialize`.

``ELEMENT_MODELS`` maps every DocLang element name to its model class; the sync
tests (``tests/test_doclang_sync.py``) use it to verify the models cover the
XSD element inventory.

Notes
-----
* ``class`` is a Python keyword, so the four elements with a ``class``
  attribute (``checkbox``, ``list``, ``picture``, ``value``) expose it as
  ``class_`` with a ``class`` alias (``populate_by_name=True``): the JSON
  schema and ``model_validate`` use ``class`` (matching the XML attribute)
  while Python code may use either ``class_`` or ``class``. The serializer
  maps ``class_`` → the ``class`` XML attribute.
* ``head`` / ``meta`` / ``custom`` are ``xs:any`` in the XSD (arbitrary
  content). The models restrict them to the known simple items (``head`` →
  ``default_resolution`` | ``meta``; ``meta`` / ``custom`` → empty) — a sound
  subset, matching the grammar.
* ``Cell`` and ``ListItem`` are *structural helpers* (a table cell / list item
  is a group, not a named XSD element) and are therefore **not** in
  ``ELEMENT_MODELS``.
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    # root / head
    "DocLang", "Head", "ElementHead", "DefaultResolution", "Meta", "PageBreak",
    # element-head property elements
    "Label", "Thread", "Xref", "Href", "Layer", "Location",
    "Description", "Summary", "Custom",
    # payload
    "Src", "Checkbox", "Content",
    # structural
    "TableSep", "Ldiv", "Tabular", "Cell", "ListItem",
    # formatting
    "Formatting",
    # marker / hint
    "Marker", "Hint",
    # semantic elements
    "Text", "Heading", "Code", "Formula", "Caption", "PageHeader", "PageFooter",
    "Footnote", "ListEl", "Group", "FieldRegion", "FieldHeading", "FieldItem",
    "Key", "Value", "Picture", "Table", "Index",
    # registry
    "ELEMENT_MODELS",
    # body-node type aliases
    "FmtBodyNode", "MarkerBodyNode", "DescBodyNode", "TextBodyNode",
    "CodeBodyNode", "FieldBodyNode", "GroupBodyNode", "DocLangBodyNode",
    "HeadBodyNode", "TopLevelNode", "HeadlessTxtNode", "HeadlessNode",
]

# A shared config: reject unknown fields → ``additionalProperties: False`` in
# the JSON schema (strict / decoder-friendly).
_FORBID = ConfigDict(extra="forbid", populate_by_name=True)


# --------------------------------------------------------------------------- #
# Text / content primitives
# --------------------------------------------------------------------------- #

class TextChunk(BaseModel):
    """A run of raw text (may include whitespace)."""

    model_config = _FORBID
    type: Literal["text_chunk"]
    text: str


class Content(BaseModel):
    """``<content>`` — explicit whitespace-preserving text (optional)."""

    model_config = _FORBID
    type: Literal["content"]
    text: Optional[str] = None


# --------------------------------------------------------------------------- #
# Element-head property elements
# --------------------------------------------------------------------------- #

class Label(BaseModel):
    model_config = _FORBID
    type: Literal["label"]
    value: Optional[str] = None


class Thread(BaseModel):
    model_config = _FORBID
    type: Literal["thread"]
    thread_id: int = Field(ge=1)


class Xref(BaseModel):
    model_config = _FORBID
    type: Literal["xref"]
    thread_id: int = Field(ge=1)


class Href(BaseModel):
    model_config = _FORBID
    type: Literal["href"]
    uri: str


class Layer(BaseModel):
    model_config = _FORBID
    type: Literal["layer"]
    value: Optional[Literal["body", "background", "furniture"]] = None


class Location(BaseModel):
    model_config = _FORBID
    type: Literal["location"]
    value: int = Field(ge=0)
    resolution: Optional[int] = Field(default=None, ge=1)


class Description(BaseModel):
    model_config = _FORBID
    type: Literal["description"]
    body: list[DescBodyNode] = Field(default_factory=list)


class Summary(BaseModel):
    model_config = _FORBID
    type: Literal["summary"]
    body: list[DescBodyNode] = Field(default_factory=list)


class Custom(BaseModel):
    model_config = _FORBID
    type: Literal["custom"]


# --------------------------------------------------------------------------- #
# Element head (ordered, all optional)
# Defined early because many node models use it as an eager
# ``default_factory``. Its own field references are string annotations
# (resolved at rebuild) or ``None``/``list`` defaults, so it has no eager
# forward references.
# --------------------------------------------------------------------------- #

class ElementHead(BaseModel):
    """The ordered, all-optional element head.

    Order: ``label? thread? (xref|href)? layer? location_block? caption?
    description? summary? custom?``. ``xref`` and ``href`` are mutually
    exclusive; ``locations`` must be 0 or exactly 4.
    """

    model_config = _FORBID
    label: Optional[Label] = None
    thread: Optional[Thread] = None
    xref: Optional[Xref] = None
    href: Optional[Href] = None
    layer: Optional[Layer] = None
    locations: list[Location] = Field(default_factory=list)
    caption: Optional[Caption] = None
    description: Optional[Description] = None
    summary: Optional[Summary] = None
    custom: Optional[Custom] = None

    @model_validator(mode="after")
    def _check_invariants(self) -> ElementHead:
        if self.xref is not None and self.href is not None:
            raise ValueError("xref and href are mutually exclusive")
        if len(self.locations) not in (0, 4):
            raise ValueError(f"locations must be 0 or 4, got {len(self.locations)}")
        return self


# --------------------------------------------------------------------------- #
# Payload elements
# --------------------------------------------------------------------------- #

class Src(BaseModel):
    model_config = _FORBID
    type: Literal["src"]
    uri: str


class Checkbox(BaseModel):
    model_config = _FORBID
    type: Literal["checkbox"]
    class_: Optional[Literal["unselected", "selected"]] = Field(default=None, alias="class")


# --------------------------------------------------------------------------- #
# Structural elements
# --------------------------------------------------------------------------- #

class TableSep(BaseModel):
    """One of the ten (empty) table-separator tokens."""

    model_config = _FORBID
    type: Literal[
        "fcel", "ecel", "ched", "rhed", "corn",
        "srow", "lcel", "ucel", "xcel", "nl",
    ]


class Ldiv(BaseModel):
    model_config = _FORBID
    type: Literal["ldiv"]
    marker: Optional[Marker] = None


class Tabular(BaseModel):
    model_config = _FORBID
    type: Literal["tabular"]
    cells: list[Cell] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Inline formatting (nestable) — one model, eight ``type`` values
# --------------------------------------------------------------------------- #

class Formatting(BaseModel):
    model_config = _FORBID
    type: Literal[
        "bold", "italic", "strikethrough", "underline",
        "superscript", "subscript", "rtl", "handwriting",
    ]
    body: list[FmtBodyNode] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# marker / hint
# --------------------------------------------------------------------------- #

class Marker(BaseModel):
    model_config = _FORBID
    type: Literal["marker"]
    head: ElementHead = Field(default_factory=ElementHead)
    body: list[MarkerBodyNode] = Field(default_factory=list)


class Hint(BaseModel):
    model_config = _FORBID
    type: Literal["hint"]
    body: list[FmtBodyNode] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Semantic elements (element head + body)
# --------------------------------------------------------------------------- #

class Text(BaseModel):
    model_config = _FORBID
    type: Literal["text"]
    head: ElementHead = Field(default_factory=ElementHead)
    body: list[TextBodyNode] = Field(default_factory=list)


class Heading(BaseModel):
    model_config = _FORBID
    type: Literal["heading"]
    level: Optional[int] = Field(default=None, ge=1)
    head: ElementHead = Field(default_factory=ElementHead)
    body: list[TextBodyNode] = Field(default_factory=list)


class Code(BaseModel):
    model_config = _FORBID
    type: Literal["code"]
    head: ElementHead = Field(default_factory=ElementHead)
    body: list[CodeBodyNode] = Field(default_factory=list)


class Formula(BaseModel):
    model_config = _FORBID
    type: Literal["formula"]
    head: ElementHead = Field(default_factory=ElementHead)
    body: list[CodeBodyNode] = Field(default_factory=list)


class Caption(BaseModel):
    model_config = _FORBID
    type: Literal["caption"]
    head: ElementHead = Field(default_factory=ElementHead)
    body: list[TextBodyNode] = Field(default_factory=list)


class PageHeader(BaseModel):
    model_config = _FORBID
    type: Literal["page_header"]
    head: ElementHead = Field(default_factory=ElementHead)
    body: list[TextBodyNode] = Field(default_factory=list)


class PageFooter(BaseModel):
    model_config = _FORBID
    type: Literal["page_footer"]
    head: ElementHead = Field(default_factory=ElementHead)
    body: list[TextBodyNode] = Field(default_factory=list)


class Footnote(BaseModel):
    model_config = _FORBID
    type: Literal["footnote"]
    head: ElementHead = Field(default_factory=ElementHead)
    body: list[TextBodyNode] = Field(default_factory=list)


class ListEl(BaseModel):
    model_config = _FORBID
    type: Literal["list"]
    class_: Optional[Literal["ordered", "unordered"]] = Field(default=None, alias="class")
    head: ElementHead = Field(default_factory=ElementHead)
    items: list[ListItem] = Field(default_factory=list)


class Group(BaseModel):
    model_config = _FORBID
    type: Literal["group"]
    head: ElementHead = Field(default_factory=ElementHead)
    body: list[GroupBodyNode] = Field(default_factory=list)


class FieldRegion(BaseModel):
    model_config = _FORBID
    type: Literal["field_region"]
    head: ElementHead = Field(default_factory=ElementHead)
    body: list[FieldBodyNode] = Field(default_factory=list)


class FieldHeading(BaseModel):
    model_config = _FORBID
    type: Literal["field_heading"]
    level: Optional[int] = Field(default=None, ge=1)
    head: ElementHead = Field(default_factory=ElementHead)
    body: list[FieldBodyNode] = Field(default_factory=list)


class FieldItem(BaseModel):
    model_config = _FORBID
    type: Literal["field_item"]
    head: ElementHead = Field(default_factory=ElementHead)
    body: list[FieldBodyNode] = Field(default_factory=list)


class Key(BaseModel):
    model_config = _FORBID
    type: Literal["key"]
    head: ElementHead = Field(default_factory=ElementHead)
    body: list[TextBodyNode] = Field(default_factory=list)


class Value(BaseModel):
    model_config = _FORBID
    type: Literal["value"]
    class_: Optional[Literal["read_only", "fillable"]] = Field(default=None, alias="class")
    head: ElementHead = Field(default_factory=ElementHead)
    body: list[TextBodyNode] = Field(default_factory=list)


class Picture(BaseModel):
    model_config = _FORBID
    type: Literal["picture"]
    class_: Optional[Literal["undefined", "chart"]] = Field(default=None, alias="class")
    head: ElementHead = Field(default_factory=ElementHead)
    src: Optional[Src] = None
    tabular: Optional[Tabular] = None
    body: list[TextBodyNode] = Field(default_factory=list)


class Table(BaseModel):
    model_config = _FORBID
    type: Literal["table"]
    head: ElementHead = Field(default_factory=ElementHead)
    cells: list[Cell] = Field(default_factory=list)


class Index(BaseModel):
    model_config = _FORBID
    type: Literal["index"]
    head: ElementHead = Field(default_factory=ElementHead)
    cells: list[Cell] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Head / document-level elements
# --------------------------------------------------------------------------- #

class DefaultResolution(BaseModel):
    model_config = _FORBID
    type: Literal["default_resolution"]
    width: Optional[int] = Field(default=None, ge=0)
    height: Optional[int] = Field(default=None, ge=0)


class Meta(BaseModel):
    model_config = _FORBID
    type: Literal["meta"]


class PageBreak(BaseModel):
    model_config = _FORBID
    type: Literal["page_break"]


class Head(BaseModel):
    """``<head>`` — document metadata (restricted to the known simple items)."""

    model_config = _FORBID
    items: list[HeadBodyNode] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Structural helpers (groups, not named XSD elements)
# --------------------------------------------------------------------------- #

class Cell(BaseModel):
    """A table / tabular cell: separator + optional head + body."""

    model_config = _FORBID
    sep: TableSep
    head: ElementHead = Field(default_factory=ElementHead)
    body: list[TextBodyNode] = Field(default_factory=list)


class ListItem(BaseModel):
    """A list item: ``ldiv`` + optional head + body."""

    model_config = _FORBID
    ldiv: Ldiv
    head: ElementHead = Field(default_factory=ElementHead)
    body: list[TextBodyNode] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Root document
# --------------------------------------------------------------------------- #

class DocLang(BaseModel):
    """The ``<doclang>`` root: optional ``head`` + top-level body."""

    model_config = _FORBID
    version: Optional[Literal["0.7"]] = None
    head: Optional[Head] = None
    body: list[DocLangBodyNode] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Body-node discriminated unions
# --------------------------------------------------------------------------- #
# Each is a *flat* union of concrete node models, discriminated on ``type``.
# (Intermediate aliases are not nested inside the unions — the discriminator
# requires a flat set of members.)

# formatting body: text_chunk | content | formatting
FmtBodyNode = Annotated[Union[
    TextChunk, Content, Formatting,
], Field(discriminator="type")]

# marker body: text_chunk | content | formatting | checkbox
MarkerBodyNode = Annotated[Union[
    TextChunk, Content, Formatting, Checkbox,
], Field(discriminator="type")]

# description / summary body: text_chunk | content
DescBodyNode = Annotated[Union[
    TextChunk, Content,
], Field(discriminator="type")]

# headless_txt: formatting | marker | hint | checkbox
HeadlessTxtNode = Annotated[Union[
    Formatting, Marker, Hint, Checkbox,
], Field(discriminator="type")]

# top_level_cat: the 17 top-level semantic elements
TopLevelNode = Annotated[Union[
    Text, Heading, Code, Formula, PageHeader, PageFooter, Footnote,
    ListEl, Group, FieldRegion, FieldHeading, FieldItem, Key, Value,
    Picture, Table, Index,
], Field(discriminator="type")]

# headless: headless_txt | code | formula | picture | field_* | key | value
HeadlessNode = Annotated[Union[
    Formatting, Marker, Hint, Checkbox,
    Code, Formula, Picture, FieldRegion, FieldHeading, FieldItem, Key, Value,
], Field(discriminator="type")]

# text-like body (text/heading/caption/page_header/page_footer/footnote/key/
# value/list_item/table_cell/picture): text_chunk | content | headless_txt |
# top_level
TextBodyNode = Annotated[Union[
    TextChunk, Content,
    Formatting, Marker, Hint, Checkbox,
    Text, Heading, Code, Formula, PageHeader, PageFooter, Footnote,
    ListEl, Group, FieldRegion, FieldHeading, FieldItem, Key, Value,
    Picture, Table, Index,
], Field(discriminator="type")]

# code / formula body: text_chunk | content | headless
CodeBodyNode = Annotated[Union[
    TextChunk, Content,
    Formatting, Marker, Hint, Checkbox,
    Code, Formula, Picture, FieldRegion, FieldHeading, FieldItem, Key, Value,
], Field(discriminator="type")]

# field_region / field_heading / field_item body: text_chunk | headless_txt |
# top_level  (NO <content>)
FieldBodyNode = Annotated[Union[
    TextChunk,
    Formatting, Marker, Hint, Checkbox,
    Text, Heading, Code, Formula, PageHeader, PageFooter, Footnote,
    ListEl, Group, FieldRegion, FieldHeading, FieldItem, Key, Value,
    Picture, Table, Index,
], Field(discriminator="type")]

# group body: top_level only (non-mixed)
GroupBodyNode = Annotated[Union[
    Text, Heading, Code, Formula, PageHeader, PageFooter, Footnote,
    ListEl, Group, FieldRegion, FieldHeading, FieldItem, Key, Value,
    Picture, Table, Index,
], Field(discriminator="type")]

# doclang body: top_level | page_break
DocLangBodyNode = Annotated[Union[
    Text, Heading, Code, Formula, PageHeader, PageFooter, Footnote,
    ListEl, Group, FieldRegion, FieldHeading, FieldItem, Key, Value,
    Picture, Table, Index,
    PageBreak,
], Field(discriminator="type")]

# head body: default_resolution | meta
HeadBodyNode = Annotated[Union[
    DefaultResolution, Meta,
], Field(discriminator="type")]


# --------------------------------------------------------------------------- #
# Element registry (element name → model class) for the sync tests
# --------------------------------------------------------------------------- #

ELEMENT_MODELS: dict[str, type[BaseModel]] = {
    # root / head
    "doclang": DocLang,
    "head": Head,
    "default_resolution": DefaultResolution,
    "meta": Meta,
    "page_break": PageBreak,
    # semantic
    "text": Text,
    "heading": Heading,
    "code": Code,
    "formula": Formula,
    "caption": Caption,
    "page_header": PageHeader,
    "page_footer": PageFooter,
    "footnote": Footnote,
    "list": ListEl,
    "group": Group,
    "field_region": FieldRegion,
    "field_heading": FieldHeading,
    "field_item": FieldItem,
    "key": Key,
    "value": Value,
    "picture": Picture,
    "table": Table,
    "index": Index,
    # formatting (local elements in the XSD formatting_group)
    "bold": Formatting,
    "italic": Formatting,
    "strikethrough": Formatting,
    "underline": Formatting,
    "superscript": Formatting,
    "subscript": Formatting,
    "rtl": Formatting,
    "handwriting": Formatting,
    # property / payload
    "label": Label,
    "thread": Thread,
    "xref": Xref,
    "href": Href,
    "layer": Layer,
    "location": Location,
    "description": Description,
    "summary": Summary,
    "custom": Custom,
    "src": Src,
    "tabular": Tabular,
    "checkbox": Checkbox,
    "content": Content,
    # structural
    "fcel": TableSep,
    "ecel": TableSep,
    "ched": TableSep,
    "rhed": TableSep,
    "corn": TableSep,
    "srow": TableSep,
    "lcel": TableSep,
    "ucel": TableSep,
    "xcel": TableSep,
    "nl": TableSep,
    "ldiv": Ldiv,
    # marker / hint
    "marker": Marker,
    "hint": Hint,
}


# --------------------------------------------------------------------------- #
# Rebuild all models so the recursive forward references (defined above as
# string annotations via ``from __future__ import annotations``) resolve
# against the now-defined type aliases.
# --------------------------------------------------------------------------- #

def _rebuild_all() -> None:
    _models = [
        TextChunk, Content,
        Label, Thread, Xref, Href, Layer, Location, Description, Summary, Custom,
        Src, Checkbox, TableSep, Ldiv, Tabular, Formatting, Marker, Hint,
        Text, Heading, Code, Formula, Caption, PageHeader, PageFooter, Footnote,
        ListEl, Group, FieldRegion, FieldHeading, FieldItem, Key, Value,
        Picture, Table, Index,
        DefaultResolution, Meta, PageBreak, Head, Cell, ListItem, ElementHead,
        DocLang,
    ]
    for _m in _models:
        _m.model_rebuild()


_rebuild_all()
