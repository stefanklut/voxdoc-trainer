"""Parse a ``.dclg`` XML document into a :class:`~src.doclang_structured.models.DocLang` model.

:func:`doclang_xml_to_model` uses the standard-library
:mod:`xml.etree.ElementTree` and walks the tree in document order, reconstructing
the mixed content (raw text interleaved with elements) via each element's
``.text`` (text before its first child) and ``.tail`` (text after it).

The parser is the inverse of
:func:`src.doclang_structured.serialize.model_to_doclang_xml`: for any
XSD-valid document the grammar accepts, ``doclang_xml_to_model`` succeeds and
``model_to_doclang_xml(doclang_xml_to_model(xml))`` round-trips to an
equivalent document.

Notes
-----
* The DocLang namespace (``https://www.doclang.ai/ns/v0``) is stripped from
  tags; documents without the namespace also parse (tags are matched by local
  name).
* Non-mixed elements (``doclang``, ``head``, ``group``, ``ldiv``, ``custom``)
  ignore whitespace text; mixed elements treat it as content.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

from .models import (
    Caption,
    Cell,
    Checkbox,
    Code,
    Content,
    Custom,
    DefaultResolution,
    DocLang,
    ElementHead,
    FieldHeading,
    FieldItem,
    FieldRegion,
    Footnote,
    Formula,
    Group,
    Head,
    Heading,
    Hint,
    Href,
    Index,
    Key,
    Label,
    Layer,
    Ldiv,
    ListItem,
    ListEl,
    Location,
    Marker,
    Meta,
    PageBreak,
    PageFooter,
    PageHeader,
    Picture,
    Src,
    Summary,
    Tabular,
    Table,
    TableSep,
    Text,
    TextChunk,
    Thread,
    Value,
    Xref,
)

__all__ = ["doclang_xml_to_model"]

_FORMATTING_TAGS = frozenset(
    {"bold", "italic", "strikethrough", "underline", "superscript", "subscript", "rtl", "handwriting"}
)
_TABLE_SEP_TAGS = frozenset(
    {"fcel", "ecel", "ched", "rhed", "corn", "srow", "lcel", "ucel", "xcel", "nl"}
)


def _tag(elem: ET.Element) -> str:
    """Local tag name (namespace stripped)."""
    t = elem.tag
    if isinstance(t, str) and "}" in t:
        return t.split("}", 1)[1]
    return t


def _text_node(text: str) -> TextChunk:
    return TextChunk(type="text_chunk", text=text)


# --------------------------------------------------------------------------- #
# Element head
# --------------------------------------------------------------------------- #

def _parse_head_prefix(children: list[ET.Element], start: int) -> tuple[ElementHead, int]:
    """Parse the ordered element head from ``children[start:]``.

    Returns the head and the index of the first non-head child.
    """
    head = ElementHead()
    i = start
    n = len(children)

    def cur() -> str | None:
        return _tag(children[i]) if i < n else None

    if cur() == "label":
        head.label = Label(type="label", value=children[i].get("value"))
        i += 1
    if cur() == "thread":
        head.thread = Thread(type="thread", thread_id=int(children[i].get("thread_id")))
        i += 1
    if cur() in ("xref", "href"):
        if cur() == "xref":
            head.xref = Xref(type="xref", thread_id=int(children[i].get("thread_id")))
        else:
            head.href = Href(type="href", uri=children[i].get("uri"))
        i += 1
    if cur() == "layer":
        head.layer = Layer(type="layer", value=children[i].get("value"))
        i += 1
    while cur() == "location":
        res = children[i].get("resolution")
        head.locations.append(
            Location(type="location", value=int(children[i].get("value")), resolution=int(res) if res else None)
        )
        i += 1
    if cur() == "caption":
        head.caption = _parse_caption(children[i])
        i += 1
    if cur() == "description":
        head.description = _parse_desc(children[i], "description")
        i += 1
    if cur() == "summary":
        head.summary = _parse_desc(children[i], "summary")
        i += 1
    if cur() == "custom":
        head.custom = Custom(type="custom")
        i += 1
    return head, i


# --------------------------------------------------------------------------- #
# Mixed body (text + elements in document order)
# --------------------------------------------------------------------------- #

def _mixed_body_from(children: list[ET.Element], i: int, elem: ET.Element) -> list[Any]:
    """Build the mixed body from ``children[i:]``.

    If ``i == 0`` (no head / preceding element) the body starts with
    ``elem.text``; otherwise it starts with the ``.tail`` of the element
    immediately before the body (``children[i-1]``).
    """
    body: list[Any] = []
    if i == 0:
        if elem.text:
            body.append(_text_node(elem.text))
    else:
        anchor = children[i - 1]
        if anchor.tail:
            body.append(_text_node(anchor.tail))
    for child in children[i:]:
        body.append(_parse_element(child))
        if child.tail:
            body.append(_text_node(child.tail))
    return body


def _parse_headed_mixed(elem: ET.Element, make: Any) -> Any:
    children = list(elem)
    head, i = _parse_head_prefix(children, 0)
    body = _mixed_body_from(children, i, elem)
    return make(head=head, body=body)


def _parse_body_only_mixed(elem: ET.Element, make: Any) -> Any:
    children = list(elem)
    body = _mixed_body_from(children, 0, elem)
    return make(body=body)


# --------------------------------------------------------------------------- #
# Element dispatch
# --------------------------------------------------------------------------- #

def _parse_element(elem: ET.Element) -> Any:
    t = _tag(elem)
    if t == "text":
        return _parse_headed_mixed(elem, lambda head, body: Text(type="text", head=head, body=body))
    if t == "heading":
        level = elem.get("level")
        return _parse_headed_mixed(
            elem, lambda head, body: Heading(type="heading", level=int(level) if level else None, head=head, body=body)
        )
    if t == "code":
        return _parse_headed_mixed(elem, lambda head, body: Code(type="code", head=head, body=body))
    if t == "formula":
        return _parse_headed_mixed(elem, lambda head, body: Formula(type="formula", head=head, body=body))
    if t == "caption":
        return _parse_caption(elem)
    if t == "page_header":
        return _parse_headed_mixed(elem, lambda head, body: PageHeader(type="page_header", head=head, body=body))
    if t == "page_footer":
        return _parse_headed_mixed(elem, lambda head, body: PageFooter(type="page_footer", head=head, body=body))
    if t == "footnote":
        return _parse_headed_mixed(elem, lambda head, body: Footnote(type="footnote", head=head, body=body))
    if t == "list":
        return _parse_list(elem)
    if t == "group":
        return _parse_group(elem)
    if t == "field_region":
        return _parse_headed_mixed(elem, lambda head, body: FieldRegion(type="field_region", head=head, body=body))
    if t == "field_heading":
        level = elem.get("level")
        return _parse_headed_mixed(
            elem, lambda head, body: FieldHeading(type="field_heading", level=int(level) if level else None, head=head, body=body)
        )
    if t == "field_item":
        return _parse_headed_mixed(elem, lambda head, body: FieldItem(type="field_item", head=head, body=body))
    if t == "key":
        return _parse_headed_mixed(elem, lambda head, body: Key(type="key", head=head, body=body))
    if t == "value":
        cls = elem.get("class")
        return _parse_headed_mixed(elem, lambda head, body: Value(type="value", class_=cls, head=head, body=body))
    if t == "picture":
        return _parse_picture(elem)
    if t == "table":
        return _parse_tablelike(elem, lambda head, cells: Table(type="table", head=head, cells=cells))
    if t == "index":
        return _parse_tablelike(elem, lambda head, cells: Index(type="index", head=head, cells=cells))
    if t in _FORMATTING_TAGS:
        return _parse_body_only_mixed(elem, lambda body: _formatting(t, body))
    if t == "marker":
        return _parse_headed_mixed(elem, lambda head, body: Marker(type="marker", head=head, body=body))
    if t == "hint":
        return _parse_body_only_mixed(elem, lambda body: Hint(type="hint", body=body))
    if t == "checkbox":
        return Checkbox(type="checkbox", class_=elem.get("class"))
    if t == "content":
        return Content(type="content", text=elem.text)
    if t == "page_break":
        return PageBreak(type="page_break")
    raise ValueError(f"unknown DocLang element: <{t}>")


def _formatting(tag: str, body: list[Any]) -> Any:
    from .models import Formatting

    return Formatting(type=tag, body=body)


def _parse_caption(elem: ET.Element) -> Caption:
    return _parse_headed_mixed(elem, lambda head, body: Caption(type="caption", head=head, body=body))


def _parse_desc(elem: ET.Element, tag: str) -> Any:
    body = _mixed_body_from(list(elem), 0, elem)
    if tag == "description":
        from .models import Description

        return Description(type="description", body=body)
    return Summary(type="summary", body=body)


def _parse_group(elem: ET.Element) -> Group:
    # Non-mixed: body is top-level elements only (whitespace is ignorable).
    children = list(elem)
    head, i = _parse_head_prefix(children, 0)
    body = [_parse_element(child) for child in children[i:]]
    return Group(type="group", head=head, body=body)


def _parse_list(elem: ET.Element) -> ListEl:
    cls = elem.get("class")
    children = list(elem)
    head, i = _parse_head_prefix(children, 0)
    items: list[ListItem] = []
    n = len(children)
    while i < n:
        ldiv = _parse_ldiv(children[i])
        i += 1
        item_head, i = _parse_head_prefix(children, i)
        body = _cell_like_body(children, i, stop_tags={"ldiv"})
        i = body[1]
        items.append(ListItem(ldiv=ldiv, head=item_head, body=body[0]))
    return ListEl(type="list", class_=cls, head=head, items=items)


def _parse_tablelike(elem: ET.Element, make: Any) -> Any:
    children = list(elem)
    head, i = _parse_head_prefix(children, 0)
    cells = _parse_cells(children, i)
    return make(head=head, cells=cells)


def _parse_tabular(elem: ET.Element) -> Tabular:
    children = list(elem)
    cells = _parse_cells(children, 0)
    return Tabular(type="tabular", cells=cells)


def _parse_cells(children: list[ET.Element], start: int) -> list[Cell]:
    cells: list[Cell] = []
    i = start
    n = len(children)
    while i < n:
        sep = TableSep(type=_tag(children[i]))
        i += 1
        cell_head, i = _parse_head_prefix(children, i)
        body, i = _cell_like_body(children, i, stop_tags=_TABLE_SEP_TAGS)
        cells.append(Cell(sep=sep, head=cell_head, body=body))
    return cells


def _cell_like_body(
    children: list[ET.Element], i: int, stop_tags: frozenset[str]
) -> tuple[list[Any], int]:
    """Parse a cell / list-item body from ``children[i:]`` until a stop tag.

    Returns ``(body, next_index)``. The body's first text is the ``.tail`` of
    the element immediately before it (``children[i-1]``).
    """
    body: list[Any] = []
    n = len(children)
    if i > 0:
        anchor = children[i - 1]
        if anchor.tail:
            body.append(_text_node(anchor.tail))
    body_elems: list[ET.Element] = []
    while i < n and _tag(children[i]) not in stop_tags:
        body_elems.append(children[i])
        i += 1
    for be in body_elems:
        body.append(_parse_element(be))
        if be.tail:
            body.append(_text_node(be.tail))
    return body, i


def _parse_ldiv(elem: ET.Element) -> Ldiv:
    children = list(elem)
    if children and _tag(children[0]) == "marker":
        return Ldiv(type="ldiv", marker=_parse_element(children[0]))
    return Ldiv(type="ldiv", marker=None)


def _parse_picture(elem: ET.Element) -> Picture:
    cls = elem.get("class")
    children = list(elem)
    head, i = _parse_head_prefix(children, 0)
    src: Src | None = None
    tabular: Tabular | None = None
    if i < len(children) and _tag(children[i]) == "src":
        src = Src(type="src", uri=children[i].get("uri"))
        i += 1
    if i < len(children) and _tag(children[i]) == "tabular":
        tabular = _parse_tabular(children[i])
        i += 1
    body = _mixed_body_from(children, i, elem)
    return Picture(type="picture", class_=cls, head=head, src=src, tabular=tabular, body=body)


# --------------------------------------------------------------------------- #
# Head / root
# --------------------------------------------------------------------------- #

def _parse_head(elem: ET.Element) -> Head:
    items: list[Any] = []
    for child in elem:
        t = _tag(child)
        if t == "default_resolution":
            w = child.get("width")
            h = child.get("height")
            items.append(
                DefaultResolution(
                    type="default_resolution",
                    width=int(w) if w else None,
                    height=int(h) if h else None,
                )
            )
        elif t == "meta":
            items.append(Meta(type="meta"))
        # Other (xs:any) head items are out of scope and ignored.
    return Head(items=items)


def doclang_xml_to_model(xml: str) -> DocLang:
    """Parse a ``.dclg`` document string into a :class:`DocLang` model."""
    root = ET.fromstring(xml)
    if _tag(root) != "doclang":
        raise ValueError(f"expected <doclang> root, got <{_tag(root)}>")
    children = list(root)
    head: Head | None = None
    i = 0
    if children and _tag(children[0]) == "head":
        head = _parse_head(children[0])
        i = 1
    body = [_parse_element(child) for child in children[i:]]
    return DocLang(version=root.get("version"), head=head, body=body)
