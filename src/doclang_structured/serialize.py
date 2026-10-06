"""Serialize a :class:`~src.doclang_structured.models.DocLang` model to ``.dclg`` XML.

:func:`model_to_doclang_xml` renders the model as a compact, XSD-valid DocLang
document:

* the element head is emitted in the canonical XSD order
  (``label? thread? (xref|href)? layer? location×4? caption? description?
  summary? custom?``);
* XML special characters are escaped (``& < >`` in text, plus ``"`` in
  attribute values);
* a text run with leading/trailing whitespace is wrapped in ``<content>``
  (``xml:space="preserve"``) so the whitespace is explicit;
* the root always carries ``xmlns`` and ``version="0.7"``.

The output is a *sound* rendering: for any model that passes Pydantic
validation, the emitted XML passes ``validate_doclang(xsd_only=True)``. The
inverse direction (XML → model) is :mod:`src.doclang_structured.parse`.
"""

from __future__ import annotations

from typing import Any

from .models import (
    Cell,
    DocLang,
    ElementHead,
    Head,
    ListItem,
)

__all__ = ["model_to_doclang_xml"]

_XMLNS = "https://www.doclang.ai/ns/v0"
_VERSION = "0.7"

_FORMATTING_TAGS = frozenset(
    {"bold", "italic", "strikethrough", "underline", "superscript", "subscript", "rtl", "handwriting"}
)


# --------------------------------------------------------------------------- #
# Escaping
# --------------------------------------------------------------------------- #

def _escape_text(text: str) -> str:
    """Escape XML special characters for element content."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _escape_attr(text: str) -> str:
    """Escape XML special characters for a double-quoted attribute value."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# --------------------------------------------------------------------------- #
# Element head
# --------------------------------------------------------------------------- #

def _serialize_element_head(head: ElementHead) -> str:
    parts: list[str] = []
    if head.label is not None:
        parts.append(_serialize_label(head.label))
    if head.thread is not None:
        parts.append(f'<thread thread_id="{head.thread.thread_id}"/>')
    if head.xref is not None:
        parts.append(f'<xref thread_id="{head.xref.thread_id}"/>')
    elif head.href is not None:
        parts.append(f'<href uri="{_escape_attr(head.href.uri)}"/>')
    if head.layer is not None:
        if head.layer.value is not None:
            parts.append(f'<layer value="{_escape_attr(head.layer.value)}"/>')
        else:
            parts.append("<layer/>")
    for loc in head.locations:
        res = f' resolution="{loc.resolution}"' if loc.resolution is not None else ""
        parts.append(f'<location value="{loc.value}"{res}/>')
    if head.caption is not None:
        parts.append(_serialize_caption(head.caption))
    if head.description is not None:
        parts.append(_serialize_desc(head.description, "description"))
    if head.summary is not None:
        parts.append(_serialize_desc(head.summary, "summary"))
    if head.custom is not None:
        parts.append("<custom/>")
    return "".join(parts)


def _serialize_label(label: Any) -> str:
    if label.value is not None:
        return f'<label value="{_escape_attr(label.value)}"/>'
    return "<label/>"


def _serialize_desc(desc: Any, tag: str) -> str:
    body = "".join(_serialize_body_node(n) for n in desc.body)
    return f"<{tag}>{body}</{tag}>"


def _serialize_caption(caption: Any) -> str:
    head = _serialize_element_head(caption.head)
    body = "".join(_serialize_body_node(n) for n in caption.body)
    return f"<caption>{head}{body}</caption>"


# --------------------------------------------------------------------------- #
# Body nodes (dispatch on ``type``)
# --------------------------------------------------------------------------- #

def _serialize_body_node(node: Any) -> str:
    t = node.type
    if t == "text_chunk":
        return _serialize_text_chunk(node)
    if t == "content":
        if node.text is None:
            return "<content/>"
        return f"<content>{_escape_text(node.text)}</content>"
    if t in _FORMATTING_TAGS:
        body = "".join(_serialize_body_node(n) for n in node.body)
        return f"<{t}>{body}</{t}>"
    if t == "marker":
        head = _serialize_element_head(node.head)
        body = "".join(_serialize_body_node(n) for n in node.body)
        return f"<marker>{head}{body}</marker>"
    if t == "hint":
        body = "".join(_serialize_body_node(n) for n in node.body)
        return f"<hint>{body}</hint>"
    if t == "checkbox":
        if node.class_ is not None:
            return f'<checkbox class="{_escape_attr(node.class_)}"/>'
        return "<checkbox/>"
    if t == "text":
        return _open_close("text", "", node)
    if t == "heading":
        return _open_close("heading", _level_attr(node.level), node)
    if t == "code":
        return _open_close("code", "", node)
    if t == "formula":
        return _open_close("formula", "", node)
    if t == "caption":
        return _serialize_caption(node)
    if t == "page_header":
        return _open_close("page_header", "", node)
    if t == "page_footer":
        return _open_close("page_footer", "", node)
    if t == "footnote":
        return _open_close("footnote", "", node)
    if t == "list":
        return _serialize_list(node)
    if t == "group":
        return _open_close("group", "", node)
    if t == "field_region":
        return _open_close("field_region", "", node)
    if t == "field_heading":
        return _open_close("field_heading", _level_attr(node.level), node)
    if t == "field_item":
        return _open_close("field_item", "", node)
    if t == "key":
        return _open_close("key", "", node)
    if t == "value":
        return _open_close("value", _class_attr(node.class_), node)
    if t == "picture":
        return _serialize_picture(node)
    if t == "table":
        return _serialize_tablelike("table", node)
    if t == "index":
        return _serialize_tablelike("index", node)
    if t == "page_break":
        return "<page_break/>"
    raise ValueError(f"unknown body node type: {t!r}")


def _open_close(tag: str, attrs: str, node: Any) -> str:
    head = _serialize_element_head(node.head)
    body = "".join(_serialize_body_node(n) for n in node.body)
    return f"<{tag}{attrs}>{head}{body}</{tag}>"


def _level_attr(level: int | None) -> str:
    return f' level="{level}"' if level is not None else ""


def _class_attr(cls: str | None) -> str:
    return f' class="{_escape_attr(cls)}"' if cls is not None else ""


def _serialize_text_chunk(node: Any) -> str:
    text = node.text
    if not text:
        return ""
    escaped = _escape_text(text)
    if text != text.strip():
        return f"<content>{escaped}</content>"
    return escaped


def _serialize_list(node: Any) -> str:
    head = _serialize_element_head(node.head)
    items = "".join(_serialize_list_item(item) for item in node.items)
    return f"<list{_class_attr(node.class_)}>{head}{items}</list>"


def _serialize_list_item(item: ListItem) -> str:
    ldiv = _serialize_ldiv(item.ldiv)
    head = _serialize_element_head(item.head)
    body = "".join(_serialize_body_node(n) for n in item.body)
    return f"{ldiv}{head}{body}"


def _serialize_ldiv(ldiv: Any) -> str:
    if ldiv.marker is not None:
        head = _serialize_element_head(ldiv.marker.head)
        body = "".join(_serialize_body_node(n) for n in ldiv.marker.body)
        return f"<ldiv><marker>{head}{body}</marker></ldiv>"
    return "<ldiv/>"


def _serialize_picture(node: Any) -> str:
    head = _serialize_element_head(node.head)
    src = f'<src uri="{_escape_attr(node.src.uri)}"/>' if node.src is not None else ""
    tabular = _serialize_tabular(node.tabular) if node.tabular is not None else ""
    body = "".join(_serialize_body_node(n) for n in node.body)
    return f"<picture{_class_attr(node.class_)}>{head}{src}{tabular}{body}</picture>"


def _serialize_tabular(node: Any) -> str:
    cells = "".join(_serialize_cell(c) for c in node.cells)
    return f"<tabular>{cells}</tabular>"


def _serialize_cell(cell: Cell) -> str:
    sep = f"<{cell.sep.type}/>"
    head = _serialize_element_head(cell.head)
    body = "".join(_serialize_body_node(n) for n in cell.body)
    return f"{sep}{head}{body}"


def _serialize_tablelike(tag: str, node: Any) -> str:
    head = _serialize_element_head(node.head)
    cells = "".join(_serialize_cell(c) for c in node.cells)
    return f"<{tag}>{head}{cells}</{tag}>"


# --------------------------------------------------------------------------- #
# Head / root
# --------------------------------------------------------------------------- #

def _serialize_head(head: Head) -> str:
    items: list[str] = []
    for item in head.items:
        if item.type == "default_resolution":
            w = f' width="{item.width}"' if item.width is not None else ""
            h = f' height="{item.height}"' if item.height is not None else ""
            items.append(f"<default_resolution{w}{h}/>")
        elif item.type == "meta":
            items.append("<meta/>")
        else:  # pragma: no cover - guarded by the model
            raise ValueError(f"unknown head item type: {item.type!r}")
    return f"<head>{''.join(items)}</head>"


def model_to_doclang_xml(model: DocLang) -> str:
    """Render ``model`` as a compact, XSD-valid ``.dclg`` document string."""
    parts = [f'<doclang xmlns="{_XMLNS}" version="{_VERSION}">']
    if model.head is not None:
        parts.append(_serialize_head(model.head))
    for node in model.body:
        parts.append(_serialize_body_node(node))
    parts.append("</doclang>")
    return "".join(parts)
