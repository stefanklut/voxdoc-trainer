"""Behavioural tests for the pydantic DocLang models.

These check the *structural* invariants that the models enforce directly (as
opposed to the context-sensitive rules that only the validator can check):

* a well-formed document validates;
* ``xref`` and ``href`` are mutually exclusive in the element head;
* the location block must be empty or exactly four locations;
* ``extra="forbid`` rejects unknown fields;
* the ``class`` attribute is exposed via the ``class_`` alias (and accepted by
  both names);
* enumerated attributes reject out-of-domain values;
* numeric attributes enforce their bounds.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.doclang_structured.models import (
    Checkbox,
    DocLang,
    ElementHead,
    Layer,
    Thread,
)


def _text_doc(body: list | None = None) -> dict:
    return {"body": body or [{"type": "text", "body": [{"type": "text_chunk", "text": "hi"}]}]}


def test_valid_text_document() -> None:
    doc = DocLang.model_validate(_text_doc())
    assert doc.body[0].type == "text"


def test_valid_document_with_head() -> None:
    doc = DocLang.model_validate(
        {
            "head": {"items": [{"type": "default_resolution", "width": 100, "height": 100}]},
            "body": [{"type": "text", "body": [{"type": "text_chunk", "text": "x"}]}],
        }
    )
    assert doc.head is not None


def test_xref_and_href_mutually_exclusive() -> None:
    with pytest.raises(ValidationError):
        ElementHead.model_validate(
            {
                "xref": {"type": "xref", "thread_id": 1},
                "href": {"type": "href", "uri": "http://x"},
            }
        )


def test_xref_alone_is_fine() -> None:
    head = ElementHead.model_validate({"xref": {"type": "xref", "thread_id": 1}})
    assert head.xref is not None and head.href is None


def test_locations_must_be_0_or_4() -> None:
    two = [{"type": "location", "value": i} for i in range(2)]
    with pytest.raises(ValidationError):
        ElementHead.model_validate({"locations": two})
    four = [{"type": "location", "value": i} for i in range(4)]
    head = ElementHead.model_validate({"locations": four})
    assert len(head.locations) == 4


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        DocLang.model_validate({**_text_doc(), "bogus": 1})


def test_class_alias_accepted() -> None:
    # The JSON/XML attribute is ``class``; the Python field is ``class_``.
    by_alias = Checkbox.model_validate({"type": "checkbox", "class": "selected"})
    assert by_alias.class_ == "selected"
    by_name = Checkbox.model_validate({"type": "checkbox", "class_": "unselected"})
    assert by_name.class_ == "unselected"


def test_class_enum_rejects_out_of_domain() -> None:
    with pytest.raises(ValidationError):
        Checkbox.model_validate({"type": "checkbox", "class": "bogus"})


def test_layer_enum_rejects_out_of_domain() -> None:
    with pytest.raises(ValidationError):
        Layer.model_validate({"type": "layer", "value": "bogus"})


def test_thread_id_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        Thread.model_validate({"type": "thread", "thread_id": 0})
    ok = Thread.model_validate({"type": "thread", "thread_id": 1})
    assert ok.thread_id == 1


def test_version_is_pinned() -> None:
    doc = DocLang.model_validate({**_text_doc(), "version": "0.7"})
    assert doc.version == "0.7"
    with pytest.raises(ValidationError):
        DocLang.model_validate({**_text_doc(), "version": "0.8"})
