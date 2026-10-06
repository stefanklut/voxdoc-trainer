"""JSON schema for the DocLang fallback path.

The fallback path constrains the model to emit JSON conforming to the schema
derived from :class:`src.doclang_structured.models.DocLang`, then parses that
JSON into the Pydantic models and serializes it to ``.dclg`` XML.

* :func:`get_doclang_json_schema` — the (strict-compatible) JSON schema.
* :func:`response_format` — the OpenAI ``response_format`` payload for
  ``chat.completions.create``.

The schema is a recursive set of discriminated unions (on the ``type`` field),
which structured decoders (vLLM ``xgrammar`` / ``outlines`` backends) handle
well. :func:`_make_strict_compatible` normalizes it for strict decoding.
"""

from __future__ import annotations

import copy
from typing import Any

from .models import DocLang

__all__ = ["get_doclang_json_schema", "response_format", "_make_strict_compatible"]

# Keywords that structured decoders either ignore or choke on. We drop them so
# the schema is a clean constraint. (``default`` is the main offender — it is
# not a constraint and some backends reject it in strict mode.)
_DROP_KEYWORDS = {"default", "title", "description", "examples", "deprecated"}


def _make_strict_compatible(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``schema`` normalized for strict JSON decoding.

    * Forces ``additionalProperties: False`` on every object schema (required
      for OpenAI ``strict: True`` and for unambiguous decoding).
    * Drops non-constraint keywords (``default``, ``title``, ``description``,
      ``examples``, ``deprecated``).
    * Preserves ``$defs`` / ``$ref`` and all constraint keywords.
    """
    schema = copy.deepcopy(schema)
    _walk(schema)
    return schema


def _walk(node: Any) -> None:
    if isinstance(node, dict):
        # Drop non-constraint keywords.
        for key in _DROP_KEYWORDS:
            node.pop(key, None)
        # Force strict objects.
        if node.get("type") == "object" or "properties" in node:
            node["additionalProperties"] = False
        for value in node.values():
            _walk(value)
    elif isinstance(node, list):
        for item in node:
            _walk(item)


def get_doclang_json_schema() -> dict[str, Any]:
    """Return the strict-compatible JSON schema for :class:`DocLang`."""
    return _make_strict_compatible(DocLang.model_json_schema())


def response_format(strict: bool = False) -> dict[str, Any]:
    """Return the OpenAI ``response_format`` payload (json_schema).

    ``strict`` defaults to ``False`` because the DocLang schema uses a
    discriminated union (``oneOf`` + ``discriminator``) for the body, which is
    outside OpenAI's strict-mode subset (strict mode forbids ``oneOf`` /
    ``anyOf`` / ``allOf``). With ``strict=False`` the schema guides the model
    but is not hard-enforced by the endpoint; correctness is guaranteed
    client-side by :func:`~src.doclang_structured.validate.validate_doclang`
    (see :mod:`~src.doclang_structured.client`).
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "doclang",
            "schema": get_doclang_json_schema(),
            "strict": strict,
        },
    }
