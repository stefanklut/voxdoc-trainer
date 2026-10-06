"""Structured DocLang output: EBNF grammar + Pydantic/JSON-schema fallback.

This package constrains an OpenAI-compatible endpoint (e.g. local vLLM) to emit
only valid DocLang (``.dclg`` XML) via two complementary mechanisms:

1. **Primary** — an EBNF grammar (``grammar.DOCLANG_GRAMMAR``) passed to vLLM's
   structured-output API (``structured_outputs`` + ``guided_decoding_backend``
   for vLLM >= 0.29) so the model emits ``.dclg`` directly.
2. **Fallback** — Pydantic v2 models (``models``) exposing a JSON schema
   (``schema``) for portable ``response_format`` use, plus a JSON→``.dclg``
   serializer (``serialize``) and a ``.dclg``→model parser (``parse``).

Both paths are gated by :func:`validate_doclang` (the official ``doclang``
validator) because a CFG / JSON schema cannot express DocLang's
context-sensitive rules (table rectangularity, location ordering, field
nesting, ...).
"""

from .grammar import DOCLANG_GRAMMAR, get_grammar
from .models import DocLang, ELEMENT_MODELS
from .schema import get_doclang_json_schema, response_format
from .serialize import model_to_doclang_xml
from .parse import doclang_xml_to_model
from .validate import DocLangValidationError, validate_doclang

__all__ = [
    "DOCLANG_GRAMMAR",
    "get_grammar",
    "DocLang",
    "ELEMENT_MODELS",
    "get_doclang_json_schema",
    "response_format",
    "model_to_doclang_xml",
    "doclang_xml_to_model",
    "DocLangValidationError",
    "validate_doclang",
]
