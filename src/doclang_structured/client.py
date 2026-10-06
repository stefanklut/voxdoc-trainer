"""OpenAI-compatible clients for structured DocLang generation.

Two generation paths, both gated by
:func:`~src.doclang_structured.validate.validate_doclang`:

* :func:`generate_doclang_grammar` — **primary**. Sends the EBNF grammar
  (:data:`~src.doclang_structured.grammar.DOCLANG_GRAMMAR`) to a vLLM server so
  the model emits ``.dclg`` directly.
* :func:`generate_doclang_json_schema` — **fallback**. Constrains the model to
  emit JSON matching the DocLang schema (``response_format``), then converts
  JSON → :class:`~src.doclang_structured.models.DocLang` → ``.dclg``.

Both retry up to ``max_retries`` times if the output fails validation, so the
returned string is always guaranteed valid.

vLLM API note
-------------
vLLM >= 0.29 exposes structured outputs through the request fields
``structured_outputs`` (an object with ``grammar`` / ``json`` / ``regex`` /
``choice`` / ``json_object``) and a separate top-level
``guided_decoding_backend`` string (``auto`` / ``xgrammar`` / ``guidance`` /
``outlines`` / ``lm-format-enforcer``). The older ``guided_decoding`` dict is
no longer accepted, so the grammar is sent as::

    extra_body={
        "structured_outputs": {"grammar": DOCLANG_GRAMMAR},
        "guided_decoding_backend": backend,
    }
"""

from __future__ import annotations

import json
from typing import Any, Sequence

from openai import OpenAI
from pydantic import ValidationError

from .grammar import DOCLANG_GRAMMAR
from .models import DocLang
from .schema import response_format
from .serialize import model_to_doclang_xml
from .validate import DocLangValidationError, validate_doclang

__all__ = ["generate_doclang_grammar", "generate_doclang_json_schema"]

Messages = Sequence[dict[str, Any]]


def _make_client(base_url: str, api_key: str) -> OpenAI:
    """Construct an OpenAI-compatible client (separated for testability)."""
    return OpenAI(base_url=base_url, api_key=api_key)


def _content(resp: Any) -> str:
    """Extract the assistant message content, failing on empty output."""
    text = resp.choices[0].message.content
    if not text:
        raise DocLangValidationError("model returned empty content")
    return text


def generate_doclang_grammar(
    base_url: str,
    model: str,
    messages: Messages,
    *,
    api_key: str = "EMPTY",
    backend: str = "xgrammar",
    max_retries: int = 3,
    temperature: float = 0.0,
) -> str:
    """Generate a validated ``.dclg`` document using vLLM's EBNF grammar.

    The grammar is sent via ``extra_body`` (see module docstring for the exact
    vLLM >= 0.29 field names). The returned string is guaranteed to pass
    :func:`validate_doclang`.

    Args:
        base_url: vLLM OpenAI-compatible endpoint (e.g.
            ``http://localhost:8000/v1``).
        model: Served model identifier.
        messages: Chat messages in OpenAI format. Each message's ``content``
            may be a string or a list of content blocks (e.g.
            ``{"type": "image_url", "image_url": {"url": ...}}``) for
            multimodal input.
        api_key: API key (``"EMPTY"`` for a local, unauthenticated vLLM).
        backend: Structured-output backend — ``xgrammar``, ``guidance``,
            ``outlines``, ``lm-format-enforcer``, or ``auto``.
        max_retries: Number of attempts if the output fails validation.
        temperature: Sampling temperature (``0.0`` for deterministic output).

    Returns:
        A validated ``.dclg`` XML string.

    Raises:
        DocLangValidationError: If no attempt produces valid DocLang.
    """
    client = _make_client(base_url, api_key)
    extra_body = {
        "structured_outputs": {"grammar": DOCLANG_GRAMMAR},
        "guided_decoding_backend": backend,
    }
    last_error: Exception | None = None
    for _ in range(max(1, max_retries)):
        resp = client.chat.completions.create(
            model=model,
            messages=list(messages),
            temperature=temperature,
            extra_body=extra_body,
        )
        raw = _content(resp)
        try:
            validate_doclang(raw)
            return raw
        except DocLangValidationError as exc:
            last_error = exc
    raise DocLangValidationError(
        f"failed to produce valid DocLang after {max_retries} attempt(s); "
        f"last error: {last_error}"
    )


def generate_doclang_json_schema(
    base_url: str,
    model: str,
    messages: Messages,
    *,
    api_key: str = "EMPTY",
    max_retries: int = 3,
    temperature: float = 0.0,
    strict: bool = False,
) -> str:
    """Generate a validated ``.dclg`` document via the JSON-schema fallback.

    Constrains the model to emit JSON matching the DocLang schema
    (``response_format``), parses it into :class:`DocLang`, serializes to
    ``.dclg``, and validates. Retries on failure.

    Args:
        base_url: OpenAI-compatible endpoint.
        model: Served model identifier.
        messages: Chat messages in OpenAI format (string or content blocks).
        api_key: API key (``"EMPTY"`` for a local server).
        max_retries: Number of attempts if the output fails validation.
        temperature: Sampling temperature.
        strict: Passed to :func:`response_format`; ``False`` by default because
            the DocLang schema uses a discriminated union, which is outside
            OpenAI's strict-mode subset.

    Returns:
        A validated ``.dclg`` XML string.

    Raises:
        DocLangValidationError: If no attempt produces valid DocLang.
    """
    client = _make_client(base_url, api_key)
    rf = response_format(strict=strict)
    last_error: Exception | None = None
    for _ in range(max(1, max_retries)):
        resp = client.chat.completions.create(
            model=model,
            messages=list(messages),
            temperature=temperature,
            response_format=rf,
        )
        raw = _content(resp)
        try:
            data = json.loads(raw)
            doc = DocLang.model_validate(data)
            xml = model_to_doclang_xml(doc)
            validate_doclang(xml)
            return xml
        except (json.JSONDecodeError, ValidationError, DocLangValidationError) as exc:
            last_error = exc
    raise DocLangValidationError(
        f"failed to produce valid DocLang after {max_retries} attempt(s); "
        f"last error: {last_error}"
    )
