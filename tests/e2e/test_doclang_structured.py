"""End-to-end structured-output tests against a live vLLM server.

These tests exercise both generation paths (grammar + JSON-schema) against a
real, running vLLM OpenAI-compatible endpoint. They are gated behind the
``e2e`` marker *and* skipped unless ``VOXDOC_VLLM_BASE_URL`` is set, so they
never run in normal CI.

To run them, start a vLLM server (e.g. ``vllm serve <model> --port 8000``) and
then::

    VOXDOC_VLLM_BASE_URL=http://localhost:8000/v1 \
    VOXDOC_VLLM_MODEL=<served-model-name> \
    python -m pytest tests/e2e/test_doclang_structured.py -m e2e -v

``VOXDOC_VLLM_MODEL`` defaults to ``default`` if the server serves a single
model under that name.
"""

from __future__ import annotations

import os

import pytest

from src.doclang_structured.client import (
    generate_doclang_grammar,
    generate_doclang_json_schema,
)
from src.doclang_structured.validate import validate_doclang

pytestmark = pytest.mark.e2e

BASE_URL = os.environ.get("VOXDOC_VLLM_BASE_URL")
MODEL = os.environ.get("VOXDOC_VLLM_MODEL", "default")

requires_server = pytest.mark.skipif(
    not BASE_URL, reason="VOXDOC_VLLM_BASE_URL not set"
)

_MESSAGES = [{"role": "user", "content": "Emit a short DocLang document."}]


@requires_server
def test_grammar_path_live() -> None:
    """The grammar path should return a validated ``.dclg`` document."""
    result = generate_doclang_grammar(BASE_URL, MODEL, _MESSAGES)
    assert result.lstrip().startswith("<doclang")
    # The client already validated it; re-validate to be certain.
    validate_doclang(result)


@requires_server
def test_json_schema_path_live() -> None:
    """The JSON-schema path should return a validated ``.dclg`` document."""
    result = generate_doclang_json_schema(BASE_URL, MODEL, _MESSAGES)
    assert result.lstrip().startswith("<doclang")
    validate_doclang(result)
