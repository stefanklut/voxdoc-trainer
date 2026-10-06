"""Unit tests for the OpenAI-compatible generation clients.

These tests exercise the *client control flow* — what is sent on the wire, the
retry behaviour, and error handling — by mocking the OpenAI client
(``_make_client``) and stubbing the validator (``validate_doclang``) so the
tests are fast and deterministic. Real end-to-end validation against a live
vLLM server is covered by the ``e2e`` tests.

The stub validator treats any output containing ``<bogus>`` as invalid and
everything else as valid, which is enough to drive the retry / success paths.
"""

from __future__ import annotations

import json

import pytest

from src.doclang_structured import client
from src.doclang_structured.client import (
    generate_doclang_grammar,
    generate_doclang_json_schema,
)
from src.doclang_structured.grammar import DOCLANG_GRAMMAR
from src.doclang_structured.validate import DocLangValidationError

VALID = '<doclang xmlns="https://www.doclang.ai/ns/v0" version="0.7"><text>hello</text></doclang>'
INVALID = '<doclang xmlns="https://www.doclang.ai/ns/v0" version="0.7"><bogus>x</bogus></doclang>'
VALID_JSON = json.dumps(
    {"body": [{"type": "text", "body": [{"type": "text_chunk", "text": "hi"}]}]}
)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class _Msg:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = _Msg(content)


class _Resp:
    def __init__(self, content: str) -> None:
        self.choices = [_Choice(content)]


class _Completions:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kw) -> _Resp:
        self.calls.append(kw)
        return _Resp(self._responses.pop(0))


class _Chat:
    def __init__(self, completions: _Completions) -> None:
        self.completions = completions


class _Client:
    def __init__(self, completions: _Completions) -> None:
        self.chat = _Chat(completions)


@pytest.fixture
def mock(monkeypatch: pytest.MonkeyPatch):
    """Install a fake OpenAI client and a stub validator; return an installer."""

    def _install(responses: list[str]) -> _Completions:
        completions = _Completions(responses)
        monkeypatch.setattr(
            client, "_make_client", lambda base_url, api_key: _Client(completions)
        )
        return completions

    def _fake_validate(xml: str, *, xsd_only: bool = False) -> None:
        if "<bogus>" in xml:
            raise DocLangValidationError("invalid: bogus element")

    monkeypatch.setattr(client, "validate_doclang", _fake_validate)
    return _install


# --------------------------------------------------------------------------- #
# Grammar path
# --------------------------------------------------------------------------- #
def test_grammar_path_sends_structured_outputs(mock) -> None:
    completions = mock([VALID])
    result = generate_doclang_grammar(
        "http://x/v1", "model", [{"role": "user", "content": "hi"}], backend="xgrammar"
    )
    assert result == VALID
    assert len(completions.calls) == 1  # no retry
    extra = completions.calls[0]["extra_body"]
    assert extra["structured_outputs"]["grammar"] == DOCLANG_GRAMMAR
    assert extra["guided_decoding_backend"] == "xgrammar"
    # the grammar path must NOT use response_format
    assert "response_format" not in completions.calls[0]


def test_grammar_retry_then_success(mock) -> None:
    completions = mock([INVALID, VALID])
    result = generate_doclang_grammar(
        "http://x/v1", "model", [{"role": "user", "content": "hi"}], max_retries=3
    )
    assert result == VALID
    assert len(completions.calls) == 2  # first failed, second succeeded


def test_grammar_exhausts_retries(mock) -> None:
    completions = mock([INVALID, INVALID, INVALID])
    with pytest.raises(DocLangValidationError):
        generate_doclang_grammar(
            "http://x/v1", "model", [{"role": "user", "content": "hi"}], max_retries=3
        )
    assert len(completions.calls) == 3


def test_grammar_empty_content_raises(mock) -> None:
    completions = mock([""])
    with pytest.raises(DocLangValidationError):
        generate_doclang_grammar(
            "http://x/v1", "model", [{"role": "user", "content": "hi"}], max_retries=3
        )
    # empty content fails fast (before the validation try/except), no retry
    assert len(completions.calls) == 1


def test_grammar_image_content_blocks_pass_through(mock) -> None:
    completions = mock([VALID])
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe this"},
                {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
            ],
        }
    ]
    result = generate_doclang_grammar("http://x/v1", "model", messages)
    assert result == VALID
    assert completions.calls[0]["messages"] == messages


# --------------------------------------------------------------------------- #
# JSON-schema path
# --------------------------------------------------------------------------- #
def test_json_schema_path_sends_response_format(mock) -> None:
    completions = mock([VALID_JSON])
    result = generate_doclang_json_schema(
        "http://x/v1", "model", [{"role": "user", "content": "hi"}]
    )
    assert result.startswith("<doclang")  # serialized .dclg, not raw JSON
    assert len(completions.calls) == 1
    assert "response_format" in completions.calls[0]
    # the JSON-schema path must NOT send a grammar
    assert "extra_body" not in completions.calls[0]


def test_json_schema_retry_then_success(mock) -> None:
    completions = mock(["this is not json", VALID_JSON])
    result = generate_doclang_json_schema(
        "http://x/v1", "model", [{"role": "user", "content": "hi"}], max_retries=3
    )
    assert result.startswith("<doclang")
    assert len(completions.calls) == 2


def test_json_schema_exhausts_retries(mock) -> None:
    completions = mock(["bad", "still bad"])
    with pytest.raises(DocLangValidationError):
        generate_doclang_json_schema(
            "http://x/v1", "model", [{"role": "user", "content": "hi"}], max_retries=2
        )
    assert len(completions.calls) == 2
