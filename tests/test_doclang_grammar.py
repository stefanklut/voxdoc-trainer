"""Behavioural tests for the EBNF grammar (via lark).

The canonical grammar is GBNF (for xgrammar / vLLM); it is unit-tested here by
rewriting it to lark's EBNF dialect (``to_lark_grammar``) and parsing with the
unambiguous earley parser.

Guarantees checked:

* the grammar **compiles**;
* it **accepts** every document in the valid corpus (it is not over-strict);
* it **rejects** every invalid document whose violation is CFG-expressible
  (the ``grammar+validator`` subset in the invalid manifest) — i.e. it is a
  *sound* filter for the rules a CFG can express.

Context-sensitive invalid documents (the ``validator`` subset) are *not*
required to be rejected here; a CFG cannot express them, and they are caught by
the final validation gate (see ``test_doclang_corpus``).
"""

from __future__ import annotations

import json
from pathlib import Path

import lark
import lark.exceptions
import pytest

from src.doclang_structured.grammar import to_lark_grammar

DATA = Path(__file__).parent / "data"
CORPUS_DIR = DATA / "doclang_corpus"
INVALID_DIR = DATA / "doclang_invalid"


@pytest.fixture(scope="module")
def parser():
    return lark.Lark(to_lark_grammar(), start="root", parser="earley")


def _corpus_files() -> list[Path]:
    return sorted(CORPUS_DIR.glob("*.dclg"))


def _invalid_manifest() -> dict:
    return json.loads((INVALID_DIR / "manifest.json").read_text(encoding="utf-8"))


def test_grammar_compiles(parser) -> None:
    """The GBNF grammar must rewrite to a lark grammar that compiles."""
    assert parser is not None


def test_grammar_accepts_valid_corpus(parser) -> None:
    """Every valid corpus document must be accepted (grammar not over-strict)."""
    files = _corpus_files()
    assert files, f"valid corpus is empty: {CORPUS_DIR}"
    for path in files:
        parser.parse(path.read_text(encoding="utf-8"))  # raises on rejection


def test_grammar_rejects_cfg_expressible_invalid(parser) -> None:
    """Every CFG-expressible invalid document must be rejected by the grammar."""
    manifest = _invalid_manifest()
    cfg_expressible = [
        fn for fn, info in manifest.items() if info["mechanism"] == "grammar+validator"
    ]
    assert cfg_expressible, "invalid manifest has no 'grammar+validator' documents"
    for fn in cfg_expressible:
        xml = (INVALID_DIR / fn).read_text(encoding="utf-8")
        with pytest.raises(lark.exceptions.LarkError):
            parser.parse(xml)
