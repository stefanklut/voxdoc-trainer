"""Shared pytest fixtures."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
GENERATORS_DIR = TESTS_DIR / "generators"
CORPUS_DIR = TESTS_DIR / "data" / "doclang_corpus"
INVALID_DIR = TESTS_DIR / "data" / "doclang_invalid"


def _needs_generation(directory: Path) -> bool:
    """Return ``True`` if a corpus dir is missing or holds no ``.dclg`` docs."""
    return not directory.is_dir() or not any(directory.glob("*.dclg"))


def _run_builder(script: Path) -> None:
    """Run a corpus builder script, raising with its output on failure."""
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"DocLang corpus generation failed ({script.name}, exit {result.returncode}):\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )


def pytest_configure(config: pytest.Config) -> None:
    """Regenerate the gitignored DocLang corpora before collection.

    The corpora under ``tests/data/doclang_{corpus,invalid}/`` are gitignored
    build artifacts produced by ``tests/generators/build_doclang_{corpus,invalid}.py``.
    ``test_doclang_fuzz`` loads its corpus at import time, so the data must exist
    before collection — hence this hook rather than a session fixture. No-op when
    the data is already present.
    """
    if _needs_generation(CORPUS_DIR):
        _run_builder(GENERATORS_DIR / "build_doclang_corpus.py")
    if _needs_generation(INVALID_DIR):
        _run_builder(GENERATORS_DIR / "build_doclang_invalid.py")


@pytest.fixture
def tiny_image(tmp_path: Path) -> Path:
    """Create a tiny test image."""
    image_path = tmp_path / "doc.png"
    Image.new("RGB", (8, 8), color="white").save(image_path)
    return image_path


@pytest.fixture
def sft_jsonl(tmp_path: Path, tiny_image: Path) -> Path:
    """Create a tiny SFT JSONL file."""
    rows = [
        {"image": str(tiny_image), "prompt": "Read this document.", "completion": "The text is here."},
        {"image": str(tiny_image), "prompt": "Read this document.", "completion": "More text."},
    ]
    path = tmp_path / "sft.jsonl"
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path
