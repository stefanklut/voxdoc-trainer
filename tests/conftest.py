"""Shared pytest fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image


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
