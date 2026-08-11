"""Unit tests for the data converters."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from tools.convert_data import to_dpo, to_kto, to_sft


def _make_image(tmp_path: Path) -> str:
    """Create a tiny test image and return its path."""
    image_path = tmp_path / "doc.png"
    Image.new("RGB", (8, 8), color="white").save(image_path)
    return str(image_path)


def test_to_sft(tmp_path: Path) -> None:
    """SFT conversion should produce prompt-completion records."""
    image = _make_image(tmp_path)
    rows = [{"image": image, "prompt": "Read this.", "completion": "Output."}]
    dataset = to_sft(rows)
    assert len(dataset) == 1
    record = dataset[0]
    assert record["prompt"][0]["role"] == "user"
    assert record["completion"][0]["role"] == "assistant"
    assert len(record["images"]) == 1


def test_to_dpo(tmp_path: Path) -> None:
    """DPO conversion should produce chosen/rejected records."""
    image = _make_image(tmp_path)
    rows = [{"image": image, "prompt": "Read this.", "chosen": "Good.", "rejected": "Bad."}]
    dataset = to_dpo(rows)
    record = dataset[0]
    assert record["chosen"][0]["role"] == "assistant"
    assert record["rejected"][0]["role"] == "assistant"


def test_to_kto(tmp_path: Path) -> None:
    """KTO conversion should produce completion/label records."""
    image = _make_image(tmp_path)
    rows = [{"image": image, "prompt": "Read this.", "completion": "Output.", "label": True}]
    dataset = to_kto(rows)
    record = dataset[0]
    assert record["label"] is True
