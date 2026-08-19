"""Unit tests for the data converters."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from tools.convert_data import to_dpo, to_grpo, to_kto, to_sft


def _make_image(tmp_path: Path, name: str = "doc.png") -> str:
    """Create a tiny test image and return its path."""
    image_path = tmp_path / name
    Image.new("RGB", (8, 8), color="white").save(image_path)
    return str(image_path)


def _history(image: str | None = None) -> list[dict]:
    """Build a two-turn message history, optionally with an image in turn one."""
    first = {"role": "user", "content": "Read this document."}
    if image:
        first = {
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": "Read this document."}],
        }
    return [
        first,
        {"role": "assistant", "content": "I found an uncertain section."},
        {"role": "user", "content": "Explain that section."},
    ]


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


def test_to_grpo(tmp_path: Path) -> None:
    """GRPO conversion should produce prompt-only records."""
    image = _make_image(tmp_path)
    rows = [{"image": image, "prompt": "Read this."}]
    dataset = to_grpo(rows)
    record = dataset[0]
    assert record["prompt"][0]["role"] == "user"
    assert len(record["images"]) == 1


def test_to_sft_multi_turn(tmp_path: Path) -> None:
    """SFT should preserve a multi-turn history in the prompt."""
    image = _make_image(tmp_path)
    rows = [{"messages": _history(image), "completion": "The section describes..."}]
    dataset = to_sft(rows)
    record = dataset[0]
    assert [m["role"] for m in record["prompt"]] == ["user", "assistant", "user"]
    assert record["completion"][0]["role"] == "assistant"
    assert len(record["images"]) == 1


def test_to_dpo_multi_turn(tmp_path: Path) -> None:
    """DPO should keep a shared multi-turn prompt with two alternatives."""
    image = _make_image(tmp_path)
    rows = [
        {
            "messages": _history(image),
            "chosen": "The section describes...",
            "rejected": "I cannot determine anything.",
        }
    ]
    dataset = to_dpo(rows)
    record = dataset[0]
    assert [m["role"] for m in record["prompt"]] == ["user", "assistant", "user"]
    assert record["chosen"][0]["role"] == "assistant"
    assert record["rejected"][0]["role"] == "assistant"
    assert len(record["images"]) == 1


def test_to_kto_multi_turn(tmp_path: Path) -> None:
    """KTO should preserve a multi-turn history with both label values."""
    image = _make_image(tmp_path)
    rows = [
        {"messages": _history(image), "completion": "Good.", "label": True},
        {"messages": _history(image), "completion": "Bad.", "label": False},
    ]
    dataset = to_kto(rows)
    assert [m["role"] for m in dataset[0]["prompt"]] == ["user", "assistant", "user"]
    assert dataset[0]["label"] is True
    assert dataset[1]["label"] is False
    assert len(dataset[0]["images"]) == 1


def test_to_grpo_multi_turn(tmp_path: Path) -> None:
    """GRPO should use a multi-turn history as its generation prompt."""
    image = _make_image(tmp_path)
    rows = [{"messages": _history(image)}]
    dataset = to_grpo(rows)
    record = dataset[0]
    assert [m["role"] for m in record["prompt"]] == ["user", "assistant", "user"]
    assert len(record["images"]) == 1


def test_legacy_single_turn_matches_messages(tmp_path: Path) -> None:
    """Legacy prompt rows should produce the same structure as messages rows."""
    image = _make_image(tmp_path)
    legacy = to_sft([{"image": image, "prompt": "Read this.", "completion": "Output."}])
    structured = to_sft(
        [
            {
                "messages": [
                    {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "Read this."}]}
                ],
                "completion": "Output.",
            }
        ]
    )
    assert legacy[0]["prompt"] == structured[0]["prompt"]
    assert legacy[0]["completion"] == structured[0]["completion"]
    assert len(legacy[0]["images"]) == len(structured[0]["images"]) == 1


def test_multiple_images_ordering(tmp_path: Path) -> None:
    """Multiple images should be collected in placeholder order."""
    image_a = _make_image(tmp_path, "a.png")
    image_b = _make_image(tmp_path, "b.png")
    rows = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_a},
                        {"type": "text", "text": "First."},
                        {"type": "image", "image": image_b},
                    ],
                }
            ],
            "completion": "Output.",
        }
    ]
    dataset = to_sft(rows)
    record = dataset[0]
    content = record["prompt"][0]["content"]
    assert [b["type"] for b in content] == ["image", "text", "image"]
    assert len(record["images"]) == 2


def test_missing_messages_raises() -> None:
    """A row with neither messages nor prompt should raise."""
    with pytest.raises(ValueError, match="messages.*prompt"):
        to_sft([{"completion": "Output."}])


def test_empty_messages_raises() -> None:
    """An empty messages history should raise."""
    with pytest.raises(ValueError, match="empty"):
        to_sft([{"messages": [], "completion": "Output."}])


def test_invalid_role_raises() -> None:
    """An unknown message role should raise."""
    rows = [{"messages": [{"role": "robot", "content": "Hi"}], "completion": "Output."}]
    with pytest.raises(ValueError, match="invalid role"):
        to_sft(rows)


def test_history_ending_with_assistant_raises() -> None:
    """A history ending with an assistant message should raise."""
    rows = [
        {
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ],
            "completion": "Output.",
        }
    ]
    with pytest.raises(ValueError, match="final message"):
        to_sft(rows)


def test_missing_completion_raises() -> None:
    """SFT rows missing a completion should raise."""
    with pytest.raises(ValueError, match="completion"):
        to_sft([{"messages": [{"role": "user", "content": "Hi"}]}])


def test_missing_dpo_fields_raises() -> None:
    """DPO rows missing chosen/rejected should raise."""
    with pytest.raises(ValueError, match="chosen.*rejected"):
        to_dpo([{"messages": [{"role": "user", "content": "Hi"}], "chosen": "Good."}])


def test_invalid_kto_label_raises() -> None:
    """KTO rows with a non-boolean label should raise."""
    rows = [{"messages": [{"role": "user", "content": "Hi"}], "completion": "Out.", "label": "yes"}]
    with pytest.raises(ValueError, match="boolean"):
        to_kto(rows)


def test_missing_image_file_raises(tmp_path: Path) -> None:
    """A missing image file should raise when opened."""
    rows = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "image", "image": str(tmp_path / "nope.png")}, {"type": "text", "text": "Hi"}],
                }
            ],
            "completion": "Output.",
        }
    ]
    with pytest.raises(FileNotFoundError):
        to_sft(rows)
