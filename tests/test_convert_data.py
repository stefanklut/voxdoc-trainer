"""Unit tests for the data converters."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

from tools.convert_data import load_rows, to_dpo, to_grpo, to_kto, to_sft
from utils.discovery import DiscoveryError


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
    """GRPO conversion should carry the ground truth for the reward."""
    image = _make_image(tmp_path)
    rows = [{"image": image, "prompt": "Read this.", "ground_truth": "A letter."}]
    dataset = to_grpo(rows)
    record = dataset[0]
    assert record["prompt"][0]["role"] == "user"
    assert len(record["images"]) == 1
    assert record["ground_truth"] == "A letter."


def test_to_grpo_requires_ground_truth(tmp_path: Path) -> None:
    """GRPO rows without a ground truth should be rejected."""
    image = _make_image(tmp_path)
    with pytest.raises(ValueError, match="ground_truth"):
        to_grpo([{"image": image, "prompt": "Read this."}])


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
    rows = [{"messages": _history(image), "ground_truth": "A marginal note."}]
    dataset = to_grpo(rows)
    record = dataset[0]
    assert [m["role"] for m in record["prompt"]] == ["user", "assistant", "user"]
    assert len(record["images"]) == 1
    assert record["ground_truth"] == "A marginal note."


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


def test_to_sft_embed_resolves_relative_image_against_roots(tmp_path: Path) -> None:
    """Embed mode resolves a row's relative image path against the JSONL directories."""
    corpus = tmp_path / "corpus"
    (corpus / "document").mkdir(parents=True)
    Image.new("RGB", (8, 8), color="white").save(corpus / "document" / "doc.png")
    rows = [{"image": "document/doc.png", "prompt": "Read this.", "completion": "Output."}]
    dataset = to_sft(rows, "embed", [corpus])
    image = dataset[0]["images"][0]
    assert image.size == (8, 8)


def test_to_sft_embed_missing_image_names_searched_dirs(tmp_path: Path) -> None:
    """A row whose image cannot be found reports the roots that were tried."""
    rows = [{"image": "missing.png", "prompt": "Read this.", "completion": "Output."}]
    with pytest.raises(FileNotFoundError, match=str(tmp_path / "corpus")):
        to_sft(rows, "embed", [tmp_path / "corpus"])


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


# --------------------------------------------------------------------------- #
# load_rows — file and directory inputs
# --------------------------------------------------------------------------- #


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write rows as JSONL, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_load_rows_single_file(tmp_path: Path) -> None:
    """A single JSONL file loads its rows."""
    path = tmp_path / "raw.jsonl"
    _write_jsonl(path, [{"a": 1}, {"a": 2}])
    assert load_rows(path) == [{"a": 1}, {"a": 2}]


def test_load_rows_expands_directory(tmp_path: Path) -> None:
    """A directory input contributes its .jsonl files in sorted order."""
    _write_jsonl(tmp_path / "b.jsonl", [{"src": "b"}])
    _write_jsonl(tmp_path / "a.jsonl", [{"src": "a"}])
    (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")
    assert load_rows(tmp_path) == [{"src": "a"}, {"src": "b"}]


def test_load_rows_merges_files_and_directories(tmp_path: Path) -> None:
    """Files and directories can be mixed in one invocation."""
    _write_jsonl(tmp_path / "corpus" / "a.jsonl", [{"src": "a"}])
    extra = tmp_path / "extra.jsonl"
    _write_jsonl(extra, [{"src": "extra"}])
    assert load_rows([tmp_path / "corpus", extra]) == [{"src": "a"}, {"src": "extra"}]


def test_load_rows_raises_when_no_jsonl_found(tmp_path: Path) -> None:
    """An input without any JSONL is reported rather than silently empty."""
    (tmp_path / "notes.txt").write_text("nope", encoding="utf-8")
    with pytest.raises(DiscoveryError):
        load_rows(tmp_path)


def test_load_rows_raises_for_missing_input(tmp_path: Path) -> None:
    """A nonexistent input path is reported."""
    with pytest.raises(FileNotFoundError):
        load_rows(tmp_path / "missing.jsonl")


def test_to_sft_reference_mode_stores_relative_path(tmp_path: Path) -> None:
    """Reference mode stores the image path relative to the JSONL's directory."""
    image = _make_image(tmp_path)
    rows = [{"image": image, "prompt": "Read this.", "completion": "Output."}]
    dataset = to_sft(rows, "reference", [tmp_path])
    assert dataset[0]["images"] == ["doc.png"]


def test_to_dpo_reference_mode_stores_relative_path(tmp_path: Path) -> None:
    """DPO conversion keeps stored paths relative in reference mode."""
    image = _make_image(tmp_path)
    rows = [{"image": image, "prompt": "Read this.", "chosen": "Good.", "rejected": "Bad."}]
    dataset = to_dpo(rows, "reference", [tmp_path])
    assert dataset[0]["images"] == ["doc.png"]


def test_to_kto_reference_mode_stores_relative_path(tmp_path: Path) -> None:
    """KTO conversion keeps stored paths relative in reference mode."""
    image = _make_image(tmp_path)
    rows = [{"image": image, "prompt": "Read this.", "completion": "Output.", "label": True}]
    dataset = to_kto(rows, "reference", [tmp_path])
    assert dataset[0]["images"] == ["doc.png"]


def test_to_grpo_reference_mode_stores_relative_path(tmp_path: Path) -> None:
    """GRPO conversion keeps stored paths relative in reference mode."""
    image = _make_image(tmp_path)
    rows = [{"image": image, "prompt": "Read this.", "ground_truth": "Output."}]
    dataset = to_grpo(rows, "reference", [tmp_path])
    assert dataset[0]["images"] == ["doc.png"]


def test_to_sft_multi_turn_reference_mode(tmp_path: Path) -> None:
    """Images inside a structured message history are stored as paths too."""
    image = _make_image(tmp_path)
    rows = [{"messages": _history(image), "completion": "It is an invoice."}]
    dataset = to_sft(rows, "reference", [tmp_path])
    assert dataset[0]["images"] == ["doc.png"]
    assert dataset[0]["prompt"][0]["content"][0]["type"] == "image"


def test_to_sft_embed_mode_unchanged(tmp_path: Path) -> None:
    """The default mode still embeds decoded images, so existing datasets are unaffected."""
    image = _make_image(tmp_path)
    dataset = to_sft([{"image": image, "prompt": "Read this.", "completion": "Output."}])
    assert isinstance(dataset[0]["images"][0], Image.Image)


def test_image_outside_roots_stays_absolute(tmp_path: Path) -> None:
    """An image outside every root is stored absolute rather than guessed at."""
    (tmp_path / "elsewhere").mkdir()
    image = _make_image(tmp_path / "elsewhere", "doc.png")
    dataset = to_sft([{"image": image, "prompt": "Read this.", "completion": "Output."}], "reference", [tmp_path / "data"])
    assert dataset[0]["images"] == [str(Path(image).resolve())]


def test_to_sft_records_returns_plain_records(tmp_path: Path) -> None:
    """The record-level converter returns dicts, letting callers choose how to save."""
    from tools.convert_data import to_sft_records

    image = _make_image(tmp_path)
    records = to_sft_records([{"image": image, "prompt": "Read this.", "completion": "Output."}], "reference", [tmp_path])
    assert records == [
        {
            "prompt": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Read this."}]}],
            "completion": [{"role": "assistant", "content": [{"type": "text", "text": "Output."}]}],
            "images": ["doc.png"],
        }
    ]


def test_main_writes_manifest_for_reference_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI records the image mode and the JSONL directory next to the dataset."""
    from tools import convert_data

    image = _make_image(tmp_path)
    _write_jsonl(tmp_path / "raw.jsonl", [{"image": image, "prompt": "Read this.", "completion": "Output."}])
    output = tmp_path / "dataset"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "convert_data.py",
            "--input",
            str(tmp_path / "raw.jsonl"),
            "--output",
            str(output),
            "--method",
            "sft",
            "--image-mode",
            "reference",
        ],
    )

    convert_data.main()

    manifest = json.loads((output / "image_manifest.json").read_text(encoding="utf-8"))
    assert manifest["image_mode"] == "reference"
    assert manifest["roots"] == [str(tmp_path.resolve())]


def test_main_omits_manifest_in_embed_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The default CLI mode saves exactly what it saved before, with no sidecar."""
    from tools import convert_data

    image = _make_image(tmp_path)
    _write_jsonl(tmp_path / "raw.jsonl", [{"image": image, "prompt": "Read this.", "completion": "Output."}])
    output = tmp_path / "dataset"
    monkeypatch.setattr(
        sys, "argv", ["convert_data.py", "--input", str(tmp_path / "raw.jsonl"), "--output", str(output), "--method", "sft"]
    )

    convert_data.main()

    assert not (output / "image_manifest.json").exists()
