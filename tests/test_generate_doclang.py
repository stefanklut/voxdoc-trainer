"""Unit tests for the DocLang generation CLI, including directory batch mode."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
from PIL import Image

from tools import generate_doclang


def _make_image(path: Path) -> None:
    """Create a tiny test image at the given path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color="white").save(path)


def _args(**overrides) -> argparse.Namespace:
    """Build a parsed-args namespace with CLI defaults, overridden as needed."""
    defaults = dict(
        base_url="http://localhost:8000/v1",
        model="mymodel",
        prompt="Transcribe this document as DocLang.",
        image=None,
        out=None,
        mode="grammar",
        backend="xgrammar",
        max_retries=3,
        temperature=0.0,
        api_key="EMPTY",
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# --------------------------------------------------------------------------- #
# Single-image mode
# --------------------------------------------------------------------------- #


def test_single_image_writes_out_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A single image path produces one .dclg file."""
    image = tmp_path / "scan.png"
    _make_image(image)
    out = tmp_path / "out" / "scan.dclg"
    calls: list[str] = []

    def fake_generate(base_url, model, messages, args):
        calls.append(str(messages[0]["content"][0]["image_url"]["url"])[:5])
        return "<doclang>ok</doclang>"

    monkeypatch.setattr(generate_doclang, "_generate", fake_generate)
    monkeypatch.setattr(
        generate_doclang.sys, "argv", ["prog", "--base-url", "u", "--model", "m", "--image", str(image), "--out", str(out)]
    )

    generate_doclang.main()

    assert calls == ["data:"]
    assert out.read_text(encoding="utf-8") == "<doclang>ok</doclang>\n"


def test_single_image_without_out_prints_to_stdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without --out the document is printed on stdout."""
    monkeypatch.setattr(generate_doclang, "_generate", lambda *_, **__: "<doclang>ok</doclang>")
    monkeypatch.setattr(generate_doclang.sys, "argv", ["prog", "--base-url", "u", "--model", "m"])

    generate_doclang.main()

    assert capsys.readouterr().out.strip() == "<doclang>ok</doclang>"


# --------------------------------------------------------------------------- #
# Batch mode
# --------------------------------------------------------------------------- #


def test_batch_mode_writes_one_dclg_per_image(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A directory --image generates one <stem>.dclg per image into --out."""
    images = tmp_path / "scans"
    _make_image(images / "a.png")
    _make_image(images / "b.jpg")
    (images / "notes.txt").write_text("not an image", encoding="utf-8")
    out_dir = tmp_path / "doclang"
    seen: list[str] = []

    def fake_generate(base_url, model, messages, args):
        seen.append(messages[0]["content"][0]["image_url"]["url"])
        return f"<doclang>{len(seen)}</doclang>"

    monkeypatch.setattr(generate_doclang, "_generate", fake_generate)
    monkeypatch.setattr(
        generate_doclang.sys,
        "argv",
        ["prog", "--base-url", "u", "--model", "m", "--image", str(images), "--out", str(out_dir)],
    )

    generate_doclang.main()

    assert sorted(p.name for p in out_dir.iterdir()) == ["a.dclg", "b.dclg"]
    assert (out_dir / "a.dclg").read_text(encoding="utf-8") == "<doclang>1</doclang>\n"
    assert all(url.startswith("data:") for url in seen)


def test_batch_mode_continues_after_failure_and_exits_nonzero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing image is reported and skipped; the run exits non-zero."""
    images = tmp_path / "scans"
    _make_image(images / "a.png")
    _make_image(images / "b.png")
    out_dir = tmp_path / "doclang"

    def fake_generate(base_url, model, messages, args):
        if messages[0]["content"][0]["image_url"]["url"] == "FAIL":
            raise RuntimeError("boom")
        return "<doclang>ok</doclang>"

    monkeypatch.setattr(generate_doclang, "_generate", fake_generate)
    monkeypatch.setattr(
        generate_doclang,
        "build_messages",
        lambda prompt, image: [
            {
                "role": "user",
                "content": [{"image_url": {"url": "FAIL" if "b." in image else "data:x"}}, {"type": "text", "text": prompt}],
            }
        ],
    )
    monkeypatch.setattr(
        generate_doclang.sys,
        "argv",
        ["prog", "--base-url", "u", "--model", "m", "--image", str(images), "--out", str(out_dir)],
    )

    with pytest.raises(SystemExit) as exc_info:
        generate_doclang.main()

    assert exc_info.value.code == 1
    assert [p.name for p in out_dir.iterdir()] == ["a.dclg"]


def test_batch_mode_requires_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Batch mode without --out is a usage error."""
    images = tmp_path / "scans"
    _make_image(images / "a.png")
    monkeypatch.setattr(generate_doclang.sys, "argv", ["prog", "--base-url", "u", "--model", "m", "--image", str(images)])

    with pytest.raises(SystemExit) as exc_info:
        generate_doclang.main()

    assert exc_info.value.code == 2


def test_batch_mode_empty_directory_exits_nonzero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A directory without images is reported as a failure."""
    images = tmp_path / "scans"
    images.mkdir()
    monkeypatch.setattr(
        generate_doclang.sys,
        "argv",
        ["prog", "--base-url", "u", "--model", "m", "--image", str(images), "--out", str(tmp_path / "out")],
    )

    with pytest.raises(SystemExit) as exc_info:
        generate_doclang.main()

    assert exc_info.value.code == 1
