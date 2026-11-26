"""Unit tests for the lightweight inference helper.

The model and processor factories are replaced with fakes, so these tests
exercise the request plumbing — which model class is chosen, how the processor
is called, device placement, and what gets decoded — without a GPU or weights.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image as PILImage

from tools import serve


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class _Inputs(dict):
    """Stand-in for a ``BatchFeature``: a mapping that can be moved to a device."""

    def __init__(self, mapping: dict) -> None:
        super().__init__(mapping)
        self.moved_to: str | None = None

    def to(self, device: str) -> "_Inputs":
        self.moved_to = device
        return self


class _Model:
    """Stand-in for a loaded model."""

    def __init__(self, prompt_length: int = 5, new_tokens: int = 3) -> None:
        self.device = "cuda:0"
        self.moved_to: str | None = None
        self.eval_called = False
        self.generate_kwargs: dict = {}
        self.grad_enabled_at_generate: bool | None = None
        self._prompt_length = prompt_length
        self._new_tokens = new_tokens

    def to(self, device: str) -> "_Model":
        self.moved_to = device
        return self

    def eval(self) -> "_Model":
        self.eval_called = True
        return self

    def generate(self, **kwargs) -> torch.Tensor:
        self.generate_kwargs = kwargs
        self.grad_enabled_at_generate = torch.is_grad_enabled()
        total = self._prompt_length + self._new_tokens
        return torch.arange(total, dtype=torch.long).unsqueeze(0)


class _Processor:
    """Stand-in for a processor: records how it was called and what was decoded."""

    def __init__(self, prompt_length: int = 5) -> None:
        self.calls: list[dict] = []
        self.inputs: list[_Inputs] = []
        self.decoded: list = []
        self._prompt_length = prompt_length

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        return f"<s>{messages[0]['content']}</s>"

    def __call__(self, **kwargs) -> _Inputs:
        self.calls.append(kwargs)
        inputs = _Inputs({"input_ids": torch.zeros(1, self._prompt_length, dtype=torch.long)})
        self.inputs.append(inputs)
        return inputs

    def decode(self, token_ids, skip_special_tokens=True) -> str:
        self.decoded.append(token_ids)
        return "generated answer"


class _Recorder:
    """Factory stand-in that records which model class was used."""

    def __init__(self, name: str) -> None:
        self.__name__ = name
        self.calls: list[tuple[str, dict]] = []
        self.model = _Model()

    def from_pretrained(self, model_id: str, **kwargs) -> _Model:
        self.calls.append((model_id, kwargs))
        return self.model


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Replace the model/processor factories and pin CUDA availability to True."""
    causal = _Recorder("AutoModelForCausalLM")
    vision = _Recorder("AutoModelForImageTextToText")
    processor = _Processor()
    monkeypatch.setattr(serve, "AutoModelForCausalLM", causal)
    monkeypatch.setattr(serve, "AutoModelForImageTextToText", vision)
    monkeypatch.setattr(serve, "AutoProcessor", _RecordingProcessor(processor))
    monkeypatch.setattr(serve.torch.cuda, "is_available", lambda: True)
    return {"causal": causal, "vision": vision, "processor": processor}


class _RecordingProcessor:
    """Factory for a shared :class:`_Processor` instance."""

    def __init__(self, processor: _Processor) -> None:
        self._processor = processor
        self.calls: list[str] = []

    def from_pretrained(self, model_id: str) -> _Processor:
        self.calls.append(model_id)
        return self._processor


def _make_image(tmp_path: Path, name: str = "scan.png", mode: str = "RGB") -> str:
    """Create a tiny test image and return its path."""
    image_path = tmp_path / name
    PILImage.new(mode, (8, 8), color="white").save(image_path)
    return str(image_path)


# --------------------------------------------------------------------------- #
# Model class selection
# --------------------------------------------------------------------------- #
def test_text_only_uses_causal_lm(fakes) -> None:
    serve.generate("some/model", "Read this document.")
    assert len(fakes["causal"].calls) == 1
    assert fakes["vision"].calls == []


def test_images_use_image_text_to_text(fakes, tmp_path) -> None:
    image = _make_image(tmp_path)
    serve.generate("some/model", "Transcribe this.", images=[image])
    assert len(fakes["vision"].calls) == 1
    assert fakes["causal"].calls == []


def test_model_and_inputs_are_placed_on_device(fakes, tmp_path) -> None:
    image = _make_image(tmp_path)
    serve.generate("some/model", "Transcribe this.", images=[image])
    model = fakes["vision"].model
    assert model.moved_to == "cuda"  # auto-detected device, not the processor's
    assert model.eval_called
    assert fakes["processor"].inputs[0].moved_to == model.device


def test_dtype_and_device_are_forwarded(fakes, monkeypatch) -> None:
    serve.generate("some/model", "hi", dtype="bfloat16", device="cpu")
    model_id, kwargs = fakes["causal"].calls[0]
    assert model_id == "some/model"
    assert kwargs == {"dtype": "bfloat16"}
    assert fakes["causal"].model.moved_to == "cpu"


# --------------------------------------------------------------------------- #
# Processor call
# --------------------------------------------------------------------------- #
def test_processor_receives_batched_text(fakes) -> None:
    serve.generate("some/model", "Read this document.")
    call = fakes["processor"].calls[0]
    assert isinstance(call["text"], list) and len(call["text"]) == 1
    assert "Read this document." in call["text"][0]
    assert call["return_tensors"] == "pt"


def test_processor_receives_no_images_keyword_for_text_only(fakes) -> None:
    serve.generate("some/model", "Read this document.")
    assert "images" not in fakes["processor"].calls[0]


def test_processor_receives_loaded_pil_images(fakes, tmp_path) -> None:
    first = _make_image(tmp_path, "a.png")
    second = _make_image(tmp_path, "b.png")
    serve.generate("some/model", "Compare these.", images=[first, second])
    images = fakes["processor"].calls[0]["images"]
    assert len(images) == 2
    assert all(isinstance(image, PILImage.Image) for image in images)
    assert [image.size for image in images] == [(8, 8), (8, 8)]


def test_grayscale_image_is_converted_to_rgb(fakes, tmp_path) -> None:
    image = _make_image(tmp_path, "gray.png", mode="L")
    serve.generate("some/model", "Transcribe this.", images=[image])
    assert fakes["processor"].calls[0]["images"][0].mode == "RGB"


def test_generate_respects_max_new_tokens(fakes) -> None:
    serve.generate("some/model", "hi", max_new_tokens=42)
    assert fakes["causal"].model.generate_kwargs["max_new_tokens"] == 42


def test_generate_runs_without_grad(fakes) -> None:
    serve.generate("some/model", "hi")
    assert fakes["causal"].model.grad_enabled_at_generate is False


# --------------------------------------------------------------------------- #
# Decoding
# --------------------------------------------------------------------------- #
def test_only_the_continuation_is_decoded(fakes) -> None:
    serve.generate("some/model", "Read this document.")
    decoded = fakes["processor"].decoded[0]
    # The fake model returns prompt_length + new_tokens ids; only the tail is decoded.
    assert len(decoded) == 3


def test_response_is_stripped(fakes, monkeypatch) -> None:
    monkeypatch.setattr(fakes["processor"], "decode", lambda token_ids, skip_special_tokens=True: "  padded  ")
    assert serve.generate("some/model", "hi") == "padded"


# --------------------------------------------------------------------------- #
# Image loading
# --------------------------------------------------------------------------- #
def test_load_images_rejects_urls(tmp_path) -> None:
    with pytest.raises(ValueError):
        serve.load_images(["https://example.com/scan.png"])


def test_load_images_rejects_missing_files(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        serve.load_images([str(tmp_path / "absent.png")])


def test_load_images_preserves_order(tmp_path) -> None:
    first = _make_image(tmp_path, "1.png")
    second = _make_image(tmp_path, "2.png")
    images = serve.load_images([first, second])
    assert [image.size for image in images] == [(8, 8), (8, 8)]


# --------------------------------------------------------------------------- #
# Device resolution
# --------------------------------------------------------------------------- #
def test_resolve_device_prefers_explicit_value(monkeypatch) -> None:
    monkeypatch.setattr(serve.torch.cuda, "is_available", lambda: True)
    assert serve.resolve_device("cpu") == "cpu"


def test_resolve_device_detects_cuda(monkeypatch) -> None:
    monkeypatch.setattr(serve.torch.cuda, "is_available", lambda: True)
    assert serve.resolve_device() == "cuda"


def test_resolve_device_falls_back_to_cpu(monkeypatch) -> None:
    monkeypatch.setattr(serve.torch.cuda, "is_available", lambda: False)
    assert serve.resolve_device() == "cpu"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_image_flag_is_repeatable(monkeypatch, tmp_path) -> None:
    captured: dict = {}

    def fake_generate(model_id, prompt, max_new_tokens=256, images=None, dtype="auto", device=None) -> str:
        captured.update(
            model_id=model_id, prompt=prompt, max_new_tokens=max_new_tokens, images=images, dtype=dtype, device=device
        )
        return "ok"

    monkeypatch.setattr(serve, "generate", fake_generate)
    monkeypatch.setattr(
        "sys.argv",
        [
            "serve.py",
            "--model",
            "outputs/merged",
            "--prompt",
            "Transcribe this.",
            "--image",
            _make_image(tmp_path, "a.png"),
            "--image",
            _make_image(tmp_path, "b.png"),
            "--dtype",
            "bfloat16",
            "--device",
            "cpu",
        ],
    )
    serve.main()
    assert captured["model_id"] == "outputs/merged"
    assert len(captured["images"]) == 2
    assert captured["dtype"] == "bfloat16"
    assert captured["device"] == "cpu"


def test_image_flag_defaults_to_none(monkeypatch) -> None:
    captured: dict = {}

    def fake_generate(model_id, prompt, max_new_tokens=256, images=None, dtype="auto", device=None) -> str:
        captured.update(images=images, dtype=dtype, device=device)
        return "ok"

    monkeypatch.setattr(serve, "generate", fake_generate)
    monkeypatch.setattr("sys.argv", ["serve.py", "--model", "m", "--prompt", "p"])
    serve.main()
    assert captured["images"] is None
    assert captured["dtype"] == "auto"
    assert captured["device"] is None
