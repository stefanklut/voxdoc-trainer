"""Tests for :mod:`tools.merge_adapter`.

The merge tool used to load every adapter onto ``AutoModelForCausalLM``, which for a
vision-language adapter silently discarded the vision-tower tensors -- PEFT warns, it
does not raise -- and still saved a model that looked like a successful merge.
"""

from __future__ import annotations

import warnings

import pytest

from tools.merge_adapter import _MISSING_KEYS_RE, load_adapter, resolve_model_class


class _FakeConfig:
    """Stand-in for a transformers config; only the attributes that decide the class."""

    def __init__(self, vision_config=None, image_token_id=None):
        if vision_config is not None:
            self.vision_config = vision_config
        if image_token_id is not None:
            self.image_token_id = image_token_id


@pytest.fixture
def fake_config(monkeypatch):
    """Make ``resolve_model_class`` read a config we hand it."""
    import tools.merge_adapter as module

    def _set(config):
        monkeypatch.setattr(module.AutoConfig, "from_pretrained", staticmethod(lambda _model: config))
        return config

    return _set


class TestResolveModelClass:
    def test_vision_config_selects_image_text_to_text(self, fake_config):
        from transformers import AutoModelForImageTextToText

        fake_config(_FakeConfig(vision_config={"hidden_size": 1}))
        assert resolve_model_class("any/model") is AutoModelForImageTextToText

    def test_text_only_config_selects_causal_lm(self, fake_config):
        from transformers import AutoModelForCausalLM

        fake_config(_FakeConfig())
        assert resolve_model_class("any/model") is AutoModelForCausalLM

    def test_image_token_id_also_selects_vlm(self, fake_config):
        from transformers import AutoModelForImageTextToText

        fake_config(_FakeConfig(image_token_id=151655))
        assert resolve_model_class("any/model") is AutoModelForImageTextToText

    def test_none_vision_config_selects_causal_lm(self, fake_config):
        """A config that declares ``vision_config = None`` is still text-only."""
        from transformers import AutoModelForCausalLM

        class _Cfg:
            vision_config = None
            image_token_id = None

        fake_config(_Cfg())
        assert resolve_model_class("any/model") is AutoModelForCausalLM


class TestLoadAdapter:
    def test_missing_keys_raise(self, monkeypatch):
        import tools.merge_adapter as module

        keys = ["base_model.model.model.layers.0.linear_attn.out_proj.lora_A.default.weight"]

        def _fake_from_pretrained(_model, _path):
            warnings.warn(f"Found missing adapter keys while loading the checkpoint: {keys}.", UserWarning)
            return "loaded"

        monkeypatch.setattr(module.PeftModel, "from_pretrained", staticmethod(_fake_from_pretrained))
        with pytest.raises(RuntimeError, match="had no matching module"):
            load_adapter(object(), "/adapter")

    def test_error_reports_the_count_and_an_example(self, monkeypatch):
        import tools.merge_adapter as module

        keys = [f"base_model.model.model.layers.{i}.visual.blocks.0.attn.qkv.lora_A.default.weight" for i in range(3)]

        def _fake_from_pretrained(_model, _path):
            warnings.warn(f"Found missing adapter keys while loading the checkpoint: {keys}.", UserWarning)
            return "loaded"

        monkeypatch.setattr(module.PeftModel, "from_pretrained", staticmethod(_fake_from_pretrained))
        with pytest.raises(RuntimeError) as excinfo:
            load_adapter(object(), "/adapter")
        message = str(excinfo.value)
        assert "3 of its tensors" in message
        assert keys[0] in message

    def test_clean_load_returns_model(self, monkeypatch):
        import tools.merge_adapter as module

        def _fake_from_pretrained(_model, _path):
            return "loaded"

        monkeypatch.setattr(module.PeftModel, "from_pretrained", staticmethod(_fake_from_pretrained))
        assert load_adapter(object(), "/adapter") == "loaded"

    def test_unrelated_warnings_do_not_raise(self, monkeypatch):
        import tools.merge_adapter as module

        def _fake_from_pretrained(_model, _path):
            warnings.warn("Some other PEFT notice.", UserWarning)
            return "loaded"

        monkeypatch.setattr(module.PeftModel, "from_pretrained", staticmethod(_fake_from_pretrained))
        assert load_adapter(object(), "/adapter") == "loaded"

    def test_regex_matches_peft_wording(self):
        """Guard against PEFT rewording the message out from under the matcher."""
        assert _MISSING_KEYS_RE.search("Found missing adapter keys while loading the checkpoint: ['a.lora_A.weight'].")
