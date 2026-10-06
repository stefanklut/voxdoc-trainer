# voxdoc-trainer

Training LLMs for Historical Document Understanding.

A production-quality, model-agnostic codebase for fine-tuning vision-language LLMs (Qwen3.5-9B and newer) on historical document understanding data, built on Hugging Face TRL + PEFT (LoRA/QLoRA) + accelerate.

## Training methods

| Data type | Method | Trainer |
| --- | --- | --- |
| Corrections to LLM answers | SFT | `SFTTrainer` |
| A/B comparisons | DPO | `DPOTrainer` |
| Good/bad feedback | KTO | `KTOTrainer` |
| Feedback / reward signals | GRPO (online RL) | `GRPOTrainer` |

## Quick start

See [docs/setup.md](docs/setup.md) for full setup instructions.

```bash
mamba env create -f environment.yml
mamba activate voxdoc-trainer
```

## Project layout

- `src/` — training entry points and config
  - `src/config.py` — model-agnostic config system (`RunConfig`, `ModelConfig`, `DataConfig`, `TrainingConfig`, `GRPOConfig`)
  - `src/peft_utils.py` — LoRA/QLoRA config builders
  - `src/train_sft.py`, `src/train_dpo.py`, `src/train_kto.py`, `src/train_grpo.py` — TRL trainers
  - `src/doclang_structured/` — constrained DocLang (`.dclg`) generation: EBNF grammar + Pydantic/JSON-schema fallback, validator-gated
- `configs/` — example YAML configs for each method (chained pipeline + dry-run)
- `tools/` — data conversion, evaluation, post-training utilities
  - `tools/convert_data.py` — raw JSONL → TRL-format HF datasets
  - `tools/data_creators/build_doclang_data.py` — build SFT data teaching the model to emit DocLang natively
  - `tools/data_creators/build_transcription_data.py` — build SFT data teaching the model to transcribe PAGE XML to plain unicode
  - `tools/evaluate.py` — lm-eval-harness wrapper
  - `tools/merge_adapter.py` — merge a LoRA adapter into the base model
  - `tools/serve.py` — lightweight inference
  - `tools/generate_doclang.py` — generate a validated `.dclg` document from a vLLM server (grammar or JSON-schema path)
- `scripts/` — shell utilities
  - `scripts/find_images.sh` — find images, shuffle, and save filenames to a list
  - `scripts/sample_lists.sh` — sample from the top of multiple shuffled image lists
- `utils/` — shared helpers
  - `utils/logging_utils.py` — unified logging (logger, setup, git commit, config hash, run metadata)
  - `utils/page_xml_editor.py` — PAGE XML parsing and transcription extraction
  - `utils/paths.py` — output directory helpers
  - `utils/tempdir.py` — optional temp dir + atomic file name helpers
- `tests/` — unit tests; `tests/e2e/` — end-to-end integration tests
- `docs/` — documentation and the implementation plan

## Documentation

- [docs/PLAN.md](docs/PLAN.md) — the full implementation plan
- [docs/setup.md](docs/setup.md) — environment setup
- [docs/usage.md](docs/usage.md) — running training, evaluation, and post-training
- [docs/data_formats.md](docs/data_formats.md) — data formats and conversion (single-turn `prompt` and multi-turn `messages`)
- [docs/structured_output.md](docs/structured_output.md) — constrained DocLang (`.dclg`) generation via grammar / JSON schema, with the XSD sync-guarantee test suite
