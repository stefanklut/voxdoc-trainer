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
| Transcription pretraining (e.g. Old Dutch) | GRPO pretrain (CER reward) | `GRPOTrainer` |

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
  - `src/rewards.py` — finetune GRPO rewards (`format_reward`, `accuracy_reward`)
  - `src/rewards_pretrain.py` — pretrain GRPO reward (`cer_reward`, character error rate)
  - `src/doclang_structured/` — constrained DocLang (`.dclg`) generation: EBNF grammar + Pydantic/JSON-schema fallback, validator-gated
- `configs/` — example YAML configs for each method (chained pipeline + dry-run + GRPO pretrain)
- `eval/` — post-training evaluation
  - `eval/benchmarks.py` — thin lm-eval-harness wrapper for external benchmark tasks
  - `eval/documents.py` — bbox IoU, box-label accuracy, and transcription CER CLI
  - `eval/scoring.py`, `eval/runner.py` — strict scoring, dataset adapters, paired stochastic runs, and offline rescoring
  - `eval/configs/documents.yaml` — example grid/pixel experiment configuration
- `tools/` — data conversion, evaluation, post-training utilities
  - `tools/convert_data.py` — raw JSONL (files or directories) → TRL-format HF datasets
  - `tools/data_creators/build_doclang_data.py` — build SFT data teaching the model to emit DocLang natively
  - `tools/data_creators/build_transcription_data.py` — build SFT data teaching the model to transcribe PAGE XML to plain unicode
  - `tools/data_creators/build_region_data.py` — build SFT data teaching the model to detect and label PAGE XML regions
  - `tools/merge_adapter.py` — merge a LoRA adapter into the base model
  - `tools/serve.py` — lightweight local inference, with optional document images
  - `tools/generate_doclang.py` — generate validated `.dclg` documents from a vLLM server (grammar or JSON-schema path; `--image` may be a directory for batch mode)
- `scripts/` — shell utilities
  - `scripts/find_images.sh` — find images, shuffle, and save filenames to a list
  - `scripts/sample_lists.sh` — sample from the top of multiple shuffled image lists
- `utils/` — shared helpers
  - `utils/discovery.py` — corpus autodiscovery (PAGE XML in `page` dirs, DocLang, JSONL) and image resolution
  - `utils/image_loading.py` — embedded vs. referenced image storage (`--image-mode`), manifests, and lazy decoding during training
  - `utils/logging_utils.py` — unified logging (logger, setup, git commit, config hash, run metadata)
  - `utils/metrics.py` — shared IoU, Levenshtein distance, text normalization, and unclipped CER primitives
  - `utils/page_xml_editor.py` — PAGE XML parsing and transcription extraction
  - `utils/paths.py` — output directory helpers
  - `utils/tempdir.py` — optional temp dir + atomic file name helpers
- `tests/` — unit tests; `tests/e2e/` — end-to-end integration tests
- `docs/` — documentation and the implementation plan

### Storing images by reference

Data creators embed images in the dataset by default, which is portable but
heavy (~4.5 GB per 1000 300-dpi pages). Pass `--image-mode reference` to any
creator to store image paths plus an `image_manifest.json` instead, keeping the
dataset at a few megabytes; training decodes images lazily per batch, downscaled
to at most `data.max_pixels` (1 003 520 by default) so a 20-megapixel scan does
not become a 16 000-token prompt. Point `data.image_root` at the corpus if it has
moved. See
[storing images](docs/data_formats.md#storing-images-embedded-vs-referenced).

## Documentation

For document evaluation, prepare a [typed evaluation manifest](docs/data_formats.md#evaluation-manifests),
edit [eval/configs/documents.yaml](eval/configs/documents.yaml), and run:

```bash
python -m eval.documents generate --config eval/configs/documents.yaml
python -m eval.documents score --run-dir outputs/evaluation
```

The example compares requested 0–1000 grid output against original-pixel output
over five paired sampled runs. Generation requires a running OpenAI-compatible
vision endpoint; rescoring is offline. See [evaluation usage](docs/usage.md#document-evaluation)
for HF dataset inputs, failure policies, and metric definitions.

- [docs/PLAN.md](docs/PLAN.md) — the full implementation plan
- [docs/setup.md](docs/setup.md) — environment setup
- [docs/usage.md](docs/usage.md) — running training, evaluation, and post-training
- [docs/data_formats.md](docs/data_formats.md) — data formats and conversion (single-turn `prompt` and multi-turn `messages`)
- [docs/structured_output.md](docs/structured_output.md) — constrained DocLang (`.dclg`) generation via grammar / JSON schema, with the XSD sync-guarantee test suite
