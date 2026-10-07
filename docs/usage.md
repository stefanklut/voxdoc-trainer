# Usage

All training entry points accept a YAML config file. The model id lives in the
config, so you can swap models (e.g. Qwen3.5-9B -> Qwen3.8+) without
code changes.

## Example config

```yaml
method: sft
model:
  model_id: Qwen/Qwen3.5-9B
  use_qlora: false
  lora_r: 16
  lora_alpha: 32
data:
  train_path: data/train_sft
  eval_path: null
  max_length: null
training:
  output_dir: outputs/sft_run
  per_device_train_batch_size: 2
  num_train_epochs: 3
  learning_rate: 1e-4
  report_to: [tensorboard]
  dry_run: false
  assistant_only_loss: false
  precompute_ref_log_probs: false
```

## Message history

Raw JSONL supports both legacy `prompt` rows and structured `messages` rows
(see [docs/data_formats.md](data_formats.md)). Conversion happens before the
model chat template is applied. SFT, DPO, and KTO learn the next response
conditioned on the complete history; GRPO uses the history as its generation
prompt. The model's chat template must support the selected message roles and
multimodal content format.

The training entry points consume the converted conversational datasets
directly and require no separate configuration for ordinary message history.
Interactive follow-up turns during GRPO generation (via tools or an
`environment_factory`) are a separate, later feature.

## Run training

Example configs are provided in `configs/` for each method, with sensible
defaults. The pipeline configs are chained: `dpo.yaml`/`kto.yaml` point at
the SFT output, and `grpo.yaml` points at the DPO output.

```bash
python src/train_sft.py --config configs/sft.yaml
python src/train_dpo.py --config configs/dpo.yaml
python src/train_kto.py --config configs/kto.yaml
python src/train_grpo.py --config configs/grpo.yaml
```

A dry-run config validates the pipeline before committing GPU time:

```bash
python src/train_sft.py --config configs/sft_dry_run.yaml
```

## Training pipeline (recommended order)

You can combine all data sources, but the standard, most reliable recipe is a
**sequential pipeline** rather than one mixed run:

```
SFT (corrections + good completions + DocLang examples)
        │
        ▼
DPO (A/B comparisons)  ── or ──  KTO (good/bad feedback)
        │
        ▼
GRPO (optional, online RL refinement with reward functions)
```

1. **SFT first** — teaches the model *what* to output (the task and the DocLang
   format) using corrections and good completions. Preference methods re-rank
   behavior rather than teach from scratch, so they need a model that can already
   produce reasonable completions.
2. **DPO or KTO second** — starting from the SFT checkpoint, align toward
   preferred/good outputs using A/B comparisons (DPO) or good/bad feedback (KTO).
3. **GRPO (optional)** — online RL refinement using reward functions derived from your
   feedback data.

To chain stages, point the next stage's `model_id` at the previous stage's output
(e.g. `dpo.yaml` uses `model_id: outputs/sft_run`). Each trainer can load a
checkpoint as its base model.

You can also mix data sources within a single SFT run (corrections + good
completions + DocLang examples are all valid SFT examples). Mixing preference
objectives in one run is possible via TRL's multi-loss DPO (MPO), but is more
advanced and best treated as a later optimization.

## Build DocLang training data

To teach the model to emit DocLang natively, build SFT examples from existing
OCR already in DocLang format (see [docs/data_formats.md](data_formats.md)).
Each document needs a matching source image (any format Pillow can open — png,
jpg, webp, tiff, etc.); documents without one are skipped:

```bash
python tools/data_creators/build_doclang_data.py --input path/to/doclang/ --images path/to/images/ --output data/transcription
```

## Build transcription training data

To teach the model to transcribe a document image to plain unicode, build SFT
examples from PAGE XML ground-truth files and their source images. The
transcription is extracted from the PAGE XML (lines ordered by reading order when
available); files that cannot be parsed, that contain no text, or that have no
matching image are skipped:

```bash
python tools/data_creators/build_transcription_data.py --input path/to/pagexml/ --images path/to/images/ --output data/unicode_transcription
```

## Build bbox grounding training data

To teach the model to ground text in bounding boxes — or to predict a bounding
box for a line of text — build SFT examples from PAGE XML and/or DocLang files
and their source images. Choose the task with `--mode` and the bbox
serialization with `--bbox-format` (`qwen` for `[x1, y1, x2, y2]` in 0–1000
space, or `doclang` for the four `<location>` elements).

Line-level modes (PAGE XML and DocLang):

- `bbox_to_text` — the model sees the image and a bounding box and must produce the text inside it.
- `text_to_bbox` — the model sees the image and a line of text and must produce the bounding box.

Region-based modes (PAGE XML only; the reading order is the document order of
the lines within a `TextRegion`):

- `line_neighbor` — given a line (by text or bbox) and an offset, produce the text or bbox of the Nth line above/below it (`--max-n` bounds the offset, default 3).
- `region_to_lines` — given a paragraph's bbox, list all its line bboxes in reading order.
- `lines_to_region` — given a paragraph's line bboxes, produce the paragraph's bbox.
- `region_to_transcription` — given a paragraph's bbox, transcribe all its lines in reading order.
- `line_index` — given a line (by text or bbox), state its position, e.g. "line 3 of 7".
- `line_ordering` — given a paragraph's line bboxes in shuffled order, give the reading order as 1-based indices.
- `line_count` — given a paragraph's bbox, state how many lines it contains.

Tasks that emit several boxes serialize them as a `[[x1, y1, x2, y2], ...]`
array in the `--bbox-format` coordinate space. Use `--max-lines` to randomly
sample at most N examples from the global pool of all input files (`--seed`
makes the selection reproducible). Files that cannot be parsed, that contain
no usable pairs, or that have no matching image are skipped:

```bash
python tools/data_creators/build_bbox_data.py --input path/to/docs/ --images path/to/images/ --output data/bbox --mode text_to_bbox --bbox-format qwen --max-lines 5000 --seed 42
python tools/data_creators/build_bbox_data.py --input path/to/pagexml/ --images path/to/images/ --output data/structure --mode line_neighbor --bbox-format qwen --max-n 3
```

## Dry-run / smoke mode

Set `dry_run: true` in the config (or `max_steps` to a tiny value) to
validate a run before committing GPU time.

## Resume training

TRL/transformers native resume is supported via `trainer.train(resume_from_checkpoint=True)`.

## Multi-GPU

Use accelerate:

```bash
accelerate launch --num_processes 2 src/train_sft.py --config configs/sft.yaml
```

For DeepSpeed ZeRO-3, provide an accelerate config file.

## Evaluation

```bash
python tools/evaluate.py --model outputs/sft_run --tasks some_task
```

## Post-training

Merge a LoRA adapter:

```bash
python tools/merge_adapter.py --base_model Qwen/Qwen3.5-9B --adapter_path outputs/sft_run --output_dir outputs/merged
```

Run inference:

```bash
python tools/serve.py --model outputs/merged --prompt "Read this document."
```

## Structured DocLang output (inference)

To make a served model emit **only valid DocLang** (`.dclg`), use the
structured-output package (`src/doclang_structured/`). It constrains an
OpenAI-compatible endpoint (e.g. local vLLM) via an EBNF grammar (primary) or a
JSON schema (fallback), and gates every result on the official `doclang`
validator. See [docs/structured_output.md](structured_output.md) for the full
API, the CLI, and the test suite (including the XSD sync guarantee).

```bash
python tools/generate_doclang.py \
    --base-url http://localhost:8000/v1 --model outputs/merged \
    --image scan.png --out doc.dclg
```
