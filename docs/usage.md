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
the SFT output, and `grpo.yaml` points at the DPO output. There is also a
pretrain config (`grpo_pretrain.yaml`) that optimizes CER directly (see
[Pretraining (Old Dutch) with GRPO + CER](#pretraining-old-dutch-with-grpo--cer)).

```bash
python src/train_sft.py --config configs/sft.yaml
python src/train_dpo.py --config configs/dpo.yaml
python src/train_kto.py --config configs/kto.yaml
python src/train_grpo.py --config configs/grpo.yaml
python src/train_grpo.py --config configs/grpo_pretrain.yaml
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

## Pretraining (Old Dutch) with GRPO + CER

To pretrain the model to learn a language (e.g. Old Dutch) by directly
optimizing the **character error rate (CER)**, use the GRPO pretrain stage.
CER is the standard OCR/transcription metric, so it is the natural objective
for teaching the model to transcribe historical documents.

- **Reward:** `cer_reward` in `src/rewards_pretrain.py` — a continuous
  `max(0.0, 1.0 - CER)` score in `[0, 1]`, computed on the **final answer only**
  (the `think ... endthink` reasoning is stripped first, so the chain-of-thought
  is never reinforced). CER is **case-sensitive** and **whitespace counts**
  (raw Levenshtein over the final answer), matching the primary eval metric.
  Set `grpo.cer_metric: normalized` to instead use `cer_reward_normalized`,
  which casefolds and collapses whitespace before scoring (the previous
  behaviour).
- **Why CER works for GRPO:** GRPO computes advantages *within a group* of
  `num_generations` rollouts. A binary exact-match reward collapses most groups
  to a single value → zero advantage → no gradient. CER gives **partial credit**,
  so groups stay informative.
- **Stage selection:** `grpo.stage: pretrain` selects the CER reward —
  `cer_reward` (raw, case-sensitive) by default, or `cer_reward_normalized`
  when `grpo.cer_metric: normalized`; the default `grpo.stage: finetune`
  selects `[format_reward, accuracy_reward]` (`src/rewards.py`). The two stages
  use separate reward files so the pretrain and finetune reward surfaces stay
  distinct. `grpo.reward_weights` must have one weight per selected reward
  function (pretrain: `[1.0]`).

```bash
python src/train_grpo.py --config configs/grpo_pretrain.yaml
```

**Recommended recipe:** a short **SFT warm-start** (so the model can already emit
reasonable transcriptions) → **GRPO pretrain with the CER reward** to drive CER
down. GRPO is the only method that directly optimizes CER.

## Finetuning a reasoning model

The target model (Qwen3.5-9B) is a **thinking/reasoning** model. Options for
finetuning it, and when to use each:

| Option | What it does | When to use |
| --- | --- | --- |
| **SFT** | Imitate corrected/good completions | Teach *what* to output (task + format). Cheap, stable baseline. |
| **GRPO + CER reward** | Online RL; reward = closeness to ground truth | Optimize a *measurable* objective (CER) after the model can already transcribe. |
| **DPO / KTO** | Preference alignment from A/B or good/bad feedback | Re-rank behavior when you have preference data. |
| **GRPO + format reward** | Reward well-formed output | Enforce structure (DocLang/bbox). |

Reasoning-model considerations:

- Keep thinking **enabled** during rollouts; `max_completion_length` must exceed
  thinking + answer length (512 truncates reasoning models — the configs use 2048).
- Score only the **final answer** (the rewards strip the CoT via `extract_answer`).
- Use `loss_type: dapo` (the default) to avoid length bias / reward hacking.
- Watch the `reward/truncated` metric to size `max_completion_length`.
- Keep `beta` low (0.0) and `temperature` ~1.0 for exploration.

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
space, or `doclang` for the four `<location>` elements). With `qwen`, every
prompt that involves a box states that coordinates are integers normalized to
the 0–1000 range, so the model knows the convention to expect and to use in its
answer.

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

## Build region detection training data

To teach the model to detect and label document regions, build SFT examples
from PAGE XML ground-truth files and their source images. The tool extracts
every PAGE XML region (any element whose tag ends in `Region`, at any nesting
depth) and fits each polygon to an axis-aligned bounding box. Choose the task
with `--mode` and the bbox serialization with `--bbox-format` (`qwen` for
`[x1, y1, x2, y2]` in 0–1000 space, or `native` for `[x1, y1, x2, y2]` in the
declared pixel space). Every prompt states the coordinate space, so the model
knows the convention to use in its answer.

Task modes (PAGE XML only):

- `detect_all` — one example per page: the model detects and labels every region in the document. The prompt lists the label vocabulary present on the page.
- `find_region` — two example styles per page: for every label, "find all `<label>` regions", and for every region, "find the Nth `<label>` region in reading order".

Completions are compact JSON lists of `{"bbox": [x1, y1, x2, y2], "label": ...}`
objects (a single-element list for the ordinal style) — the same shape as the
evaluation `labeled_bbox` contract. Labels come from the region's `custom`
attribute (`structure {type:X;}`) when present, otherwise from the element tag
lowercased (`TextRegion` → `textregion`). Reading order is the
`readingOrder {index:N;}` value from the `custom` attribute when present;
regions without one keep their document order and sort last.

Use `--max-lines` to randomly sample at most N examples from the global pool
of all input files (`--seed` makes the selection reproducible). Files that
cannot be parsed, that contain no regions, or that have no matching image are
skipped:

```bash
python tools/data_creators/build_region_data.py --input path/to/pagexml/ --images path/to/images/ --output data/regions --mode detect_all --bbox-format qwen
python tools/data_creators/build_region_data.py --input path/to/pagexml/ --images path/to/images/ --output data/regions_find --mode find_region --bbox-format native --max-lines 5000 --seed 42
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
python -m eval.benchmarks --model hf --tasks some_task
```

`eval.benchmarks` is only a thin wrapper around `lm_eval.simple_evaluate`:
it forwards `--model`, task names and an optional `--limit`, then prints the
returned results. `--model` is passed directly to the harness's backend selector;
this wrapper does not configure checkpoint loading via `model_args`. It does
not implement document-specific metrics or repeated runs. Those live in the
separate document evaluator below. Evaluation code and example configurations
are grouped under `eval/`; shared training/evaluation metrics remain in `utils/`.

### Document evaluation

The document evaluator supports bbox IoU, spatially matched box-label accuracy,
and transcription CER. It calls a running OpenAI-compatible vision endpoint
(for example vLLM); it does not load or serve model weights itself. Prepare a
[typed manifest](data_formats.md#evaluation-manifests), then edit
[`eval/configs/documents.yaml`](../eval/configs/documents.yaml):

```yaml
data:
  format: manifest
  path: ../../data/evaluation.jsonl
model: Qwen/Qwen3.5-9B  # Must match the name served by your endpoint.
base_url: http://localhost:8000/v1
output_dir: ../../outputs/evaluation
comparison: grid_vs_native
seeds: [0, 1, 2, 3, 4]
temperature: 0.7
top_p: 1.0
max_tokens: 2048
send_seed: true
enable_thinking: false
iou_threshold: 0.5
```

Paths in YAML are relative to the config file, not the current directory.
Manifest image paths are relative to the manifest. Set `OPENAI_API_KEY` in the
environment for authenticated endpoints; `api_key_env` changes its name. Local
endpoints default to `EMPTY` if the variable is unset. Credentials are not saved
in artifacts; endpoint URLs cannot contain credentials or query parameters.

```bash
python -m eval.documents generate --config eval/configs/documents.yaml
python -m eval.documents score --run-dir outputs/evaluation
python -m eval.documents score --run-dir outputs/evaluation --iou-threshold 0.75
```

`grid_vs_native` runs two paired variants, named `grid` and `pixels`, with the
same images, GT, task questions, seeds, and sampling settings. One asks for
integer-valued 0–1000 coordinates; the other asks for original-pixel coordinates
and provides annotation dimensions. Both are scored against unchanged native
GT after converting grid predictions. Quantization can itself affect IoU;
native results are not rounded to force agreement. This comparison tests output
coordinate conventions, not merely mentioning versus omitting a grid instruction.

Five seeds mean five runs. Change the seed list to change the number of repeats.
Default temperature is `0.7`; `0` is permitted for a deterministic control.
Seeds are best-effort: server implementation, model, and hardware still matter.
Set `send_seed: false` explicitly for a backend that does not support seeds.
Unsupported request options are never silently removed. Optional `model_revision`
and `server_revision` strings let you identify the deployed versions in artifacts.
Use a new output directory when changing the model, settings, images, or GT.

`enable_thinking` is optional and, when set, sends vLLM-style
`extra_body.chat_template_kwargs.enable_thinking`. Omit it for endpoints without
that extension. To evaluate reasoning-enabled models, configure
`answer_extraction: endthink` or `think_tag` (closing `</think>`) for the built-in
comparison/default variant. This requires the marker on every response and
removes up to two separator newlines after it; all other answer whitespace is
preserved. Missing markers are failures. The default `none` scores content as
returned. For custom variants, set extraction on each variant instead:

```yaml
variants:
  - name: baseline
    units: native
    prompt_template: "{question}"
    answer_extraction: none
```

Use `variants` instead of `comparison`. Available template fields are `question`,
`units`, `width`, `height`, and `output_format`. Coordinate instructions are added
only when your template uses them; prompts containing input coordinates must
already be correct for that variant. Transcription experiments normally use the
default unchanged prompt and no `comparison` field.

For an existing transcription HF dataset, replace the `data` section with:

```yaml
data:
  format: hf
  path: ../../data/transcriptions_sft
  task: transcription
```

Remove `comparison`. Legacy Qwen single-box HF datasets instead require
`data.task: bbox`, `data.gt_units: qwen`, and `data.output_format: single_box`;
the default variant then uses Qwen units. Use `boxes` for nested box lists.
Legacy prompts and message history remain unchanged; coordinate experiments
require a manifest rather than automatic prompt rewriting. See the
[HF input requirements](data_formats.md#existing-hf-evaluation-inputs).

Externally saved responses also need no endpoint:

```bash
python -m eval.documents score --config eval/configs/documents.yaml \
  --predictions predictions.jsonl --out outputs/external_scores
```

#### Metrics And Failures

- **IoU:** maximize total IoU with one-to-one spatial matching, independently of
  labels. Per-image score is summed matched IoU divided by `max(GT count,
  prediction count)`, so missing/extra boxes count as zero. Report the mean over
  images; this is not mAP. Empty/empty scores 1, one-sided empty scores 0.
- **Detection:** precision, recall, and F1 use matched IoU at or above the
  configured threshold (default `0.5`). Matching maximizes total IoU, not the
  number of threshold-qualified matches. These are localization metrics;
  labels do not change detection TP counts.
- **Labels:** exact correct labels on threshold-qualified matches divided by
  all labeled GT boxes. Missed/low-IoU GT is incorrect. Conditional accuracy on
  matched boxes is secondary; detection precision exposes extra predictions.
- **CER:** Unicode-codepoint Levenshtein edits divided by GT characters,
  including case, whitespace, and newlines. CER can exceed 1 and is not clipped.
  Report corpus CER (summed edits / summed GT characters) and macro CER (mean
  per-example CER). Secondary normalized CER casefolds and collapses whitespace.
  An empty reference scores 0 if prediction is empty, otherwise 1, while edit
  counts retain insertions. An all-empty-reference corpus has null corpus CER.
- **Failures:** malformed structured output scores zero IoU/accuracy without
  partial box salvage. Failed/truncated text attempts score as empty predictions
  in primary CER; partial-text CER is retained as a diagnostic. Finish reasons,
  transport errors, and format failures remain separate status categories.
  All-attempt and valid-only metrics are both reported. Undefined ratios are
  null, not NaN or arbitrary perfect scores.
- **Variation:** metrics are grouped by task, model, variant, and run. Reports
  include between-run mean/sample standard deviation and paired per-run deltas
  (`variant - baseline`, where baseline is the first configured variant).
  Incomplete inputs report missing attempts; paired deltas use common samples
  and show their count. These are descriptive statistics, not confidence intervals
  or best-of-N results.

The output directory contains `metadata.json` (configuration, image hashes,
dimensions, implementation hashes and package versions), `samples.jsonl` (frozen
GT), append-only `attempts.jsonl` (raw responses and request metadata),
`scores.jsonl` (per-attempt parsed results), and `summary.json`. HF image payloads
are frozen once in the sample snapshot; request records retain hashes instead
of repeating base64 data. Rescoring a saved run needs neither original images
nor the original dataset nor API credentials, and does not change raw attempts.

Rerunning `generate` resumes only an identical dataset/configuration/image
fingerprint. Completed malformed or failed attempts are not regenerated. SDK
automatic retries are disabled; a rejected request is saved and stops generation
with an actionable error. Completed transport failures remain visible. Generation
is sequential and interrupted runs can be resumed. Do not edit saved GT or raw
attempt identities; mismatches and duplicate attempts are rejected.

## Post-training

Merge a LoRA adapter:

```bash
python tools/merge_adapter.py --base_model Qwen/Qwen3.5-9B --adapter_path outputs/sft_run --output_dir outputs/merged
```

Run inference:

```bash
python tools/serve.py --model outputs/merged --prompt "Read this document."
```

### Serving with vLLM

For production serving (and for the document evaluator and structured DocLang
generation, which expect an OpenAI-compatible endpoint), serve the merged model
with [vLLM](https://docs.vllm.ai) in a separate environment:

```bash
vllm serve outputs/merged --served-model-name voxdoc
```

Useful flags for a 9B vision model: `--max-model-len 8192`,
`--gpu-memory-utilization 0.9`, and `--limit-mm-per-prompt image=4` when
sending multiple images per request.

Alternatively, skip the merge and load the adapter directly:

```bash
vllm serve Qwen/Qwen3.5-9B \
  --enable-lora --lora-modules voxdoc=outputs/sft_run
```

Then request with model name `voxdoc`. Merging is simpler for production;
direct LoRA serving is handy for quick A/B checks.

The served model name must match the `model` field of
[`eval/configs/documents.yaml`](../eval/configs/documents.yaml) (and of
`tools/generate_doclang.py --model`) so the evaluator and the DocLang
generator talk to the right model.

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
