# Data Formats

Data is stored as JSONL (source of truth) and converted to Hugging Face datasets
for training. Conversion happens **before** applying the chat template, per TRL
recommendation.

Each row may use either a structured `messages` history (multi-turn) or the
legacy `prompt` field (single-turn). The `messages` field holds the prior
conversation; the task-specific field (`completion`, `chosen`, `rejected`,
`label`) describes the next assistant response.

## Message history

A `messages` history is a list of `system`, `user`, and `assistant` messages.
The final message must be from the user. `content` may be a plain string or a
list of content blocks for multimodal input:

```json
{
  "messages": [
    {"role": "user", "content": "Read this document."},
    {"role": "assistant", "content": "I found an uncertain section."},
    {"role": "user", "content": "Explain that section."}
  ],
  "completion": "The section appears to describe..."
}
```

Images are referenced inside a message's content blocks with
`{"type": "image", "image": "path/to/doc.png"}`. The converter collects the
actual images into the dataset's `images` column in placeholder order and
replaces each block with a `{"type": "image"}` placeholder, matching TRL's
vision convention:

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "path/to/doc.png"},
        {"type": "text", "text": "Read this document."}
      ]
    },
    {"role": "assistant", "content": "I found an uncertain section."},
    {"role": "user", "content": "Explain that section."}
  ],
  "completion": "The section appears to describe..."
}
```

## SFT (corrections)

```json
{"image": "path/to/doc.png", "prompt": "Read this document.", "completion": "The corrected transcription."}
```

Multi-turn:

```json
{"messages": [{"role": "user", "content": "Read this document."}, {"role": "assistant", "content": "I found an uncertain section."}, {"role": "user", "content": "Explain that section."}], "completion": "The section appears to describe..."}
```

## DPO (A/B comparisons)

```json
{"image": "path/to/doc.png", "prompt": "Read this document.", "chosen": "The preferred answer.", "rejected": "The dispreferred answer."}
```

Multi-turn:

```json
{"messages": [{"role": "user", "content": "Read this document."}, {"role": "assistant", "content": "I found an uncertain section."}, {"role": "user", "content": "Explain that section."}], "chosen": "The preferred answer.", "rejected": "The dispreferred answer."}
```

## KTO (good/bad feedback)

```json
{"image": "path/to/doc.png", "prompt": "Read this document.", "completion": "The answer.", "label": true}
```

Multi-turn:

```json
{"messages": [{"role": "user", "content": "Read this document."}, {"role": "assistant", "content": "I found an uncertain section."}, {"role": "user", "content": "Explain that section."}], "completion": "The answer.", "label": true}
```

## GRPO (feedback / reward)

GRPO rows require a `ground_truth` field: the answer the reward functions score
generated completions against. The model generates the response online; the
rewards (`src/rewards.py`) compare it to `ground_truth`.

```json
{"image": "path/to/doc.png", "prompt": "Read this document.", "ground_truth": "The corrected transcription."}
```

Multi-turn (the model generates the next response):

```json
{"messages": [{"role": "user", "content": "Read this document."}, {"role": "assistant", "content": "I found an uncertain section."}, {"role": "user", "content": "Explain that section."}], "ground_truth": "The answer."}
```

The rewards score only the **final answer** (the text after the model's
`endthink` marker), so the chain-of-thought is never reinforced. The task is
inferred from the shape of `ground_truth` (DocLang XML, bbox / bbox list,
integer, ordinal, index list, or plain text), so no extra tagging is needed.

The **pretrain** stage (`grpo.stage: pretrain`, see
[docs/usage.md](usage.md#pretraining-old-dutch-with-grpo--cer)) uses the same
row format — a `ground_truth` field — but scores it with the CER reward
(`src/rewards_pretrain.py`) instead of the finetune format + accuracy rewards.
No extra tagging is needed; plain-text `ground_truth` (transcriptions) is the
expected shape for pretraining.

## Evaluation manifests

Document evaluation uses a separate typed JSONL contract. Unlike training rewards,
it never infers the task from the shape of the GT: a transcription of `123` is
still text. IDs must be unique within the selected evaluation set. Unknown row
fields, invalid GT, and ambiguous coordinate spaces are rejected before inference.

For the grid/pixel comparison, put rows like this in `data/evaluation.jsonl`:

```json
{"sample_id":"page-001","document_id":"document-01","task":"labeled_bbox","prompt":"Detect and label all text and table regions.","images":["images/page-001.png"],"width":2000,"height":3000,"gt_units":"native","label_vocabulary":["text","table"],"boxes":[{"bbox":[100,200,600,700],"label":"text"}]}
```

`prompt` is the task question, without prewritten coordinate instructions. The
comparison appends consistent output-format instructions and either grid units
or native dimensions. Images are local single-frame PNG, JPEG, or WebP files,
relative to the manifest; supported base64 data URIs are also accepted. Images
are neither resized nor EXIF-transposed. Annotation `width`/`height` define the
GT coordinate space; actual raster dimensions are recorded independently.

Tasks and expected responses:

| Task | GT field | Default response format |
| --- | --- | --- |
| `bbox` | `boxes`, each containing `bbox` only | JSON list of `[xmin,ymin,xmax,ymax]` arrays |
| `labeled_bbox` | `boxes`, each containing `bbox` and nonempty `label` | JSON list of objects with exactly `bbox` and `label` |
| `transcription` | `text` | Plain transcription text |

Unlabeled localization example:

```json
{"sample_id":"line-001","task":"bbox","prompt":"Find the text \"hello\".","images":["images/page-001.png"],"width":2000,"height":3000,"boxes":[{"bbox":[100,200,600,700]}]}
```

Transcription example, in a separate manifest for a transcription experiment:

```json
{"sample_id":"transcription-001","task":"transcription","prompt":"Transcribe this page.","images":["images/page-001.png"],"text":"First line.\nSecond line."}
```

Native box GT requires positive integer annotation dimensions. `gt_units: qwen`
instead permits integer-valued GT on the inclusive 0–1000 coordinate range, but
such rounded GT is not accepted for a native/grid comparison. That experiment
requires exact native GT; normalized coordinates are never inverted and claimed
to be exact pixel annotations. GT boxes must have positive area and stay within
their declared bounds. Empty `boxes: []` is valid for no-detection images.

Optional `output_format` overrides for `bbox` are `single_box` (one JSON array)
and `doclang_box` (one `<text>` element with exactly four empty `<location
value="integer"/>` children). Both require one GT box. Legacy multi-box outputs
are arrays, even if a data creator used its `doclang` option; set their units
explicitly. `label_vocabulary`, when supplied, rejects unknown predicted labels;
otherwise labels are arbitrary nonempty, exact-match strings.

Model responses must consume the entire declared format: embedded boxes,
Markdown fences, trailing prose, missing/extra object keys, booleans as numbers,
nonfinite coordinates, inverted boxes, and out-of-bounds boxes are failures.
Missing or extra *detections* are allowed and scored, rather than being confused
with malformed syntax. Plain transcription has no structural wrapper: every
character is evaluated, including additional commentary and whitespace.

### Existing HF evaluation inputs

The HF adapter accepts saved datasets with `prompt`, `images`, and either
`completion` (text or one assistant message) or `ground_truth`. If both GT
columns exist they must agree. Set `data.task` explicitly. Bbox tasks also need
`data.gt_units` and the appropriate `data.output_format`; native GT additionally
needs `data.width` and `data.height`. These dimensions apply to every selected
row; use a manifest for varying dimensions. DatasetDict inputs require an
explicit `data.split`.

Chat history and image-placeholder order are preserved. Arrow's null-padded
content-block fields are normalized only at the data-loading boundary. Source
image pixels are encoded as PNG; single-frame PNG-compatible PIL images are
required. Numeric transcriptions remain text. Existing HF prompts are used
unchanged and cannot be automatically rewritten into grid/pixel variants.

Current data creators do not retain native box metadata or semantic labels.
Supply a manifest for the grid/pixel experiment or labeled-box evaluation when
those fields are absent. `build_region_data.py` produces SFT *training* data
whose completions carry semantic labels in the `labeled_bbox` shape, but it is
not an evaluation manifest. KTO's Boolean `label` is preference feedback, not a
semantic box label, and is not used as one.

### Saved prediction input

Externally generated predictions can be scored offline as JSONL:

```json
{"sample_id":"page-001","variant":"grid","run":0,"seed":0,"raw_response":"[{\"bbox\":[50,67,300,233],\"label\":\"text\"}]","finish_reason":"stop"}
```

`sample_id`, `variant`, and zero-based `run` must identify an attempt in the
experiment configuration. `raw_response` is text or null; optional `model` and
`seed` must match the configuration. `finish_reason` defaults to `stop`;
`length` marks truncation. An optional string `transport_error` records a failed
request. Duplicate attempts, incorrect types, and unknown IDs/runs are rejected.
Partial prediction files are allowed, but missing-attempt counts are explicit
and paired comparisons use only shared samples within the same run.

## Migrating from `prompt` to `messages`

A legacy single-turn row maps directly to a one-message history:

```json
{"image": "path/to/doc.png", "prompt": "Read this document.", "completion": "The corrected transcription."}
```

becomes:

```json
{"messages": [{"role": "user", "content": [{"type": "image", "image": "path/to/doc.png"}, {"type": "text", "text": "Read this document."}]}], "completion": "The corrected transcription."}
```

Both forms are accepted by the converter; `messages` is preferred for new data
because it supports follow-up turns.

## DocLang (historical document transcription)

DocLang (<https://doclang.ai/>) is an AI-native XML document format designed for LLMs
to consume directly. If the model does not support it natively, we teach it to by
giving it many examples of existing OCR already in DocLang format.

Build SFT training data from a directory of existing DocLang documents (`.dclg`
files) and their source images (in any format Pillow can open — png, jpg, webp,
tiff, etc.). Each document is validated with the official `doclang` package before
it is added; documents that fail validation or have no matching source image are
skipped. The model sees a document image and must produce the DocLang
representation as its completion, so it learns the format itself:

```bash
python tools/data_creators/build_doclang_data.py --input path/to/doclang/ --images path/to/images/ --output data/transcription
```

## PAGE XML (plain unicode transcription)

Build SFT training data that teaches the model to transcribe a document image to
plain unicode. The tool reads a directory of PAGE XML ground-truth files (`.xml`)
and their source images, extracts each document's transcription (lines ordered by
their reading order when available), and emits one SFT example per document: the
model sees the image and must produce the transcription as its completion. Files
that cannot be parsed, that contain no text, or that have no matching image are
skipped:

```bash
python tools/data_creators/build_transcription_data.py --input path/to/pagexml/ --images path/to/images/ --output data/unicode_transcription
```

## BBox <-> Text (grounding)

Build SFT training data that teaches the model to ground text in bounding boxes
(and vice versa). The tool reads a directory of PAGE XML (`.xml`) and DocLang
(`.dclg`) files and their source images: PAGE XML `TextLine` polygons are fit
to bounding boxes, and DocLang `<text>` elements use their four `<location>`
values (interpreted as `x_min, y_min, x_max, y_max`, per the DocLang spec).

Line-level task modes (`--mode`, PAGE XML and DocLang):

- `bbox_to_text` — the model sees the image and a bounding box and must produce the text inside it.
- `text_to_bbox` — the model sees the image and a line of text and must produce the bounding box.

Region-based task modes (`--mode`, PAGE XML only — the reading order is the
document order of the lines within a `TextRegion`):

- `line_neighbor` — given a line (by text or bbox) and an offset, produce the text or bbox of the Nth line above/below it (`--max-n` bounds the offset, default 3).
- `region_to_lines` — given a paragraph's bbox, list all its line bboxes in reading order.
- `lines_to_region` — given a paragraph's line bboxes, produce the paragraph's bbox.
- `region_to_transcription` — given a paragraph's bbox, transcribe all its lines in reading order.
- `line_index` — given a line (by text or bbox), state its position, e.g. "line 3 of 7".
- `line_ordering` — given a paragraph's line bboxes in shuffled order, give the reading order as 1-based indices.
- `line_count` — given a paragraph's bbox, state how many lines it contains.

Two bbox serializations (`--bbox-format`):

- `qwen` — `[x1, y1, x2, y2]` integers normalized to the 0–1000 range (the Qwen-VL grounding convention). Every prompt that involves a box states that coordinates are integers in the 0–1000 range, so the model knows the convention to expect and to use in its answer.
- `doclang` — the four `<location value="N"/>` elements in the document's native coordinate space (teaches the model to emit correct DocLang).

Tasks that emit several boxes serialize them as a `[[x1, y1, x2, y2], ...]`
array in the `--bbox-format` coordinate space. Normalization uses the source's
declared size (PAGE XML `imageWidth`/`imageHeight`; DocLang
`<default_resolution>`), falling back to the actual image size. When the total
number of candidates across all input files exceeds `--max-lines`, a random
subset of that size is sampled globally (`--seed` makes the selection
reproducible). Files that cannot be parsed, that contain no usable pairs, or
that have no matching image are skipped:

```bash
python tools/data_creators/build_bbox_data.py --input path/to/docs/ --images path/to/images/ --output data/bbox --mode text_to_bbox --bbox-format qwen
```

## Region detection and labeling

Build SFT training data that teaches the model to detect and label document
regions. The tool reads a directory of PAGE XML (`.xml`) files and their
source images: every element whose tag ends in `Region` (at any nesting depth)
is fit to an axis-aligned bounding box.

Task modes (`--mode`, PAGE XML only):

- `detect_all` — one example per page: detect and label every region. The prompt lists the label vocabulary present on the page.
- `find_region` — per-label ("find all `<label>` regions") and per-region ordinal ("find the Nth `<label>` region in reading order") examples.

Completions are compact JSON lists of `{"bbox": [x1, y1, x2, y2], "label": ...}`
objects — the same shape as the `labeled_bbox` evaluation contract above.
Labels come from the region's `custom` attribute (`structure {type:X;}`) when
present, otherwise from the element tag lowercased. Reading order is the
`readingOrder {index:N;}` value from the `custom` attribute when present;
regions without one keep their document order and sort last.

Bbox serializations (`--bbox-format`):

- `qwen` — `[x1, y1, x2, y2]` integers normalized to the 0–1000 range.
- `native` — `[x1, y1, x2, y2]` integers in the document's declared coordinate space (PAGE XML `imageWidth`/`imageHeight`), falling back to the actual image size.

When the total number of candidates across all input files exceeds
`--max-lines`, a random subset of that size is sampled globally (`--seed`
makes the selection reproducible). Files that cannot be parsed, that contain
no regions, or that have no matching image are skipped:

```bash
python tools/data_creators/build_region_data.py --input path/to/pagexml/ --images path/to/images/ --output data/regions --mode detect_all --bbox-format qwen
```

## Converting JSONL to HF datasets

```bash
python tools/convert_data.py --input data/train.jsonl --output data/train_sft --method sft
python tools/convert_data.py --input data/train.jsonl --output data/train_dpo --method dpo
python tools/convert_data.py --input data/train.jsonl --output data/train_kto --method kto
python tools/convert_data.py --input data/train.jsonl --output data/train_grpo --method grpo
```
