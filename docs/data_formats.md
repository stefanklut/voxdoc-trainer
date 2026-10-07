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

```json
{"image": "path/to/doc.png", "prompt": "Read this document."}
```

Multi-turn (the model generates the next response):

```json
{"messages": [{"role": "user", "content": "Read this document."}, {"role": "assistant", "content": "I found an uncertain section."}, {"role": "user", "content": "Explain that section."}]}
```

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

## Converting JSONL to HF datasets

```bash
python tools/convert_data.py --input data/train.jsonl --output data/train_sft --method sft
python tools/convert_data.py --input data/train.jsonl --output data/train_dpo --method dpo
python tools/convert_data.py --input data/train.jsonl --output data/train_kto --method kto
python tools/convert_data.py --input data/train.jsonl --output data/train_grpo --method grpo
```
