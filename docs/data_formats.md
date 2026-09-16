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

## Converting JSONL to HF datasets

```bash
python tools/convert_data.py --input data/train.jsonl --output data/train_sft --method sft
python tools/convert_data.py --input data/train.jsonl --output data/train_dpo --method dpo
python tools/convert_data.py --input data/train.jsonl --output data/train_kto --method kto
python tools/convert_data.py --input data/train.jsonl --output data/train_grpo --method grpo
```
