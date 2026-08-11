# Data Formats

Data is stored as JSONL (source of truth) and converted to Hugging Face datasets
for training. Conversion happens **before** applying the chat template, per TRL
recommendation.

## SFT (corrections)

```json
{"image": "path/to/doc.png", "prompt": "Read this document.", "completion": "The corrected transcription."}
```

## DPO (A/B comparisons)

```json
{"image": "path/to/doc.png", "prompt": "Read this document.", "chosen": "The preferred answer.", "rejected": "The dispreferred answer."}
```

## KTO (good/bad feedback)

```json
{"image": "path/to/doc.png", "prompt": "Read this document.", "completion": "The answer.", "label": true}
```

## GRPO (feedback / reward)

```json
{"image": "path/to/doc.png", "prompt": "Read this document."}
```

## DocLang (historical document transcription)

DocLang (<https://doclang.ai/>) is an AI-native XML document format designed for LLMs
to consume directly. If the model does not support it natively, we teach it to by
giving it many examples of existing OCR already in DocLang format.

Build SFT training data from a directory of existing DocLang XML documents (plus their
source images). The model sees a document image and must produce the DocLang
representation as its completion, so it learns the format itself:

```bash
python tools/build_doclang_data.py --input path/to/doclang_xml/ --images path/to/images/ --output data/transcription
```

## Converting JSONL to HF datasets

```bash
python tools/convert_data.py --input data/train.jsonl --output data/train_sft --method sft
python tools/convert_data.py --input data/train.jsonl --output data/train_dpo --method dpo
python tools/convert_data.py --input data/train.jsonl --output data/train_kto --method kto
python tools/convert_data.py --input data/train.jsonl --output data/train_grpo --method grpo
```
