# Setup

## Prerequisites

- [mamba](https://mamba.readthedocs.io/) (or conda)
- A CUDA-capable GPU (pro GPU 24-48GB recommended for Qwen3.5-9B)

## Create the environment

```bash
mamba env create -f environment.yml
mamba activate voxdoc-trainer
```

## Verify the environment

```bash
python -c "import trl, transformers, peft, accelerate; print('OK')"
```

## Run tests

Unit tests (no GPU required):

```bash
pytest -m "not e2e"
```

End-to-end tests (requires GPU + installed deps):

```bash
pytest -m e2e
```

## Formatting

This project uses [black](https://black.readthedocs.io/) with a line width of 128:

```bash
black --line-length 128 src/ tools/ utils/ tests/
```
