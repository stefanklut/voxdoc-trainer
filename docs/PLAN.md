# voxdoc-trainer — LLM Finetuning Codebase (TRL)

**Status:** Implementation plan (saved as first step of the plan)

## Goal

Build a production-quality, **model-agnostic** codebase for finetuning vision-language LLMs on historical document understanding. Data types map to TRL trainers:

- **corrections → SFT** (`SFTTrainer`)
- **A/B comparisons → DPO** (`DPOTrainer`)
- **good/bad feedback → KTO** (`KTOTrainer`)
- **feedback → GRPO** (online RL, 4th method) (`GRPOTrainer`)

Built on **TRL + PEFT (LoRA/QLoRA) + accelerate**, targeting **Qwen3.5-9B** on a single pro GPU first, scaling to multi-GPU. Uses TRL/transformers native features wherever possible (only building custom where TRL lacks them). Formatter: **black (line width 128)**. **DocLang** as a data source for historical document transcription. Includes **unit + E2E tests**.

## Model strategy

- **No fallback to older Qwen.** Stay state-of-the-art: Qwen3.5-9B now, Qwen3.8 or newer as they release.
- Codebase is **model-agnostic and config-driven** — model id lives in config, so swapping models requires **zero code changes**.
- TRL already tracks the latest Qwen (Qwen3.5 and Qwen3.6 in its supported-models list).

## Framework comparison (why TRL)

| | **TRL (HF)** | **Axolotl** | **torchtune** |
| --- | --- | --- | --- |
| SFT / DPO / KTO / GRPO | all | all | no KTO/GRPO |
| Vision (Qwen-VL) | yes | yes | limited |
| (Q)LoRA | PEFT | yes | yes |
| Single→multi GPU | accelerate/DDP/FSDP/DeepSpeed | yes | yes |
| Programmatic control | High | Low (YAML) | High |
| Maintenance | **Active** | **Active** | **wound down 2025** |

## Additional TRL methods (beyond SFT/KTO/DPO) — vision applicability

- **GRPO** — Online RL, **strong vision support** (Qwen2-VL/2.5-VL, Gemma3, LLaVA-NeXT, SmolVLM2). **High relevance** — feedback data becomes reward functions.
- **RewardTrainer** — Reward modeling, limited vision. Medium — train reward model from A/B data to feed GRPO.
- **Experimental (defer):** RLOO, OnlineDPO, NashMD, PPO, XPO, BCO, CPO, ORPO, PRM, GKD, MiniLLM.

## TRL-aligned specifics (verified against official docs)

- **SFT** — conversational prompt-completion; vision `images` column + content dicts; `max_length=None`; `assistant_only_loss=False` (not supported for VLMs in TRL); QLoRA = BitsAndBytes + `peft_config`; adapter LR ~1e-4.
- **DPO** — preference `{"prompt","chosen","rejected"}`; `precompute_ref_log_probs=False` (not supported for vision datasets in TRL); adapter LR ~1e-5.
- **KTO** — unpaired `{"prompt","completion","label"}`; **no `precompute_ref_log_probs` for vision**; `train_sampling_strategy="sequential"` + batch>1; LR 5e-7..5e-6.
- **GRPO** — `prompt` + `image`/`images`; custom reward functions; vLLM (colocate/server) or transformers continuous batching; LoRA on vision projection layers.
- **MoE (Qwen3.5-9B is sparse MoE)** — `router_aux_loss_coef` for load-balancing aux loss.

## Steps

### Phase 0 — Repo scaffold + plan (first)

1. Create folders `tools/`, `utils/`, `src/`, `docs/`, `tests/`
2. **Save this plan to the repo** as `docs/PLAN.md` (the requested first step)
3. `environment.yml` (mamba): python, pytorch-cuda, transformers, trl, peft, accelerate, datasets, bitsandbytes, wandb, tensorboard, lm-eval, vllm, pytest, **black** + `pyproject.toml`
4. `README.md`, `.gitignore`

### Phase 1 — Data organization (tools/ + utils/)

1. Data converters (mirror TRL `examples/datasets/`): corrections → SFT; A/B → DPO; good/bad → KTO; feedback → GRPO reward data. Vision `images` column + content dicts
2. **DocLang integration (doclang.ai)** — AI-native document format (Linux Foundation/IBM/ABBYY/NVIDIA standard) for loading existing OCR of historical documents. **Teach the model to emit DocLang natively** by giving it many examples of existing OCR already in DocLang format. `tools/data_creators/build_doclang_data.py` builds SFT training data from `.dclg` files: the model sees a document image and must produce the DocLang representation as its completion. Each document is validated with the official `doclang` package (XSD + Schematron) before it is added; invalid documents are skipped. *(Reference implementation integrated via the `doclang[schematron-saxon]` pip package in `environment.yml`.)*
3. `utils/` shared helpers (paths, logging, config loading)

### Phase 2 — Training (src/)

1. Config system (YAML/dataclasses) — **model-agnostic, model id in config**
2. SFT trainer (`SFTTrainer` + PEFT LoRA/QLoRA)
3. DPO trainer (`DPOTrainer`)
4. KTO trainer (`KTOTrainer`)
5. **GRPO trainer (`GRPOTrainer`)** — reward functions from feedback data
6. Logging (`report_to`), checkpointing, resume; `router_aux_loss_coef` for MoE

### Phase 3 — Evaluation (tools/ + src/)

 1. **Use TRL/transformers native eval-during-training** (`eval_dataset` + `compute_metrics`) for in-loop metrics. **lm-eval-harness is external** (not native to TRL) → wrap it in `tools/evaluate.py` for post-training benchmark evals. Custom document-understanding eval (transcription accuracy) built on top of lm-eval-harness or a small custom harness.

### Phase 4 — Multi-GPU scaling

 1. accelerate config for DDP/FSDP/DeepSpeed; single-GPU default; GRPO vLLM server mode on separate GPUs

### Phase 5 — Quality-of-life features (src/ + tools/)

 1. **QoL features** — **use TRL/transformers native features first; only build custom where TRL lacks it**:
    - **NATIVE (no custom code)**: checkpoint mgmt (`save_strategy`, `save_steps`, `save_total_limit`, `save_only_model`); `report_to` (wandb/tensorboard); `seed`; `EarlyStoppingCallback`; `eval_dataset` + `compute_metrics`; speed/memory (`gradient_checkpointing`, `bf16`/`fp16`, `torch.compile`, `max_grad_norm`, `gradient_accumulation_steps`, FlashAttention, Liger kernel, Unsloth, vLLM for GRPO); data opts (`packing`, `dataset_num_proc`, `shuffle`, `pad_to_multiple_of`, dataset caching); `push_to_hub` + model card; GRPO `log_completions`/`log_multimodal`; `HfArgumentParser` for CLI overrides
    - **CUSTOM (build these)**: `--dry-run`/`--smoke` mode (tiny `max_steps` to validate a run before committing GPU time); log git commit + env + config hash to run dir; run naming / organized `output_dir`; LoRA adapter merge tool (PEFT `merge_and_unload` wrapper); inference/serving script; `Makefile`/task runner; pre-commit (**black**); `.env`/`HF_TOKEN` handling
 2. `pytest` unit tests (data converters, config, utils)

### Phase 6 — E2E tests + docs

 1. **E2E tests (`tests/e2e/`)** — full-pipeline integration tests on tiny synthetic data + a tiny model (e.g. Qwen3.5-0.8B), marked `@pytest.mark.e2e` / gated behind an env var so they don't run in normal CI. Cover:
    - **Training startup**: config → data load → model load → a few steps → checkpoint saved → loss decreases
    - **Data pipeline**: raw JSONL → converter → HF dataset → trainer consumes it (each of SFT/DPO/KTO/GRPO)
    - **Checkpoint/resume**: save mid-training, resume, verify it continues from the saved step
    - **Evaluation**: trained adapter → eval harness → metrics produced
    - **Model swap**: change model id in config → pipeline runs (validates model-agnostic design)
    - **Multi-GPU launch**: `accelerate launch` smoke test (optional, gated on GPU count)
 2. `docs/` (setup, usage, data formats, scaling guide, model-swap guide for new Qwen releases)

## Verification

1. `mamba env create -f environment.yml`
2. `pytest` (unit) passes
3. `pytest -m e2e` (E2E) passes on a tiny model
4. Smoke-train each method on a tiny dataset on single GPU
5. `accelerate launch` multi-GPU smoke test
6. **Compatibility spike first:** confirm Qwen3.5-9B loads with PEFT LoRA in TRL before full build

## Decisions

- **TRL** over axolotl/torchtune
- **Four methods** in PoC: SFT + DPO + KTO + GRPO
- **Qwen3.5-9B** now; **model-agnostic** so Qwen3.8+ plugs in later (no fallback to older models)
- **Formatter: black (line width 128)** over ruff
- **DocLang** as a data source for historical document transcription training
- **Training pipeline order: SFT → DPO/KTO → GRPO (optional)**. SFT first teaches the task + DocLang format; DPO/KTO then align behavior from the SFT checkpoint; GRPO optionally refines via reward functions. Chain stages by pointing the next stage's `model_id` at the previous stage's output. Mixing all sources into one SFT run is also valid; mixing preference objectives (MPO) is a later optimization.
- **Use TRL/transformers native QoL + eval features first; only build custom where TRL lacks it**

## Further Considerations

1. **Logging backend** — W&B vs TensorBoard. Recommendation: support both, default TensorBoard for PoC.
2. **Data storage format** — JSONL as source of truth, convert to HF `datasets` for training.
3. **GRPO is memory-intensive** — needs vLLM or continuous batching; implement after SFT/DPO/KTO are proven.
4. **DocLang tooling maturity** — the spec is open (Linux Foundation); the reference `doclang` pip package (v0.7.x) is available and used for validation. Full Schematron validation needs a JRE (via the `schematron-saxon` extra); XSD-only validation avoids that dependency.
