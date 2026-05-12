# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository

Doc-to-LoRA (D2L): a hypernetwork that reads a context document and emits per-document LoRA weights, which are then merged into a frozen base LLM so the model "internalizes" the document for downstream queries. This fork extends the original Sakana AI release with a BabiLong task and an experimental random-representation Perceiver variant — see `CHANGES.md` (Russian) and `IDEFICS2_PERCEIVER_DIFF.md` for what was changed and why.

## Environment & install

The project uses `uv` (not pip/conda directly). All commands are expected to run via `uv run …`.

```bash
./install.sh              # creates .venv, syncs uv.lock, installs torch 2.6 cu124, flash-attn, flashinfer, downloads SQuAD/DROP/PWC/ROPES
uv run pre-commit install # optional; pre-commit runs ruff (lint+format), isort (black profile), pyupgrade --py310-plus
```

Some local scripts (`scripts/random_repr/train.sh`, `scripts/babilong/train_qa_1.sh`) hard-code `conda run --live-stream -n D2L python …` and a self-hosted W&B endpoint (`WANDB_BASE_URL=https://wandb-radfan.ru`). On a fresh machine that conda env does not exist — switch the launcher to `uv run python …` or create a `D2L` env, and override `WANDB_*` env vars (or set `WANDB_MODE=disabled`).

There is a separate Qwen3.5/transformers-5.3.0 environment (`prag_req`) recorded in user memory; it is **not** the env this repo trains in. D2L training requires `transformers==4.51.3` exactly (pinned in `pyproject.toml`) — any other version will break model loading.

## Common commands

Training is multi-GPU via `accelerate launch --config_file accelerate_config.yaml` (8 processes, bf16). The single-GPU local scripts here use `CUDA_VISIBLE_DEVICES=1` and bypass accelerate.

```bash
# Main D2L training (multi-GPU, gemma-2-2b)
uv run bash scripts/main_exp/1-train.sh

# NIAH "magic-number" training / eval
uv run bash scripts/niah/1-train.sh
uv run bash scripts/niah/2-eval.sh

# BabiLong fine-tuning from a pretrained D2L checkpoint (this fork)
bash scripts/babilong/train_qa_1.sh
bash scripts/babilong/eval_d2l_model_qa_1.sh

# Random-representation Perceiver experiment (this fork; requires USE_RANDOM_REPR=True in hypernet.py)
bash scripts/random_repr/train.sh

# Evaluate a checkpoint directly
uv run run_eval.py --checkpoint_path train_outputs/runs/$RUN/checkpoint-$STEP/pytorch_model.bin \
    --datasets squad drop ropes --split test --eval_batch_size_gen 1 --max_ctx_chunk_len 8192

# Auto-eval new checkpoints as they appear under train_outputs/runs/*/checkpoint*/pytorch_model.bin
uv run watcher.py

# Interactive demo / dataset viewer
uv run demo/app.py
uv run webui/self_gen_viewer.py
```

There is no test suite. Lint/format only via `pre-commit run --all-files`.

## Architecture

`train.py` and `run_eval.py` both start with `sys.path.append("src")` because `ctx_to_lora` lives under `src/` but is imported as a top-level package — keep that line if you add new entry points.

The training graph is built from three layers, in this order (see `train.py:191-261`):

1. **Base model** — frozen pretrained LLM (Gemma-2-2b / Qwen3-4B / Mistral-7B), wrapped in PEFT to register LoRA adapters on `target_modules` (typically `down_proj`).
2. **Context encoder** (`src/ctx_to_lora/modeling/ctx_encoder.py`) — reads the document. `ctx_encoder_type=early_exit` returns hidden states from a single layer (much smaller memory); `per_layer_activations` returns a per-layer stack and is what production main_exp uses. The default encoder layer index is `num_hidden_layers // 4`.
3. **Aggregator + HyperLoRA head** (`modeling/aggregator.py`, `modeling/hypernet.py`) — a Perceiver Resampler (encoder + decoder, both trainable) compresses encoder activations into `n_latent_queries` latents, then a `ResMLPBlock(PerLayer)` head predicts low-rank LoRA factors `(A, B)` per target module per layer. Output is plugged into the frozen base model's LoRA slots via `model.generate_weights(...)` before the forward pass.

`ModulatedPretrainedModel` (`hypernet.py:518`) is the top-level training/inference object. `internalize(text)` runs the encoder→aggregator→head once and stashes the generated LoRA weights; `reset()` clears them; `generate(...)` then behaves like a normal HF generate that reflects the internalized doc. `from_state_dict(...)` is the supported reload path — it reconstructs the wrapped base model + ctx_encoder + hypernet from a single `pytorch_model.bin`, so checkpoints are self-contained.

**Sequence packing is mandatory**: `train.py` asserts `ctx_args.use_sequence_packing=True`, forces `per_device_train_batch_size=1`, and packs samples up to `max_packed_inp_len` / `max_packed_ctx_len`. Position ids encode chunk boundaries (a new chunk starts wherever `position_ids==0`). Documents are split into chunks of length in `[min_ctx_chunk_len, max_ctx_chunk_len]`; the number of chunks per sample is sampled from `num_chunk_probs` (a JSON-string mapping `n_chunks → probability`).

### Argument layout

CLI args are split across eight dataclasses in `src/ctx_to_lora/configs.py`, parsed by `ArgumentParser` (`HfArgumentParser` subclass) which can also load YAML configs (`configs/main_exp/*.yaml`, `configs/niah_exp/*.yaml`) as the first positional argument. Validation refuses overlapping field names across the dataclasses, so when adding a flag, pick the dataclass it semantically belongs to:

- `DataArguments` — train/val dataset names, sample caps, custom splits
- `CtxTrainingArguments` — `exp_setup` (`hyper_lora` vs plain `lora`), packing limits, KL/L1 losses, chunk size limits, `from_pretrained_checkpoint`
- `ModelArguments`, `LoRAArguments` — base model + PEFT LoRA config
- `TrainingArguments` — extends HF `TrainingArguments`
- `HypernetArguments`, `AggregatorArguments`, `CtxEncoderArguments` — hypernet/Perceiver/encoder shape

### Random-representation flag (this fork)

`src/ctx_to_lora/modeling/hypernet.py` has a module-level `USE_RANDOM_REPR = True` (line 67). When True the Perceiver path samples random low-rank `(A, B)` matrices keyed by per-context seeds (computed from the tokenized chunks) and adds their encoder output to the text latents — see `IDEFICS2_PERCEIVER_DIFF.md` for the exact dataflow and `lora_layer.py:apply_random_repr / random_repr_forward` for the inference-side resampling. This branch only fits in 80GB with `ctx_encoder_type=early_exit`; `per_layer_activations` OOMs. The data path expects an already-generated NIAH dataset — `data/definitions.py` has paths pointing at a sibling repo, adjust them if reproducing.

### Known regularization gotcha

`gen_lora_l1_reg_coef` defaults to 0.0 in the paper, but several launchers in this repo set it to 0.1+ (e.g. `scripts/main_exp/1-train.sh`, `scripts/babilong/train_qa_1.sh`). High values drive the generated LoRA toward zero within ~150 steps and the model regresses to base behavior — watch `lora_cossim_mean` and the L1 term early in training before assuming a config bug elsewhere.

## Outputs and W&B

Runs land in `train_outputs/runs/<run_name>/`, with `args.yaml` + `cli_args.yaml` written before training starts (useful for `watcher.py` and for re-loading via `--from_pretrained_checkpoint`). W&B project defaults to `ctx_to_lora` (set by `train.py`); local fork scripts override it to `doc-to-lora` and point at a self-hosted instance. Set `WANDB_MODE=disabled` for local debugging.
