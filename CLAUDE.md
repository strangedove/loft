# Loft — SFT/DPO Fine-Tuning Toolkit

## Overview
Loft is a power-user SFT/DPO fine-tuning toolkit (v0.1.0) built on Transformers 5.x, PEFT, Accelerate, and DeepSpeed. It targets 1-2x RTX 3090 (24GB) and occasionally H100 (80GB).

## Project Structure
```
loft/
  cli.py              — CLI entry point (`loft` command)
  data_utils.py       — Dataset loading and preprocessing
  import_utils.py     — Lazy import helpers
  models/             — Model utilities, activation offloading
  kernels/            — Custom Triton kernels (fused logprobs, ScatterMoE + NF4 LoRA)
  patches/            — Monkey-patches (FLA Triton fixes, TRL compatibility)
  scripts/
    sft.py            — SFT training script (via `loft sft` or `loft train`)
    dpo.py            — DPO training script (via `python -m loft.scripts.dpo`)
    prepare.py        — Dataset preparation (`loft prepare`)
    eval.py           — Evaluation with vLLM (`loft eval`)
    merge.py          — LoRA merge + upload (`loft merge`)
    utils.py          — Shared training utilities
  trainer/
    sft_trainer.py    — Custom SFT trainer
    sft_config.py     — SFT config dataclass
    model_config.py   — Model/LoRA config with custom fields
    base_trainer.py   — Base trainer class
    chunked_mlp.py    — Chunked MLP for memory-efficient long-context
    chunked_logprobs.py — Vocab-chunked log probs for DPO
    triton_rl_kernels.py — Triton-fused RL kernels for DPO
    callbacks.py       — Training callbacks
    utils/             — Collators, padding, loss, distributed helpers
configs/              — Training configs (train.yaml + data.yaml per run)
runs/                 — Training output directories
scripts/              — Standalone utility scripts (not part of loft package)
```

## Commands
- `loft init <name>` — Scaffold a new project
- `loft prepare <config.yaml>` — Tokenize and prepare datasets
- `loft sft <config.yaml>` (alias: `loft train`) — Run SFT training via accelerate
- `loft eval` — Run vLLM-based evaluation
- `loft merge` — Merge LoRA adapters and optionally upload
- `python -m loft.scripts.dpo <args>` — Run DPO training (separate from CLI)

## Development
- **Python**: >=3.10, managed with `uv`
- **Package manager**: `uv` (mandatory — never use pip directly)
- **Linting**: ruff (line-length 119, target py310)
- **Type checking**: basedpyright (relaxed settings)
- **Tests**: pytest (markers: `slow`)
- **Branch**: `transformers-5x` is the active development branch

## Key Conventions
- Config files: `configs/<run-name>/train.yaml` + `data.yaml`, output to `runs/<run-name>/`
- Qwen3.5 models have 248K vocab — always use CCE (cut-cross-entropy) or chunked logprobs to avoid OOM
- Memory optimization stacking: `chunked_mlp` + `activation_offloading` for max savings
- Liger SwiGLU conflicts with chunked_mlp — use one or the other, not both

## Memory Management (CRITICAL)
After any training run fails, crashes, or is manually stopped, you MUST check and clean up system and GPU memory before starting another memory-heavy task. Failure to do so has caused system crashes from memory overflow.

Cleanup steps:
```bash
# Check GPU memory
nvidia-smi
# Kill any leftover Python/training processes holding GPU memory
# Check system RAM
free -h
# Clear Python/PyTorch caches
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null
# Clear system page cache if RAM is still high (needs sudo)
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
```

Always verify memory is reasonably free before launching the next training run or heavy GPU task.

## Running Commands
```bash
uv run loft sft configs/my-run/train.yaml
uv run python -m loft.scripts.dpo --config configs/my-run/dpo.yaml
uv run pytest tests/
```
