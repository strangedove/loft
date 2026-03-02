# loft

Power-user SFT training toolkit. Config-driven dataset preparation, training, and checkpoint evaluation for language model fine-tuning.

Built on HuggingFace Transformers, PEFT, and Accelerate.

## Install

```bash
pip install -e .

# Optional extras
pip install -e ".[liger]"          # Liger kernel fused ops
pip install -e ".[quantization]"   # QLoRA (bitsandbytes)
pip install -e ".[eval]"           # loft eval (vLLM)
pip install -e ".[dev]"            # Everything + dev tools
```

## Quick start

```bash
# Create a new project with template configs
loft init my-project
cd my-project

# Edit configs:
#   base.yaml  — model path, LoRA settings, training defaults
#   data.yaml  — datasets, preprocessing, eval split
#   train.yaml — epochs, overrides, output directory

# Prepare datasets (tokenize, combine, split)
loft prepare train.yaml

# Inspect tokenization (verify loss masking and turn boundaries)
loft prepare train.yaml --debug

# Train
loft train train.yaml

# Merge LoRA into base model
loft merge train.yaml

# Evaluate a checkpoint
loft eval train.yaml --checkpoint checkpoint-500
```

## Commands

### `loft init [name]`

Creates a project directory with template config files (`base.yaml`, `data.yaml`, `train.yaml`).

### `loft prepare <config.yaml>`

Prepares datasets for training. Loads datasets from HuggingFace or local paths, preprocesses conversations, tokenizes with the model's chat template, applies truncation, and saves as parquet.

```bash
loft prepare train.yaml              # Full prepare
loft prepare train.yaml --dry-run    # Preview without writing
loft prepare train.yaml --debug      # Inspect tokenization and loss mask
```

The data config (`data.yaml`) supports:
- Multiple datasets with per-dataset overrides (system message, truncation strategy, max length)
- Truncation strategies: `truncate`, `drop`, `split`, `truncate_turns`
- Eval split with configurable ratio
- Assistant-only loss masking

### `loft train <config.yaml>`

Runs SFT training via Accelerate. Alias for `loft sft`.

Supports:
- Full fine-tuning or LoRA/QLoRA
- DeepSpeed ZeRO stages and FSDP
- Gradient checkpointing
- W&B logging
- Convenience scheduling (`saves_per_epoch`, `evals_per_epoch`)

```bash
loft train train.yaml
loft train train.yaml --accelerate_config ds_zero3.yaml
```

### `loft eval`

Runs test prompts against a model using vLLM offline inference. Requires `pip install vllm` or `pip install -e ".[eval]"`.

```bash
# From training config (reads model and output paths)
loft eval train.yaml
loft eval train.yaml --checkpoint checkpoint-500

# Explicit paths
loft eval --model /path/to/model
loft eval --model base-model --lora /path/to/lora

# Prompt selection
loft eval train.yaml --all-prompts
loft eval train.yaml --prompts instruct_writing_1 character_roleplay_2

# Sampling parameters
loft eval train.yaml --temperature 1.1 --min-p 0.15 --max-tokens 1024
```

When a LoRA adapter is detected, eval generates with both the LoRA and the base model for side-by-side comparison. Use `--base-only` to skip the LoRA.

Outputs `{prefix}.json` (raw results) and `{prefix}.md` (human-readable markdown).

### `loft merge`

Merges a LoRA adapter into its base model, shard by shard, without loading the full model into memory.

```bash
# From training config (reads base model and output dir automatically)
loft merge train.yaml
loft merge train.yaml --checkpoint checkpoint-1000

# Explicit paths
loft merge --base-model ./model --lora ./lora-adapter --output ./merged

# Adjust LoRA strength
loft merge train.yaml --weight 0.5      # Half strength (multiplier on adapter's scale)
loft merge train.yaml --scale 1.0       # Direct scale factor (replaces adapter config)
```

- **`--weight`**: Multiplier on the adapter's own scale factor (alpha/r). Useful when you trained the LoRA yourself and want to dial it up or down relative to training strength.
- **`--scale`**: Replaces the scale factor entirely. Useful for third-party LoRAs where you want an exact value.
- **`--checkpoint`**: Picks a specific `checkpoint-*` directory inside the output folder. Without it, the latest checkpoint is used.
- Output defaults to `<output_dir>/merged`.

## Multi-GPU training

Loft supports multi-GPU training via FSDP and DeepSpeed ZeRO, passed as an accelerate config:

```bash
loft train train.yaml --accelerate_config fsdp.yaml
loft train train.yaml --accelerate_config ds_zero2.yaml
```

### Compatibility matrix

Tested with Transformers 5.x, PEFT 0.15, Accelerate 1.12, DeepSpeed 0.18.

| Method | FSDP | DeepSpeed ZeRO-2 | DeepSpeed ZeRO-3 |
|--------|------|------------------|------------------|
| Full fine-tuning | yes | yes | yes |
| LoRA | yes | yes | yes (no grad ckpt) |
| QLoRA (4-bit) | no | yes | no |

**Notes:**
- **QLoRA + FSDP**: Incompatible. FSDP requires uniform dtypes; QLoRA mixes bf16/int4.
- **QLoRA + ZeRO-3**: Incompatible. ZeRO-3 manages its own parameter partitioning and conflicts with bitsandbytes' `device_map`.
- **LoRA + ZeRO-3**: Works, but requires `gradient_checkpointing: false`. ZeRO-3's parameter gather/partition cycle causes metadata mismatches during gradient checkpointing recomputation.

### Tested architectures

| Architecture | Example model | LoRA | QLoRA | Multimodal | Notes |
|-------------|---------------|------|-------|------------|-------|
| Qwen2 | Qwen2-0.5B | yes | yes | n/a | |
| Qwen3 | Qwen3-0.6B | yes | — | n/a | |
| Qwen3-VL | Qwen3-VL-2B-Instruct | yes | — | text-only SFT | Single GPU + FSDP |
| Mistral | Voxtral-Mini-3B | yes | — | n/a | |
| AfmoeSCM (MoE) | Trinity-Nano-Preview-SCM | yes | — | n/a | ScatterMoE, trust_remote_code |
| Ministral3 | Ministral-3-14B | — | yes | n/a | Text-only Mistral3-family |
| Mistral3 | Devstral-24B | — | yes | text-only SFT | QLoRA + model_parallel (2 GPUs) |

**Features tested:** Liger kernel, packing (BFD), Liger + FSDP, multimodal + FSDP.

**Large model support:** Models that don't fit on a single GPU can use `model_parallel: true` with QLoRA to distribute across multiple GPUs via `device_map`. FP8-quantized models (e.g. Devstral) are automatically re-quantized to 4-bit when `load_in_4bit` is set.

## Config inheritance

Training configs support a `base_config` field that loads defaults from another YAML file. Child values override base values. Chains are supported (a base can reference its own base).

```yaml
# train.yaml
base_config: base.yaml
num_train_epochs: 3
learning_rate: 1.0e-5  # overrides base.yaml
```

## Project layout

```
my-project/
  base.yaml       # Model, LoRA, training defaults
  data.yaml       # Datasets, preprocessing, eval split
  train.yaml      # Inherits base.yaml, references data.yaml
  prepared/        # Output of loft prepare (parquet + metadata)
  output/          # Training checkpoints and final model
```

## License

Apache 2.0
