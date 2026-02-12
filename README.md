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
