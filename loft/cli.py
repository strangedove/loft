import importlib.resources as resources
import os
import sys

from accelerate.commands.launch import launch_command, launch_command_parser

from .scripts.prepare import main as prepare_main
from .scripts.prepare import make_parser as make_prepare_parser
from .scripts.sft import make_parser as make_sft_parser
from .scripts.utils import TrlParser


def _set_wandb_env_from_config(config_path: str) -> None:
    """
    Read wandb_project from config and set WANDB_PROJECT env var if not already set.
    """
    import yaml

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}

        if config.get("base_config"):
            base_path = config["base_config"]
            if not os.path.isabs(base_path):
                base_path = os.path.join(os.path.dirname(config_path), base_path)
            if os.path.exists(base_path):
                with open(base_path) as f:
                    base_config = yaml.safe_load(f) or {}
                if "wandb_project" in base_config and "wandb_project" not in config:
                    config["wandb_project"] = base_config["wandb_project"]

        if config.get("wandb_project") and not os.environ.get("WANDB_PROJECT"):
            os.environ["WANDB_PROJECT"] = config["wandb_project"]

    except Exception:
        pass


_DATA_CONFIG_TEMPLATE = """\
# Data config — defines which datasets to combine and how to preprocess them.
# This file is model-agnostic and reusable across different training runs.
#
# Usage:
#   loft prepare train.yaml          (referenced via train.yaml's data_config field)
#   loft prepare train.yaml --debug  (preview tokenization for one sample per dataset)

# Dataset list — each entry loads a HuggingFace dataset or local path
datasets:
  - path:                     # HF dataset name or local path
    # split: train            # dataset split (default: train)
    # subset:                 # number of samples (int) or fraction (float) to use
    # system_message:         # per-dataset system message override
    # truncation_strategy:    # per-dataset override: truncate, drop, split, truncate_turns
    # max_length:             # per-dataset max token length override

  # - path:                   # add more datasets here
  #   subset: 5000

# Shuffle settings
shuffle_datasets: true        # shuffle each dataset individually before combining
shuffle_combined: true        # shuffle the combined dataset
shuffle_seed: 42

# Eval split
eval_split: 0.05              # fraction to hold out for evaluation (0 to disable)
split_seed: 42

# Preprocessing
assistant_only_loss: true     # only compute loss on assistant turns
# last_assistant_only_loss: false
# train_on_incomplete_assistant: false
# fix_turn_order: false       # fix strict turn order (adds filler user msg if needed)
# default_system_message:     # system message for conversations that don't have one
# truncation_strategy: truncate  # global default: truncate, drop, split, truncate_turns
"""

_BASE_CONFIG_TEMPLATE = """\
# Base training config — shared settings inherited by training configs via base_config.
# Put model-specific and hardware-specific defaults here.
#
# Training configs inherit these values and can override any of them.

# Model
model_name_or_path:           # HF model name or local path

# Precision & performance
bf16: true
gradient_checkpointing: true
# attn_implementation: flash_attention_2

# Sequence length
max_length: 4096
# truncation_strategy: truncate

# Training hyperparameters
learning_rate: 2.0e-5
warmup_ratio: 0.05
weight_decay: 0.01
lr_scheduler_type: cosine
per_device_train_batch_size: 2
gradient_accumulation_steps: 4

# LoRA (comment out use_peft to do full fine-tuning)
use_peft: true
lora_r: 32
lora_alpha: 64
lora_dropout: 0.05
# lora_target_modules:        # default: all linear layers
# load_in_4bit: false         # QLoRA

# Logging & saving
logging_steps: 10
save_strategy: steps
save_steps: 200
# eval_strategy: steps
# eval_steps: 200
report_to: wandb
# wandb_project:              # set WANDB_PROJECT or use this field

# Output
output_dir: ./output
"""

_TRAIN_CONFIG_TEMPLATE = """\
# Training config — the main config file for a training run.
# Inherits from base.yaml and references data.yaml for dataset preparation.
#
# Usage:
#   loft prepare train.yaml   (prepare datasets)
#   loft train train.yaml     (run training)

# Config inheritance
base_config: base.yaml

# Data
data_config: data.yaml
prepared_dataset: ./prepared  # where prepared data is saved/loaded

# Training
num_train_epochs: 3
# max_steps: -1               # override num_train_epochs with exact step count

# Per-run overrides (uncomment to override base.yaml values)
# learning_rate: 1.0e-5
# max_length: 8192
# per_device_train_batch_size: 1
# gradient_accumulation_steps: 8
# lora_r: 64

# Convenience scheduling (auto-calculates save_steps / eval_steps)
# saves_per_epoch: 2
# evals_per_epoch: 4

# Output (overrides base)
# output_dir: ./output/my-run
"""


def _run_init(project_name: str) -> None:
    """Create a new loft project directory with template config files."""
    if os.path.exists(project_name):
        print(f"Error: '{project_name}' already exists.")
        sys.exit(1)

    os.makedirs(project_name)

    configs = {
        "data.yaml": _DATA_CONFIG_TEMPLATE,
        "base.yaml": _BASE_CONFIG_TEMPLATE,
        "train.yaml": _TRAIN_CONFIG_TEMPLATE,
    }
    for filename, content in configs.items():
        path = os.path.join(project_name, filename)
        with open(path, "w") as f:
            f.write(content)

    print(f"Created project '{project_name}/' with:")
    for filename in configs:
        print(f"  {project_name}/{filename}")
    print()
    print("Next steps:")
    print(f"  1. Edit {project_name}/base.yaml    — set model_name_or_path and training defaults")
    print(f"  2. Edit {project_name}/data.yaml    — add your datasets")
    print(f"  3. Edit {project_name}/train.yaml   — adjust epochs, overrides, output")
    print(f"  4. loft prepare {project_name}/train.yaml")
    print(f"  5. loft train {project_name}/train.yaml")


def main():
    # Normalize positional config: "loft sft config.yaml" → "loft sft --config config.yaml"
    if len(sys.argv) >= 3 and "--config" not in sys.argv[2:]:
        candidate = sys.argv[2]
        if not candidate.startswith("-") and (candidate.endswith(".yaml") or candidate.endswith(".yml")):
            sys.argv.insert(2, "--config")

    # Accept command aliases
    if len(sys.argv) >= 2:
        _aliases = {"train": "sft", "prep": "prepare"}
        if sys.argv[1] in _aliases:
            sys.argv[1] = _aliases[sys.argv[1]]

    if len(sys.argv) >= 2:
        command = sys.argv[1]

        if command == "init":
            name = sys.argv[2] if len(sys.argv) >= 3 else "loft-project"
            _run_init(name)
            return

        if command == "sft":
            for i, arg in enumerate(sys.argv):
                if arg == "--config" and i + 1 < len(sys.argv):
                    _set_wandb_env_from_config(sys.argv[i + 1])
                    break

        if command == "prepare":
            prepare_parser = make_prepare_parser()
            prepare_args = prepare_parser.parse_args(sys.argv[2:])
            prepare_main(
                prepare_args.config,
                prepare_args.output,
                is_dry_run=prepare_args.dry_run,
                is_debug=prepare_args.debug,
                debug_max_tokens=prepare_args.debug_max_tokens,
            )
            return

    parser = TrlParser(prog="loft", usage="loft", allow_abbrev=False)
    subparsers = parser.add_subparsers(help="available commands", dest="command", parser_class=TrlParser)

    make_prepare_parser(subparsers)
    make_sft_parser(subparsers, include_dataset_args=False)

    args, launch_args = parser.parse_args_and_config(return_remaining_strings=True)

    if "--accelerate_config" in launch_args:
        config_index = launch_args.index("--accelerate_config")
        config_name = launch_args[config_index + 1]

        if os.path.isfile(config_name):
            accelerate_config_path = config_name
        else:
            raise ValueError(
                f"Accelerate config '{config_name}' is not a valid file path. "
                "Provide a path to an accelerate config YAML file."
            )

        launch_args.pop(config_index)
        launch_args.pop(config_index)
        launch_args = ["--config_file", str(accelerate_config_path)] + launch_args

    # Filter out custom fields not recognized by accelerate/transformers
    custom_fields_to_filter = [
        "--wandb_project",
        "--saves_per_epoch",
        "--evals_per_epoch",
    ]
    for field in custom_fields_to_filter:
        if field in launch_args:
            idx = launch_args.index(field)
            launch_args.pop(idx)
            if idx < len(launch_args) and not launch_args[idx].startswith("--"):
                launch_args.pop(idx)

    if args.command == "prepare":
        prepare_parser = make_prepare_parser()
        prepare_args = prepare_parser.parse_args(sys.argv[2:])
        prepare_main(
            prepare_args.config,
            prepare_args.output,
            is_dry_run=prepare_args.dry_run,
            is_debug=prepare_args.debug,
            debug_max_tokens=prepare_args.debug_max_tokens,
        )

    elif args.command == "sft":
        sft_training_script = resources.files("loft.scripts").joinpath("sft.py")
        training_script_args = sys.argv[2:]
        args = launch_command_parser().parse_args(launch_args + [str(sft_training_script)] + training_script_args)
        launch_command(args)


if __name__ == "__main__":
    main()
