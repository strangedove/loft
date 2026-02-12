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
