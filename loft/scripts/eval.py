"""Run test prompts against a model (with optional LoRA) using vLLM offline inference.

Usage:
    loft eval train.yaml                                # from training config
    loft eval train.yaml --checkpoint checkpoint-500    # specific checkpoint
    loft eval --model /path/to/model                    # explicit model
    loft eval --model base --lora /path/to/lora         # base + lora side-by-side
    loft eval train.yaml --all-prompts                  # all prompts
    loft eval train.yaml --prompts id1 id2              # specific prompts
"""

import argparse
import importlib.resources as resources
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from .utils import resolve_config_inheritance

# Default subset for quick testing (5 prompts across categories)
DEFAULT_SELECTED = [
    "instruct_writing_1",
    "character_roleplay_2",
    "interactive_fiction_2",
    "general_assistant_1",
    "general_assistant_6",
]


def load_prompts(path: Optional[str] = None) -> dict:
    """Load test prompts from JSON file. Uses bundled prompts if no path given."""
    if path is None:
        prompts_file = resources.files("loft.eval").joinpath("test_prompts.json")
        data = prompts_file.read_text(encoding="utf-8")
        prompts = json.loads(data)
    else:
        with open(path) as f:
            prompts = json.load(f)
    return {p["id"]: p for p in prompts}


def detect_lora(path: str) -> bool:
    """Check if a path contains a LoRA adapter (has adapter_config.json)."""
    return os.path.isfile(os.path.join(path, "adapter_config.json"))


def load_config(config_path: str) -> dict:
    """Load a training config YAML with base_config inheritance."""
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config {config_path} must be a YAML mapping, got {type(config).__name__}.")
    config = resolve_config_inheritance(config, config_path)
    return config


def resolve_paths(args) -> dict:
    """Resolve model, lora, and output paths from config + CLI overrides.

    Returns dict with keys: model, lora (or None), output, config (or None).
    """
    config = None
    model = args.model
    lora = args.lora
    output = args.output

    # Load config if provided
    if args.config:
        config = load_config(args.config)
        if not model:
            model = config.get("model_name_or_path")
        if not lora:
            output_dir = config.get("output_dir", "./output")
            if args.checkpoint:
                # Try exact name first, then checkpoint-N pattern
                candidate = os.path.join(output_dir, args.checkpoint)
                if not os.path.isdir(candidate) and not args.checkpoint.startswith("checkpoint-"):
                    candidate = os.path.join(output_dir, f"checkpoint-{args.checkpoint}")
                lora = candidate
            else:
                # Use output_dir itself (final adapter)
                lora = output_dir

    # CLI overrides
    if args.lora:
        lora = args.lora

    # Auto-detect: is the lora path actually a LoRA, or a full model?
    if lora and os.path.isdir(lora) and not detect_lora(lora):
        # Not a LoRA — treat as a full fine-tuned model
        print(f"No adapter_config.json in {lora}, treating as full model (not LoRA).")
        model = lora
        lora = None

    if not model:
        print("Error: No model specified. Use --model or provide a config with model_name_or_path.", file=sys.stderr)
        sys.exit(1)

    # Default output path
    if not output:
        if config and config.get("output_dir"):
            output = os.path.join(config["output_dir"], f"eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        else:
            output = f"eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    return {
        "model": model,
        "lora": lora,
        "output": output,
        "config": config,
    }


def _get_lora_rank(lora_path: str) -> int:
    """Read max LoRA rank from adapter_config.json."""
    config_path = os.path.join(lora_path, "adapter_config.json")
    try:
        with open(config_path) as f:
            cfg = json.load(f)
        return cfg.get("r", 64)
    except Exception:
        return 64


def run_eval(args):
    """Main eval function: load model, generate, save results."""
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer

    paths = resolve_paths(args)
    model_path = paths["model"]
    lora_path = paths["lora"]
    output_prefix = paths["output"]

    # Skip LoRA if --base-only
    if args.base_only:
        lora_path = None

    # Load prompts
    all_prompts = load_prompts(args.prompt_file)

    if args.all_prompts:
        selected = list(all_prompts.values())
    elif args.prompts:
        missing = [pid for pid in args.prompts if pid not in all_prompts]
        if missing:
            print(f"Warning: Unknown prompt IDs: {missing}", file=sys.stderr)
        selected = [all_prompts[pid] for pid in args.prompts if pid in all_prompts]
    else:
        selected = [all_prompts[pid] for pid in DEFAULT_SELECTED if pid in all_prompts]

    if not selected:
        print("Error: No prompts selected.", file=sys.stderr)
        sys.exit(1)

    # Load tokenizer for chat template
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=args.trust_remote_code,
    )

    # Build formatted prompts
    formatted_prompts = []
    for prompt in selected:
        text = tokenizer.apply_chat_template(
            prompt["messages"],
            tokenize=False,
            add_generation_prompt=True,
        )
        formatted_prompts.append(text)

    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_tokens,
    )

    use_lora = lora_path is not None and os.path.isdir(lora_path)

    # Print plan
    print(f"\nModel:       {model_path}")
    if use_lora:
        print(f"LoRA:        {lora_path}")
    print(f"Prompts:     {len(selected)}")
    print(f"Sampling:    temp={args.temperature}, min_p={args.min_p}, "
          f"rep_penalty={args.repetition_penalty}, max_tokens={args.max_tokens}")
    print(f"Output:      {output_prefix}.md / .json")
    print()

    # Initialize vLLM
    llm_kwargs = dict(
        model=model_path,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=args.trust_remote_code,
    )

    if use_lora:
        lora_rank = _get_lora_rank(lora_path)
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = lora_rank

    print("Loading model...")
    llm = LLM(**llm_kwargs)

    results = {}

    if use_lora:
        # Generate with LoRA
        lora_name = os.path.basename(os.path.normpath(lora_path))
        lora_request = LoRARequest(lora_name, 1, lora_path)

        print(f"\nGenerating with LoRA ({lora_name})...")
        lora_outputs = llm.generate(formatted_prompts, sampling_params, lora_request=lora_request)
        results[f"lora:{lora_name}"] = {}
        for prompt, output in zip(selected, lora_outputs):
            text = output.outputs[0].text
            results[f"lora:{lora_name}"][prompt["id"]] = {
                "category": prompt["category"],
                "output": text,
            }
            preview = text[:100].replace("\n", " ")
            print(f"  [{prompt['id']}] {preview}...")

        # Generate without LoRA (base comparison)
        print(f"\nGenerating with base model...")
        base_outputs = llm.generate(formatted_prompts, sampling_params)
        results["base"] = {}
        for prompt, output in zip(selected, base_outputs):
            text = output.outputs[0].text
            results["base"][prompt["id"]] = {
                "category": prompt["category"],
                "output": text,
            }
            preview = text[:100].replace("\n", " ")
            print(f"  [{prompt['id']}] {preview}...")
    else:
        # Single model (no LoRA)
        model_label = os.path.basename(os.path.normpath(model_path))
        print(f"\nGenerating with {model_label}...")
        outputs = llm.generate(formatted_prompts, sampling_params)
        results[model_label] = {}
        for prompt, output in zip(selected, outputs):
            text = output.outputs[0].text
            results[model_label][prompt["id"]] = {
                "category": prompt["category"],
                "output": text,
            }
            preview = text[:100].replace("\n", " ")
            print(f"  [{prompt['id']}] {preview}...")

    # Metadata
    meta = {
        "timestamp": datetime.now().isoformat(),
        "model": model_path,
        "lora": lora_path,
        "prompts": [p["id"] for p in selected],
        "sampling": {
            "temperature": args.temperature,
            "min_p": args.min_p,
            "repetition_penalty": args.repetition_penalty,
            "max_tokens": args.max_tokens,
        },
    }

    # Write outputs
    os.makedirs(os.path.dirname(output_prefix) or ".", exist_ok=True)

    write_json(results, meta, output_prefix)
    write_markdown(results, selected, meta, output_prefix)

    print(f"\nDone! Saved {output_prefix}.json and {output_prefix}.md")


def write_json(results: dict, meta: dict, path: str):
    """Write raw results + metadata as JSON."""
    data = {"meta": meta, "results": results}
    with open(f"{path}.json", "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_markdown(results: dict, prompts: list[dict], meta: dict, path: str):
    """Write human-readable markdown output."""
    model_names = list(results.keys())

    with open(f"{path}.md", "w") as f:
        f.write("# Eval Results\n\n")
        f.write(f"**Model**: {meta['model']}\n\n")
        if meta.get("lora"):
            f.write(f"**LoRA**: {meta['lora']}\n\n")
        f.write(f"**Prompts**: {len(prompts)}\n\n")
        sampling = meta["sampling"]
        f.write(f"**Sampling**: temperature={sampling['temperature']}, "
                f"min_p={sampling['min_p']}, "
                f"repetition_penalty={sampling['repetition_penalty']}, "
                f"max_tokens={sampling['max_tokens']}\n\n")
        f.write(f"**Timestamp**: {meta['timestamp']}\n\n")
        f.write("---\n\n")

        for prompt in prompts:
            pid = prompt["id"]
            cat = prompt["category"]
            user_msgs = [m for m in prompt["messages"] if m["role"] == "user"]
            user_msg = user_msgs[-1]["content"] if user_msgs else "(no user message)"
            sys_msgs = [m for m in prompt["messages"] if m["role"] == "system"]
            sys_msg = sys_msgs[0]["content"] if sys_msgs else "(no system message)"

            f.write(f"## {pid} ({cat})\n\n")
            f.write(f"**System**: {sys_msg}\n\n")
            f.write(f"**User**: {user_msg}\n\n")

            for model_name in model_names:
                if pid in results[model_name]:
                    output = results[model_name][pid]["output"]
                    label = model_name.upper() if model_name == "base" else model_name
                    f.write(f"### {label}\n\n")
                    f.write(f"{output}\n\n")
                    f.write("---\n\n")


def make_parser(subparsers: Optional[argparse._SubParsersAction] = None):
    """Create the argument parser for the eval command."""
    if subparsers is not None:
        parser = subparsers.add_parser(
            "eval",
            help="Run test prompts against a model checkpoint for evaluation",
        )
    else:
        parser = argparse.ArgumentParser(
            description="Run test prompts against a model checkpoint for evaluation",
        )

    # Model sources (config or explicit)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to training config YAML (reads model_name_or_path and output_dir)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Explicit model path (overrides config model_name_or_path)",
    )
    parser.add_argument(
        "--lora",
        type=str,
        default=None,
        help="Explicit LoRA adapter path (overrides config output_dir)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint directory name within output_dir (e.g. checkpoint-500)",
    )

    # Prompt selection
    parser.add_argument(
        "--prompts",
        nargs="*",
        default=None,
        help="Specific prompt IDs to run (default: 5-prompt quick subset)",
    )
    parser.add_argument(
        "--all-prompts",
        action="store_true",
        default=False,
        help="Run all prompts instead of the default subset",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="Path to custom test_prompts.json",
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path prefix (default: {output_dir}/eval-{timestamp})",
    )

    # Sampling parameters
    parser.add_argument("--temperature", type=float, default=1.1, help="Sampling temperature (default: 1.1)")
    parser.add_argument("--min-p", type=float, default=0.15, help="Min-p sampling (default: 0.15)")
    parser.add_argument("--repetition-penalty", type=float, default=1.1, help="Repetition penalty (default: 1.1)")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max generation tokens (default: 1024)")

    # vLLM parameters
    parser.add_argument("--max-model-len", type=int, default=4096, help="vLLM max model length (default: 4096)")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size (default: 1)")

    # Flags
    parser.add_argument("--base-only", action="store_true", default=False, help="Skip LoRA, only test base model")
    parser.add_argument("--trust-remote-code", action="store_true", default=False, help="Trust remote code in model")

    return parser


def main(args=None):
    """Entry point for loft eval."""
    if args is None:
        parser = make_parser()
        args = parser.parse_args()

    # Validate: need at least --config or --model
    if not args.config and not args.model:
        print("Error: Provide either --config or --model.", file=sys.stderr)
        sys.exit(1)

    run_eval(args)


if __name__ == "__main__":
    main()
