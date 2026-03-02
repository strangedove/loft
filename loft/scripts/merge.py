"""
LoRA merge script — tensor-wise merge without loading the full model into memory.

Reads the base model and LoRA adapter, applies LoRA weights shard-by-shard,
and writes the merged model to an output directory.

Usage:
    loft merge configs/train.yaml
    loft merge configs/train.yaml --checkpoint checkpoint-1000
    loft merge configs/train.yaml --weight 1.5
    loft merge --base-model ./model --lora ./lora-adapter --output ./merged
"""

import argparse
import json
import logging
import math
import os
import re
import shutil
from pathlib import Path
from typing import Optional

import safetensors
import safetensors.torch
import torch
import yaml


logger = logging.getLogger(__name__)


def resolve_path(path_or_repo_id: str) -> Path:
    """Resolve a local path or HuggingFace repo ID to a local directory."""
    path = Path(path_or_repo_id)
    if path.is_dir():
        return path
    if "/" in path_or_repo_id and not path.exists():
        try:
            from huggingface_hub import snapshot_download

            downloaded = snapshot_download(
                repo_id=path_or_repo_id,
                local_dir_use_symlinks=False,
                repo_type="model",
            )
            return Path(downloaded)
        except Exception as e:
            raise RuntimeError(f"Failed to download '{path_or_repo_id}': {e}") from e
    raise FileNotFoundError(f"'{path_or_repo_id}' is not a local directory or valid HuggingFace repo ID.")


def _load_adapter_config(lora_path: Path) -> dict:
    """Load adapter_config.json from a LoRA directory."""
    config_file = lora_path / "adapter_config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"No adapter_config.json found in {lora_path}")
    with open(config_file) as f:
        return json.load(f)


def _compute_scale(
    adapter_config: dict,
    weight: Optional[float] = None,
    scale_override: Optional[float] = None,
) -> float:
    """Compute the LoRA scaling factor.

    Base scale is alpha/r (or alpha/sqrt(r) for rslora) from the adapter config.
    ``weight`` multiplies the base scale (e.g. 0.5 = half strength).
    ``scale_override`` replaces the base scale entirely.
    """
    if scale_override is not None:
        return scale_override

    alpha = adapter_config["lora_alpha"]
    r = adapter_config["r"]
    use_rslora = adapter_config.get("use_rslora", False)

    if use_rslora:
        scale = alpha / math.sqrt(r)
    else:
        scale = alpha / r

    if weight is not None:
        scale *= weight

    return scale


def _find_lora_weights(key: str, lora_state: dict):
    """Find matching LoRA A and B matrices for a given base model key."""
    lora_A = None
    lora_B = None
    prefix = "base_model.model."

    for lora_key, lora_weight in lora_state.items():
        unprefixed = lora_key[len(prefix):] if lora_key.startswith(prefix) else lora_key
        if key.strip(".weight") in unprefixed:
            if "lora_A" in lora_key:
                lora_A = lora_weight
            elif "lora_B" in lora_key:
                lora_B = lora_weight

    if lora_A is not None and lora_B is not None:
        return lora_A, lora_B
    return None, None


def merge(
    base_model_path: str | Path,
    lora_path: str | Path,
    output_path: str | Path,
    weight: Optional[float] = None,
    scale: Optional[float] = None,
    use_gpu: bool = True,
) -> Path:
    """Merge a LoRA adapter into a base model, shard by shard.

    Args:
        base_model_path: Path or HF repo ID for the base model.
        lora_path: Path or HF repo ID for the LoRA adapter.
        output_path: Where to write the merged model.
        weight: Optional multiplier on the LoRA scale factor.
            None = use the scale from adapter_config (alpha/r or alpha/sqrt(r)).
            1.0 = same as None. 0.5 = half strength. 2.0 = double strength.
        scale: Optional direct scale factor, replacing the adapter config value.
            Mutually exclusive with weight.
        use_gpu: Use CUDA for merge computation if available.

    Returns:
        The output path.
    """
    base_model_path = resolve_path(str(base_model_path))
    lora_path = resolve_path(str(lora_path))
    output_path = Path(output_path)
    os.makedirs(output_path, exist_ok=True)

    adapter_config = _load_adapter_config(lora_path)
    effective_scale = _compute_scale(adapter_config, weight=weight, scale_override=scale)
    if scale is not None:
        logger.info(f"LoRA scale factor: {effective_scale:.4f} (explicit --scale)")
    elif weight is not None:
        logger.info(f"LoRA scale factor: {effective_scale:.4f} (base * weight={weight})")
    else:
        logger.info(f"LoRA scale factor: {effective_scale:.4f} (from adapter config)")

    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

    # Load LoRA state dict
    logger.info("Loading LoRA weights...")
    if (lora_path / "adapter_model.safetensors").exists():
        lora_state = safetensors.torch.load_file(lora_path / "adapter_model.safetensors")
        if device == "cuda":
            for key in lora_state:
                lora_state[key] = lora_state[key].to("cuda")
    elif (lora_path / "adapter_model.bin").exists():
        lora_state = torch.load(lora_path / "adapter_model.bin", map_location=device)
    else:
        raise FileNotFoundError(f"No adapter_model.safetensors or adapter_model.bin in {lora_path}")

    # Find model shards
    shards = sorted(base_model_path.glob("model*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No model*.safetensors files found in {base_model_path}")

    # Copy non-model files from base model
    logger.info("Copying non-model files from base model...")
    for filepath in base_model_path.iterdir():
        if filepath in shards or filepath.is_dir():
            continue
        if filepath.name.startswith("."):
            continue
        if filepath.suffix == ".gguf":
            continue
        if filepath.suffix == ".md":
            continue
        if filepath.suffix == ".safetensors" and "model" in filepath.name:
            continue
        shutil.copy(filepath, output_path)

    # Merge shards
    found = 0
    for i, shard in enumerate(shards, 1):
        logger.info(f"Processing shard {i}/{len(shards)}: {shard.name}")
        tensors = {}
        with safetensors.safe_open(shard, framework="pt", device=device) as f:
            metadata = f.metadata()
            for key in f.keys():
                tensor = f.get_tensor(key)
                # Strip model prefix for LoRA key lookup (composite models may
                # have 'model.' or 'language_model.' prefixes)
                lora_key_lookup = re.sub(r"^(model|language_model)\.", "", key)

                lora_A, lora_B = _find_lora_weights(lora_key_lookup, lora_state)
                if lora_A is not None:
                    found += 1
                    old_dtype = tensor.dtype
                    tensor = tensor.to(torch.float32)
                    tensor += effective_scale * lora_B.to(torch.float32) @ lora_A.to(torch.float32)
                    tensor = tensor.to(old_dtype)
                tensors[key] = tensor

        safetensors.torch.save_file(tensors, output_path / shard.name, metadata=metadata)

    logger.info(f"Applied LoRA to {found} tensors across {len(shards)} shards.")

    # Copy tokenizer files from LoRA directory if present (may have been modified)
    for tok_file in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.model"]:
        src = lora_path / tok_file
        if src.exists():
            shutil.copy(src, output_path)

    # Copy training README.md from LoRA adapter (contains model card with
    # hyperparams, dataset stats, configs) so it survives the merge
    lora_readme = lora_path / "README.md"
    if lora_readme.exists():
        shutil.copy(lora_readme, output_path / "README.md")
        logger.info("Copied training README.md from LoRA adapter to merged output.")

    logger.info(f"Merge complete. Output: {output_path}")
    return output_path


def _resolve_from_config(config_path: str, checkpoint: Optional[str] = None):
    """Read base_model, lora_path, and output_path from a loft training config.

    Returns (base_model_path, lora_path, output_path).
    """
    from loft.scripts.utils import resolve_config_inheritance

    with open(config_path) as f:
        config = yaml.safe_load(f) or {}
    config = resolve_config_inheritance(config, config_path)

    base_model = config.get("model_name_or_path")
    if not base_model:
        raise ValueError(f"Config {config_path} has no model_name_or_path")

    output_dir = config.get("output_dir", ".")

    # Determine LoRA path: either a specific checkpoint or the final adapter
    if checkpoint:
        lora_path = os.path.join(output_dir, checkpoint)
    else:
        lora_path = output_dir

    # If the output_dir itself doesn't have adapter_config.json, check for
    # the latest checkpoint inside it
    if not os.path.exists(os.path.join(lora_path, "adapter_config.json")):
        # Look for checkpoint-* subdirectories and pick the highest-numbered one
        checkpoints = sorted(
            Path(output_dir).glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0,
        )
        if checkpoints:
            lora_path = str(checkpoints[-1])
            logger.info(f"Using latest checkpoint: {lora_path}")
        else:
            raise FileNotFoundError(
                f"No adapter_config.json found in {output_dir} and no checkpoint-* subdirectories. "
                "Has training completed or saved a checkpoint?"
            )

    # Default merged output path: <output_dir>/merged
    merged_path = os.path.join(output_dir, "merged")

    return base_model, lora_path, merged_path


def make_parser(subparsers: Optional[argparse._SubParsersAction] = None):
    """Create the argument parser for the merge command."""
    if subparsers is not None:
        parser = subparsers.add_parser(
            "merge",
            help="Merge a LoRA adapter into a base model",
        )
    else:
        parser = argparse.ArgumentParser(
            description="Merge a LoRA adapter into a base model",
        )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to loft training config YAML (reads base model and output dir)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model path or HF repo ID (overrides config)",
    )
    parser.add_argument(
        "--lora",
        type=str,
        default=None,
        help="LoRA adapter path or HF repo ID (overrides config)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for merged model (default: <output_dir>/merged)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint directory name inside output_dir (e.g. 'checkpoint-1000')",
    )
    parser.add_argument(
        "--weight",
        type=float,
        default=None,
        help="Multiplier on the adapter's scale factor (e.g. 0.5 = half strength, 2.0 = double)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Direct scale factor, replacing the adapter config value entirely",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Force CPU merge (default: use GPU if available)",
    )

    return parser


def main(args):
    """Entry point for the merge command."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    if args.weight is not None and args.scale is not None:
        raise SystemExit("Error: --weight and --scale are mutually exclusive.")

    if args.config:
        base_model, lora_path, default_output = _resolve_from_config(
            args.config, checkpoint=args.checkpoint,
        )
        # CLI args override config-derived values
        base_model = args.base_model or base_model
        lora_path = args.lora or lora_path
        output_path = args.output or default_output
    elif args.base_model and args.lora:
        base_model = args.base_model
        lora_path = args.lora
        if args.checkpoint:
            lora_path = os.path.join(lora_path, args.checkpoint)
        output_path = args.output
        if not output_path:
            # Default: sibling directory named "merged" next to the lora path
            output_path = os.path.join(os.path.dirname(lora_path.rstrip("/")), "merged")
    else:
        raise SystemExit(
            "Error: Provide either --config (loft training config) or both --base-model and --lora.\n"
            "\n"
            "Examples:\n"
            "  loft merge --config configs/train.yaml\n"
            "  loft merge --config configs/train.yaml --checkpoint checkpoint-1000\n"
            "  loft merge --base-model ./model --lora ./lora-adapter --output ./merged"
        )

    logger.info(f"Base model: {base_model}")
    logger.info(f"LoRA adapter: {lora_path}")
    logger.info(f"Output: {output_path}")

    merge(
        base_model_path=base_model,
        lora_path=lora_path,
        output_path=output_path,
        weight=args.weight,
        scale=args.scale,
        use_gpu=not args.no_gpu,
    )
