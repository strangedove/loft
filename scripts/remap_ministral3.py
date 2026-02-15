#!/usr/bin/env python3
"""
Remap a Ministral3 (transformers 5.x) model to MistralForCausalLM (4.57).

Creates a new directory that symlinks all weight/tokenizer files from the
source model and writes a remapped config.json compatible with transformers
4.57's MistralForCausalLM.  The Llama 4 attention-scaling parameters are
preserved as private config fields so the monkey-patch in
``loft.models.ministral3_compat`` can pick them up at runtime.

Usage:
    python scripts/remap_ministral3.py <source> <dest>

    source: HuggingFace model ID (resolved from cache) or local path.
    dest:   Output directory for the remapped model.

Example:
    python scripts/remap_ministral3.py \
        estrogen/SomehowMinistralReturnedButCommunist \
        ./models/ministral3-remapped
"""

import argparse
import json
import os
import sys
from pathlib import Path


def resolve_hf_cache_path(model_id: str) -> Path:
    """Resolve a HuggingFace model ID to its snapshot path in the local cache."""
    cache_dir = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    # HF cache uses -- as separator: "org/model" -> "models--org--model"
    folder_name = "models--" + model_id.replace("/", "--")
    model_dir = cache_dir / folder_name

    if not model_dir.exists():
        print(f"Error: Model not found in HF cache at {model_dir}", file=sys.stderr)
        sys.exit(1)

    snapshots_dir = model_dir / "snapshots"
    if not snapshots_dir.exists():
        print(f"Error: No snapshots directory at {snapshots_dir}", file=sys.stderr)
        sys.exit(1)

    # Use the most recently modified snapshot
    snapshots = sorted(snapshots_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    if not snapshots:
        print(f"Error: No snapshots found in {snapshots_dir}", file=sys.stderr)
        sys.exit(1)

    return snapshots[0]


def remap_config(config: dict) -> dict:
    """
    Remap a Ministral3 config.json to be loadable as MistralForCausalLM on
    transformers 4.57.

    Changes:
      - model_type: "ministral3" -> "mistral"
      - architectures: ["Ministral3ForCausalLM"] -> ["MistralForCausalLM"]
      - rope_parameters (5.x dict) -> rope_theta (top-level) + rope_scaling (4.57 dict)
      - Stash llama_4_scaling_beta as _ministral3_llama4_beta for the monkey-patch
      - Update transformers_version to 4.57.3
    """
    out = dict(config)

    out["model_type"] = "mistral"
    out["architectures"] = ["MistralForCausalLM"]
    out["transformers_version"] = "4.57.3"

    rope_params = config.get("rope_parameters", {})

    # Extract rope_theta to top-level (4.57 MistralConfig expects this)
    out["rope_theta"] = rope_params.get("rope_theta", 10000.0)

    # Build rope_scaling dict for 4.57's YARN implementation
    rope_scaling = {}
    for key in ("rope_type", "type", "factor", "original_max_position_embeddings",
                "beta_fast", "beta_slow", "mscale", "mscale_all_dim"):
        if key in rope_params:
            rope_scaling[key] = rope_params[key]
    out["rope_scaling"] = rope_scaling

    # Stash Llama 4 scaling params for the monkey-patch
    llama4_beta = rope_params.get("llama_4_scaling_beta")
    if llama4_beta is not None:
        out["_ministral3_llama4_beta"] = llama4_beta
        out["_ministral3_orig_max_pos"] = rope_params.get(
            "original_max_position_embeddings", 16384
        )

    # Remove the 5.x-only field
    out.pop("rope_parameters", None)

    return out


def main():
    parser = argparse.ArgumentParser(
        description="Remap Ministral3 model config for transformers 4.57 compatibility."
    )
    parser.add_argument("source", help="HuggingFace model ID or local path to the source model.")
    parser.add_argument("dest", help="Output directory for the remapped model.")
    args = parser.parse_args()

    # Resolve source path
    source = Path(args.source)
    if not source.exists():
        # Try resolving as HF model ID
        source = resolve_hf_cache_path(args.source)
    source = source.resolve()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Source: {source}")
    print(f"Dest:   {dest}")

    # Read and remap config
    config_path = source / "config.json"
    if not config_path.exists():
        print(f"Error: No config.json found at {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)

    if config.get("model_type") != "ministral3":
        print(f"Warning: model_type is '{config.get('model_type')}', not 'ministral3'. Proceeding anyway.")

    remapped = remap_config(config)

    # Write remapped config
    with open(dest / "config.json", "w") as f:
        json.dump(remapped, f, indent=2)
        f.write("\n")
    print("Wrote remapped config.json")

    # Symlink everything else from source
    skipped = {"config.json"}
    linked = 0
    for item in source.iterdir():
        if item.name in skipped:
            continue
        dest_item = dest / item.name
        if dest_item.exists() or dest_item.is_symlink():
            dest_item.unlink()

        # Resolve through any existing symlinks to get the actual blob
        real_path = item.resolve()
        os.symlink(real_path, dest_item)
        linked += 1

    print(f"Symlinked {linked} files from source.")
    print()
    print("Remapped config summary:")
    print(f"  model_type:    {config['model_type']} -> {remapped['model_type']}")
    print(f"  architectures: {config['architectures']} -> {remapped['architectures']}")
    print(f"  rope_theta:    {remapped['rope_theta']}")
    print(f"  rope_scaling:  type={remapped['rope_scaling'].get('rope_type')}, factor={remapped['rope_scaling'].get('factor')}")
    if "_ministral3_llama4_beta" in remapped:
        print(f"  llama4_beta:   {remapped['_ministral3_llama4_beta']} (stashed for monkey-patch)")
        print(f"  orig_max_pos:  {remapped['_ministral3_orig_max_pos']}")
    print()
    print(f"Use '{dest}' as model_name_or_path for training.")


if __name__ == "__main__":
    main()
