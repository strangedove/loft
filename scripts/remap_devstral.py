#!/usr/bin/env python3
"""
Convert Devstral (Mistral3ForConditionalGeneration, FP8, transformers 5.x)
to a text-only MistralForCausalLM (bf16, transformers 4.57).

This script:
  1. Extracts text_config from the multimodal config -> top-level MistralForCausalLM config
  2. Remaps rope_parameters (5.x) -> rope_theta + rope_scaling (4.57)
  3. Stashes Llama 4 attention-scaling params for the ministral3_compat monkey-patch
  4. Strips vision_tower and multi_modal_projector weights
  5. Renames language_model.X -> X
  6. Dequantizes FP8 (float8_e4m3fn) weights to bf16 using weight_scale_inv
  7. Drops activation_scale and weight_scale_inv tensors (not needed after dequant)
  8. Fixes tokenizer_config.json for transformers 4.57 compatibility

Usage:
    python scripts/remap_devstral.py <source> <dest>

    source: HuggingFace model ID (resolved from cache) or local path.
    dest:   Output directory for the converted model.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def resolve_hf_cache_path(model_id: str) -> Path:
    """Resolve a HuggingFace model ID to its snapshot path in the local cache."""
    cache_dir = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    folder_name = "models--" + model_id.replace("/", "--")
    model_dir = cache_dir / folder_name

    if not model_dir.exists():
        print(f"Error: Model not found in HF cache at {model_dir}", file=sys.stderr)
        sys.exit(1)

    snapshots_dir = model_dir / "snapshots"
    if not snapshots_dir.exists():
        print(f"Error: No snapshots directory at {snapshots_dir}", file=sys.stderr)
        sys.exit(1)

    snapshots = sorted(snapshots_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    if not snapshots:
        print(f"Error: No snapshots found in {snapshots_dir}", file=sys.stderr)
        sys.exit(1)

    return snapshots[0]


def remap_config(config: dict) -> dict:
    """
    Convert Mistral3ForConditionalGeneration config to MistralForCausalLM config.

    Extracts text_config to top-level and remaps rope parameters.
    """
    text_cfg = config.get("text_config", {})

    out = {}
    out["architectures"] = ["MistralForCausalLM"]
    out["model_type"] = "mistral"
    out["transformers_version"] = "4.57.3"

    # Copy text_config fields
    for key in ("attention_dropout", "head_dim", "hidden_act", "hidden_size",
                "initializer_range", "intermediate_size", "max_position_embeddings",
                "num_attention_heads", "num_hidden_layers", "num_key_value_heads",
                "rms_norm_eps", "sliding_window", "use_cache", "vocab_size"):
        if key in text_cfg:
            out[key] = text_cfg[key]

    # Copy top-level fields
    out["bos_token_id"] = config.get("bos_token_id", 1)
    out["eos_token_id"] = config.get("eos_token_id", 2)
    out["dtype"] = "bfloat16"
    out["tie_word_embeddings"] = config.get("tie_word_embeddings", False)

    # Remap rope_parameters -> rope_theta + rope_scaling
    rope_params = text_cfg.get("rope_parameters", {})
    out["rope_theta"] = rope_params.get("rope_theta", 10000.0)

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
            "original_max_position_embeddings", 8192
        )

    return out


def convert_weights(source: Path, dest: Path):
    """
    Read source safetensors, strip vision/projector weights, rename
    language_model.X -> X, dequantize FP8 -> bf16, and write new files.
    """
    # Read the index
    index_path = source / "model.safetensors.index.json"
    with open(index_path, "rb") as f:
        index = json.loads(f.read())

    weight_map = index["weight_map"]

    # Group weights by source file, filtering to text-only
    file_groups = {}
    for key, filename in weight_map.items():
        # Skip vision and projector weights
        if any(prefix in key for prefix in ("vision_tower", "multi_modal_projector")):
            continue
        # Skip FP8 scale tensors (we'll use them during dequant but not save them)
        if key.endswith(".weight_scale_inv") or key.endswith(".activation_scale"):
            continue
        if filename not in file_groups:
            file_groups[filename] = []
        file_groups[filename].append(key)

    new_weight_map = {}
    total_dequantized = 0
    total_copied = 0
    out_file_idx = 1

    # Determine total output files (repack into similar-sized chunks)
    # Process each source file and write output files
    all_tensors = {}

    for src_filename in sorted(file_groups.keys()):
        keys = file_groups[src_filename]
        src_path = source / src_filename

        print(f"  Reading {src_filename} ({len(keys)} text weights)...")

        with safe_open(str(src_path), framework="pt") as f:
            # Also load scale tensors from this file for dequantization
            all_file_keys = list(f.keys())

            for key in keys:
                tensor = f.get_tensor(key)

                # Dequantize FP8 weights
                if tensor.dtype == torch.float8_e4m3fn:
                    scale_key = key.replace(".weight", ".weight_scale_inv")
                    if scale_key in all_file_keys:
                        scale_inv = f.get_tensor(scale_key)
                        tensor = tensor.to(torch.bfloat16) * scale_inv
                        total_dequantized += 1
                    else:
                        print(f"    WARNING: No scale found for {key}, casting directly")
                        tensor = tensor.to(torch.bfloat16)

                # Rename: strip "language_model." prefix
                new_key = key
                if new_key.startswith("language_model."):
                    new_key = new_key[len("language_model."):]

                all_tensors[new_key] = tensor

    # Write output files, splitting at ~5GB boundaries
    max_file_bytes = 5 * 1024 * 1024 * 1024  # 5 GB
    current_tensors = {}
    current_bytes = 0

    for key in sorted(all_tensors.keys()):
        tensor = all_tensors[key]
        tensor_bytes = tensor.nelement() * tensor.element_size()

        if current_bytes + tensor_bytes > max_file_bytes and current_tensors:
            # Write current chunk
            out_name = f"model-{out_file_idx:05d}-of-PLACEHOLDER.safetensors"
            out_path = dest / out_name
            print(f"  Writing {out_name} ({len(current_tensors)} tensors, {current_bytes / 1e9:.1f} GB)")
            save_file(current_tensors, str(out_path))
            for k in current_tensors:
                new_weight_map[k] = out_name
            total_copied += len(current_tensors)
            out_file_idx += 1
            current_tensors = {}
            current_bytes = 0

        current_tensors[key] = tensor
        current_bytes += tensor_bytes

    # Write final chunk
    if current_tensors:
        out_name = f"model-{out_file_idx:05d}-of-PLACEHOLDER.safetensors"
        out_path = dest / out_name
        print(f"  Writing {out_name} ({len(current_tensors)} tensors, {current_bytes / 1e9:.1f} GB)")
        save_file(current_tensors, str(out_path))
        for k in current_tensors:
            new_weight_map[k] = out_name
        total_copied += len(current_tensors)

    total_files = out_file_idx

    # Rename placeholder files and fix weight map
    final_weight_map = {}
    for key, name in new_weight_map.items():
        final_name = name.replace("PLACEHOLDER", f"{total_files:05d}")
        final_weight_map[key] = final_name

    for i in range(1, total_files + 1):
        old_name = f"model-{i:05d}-of-PLACEHOLDER.safetensors"
        new_name = f"model-{i:05d}-of-{total_files:05d}.safetensors"
        old_path = dest / old_name
        new_path = dest / new_name
        if old_path.exists():
            old_path.rename(new_path)

    # Write new index
    new_index = {
        "metadata": {"total_size": sum(t.nelement() * t.element_size() for t in all_tensors.values())},
        "weight_map": final_weight_map,
    }
    with open(dest / "model.safetensors.index.json", "w") as f:
        json.dump(new_index, f, indent=2)
        f.write("\n")

    print(f"\n  Dequantized {total_dequantized} FP8 weights to bf16")
    print(f"  Wrote {total_copied} weights across {total_files} files")

    # Free memory
    del all_tensors
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


def fix_tokenizer_config(source: Path, dest: Path):
    """Copy and fix tokenizer_config.json for transformers 4.57."""
    src_path = source / "tokenizer_config.json"
    if not src_path.exists():
        return

    with open(src_path) as f:
        cfg = json.load(f)

    # Fix class name
    cfg["tokenizer_class"] = "PreTrainedTokenizerFast"

    # Remove 5.x-only fields
    for field in ("extra_special_tokens", "model_specific_special_tokens",
                  "backend", "is_local", "processor_class"):
        cfg.pop(field, None)

    with open(dest / "tokenizer_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    print("Wrote fixed tokenizer_config.json")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Devstral (multimodal FP8) to text-only MistralForCausalLM (bf16)."
    )
    parser.add_argument("source", help="HuggingFace model ID or local path.")
    parser.add_argument("dest", help="Output directory for the converted model.")
    args = parser.parse_args()

    # Resolve source path
    source = Path(args.source)
    if not source.exists():
        source = resolve_hf_cache_path(args.source)
    source = source.resolve()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Source: {source}")
    print(f"Dest:   {dest}")
    print()

    # Read and remap config
    config_path = source / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    print("Remapping config...")
    remapped = remap_config(config)
    with open(dest / "config.json", "w") as f:
        json.dump(remapped, f, indent=2)
        f.write("\n")

    text_cfg = config.get("text_config", {})
    rope_params = text_cfg.get("rope_parameters", {})
    print(f"  model_type: {config.get('model_type')} -> {remapped['model_type']}")
    print(f"  architectures: {config['architectures']} -> {remapped['architectures']}")
    print(f"  rope_theta: {remapped['rope_theta']}")
    print(f"  rope_scaling: type={remapped['rope_scaling'].get('rope_type')}, factor={remapped['rope_scaling'].get('factor')}")
    if "_ministral3_llama4_beta" in remapped:
        print(f"  llama4_beta: {remapped['_ministral3_llama4_beta']}")
        print(f"  orig_max_pos: {remapped['_ministral3_orig_max_pos']}")
    print()

    # Convert weights
    print("Converting weights (FP8 -> bf16, stripping vision)...")
    convert_weights(source, dest)
    print()

    # Fix tokenizer
    print("Fixing tokenizer...")
    fix_tokenizer_config(source, dest)

    # Symlink tokenizer.json and other non-weight files
    symlink_files = ["tokenizer.json", "chat_template.jinja", "generation_config.json"]
    for fname in symlink_files:
        src_file = source / fname
        if src_file.exists():
            dest_file = dest / fname
            if dest_file.exists() or dest_file.is_symlink():
                dest_file.unlink()
            os.symlink(src_file.resolve(), dest_file)
            print(f"Symlinked {fname}")

    print()
    print(f"Done! Use '{dest}' as model_name_or_path for training.")


if __name__ == "__main__":
    main()
