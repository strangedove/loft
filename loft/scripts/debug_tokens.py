# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tokenization debug inspector.

Shows the final tokenized form with loss mask coloring so you can verify
turn boundaries and which tokens the model trains on.

Usage:
    loft prepare configs/my-training.yaml --debug
    python -m loft.scripts.debug_tokens --config configs/my-training.yaml
"""

import argparse
import logging
import os
import sys
from typing import Optional

import yaml
from transformers import AutoTokenizer

from loft.data_utils import (
    add_system_message_to_example,
    fix_example_turn_order,
    is_conversational,
    is_conversational_from_value,
    maybe_convert_to_chatml,
    tokenize_sft_example,
)
from loft.scripts.utils import DatasetConfig, DatasetMixtureConfig, resolve_config_inheritance

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Color helpers
# ──────────────────────────────────────────────────────────────────────────────

def _supports_color() -> bool:
    """Check if the terminal supports ANSI colors."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


USE_COLOR = _supports_color()

# ANSI color codes
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RED = "\033[31m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_BLUE = "\033[34m"
_MAGENTA = "\033[35m"
_CYAN = "\033[36m"
_BG_RED = "\033[41m"
_BG_GREEN = "\033[42m"
_BG_YELLOW = "\033[43m"
_BG_BLUE = "\033[44m"


def _c(text: str, *codes: str) -> str:
    """Apply ANSI color codes to text if color is supported."""
    if not USE_COLOR:
        return text
    return "".join(codes) + text + _RESET


def _label(tag: str, color_codes: tuple) -> str:
    """Format a tag label with color."""
    if USE_COLOR:
        return "".join(color_codes) + f"[{tag}]" + _RESET
    return f"[{tag}]"


# ──────────────────────────────────────────────────────────────────────────────
# Sample loading
# ──────────────────────────────────────────────────────────────────────────────

def _load_one_sample_per_dataset(data_config_path: str) -> tuple[list[tuple[str, dict, DatasetConfig]], DatasetMixtureConfig]:
    """
    Load one sample from each dataset in the data config.

    Uses DatasetMixtureConfig to properly expand per-file configs
    into individual datasets with correct data_files parameters.

    Returns:
        Tuple of (samples_list, mixture_config) where samples_list contains
        (dataset_label, sample_dict, dataset_config) tuples.
    """
    from datasets import load_dataset

    with open(data_config_path) as f:
        raw = yaml.safe_load(f)

    # Use DatasetMixtureConfig to properly expand files: entries into individual DatasetConfigs
    mixture_config = DatasetMixtureConfig(**raw)

    if not mixture_config.datasets:
        raise ValueError(f"No datasets found in {data_config_path}")

    samples = []
    for ds_config in mixture_config.datasets:
        # Build a descriptive label
        label = ds_config.path
        if ds_config.data_files:
            if isinstance(ds_config.data_files, str):
                label = f"{ds_config.path}/{ds_config.data_files}"
            elif isinstance(ds_config.data_files, list) and len(ds_config.data_files) == 1:
                label = f"{ds_config.path}/{ds_config.data_files[0]}"

        try:
            ds = load_dataset(
                path=ds_config.path,
                name=ds_config.name,
                data_dir=ds_config.data_dir,
                data_files=ds_config.data_files,
                split=ds_config.split,
            )
            sample = next(iter(ds))
            samples.append((label, sample, ds_config))
        except Exception as e:
            logger.warning(f"Failed to load sample from {label}: {e}")
            samples.append((label, {"_error": str(e)}, ds_config))

    return samples, mixture_config


# ──────────────────────────────────────────────────────────────────────────────
# Debug display
# ──────────────────────────────────────────────────────────────────────────────

def _truncate_text(text: str, max_chars: int = 500) -> str:
    """Truncate text for display, showing start and end."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + f"\n  ... ({len(text) - max_chars} chars omitted) ...\n" + text[-half:]


def _format_raw_sample(sample: dict) -> str:
    """Format a raw sample for display."""
    lines = []
    for key, value in sample.items():
        if key.startswith("_"):
            continue
        if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
            # Chat messages
            lines.append(f"  {_c(key, _BOLD)}:")
            for msg in value[:5]:  # Show first 5 messages
                role = msg.get("role") or msg.get("from", "?")
                content = msg.get("content") or msg.get("value", "")
                content_preview = _truncate_text(content, 200)
                lines.append(f"    [{_c(role, _CYAN)}]: {content_preview}")
            if len(value) > 5:
                lines.append(f"    ... ({len(value) - 5} more messages)")
        elif isinstance(value, str):
            lines.append(f"  {_c(key, _BOLD)}: {_truncate_text(value, 300)}")
        else:
            lines.append(f"  {_c(key, _BOLD)}: {repr(value)[:200]}")
    return "\n".join(lines)


def _build_loss_mask(tokenized: dict, num_tokens: int) -> list[int]:
    """
    Build a per-token loss mask from the tokenized output.

    Returns a list of 0/1 where 1 = token is trained on, 0 = masked.
    """
    # For prompt-completion: completion_mask marks completions
    if "completion_mask" in tokenized:
        mask = tokenized["completion_mask"][:num_tokens]
        # If assistant_masks is also present, intersect
        if "assistant_masks" in tokenized:
            asst = tokenized["assistant_masks"][:num_tokens]
            mask = [c & a for c, a in zip(mask, asst)]
        return mask

    # For chat with assistant_masks: train only on assistant tokens
    if "assistant_masks" in tokenized:
        return tokenized["assistant_masks"][:num_tokens]

    # For plain text: train on all tokens
    return [1] * num_tokens


def _format_token_view(
    input_ids: list[int],
    loss_mask: list[int],
    tokenizer,
    max_display_tokens: int = 500,
) -> str:
    """
    Format a token-level view with color coding.

    Shows each token classified by loss mask:
    - [TRAIN] tokens: green - model learns from these
    - [MASKED] tokens: dim - not trained on (prompt, system message)

    Special tokens are classified as TRAIN or MASKED based on the loss mask
    (not given their own category), so it's immediately clear whether they
    are trained on. They are visually marked with angle brackets in the
    decoded text.
    """
    lines = []
    num_tokens = len(input_ids)

    # If too many tokens, show first N and last N
    show_start = max_display_tokens // 2
    show_end = max_display_tokens // 2
    if num_tokens > max_display_tokens:
        display_ranges = list(range(show_start)) + [-1] + list(range(num_tokens - show_end, num_tokens))
    else:
        display_ranges = list(range(num_tokens))

    current_type = None
    current_text = []
    current_ids = []

    def flush():
        nonlocal current_type, current_text, current_ids
        if current_type is None:
            return
        decoded = "".join(current_text)
        id_str = ",".join(str(i) for i in current_ids[:8])
        if len(current_ids) > 8:
            id_str += f"...+{len(current_ids)-8}"

        if current_type == "TRAIN":
            tag = _label("TRAIN", (_GREEN, _BOLD))
            text_display = _c(repr(decoded), _GREEN)
        else:  # MASKED
            tag = _label("MASKED", (_DIM,))
            text_display = _c(repr(decoded), _DIM)

        lines.append(f"  {tag:>20s} {text_display}  {_c(f'({id_str})', _DIM)}")
        current_type = None
        current_text = []
        current_ids = []

    for idx in display_ranges:
        if idx == -1:
            flush()
            lines.append(f"  {'':>20s} {_c(f'... ({num_tokens - max_display_tokens} tokens omitted) ...', _DIM)}")
            continue

        token_id = input_ids[idx]
        is_trained = loss_mask[idx] if idx < len(loss_mask) else 0
        decoded_token = tokenizer.decode([token_id])

        if is_trained:
            token_type = "TRAIN"
        else:
            token_type = "MASKED"

        # Group consecutive tokens of the same type
        if token_type != current_type:
            flush()
            current_type = token_type
            current_text = [decoded_token]
            current_ids = [token_id]
        else:
            current_text.append(decoded_token)
            current_ids.append(token_id)

    flush()
    return "\n".join(lines)


def debug_sample(
    sample: dict,
    tokenizer,
    dataset_label: str,
    dataset_text_field: str = "text",
    default_system_message: Optional[str] = None,
    fix_turn_order: bool = False,
    fix_turn_order_filler: str = "Let's begin.",
    assistant_only_loss: bool = False,
    last_assistant_only_loss: bool = False,
    train_on_incomplete_assistant: bool = False,
    max_length: Optional[int] = None,
    truncation_strategy: str = "truncate",
    max_display_tokens: int = 500,
) -> str:
    """
    Produce a debug report for a single sample.

    Shows the final tokenized form with loss mask coloring so you can verify
    turn boundaries and which tokens the model trains on.
    """
    lines = []
    warnings = []
    sep = "─" * 72

    lines.append("")
    lines.append(_c(f"{'═' * 72}", _BOLD))
    lines.append(_c(f"  Dataset: {dataset_label}", _BOLD))
    lines.append(_c(f"{'═' * 72}", _BOLD))

    # Preprocess (same steps as preprocess_dataset, but silently)
    is_chat = is_conversational(sample) or is_conversational_from_value(sample)
    processed = dict(sample)

    if is_conversational_from_value(processed) and "conversations" in processed:
        processed = maybe_convert_to_chatml(processed)
        if "conversations" in processed:
            del processed["conversations"]

    if (default_system_message or "_system_message" in processed) and is_conversational(processed):
        processed = add_system_message_to_example(
            processed, system_message=default_system_message or ""
        )

    if fix_turn_order and is_conversational(processed):
        processed = fix_example_turn_order(processed, filler_message=fix_turn_order_filler)

    # Tokenize
    lines.append("")
    lines.append(_c("  Tokenized (loss mask view)", _BOLD, _CYAN))
    lines.append(f"  {sep}")

    # Add EOS for plain text (matching prepare pipeline behavior)
    tokenize_sample = dict(processed)
    if not is_conversational(processed):
        eos_token = tokenizer.eos_token
        text_key = dataset_text_field if dataset_text_field in tokenize_sample else "text"
        if text_key in tokenize_sample and eos_token and not tokenize_sample[text_key].endswith(eos_token):
            tokenize_sample[text_key] = tokenize_sample[text_key] + eos_token

    try:
        tokenized = tokenize_sft_example(
            tokenize_sample,
            tokenizer,
            dataset_text_field=dataset_text_field,
            assistant_only_loss=assistant_only_loss,
            last_assistant_only_loss=last_assistant_only_loss,
            train_on_incomplete_assistant=train_on_incomplete_assistant,
            eos_token_id=tokenizer.eos_token_id,
        )
    except Exception as e:
        lines.append(f"  {_c(f'ERROR during tokenization: {e}', _RED)}")
        warnings.append(f"Tokenization failed: {e}")
        tokenized = None
        import traceback; traceback.print_exc()

    if tokenized:
        input_ids = tokenized["input_ids"]
        num_tokens = len(input_ids)
        loss_mask = _build_loss_mask(tokenized, num_tokens)
        train_tokens = sum(loss_mask)
        train_pct = train_tokens / num_tokens * 100 if num_tokens > 0 else 0

        lines.append(f"  Total tokens: {num_tokens:,}")
        lines.append(f"  Training on: {train_tokens:,}/{num_tokens:,} tokens ({train_pct:.1f}%)")

        if max_length and num_tokens > max_length:
            if truncation_strategy == "split":
                import math
                n_chunks = math.ceil(num_tokens / max_length)
                lines.append(f"  ⚠ Would be split into {n_chunks} chunks of ≤{max_length:,} tokens")
                warnings.append(f"Sample would be split into {n_chunks} chunks by '{truncation_strategy}' strategy")
            elif truncation_strategy == "drop":
                lines.append(f"  ⚠ Would be DROPPED (exceeds max_length={max_length:,})")
                warnings.append(f"Sample would be dropped (exceeds max_length={max_length:,})")
            else:
                lines.append(f"  ⚠ Would be truncated from {num_tokens:,} to {max_length:,} tokens")
                warnings.append(f"Sample would be truncated from {num_tokens:,} to {max_length:,} tokens")

        # Check for EOS at end
        if tokenizer.eos_token_id is not None and input_ids[-1] != tokenizer.eos_token_id:
            warnings.append("No EOS token at end of sequence")

        # Check train percentage
        if train_pct < 5.0:
            warnings.append(f"Very low train percentage ({train_pct:.1f}%) — check masking settings")
        elif train_pct > 95.0 and is_conversational(processed):
            warnings.append(f"Very high train percentage ({train_pct:.1f}%) for chat data — system/user turns may not be masked")

        lines.append("")
        lines.append(_format_token_view(input_ids, loss_mask, tokenizer, max_display_tokens=max_display_tokens))
    # 5. Warnings
    if warnings:
        lines.append("")
        lines.append(_c("  ⚠ Warnings", _BOLD, _YELLOW))
        lines.append(f"  {sep}")
        for w in warnings:
            lines.append(f"  • {_c(w, _YELLOW)}")

    lines.append("")
    return "\n".join(lines)


def run_debug(training_config: dict, max_display_tokens: int = 500, config_path: Optional[str] = None) -> None:
    """
    Run the debug inspector on one sample from each dataset.

    Args:
        training_config: Raw dict from the training YAML config.
        max_display_tokens: Max tokens to show in the token-level view.
        config_path: Path to the training config file (for resolving relative paths).
    """
    data_config_path = training_config.get("data_config")
    if not data_config_path:
        raise ValueError("Training config must have a 'data_config' field.")

    # Resolve relative data_config path relative to training config location
    if config_path and not os.path.isabs(data_config_path):
        config_dir = os.path.dirname(os.path.abspath(config_path))
        data_config_path = os.path.join(config_dir, data_config_path)

    model_name_or_path = training_config.get("model_name_or_path")
    if not model_name_or_path:
        raise ValueError("Training config must have 'model_name_or_path'.")

    trust_remote_code = training_config.get("trust_remote_code", False)
    max_length = training_config.get("max_length")
    truncation_strategy = training_config.get("truncation_strategy", "truncate")
    dataset_text_field = training_config.get("dataset_text_field", "text")
    fix_turn_order = training_config.get("fix_turn_order", False)
    fix_turn_order_filler = training_config.get("fix_turn_order_filler", "Let's begin.")
    last_assistant_only_loss = training_config.get("last_assistant_only_loss", False)
    train_on_incomplete_assistant = training_config.get("train_on_incomplete_assistant", False)
    chat_template_path = training_config.get("chat_template_path")

    model_short = os.path.basename(model_name_or_path.rstrip("/")) if "/" in model_name_or_path else model_name_or_path

    print(f"\n🔬 Tokenization Debug Inspector")
    print(f"   Model: {model_short}")
    print(f"   Data config: {data_config_path}")
    if max_length:
        print(f"   Max length: {max_length:,}, strategy: {truncation_strategy}")
    print()

    # Load tokenizer
    print(f"   Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)

    if chat_template_path:
        if os.path.isfile(chat_template_path):
            with open(chat_template_path) as f:
                tokenizer.chat_template = f.read()
            print(f"   Loaded chat template from: {chat_template_path}")
        else:
            template_tokenizer = AutoTokenizer.from_pretrained(chat_template_path, trust_remote_code=trust_remote_code)
            tokenizer.chat_template = template_tokenizer.chat_template
            print(f"   Loaded chat template from model: {chat_template_path}")

    # Load one sample per dataset
    print(f"   Loading samples...")
    samples, mixture_config = _load_one_sample_per_dataset(data_config_path)

    # Get default_system_message and assistant_only_loss from data config (mixture_config)
    # with fallback to training config
    default_system_message = mixture_config.default_system_message or training_config.get("default_system_message")
    assistant_only_loss = mixture_config.assistant_only_loss or training_config.get("assistant_only_loss", False)

    if default_system_message:
        print(f"   Default system message: {default_system_message[:50]}...")
    if assistant_only_loss:
        print(f"   Assistant-only loss: enabled")

    if not samples:
        print("   No samples found!")
        return

    print(f"   Found {len(samples)} dataset(s)")

    # Debug each sample
    for label, sample, ds_config in samples:
        if "_error" in sample:
            print(f"\n   ⚠ Skipping {label}: {sample['_error']}")
            continue

        # Use per-dataset settings if available, falling back to global config
        sample_max_length = ds_config.max_length if ds_config.max_length is not None else max_length
        sample_trunc_strategy = ds_config.truncation_strategy if ds_config.truncation_strategy is not None else truncation_strategy
        sample_system_message = ds_config.system_message if ds_config.system_message is not None else default_system_message
        sample_train_incomplete = ds_config.train_on_incomplete_assistant if ds_config.train_on_incomplete_assistant else train_on_incomplete_assistant

        report = debug_sample(
            sample=sample,
            tokenizer=tokenizer,
            dataset_label=label,
            dataset_text_field=dataset_text_field,
            default_system_message=sample_system_message,
            fix_turn_order=fix_turn_order,
            fix_turn_order_filler=fix_turn_order_filler,
            assistant_only_loss=assistant_only_loss,
            last_assistant_only_loss=last_assistant_only_loss,
            train_on_incomplete_assistant=sample_train_incomplete,
            max_length=sample_max_length,
            truncation_strategy=sample_trunc_strategy,
            max_display_tokens=max_display_tokens,
        )
        print(report)

    print(f"\n{'═' * 72}")
    print(f"  Debug inspection complete.")
    print(f"{'═' * 72}\n")


def main():
    """CLI entry point for standalone debug usage."""
    parser = argparse.ArgumentParser(
        description="Tokenization debug inspector — show how samples get processed end-to-end",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=500,
        help="Max tokens to display in token-level view (default: 500)",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Load and resolve config
    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)
    config = resolve_config_inheritance(config, args.config)

    run_debug(config, max_display_tokens=args.max_tokens, config_path=args.config)


if __name__ == "__main__":
    main()
