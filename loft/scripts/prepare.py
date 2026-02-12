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
Dataset preparation script.

Prepares datasets with full tokenization, truncation/splitting, and eval splitting.
Takes a training config (which references a data config) and produces a pre-tokenized
dataset ready for training.

Usage:
    loft prepare configs/my-training.yaml
    loft prepare configs/my-training.yaml --output data/prepared
"""

import argparse
from collections import Counter
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import yaml
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

from loft.data_utils import (
    add_system_message_to_example,
    apply_truncation_to_dataset,
    convert_binary_preference_to_sft,
    convert_preference_to_sft,
    fix_example_turn_order,
    is_binary_preference_dataset,
    is_conversational,
    is_conversational_from_value,
    is_preference_dataset,
    maybe_convert_to_chatml,
    tokenize_sft_example,
    truncate_conversation_by_turns,
)
from loft.scripts.utils import DatasetMixtureConfig, get_dataset, resolve_config_inheritance


logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline statistics tracker
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineStats:
    """Tracks sample counts through each stage of the prepare pipeline."""

    steps: list = field(default_factory=list)

    def record(self, stage: str, count: int, detail: str = ""):
        self.steps.append({"stage": stage, "count": count, "detail": detail})

    def format_summary(self) -> str:
        """Format a waterfall summary of the pipeline."""
        if not self.steps:
            return ""

        lines = ["", "📊 Pipeline summary:"]
        max_stage_len = max(len(s["stage"]) for s in self.steps)
        prev_count = None

        for step in self.steps:
            stage = step["stage"].ljust(max_stage_len)
            count = step["count"]
            detail = step["detail"]

            if prev_count is not None and count < prev_count:
                delta = f"  (-{prev_count - count} {detail})" if detail else f"  (-{prev_count - count})"
            elif prev_count is not None and count > prev_count:
                delta = f"  (+{count - prev_count} {detail})" if detail else f"  (+{count - prev_count})"
            else:
                delta = f"  ({detail})" if detail else ""

            lines.append(f"  {stage}  {count:>8,}{delta}")
            prev_count = count

        lines.append("")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Config loading
# ──────────────────────────────────────────────────────────────────────────────

def _load_data_config(data_config_path: str) -> DatasetMixtureConfig:
    """Load a data config YAML and return a DatasetMixtureConfig."""
    if not os.path.isfile(data_config_path):
        raise FileNotFoundError(
            f"Data config file not found: {data_config_path}\n"
            f"Check that the 'data_config' path in your training config is correct."
        )
    try:
        with open(data_config_path) as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in data config {data_config_path}: {e}") from e

    if not isinstance(raw, dict):
        raise ValueError(
            f"Data config {data_config_path} must be a YAML mapping, got {type(raw).__name__}."
        )
    if "datasets" not in raw:
        raise ValueError(
            f"Data config {data_config_path} must have a 'datasets' list. "
            f"Example:\n  datasets:\n    - path: /path/to/dataset\n      split: train"
        )

    # Force eval_split to 0 — we do eval splitting AFTER tokenization
    raw["eval_split"] = 0.0

    return DatasetMixtureConfig(**raw)


def _load_training_config(config_path: str) -> dict:
    """Load a training config YAML and return the raw dict."""
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Training config file not found: {config_path}")
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in training config {config_path}: {e}") from e

    if not isinstance(config, dict):
        raise ValueError(f"Training config {config_path} must be a YAML mapping, got {type(config).__name__}.")

    # Resolve config inheritance (base_config field)
    config = resolve_config_inheritance(config, config_path)

    return config


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_dataset(
    dataset: Dataset,
    stats: PipelineStats,
    trainer_type: str = "sft",
    default_system_message: Optional[str] = None,
    fix_turn_order: bool = False,
    fix_turn_order_filler: str = "Let's begin.",
    num_proc: Optional[int] = None,
) -> Dataset:
    """
    Apply model-agnostic preprocessing to a dataset.

    Handles format conversion, system messages, turn order fixing.
    """
    map_kwargs = {}
    if num_proc is not None:
        map_kwargs["num_proc"] = num_proc

    if trainer_type != "sft":
        logger.warning(f"trainer_type={trainer_type} is not fully supported in prepare yet. Proceeding with SFT logic.")

    if len(dataset) == 0:
        raise ValueError("Dataset is empty after loading. Check your data config paths and split names.")

    # Auto-convert preference datasets to SFT format
    first_example = next(iter(dataset))
    if is_preference_dataset(first_example):
        logger.info("Detected preference dataset format — converting to SFT format.")
        column_names = dataset.column_names
        remove_cols = [c for c in ["prompt", "chosen", "rejected"] if c in column_names]
        try:
            dataset = dataset.map(convert_preference_to_sft, remove_columns=remove_cols, **map_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to convert preference dataset to SFT format: {e}\n"
                f"Expected columns: 'prompt', 'chosen', 'rejected' (each with 'role'/'content' dicts)."
            ) from e
        stats.record("Preference → SFT", len(dataset))

    elif is_binary_preference_dataset(first_example):
        logger.info("Detected binary preference dataset — converting to SFT format.")
        column_names = dataset.column_names
        remove_cols = [c for c in ["prompt", "completion", "label"] if c in column_names]
        before = len(dataset)

        def convert_and_filter(example):
            result = convert_binary_preference_to_sft(example)
            return result if result is not None else {}

        try:
            dataset = dataset.map(convert_and_filter, remove_columns=remove_cols, **map_kwargs)
            dataset = dataset.filter(lambda x: "messages" in x and len(x["messages"]) > 0, **map_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to convert binary preference dataset to SFT format: {e}\n"
                f"Expected columns: 'prompt', 'completion', 'label'."
            ) from e
        stats.record("Binary pref → SFT", len(dataset), f"filtered {before - len(dataset)} bad examples")

    # Convert legacy conversation format to ChatML
    first_example = next(iter(dataset))
    if is_conversational_from_value(first_example):
        column_names = dataset.column_names
        try:
            dataset = dataset.map(
                maybe_convert_to_chatml,
                remove_columns="conversations" if "conversations" in column_names else None,
                desc="Converting to ChatML",
                **map_kwargs,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to convert legacy conversation format to ChatML: {e}\n"
                f"Expected 'conversations' column with list of {{'from': ..., 'value': ...}} dicts."
            ) from e
        stats.record("Convert to ChatML", len(dataset))

    # Add system messages
    column_names = dataset.column_names
    has_per_dataset_system_msg = "_system_message" in column_names
    if (default_system_message or has_per_dataset_system_msg) and is_conversational(next(iter(dataset))):
        remove_cols = "_system_message" if has_per_dataset_system_msg else None
        dataset = dataset.map(
            add_system_message_to_example,
            fn_kwargs={"system_message": default_system_message or ""},
            remove_columns=remove_cols,
            desc="Adding system messages",
            **map_kwargs,
        )
        stats.record("Add system messages", len(dataset))

    # Fix turn order
    if fix_turn_order and is_conversational(next(iter(dataset))):
        before = len(dataset)
        dataset = dataset.map(
            fix_example_turn_order,
            fn_kwargs={"filler_message": fix_turn_order_filler},
            desc="Fixing turn order",
            **map_kwargs,
        )
        dataset = dataset.filter(
            lambda x: any(
                isinstance(x.get(k), list) and len(x.get(k, [])) > 0
                for k in ["messages", "prompt", "completion"]
            ),
            **map_kwargs,
        )
        dropped = before - len(dataset)
        if dropped:
            logger.warning(f"fix_turn_order: Dropped {dropped} invalid examples.")
        stats.record("Fix turn order", len(dataset), f"dropped {dropped} invalid" if dropped else "")

    return dataset


# ──────────────────────────────────────────────────────────────────────────────
# Tokenization
# ──────────────────────────────────────────────────────────────────────────────

def tokenize_dataset(
    dataset: Dataset,
    processing_class,
    stats: PipelineStats,
    dataset_text_field: str = "text",
    truncation_strategy: str = "truncate",
    max_length: Optional[int] = None,
    assistant_only_loss: bool = False,
    last_assistant_only_loss: bool = False,
    train_on_incomplete_assistant: bool = False,
    num_proc: Optional[int] = None,
) -> Dataset:
    """
    Tokenize a preprocessed dataset and apply truncation.

    Handles:
    - truncate_turns (pre-tokenization, on messages)
    - EOS addition for plain text
    - Tokenization via chat template or plain tokenizer
    - Truncation strategy (split/drop/truncate)
    """
    map_kwargs = {}
    if num_proc is not None:
        map_kwargs["num_proc"] = num_proc

    # Handle truncate_turns before tokenization (needs message-level structure)
    column_names = dataset.column_names
    has_per_dataset_strategy = "_truncation_strategy" in column_names
    has_per_dataset_max_length = "_max_length" in column_names
    has_dataset_name = "_dataset_name" in column_names

    # Print per-dataset breakdown if we have the info
    if has_per_dataset_strategy and has_dataset_name:
        # Count samples per dataset with truncate_turns strategy
        truncate_turns_datasets = []
        for ex in dataset:
            if ex.get("_truncation_strategy") == "truncate_turns":
                truncate_turns_datasets.append(ex.get("_dataset_name", "unknown"))
        if truncate_turns_datasets:
            counts = Counter(truncate_turns_datasets)
            print("   Datasets using truncate_turns strategy:")
            for ds_name, count in counts.most_common():
                short_name = ds_name.split("/")[-1] if "/" in ds_name else ds_name
                print(f"     • {short_name}: {count:,} samples")

    if (truncation_strategy == "truncate_turns" or has_per_dataset_strategy) and max_length is not None:
        first_example = next(iter(dataset))
        if is_conversational(first_example):
            def truncate_turns_fn(example, tokenizer, default_max_length, default_strategy):
                # Get strategy without popping - we need to preserve it for non-truncate_turns samples
                strategy = example.get("_truncation_strategy") or default_strategy
                # Always set _truncation_drop to ensure consistent schema across all workers
                example["_truncation_drop"] = False
                if strategy != "truncate_turns":
                    return example
                # For truncate_turns samples, mark as handled by setting strategy to "truncate"
                # (truncate_turns becomes truncate after pre-tokenization handling)
                example["_truncation_strategy"] = "truncate"
                # Use per-example max_length if available, otherwise use default
                effective_max_length = example.get("_max_length") or default_max_length
                truncated = truncate_conversation_by_turns(
                    example.get("messages", []), tokenizer, effective_max_length
                )
                if truncated is None:
                    example["_truncation_drop"] = True
                else:
                    example["messages"] = truncated
                return example

            # Don't remove _truncation_strategy - we need it for post-tokenization truncation
            before = len(dataset)
            dataset = dataset.map(
                truncate_turns_fn,
                fn_kwargs={
                    "tokenizer": processing_class,
                    "default_max_length": max_length,
                    "default_strategy": truncation_strategy,
                },
                desc="Truncating by turns",
                **map_kwargs,
            )
            dataset = dataset.filter(lambda x: not x.get("_truncation_drop", False), **map_kwargs)
            dropped = before - len(dataset)
            if dropped:
                logger.info(
                    f"truncate_turns: Dropped {dropped} samples that couldn't fit "
                    f"even one turn pair in max_length."
                )
            column_names = dataset.column_names
            if "_truncation_drop" in column_names:
                dataset = dataset.remove_columns(["_truncation_drop"])
            stats.record("Truncate turns", len(dataset), f"dropped {dropped}" if dropped else "")

    # Add EOS for plain text datasets
    first_example = next(iter(dataset))
    if not is_conversational(first_example):
        eos_token = processing_class.eos_token

        def add_eos(example, _eos_token):
            text_val = example.get("text")
            completion_val = example.get("completion")
            if text_val is not None and isinstance(text_val, str) and not text_val.endswith(_eos_token):
                example["text"] = text_val + _eos_token
            elif completion_val is not None and isinstance(completion_val, str) and not completion_val.endswith(_eos_token):
                example["completion"] = completion_val + _eos_token
            return example

        dataset = dataset.map(
            add_eos,
            fn_kwargs={"_eos_token": eos_token},
            desc="Adding EOS",
            **map_kwargs,
        )

    # Detect dataset format for helpful error messages
    first_example = next(iter(dataset))
    is_chat = is_conversational(first_example)
    has_text_field = dataset_text_field in first_example
    if not is_chat and not has_text_field:
        available = list(first_example.keys())
        raise ValueError(
            f"Dataset has neither chat format ('messages' column) nor the text field '{dataset_text_field}'.\n"
            f"Available columns: {available}\n"
            f"Set 'dataset_text_field' in your config to match your data, or convert to ChatML format."
        )

    # Tokenize
    try:
        dataset = dataset.map(
            tokenize_sft_example,
            fn_kwargs={
                "processing_class": processing_class,
                "dataset_text_field": dataset_text_field,
                "assistant_only_loss": assistant_only_loss,
                "last_assistant_only_loss": last_assistant_only_loss,
                "train_on_incomplete_assistant": train_on_incomplete_assistant,
                "eos_token_id": processing_class.eos_token_id,
            },
            desc="Tokenizing",
            **map_kwargs,
        )
    except Exception as e:
        if is_chat and "chat_template" in str(e).lower():
            raise RuntimeError(
                f"Tokenization failed — likely a chat template issue: {e}\n"
                f"The model's tokenizer may not have a chat template, or it's incompatible with your data.\n"
                f"Try setting 'chat_template_path' in your config to a compatible template."
            ) from e
        elif is_chat:
            raise RuntimeError(
                f"Tokenization failed on chat-format data: {e}\n"
                f"Check that your 'messages' column has the expected format: "
                f"[{{'role': 'user', 'content': '...'}}, {{'role': 'assistant', 'content': '...'}}]"
            ) from e
        else:
            raise RuntimeError(
                f"Tokenization failed on text-format data: {e}\n"
                f"Check that the '{dataset_text_field}' column contains text strings."
            ) from e

    stats.record("Tokenize", len(dataset))

    # Apply truncation strategy
    if max_length is not None:
        effective_strategy = truncation_strategy
        if effective_strategy == "truncate_turns":
            effective_strategy = "truncate"  # already handled above

        # Check if we have per-sample strategies
        has_per_sample_strategy = "_truncation_strategy" in dataset.column_names
        if has_per_sample_strategy:
            # Show per-strategy breakdown before applying
            strategy_counts = Counter(dataset["_truncation_strategy"])
            print("   Per-dataset truncation strategies:")
            for strat, count in strategy_counts.most_common():
                strat_name = strat if strat else f"default ({effective_strategy})"
                print(f"     • {strat_name}: {count:,} samples")

        before = len(dataset)
        try:
            dataset = apply_truncation_to_dataset(
                dataset, processing_class, max_length, strategy=effective_strategy, num_proc=num_proc
            )
        except Exception as e:
            raise RuntimeError(
                f"Truncation failed (strategy='{effective_strategy}', max_length={max_length}): {e}"
            ) from e

        detail = ""
        if has_per_sample_strategy:
            detail = f"mixed strategies, {before:,} → {len(dataset):,}"
            print(f"   Truncation complete: {before:,} → {len(dataset):,} samples ({len(dataset) - before:+,} delta)")
        elif effective_strategy == "split":
            detail = f"split into {max_length}-token chunks"
            print(f"   Split strategy: {before:,} → {len(dataset):,} samples ({len(dataset) - before:+,} from chunking)")
        elif effective_strategy == "drop" and len(dataset) < before:
            detail = f"dropped {before - len(dataset)} over-length samples"
            print(f"   Drop strategy: {before:,} → {len(dataset):,} samples (dropped {before - len(dataset):,})")
        elif effective_strategy == "truncate":
            detail = f"truncated to {max_length} tokens"
            print(f"   Truncate strategy: all samples capped at {max_length} tokens")

        stats.record("Truncation ({})".format(effective_strategy if not has_per_sample_strategy else "mixed"), len(dataset), detail)

    return dataset


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def prepare_dataset(
    training_config: dict,
    output_dir: Optional[str] = None,
) -> tuple[DatasetDict, PipelineStats]:
    """
    Full pipeline: load data → preprocess → tokenize → truncate → eval split.

    Args:
        training_config: Raw dict from the training YAML config.
        output_dir: Override for output directory.

    Returns:
        Tuple of (DatasetDict with "train" and optionally "test" splits, PipelineStats).

    Config value precedence (highest to lowest):
      1. training_config (main training config, after base_config inheritance)
      2. data_config file
      3. hardcoded defaults

    This allows users to set defaults in data_config (tied to the dataset) while
    overriding specific values in the main training config for experiments.
    """
    stats = PipelineStats()

    # Extract settings from training config
    data_config_path = training_config.get("data_config")
    if not data_config_path:
        raise ValueError(
            "Training config must have a 'data_config' field pointing to the data config YAML. "
            "Example: data_config: data/marvin.yaml"
        )

    model_name_or_path = training_config.get("model_name_or_path")
    if not model_name_or_path:
        raise ValueError("Training config must have 'model_name_or_path'.")

    # Load data_config for fallback values
    data_config_values = {}
    if os.path.exists(data_config_path):
        with open(data_config_path) as f:
            data_config_values = yaml.safe_load(f) or {}

    # Helper to get value with precedence: training_config > data_config > default
    def get_with_precedence(key, default=None):
        if key in training_config and training_config[key] is not None:
            return training_config[key]
        if key in data_config_values and data_config_values[key] is not None:
            return data_config_values[key]
        return default

    trust_remote_code = training_config.get("trust_remote_code", False)
    max_length = training_config.get("max_length", 1024)
    truncation_strategy = training_config.get("truncation_strategy", "truncate")
    dataset_text_field = training_config.get("dataset_text_field", "text")
    eval_split = get_with_precedence("eval_split", 0.0)
    split_seed = get_with_precedence("split_seed", 42)

    # Preprocessing options (support data_config fallback)
    default_system_message = get_with_precedence("default_system_message")
    fix_turn_order = get_with_precedence("fix_turn_order", False)
    fix_turn_order_filler = get_with_precedence("fix_turn_order_filler", "Let's begin.")
    assistant_only_loss = get_with_precedence("assistant_only_loss", False)
    last_assistant_only_loss = get_with_precedence("last_assistant_only_loss", False)
    train_on_incomplete_assistant = get_with_precedence("train_on_incomplete_assistant", False)
    num_proc = get_with_precedence("dataset_num_proc")
    chat_template_path = get_with_precedence("chat_template_path")

    print("\n🍹 Preparing datasets...")
    print(f"   Config: max_length={max_length}, strategy={truncation_strategy}, "
          f"eval_split={eval_split}, text_field={dataset_text_field}")

    # Step 1: Load data config
    print(f"\n📦 Loading data config from {data_config_path}...")
    data_config = _load_data_config(data_config_path)
    print(f"   Found {len(data_config.datasets)} dataset(s)")

    # Load datasets (NO eval split — we do that after tokenization)
    try:
        dataset_dict = get_dataset(data_config)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load datasets from {data_config_path}: {e}\n"
            f"Check that dataset paths exist and are readable."
        ) from e

    dataset = dataset_dict["train"]
    stats.record("Loaded", len(dataset))
    print(f"   {len(dataset):,} samples loaded")

    # Print per-dataset breakdown if we have dataset names
    if "_dataset_name" in dataset.column_names:
        dataset_counts = Counter(dataset["_dataset_name"])
        print("   Per-dataset breakdown:")
        for ds_name, count in dataset_counts.most_common():
            # Shorten the name for display
            short_name = ds_name.split("/")[-1] if "/" in ds_name else ds_name
            print(f"     • {short_name}: {count:,} samples")

    # Step 2: Load tokenizer
    model_short = os.path.basename(model_name_or_path.rstrip("/")) if "/" in model_name_or_path else model_name_or_path
    print(f"\n🤖 Loading tokenizer: {model_short}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load tokenizer from '{model_name_or_path}': {e}\n"
            f"Check that model_name_or_path is correct and the model is downloaded."
        ) from e

    # Apply chat template if specified
    if chat_template_path:
        try:
            if os.path.isfile(chat_template_path):
                with open(chat_template_path) as f:
                    tokenizer.chat_template = f.read()
                print(f"   Loaded chat template from file: {chat_template_path}")
            else:
                from transformers import AutoTokenizer as _AT
                template_tokenizer = _AT.from_pretrained(chat_template_path, trust_remote_code=trust_remote_code)
                tokenizer.chat_template = template_tokenizer.chat_template
                print(f"   Loaded chat template from model: {chat_template_path}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load chat template from '{chat_template_path}': {e}\n"
                f"Provide either a path to a jinja2 template file, or a HuggingFace model ID."
            ) from e

    # Step 3: Preprocess
    print("\n🔄 Preprocessing...")
    dataset = preprocess_dataset(
        dataset,
        stats,
        trainer_type="sft",
        default_system_message=default_system_message,
        fix_turn_order=fix_turn_order,
        fix_turn_order_filler=fix_turn_order_filler,
        num_proc=num_proc,
    )
    stats.record("After preprocessing", len(dataset))

    # Step 4: Tokenize + truncate
    print(f"\n🔤 Tokenizing with {model_short}...")
    print(f"✂️  Truncation strategy: {truncation_strategy} (max_length={max_length})")
    dataset = tokenize_dataset(
        dataset,
        tokenizer,
        stats,
        dataset_text_field=dataset_text_field,
        truncation_strategy=truncation_strategy,
        max_length=max_length,
        assistant_only_loss=assistant_only_loss,
        last_assistant_only_loss=last_assistant_only_loss,
        train_on_incomplete_assistant=train_on_incomplete_assistant,
        num_proc=num_proc,
    )

    # Step 5: Eval split (AFTER tokenization + chunking)
    if eval_split and eval_split > 0:
        # Check for _no_eval column - samples marked with this should ONLY go to train
        if "_no_eval" in dataset.column_names:
            # Separate samples that should never be in eval
            no_eval_mask = dataset["_no_eval"]
            no_eval_indices = [i for i, v in enumerate(no_eval_mask) if v]
            eval_eligible_indices = [i for i, v in enumerate(no_eval_mask) if not v]

            no_eval_samples = dataset.select(no_eval_indices) if no_eval_indices else None
            eval_eligible = dataset.select(eval_eligible_indices) if eval_eligible_indices else dataset

            if no_eval_samples:
                print(f"\n📊 Eval split: {len(no_eval_samples):,} samples excluded from eval (eval_split: false)")

            # Split only the eval-eligible samples
            if len(eval_eligible) > 0:
                split_result = eval_eligible.train_test_split(test_size=eval_split, seed=split_seed)
                train_split = split_result["train"]
                test_split = split_result["test"]
            else:
                train_split = eval_eligible
                test_split = None

            # Add back the no_eval samples to training set
            if no_eval_samples and len(no_eval_samples) > 0:
                from datasets import concatenate_datasets
                train_split = concatenate_datasets([train_split, no_eval_samples])

            if test_split and len(test_split) > 0:
                result = DatasetDict({"train": train_split, "test": test_split})
            else:
                result = DatasetDict({"train": train_split})
        else:
            split_result = dataset.train_test_split(test_size=eval_split, seed=split_seed)
            train_split = split_result["train"]
            test_split = split_result["test"]
            result = DatasetDict({"train": train_split, "test": test_split})

        stats.record("Train split", len(result["train"]))
        if "test" in result:
            stats.record("Eval split", len(result["test"]), f"{eval_split:.1%} of eligible")
            print(f"   {eval_split:.1%} of eligible → {len(result['train']):,} train, "
                  f"{len(result['test']):,} eval")

            # Show per-dataset breakdown of eval split
            if "_dataset_name" in result["test"].column_names:
                eval_counts = Counter(result["test"]["_dataset_name"])
                print("   Eval samples by dataset:")
                for ds_name, count in eval_counts.most_common():
                    short_name = ds_name.split("/")[-1] if "/" in ds_name else ds_name
                    print(f"     • {short_name}: {count:,}")
        else:
            stats.record("Eval split", 0, "no eligible samples")
            print(f"   No samples eligible for eval split")
    else:
        result = DatasetDict({"train": dataset})
        stats.record("Final (no eval)", len(dataset))
        print("\n📊 No eval split requested.")

    # Print waterfall summary
    print(stats.format_summary())

    return result, stats


def _get_cfg(training_config, data_config_values, key, default=None):
    """Get config value with precedence: training_config > data_config > default."""
    if key in training_config and training_config[key] is not None:
        return training_config[key]
    if key in data_config_values and data_config_values[key] is not None:
        return data_config_values[key]
    return default


def save_prepared_dataset(
    dataset_dict: DatasetDict,
    output_dir: str,
    training_config: dict,
    stats: Optional[PipelineStats] = None,
) -> None:
    """Save the prepared dataset to disk with metadata."""
    os.makedirs(output_dir, exist_ok=True)

    # Load data config for accurate metadata (training_config may not have all keys)
    data_config_values = {}
    data_config_path = training_config.get("data_config")
    if data_config_path and os.path.exists(data_config_path):
        with open(data_config_path) as f:
            data_config_values = yaml.safe_load(f) or {}

    print(f"💾 Saving to {output_dir}...")

    # Save each split as parquet
    for split_name, dataset in dataset_dict.items():
        split_path = os.path.join(output_dir, f"{split_name}.parquet")
        dataset.to_parquet(split_path)
        size_mb = os.path.getsize(split_path) / (1024 * 1024)
        print(f"   {split_name}: {len(dataset):,} samples ({size_mb:.1f} MB)")

    # Save metadata
    metadata = {
        "created_at": datetime.now().isoformat(),
        "tokenized": True,
        "model_name_or_path": training_config.get("model_name_or_path"),
        "max_length": training_config.get("max_length"),
        "truncation_strategy": training_config.get("truncation_strategy", "truncate"),
        "eval_split": training_config.get("eval_split", 0.0),
        "split_seed": training_config.get("split_seed", 42),
        "data_config": training_config.get("data_config"),
        "dataset_text_field": training_config.get("dataset_text_field", "text"),
        "preprocessing": {
            "assistant_only_loss": _get_cfg(training_config, data_config_values, "assistant_only_loss", False),
            "last_assistant_only_loss": _get_cfg(training_config, data_config_values, "last_assistant_only_loss", False),
            "train_on_incomplete_assistant": _get_cfg(training_config, data_config_values, "train_on_incomplete_assistant", False),
            "fix_turn_order": _get_cfg(training_config, data_config_values, "fix_turn_order", False),
            "default_system_message": _get_cfg(training_config, data_config_values, "default_system_message", None),
        },
        "splits": {
            split_name: len(dataset) for split_name, dataset in dataset_dict.items()
        },
    }

    # Include pipeline stats in metadata if available
    if stats is not None:
        metadata["pipeline_steps"] = stats.steps

    # Config hash for cache invalidation
    config_str = json.dumps(training_config, sort_keys=True, default=str)
    metadata["config_hash"] = hashlib.sha256(config_str.encode()).hexdigest()[:16]

    metadata_path = os.path.join(output_dir, "blend_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Compute token statistics
    def count_tokens(dataset):
        """Count total tokens and trainable tokens in a dataset."""
        import pyarrow.compute as pc

        table = dataset.data

        # Arrow-native: sum of list lengths avoids pulling entire columns to Python
        lengths = pc.list_value_length(table.column("input_ids"))
        total_tokens = pc.sum(lengths).as_py()

        has_assistant_masks = "assistant_masks" in dataset.column_names
        if has_assistant_masks:
            # Flatten all mask lists and sum (1 = trainable token)
            flat_masks = pc.list_flatten(table.column("assistant_masks"))
            trainable_tokens = pc.sum(flat_masks).as_py()
        else:
            trainable_tokens = total_tokens

        return total_tokens, trainable_tokens

    train_tokens, train_trainable = count_tokens(dataset_dict["train"])
    if "test" in dataset_dict:
        eval_tokens, eval_trainable = count_tokens(dataset_dict["test"])
    else:
        eval_tokens, eval_trainable = 0, 0

    # Add token stats to metadata
    metadata["token_stats"] = {
        "train_total_tokens": train_tokens,
        "train_trainable_tokens": train_trainable,
        "eval_total_tokens": eval_tokens,
        "eval_trainable_tokens": eval_trainable,
    }

    # Re-save metadata with token stats
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Final summary
    train_count = metadata["splits"].get("train", 0)
    eval_count = metadata["splits"].get("test", 0)
    total_count = train_count + eval_count
    print(f"\n✅ Dataset ready! {total_count:,} total ({train_count:,} train, {eval_count:,} eval)")
    print(f"   Saved to {output_dir}")

    # Token statistics
    total_tokens = train_tokens + eval_tokens
    total_trainable = train_trainable + eval_trainable
    print(f"\n📊 Token statistics:")
    print(f"   Train: {train_tokens:,} tokens ({train_trainable:,} trainable)")
    if eval_tokens > 0:
        print(f"   Eval:  {eval_tokens:,} tokens ({eval_trainable:,} trainable)")
    print(f"   Total: {total_tokens:,} tokens ({total_trainable:,} trainable)")


def dry_run(training_config: dict) -> None:
    """
    Show what prepare WOULD produce without actually tokenizing or saving.

    Loads datasets and tokenizes a small sample to estimate:
    - Dataset sizes
    - Estimated chunk counts (for split strategy)
    - Eval split sizes
    - Estimated output disk usage

    Config value precedence: training_config > data_config > defaults
    """
    import math

    data_config_path = training_config.get("data_config")
    if not data_config_path:
        raise ValueError("Training config must have a 'data_config' field.")

    model_name_or_path = training_config.get("model_name_or_path")
    if not model_name_or_path:
        raise ValueError("Training config must have 'model_name_or_path'.")

    # Load data_config for fallback values
    data_config_values = {}
    if os.path.exists(data_config_path):
        with open(data_config_path) as f:
            data_config_values = yaml.safe_load(f) or {}

    # Helper to get value with precedence: training_config > data_config > default
    def get_with_precedence(key, default=None):
        if key in training_config and training_config[key] is not None:
            return training_config[key]
        if key in data_config_values and data_config_values[key] is not None:
            return data_config_values[key]
        return default

    trust_remote_code = training_config.get("trust_remote_code", False)
    max_length = training_config.get("max_length", 1024)
    truncation_strategy = training_config.get("truncation_strategy", "truncate")
    dataset_text_field = training_config.get("dataset_text_field", "text")
    eval_split = get_with_precedence("eval_split", 0.0)
    model_short = os.path.basename(model_name_or_path.rstrip("/")) if "/" in model_name_or_path else model_name_or_path

    print("\n🔍 Dry run — showing what prepare would produce\n")
    print("  Settings:")
    print(f"    Model:               {model_short}")
    print(f"    Max length:          {max_length:,} tokens")
    print(f"    Truncation strategy: {truncation_strategy}")
    print(f"    Text field:          {dataset_text_field}")
    print(f"    Eval split:          {eval_split:.1%}" if eval_split else "    Eval split:          none")

    # Load data config and datasets
    print(f"\n  Loading data config from {data_config_path}...")
    data_config = _load_data_config(data_config_path)
    print(f"    Found {len(data_config.datasets)} dataset(s):")
    for i, ds_cfg in enumerate(data_config.datasets):
        label = ds_cfg.path
        if hasattr(ds_cfg, 'subset') and ds_cfg.subset:
            label += f" (subset: {ds_cfg.subset})"
        print(f"      [{i+1}] {label}")

    try:
        dataset_dict = get_dataset(data_config)
    except Exception as e:
        raise RuntimeError(f"Failed to load datasets: {e}") from e

    dataset = dataset_dict["train"]
    total_samples = len(dataset)
    print(f"\n    Total samples loaded: {total_samples:,}")

    # Load tokenizer and sample to estimate token lengths
    print(f"\n  Loading tokenizer: {model_short}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer: {e}") from e

    # Tokenize a sample (up to 100 examples) to estimate token length distribution
    sample_size = min(100, total_samples)
    sample_ds = dataset.select(range(sample_size))

    token_lengths = []
    first = next(iter(sample_ds))
    is_chat = is_conversational(first)

    for example in sample_ds:
        if is_chat:
            messages = example.get("messages", [])
            try:
                text = tokenizer.apply_chat_template(messages, tokenize=False)
            except Exception:
                text = str(messages)
        else:
            text = example.get(dataset_text_field, "")
        # Ensure text is a string (may be None or other type in malformed data)
        if text is None:
            text = ""
        elif not isinstance(text, str):
            text = str(text)
        tokens = tokenizer.encode(text, add_special_tokens=False)
        token_lengths.append(len(tokens))

    if not token_lengths:
        print("\n  ⚠️  No samples to estimate from.")
        return

    avg_tokens = sum(token_lengths) / len(token_lengths)
    min_tokens = min(token_lengths)
    max_tokens = max(token_lengths)
    median_tokens = sorted(token_lengths)[len(token_lengths) // 2]

    print(f"\n  Token length distribution (sampled {sample_size} examples):")
    print(f"    Min:    {min_tokens:>8,} tokens")
    print(f"    Median: {median_tokens:>8,} tokens")
    print(f"    Mean:   {avg_tokens:>8,.0f} tokens")
    print(f"    Max:    {max_tokens:>8,} tokens")

    # Estimate chunk counts based on truncation strategy
    if truncation_strategy == "split":
        # Each sample produces ceil(token_length / max_length) chunks
        est_chunks_per_sample = [math.ceil(tl / max_length) for tl in token_lengths]
        avg_chunks = sum(est_chunks_per_sample) / len(est_chunks_per_sample)
        est_total_chunks = int(avg_chunks * total_samples)

        over_length = sum(1 for tl in token_lengths if tl > max_length)
        over_pct = over_length / len(token_lengths) * 100

        print(f"\n  Estimated output (split strategy):")
        print(f"    Samples over max_length:    {over_pct:.0f}% ({over_length}/{sample_size} sampled)")
        print(f"    Avg chunks per sample:      {avg_chunks:.1f}")
        print(f"    Estimated total chunks:     ~{est_total_chunks:,}")

    elif truncation_strategy == "drop":
        over_length = sum(1 for tl in token_lengths if tl > max_length)
        drop_pct = over_length / len(token_lengths) * 100
        est_remaining = int(total_samples * (1 - drop_pct / 100))

        print(f"\n  Estimated output (drop strategy):")
        print(f"    Samples over max_length:    {drop_pct:.0f}% ({over_length}/{sample_size} sampled)")
        print(f"    Estimated samples dropped:  ~{total_samples - est_remaining:,}")
        print(f"    Estimated samples kept:     ~{est_remaining:,}")
        est_total_chunks = est_remaining

    else:  # truncate
        over_length = sum(1 for tl in token_lengths if tl > max_length)
        over_pct = over_length / len(token_lengths) * 100

        print(f"\n  Estimated output (truncate strategy):")
        print(f"    Samples over max_length:    {over_pct:.0f}% ({over_length}/{sample_size} sampled) — will be truncated")
        print(f"    Total samples:              {total_samples:,} (unchanged)")
        est_total_chunks = total_samples

    # Eval split estimate
    if eval_split and eval_split > 0:
        est_eval = int(est_total_chunks * eval_split)
        est_train = est_total_chunks - est_eval
        print(f"\n  Eval split ({eval_split:.1%}):")
        print(f"    Estimated train: ~{est_train:,}")
        print(f"    Estimated eval:  ~{est_eval:,}")
    else:
        est_train = est_total_chunks
        print(f"\n  No eval split — all {est_total_chunks:,} chunks go to train")

    # Disk usage estimate (rough: ~4 bytes per token for input_ids + labels + attention_mask)
    bytes_per_token = 4 * 3  # input_ids, labels, attention_mask (int32 each)
    avg_chunk_tokens = min(avg_tokens, max_length) if truncation_strategy != "split" else max_length
    est_bytes = est_total_chunks * avg_chunk_tokens * bytes_per_token
    est_mb = est_bytes / (1024 * 1024)

    print(f"\n  Estimated disk usage:           ~{est_mb:.0f} MB")

    print("\n  ✅ Dry run complete — no data was written.\n")


def main(
    config_path: str,
    output_override: Optional[str] = None,
    is_dry_run: bool = False,
    is_debug: bool = False,
    debug_max_tokens: int = 500,
):
    """Main entry point for the prepare command."""
    # Load training config
    training_config = _load_training_config(config_path)

    if is_debug:
        from loft.scripts.debug_tokens import run_debug
        run_debug(training_config, max_display_tokens=debug_max_tokens)
        return

    if is_dry_run:
        dry_run(training_config)
        return

    # Determine output directory
    output_dir = output_override or training_config.get("prepared_dataset") or training_config.get("output_dir")
    if not output_dir:
        raise ValueError(
            "Output directory must be specified via --output, or as 'prepared_dataset' "
            "or 'output_dir' in the training config."
        )

    # Run the pipeline
    dataset_dict, stats = prepare_dataset(training_config)

    # Save
    save_prepared_dataset(dataset_dict, output_dir, training_config, stats)


def make_parser(subparsers: Optional[argparse._SubParsersAction] = None):
    """Create the argument parser for the prepare command."""
    if subparsers is not None:
        parser = subparsers.add_parser(
            "prepare",
            help="Prepare, tokenize, and combine datasets for training (alias: prep)",
        )
    else:
        parser = argparse.ArgumentParser(
            description="Prepare, tokenize, and combine datasets for training (alias: prep)",
        )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config YAML (must have data_config and model_name_or_path)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory (overrides prepared_dataset/output_dir from config)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Preview what prepare would produce without tokenizing or saving anything",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Show tokenized loss mask view for one sample per dataset",
    )
    parser.add_argument(
        "--debug-max-tokens",
        type=int,
        default=500,
        help="Max tokens to display in debug token-level view (default: 500)",
    )

    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = make_parser()
    args = parser.parse_args()
    main(
        args.config,
        args.output,
        is_dry_run=args.dry_run,
        is_debug=args.debug,
        debug_max_tokens=args.debug_max_tokens,
    )
