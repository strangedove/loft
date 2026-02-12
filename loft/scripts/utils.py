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

import argparse
import importlib
import inspect
import logging
import os
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Optional, Union

import datasets
import yaml
from datasets import DatasetDict, concatenate_datasets
from transformers import HfArgumentParser
from transformers.hf_argparser import DataClass, DataClassType
from transformers.utils import is_rich_available


def _ensure_transformers_parallelism_config() -> None:
    """
    Ensure that ``transformers.training_args`` always defines the symbol `ParallelismConfig` so that Python's
    `typing.get_type_hints` can resolve annotations on `transformers.TrainingArguments` without raising a `NameError`.

    This is needed when running with ``accelerate<1.10.1``, where the module ``accelerate.parallelism_config`` did not
    exist and therefore the type alias is not imported by Transformers.

    See upstream fix PR in transformers#40818.
    """
    from typing import Any

    import transformers.training_args

    if not hasattr(transformers.training_args, "ParallelismConfig"):
        transformers.training_args.ParallelismConfig = Any


_ensure_transformers_parallelism_config()  # before creating HfArgumentParser

logger = logging.getLogger(__name__)


_MAX_CONFIG_INHERITANCE_DEPTH = 10


def resolve_config_inheritance(config: dict, config_path: Optional[str] = None) -> dict:
    """
    Resolve ``base_config`` inheritance for a YAML config.

    If ``config`` contains a ``base_config`` key, the referenced YAML is loaded
    first, then the child config is overlaid on top.  Supports chaining (a base
    can reference its own ``base_config``) with a depth limit to prevent cycles.

    For list-valued keys (e.g. ``lora_target_modules``, ``report_to``), the
    child value **replaces** the base value entirely (no merging). For
    dict-valued keys (e.g. ``env``), the child dict is shallow-merged on top of
    the base dict so you can override individual keys.

    Args:
        config: Raw dict loaded from a YAML file.
        config_path: Path to the YAML file that ``config`` came from, used to
            resolve relative ``base_config`` paths.  May be ``None`` if paths
            are absolute.

    Returns:
        A new dict with the inheritance chain fully resolved.  The
        ``base_config`` key is removed from the result.
    """
    if "base_config" not in config:
        return config

    seen: list[str] = []
    if config_path:
        seen.append(os.path.abspath(config_path))

    merged = _resolve_chain(config, config_path, seen, depth=0)
    return merged


def _resolve_chain(config: dict, config_path: Optional[str], seen: list[str], depth: int) -> dict:
    """Recursive helper for :func:`resolve_config_inheritance`."""
    if depth > _MAX_CONFIG_INHERITANCE_DEPTH:
        raise ValueError(
            f"Config inheritance depth exceeded {_MAX_CONFIG_INHERITANCE_DEPTH}. "
            f"Check for circular base_config references.\n  Chain: {' -> '.join(seen)}"
        )

    base_config_ref = config.get("base_config")
    if not base_config_ref:
        # No base — strip the key and return
        result = dict(config)
        result.pop("base_config", None)
        return result

    # Resolve relative paths: try CWD first (matching how other config paths
    # like data_config/output_dir work), then fall back to the config file's
    # directory for sibling-relative references.
    if not os.path.isabs(base_config_ref):
        cwd_resolved = os.path.abspath(base_config_ref)
        if os.path.isfile(cwd_resolved):
            base_config_ref = cwd_resolved
        elif config_path:
            base_config_ref = os.path.join(os.path.dirname(os.path.abspath(config_path)), base_config_ref)
            base_config_ref = os.path.abspath(base_config_ref)
        else:
            base_config_ref = cwd_resolved
    base_config_ref = os.path.abspath(base_config_ref)

    # Cycle detection
    if base_config_ref in seen:
        raise ValueError(
            f"Circular config inheritance detected: {base_config_ref}\n"
            f"  Chain: {' -> '.join(seen)} -> {base_config_ref}"
        )
    seen.append(base_config_ref)

    # Load the base config
    if not os.path.isfile(base_config_ref):
        raise FileNotFoundError(
            f"Base config file not found: {base_config_ref}\n"
            f"Referenced by base_config in: {config_path or '(unknown)'}"
        )
    with open(base_config_ref) as f:
        base = yaml.safe_load(f)
    if not isinstance(base, dict):
        raise ValueError(f"Base config {base_config_ref} must be a YAML mapping, got {type(base).__name__}.")

    # Recurse in case the base also has a base_config
    base = _resolve_chain(base, base_config_ref, seen, depth + 1)

    # Overlay: child values override base values
    child = dict(config)
    child.pop("base_config", None)

    merged = dict(base)
    for key, value in child.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            # Shallow-merge dicts (e.g. env)
            merged_dict = dict(merged[key])
            merged_dict.update(value)
            merged[key] = merged_dict
        else:
            # Everything else: child replaces base
            merged[key] = value

    return merged


@dataclass
class FileConfig:
    """
    Configuration for a single file within a dataset repository.

    Use this when you have multiple files in a single HuggingFace repo that need different
    processing settings (e.g., some are conversational, some are text).

    Parameters:
        file (`str`):
            Path to the file within the dataset (e.g., "conversations.parquet", "text/*.jsonl").
        columns (`list[str]`, *optional*):
            List of column names to select from this file.
        system_message (`str`, *optional*):
            System message to add to conversations in this file.
        truncation_strategy (`str`, *optional*):
            How to handle samples exceeding max_length for this file.
        subset (`int` or `float`, *optional*):
            Number of samples (if int) or fraction (if float 0-1) to take from this file.
        shuffle (`bool`, *optional*):
            Whether to shuffle this file before subsetting.
        eval_split (`float` or `False`, *optional*):
            Eval split fraction for this file, or `False` to exclude from eval.
        eval_before_subset (`bool`, *optional*):
            Whether to split eval before subsetting for this file.
        max_length (`int`, *optional*):
            Maximum sequence length for this file. Overrides the global `max_length` setting.
            Useful when you want shorter truncation for specific data (e.g., 2048 for short-form
            data while training at 4096 context).
        type (`str`, *optional*):
            Type of data in this file: "conversational" or "text". Overrides the dataset-level type.
        train_on_incomplete_assistant (`bool`, *optional*):
            If True, don't add the assistant end token (EOS) to the last assistant turn when the
            response is incomplete (truncated). Useful for training on samples where the assistant
            response was cut off. This per-file setting overrides the dataset-level setting.
    """

    file: str
    columns: Optional[list[str]] = None
    system_message: Optional[str] = None
    truncation_strategy: Optional[str] = None
    subset: Optional[Union[int, float]] = None
    shuffle: Optional[bool] = None
    eval_split: Optional[Union[float, bool]] = None
    eval_before_subset: Optional[bool] = None
    max_length: Optional[int] = None
    type: Optional[str] = None  # "conversational" or "text"
    train_on_incomplete_assistant: Optional[bool] = None


@dataclass
class DatasetConfig:
    """
    Configuration for a dataset.

    This class matches the signature of [`~datasets.load_dataset`] and the arguments are used directly in the
    `datasets.load_dataset` function. You can refer to the `datasets.load_dataset` documentation for more details.

    Parameters:
        path (`str`):
            Path or name of the dataset.
        name (`str`, *optional*):
            Defining the name of the dataset configuration.
        data_dir (`str`, *optional*):
            Defining the `data_dir` of the dataset configuration. If specified for the generic builders(csv, text etc.)
            or the Hub datasets and `data_files` is `None`, the behavior is equal to passing `os.path.join(data_dir,
            **)` as `data_files` to reference all the files in a directory.
        data_files (`str` or `Sequence` or `Mapping` or `list[FileConfig]`, *optional*):
            Path(s) to source data file(s). Can be:
            - A string: `"data.parquet"`
            - A list of strings: `["file1.parquet", "file2.parquet"]`
            - A list of FileConfig objects with per-file settings:
              ```yaml
              data_files:
                - file: conversations.parquet
                  truncation_strategy: truncate_turns
                - file: text_data.parquet
                  columns: [text]
                  truncation_strategy: split
              ```
        split (`str`, *optional*, defaults to `"train"`):
            Which split of the data to load.
        columns (`list[str]`, *optional*):
            List of column names to select from the dataset. If `None`, all columns are selected.
        system_message (`str`, *optional*):
            System message to add to conversations that don't have one. This per-dataset setting overrides
            the global `default_system_message` in the trainer config.
        truncation_strategy (`str`, *optional*):
            How to handle samples exceeding max_length. Options:
            - `"truncate"`: Cut off at max_length (default behavior)
            - `"drop"`: Filter out samples exceeding max_length
            - `"split"`: Split into chunks (text/CPT data only)
            - `"truncate_turns"`: Drop complete turn pairs from start, keep system message (conversational only)
            This per-dataset setting overrides the global `truncation_strategy` in the trainer config.
        subset (`int` or `float`, *optional*):
            Number of samples (if int) or fraction (if float 0-1) to take from this dataset.
            Applied after shuffling if `shuffle` is True.
        shuffle (`bool`, *optional*):
            Whether to shuffle this dataset before subsetting. If `None`, uses the global `shuffle_datasets` setting.
        eval_split (`float` or `False`, *optional*):
            Fraction of samples to split off for evaluation (0-1), or `False` to exclude this dataset from eval.
            If `None`, uses the global `eval_split` setting.
        eval_before_subset (`bool`, *optional*):
            Whether to split off eval data before applying subset. If `None`, uses the global setting.
            When `True`, eval is representative of full dataset. When `False` (default), eval size scales with subset.
        max_length (`int`, *optional*):
            Maximum sequence length for this dataset. Overrides the global `max_length` setting.
            Useful when you want shorter truncation for specific data (e.g., 2048 for short-form
            data while training at 4096 context).
        type (`str`, *optional*):
            Type of dataset: "conversational" or "text". If not specified, the format is auto-detected
            from the data (presence of "messages" key indicates conversational).
        files (`list[FileConfig]`, *optional*):
            Alternative to `data_files` for per-file configuration within a single repo. Each entry
            must have a `file` key and can override other settings per-file. Example:
            ```yaml
            files:
              - file: conversations.json
                truncation_strategy: truncate_turns
              - file: prose.json
                type: text
                truncation_strategy: split
            ```
        train_on_incomplete_assistant (`bool`, *optional*):
            If True, don't add the assistant end token (EOS) to the last assistant turn when the
            response is incomplete (truncated). Useful for training on samples where the assistant
            response was cut off. This per-dataset setting overrides the global setting.
    """

    path: str
    name: Optional[str] = None
    data_dir: Optional[str] = None
    data_files: Optional[Union[str, list, dict[str, str]]] = None
    split: str = "train"
    columns: Optional[list[str]] = None
    system_message: Optional[str] = None
    truncation_strategy: Optional[str] = None
    subset: Optional[Union[int, float]] = None
    shuffle: Optional[bool] = None
    eval_split: Optional[Union[float, bool]] = None
    eval_before_subset: Optional[bool] = None
    max_length: Optional[int] = None
    type: Optional[str] = None  # "conversational" or "text"
    files: Optional[list] = None  # Alternative to data_files with per-file settings
    train_on_incomplete_assistant: Optional[bool] = None


def _load_dataset_registry(registry_path: Optional[str] = None) -> dict:
    """
    Load the dataset registry YAML.

    The registry maps short names to dataset specifications::

        marvin:
          path: /tmp/marvin-dataset
          split: train
          description: "Marvin prose, 154 texts"

    Args:
        registry_path: Explicit path to registry YAML. If None, tries
            ``data/registry.yaml`` relative to CWD.

    Returns:
        Dict mapping dataset names to their config dicts. Empty dict if
        no registry file is found.
    """
    if registry_path is None:
        registry_path = os.path.join(os.getcwd(), "data", "registry.yaml")

    if not os.path.isfile(registry_path):
        return {}

    with open(registry_path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        logger.warning(f"Dataset registry {registry_path} is not a mapping, ignoring.")
        return {}

    return raw


def _resolve_dataset_entry(entry: dict, registry: dict) -> dict:
    """
    If a dataset entry uses ``dataset: <name>`` instead of ``path:``,
    resolve it from the registry. Per-entry overrides (subset, shuffle, etc.)
    are merged on top of the registry defaults.
    """
    name = entry.get("dataset")
    if name is None:
        return entry  # no resolution needed

    if name not in registry:
        available = ", ".join(sorted(registry.keys())) if registry else "(registry is empty)"
        raise ValueError(
            f"Dataset '{name}' not found in registry. Available: {available}\n"
            f"Add it to data/registry.yaml or use 'path:' instead."
        )

    # Start with registry defaults, overlay with per-entry overrides
    resolved = dict(registry[name])
    for key, value in entry.items():
        if key == "dataset":
            continue  # consumed, not passed through
        resolved[key] = value

    # Strip fields that aren't valid DatasetConfig params
    _valid_fields = {f.name for f in DatasetConfig.__dataclass_fields__.values()}
    resolved = {k: v for k, v in resolved.items() if k in _valid_fields}

    return resolved


@dataclass
class DatasetMixtureConfig:
    """
    Configuration class for a mixture of datasets.

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        datasets (`list[DatasetConfig]`):
            List of dataset configurations to include in the mixture.
        streaming (`bool`, *optional*, defaults to `False`):
            Whether to stream the datasets. If `True`, the datasets will be loaded in streaming mode.
        test_split_size (`float`, *optional*):
            Size of the test split. Refer to the `test_size` parameter in the [`~datasets.train_test_split`] function
            for more details. If `None`, the dataset will not be split into train and test sets.
            **Deprecated**: Use `eval_split` instead for more control over per-dataset splitting.
        shuffle_datasets (`bool`, *optional*, defaults to `True`):
            Whether to shuffle each dataset before subsetting. Can be overridden per-dataset.
        shuffle_combined (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the final combined dataset after all datasets are processed.
        eval_split (`float`, *optional*, defaults to `0.0`):
            Default fraction of samples to split off for evaluation from each dataset (0-1).
            Can be overridden per-dataset. Set to 0.0 to disable eval splitting.
        eval_before_subset (`bool`, *optional*, defaults to `False`):
            Whether to split off eval data before applying subset (globally).
            When `True`, eval is representative of full dataset.
            When `False` (default), eval size scales with subset.
        shuffle_seed (`int`, *optional*, defaults to `42`):
            Seed for all shuffle operations (dataset shuffling and combined shuffling).
        split_seed (`int`, *optional*, defaults to `42`):
            Seed for train/eval splits. Separate from shuffle_seed for reproducibility.

    Usage:
        When using the CLI, you can add the following section to your YAML config file:

        ```yaml
        # Global data processing options
        shuffle_datasets: true
        shuffle_combined: true
        eval_split: 0.05
        eval_before_subset: false
        shuffle_seed: 42
        split_seed: 42

        datasets:
          - path: dataset_a
            subset: 500
            shuffle: true
            eval_split: 0.05
          - path: dataset_b
            subset: 0.15
            shuffle: false
            eval_split: false
        ```
    """

    datasets: list[DatasetConfig] = field(
        default_factory=list,
        metadata={"help": "List of dataset configurations to include in the mixture."},
    )
    streaming: bool = field(
        default=False,
        metadata={"help": "Whether to stream the datasets. If True, the datasets will be loaded in streaming mode."},
    )
    test_split_size: Optional[float] = field(
        default=None,
        metadata={
            "help": "Size of the test split. Refer to the `test_size` parameter in the `datasets.train_test_split` "
            "function for more details. If None, the dataset will not be split into train and test sets. "
            "Deprecated: Use `eval_split` instead for more control over per-dataset splitting."
        },
    )
    shuffle_datasets: bool = field(
        default=True,
        metadata={"help": "Whether to shuffle each dataset before subsetting. Can be overridden per-dataset."},
    )
    shuffle_combined: bool = field(
        default=True,
        metadata={"help": "Whether to shuffle the final combined dataset after all datasets are processed."},
    )
    eval_split: float = field(
        default=0.0,
        metadata={
            "help": "Default fraction of samples to split off for evaluation from each dataset (0-1). "
            "Can be overridden per-dataset. Set to 0.0 to disable eval splitting."
        },
    )
    eval_before_subset: bool = field(
        default=False,
        metadata={
            "help": "Whether to split off eval data before applying subset. "
            "When True, eval is representative of full dataset. When False (default), eval size scales with subset."
        },
    )
    shuffle_seed: int = field(
        default=42,
        metadata={"help": "Seed for all shuffle operations (dataset shuffling and combined shuffling)."},
    )
    split_seed: int = field(
        default=42,
        metadata={"help": "Seed for train/eval splits. Separate from shuffle_seed for reproducibility."},
    )
    default_system_message: Optional[str] = field(
        default=None,
        metadata={
            "help": "Default system message to add to conversational data that lacks one. "
            "Can be overridden per-dataset with the `system_message` field."
        },
    )
    assistant_only_loss: bool = field(
        default=False,
        metadata={
            "help": "Whether to mask non-assistant tokens in the loss. "
            "When True, only assistant responses contribute to the loss."
        },
    )
    registry: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to a dataset registry YAML. If not set, auto-discovers data/registry.yaml. "
            "The registry maps short names to full dataset specs, so you can write "
            "'dataset: marvin' instead of 'path: /tmp/marvin-dataset'."
        },
    )

    def __post_init__(self):
        # Load dataset registry for name resolution
        _registry = _load_dataset_registry(self.registry)

        # Convert any dataset dicts (from CLI/config parsing) into DatasetConfig objects
        # and expand per-file configs into separate DatasetConfigs
        expanded_datasets = []
        for dataset in self.datasets:
            if isinstance(dataset, dict):
                # Resolve registry references (dataset: name -> path: ...)
                if "dataset" in dataset and "path" not in dataset:
                    dataset = _resolve_dataset_entry(dataset, _registry)
                # Handle 'files' as alias for 'data_files'
                if "files" in dataset and "data_files" not in dataset:
                    dataset["data_files"] = dataset.pop("files")
                dataset = DatasetConfig(**dataset)

            # Handle 'files' field as alias for 'data_files' on DatasetConfig objects
            if dataset.files is not None and dataset.data_files is None:
                dataset = DatasetConfig(
                    path=dataset.path,
                    name=dataset.name,
                    data_dir=dataset.data_dir,
                    data_files=dataset.files,
                    split=dataset.split,
                    columns=dataset.columns,
                    system_message=dataset.system_message,
                    truncation_strategy=dataset.truncation_strategy,
                    subset=dataset.subset,
                    shuffle=dataset.shuffle,
                    eval_split=dataset.eval_split,
                    eval_before_subset=dataset.eval_before_subset,
                    max_length=dataset.max_length,
                    type=dataset.type,
                )

            # Check if data_files contains FileConfig objects (per-file settings)
            if dataset.data_files is not None and isinstance(dataset.data_files, list):
                # Check if any item is a dict with 'file' key (FileConfig format)
                has_file_configs = any(
                    isinstance(item, dict) and "file" in item for item in dataset.data_files
                )
                if has_file_configs:
                    # Expand into separate DatasetConfigs, one per file
                    for file_item in dataset.data_files:
                        if isinstance(file_item, dict) and "file" in file_item:
                            # Create FileConfig then expand to DatasetConfig
                            file_config = FileConfig(**file_item)
                            # Create a new DatasetConfig with file-specific overrides
                            expanded = DatasetConfig(
                                path=dataset.path,
                                name=dataset.name,
                                data_dir=dataset.data_dir,
                                data_files=file_config.file,
                                split=dataset.split,
                                # Per-file settings override dataset-level settings
                                columns=file_config.columns if file_config.columns is not None else dataset.columns,
                                system_message=file_config.system_message if file_config.system_message is not None else dataset.system_message,
                                truncation_strategy=file_config.truncation_strategy if file_config.truncation_strategy is not None else dataset.truncation_strategy,
                                subset=file_config.subset if file_config.subset is not None else dataset.subset,
                                shuffle=file_config.shuffle if file_config.shuffle is not None else dataset.shuffle,
                                eval_split=file_config.eval_split if file_config.eval_split is not None else dataset.eval_split,
                                eval_before_subset=file_config.eval_before_subset if file_config.eval_before_subset is not None else dataset.eval_before_subset,
                                max_length=file_config.max_length if file_config.max_length is not None else dataset.max_length,
                                type=file_config.type if file_config.type is not None else dataset.type,
                            )
                            expanded_datasets.append(expanded)
                        elif isinstance(file_item, str):
                            # Plain string file path, use dataset-level settings
                            expanded = DatasetConfig(
                                path=dataset.path,
                                name=dataset.name,
                                data_dir=dataset.data_dir,
                                data_files=file_item,
                                split=dataset.split,
                                columns=dataset.columns,
                                system_message=dataset.system_message,
                                truncation_strategy=dataset.truncation_strategy,
                                subset=dataset.subset,
                                shuffle=dataset.shuffle,
                                eval_split=dataset.eval_split,
                                eval_before_subset=dataset.eval_before_subset,
                                max_length=dataset.max_length,
                                type=dataset.type,
                            )
                            expanded_datasets.append(expanded)
                    continue  # Don't add the original dataset

            expanded_datasets.append(dataset)

        self.datasets = expanded_datasets


@dataclass
class DataPrepConfig(DatasetMixtureConfig):
    """
    Configuration for dataset preparation.

    Extends DatasetMixtureConfig with additional options for preprocessing datasets
    in a model-agnostic way. The prepared dataset can then be tokenized for different
    models at training time.

    Parameters:
        trainer_type (`str`, *optional*, defaults to `"sft"`):
            Type of trainer this dataset is prepared for. Determines output format:
            - `"sft"`: Output has `messages` column (or `text` for non-conversational)
            - `"dpo"`, `"orpo"`: Output has `chosen` and `rejected` message columns
            - `"kto"`: Output has `completion` and `label` columns
        output_dir (`str`, *optional*):
            Directory to save the prepared dataset. If not specified, must be provided via CLI.

        > Preprocessing options (stored as metadata, applied at training time)

        default_system_message (`str`, *optional*):
            Default system message to add to conversations that don't have one.
        assistant_only_loss (`bool`, *optional*, defaults to `False`):
            Whether to compute loss only on assistant turns. Stored as `_assistant_only_loss` metadata.
        last_assistant_only_loss (`bool`, *optional*, defaults to `False`):
            Whether to compute loss only on the last assistant turn. Stored as `_last_assistant_only_loss` metadata.
        train_on_incomplete_assistant (`bool`, *optional*, defaults to `False`):
            Whether to train on incomplete assistant responses. Stored as `_train_on_incomplete_assistant` metadata.
        fix_turn_order (`bool`, *optional*, defaults to `False`):
            Whether to fix conversation turn order (add filler user message, merge consecutive roles, etc.).
        fix_turn_order_filler (`str`, *optional*, defaults to `"Let's begin."`):
            Filler message when conversation starts with assistant turn.
        truncation_strategy (`str`, *optional*, defaults to `"truncate"`):
            Default truncation strategy. Stored as `_truncation_strategy` metadata, applied at training time.
        num_proc (`int`, *optional*):
            Number of processes for dataset processing.

    Example config file (`data/my_data.yaml`):
        ```yaml
        trainer_type: sft
        output_dir: data/my_prepared

        shuffle_datasets: true
        shuffle_combined: true
        eval_split: 0.05
        shuffle_seed: 42
        split_seed: 42

        # Preprocessing options
        assistant_only_loss: true
        fix_turn_order: true
        default_system_message: "You are a helpful assistant."
        truncation_strategy: truncate

        datasets:
          - path: dataset-a
            subset: 5000
          - path: dataset-b
            subset: 0.15
            system_message: "You are a coding assistant."
            truncation_strategy: drop
        ```

    CLI usage:
        ```bash
        loft prepare data/my_data.yaml
        # or specify output via CLI
        loft prepare data/my_data.yaml --output data/my_prepared
        ```
    """

    trainer_type: str = field(
        default="sft",
        metadata={
            "help": "Type of trainer this dataset is prepared for: 'sft', 'dpo', 'orpo', 'kto'. "
            "Determines output format."
        },
    )
    output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory to save the prepared dataset."},
    )

    # Preprocessing options (stored as metadata)
    default_system_message: Optional[str] = field(
        default=None,
        metadata={"help": "Default system message to add to conversations that don't have one."},
    )
    assistant_only_loss: bool = field(
        default=False,
        metadata={"help": "Whether to compute loss only on assistant turns."},
    )
    last_assistant_only_loss: bool = field(
        default=False,
        metadata={"help": "Whether to compute loss only on the last assistant turn."},
    )
    train_on_incomplete_assistant: bool = field(
        default=False,
        metadata={"help": "Whether to train on incomplete/truncated assistant responses."},
    )
    fix_turn_order: bool = field(
        default=False,
        metadata={"help": "Whether to fix conversation turn order."},
    )
    fix_turn_order_filler: str = field(
        default="Let's begin.",
        metadata={"help": "Filler message when conversation starts with assistant turn."},
    )
    truncation_strategy: str = field(
        default="truncate",
        metadata={"help": "Default truncation strategy: 'truncate', 'drop', 'split', 'truncate_turns'."},
    )
    num_proc: Optional[int] = field(
        default=None,
        metadata={"help": "Number of processes for dataset processing."},
    )

    def __post_init__(self):
        super().__post_init__()
        # Validate trainer_type
        valid_trainer_types = {"sft", "dpo", "orpo", "kto"}
        if self.trainer_type not in valid_trainer_types:
            raise ValueError(
                f"Invalid trainer_type: {self.trainer_type}. Must be one of: {', '.join(sorted(valid_trainer_types))}"
            )
        # Validate truncation_strategy
        valid_strategies = {"truncate", "drop", "split", "truncate_turns"}
        if self.truncation_strategy not in valid_strategies:
            raise ValueError(
                f"Invalid truncation_strategy: {self.truncation_strategy}. "
                f"Must be one of: {', '.join(sorted(valid_strategies))}"
            )


@dataclass
class ScriptArguments:
    """
    Arguments common to all scripts.

    Args:
        dataset_name (`str`,, *optional*):
            Path or name of the dataset to load. If `datasets` is provided, this will be ignored.
        dataset_config (`str`, *optional*):
            Dataset configuration name. Corresponds to the `name` argument of the [`~datasets.load_dataset`] function.
            If `datasets` is provided, this will be ignored.
        dataset_train_split (`str`, *optional*, defaults to `"train"`):
            Dataset split to use for training. If `datasets` is provided, this will be ignored.
        dataset_test_split (`str`, *optional*, defaults to `"test"`):
            Dataset split to use for evaluation. If `datasets` is provided, this will be ignored.
        dataset_streaming (`bool`, *optional*, defaults to `False`):
            Whether to stream the dataset. If True, the dataset will be loaded in streaming mode. If `datasets` is
            provided, this will be ignored.
        gradient_checkpointing_use_reentrant (`bool`, *optional*, defaults to `False`):
            Whether to apply `use_reentrant` for gradient checkpointing.
        ignore_bias_buffers (`bool`, *optional*, defaults to `False`):
            Debug argument for distributed training. Fix for DDP issues with LM bias/mask buffers - invalid scalar
            type, inplace operation. See
            https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "Path or name of the dataset to load. If `datasets` is provided, this will be ignored."},
    )
    dataset_config: Optional[str] = field(
        default=None,
        metadata={
            "help": "Dataset configuration name. Corresponds to the `name` argument of the `datasets.load_dataset` "
            "function. If `datasets` is provided, this will be ignored."
        },
    )
    dataset_train_split: str = field(
        default="train",
        metadata={"help": "Dataset split to use for training. If `datasets` is provided, this will be ignored."},
    )
    dataset_test_split: str = field(
        default="test",
        metadata={"help": "Dataset split to use for evaluation. If `datasets` is provided, this will be ignored."},
    )
    dataset_streaming: bool = field(
        default=False,
        metadata={
            "help": "Whether to stream the dataset. If True, the dataset will be loaded in streaming mode. If "
            "`datasets` is provided, this will be ignored."
        },
    )
    gradient_checkpointing_use_reentrant: bool = field(
        default=False,
        metadata={"help": "Whether to apply `use_reentrant` for gradient checkpointing."},
    )
    ignore_bias_buffers: bool = field(
        default=False,
        metadata={
            "help": "Debug argument for distributed training. Fix for DDP issues with LM bias/mask buffers - invalid "
            "scalar type, inplace operation. See "
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992."
        },
    )


def init_zero_verbose():
    """
    Perform zero verbose init - use this method on top of the CLI modules to make logging and warning output cleaner.
    Uses Rich if available, falls back otherwise.
    """
    import logging
    import warnings

    FORMAT = "%(message)s"

    if is_rich_available():
        from rich.logging import RichHandler

        handler = RichHandler()
    else:
        handler = logging.StreamHandler()

    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[handler], level=logging.ERROR)

    # Custom warning handler to redirect warnings to the logging system
    def warning_handler(message, category, filename, lineno, file=None, line=None):
        logging.warning(f"{filename}:{lineno}: {category.__name__}: {message}")

    # Add the custom warning handler - we need to do that before importing anything to make sure the loggers work well
    warnings.showwarning = warning_handler


class TrlParser(HfArgumentParser):
    """
    A subclass of [`transformers.HfArgumentParser`] designed for parsing command-line arguments with dataclass-backed
    configurations, while also supporting configuration file loading and environment variable management.

    Args:
        dataclass_types (`Union[DataClassType, Iterable[DataClassType]]`, *optional*):
            Dataclass types to use for argument parsing.
        **kwargs:
            Additional keyword arguments passed to the [`transformers.HfArgumentParser`] constructor.

    Examples:

    ```yaml
    # config.yaml
    env:
        VAR1: value1
    arg1: 23
    ```

    ```python
    # main.py
    import os
    from dataclasses import dataclass
    from trl import TrlParser


    @dataclass
    class MyArguments:
        arg1: int
        arg2: str = "alpha"


    parser = TrlParser(dataclass_types=[MyArguments])
    training_args = parser.parse_args_and_config()

    print(training_args, os.environ.get("VAR1"))
    ```

    ```bash
    $ python main.py --config config.yaml
    (MyArguments(arg1=23, arg2='alpha'),) value1

    $ python main.py --arg1 5 --arg2 beta
    (MyArguments(arg1=5, arg2='beta'),) None
    ```
    """

    def __init__(
        self,
        dataclass_types: Optional[Union[DataClassType, Iterable[DataClassType]]] = None,
        **kwargs,
    ):
        # Make sure dataclass_types is an iterable
        if dataclass_types is None:
            dataclass_types = []
        elif not isinstance(dataclass_types, Iterable):
            dataclass_types = [dataclass_types]

        # Check that none of the dataclasses have the "config" field
        for dataclass_type in dataclass_types:
            if "config" in dataclass_type.__dataclass_fields__:
                raise ValueError(
                    f"Dataclass {dataclass_type.__name__} has a field named 'config'. This field is reserved for the "
                    f"config file path and should not be used in the dataclass."
                )

        super().__init__(dataclass_types=dataclass_types, **kwargs)

    def parse_args_and_config(
        self,
        args: Optional[Iterable[str]] = None,
        return_remaining_strings: bool = False,
        fail_with_unknown_args: bool = True,
    ) -> tuple[DataClass, ...]:
        """
        Parse command-line args and config file into instances of the specified dataclass types.

        This method wraps [`transformers.HfArgumentParser.parse_args_into_dataclasses`] and also parses the config file
        specified with the `--config` flag. The config file (in YAML format) provides argument values that replace the
        default values in the dataclasses. Command line arguments can override values set by the config file. The
        method also sets any environment variables specified in the `env` field of the config file.
        """
        args = list(args) if args is not None else sys.argv[1:]
        # Support positional config: "sft.py config.yaml" → "sft.py --config config.yaml"
        if "--config" not in args and args and not args[0].startswith("-") and (args[0].endswith(".yaml") or args[0].endswith(".yml")):
            args = ["--config", args[0]] + args[1:]
        if "--config" in args:
            # Get the config file path from
            config_index = args.index("--config")
            args.pop(config_index)  # remove the --config flag
            config_path = args.pop(config_index)  # get the path to the config file
            with open(config_path) as yaml_file:
                config = yaml.safe_load(yaml_file)

            # Resolve config inheritance (base_config field)
            config = resolve_config_inheritance(config, config_path)

            # Set the environment variables specified in the config file
            if "env" in config:
                env_vars = config.pop("env", {})
                if not isinstance(env_vars, dict):
                    raise ValueError("`env` field should be a dict in the YAML file.")
                for key, value in env_vars.items():
                    os.environ[key] = str(value)

            # Set the defaults from the config values
            config_remaining_strings = self.set_defaults_with_config(**config)
        else:
            config_remaining_strings = []

        # Parse the arguments from the command line
        output = self.parse_args_into_dataclasses(args=args, return_remaining_strings=return_remaining_strings)

        # Merge remaining strings from the config file with the remaining strings from the command line
        if return_remaining_strings:
            args_remaining_strings = output[-1]
            return output[:-1] + (config_remaining_strings + args_remaining_strings,)
        elif fail_with_unknown_args and config_remaining_strings:
            raise ValueError(
                f"Unknown arguments from config file: {config_remaining_strings}. Please remove them, add them to the "
                "dataclass, or set `fail_with_unknown_args=False`."
            )
        else:
            return output

    def set_defaults_with_config(self, **kwargs) -> list[str]:
        """
        Overrides the parser's default values with those provided via keyword arguments, including for subparsers.

        Any argument with an updated default will also be marked as not required if it was previously required.

        Returns a list of strings that were not consumed by the parser.
        """

        def apply_defaults(parser, kw):
            used_keys = set()
            for action in parser._actions:
                # Handle subparsers recursively
                if isinstance(action, argparse._SubParsersAction):
                    for subparser in action.choices.values():
                        used_keys.update(apply_defaults(subparser, kw))
                elif action.dest in kw:
                    action.default = kw[action.dest]
                    action.required = False
                    used_keys.add(action.dest)
            return used_keys

        used_keys = apply_defaults(self, kwargs)
        # Remaining args not consumed by the parser
        remaining = [
            item for key, value in kwargs.items() if key not in used_keys for item in (f"--{key}", str(value))
        ]
        return remaining


def get_git_commit_hash(package_name):
    try:
        # Import the package to locate its path
        package = importlib.import_module(package_name)
        # Get the path to the package using inspect
        package_path = os.path.dirname(inspect.getfile(package))

        # Navigate up to the Git repository root if the package is inside a subdirectory
        git_repo_path = os.path.abspath(os.path.join(package_path, ".."))
        git_dir = os.path.join(git_repo_path, ".git")

        if os.path.isdir(git_dir):
            # Run the git command to get the current commit hash
            commit_hash = (
                subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=git_repo_path).strip().decode("utf-8")
            )
            return commit_hash
        else:
            return None
    except Exception as e:
        return f"Error: {str(e)}"


def _apply_subset(dataset: datasets.Dataset, subset: Optional[Union[int, float]], seed: int) -> datasets.Dataset:
    """Apply subset selection to a dataset."""
    if subset is None:
        return dataset

    dataset_len = len(dataset)
    if isinstance(subset, float):
        if not 0 < subset <= 1:
            raise ValueError(f"Subset fraction must be between 0 and 1, got {subset}")
        n_samples = int(dataset_len * subset)
    else:
        n_samples = min(subset, dataset_len)

    if n_samples >= dataset_len:
        return dataset

    # Use select with indices to take the first n_samples (dataset should already be shuffled if needed)
    return dataset.select(range(n_samples))


def _process_single_dataset(
    dataset_config: DatasetConfig,
    mixture_config: DatasetMixtureConfig,
) -> tuple[Optional[datasets.Dataset], Optional[datasets.Dataset]]:
    """
    Process a single dataset: load, shuffle, subset, and split eval.

    Returns:
        Tuple of (train_dataset, eval_dataset). Either can be None.
    """
    logger.info(f"Loading dataset for mixture: {dataset_config.path} (config name: {dataset_config.name})")

    load_path = dataset_config.path
    load_data_files = dataset_config.data_files

    # If path points to a local file (not a directory or Hub ID), use the appropriate
    # format loader with data_files instead of passing the file path directly.
    # datasets.load_dataset() expects a directory, Hub ID, or builder name — not a file path.
    if os.path.isfile(load_path):
        _ext_to_loader = {
            ".json": "json",
            ".jsonl": "json",
            ".csv": "json",
            ".parquet": "parquet",
            ".arrow": "arrow",
            ".txt": "text",
        }
        ext = os.path.splitext(load_path)[1].lower()
        loader = _ext_to_loader.get(ext)
        if loader is None:
            raise ValueError(
                f"Dataset path '{load_path}' is a file but has unsupported extension '{ext}'. "
                f"Supported: {', '.join(sorted(_ext_to_loader.keys()))}"
            )
        load_data_files = load_path
        load_path = loader

    dataset = datasets.load_dataset(
        path=load_path,
        name=dataset_config.name,
        data_dir=dataset_config.data_dir,
        data_files=load_data_files,
        split=dataset_config.split,
        streaming=mixture_config.streaming,
    )

    # For streaming datasets, we can't do shuffle/subset/split operations
    if mixture_config.streaming:
        if dataset_config.columns is not None:
            dataset = dataset.select_columns(dataset_config.columns)
        return dataset, None

    original_len = len(dataset)

    # Select columns if specified
    if dataset_config.columns is not None:
        dataset = dataset.select_columns(dataset_config.columns)

    # For conversational datasets, normalize columns and message structure so that
    # datasets with extra metadata columns/keys can be concatenated safely.
    # Also converts ShareGPT format (conversations/from/value) to ChatML (messages/role/content).
    if dataset_config.type == "conversational" and dataset_config.columns is None:
        _SHAREGPT_ROLE_MAP = {"human": "user", "gpt": "assistant", "system": "system"}
        _chatml_features = datasets.Features({
            "messages": [{"role": datasets.Value("string"), "content": datasets.Value("string")}]
        })
        msg_col = None
        if "messages" in dataset.column_names:
            msg_col = "messages"
        elif "conversations" in dataset.column_names:
            msg_col = "conversations"

        if msg_col is not None:
            # Drop all columns except the message column
            extra_cols = [c for c in dataset.column_names if c != msg_col]
            if extra_cols:
                logger.info(f"  Dropping extra columns for conversational dataset: {extra_cols}")
                dataset = dataset.remove_columns(extra_cols)

            # Check format of message dicts
            sample_msg = dataset[0][msg_col][0] if len(dataset) > 0 and dataset[0][msg_col] else None
            needs_normalize = False
            if sample_msg:
                is_sharegpt = "from" in sample_msg and "value" in sample_msg
                has_extra_keys = set(sample_msg.keys()) != {"role", "content"}
                needs_normalize = is_sharegpt or has_extra_keys or msg_col != "messages"

            if needs_normalize:
                is_sharegpt = "from" in sample_msg and "value" in sample_msg
                if is_sharegpt:
                    logger.info(f"  Converting ShareGPT format to ChatML (column: {msg_col})")

                    def _convert_sharegpt(ex, _col=msg_col, _rmap=_SHAREGPT_ROLE_MAP):
                        return {
                            "messages": [
                                {"role": _rmap.get(m["from"], m["from"]), "content": m["value"]}
                                for m in ex[_col]
                            ]
                        }

                    dataset = dataset.map(
                        _convert_sharegpt,
                        features=_chatml_features,
                        remove_columns=[msg_col] if msg_col != "messages" else [],
                        desc="Converting ShareGPT to ChatML",
                    )
                else:
                    extra_keys = set(sample_msg.keys()) - {"role", "content"}
                    logger.info(f"  Stripping extra message keys: {extra_keys}")
                    dataset = dataset.map(
                        lambda ex: {
                            "messages": [{"role": m["role"], "content": m["content"]} for m in ex["messages"]]
                        },
                        features=_chatml_features,
                        desc="Normalizing messages",
                    )

    # Add per-dataset metadata columns
    # Build a short name for the dataset (path + first data_file if any)
    dataset_name = dataset_config.path
    if dataset_config.data_files:
        if isinstance(dataset_config.data_files, str):
            dataset_name = f"{dataset_config.path}/{dataset_config.data_files}"
        elif isinstance(dataset_config.data_files, list) and len(dataset_config.data_files) > 0:
            first_file = dataset_config.data_files[0]
            if isinstance(first_file, str):
                dataset_name = f"{dataset_config.path}/{first_file}"
    dataset = dataset.add_column("_dataset_name", [dataset_name] * len(dataset))

    if dataset_config.system_message is not None:
        dataset = dataset.add_column("_system_message", [dataset_config.system_message] * len(dataset))
    if dataset_config.truncation_strategy is not None:
        dataset = dataset.add_column("_truncation_strategy", [dataset_config.truncation_strategy] * len(dataset))
    if dataset_config.max_length is not None:
        dataset = dataset.add_column("_max_length", [dataset_config.max_length] * len(dataset))
    # Mark samples that should be excluded from eval split
    if dataset_config.eval_split is False:
        dataset = dataset.add_column("_no_eval", [True] * len(dataset))

    # Resolve per-dataset settings with global defaults
    should_shuffle = dataset_config.shuffle if dataset_config.shuffle is not None else mixture_config.shuffle_datasets
    eval_split_value = dataset_config.eval_split if dataset_config.eval_split is not None else mixture_config.eval_split
    eval_before_subset = (
        dataset_config.eval_before_subset
        if dataset_config.eval_before_subset is not None
        else mixture_config.eval_before_subset
    )

    # Determine if we should do eval split
    do_eval_split = eval_split_value is not False and eval_split_value > 0

    # Step 1: Shuffle if requested
    if should_shuffle:
        dataset = dataset.shuffle(seed=mixture_config.shuffle_seed)
        logger.info(f"  Shuffled dataset (seed={mixture_config.shuffle_seed})")

    train_dataset = dataset
    eval_dataset = None

    # Step 2a: If eval_before_subset, split eval first
    if do_eval_split and eval_before_subset:
        split_result = train_dataset.train_test_split(test_size=eval_split_value, seed=mixture_config.split_seed)
        train_dataset = split_result["train"]
        eval_dataset = split_result["test"]
        logger.info(
            f"  Split eval before subset: {len(eval_dataset)} eval, {len(train_dataset)} remaining "
            f"(eval_split={eval_split_value})"
        )

    # Step 3: Apply subset
    if dataset_config.subset is not None:
        before_len = len(train_dataset)
        train_dataset = _apply_subset(train_dataset, dataset_config.subset, mixture_config.shuffle_seed)
        logger.info(f"  Applied subset: {before_len} -> {len(train_dataset)} samples")

    # Step 2b: If not eval_before_subset, split eval after subset
    if do_eval_split and not eval_before_subset:
        split_result = train_dataset.train_test_split(test_size=eval_split_value, seed=mixture_config.split_seed)
        train_dataset = split_result["train"]
        eval_dataset = split_result["test"]
        logger.info(
            f"  Split eval after subset: {len(eval_dataset)} eval, {len(train_dataset)} train "
            f"(eval_split={eval_split_value})"
        )

    logger.info(
        f"  Final: {original_len} original -> {len(train_dataset)} train"
        + (f", {len(eval_dataset)} eval" if eval_dataset else "")
    )

    return train_dataset, eval_dataset


def get_dataset(mixture_config: DatasetMixtureConfig) -> DatasetDict:
    """
    Load a mixture of datasets based on the configuration.

    This function handles:
    - Per-dataset shuffling (controlled by `shuffle` or global `shuffle_datasets`)
    - Per-dataset subsetting (controlled by `subset` - int for count, float for fraction)
    - Per-dataset eval splitting (controlled by `eval_split` or global `eval_split`)
    - Eval split timing (controlled by `eval_before_subset`)
    - Final combined dataset shuffling (controlled by `shuffle_combined`)

    Args:
        mixture_config (`DatasetMixtureConfig`):
            Script arguments containing dataset configuration.

    Returns:
        `DatasetDict`:
            Combined dataset(s) from the mixture configuration. Contains "train" split and optionally
            "test" split if eval_split > 0 or test_split_size is set.

    Example:
    ```python
    from trl import DatasetMixtureConfig, get_dataset
    from trl.scripts.utils import DatasetConfig

    mixture_config = DatasetMixtureConfig(
        datasets=[DatasetConfig(path="trl-lib/tldr", subset=1000)],
        shuffle_datasets=True,
        eval_split=0.05,
    )
    dataset = get_dataset(mixture_config)
    print(dataset)
    ```

    ```
    DatasetDict({
        train: Dataset({
            features: ['prompt', 'completion'],
            num_rows: 950
        })
        test: Dataset({
            features: ['prompt', 'completion'],
            num_rows: 50
        })
    })
    ```
    """
    logger.info(f"Creating dataset mixture with {len(mixture_config.datasets)} datasets")
    logger.info(
        f"  Global settings: shuffle_datasets={mixture_config.shuffle_datasets}, "
        f"shuffle_combined={mixture_config.shuffle_combined}, eval_split={mixture_config.eval_split}, "
        f"eval_before_subset={mixture_config.eval_before_subset}"
    )

    train_datasets = []
    eval_datasets = []

    for dataset_config in mixture_config.datasets:
        train_ds, eval_ds = _process_single_dataset(dataset_config, mixture_config)
        if train_ds is not None:
            train_datasets.append(train_ds)
        if eval_ds is not None:
            eval_datasets.append(eval_ds)

    if not train_datasets:
        raise ValueError("No datasets were loaded from the mixture configuration")

    # Combine train datasets
    combined_train = concatenate_datasets(train_datasets)
    if isinstance(combined_train, datasets.Dataset):
        logger.info(f"Combined train dataset: {len(combined_train)} examples")

    # Combine eval datasets if any
    combined_eval = None
    if eval_datasets:
        combined_eval = concatenate_datasets(eval_datasets)
        if isinstance(combined_eval, datasets.Dataset):
            logger.info(f"Combined eval dataset: {len(combined_eval)} examples")

    # Shuffle combined datasets if requested
    if mixture_config.shuffle_combined and not mixture_config.streaming:
        combined_train = combined_train.shuffle(seed=mixture_config.shuffle_seed)
        logger.info(f"Shuffled combined train dataset (seed={mixture_config.shuffle_seed})")
        if combined_eval is not None:
            combined_eval = combined_eval.shuffle(seed=mixture_config.shuffle_seed)
            logger.info(f"Shuffled combined eval dataset (seed={mixture_config.shuffle_seed})")

    # Handle legacy test_split_size (deprecated but still supported)
    if mixture_config.test_split_size is not None and combined_eval is None:
        logger.warning(
            "test_split_size is deprecated. Use eval_split instead for per-dataset control. "
            "Falling back to legacy behavior."
        )
        split_result = combined_train.train_test_split(
            test_size=mixture_config.test_split_size, seed=mixture_config.split_seed
        )
        return split_result

    # Build result
    if combined_eval is not None:
        return DatasetDict({"train": combined_train, "test": combined_eval})
    else:
        return DatasetDict({"train": combined_train})


def load_prepared_dataset(prepared_path: str) -> DatasetDict:
    """
    Load a prepared dataset from disk.

    Args:
        prepared_path: Path to the prepared dataset directory (created by `loft prepare`).
            Should contain train.parquet, optionally test.parquet, and blend_metadata.json.

    Returns:
        DatasetDict with train and optionally test splits.
    """
    import os

    if not os.path.isdir(prepared_path):
        raise ValueError(f"Prepared dataset path does not exist: {prepared_path}")

    result = {}

    # Load train split
    train_path = os.path.join(prepared_path, "train.parquet")
    if os.path.exists(train_path):
        result["train"] = datasets.Dataset.from_parquet(train_path)
        logger.info(f"Loaded train split from {train_path}: {len(result['train'])} examples")
    else:
        raise ValueError(f"Train split not found at {train_path}")

    # Load test split if exists
    test_path = os.path.join(prepared_path, "test.parquet")
    if os.path.exists(test_path):
        result["test"] = datasets.Dataset.from_parquet(test_path)
        logger.info(f"Loaded test split from {test_path}: {len(result['test'])} examples")

    return DatasetDict(result)


def is_pretokenized_prepare(prepared_path: str) -> bool:
    """
    Check if a prepared dataset was created by new-style prepare (pre-tokenized).

    Returns True if blend_metadata.json exists and has ``"tokenized": true``.
    """
    import json
    import os

    metadata_path = os.path.join(prepared_path, "blend_metadata.json")
    if not os.path.exists(metadata_path):
        return False
    try:
        with open(metadata_path) as f:
            metadata = json.load(f)
        return metadata.get("tokenized", False)
    except (json.JSONDecodeError, OSError):
        return False


def _load_prepare_metadata(prepared_path: str) -> Optional[dict]:
    """Load blend_metadata.json from a prepared dataset directory, or return None."""
    import json
    import os

    metadata_path = os.path.join(prepared_path, "blend_metadata.json")
    if not os.path.exists(metadata_path):
        return None
    try:
        with open(metadata_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def validate_prepare_metadata(prepared_path: str, training_config: dict) -> list[str]:
    """
    Compare prepare metadata against the current training config.

    Returns a list of mismatch descriptions. Empty list means everything matches.

    The training_config dict should be built via build_prepare_config(), which
    handles precedence (training args > data_config > defaults).
    """
    metadata = _load_prepare_metadata(prepared_path)
    if metadata is None:
        return ["No blend_metadata.json found — cannot verify dataset matches config."]

    mismatches = []

    # Fields to compare: (metadata_key, config_key, label)
    checks = [
        ("model_name_or_path", "model_name_or_path", "Model"),
        ("max_length", "max_length", "Max length"),
        ("truncation_strategy", "truncation_strategy", "Truncation strategy"),
        ("data_config", "data_config", "Data config"),
        ("dataset_text_field", "dataset_text_field", "Dataset text field"),
    ]

    for meta_key, config_key, label in checks:
        meta_val = metadata.get(meta_key)
        config_val = training_config.get(config_key)
        if config_val is not None and meta_val is not None and str(meta_val) != str(config_val):
            mismatches.append(f"  {label}: prepared has '{meta_val}', config has '{config_val}'")

    # Check eval_split separately (float comparison)
    meta_eval = metadata.get("eval_split", 0.0)
    config_eval = training_config.get("eval_split", 0.0)
    if abs(float(meta_eval) - float(config_eval)) > 1e-6:
        mismatches.append(f"  Eval split: prepared has {meta_eval}, config has {config_eval}")

    # Check split_seed (only matters if eval_split > 0)
    if config_eval > 0:
        meta_seed = metadata.get("split_seed", 42)
        config_seed = training_config.get("split_seed", 42)
        if meta_seed != config_seed:
            mismatches.append(f"  Split seed: prepared has {meta_seed}, config has {config_seed}")

    return mismatches


def needs_prepare(prepared_path: str) -> bool:
    """Check if a prepared dataset path is missing, empty, or has no train.parquet."""
    import os

    if not prepared_path:
        return True
    if not os.path.isdir(prepared_path):
        return True
    train_path = os.path.join(prepared_path, "train.parquet")
    return not os.path.exists(train_path)


def build_prepare_config(training_args, model_args) -> dict:
    """
    Build the config dict that prepare.prepare_dataset() expects from SFT training args.

    Config value precedence (highest to lowest):
      1. training_args (main training config)
      2. data_config file
      3. hardcoded defaults

    This allows users to set defaults in data_config (tied to the dataset) while
    overriding specific values in the main training config for experiments.
    """
    # Load data_config as base values
    data_config = {}
    if training_args.data_config and os.path.exists(training_args.data_config):
        with open(training_args.data_config) as f:
            data_config = yaml.safe_load(f) or {}

    # Helper to get value with precedence: training_args > data_config > default
    def get_with_precedence(attr_name, default=None):
        # Check training_args first (main config)
        val = getattr(training_args, attr_name, None)
        if val is not None:
            return val
        # Fall back to data_config
        val = data_config.get(attr_name)
        if val is not None:
            return val
        # Fall back to default
        return default

    config = {
        "model_name_or_path": model_args.model_name_or_path,
        "trust_remote_code": model_args.trust_remote_code,
        "data_config": training_args.data_config,
        "max_length": training_args.max_length,
        "truncation_strategy": getattr(training_args, "truncation_strategy", "truncate"),
        "dataset_text_field": getattr(training_args, "dataset_text_field", "text"),
        "prepared_dataset": training_args.prepared_dataset,
        # These support override from main config
        "eval_split": get_with_precedence("eval_split", 0.0),
        "split_seed": get_with_precedence("split_seed", 42),
        "default_system_message": get_with_precedence("default_system_message"),
        "assistant_only_loss": get_with_precedence("assistant_only_loss", False),
        "last_assistant_only_loss": get_with_precedence("last_assistant_only_loss", False),
        "train_on_incomplete_assistant": get_with_precedence("train_on_incomplete_assistant", False),
        "fix_turn_order": get_with_precedence("fix_turn_order", False),
        "fix_turn_order_filler": get_with_precedence("fix_turn_order_filler"),
        "dataset_num_proc": get_with_precedence("dataset_num_proc"),
        "chat_template_path": get_with_precedence("chat_template_path"),
    }

    # Remove None values to keep config clean
    config = {k: v for k, v in config.items() if v is not None}

    return config


def run_auto_prepare(training_args, model_args) -> None:
    """
    Run the prepare pipeline automatically from within sft.py.

    Only the main process (LOCAL_RANK 0) runs prepare. Other processes wait
    for the output directory to be populated.
    """
    import os
    from loft.scripts.prepare import prepare_dataset, save_prepared_dataset

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    output_dir = training_args.prepared_dataset
    metadata_path = os.path.join(output_dir, "blend_metadata.json")

    if local_rank == 0:
        # Delete metadata file first so other processes know we're re-preparing
        if os.path.exists(metadata_path):
            os.remove(metadata_path)

        prepare_config = build_prepare_config(training_args, model_args)
        print(f"\n🔄 Auto-preparing dataset at {output_dir}...")
        dataset_dict, stats = prepare_dataset(prepare_config)
        save_prepared_dataset(dataset_dict, output_dir, prepare_config, stats)
        print()

    # Barrier: non-main processes wait for the output to exist
    if local_rank != 0:
        import time
        train_path = os.path.join(output_dir, "train.parquet")
        # Wait up to 30 minutes for main process to finish preparing
        for _ in range(1800):
            if os.path.exists(train_path) and os.path.exists(metadata_path):
                break
            time.sleep(1)
        else:
            raise TimeoutError(
                f"Waited 30 minutes for prepare to complete at {output_dir} but it never finished."
            )


def prompt_prepare_overwrite(prepared_path: str, mismatches: list[str], force: bool = False) -> bool:
    """
    Prompt user (on rank 0 only) whether to overwrite a mismatched prepared dataset.

    Returns True if user wants to re-prepare, False to abort.
    In non-interactive environments, defaults to aborting with an error message
    unless ``force=True``.
    """
    import os
    import sys

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank != 0:
        # Non-main processes shouldn't prompt; they'll follow main's decision
        return False

    print("\n⚠️  Prepared dataset metadata does not match current training config:")
    for m in mismatches:
        print(m)
    print(f"\n  Prepared dataset: {prepared_path}")
    print()

    # force_prepare skips the prompt entirely
    if force:
        print("🔄 force_prepare=True — auto-overwriting with re-prepared data.")
        return True

    # Check if we can prompt interactively
    if not sys.stdin.isatty():
        print(
            "❌ Non-interactive environment — cannot prompt for confirmation.\n"
            "   Run `loft prepare` manually, delete the prepared dataset directory,\n"
            "   or set force_prepare: true in your training config."
        )
        sys.exit(1)

    while True:
        response = input("Overwrite with re-prepared data matching current config? [y/n]: ").strip().lower()
        if response in ("y", "yes"):
            return True
        elif response in ("n", "no"):
            print("❌ Aborting. Update your config's prepared_dataset path or run `loft prepare` manually.")
            sys.exit(1)
        else:
            print("  Please enter 'y' or 'n'.")


def get_tokenized_cache_path(
    prepared_path: str,
    model_name_or_path: str,
    max_length: int,
    cache_dir: Optional[str] = None,
) -> str:
    """
    Compute the cache path for a tokenized dataset.

    The cache key is based on:
    - The prepared dataset's config hash (from blend_metadata.json)
    - The model name/path
    - The max_length

    Args:
        prepared_path: Path to the prepared dataset directory.
        model_name_or_path: Model identifier.
        max_length: Maximum sequence length.
        cache_dir: Optional cache directory. If None, uses {prepared_path}/.tokenized_cache

    Returns:
        Path to the cache directory for this configuration.
    """
    import hashlib
    import json
    import os

    # Load the prepare metadata to get config hash
    metadata_path = os.path.join(prepared_path, "blend_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)
        blend_hash = metadata.get("config_hash", "unknown")
    else:
        # Fallback: hash the prepared path
        blend_hash = hashlib.sha256(prepared_path.encode()).hexdigest()[:16]

    # Create a cache key from model + max_length
    cache_key_str = f"{model_name_or_path}|{max_length}"
    cache_key = hashlib.sha256(cache_key_str.encode()).hexdigest()[:12]

    # Determine cache directory
    if cache_dir is None:
        cache_dir = os.path.join(prepared_path, ".tokenized_cache")

    cache_path = os.path.join(cache_dir, f"{blend_hash}_{cache_key}")
    return cache_path


def load_or_tokenize_dataset(
    prepared_path: str,
    tokenize_fn,
    model_name_or_path: str,
    max_length: int,
    cache_dir: Optional[str] = None,
    num_proc: Optional[int] = None,
    force_retokenize: bool = False,
) -> DatasetDict:
    """
    Load a tokenized dataset from cache, or tokenize and cache it.

    Args:
        prepared_path: Path to the prepared dataset directory.
        tokenize_fn: Function to tokenize examples. Should take a dataset and return a tokenized dataset.
        model_name_or_path: Model identifier (for cache key).
        max_length: Maximum sequence length (for cache key).
        cache_dir: Optional cache directory.
        num_proc: Number of processes for tokenization.
        force_retokenize: If True, ignore cache and re-tokenize.

    Returns:
        Tokenized DatasetDict.
    """
    import os

    cache_path = get_tokenized_cache_path(prepared_path, model_name_or_path, max_length, cache_dir)

    # Check if cached version exists
    if not force_retokenize and os.path.isdir(cache_path):
        train_cache = os.path.join(cache_path, "train.parquet")
        if os.path.exists(train_cache):
            logger.info(f"Loading tokenized dataset from cache: {cache_path}")
            result = {}
            result["train"] = datasets.Dataset.from_parquet(train_cache)
            test_cache = os.path.join(cache_path, "test.parquet")
            if os.path.exists(test_cache):
                result["test"] = datasets.Dataset.from_parquet(test_cache)
            return DatasetDict(result)

    # Load prepared dataset
    logger.info(f"Tokenizing prepared dataset from {prepared_path}")
    dataset_dict = load_prepared_dataset(prepared_path)

    # Tokenize
    tokenized_dict = {}
    for split_name, dataset in dataset_dict.items():
        logger.info(f"Tokenizing {split_name} split ({len(dataset)} examples)")
        tokenized = tokenize_fn(dataset)
        tokenized_dict[split_name] = tokenized

    result = DatasetDict(tokenized_dict)

    # Cache the result
    os.makedirs(cache_path, exist_ok=True)
    for split_name, dataset in result.items():
        split_path = os.path.join(cache_path, f"{split_name}.parquet")
        dataset.to_parquet(split_path)
        logger.info(f"Cached {split_name} split to {split_path}")

    return result
