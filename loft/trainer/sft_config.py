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

from dataclasses import dataclass, field
from typing import Any, Optional

from transformers import TrainingArguments


@dataclass
class SFTConfig(TrainingArguments):
    r"""
    Configuration class for the [`SFTTrainer`].

    This class includes only the parameters that are specific to SFT training. For a full list of training arguments,
    please refer to the [`~transformers.TrainingArguments`] documentation. Note that default values in this class may
    differ from those in [`~transformers.TrainingArguments`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        > Parameters that control the model

        model_init_kwargs (`dict[str, Any]`, *optional*):
            Keyword arguments for [`~transformers.AutoModelForCausalLM.from_pretrained`], used when the `model`
            argument of the [`SFTTrainer`] is provided as a string. If you're training a MoE architecture and want to
            include the load balancing/auxiliary loss as part of the final loss, set `output_router_logits=True` in
            this dictionary. The loss coefficient is read from the model config's `router_aux_loss_coef` field.
        chat_template_path (`str`, *optional*):
            If specified, sets the model's chat template. This can either be the path to a tokenizer (local directory
            or Hugging Face Hub model) or a direct path to a Jinja template file. When using a Jinja file, you must
            ensure that any special tokens referenced in the template are added to the tokenizer and that the model's
            embedding layer is resized accordingly.

        > Parameters that control the data preprocessing

        dataset_text_field (`str`, *optional*, defaults to `"text"`):
            Name of the column that contains text data in the dataset.
        dataset_kwargs (`dict[str, Any]`, *optional*):
            Dictionary of optional keyword arguments for the dataset preparation. The only supported key is
            `skip_prepare_dataset`. When the model is a VLM, `skip_prepare_dataset` is automatically treated as `True`
            regardless of the provided value, since preprocessing is done on the fly.
        dataset_num_proc (`int`, *optional*):
            Number of processes to use for processing the dataset.
        eos_token (`str`, *optional*):
            Token used to indicate the end of a turn or sequence. If `None`, it defaults to
            `processing_class.eos_token`.
        pad_token (`str`, *optional*):
            Token used for padding. If `None`, it defaults to `processing_class.pad_token`, or if that is also `None`,
            it falls back to `processing_class.eos_token`.
        max_length (`int` or `None`, *optional*, defaults to `1024`):
            Maximum length of the tokenized sequence. Sequences longer than `max_length` are truncated from the right.
            If `None`, no truncation is applied. When packing is enabled, this value sets the sequence length.
        truncation_strategy (`str`, *optional*, defaults to `"truncate"`):
            How to handle samples exceeding max_length:
            - `"truncate"`: Cut off at max_length, no EOS on truncated samples (default)
            - `"drop"`: Filter out samples exceeding max_length
            - `"split"`: Split into multiple samples (text/CPT data only). First chunk gets BOS, last gets EOS.
            - `"truncate_turns"`: For conversational data, drop complete turn pairs from start while preserving
              system message. If even one turn exceeds max_length, the sample is dropped.
            Per-dataset `truncation_strategy` in the dataset mixer overrides this global default.
        packing (`bool`, *optional*, defaults to `False`):
            Whether to group multiple sequences into fixed-length blocks to improve computational efficiency and reduce
            padding. Uses `max_length` to define sequence length.
        packing_strategy (`str`, *optional*, defaults to `"bfd"`):
            Strategy for packing sequences. Can be either `"bfd"` (best-fit decreasing, default), or `"wrapped"`.
        padding_free (`bool`, *optional*, defaults to `False`):
            Whether to perform forward passes without padding by flattening all sequences in the batch into a single
            continuous sequence. This reduces memory usage by eliminating padding overhead. Currently, this is only
            supported with the FlashAttention 2 or 3, which can efficiently handle the flattened batch structure. When
            packing is enabled with strategy `"bfd"`, padding-free is enabled, regardless of the value of this
            parameter.
        pad_to_multiple_of (`int`, *optional*):
            If set, the sequences will be padded to a multiple of this value.
        eval_packing (`bool`, *optional*):
            Whether to pack the eval dataset. If `None`, uses the same value as `packing`.

        > Parameters that control the training

        completion_only_loss (`bool`, *optional*):
            Whether to compute loss only on the completion part of the sequence. If set to `True`, loss is computed
            only on the completion, which is supported only for [prompt-completion](#prompt-completion) datasets. If
            `False`, loss is computed on the entire sequence. If `None` (default), the behavior depends on the dataset:
            loss is computed on the completion for [prompt-completion](#prompt-completion) datasets, and on the full
            sequence for [language modeling](#language-modeling) datasets.
        assistant_only_loss (`bool`, *optional*, defaults to `False`):
            Whether to compute loss only on the assistant part of the sequence. If set to `True`, loss is computed only
            on the assistant responses, which is supported only for [conversational](#conversational) datasets. If
            `False`, loss is computed on the entire sequence.
        last_assistant_only_loss (`bool`, *optional*, defaults to `False`):
            Whether to compute loss only on the LAST assistant turn in multi-turn conversations. When `True`, only the
            final assistant response contributes to loss; intermediate assistant turns are masked. Implies
            `assistant_only_loss=True`. Useful for teaching specific final behaviors without affecting intermediate
            dialogue patterns.
        train_on_incomplete_assistant (`bool`, *optional*, defaults to `False`):
            Whether to train on incomplete/truncated assistant responses without adding EOS token. When `True`, if the
            last message is from the assistant, the EOS token is not appended, treating it as a continuation rather
            than a complete response. This prevents the model from learning to stop mid-thought on truncated data.
        fix_turn_order (`bool`, *optional*, defaults to `False`):
            Whether to fix conversation turn order for models with strict requirements (e.g., Llama). When `True`:
            (1) adds a filler user message if conversation starts with assistant, (2) merges consecutive messages
            from the same role, (3) drops trailing user messages so conversations end with assistant.
        fix_turn_order_filler (`str`, *optional*, defaults to `"Let's begin."`):
            The filler message to insert when `fix_turn_order=True` and the conversation starts with an assistant turn.
        default_system_message (`str`, *optional*):
            Default system message to add to conversations that don't have one. When using the dataset mixer,
            per-dataset `system_message` overrides this global default. If `None`, no system message is added.
        loss_type (`str`, *optional*, defaults to `"nll"`):
            Type of loss to use. Possible values are `"nll"` (negative log-likelihood, default) and `"dft"` (Dynamic
            Fine-Tuning, as described in [this paper](https://huggingface.co/papers/2508.05629)).
        activation_offloading (`bool`, *optional*, defaults to `False`):
            Whether to offload the activations to the CPU.
    """

    _VALID_DICT_FIELDS = TrainingArguments._VALID_DICT_FIELDS + ["model_init_kwargs"]

    # Parameters whose default values are overridden from TrainingArguments
    learning_rate: float = field(
        default=2e-5,
        metadata={"help": "The initial learning rate for AdamW."},
    )
    logging_steps: float = field(
        default=10,
        metadata={
            "help": "Log every X updates steps. Should be an integer or a float in range `[0,1)`. If smaller than 1, "
            "will be interpreted as ratio of total training steps."
        },
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={
            "help": "If True, use gradient checkpointing to save memory at the expense of slower backward pass."
        },
    )
    bf16: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA "
            "architecture or Intel XPU or using CPU (use_cpu) or Ascend NPU. If not set, it defaults to `True` if "
            "`fp16` is not set."
        },
    )

    # Parameters that control the model
    model_init_kwargs: Optional[dict[str, Any]] = field(
        default=None,
        metadata={
            "help": "Keyword arguments for `AutoModelForCausalLM.from_pretrained`, used when the `model` argument of "
            "the `SFTTrainer` is provided as a string. If you're training a MoE architecture and want to include the "
            "load balancing/auxilliary loss as a part of the final loss, remember to set `output_router_logits=True` "
            "in this dictionary."
        },
    )
    chat_template_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "If specified, sets the model's chat template. This can either be the path to a tokenizer (local "
            "directory or Hugging Face Hub model) or a direct path to a Jinja template file. When using a Jinja file, "
            "you must ensure that any special tokens referenced in the template are added to the tokenizer and "
            "that the model's embedding layer is resized accordingly."
        },
    )

    # Parameters that control the data preprocessing
    dataset_text_field: str = field(
        default="text",
        metadata={"help": "Name of the column that contains text data in the dataset."},
    )
    dataset_kwargs: Optional[dict[str, Any]] = field(
        default=None,
        metadata={
            "help": "Dictionary of optional keyword arguments for the dataset preparation. The only supported key is "
            "`skip_prepare_dataset`. If the model is a VLM, `skip_prepare_dataset` value is ignored. When the model "
            "is a VLM, `skip_prepare_dataset` is automatically treated as `True` regardless of the provided value, "
            "since preprocessing is done on the fly."
        },
    )
    dataset_num_proc: Optional[int] = field(
        default=None,
        metadata={"help": "Number of processes to use for processing the dataset."},
    )
    eos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "Token used to indicate the end of a turn or sequence. If `None`, it defaults to `processing_class.eos_token`."
        },
    )
    pad_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "Token used for padding. If `None`, it defaults to `processing_class.pad_token`, or if that "
            "is also `None`, it falls back to `processing_class.eos_token`."
        },
    )
    max_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "Maximum length of the tokenized sequence. Sequences longer than `max_length` are truncated from"
            "the right. If `None`, no truncation is applied. When packing is enabled, this value sets the "
            "sequence length."
        },
    )
    truncation_strategy: str = field(
        default="truncate",
        metadata={
            "help": (
                "How to handle samples exceeding max_length. Options: "
                "'truncate' (cut off, no EOS on truncated), "
                "'drop' (filter out), "
                "'split' (chunk text data, first gets BOS, last gets EOS), "
                "'truncate_turns' (conversational: drop turn pairs from start, keep system message)."
            ),
            "choices": ["truncate", "drop", "split", "truncate_turns"],
        },
    )
    packing: bool = field(
        default=False,
        metadata={
            "help": "Whether to group multiple sequences into fixed-length blocks to improve computational efficiency "
            "and reduce padding. Uses `max_length` to define sequence length."
        },
    )
    packing_strategy: str = field(
        default="bfd",
        metadata={
            "help": "Strategy for packing sequences. Can be either `'bfd'` (best-fit decreasing, default), or "
            "`'wrapped'`."
        },
    )
    padding_free: bool = field(
        default=False,
        metadata={
            "help": "Whether to perform forward passes without padding by flattening all sequences in the batch into "
            "a single continuous sequence. This reduces memory usage by eliminating padding overhead. Currently, this "
            "is only supported with the FlashAttention 2 or 3, which can efficiently handle the flattened batch "
            "structure. When packing is enabled with strategy `'bfd'`, padding-free is enabled, regardless of the "
            "value of this parameter."
        },
    )
    pad_to_multiple_of: Optional[int] = field(
        default=None,
        metadata={"help": "If set, the sequences will be padded to a multiple of this value."},
    )
    eval_packing: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether to pack the eval dataset. If `None`, uses the same value as `packing`."},
    )

    # Parameters that control the training
    completion_only_loss: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to compute loss only on the completion part of the sequence. If set to `True`, loss is "
                "computed only on the completion, which is supported only for prompt-completion datasets. If `False`, "
                "loss is computed on the entire sequence. If `None` (default), the behavior depends on the dataset: "
                "loss is computed on the completion for prompt-completion datasets, and on the full sequence for "
                "language modeling datasets."
            )
        },
    )
    assistant_only_loss: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to compute loss only on the assistant part of the sequence. If set to `True`, loss is "
                "computed only on the assistant responses, which is supported only for conversational datasets. If `False`, "
                "loss is computed on the entire sequence."
            )
        },
    )
    last_assistant_only_loss: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to compute loss only on the LAST assistant turn in multi-turn conversations. "
                "When `True`, only the final assistant response contributes to loss; intermediate assistant turns are masked. "
                "Implies `assistant_only_loss=True`. Useful for teaching specific final behaviors without affecting "
                "intermediate dialogue patterns."
            )
        },
    )
    train_on_incomplete_assistant: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to train on incomplete/truncated assistant responses without adding EOS token. "
                "When `True`, if the last message is from the assistant and appears truncated (no proper ending), "
                "the EOS token is not appended, treating it as a continuation rather than a complete response. "
                "This prevents the model from learning to stop mid-thought on truncated training data."
            )
        },
    )
    fix_turn_order: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to fix conversation turn order for models with strict requirements. "
                "When `True`: (1) adds a filler user message if conversation starts with assistant, "
                "(2) merges consecutive messages from the same role, "
                "(3) drops trailing user messages so conversations end with assistant. "
                "Use `fix_turn_order_filler` to customize the filler message."
            )
        },
    )
    fix_turn_order_filler: str = field(
        default="Let's begin.",
        metadata={
            "help": (
                "The filler message to insert when `fix_turn_order=True` and the conversation "
                "starts with an assistant turn (no initial user message). Default: 'Let's begin.'"
            )
        },
    )
    default_system_message: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Default system message to add to conversations that don't have one. "
                "When using the dataset mixer, per-dataset `system_message` overrides this global default. "
                "If `None`, no system message is added."
            )
        },
    )
    loss_type: str = field(
        default="nll",
        metadata={
            "help": (
                'Type of loss to use. Possible values are `"nll"` (negative log-likelihood, default) and `"dft"` '
                "(Dynamic Fine-Tuning, as described in https://huggingface.co/papers/2508.05629)."
            )
        },
    )
    label_smoothing: float = field(
        default=0.0,
        metadata={
            "help": (
                "Label smoothing factor for the cross-entropy loss. When > 0, redistributes this fraction of "
                "probability mass from the target token uniformly across all tokens, acting as a confidence "
                "penalty that prevents overconfident predictions. Applied directly via nn.functional.cross_entropy. "
                "Only used with `loss_type='nll'`. Typical values: 0.05-0.1."
            )
        },
    )
    # Auxiliary loss weights (0.0 = disabled, added on top of the primary loss_type)
    aux_loss_eos_weight: float = field(
        default=0.0,
        metadata={
            "help": (
                "Weight for the EOS calibration auxiliary loss. When > 0, adds an extra cross-entropy term on "
                "EOS token positions at the end of assistant turns, encouraging the model to assign higher "
                "probability to EOS where it should stop generating. Requires `assistant_only_loss=True` or "
                "`completion_only_loss=True` so that turn boundaries are known. Typical values: 0.05-0.2."
            )
        },
    )
    aux_loss_rep_weight: float = field(
        default=0.0,
        metadata={
            "help": (
                "Weight for the repetition penalty auxiliary loss. When > 0, penalizes the model for assigning "
                "high probability to tokens that already appeared recently in the same sequence. Uses a sliding "
                "window controlled by `aux_loss_rep_window`. Typical values: 0.01-0.1."
            )
        },
    )
    aux_loss_rep_window: int = field(
        default=64,
        metadata={
            "help": (
                "Window size (in tokens) for the repetition penalty auxiliary loss. The penalty applies to tokens "
                "that appeared within the last N positions. Only used when `aux_loss_rep_weight > 0`."
            )
        },
    )
    aux_loss_rep_ngram: int = field(
        default=1,
        metadata={
            "help": (
                "N-gram size for the repetition penalty auxiliary loss. With n=1 (default), penalizes repeating "
                "any individual token that appeared recently. With n=2, only penalizes repeating bigrams (pairs "
                "of consecutive tokens). With n=3, only penalizes repeating trigrams, etc. Higher values are more "
                "targeted but also more expensive (O(window * n) per position). Only used when "
                "`aux_loss_rep_weight > 0`."
            )
        },
    )
    aux_loss_top_prob_weight: float = field(
        default=0.0,
        metadata={
            "help": (
                "Weight for the top-probability confidence penalty auxiliary loss. When > 0, directly penalizes "
                "the model's peak (max) probability at each position. A sharper alternative to entropy-based "
                "confidence regularization — targets only the single highest probability token rather than the "
                "full distribution shape. Typical values: 0.01-0.1."
            )
        },
    )
    token_weights: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to a YAML or JSON file mapping tokens to importance weights for the cross-entropy loss. "
                "Each key is a token string (e.g. '<eos>', ' Not') which will be tokenized, and each value is "
                "a float weight (default 1.0). Tokens with weight > 1.0 get stronger gradient signal where they "
                "appear as the target; tokens with weight < 1.0 get weaker signal. Useful for upweighting "
                "control tokens like EOS or downweighting overused constructs. Example YAML:\n"
                "  '<eos>': 10.0\n"
                "  ' Not': 0.5\n"
            )
        },
    )
    activation_offloading: bool = field(
        default=False,
        metadata={"help": "Whether to offload the activations to the CPU."},
    )
    use_cce: bool = field(
        default=False,
        metadata={
            "help": "Whether to use Cut Cross-Entropy (CCE) for memory-efficient cross-entropy loss computation. "
            "Requires: pip install cut-cross-entropy"
        },
    )

    # Data config and prepared dataset support
    data_config: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to a data config YAML file containing the dataset list and shuffle/subset settings. "
            "Used by `loft prepare` to load datasets. The data config is model-agnostic and reusable across "
            "different models. When specified alongside `prepared_dataset`, `loft prepare` uses this config "
            "to build the dataset and saves to `prepared_dataset`."
        },
    )
    prepared_dataset: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to a prepared dataset directory (created by `loft prepare`). "
            "When specified, the dataset is loaded from this path instead of being loaded/processed at runtime. "
            "If the prepared dataset is pre-tokenized (from new-style prepare), it is used directly. "
            "If untokenized (from old-style prepare), it will be tokenized for the model and cached."
        },
    )
    tokenized_cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Directory to cache tokenized datasets. If not specified, caches are stored in "
            "{prepared_dataset}/.tokenized_cache. Only used when `prepared_dataset` is specified."
        },
    )
    force_retokenize: bool = field(
        default=False,
        metadata={
            "help": "If True, ignore cached tokenized datasets and re-tokenize from the prepared dataset."
        },
    )
    force_prepare: bool = field(
        default=False,
        metadata={
            "help": "If True, auto-overwrite mismatched prepared datasets without prompting. "
            "Useful for non-interactive environments (CI, automated runs, SLURM jobs) where "
            "stdin is not a TTY and the interactive prompt would otherwise abort."
        },
    )

    # Convenience fields for per-epoch save/eval scheduling
    saves_per_epoch: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of times to save per epoch. If set, automatically calculates save_steps based on "
            "total training steps. Mutually exclusive with save_steps when > 0. For example, saves_per_epoch=2 "
            "will save at the middle and end of each epoch."
        },
    )
    evals_per_epoch: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of times to evaluate per epoch. If set, automatically calculates eval_steps based on "
            "total training steps. Mutually exclusive with eval_steps when > 0. For example, evals_per_epoch=4 "
            "will evaluate 4 times per epoch."
        },
    )

    # Eval split settings (can also be specified in data_config; training config takes precedence)
    eval_split: Optional[float] = field(
        default=None,
        metadata={
            "help": "Fraction of the dataset to use for evaluation (0.0 to 1.0). Applied after tokenization "
            "during prepare. If specified here, overrides the value in data_config. Set to 0 to disable eval split. "
            "Example: eval_split=0.05 reserves 5% of samples for evaluation."
        },
    )
    split_seed: Optional[int] = field(
        default=None,
        metadata={
            "help": "Random seed for the train/eval split. If specified here, overrides the value in data_config. "
            "Use a fixed seed for reproducible splits across runs. Default (when not set anywhere): 42."
        },
    )

    # W&B project name (handled by CLI, converted to WANDB_PROJECT env var)
    wandb_project: Optional[str] = field(
        default=None,
        metadata={
            "help": "Weights & Biases project name. This field is extracted by the CLI and set as the "
            "WANDB_PROJECT environment variable before training starts. It is not passed as a CLI argument."
        },
    )

    # Custom optimizer names that are handled by SFTTrainer but aren't built into Transformers
    CUSTOM_OPTIMIZERS = {"came_pytorch"}

    def __post_init__(self):
        self.bf16 = not (self.fp16) if self.bf16 is None else self.bf16

        # Validate truncation_strategy
        valid_strategies = {"truncate", "drop", "split", "truncate_turns"}
        if self.truncation_strategy not in valid_strategies:
            raise ValueError(
                f"Invalid truncation_strategy: {self.truncation_strategy}. "
                f"Must be one of: {', '.join(sorted(valid_strategies))}"
            )

        # Handle custom optimizer names before parent validation
        # Store the original value and replace with a valid placeholder
        self._custom_optim = None
        if self.optim is not None and self.optim in self.CUSTOM_OPTIMIZERS:
            self._custom_optim = self.optim
            self.optim = "adamw_torch"  # Placeholder to pass validation

        super().__post_init__()

        # Restore the custom optimizer name after validation
        if self._custom_optim is not None:
            self.optim = self._custom_optim
