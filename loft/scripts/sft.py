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

# /// script
# dependencies = [
#     "trl",
#     "peft",
#     "trackio",
#     "kernels",
# ]
# ///

"""
# Full training
```
python trl/scripts/sft.py \
    --model_name_or_path Qwen/Qwen2-0.5B \
    --dataset_name trl-lib/Capybara \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --eos_token '<|im_end|>' \
    --eval_strategy steps \
    --eval_steps 100 \
    --output_dir Qwen2-0.5B-SFT \
    --push_to_hub
```

# LoRA
```
python trl/scripts/sft.py \
    --model_name_or_path Qwen/Qwen2-0.5B \
    --dataset_name trl-lib/Capybara \
    --learning_rate 2.0e-4 \
    --num_train_epochs 1 \
    --packing \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --eos_token '<|im_end|>' \
    --eval_strategy steps \
    --eval_steps 100 \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16 \
    --output_dir Qwen2-0.5B-SFT \
    --push_to_hub
```
"""

import argparse
import os
import sys
from typing import Optional

from accelerate import logging
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES

from loft import (
    DatasetMixtureConfig,
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_dataset,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from loft.import_utils import is_cce_available
from loft.scripts.utils import (
    build_prepare_config,
    load_prepared_dataset,
    get_tokenized_cache_path,
    is_pretokenized_prepare,
    needs_prepare,
    prompt_prepare_overwrite,
    run_auto_prepare,
    validate_prepare_metadata,
)


logger = logging.get_logger(__name__)

# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


def main(script_args, training_args, model_args, dataset_args):
    # Reduce CUDA memory fragmentation — critical for model_parallel where
    # multiple large tensors (embeddings, optimizer states) compete for space.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    ################
    # Model init kwargs
    ################
    # Infer model dtype from training config if not explicitly set
    # This ensures Flash Attention compatibility when bf16/fp16 training is enabled
    model_dtype = model_args.dtype
    if model_dtype is None:
        import torch
        if training_args.bf16:
            # Use bfloat16 if bf16 training is enabled and hardware supports it
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                model_dtype = "bfloat16"
            else:
                # Fallback to float16 if bf16 not supported
                model_dtype = "float16"
            logger.info(f"Inferred model dtype from bf16 training config: {model_dtype}")
        elif training_args.fp16:
            model_dtype = "float16"
            logger.info(f"Inferred model dtype from fp16 training config: {model_dtype}")
        # Otherwise leave as None (model default)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        dtype=model_dtype,
        low_cpu_mem_usage=model_args.low_cpu_mem_usage,
    )
    quantization_config = get_quantization_config(model_args)

    # Load model config early so we can build device maps from it
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )

    if quantization_config is not None:
        # Passing None would not be treated the same as omitting the argument, so we include it only when valid.
        model_kwargs["device_map"] = get_kbit_device_map(model_parallel=model_args.model_parallel)
        model_kwargs["quantization_config"] = quantization_config
        import torch
        if model_args.model_parallel and torch.cuda.device_count() > 1:
            n_gpus = torch.cuda.device_count()
            n_layers = config.num_hidden_layers
            last = n_gpus - 1
            # Pin special modules: embeddings on GPU 0, output head + norm
            # on the last GPU so the large logits tensor doesn't compete
            # with embeddings for memory.  Hidden layers are distributed
            # evenly across all GPUs.
            device_map = {"model.embed_tokens": 0, "lm_head": last,
                          "model.norm": last, "model.rotary_emb": last}
            layers_per_gpu = n_layers // n_gpus
            extra = n_layers % n_gpus
            layer_idx = 0
            for gpu in range(n_gpus):
                # Distribute remainder layers one each to the first `extra` GPUs
                count = layers_per_gpu + (1 if gpu < extra else 0)
                for _ in range(count):
                    device_map[f"model.layers.{layer_idx}"] = gpu
                    layer_idx += 1
            model_kwargs["device_map"] = device_map
    valid_image_text_architectures = MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.values()

    if config.architectures and any(arch in valid_image_text_architectures for arch in config.architectures):
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    # When using model_parallel with device_map, accelerate's dispatch_model
    # installs hooks that move the model's output back to the input device
    # (GPU 0).  For large-vocab models this means the logits tensor (~2 GiB
    # at batch=2) plus its fp32 gradient (~4 GiB) end up on GPU 0, wasting
    # memory that the lm_head device (last GPU) has plenty of.  Disabling
    # io_same_device keeps logits on the lm_head's GPU so backward doesn't
    # OOM on GPU 0.
    if model_args.model_parallel and hasattr(model, "_hf_hook"):
        model._hf_hook.io_same_device = False
        logger.info("Disabled io_same_device on model dispatch hook — logits stay on lm_head device")

    # Apply CCE (Cut Cross-Entropy) patching for memory-efficient loss computation
    if training_args.use_cce:
        if not is_cce_available():
            raise ImportError(
                "CCE (Cut Cross-Entropy) is not available. Please install it with: pip install cut-cross-entropy"
            )
        try:
            from cut_cross_entropy.transformers import cce_patch
        except ImportError as e:
            raise ImportError(
                f"CCE import failed due to version incompatibility: {e}\n"
                "This typically occurs when cut-cross-entropy is incompatible with your transformers version.\n"
                "Try: pip install --upgrade cut-cross-entropy transformers"
            ) from e

        model = cce_patch(model)
        logger.info("Applied CCE (Cut Cross-Entropy) patch for memory-efficient loss computation.")

        # When using model_parallel (device_map="auto"), the lm_head weight and hidden states
        # can end up on different GPUs. CCE's Triton kernel requires all tensors on the same
        # device, so we monkey-patch apply_lce to align devices before the kernel launch.
        if model_args.model_parallel:
            import cut_cross_entropy.transformers.utils as _cce_utils

            _original_apply_lce = _cce_utils.apply_lce

            def _device_aligned_apply_lce(e, c, labels, opts, bias=None, softcap=None, **loss_kwargs):
                import torch
                target_device = e.device
                if c.device != target_device:
                    c = c.to(target_device)
                if bias is not None and bias.device != target_device:
                    bias = bias.to(target_device)
                # Triton launches kernels on the current CUDA device, so we must
                # switch context to where the tensors live, then restore afterwards.
                prev_device = torch.cuda.current_device()
                if target_device.type == "cuda":
                    torch.cuda.set_device(target_device)
                try:
                    loss = _original_apply_lce(e, c, labels, opts, bias=bias, softcap=softcap, **loss_kwargs)
                finally:
                    torch.cuda.set_device(prev_device)
                # Move loss back to the original device so the trainer's loss accounting works
                return loss.to(f"cuda:{prev_device}")

            # Patch the canonical reference and all model-specific modules that imported it
            _cce_utils.apply_lce = _device_aligned_apply_lce
            import sys
            for name, mod in sys.modules.items():
                if name.startswith("cut_cross_entropy.transformers.") and hasattr(mod, "apply_lce"):
                    mod.apply_lce = _device_aligned_apply_lce
            logger.info("Patched CCE apply_lce for model_parallel device alignment.")

    # Patch CAME optimizer for model_parallel device alignment:
    # Triton kernels launch on the current CUDA device, so we must switch
    # context to match each parameter's device before running the step.
    # We patch step_param (not step) because PyTorch's LR scheduler wraps
    # optimizer.step at the instance level, bypassing class-level patches.
    if model_args.model_parallel:
        try:
            import torch as _torch_patch
            from came_pytorch import CAME as _came_cls

            _original_step_param = _came_cls.step_param

            @_torch_patch.inference_mode()
            def _device_aligned_step_param(self, p, group):
                if p.grad is None:
                    return
                if p.device.type == "cuda":
                    prev = _torch_patch.cuda.current_device()
                    _torch_patch.cuda.set_device(p.device)
                    try:
                        _original_step_param(self, p, group)
                    finally:
                        _torch_patch.cuda.set_device(prev)
                else:
                    _original_step_param(self, p, group)

            _came_cls.step_param = _device_aligned_step_param
            logger.info("Patched CAME optimizer step_param for model_parallel device alignment.")
        except (ImportError, AttributeError):
            pass  # CAME not installed or API changed

    # Load the dataset
    if training_args.prepared_dataset:
        _prepared = training_args.prepared_dataset
        _has_data_config = bool(training_args.data_config)

        # New-style: data_config is set, so we can auto-prepare if needed
        if _has_data_config:
            if needs_prepare(_prepared):
                # Path is missing/empty — auto-run prepare
                logger.info(f"Prepared dataset not found at {_prepared} — running prepare automatically.")
                run_auto_prepare(training_args, model_args)
            else:
                # Path exists — validate metadata matches current config
                prepare_config = build_prepare_config(training_args, model_args)
                mismatches = validate_prepare_metadata(_prepared, prepare_config)
                if mismatches:
                    # Prompt user to overwrite (rank 0 only, exits on 'n')
                    prompt_prepare_overwrite(_prepared, mismatches, force=training_args.force_prepare)
                    # User said yes (or force_prepare=True) — re-prepare
                    run_auto_prepare(training_args, model_args)

        # Load the prepared dataset (now guaranteed to exist if data_config was set)
        logger.info(f"Loading prepared dataset from {_prepared}")
        dataset = load_prepared_dataset(_prepared)
        logger.info(
            f"Loaded prepared dataset: {len(dataset['train'])} train"
            + (f", {len(dataset['test'])} test" if 'test' in dataset else "")
        )
        # If this is a new-style pre-tokenized prepare, tell the trainer to skip truncation
        if is_pretokenized_prepare(_prepared):
            training_args._pretokenized = True
            logger.info("Detected pre-tokenized prepared dataset — skipping tokenization and truncation in trainer.")
    elif training_args.data_config and not training_args.prepared_dataset:
        # data_config is set but no prepared_dataset path — error with helpful message
        raise ValueError(
            "Training config has 'data_config' but no 'prepared_dataset' path. "
            "Set 'prepared_dataset' to a directory where the prepared data should be stored. "
            "It will be created automatically if it doesn't exist."
        )
    elif dataset_args.datasets and script_args.dataset_name:
        logger.warning(
            "Both `datasets` and `dataset_name` are provided. The `datasets` argument will be used to load the "
            "dataset and `dataset_name` will be ignored."
        )
        dataset = get_dataset(dataset_args)
    elif dataset_args.datasets and not script_args.dataset_name:
        dataset = get_dataset(dataset_args)
    elif not dataset_args.datasets and script_args.dataset_name:
        dataset = load_dataset(
            script_args.dataset_name, name=script_args.dataset_config, streaming=script_args.dataset_streaming
        )
    else:
        raise ValueError("Either `prepared_dataset`, `datasets`, or `dataset_name` must be provided.")

    # Determine if we should pass eval dataset to trainer
    # Check for: explicit eval_strategy, OR evals_per_epoch (which will set eval_strategy in trainer init)
    has_eval_split = script_args.dataset_test_split in dataset
    wants_eval = (
        training_args.eval_strategy != "no"
        or (hasattr(training_args, "evals_per_epoch") and training_args.evals_per_epoch is not None and training_args.evals_per_epoch > 0)
    )
    eval_dataset = dataset[script_args.dataset_test_split] if has_eval_split and wants_eval else None

    # Initialize the SFT trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args),
    )

    # Train the model
    trainer.train()

    trainer.save_model(training_args.output_dir)
    logger.info(f"Training completed. Model saved to {training_args.output_dir}.")

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        logger.info(f"Model pushed to the Hub: https://huggingface.co/{trainer.hub_model_id}")


def make_parser(subparsers: Optional[argparse._SubParsersAction] = None, include_dataset_args: bool = True):
    """
    Create the argument parser for SFT training.

    Args:
        subparsers: Optional subparsers action for CLI integration.
        include_dataset_args: If True, include DatasetMixtureConfig for inline dataset definitions.
                             If False, only parse ScriptArguments, SFTConfig, and ModelConfig.
                             When using a pre-tokenized prepared_dataset, DatasetMixtureConfig
                             is not needed since all preprocessing was done during prepare.
    """
    if include_dataset_args:
        dataclass_types = (ScriptArguments, SFTConfig, ModelConfig, DatasetMixtureConfig)
    else:
        dataclass_types = (ScriptArguments, SFTConfig, ModelConfig)

    if subparsers is not None:
        parser = subparsers.add_parser("sft", help="Run the SFT training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser


def _check_needs_dataset_args(config_path: Optional[str]) -> bool:
    """
    Quick check to see if the config uses prepared_dataset with data_config.
    If so, we don't need DatasetMixtureConfig (preprocessing already done).
    """
    if not config_path:
        return True  # No config file, need dataset args for CLI

    import yaml
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
        # If prepared_dataset and data_config are set, preprocessing is done by prepare
        # We don't need DatasetMixtureConfig fields
        has_prepared = bool(config.get("prepared_dataset"))
        has_data_config = bool(config.get("data_config"))
        return not (has_prepared and has_data_config)
    except Exception:
        return True  # On error, default to including dataset args


if __name__ == "__main__":
    # First, check if we're using a config file with prepared_dataset
    # If so, we skip DatasetMixtureConfig to avoid field conflicts
    # Normalize positional config: "sft.py config.yaml" → "sft.py --config config.yaml"
    if "--config" not in sys.argv and len(sys.argv) > 1:
        first = sys.argv[1]
        if not first.startswith("-") and (first.endswith(".yaml") or first.endswith(".yml")):
            sys.argv.insert(1, "--config")

    config_path = None
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--config" and i < len(sys.argv):
            config_path = sys.argv[i + 1]
            break
        elif arg.startswith("--config="):
            config_path = arg.split("=", 1)[1]
            break

    needs_dataset_args = _check_needs_dataset_args(config_path)
    parser = make_parser(include_dataset_args=needs_dataset_args)

    # When using the trl cli, this script may be run with additional arguments, corresponding accelerate arguments.
    # To ensure that their parsing does not interfere with the script arguments, parse the arguments with
    # `return_remaining_strings=True`, then ignore the remaining strings.
    if needs_dataset_args:
        script_args, training_args, model_args, dataset_args, _ = parser.parse_args_and_config(
            return_remaining_strings=True
        )
    else:
        # No DatasetMixtureConfig needed - create empty one
        script_args, training_args, model_args, _ = parser.parse_args_and_config(
            return_remaining_strings=True
        )
        dataset_args = DatasetMixtureConfig()

    main(script_args, training_args, model_args, dataset_args)
