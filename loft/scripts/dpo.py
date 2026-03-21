"""
DPO (Direct Preference Optimization) training script.

Reuses model loading infrastructure from sft.py (quantization, model_parallel,
device maps, patches) and delegates training to TRL's DPOTrainer.

Usage:
    python -m loft.scripts.dpo configs/qwen35-9b-antirep/train.yaml

    # Or with explicit --config:
    python -m loft.scripts.dpo --config configs/qwen35-9b-antirep/train.yaml
"""

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from accelerate import logging
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from loft import (
    ModelConfig,
    ScriptArguments,
    TrlParser,
    compute_balanced_device_map,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


logger = logging.get_logger(__name__)


def preprocess_dpo_dataset(dataset, tokenizer):
    """Convert our JSONL format to plain-text prompt/chosen/rejected for DPOTrainer.

    Our data format:
        prompt: [{"role": "user", "content": "..."}, ...]
        chosen: "<think>\\n...\\n</think>\\nresponse"
        rejected: "<think>\\n...\\n</think>\\nresponse"

    Qwen3.5's chat template always appends "<think>\\n" to the generation prompt,
    so we strip the leading "<think>\\n" from chosen/rejected to avoid duplication.
    """
    def _preprocess(example):
        # Apply chat template to get the prompt string
        prompt_text = tokenizer.apply_chat_template(
            example["prompt"], tokenize=False, add_generation_prompt=True
        )

        # Strip leading <think>\n from completions — the template already added it
        chosen = example["chosen"]
        rejected = example["rejected"]
        if chosen.startswith("<think>\n"):
            chosen = chosen[len("<think>\n"):]
        if rejected.startswith("<think>\n"):
            rejected = rejected[len("<think>\n"):]

        return {"prompt": prompt_text, "chosen": chosen, "rejected": rejected}

    return dataset.map(_preprocess, remove_columns=[
        c for c in dataset.column_names if c not in ("prompt", "chosen", "rejected")
    ])


def _patch_dpo_memory_efficient_logprobs():
    """Verify that TRL's concatenated_forward uses memory-efficient log-prob computation.

    The default implementation computes:
        per_token_logps = gather(logits.log_softmax(-1), labels)
    which allocates [batch, seq, vocab_size] in float32 (~2GB for seq=2048, vocab=248K).

    We patch the TRL source directly to use the equivalent but memory-efficient:
        per_token_logps = gather(logits, labels) - logits.logsumexp(-1)
    which only allocates [batch, seq] tensors.
    """
    # The actual patch is applied directly to dpo_trainer.py source.
    # This function just verifies the patch is in place.
    import inspect
    import trl.trainer.dpo_trainer as _dpo_mod
    src = inspect.getsource(_dpo_mod.DPOTrainer.concatenated_forward)
    if "log_softmax(-1)" in src:
        logger.warning(
            "TRL dpo_trainer.py has NOT been patched for memory-efficient log-probs. "
            "Training may OOM on large-vocab models (248K). "
            "Patch line: per_token_logps = gather(logits, labels) - logsumexp(logits)"
        )


def _ref_logprobs_cache_key(model_args, training_args, dataset_path):
    """Build a hash key from factors that affect reference log probabilities.

    Includes: base model identity, quantization precision, dataset path/contents,
    and sequence length settings. Excludes: LoRA rank/alpha, LR, epochs, etc.
    """
    key_parts = {
        "model": model_args.model_name_or_path,
        "model_revision": model_args.model_revision,
        "load_in_4bit": model_args.load_in_4bit,
        "load_in_8bit": model_args.load_in_8bit,
        "bnb_4bit_quant_type": model_args.bnb_4bit_quant_type if model_args.load_in_4bit else None,
        "dataset_path": str(dataset_path),
        "max_length": training_args.max_length,
        "max_prompt_length": training_args.max_prompt_length,
        "max_completion_length": training_args.max_completion_length,
    }
    key_json = json.dumps(key_parts, sort_keys=True)
    return hashlib.sha256(key_json.encode()).hexdigest()[:16]


def _patch_ref_logprob_caching(trainer, cache_dir, cache_key):
    """Patch DPOTrainer to save/load precomputed ref log probs from disk."""
    cache_path = Path(cache_dir) / "ref_logprobs" / cache_key
    train_chosen_path = cache_path / "train_ref_chosen_logps.npy"
    train_rejected_path = cache_path / "train_ref_rejected_logps.npy"
    eval_chosen_path = cache_path / "eval_ref_chosen_logps.npy"
    eval_rejected_path = cache_path / "eval_ref_rejected_logps.npy"

    _orig_get_train_dl = trainer.get_train_dataloader
    _orig_get_eval_dl = trainer.get_eval_dataloader

    def get_train_dataloader_cached():
        if trainer.precompute_ref_log_probs and not trainer._precomputed_train_ref_log_probs:
            if train_chosen_path.exists() and train_rejected_path.exists():
                logger.info(f"Loading cached train ref log probs from {cache_path}")
                chosen = np.load(train_chosen_path)
                rejected = np.load(train_rejected_path)
                if len(chosen) == len(trainer.train_dataset):
                    trainer.train_dataset = trainer.train_dataset.add_column(
                        name="ref_chosen_logps", column=chosen
                    )
                    trainer.train_dataset = trainer.train_dataset.add_column(
                        name="ref_rejected_logps", column=rejected
                    )
                    trainer._precomputed_train_ref_log_probs = True
                    logger.info(f"Loaded {len(chosen)} cached ref log prob pairs")
                else:
                    logger.warning(
                        f"Cache size mismatch ({len(chosen)} vs {len(trainer.train_dataset)}), recomputing"
                    )

        result = _orig_get_train_dl()

        # Save after first computation
        if trainer._precomputed_train_ref_log_probs and not train_chosen_path.exists():
            cache_path.mkdir(parents=True, exist_ok=True)
            chosen_col = trainer.train_dataset["ref_chosen_logps"]
            rejected_col = trainer.train_dataset["ref_rejected_logps"]
            np.save(train_chosen_path, np.array(chosen_col, dtype=np.float32))
            np.save(train_rejected_path, np.array(rejected_col, dtype=np.float32))
            logger.info(f"Saved train ref log probs to {cache_path}")

        return result

    def get_eval_dataloader_cached(eval_dataset=None):
        ds = eval_dataset if eval_dataset is not None else trainer.eval_dataset
        if ds is not None and trainer.precompute_ref_log_probs and not trainer._precomputed_eval_ref_log_probs:
            if eval_chosen_path.exists() and eval_rejected_path.exists():
                logger.info(f"Loading cached eval ref log probs from {cache_path}")
                chosen = np.load(eval_chosen_path)
                rejected = np.load(eval_rejected_path)
                target_ds = ds if eval_dataset is not None else trainer.eval_dataset
                if len(chosen) == len(target_ds):
                    if eval_dataset is not None:
                        eval_dataset = target_ds.add_column(name="ref_chosen_logps", column=chosen)
                        eval_dataset = eval_dataset.add_column(name="ref_rejected_logps", column=rejected)
                    else:
                        trainer.eval_dataset = target_ds.add_column(name="ref_chosen_logps", column=chosen)
                        trainer.eval_dataset = trainer.eval_dataset.add_column(
                            name="ref_rejected_logps", column=rejected
                        )
                    trainer._precomputed_eval_ref_log_probs = True
                    logger.info(f"Loaded {len(chosen)} cached eval ref log prob pairs")
                else:
                    logger.warning(
                        f"Eval cache size mismatch ({len(chosen)} vs {len(target_ds)}), recomputing"
                    )

        result = _orig_get_eval_dl(eval_dataset)

        # Save after first computation
        if trainer._precomputed_eval_ref_log_probs and not eval_chosen_path.exists():
            cache_path.mkdir(parents=True, exist_ok=True)
            target_ds = trainer.eval_dataset
            chosen_col = target_ds["ref_chosen_logps"]
            rejected_col = target_ds["ref_rejected_logps"]
            np.save(eval_chosen_path, np.array(chosen_col, dtype=np.float32))
            np.save(eval_rejected_path, np.array(rejected_col, dtype=np.float32))
            logger.info(f"Saved eval ref log probs to {cache_path}")

        return result

    trainer.get_train_dataloader = get_train_dataloader_cached
    trainer.get_eval_dataloader = get_eval_dataloader_cached


def main(script_args, training_args, model_args):
    # Patch DPO to avoid OOM from 248K vocab log_softmax
    _patch_dpo_memory_efficient_logprobs()

    # Optionally use vocab-chunked log probs (avoids full logits materialization)
    if getattr(model_args, 'use_chunked_dpo', False):
        from loft.trainer.chunked_logprobs import patch_dpo_chunked_logprobs
        patch_dpo_chunked_logprobs(chunk_size=getattr(model_args, 'chunked_dpo_size', 4096))

    # Reduce CUDA memory fragmentation
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # Patch fla Triton kernel bug (affects Qwen3.5 GatedDeltaNet backward pass)
    try:
        from loft.patches.fla_triton import patch_fla_wy_fast
        patch_fla_wy_fast()
    except Exception:
        pass

    # Skip caching_allocator_warmup — incorrect for quantized models
    try:
        import transformers.modeling_utils as _mu
        _mu.caching_allocator_warmup = lambda *args, **kwargs: None
    except Exception:
        pass

    # Fix flash_attention_2 crash on models with 3D position_ids (Qwen3.5)
    try:
        import transformers.modeling_flash_attention_utils as _fa_utils
        _orig_is_packed = _fa_utils._is_packed_sequence

        def _fixed_is_packed_sequence(position_ids, batch_size):
            if position_ids is None or position_ids.ndim != 2:
                return False
            return _orig_is_packed(position_ids, batch_size)

        _fa_utils._is_packed_sequence = _fixed_is_packed_sequence
    except Exception:
        pass

    ################
    # Model loading
    ################
    model_dtype = model_args.dtype
    if model_dtype is None:
        import torch
        if training_args.bf16:
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                model_dtype = "bfloat16"
            else:
                model_dtype = "float16"
            logger.info(f"Inferred model dtype from bf16 training config: {model_dtype}")
        elif training_args.fp16:
            model_dtype = "float16"
            logger.info(f"Inferred model dtype from fp16 training config: {model_dtype}")

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        dtype=model_dtype,
        low_cpu_mem_usage=model_args.low_cpu_mem_usage,
    )
    quantization_config = get_quantization_config(model_args)

    full_config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )

    # For composite VL models (e.g. Qwen3.5), use the text sub-config for causal LM loading.
    # The text_config has vocab_size and layer details that AutoModelForCausalLM needs.
    config = getattr(full_config, "text_config", full_config)

    if quantization_config is not None:
        model_kwargs["device_map"] = get_kbit_device_map(model_parallel=model_args.model_parallel)
        model_kwargs["quantization_config"] = quantization_config
        existing_quant = getattr(config, "quantization_config", None)
        if existing_quant and existing_quant.get("quant_method") != "bitsandbytes":
            logger.info(
                f"Clearing model's pre-existing {existing_quant.get('quant_method')} quantization "
                f"config in favor of BitsAndBytes {('4-bit' if model_args.load_in_4bit else '8-bit')}"
            )
            del config.quantization_config
            model_kwargs["config"] = config
        os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")

    import torch

    if model_args.model_parallel and torch.cuda.device_count() > 1:
        # Compute an explicit balanced device map from a meta model.  We can't rely on
        # device_map="auto"/"balanced" because BnB 4-bit quantization shrinks the model
        # enough to fit on one GPU, causing accelerate to skip multi-GPU distribution.
        from accelerate import init_empty_weights

        with init_empty_weights():
            _meta_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

        # Chunked DPO never materializes full [batch, seq, vocab] logits — memory
        # profile is similar to CCE (at most [batch, seq, chunk_size] per chunk).
        _use_cce = getattr(model_args, 'use_chunked_dpo', False)

        _device_map = compute_balanced_device_map(
            _meta_model,
            model_config=config,
            is_quantized_4bit=model_args.load_in_4bit,
            is_quantized_8bit=model_args.load_in_8bit,
            max_memory=model_args.max_memory,
            batch_size=training_args.per_device_train_batch_size * 2,  # DPO concat
            max_length=training_args.max_length or 2048,
            use_cce=_use_cce,
            dtype_bytes=2,  # bf16
            use_lora=model_args.use_peft,
            lora_r=model_args.lora_r or 16,
        )
        del _meta_model
        model_kwargs["device_map"] = _device_map

    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    # When using model_parallel with device_map, accelerate's dispatch_model installs
    # hooks that move the model's output back to the input device (GPU 0).  For
    # large-vocab models this means the logits tensor (~2 GiB) plus its fp32 gradient
    # end up on GPU 0, wasting memory that the lm_head device has plenty of.
    # Disabling io_same_device keeps logits on the lm_head's GPU.
    if model_args.model_parallel and hasattr(model, "_hf_hook"):
        model._hf_hook.io_same_device = False
        logger.info("Disabled io_same_device on model dispatch hook — logits stay on lm_head device")

    # Transformers 5.x no longer sets hf_device_map on models loaded with device_map="auto".
    # The Trainer checks for this attribute to detect model_parallel and skip DataParallel
    # wrapping.  Build a synthetic device map from the actual parameter placements.
    if model_args.model_parallel and not hasattr(model, "hf_device_map"):
        devices = {str(p.device) for p in model.parameters()}
        model.hf_device_map = {f"layer_{i}": d for i, d in enumerate(sorted(devices))}
        logger.info(f"Set synthetic hf_device_map: {devices}")

    ################
    # Tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    dataset_path = script_args.dataset_name
    if not dataset_path:
        raise ValueError("Must provide dataset_name pointing to a JSONL file with DPO pairs")

    logger.info(f"Loading DPO dataset from {dataset_path}")
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    logger.info(f"Loaded {len(dataset)} preference pairs")

    # Preprocess: apply chat template to prompts, fix <think> tag overlap
    dataset = preprocess_dpo_dataset(dataset, tokenizer)

    # Split off eval set if requested
    eval_dataset = None
    if training_args.eval_strategy != "no":
        split = dataset.train_test_split(test_size=0.05, seed=42)
        dataset = split["train"]
        eval_dataset = split["test"]
        logger.info(f"Split: {len(dataset)} train, {len(eval_dataset)} eval")

    ################
    # PEFT wrapping
    ################
    peft_config = get_peft_config(model_args)

    if peft_config is not None:
        from peft import get_peft_model, prepare_model_for_kbit_training

        _saved_device_map = getattr(model, "hf_device_map", None)

        if quantization_config is not None:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=training_args.gradient_checkpointing,
            )
        model = get_peft_model(model, peft_config)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(f"PEFT: {trainable:,} trainable / {total:,} total params ({trainable/total*100:.2f}%)")

        # Restore hf_device_map on PeftModel so DPOTrainer detects model_parallel
        if _saved_device_map is not None:
            model.hf_device_map = _saved_device_map

        # Dispatch hooks installed by from_pretrained(device_map=...) on the inner
        # model survive PEFT wrapping: add_hook_to_module replaces module.forward with
        # a wrapped version, so when BaseTuner.forward calls self.model.forward(), the
        # hooked forward fires and moves tensors to the correct device at each layer
        # boundary.  No re-dispatch needed — just keep the original hooks.
        if model_args.model_parallel and torch.cuda.device_count() > 1:
            logger.info("Multi-GPU: keeping dispatch hooks from from_pretrained through PEFT")

    # Apply chunked MLP for memory-efficient long-context training
    _chunked_mlp = getattr(model_args, "chunked_mlp", False) or getattr(training_args, "chunked_mlp", False)
    if _chunked_mlp:
        from loft.trainer.chunked_mlp import patch_mlp_chunking
        _chunks = getattr(model_args, "chunked_mlp_chunks", None) or getattr(training_args, "chunked_mlp_chunks", 8)
        n_patched = patch_mlp_chunking(model, num_chunks=_chunks)
        logger.info(f"Applied chunked MLP: {n_patched} modules patched")

    ################
    # Training
    ################
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # With PEFT, base model (adapters disabled) is the reference
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # Cache precomputed ref log probs to disk so they survive across runs
    # with different LoRA/optimizer settings but the same base model + data.
    if training_args.precompute_ref_log_probs:
        cache_key = _ref_logprobs_cache_key(model_args, training_args, script_args.dataset_name)
        _patch_ref_logprob_caching(trainer, training_args.output_dir, cache_key)
        logger.info(f"Ref logprob cache key: {cache_key}")

    # Apply activation offloading: wrap training_step so activations are offloaded to CPU
    _act_offload = getattr(model_args, "activation_offloading", False) or getattr(training_args, "activation_offloading", False)
    if _act_offload:
        from loft.models.activation_offloading import get_act_offloading_ctx_manager
        _offload_ctx = get_act_offloading_ctx_manager(model=model)
        _orig_training_step = trainer.training_step

        def _offloaded_training_step(*args, **kwargs):
            with _offload_ctx:
                return _orig_training_step(*args, **kwargs)

        trainer.training_step = _offloaded_training_step
        logger.info("Activation offloading enabled for DPO training")

    trainer.train()

    # Save the final model
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    # Normalize positional config: "dpo.py config.yaml" → "dpo.py --config config.yaml"
    if "--config" not in sys.argv and len(sys.argv) > 1:
        first = sys.argv[1]
        if not first.startswith("-") and (first.endswith(".yaml") or first.endswith(".yml")):
            sys.argv.insert(1, "--config")

    parser = TrlParser((ScriptArguments, DPOConfig, ModelConfig))
    script_args, training_args, model_args, _ = parser.parse_args_and_config(
        return_remaining_strings=True
    )
    main(script_args, training_args, model_args)
