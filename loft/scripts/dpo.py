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

# Must run BEFORE importing trl: TRL references transformers symbols that
# were renamed / removed in transformers 5.x, and llm_blender (pulled in
# eagerly by TRL's judges module) imports one of them. The shim installs
# aliases if missing.
from loft.patches.trl_compat import patch_trl_imports

patch_trl_imports()

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


def _patch_think_masking(trainer, tokenizer):
    """Patch DPOTrainer to mask think block tokens from the DPO loss.

    When mask_thinking is enabled, tokens between <think> and </think>
    (inclusive) are excluded from the loss computation. This prevents
    the DPO signal from affecting the thinking style — only the actual
    response after </think> gets preference optimization.

    Works by post-processing the loss_mask in concatenated_forward.
    """
    import torch
    import inspect
    import types

    # Get the </think> token ID
    think_end_id = tokenizer.encode("</think>", add_special_tokens=False)[0]

    _orig_forward = trainer.concatenated_forward

    def _masked_forward(model, batch):
        # Call original
        result = _orig_forward(model, batch)

        # result is a tuple: (chosen_logps, rejected_logps, chosen_logits, rejected_logits, ...)
        # We need to modify the log probs — but they're already computed.
        # Instead, we need to hook into the loss_mask before log probs are computed.
        # Since TRL 0.12.0 computes everything inside concatenated_forward,
        # we can't easily intercept. Alternative: mask at the dataset level.
        return result

    # Better approach: mask at tokenization time by modifying completion_attention_mask
    # The DPOTrainer's data collator builds completion_attention_mask from the
    # tokenized sequences. We can add a post-tokenization step that zeros out
    # think tokens in the chosen/rejected sequences.

    _orig_tokenize = trainer.tokenize_row.__func__ if hasattr(trainer.tokenize_row, '__func__') else trainer.tokenize_row

    @staticmethod
    def _tokenize_with_think_mask(features, processing_class, max_prompt_length, max_completion_length, add_special_tokens):
        result = _orig_tokenize(features, processing_class, max_prompt_length, max_completion_length, add_special_tokens)

        # For each of chosen and rejected, find </think> and create attention mask
        for prefix in ("chosen", "rejected"):
            input_ids = result[f"{prefix}_input_ids"]
            # Find the </think> token
            mask = [1] * len(input_ids)
            try:
                end_idx = input_ids.index(think_end_id)
                # Mask everything up to and including </think> and the newlines after
                mask_end = end_idx + 1
                # Also mask trailing newlines after </think>
                while mask_end < len(input_ids) and input_ids[mask_end] in (198, 271):  # \n tokens
                    mask_end += 1
                for i in range(mask_end):
                    mask[i] = 0
            except ValueError:
                pass  # No </think> found, don't mask anything
            result[f"{prefix}_attention_mask"] = mask

        return result

    trainer.tokenize_row = _tokenize_with_think_mask
    logger.info("Patched DPOTrainer.tokenize_row to mask think block tokens from loss")


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


def _patch_triton_rl_kernels(trainer):
    """Replace TRL's entropy and log_softmax with fused Triton kernels.

    Provides ~5x speedup on entropy and ~3x on selective_log_softmax,
    plus significant VRAM savings by never materializing full softmax tensors.

    Adapted from Axolotl v0.16.0 (Apache 2.0).
    """
    try:
        from loft.trainer.triton_rl_kernels import (
            entropy_from_logits,
            selective_log_softmax,
            HAS_TRITON,
        )
        if not HAS_TRITON:
            logger.info("Triton not available, skipping RL kernel patches")
            return False
    except ImportError:
        logger.info("Triton RL kernels not found, skipping")
        return False

    import trl.trainer.dpo_trainer as _dpo_mod

    # Patch selective_log_softmax into TRL if it uses one
    if hasattr(_dpo_mod, 'selective_log_softmax'):
        _dpo_mod.selective_log_softmax = selective_log_softmax
        logger.info("Patched TRL selective_log_softmax with Triton kernel")

    # Patch entropy_from_logits if used
    if hasattr(_dpo_mod, 'entropy_from_logits'):
        _dpo_mod.entropy_from_logits = entropy_from_logits
        logger.info("Patched TRL entropy_from_logits with Triton kernel")

    # Also patch in the trainer's utils module if it exists
    try:
        import trl.trainer.utils as _trl_utils
        if hasattr(_trl_utils, 'selective_log_softmax'):
            _trl_utils.selective_log_softmax = selective_log_softmax
        if hasattr(_trl_utils, 'entropy_from_logits'):
            _trl_utils.entropy_from_logits = entropy_from_logits
    except ImportError:
        pass

    logger.info("Triton RL kernels patched successfully")
    return True


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


def _patch_scmoe_lora_null_ref_context(trainer, model):
    """Wrap DPOTrainer.null_ref_context so the reference forward also disables scmoe_lora.

    PEFT's ``disable_adapter()`` context toggles its own ``lora_`` params, but
    knows nothing about the plain-nn.Parameter scmoe_lora attachments on MoE
    experts. Without this patch the reference forward would still apply the
    expert-LoRA delta → corrupted KL anchor, broken DPO objective.

    Composes both contexts — enters disable_adapter() first (existing
    behavior), then disable_scmoe_lora() inside it.
    """
    from contextlib import contextmanager
    from loft.kernels.scattermoe.custom_lora import disable_scmoe_lora, has_scmoe_lora

    if not has_scmoe_lora(model):
        logger.info("No scmoe_lora attached; skipping null_ref_context patch")
        return False

    _orig_null_ref = trainer.null_ref_context

    @contextmanager
    def _null_ref_with_scmoe_disabled():
        # The original context handles PEFT adapter disable + ref_adapter_name
        # switching; nest our scmoe disable inside so both are active for the
        # reference forward.
        with _orig_null_ref():
            with disable_scmoe_lora(model):
                yield

    trainer.null_ref_context = _null_ref_with_scmoe_disabled
    logger.info(
        "Patched DPOTrainer.null_ref_context to also disable scmoe_lora during "
        "reference forward (composes with PEFT disable_adapter)"
    )
    return True


def _save_scmoe_lora_sidecar(trainer, model, output_dir):
    """Save scmoe_lora tensors alongside the PEFT adapter at training end.

    ``trainer.save_model`` only persists PEFT's adapter_model.safetensors;
    the plain-parameter scmoe_lora attachments are invisible to it.
    """
    from loft.kernels.scattermoe.custom_lora import save_scmoe_lora, has_scmoe_lora

    if not has_scmoe_lora(model):
        return
    import os as _os
    path = _os.path.join(output_dir, "scmoe_lora.safetensors")
    save_scmoe_lora(model, path)
    logger.info(f"Saved scmoe_lora sidecar to {path}")


def _install_scmoe_lora_checkpoint_callback(trainer, model):
    """Install a TrainerCallback that drops a ``scmoe_lora.safetensors`` sidecar
    next to every HF checkpoint (so intermediate + rolling checkpoints are
    complete, not just the final save).
    """
    from transformers import TrainerCallback
    from loft.kernels.scattermoe.custom_lora import save_scmoe_lora, has_scmoe_lora
    import os as _os

    if not has_scmoe_lora(model):
        return

    class _ScmoeLoraCheckpointCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            ckpt = _os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            if not _os.path.isdir(ckpt):
                return
            save_scmoe_lora(model, _os.path.join(ckpt, "scmoe_lora.safetensors"))

    trainer.add_callback(_ScmoeLoraCheckpointCallback())
    logger.info("Installed scmoe_lora checkpoint sidecar callback")


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

    # ScatterMoE — ExpertsInterface models (Gemma4, etc.) must have their
    # implementation registered and `_experts_implementation` set on the
    # config BEFORE from_pretrained, so experts are materialized directly
    # into the scattermoe layout instead of the stock fused 3D tensors.
    # Mirrors the SFT path in loft/scripts/sft.py.
    if getattr(model_args, "use_scattermoe", False):
        from loft.kernels.scattermoe.patch import EXPERTS_INTERFACE_MODELS

        _tc_full = getattr(full_config, "text_config", None)
        _mt = getattr(_tc_full, "model_type", None) or getattr(full_config, "model_type", None)
        if _mt in EXPERTS_INTERFACE_MODELS:
            from loft.kernels.scattermoe.gemma4_experts import register_scattermoe_experts

            register_scattermoe_experts()
            # Set on BOTH the top-level composite config and the text sub-config
            # so inner text layers pick up the scattermoe implementation.
            full_config._experts_implementation = "scattermoe"
            if _tc_full is not None:
                _tc_full._experts_implementation = "scattermoe"
            # For composite VL checkpoints (Gemma4ForConditionalGeneration),
            # the weights are keyed as ``model.language_model.layers.*``. We
            # MUST pass the full composite config so AutoModelForCausalLM
            # materializes the composite architecture whose parameter layout
            # matches — passing text_config alone builds the inner text-only
            # model with ``model.layers.*`` keys and every param ends up
            # reinitialized from meta (see _move_missing_keys_from_meta_to_device
            # OOM observed on Gemma4 26B-A4B-it).
            model_kwargs["config"] = full_config
            config = full_config  # meta-model build + device-map use full config
            logger.info(
                f"ScatterMoE: pre-registered ExpertsInterface for {_mt} "
                f"(config._experts_implementation=scattermoe, using full composite config)"
            )

    import torch

    # --- Fused-scattermoe NF4 cache fast path ---
    # The SFT pipeline creates a pre-quantized fused cache at
    # ``<model_dir>/experts_nf4_packed.safetensors + base.safetensors + quant_states.pkl``
    # that avoids the bf16 on-the-fly bnb path (which doesn't fit Gemma4 26B-A4B
    # on 2x3090). This skips the normal from_pretrained + compute_balanced_device_map
    # path and hands back a model already distributed across GPUs via manual .to() moves.
    _fused_cache_marker = os.path.join(model_args.model_name_or_path, "experts_nf4_packed.safetensors")
    _use_fused_cache = getattr(model_args, "use_scattermoe", False) and os.path.isfile(_fused_cache_marker)
    if _use_fused_cache:
        logger.info(
            f"Detected fused-scattermoe NF4 cache at {model_args.model_name_or_path}; "
            f"delegating to load_fused_quantized_gemma4 (bypasses generic from_pretrained)"
        )
        from loft.kernels.scattermoe.gemma4_loader import load_fused_quantized_gemma4

        _model_dtype_t = torch.bfloat16 if model_dtype in (None, "auto", "bfloat16") else torch.float16
        model, _fused_tokenizer = load_fused_quantized_gemma4(
            model_id=model_args.model_name_or_path,
            dtype=_model_dtype_t,
            register_smoe=True,
            use_cache=True,
            save_cache=False,
        )
        # Match the sdpa / flash_attention_2 user selection (the loader respects
        # GEMMA4_ATTN_IMPL env var; the config attr was already applied at build).
        if model_args.attn_implementation:
            try:
                model.config._attn_implementation = model_args.attn_implementation
                _tc_m = getattr(model.config, "text_config", None)
                if _tc_m is not None:
                    _tc_m._attn_implementation = model_args.attn_implementation
            except Exception:  # pragma: no cover
                pass
        # Build a synthetic hf_device_map reflecting the manual split so the
        # Trainer recognizes model_parallel.
        devices = sorted({str(p.device) for p in model.parameters()})
        model.hf_device_map = {f"layer_{i}": d for i, d in enumerate(devices)}
        # PreTrainedModel normally initializes these via __init__; the fused
        # loader builds from init_empty_weights + from_config + to_empty and
        # skips that path, leaving a few attributes missing. Set the ones TRL
        # / Trainer rely on.
        if not hasattr(model, "warnings_issued"):
            model.warnings_issued = {}

        # Gemma4ForConditionalGeneration.forward → inner text model requires
        # ``mm_token_type_ids`` whenever the model is in training mode (so the
        # vision/text bidirectional mask can be built). For text-only DPO no
        # image tokens are present — inject zeros when the caller omits it.
        # Wrapping the composite forward (not the inner .model) ensures TRL's
        # PEFT+model_parallel path reaches us before the inner training-mode
        # guard fires.
        _orig_gemma4_forward = model.forward

        # The manual split places embed_tokens / lm_head on cuda:0 but there's
        # no accelerate dispatch hook on the top-level model to move incoming
        # tensors from CPU. Trainer feeds the model CPU batches assuming the
        # inner hooks will place them. Move any CPU tensors up-front so embed
        # / attention-mask / labels all land on cuda:0.
        _entry_device = torch.device("cuda:0")

        def _to_entry(v):
            if isinstance(v, torch.Tensor) and v.device != _entry_device:
                return v.to(_entry_device)
            return v

        def _gemma4_text_only_forward(*_f_args, **_f_kwargs):
            _f_args = tuple(_to_entry(a) for a in _f_args)
            _f_kwargs = {k: _to_entry(v) for k, v in _f_kwargs.items()}
            if _f_kwargs.get("mm_token_type_ids") is None and "mm_token_type_ids" not in _f_kwargs:
                _ids = _f_kwargs.get("input_ids")
                if _ids is None and _f_args:
                    _ids = _f_args[0]
                if _ids is not None and hasattr(_ids, "dtype"):
                    _f_kwargs["mm_token_type_ids"] = torch.zeros_like(_ids)
            return _orig_gemma4_forward(*_f_args, **_f_kwargs)

        model.forward = _gemma4_text_only_forward
        logger.info(
            "Installed text-only forward wrapper on Gemma4ForConditionalGeneration "
            "(injects mm_token_type_ids=zeros when missing)"
        )

    elif model_args.model_parallel and torch.cuda.device_count() > 1:
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

    if not _use_fused_cache:
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

    # --- scattermoe-native LoRA for MoE experts (Gemma4 etc) ---
    # Attached AFTER get_peft_model so the base model has the PEFT adapter
    # installed on attn targets. scmoe_lora lives as plain nn.Parameters on
    # experts modules, sidestepping PEFT's target_parameters path (which
    # OOMs on 26B-A4B). The reference forward in DPO disables both PEFT
    # adapters and scmoe_lora via the composed null_ref_context patch
    # installed below.
    _scmoe_lora_enabled = getattr(model_args, "scattermoe_lora", False)
    if _scmoe_lora_enabled:
        from loft.kernels.scattermoe.custom_lora import (
            attach_scattermoe_lora,
            freeze_everything_except_scmoe_lora_and_peft,
        )
        _rank = model_args.scmoe_lora_rank if model_args.scmoe_lora_rank is not None else (model_args.lora_r or 16)
        _alpha = model_args.scmoe_lora_alpha if model_args.scmoe_lora_alpha is not None else float(model_args.lora_alpha or _rank)
        _rslora = model_args.scmoe_lora_use_rslora if model_args.scmoe_lora_use_rslora is not None else bool(model_args.use_rslora)
        _lora_dtype = torch.bfloat16 if (model_dtype in (None, "bfloat16", "auto") and torch.cuda.is_bf16_supported()) else torch.float16
        logger.info(
            f"Attaching scattermoe LoRA: rank={_rank} alpha={_alpha} rslora={_rslora} dtype={_lora_dtype}"
        )
        attach_scattermoe_lora(
            model,
            rank=_rank,
            alpha=_alpha,
            use_rslora=_rslora,
            dtype=_lora_dtype,
        )
        # Re-freeze everything except the two LoRA sets. ``prepare_model_for_kbit_training``
        # froze base params; ``get_peft_model`` unfroze the PEFT adapter;
        # ``attach_scattermoe_lora`` created new trainable params. This call makes
        # the final set explicit.
        n_trainable = freeze_everything_except_scmoe_lora_and_peft(model)
        logger.info(f"scmoe_lora attached; {n_trainable} param tensors trainable")
        # Required for gradient checkpointing when base params are frozen —
        # without this, backward can't propagate through the frozen base
        # back to the new LoRA adapters.
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

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
    # TRL's DPOTrainer auto-detects "vision models" via membership in
    # MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES and switches tokenize_row →
    # process_row, which expects a multimodal Processor with a .tokenizer
    # attribute and an ``images`` column on the dataset. For text-only DPO on
    # Gemma4 (a composite VL model whose top-level ``model_type == "gemma4"``
    # is in that mapping) we need to opt out. Temporarily override the
    # reported model_type with the text sub-config's ``gemma4_text`` across
    # the DPOTrainer.__init__ so the tokenizer path is picked. Restore after.
    _orig_model_type = getattr(model.config, "model_type", None)
    _tc_model_type = getattr(getattr(model.config, "text_config", None), "model_type", None)
    _model_type_overridden = False
    try:
        from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES

        if _orig_model_type in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES and _tc_model_type:
            model.config.model_type = _tc_model_type
            _model_type_overridden = True
            logger.info(
                f"Temporarily overriding model.config.model_type "
                f"{_orig_model_type!r} → {_tc_model_type!r} for DPOTrainer init "
                f"(text-only DPO path on composite VL model)"
            )
    except Exception:  # pragma: no cover
        pass

    try:
        trainer = DPOTrainer(
            model=model,
            ref_model=None,  # With PEFT, base model (adapters disabled) is the reference
            args=training_args,
            train_dataset=dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
        )
    finally:
        if _model_type_overridden:
            model.config.model_type = _orig_model_type

    # Apply Triton RL kernels for faster entropy/log_softmax
    _patch_triton_rl_kernels(trainer)

    # scmoe_lora: disable during reference forward (composes with PEFT
    # disable_adapter) + drop a sidecar at every checkpoint save.
    if _scmoe_lora_enabled:
        _patch_scmoe_lora_null_ref_context(trainer, model)
        _install_scmoe_lora_checkpoint_callback(trainer, model)

    # Mask think block tokens from DPO loss if requested
    _mask_thinking = getattr(training_args, "mask_thinking", False) or getattr(model_args, "mask_thinking", False)
    if _mask_thinking:
        _patch_think_masking(trainer, tokenizer)

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

    # Resume: load scmoe_lora sidecar from checkpoint (HF Trainer handles the
    # PEFT adapter + optimizer + scheduler + rng, but doesn't know about our
    # sidecar). Call BEFORE trainer.train() so the restored scmoe_lora state
    # participates in both policy and (disabled) reference forwards from step 1.
    _resume_from = getattr(training_args, "resume_from_checkpoint", None)
    if _scmoe_lora_enabled and _resume_from:
        from loft.kernels.scattermoe.custom_lora import load_scmoe_lora
        _sidecar = os.path.join(_resume_from, "scmoe_lora.safetensors")
        if os.path.exists(_sidecar):
            n = load_scmoe_lora(model, _sidecar)
            logger.info(f"Resume: loaded {n} scmoe_lora tensors from {_sidecar}")
        else:
            logger.warning(
                f"Resume: no scmoe_lora sidecar at {_sidecar} — experts will "
                f"resume from init, which is wrong for any checkpoint past step 0!"
            )

    trainer.train(resume_from_checkpoint=_resume_from)

    # Save the final model
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # scmoe_lora final sidecar (HF Trainer.save_model only persists the
    # PEFT adapter; the plain-parameter scmoe_lora is invisible to it).
    if _scmoe_lora_enabled:
        _save_scmoe_lora_sidecar(trainer, model, training_args.output_dir)


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
