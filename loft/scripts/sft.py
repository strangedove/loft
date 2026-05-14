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
    compute_balanced_device_map,
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
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # Patch fla Triton kernel bug (affects Qwen3.5 GatedDeltaNet backward pass)
    try:
        from loft.patches.fla_triton import patch_fla_wy_fast
        patch_fla_wy_fast()
    except Exception:
        pass  # Non-fatal — only needed for models using flash-linear-attention

    # Skip caching_allocator_warmup — it incorrectly estimates peak memory for quantized
    # models (counts full-precision size), causing OOM on memory-constrained GPUs.
    # This is only a CUDA allocator warm-up optimization, not required for correctness.
    try:
        import transformers.modeling_utils as _mu
        _mu.caching_allocator_warmup = lambda *args, **kwargs: None
    except Exception:
        pass

    # Fix flash_attention_2 crash on models with 3D position_ids (e.g. Qwen3.5 rope3d).
    # Transformers' _is_packed_sequence assumes 2D position_ids [batch, seq_len] but
    # Qwen3.5 passes 3D [3, batch, seq_len] for its 3D rotary embeddings.  The 3D shape
    # causes false-positive packed sequence detection, sending data through the varlen
    # path with incorrect cu_seqlens, resulting in illegal memory access.
    try:
        import transformers.modeling_flash_attention_utils as _fa_utils
        _orig_is_packed = _fa_utils._is_packed_sequence

        def _fixed_is_packed_sequence(position_ids, batch_size):
            if position_ids is None or position_ids.ndim != 2:
                return False
            return _orig_is_packed(position_ids, batch_size)

        _fa_utils._is_packed_sequence = _fixed_is_packed_sequence
    except Exception:
        pass  # Non-fatal — only needed for flash_attention_2

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
        # If the model already has a different quantization method (e.g. FP8) but the user
        # wants BitsAndBytes, clear the pre-existing config so they don't conflict.
        existing_quant = getattr(config, "quantization_config", None)
        if existing_quant and existing_quant.get("quant_method") != "bitsandbytes":
            logger.info(
                f"Clearing model's pre-existing {existing_quant.get('quant_method')} quantization "
                f"config in favor of BitsAndBytes {('4-bit' if model_args.load_in_4bit else '8-bit')}"
            )
            del config.quantization_config
            model_kwargs["config"] = config
        # Workaround for transformers 5.x OOM during quantized model loading:
        # async weight loading materializes multiple tensors to GPU at full precision
        # concurrently before quantization, causing OOM for models that don't fit in
        # full precision.  Sequential loading processes one tensor at a time:
        # load → quantize → free → next.
        os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")

    import torch
    valid_image_text_architectures = MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.values()

    if model_args.model_parallel and torch.cuda.device_count() > 1:
        # Compute a quantization-aware device map ourselves instead of relying on
        # from_pretrained's device_map="auto".  The latter uses bf16 module sizes even
        # for 4-bit models, causing heavily unbalanced distributions (e.g. 14 vs 50
        # layers for Qwen3.5-27B).  By passing dtype=int8 to infer_auto_device_map,
        # we get sizes that match actual 4-bit storage and a balanced split.
        from accelerate import init_empty_weights

        is_vl = config.architectures and any(a in valid_image_text_architectures for a in config.architectures)
        with init_empty_weights():
            if is_vl:
                from transformers import AutoModelForImageTextToText
                meta_model = AutoModelForImageTextToText.from_config(config, trust_remote_code=model_args.trust_remote_code)
            else:
                meta_model = AutoModelForCausalLM.from_config(config, trust_remote_code=model_args.trust_remote_code)

        device_map = compute_balanced_device_map(
            meta_model,
            model_config=config,
            is_quantized_4bit=model_args.load_in_4bit,
            is_quantized_8bit=model_args.load_in_8bit,
            max_memory=model_args.max_memory,
            batch_size=training_args.per_device_train_batch_size,
            max_length=training_args.max_length or 2048,
            use_cce=getattr(training_args, "use_cce", False),
            use_lora=model_args.use_peft,
            lora_r=model_args.lora_r,
        )
        del meta_model

        if device_map is not None:
            model_kwargs["device_map"] = device_map
        else:
            # Fallback to auto if compute_balanced_device_map returned None (single GPU)
            model_kwargs["device_map"] = "auto"

    # ScatterMoE — ExpertsInterface models (Gemma4, etc.) must have their
    # implementation registered and `_experts_implementation` set on the config
    # BEFORE from_pretrained, so the experts are materialized directly into the
    # scattermoe layout instead of the stock fused 3D tensors. The latter OOMs
    # at bf16 during weight load even with 4-bit quant requested, because the
    # dequant hook fires after materialize_copy.
    if training_args.use_scattermoe:
        from loft.kernels.scattermoe.patch import EXPERTS_INTERFACE_MODELS

        # Resolve the LM model_type (VLMs expose it via text_config).
        _mt = getattr(getattr(config, "text_config", None), "model_type", None) \
            or getattr(config, "model_type", None)
        if _mt in EXPERTS_INTERFACE_MODELS:
            from loft.kernels.scattermoe.gemma4_experts import register_scattermoe_experts

            register_scattermoe_experts()
            config._experts_implementation = "scattermoe"
            _tc = getattr(config, "text_config", None)
            if _tc is not None:
                _tc._experts_implementation = "scattermoe"
            # Ensure the modified config is used by from_pretrained.
            model_kwargs["config"] = config
            logger.info(
                f"ScatterMoE: pre-registered ExpertsInterface for {_mt} "
                f"(config._experts_implementation=scattermoe)"
            )

    _vl_arch = config.architectures and any(arch in valid_image_text_architectures for arch in config.architectures)

    # Gemma 4 has two head_dims: sliding layers use head_dim=256 (FA2-OK),
    # global "full_attention" layers use head_dim=512 (FA2 max=256 → rejected).
    # When FA2 is requested, force SDPA on the global layers only so the bulk
    # of layers (sliding) still get FA2's memory savings + sliding_window.
    if (model_args.attn_implementation == "flash_attention_2"
            and getattr(config, "model_type", "") == "gemma4"):
        from transformers.models.gemma4 import modeling_gemma4 as _g4mm
        _orig_g4_attn_fwd = _g4mm.Gemma4TextAttention.forward

        from torch.nn.attention import sdpa_kernel, SDPBackend
        def _attn_fwd_with_sdpa_for_global(self, *args, **kwargs):
            if getattr(self, "head_dim", 0) > 256:
                _saved = self.config._attn_implementation
                self.config._attn_implementation = "sdpa"
                try:
                    # Prefer memory-efficient + cudnn backends (both are
                    # O(L) memory). Math backend allocates O(L²) score
                    # matrix — 4.24 GiB at 8k × head_dim=512 — and OOMs.
                    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                      SDPBackend.CUDNN_ATTENTION,
                                      SDPBackend.MATH]):
                        return _orig_g4_attn_fwd(self, *args, **kwargs)
                finally:
                    self.config._attn_implementation = _saved
            return _orig_g4_attn_fwd(self, *args, **kwargs)

        _g4mm.Gemma4TextAttention.forward = _attn_fwd_with_sdpa_for_global
        logger.info(
            "Patched Gemma4TextAttention.forward: global layers (head_dim=512) "
            "fall back to SDPA; sliding layers (head_dim=256) keep FA2."
        )

    # flex_attention path: HF's `flex_attention_mask` unconditionally adds a
    # padding_mask sub-mask doing `padding_mask[batch_idx, kv_idx]`. That fancy
    # `aten.index.Tensor` op cannot be lowered inside a flex_attention mask
    # subgraph on torch 2.6 (SubgraphLoweringException: "Buffers cannot be
    # created while lowering a pointwise subgraph"), and on multi-GPU model_parallel
    # the padding_mask tensor lives on a different device than the per-layer attn
    # compute. With per_device_train_batch_size=1 there is no padding within a
    # batch, so we can drop the padding_mask entirely when attention_mask is all-1s.
    if model_args.attn_implementation == "flex_attention":
        from transformers import masking_utils as _mu
        _orig_flex_mask = _mu.flex_attention_mask
        def _flex_mask_skip_padding_if_unneeded(*args, **kwargs):
            am = kwargs.get("attention_mask", None)
            if am is None and len(args) >= 7:
                am = args[6]
            if am is not None:
                try:
                    if bool(am.all()):
                        if "attention_mask" in kwargs:
                            kwargs["attention_mask"] = None
                        else:
                            args = list(args); args[6] = None; args = tuple(args)
                except Exception:
                    pass
            return _orig_flex_mask(*args, **kwargs)
        _mu.flex_attention_mask = _flex_mask_skip_padding_if_unneeded
        # Re-register through the AttentionMaskInterface global mapping too so
        # any code that reads the dispatch dict picks up the patched function.
        try:
            _mu.AttentionMaskInterface._global_mapping["flex_attention"] = (
                _flex_mask_skip_padding_if_unneeded
            )
        except Exception:
            pass
        logger.info(
            "Patched flex_attention_mask: skip padding_mask submask when "
            "attention_mask is all-1s (avoids torch 2.6 SubgraphLoweringException "
            "on aten.index.Tensor and cross-device closures under model_parallel)."
        )

        # On RTX 3090 (SM 8.6, ~100KB shared memory per SM), torch 2.6's default
        # flex_attention configs blow past SMEM (head_dim<=256: BLOCK_M=128,
        # BLOCK_N=64, num_stages=3 -> ~196KB). Plus torch 2.6 has a skip for
        # num_stages==2 (pytorch issue #129625), leaving num_stages=1 or 3.
        # Patch _get_default_config_{fwd,bwd} to return SMEM-safe configs with
        # num_stages=1 for sm_86. SMEM ~ num_stages * BLOCK_N * head_dim * 2bytes * 2
        # head_dim=256, BLOCK_N=64, num_stages=1 -> ~75KB
        # head_dim=512, BLOCK_N=32, num_stages=1 -> ~75KB
        # Also: num_stages/num_warps are passed both as kwargs AND splatted from
        # kernel_options in inductor's autotune loop, so they MUST NOT be in
        # the user-supplied kernel_options dict — only BLOCK_M/BLOCK_N (which
        # use setdefault and so override the autotuner default if set).
        try:
            cap = torch.cuda.get_device_capability(0)
        except Exception:
            cap = (0, 0)
        if cap == (8, 6):  # RTX 3090 / 3080 / A40 etc — limited SMEM
            from torch._inductor.kernel import flex_attention as _t_flex
            def _safe_fwd_config(query):
                head_dim = query.get_size()[-1]
                if query.get_dtype() == torch.float32:
                    return (16, 16, 4, 1)
                if head_dim <= 256:
                    return (64, 64, 4, 1)
                return (32, 32, 4, 1)  # head_dim > 256 (Gemma4 global)
            def _safe_bwd_config(query):
                head_dim = query.get_size()[-1]
                if query.get_dtype() == torch.float32:
                    return (16, 16, 4, 1)
                if head_dim <= 256:
                    return (32, 32, 4, 1)
                return (16, 16, 4, 1)
            _t_flex._get_default_config_fwd = _safe_fwd_config
            _t_flex._get_default_config_bwd = _safe_bwd_config
            logger.info(
                "Patched torch._inductor flex_attention default configs for SM 8.6 "
                "(RTX 3090): fwd (64,64,4,1) head_dim<=256 / (32,32,4,1) head_dim>256; "
                "bwd (32,32,4,1) head_dim<=256 / (16,16,4,1) head_dim>256. "
                "num_stages=1 fits in 100KB SMEM."
            )

            # HF transformers/integrations/flex_attention.py force-compiles flex
            # with mode="max-autotune-no-cudagraphs" on torch 2.6.0 + training
            # (line 86-89, workaround for pytorch#146260). That adds 5 alternate
            # configs per layer, all of which exceed our SMEM and add tons of
            # wasted compile time. Override WrappedFlexAttention to use the
            # plain compile path; with our num_stages=1 default config we
            # don't need the workaround.
            from transformers.integrations import flex_attention as _hf_flex_int
            from torch.nn.attention.flex_attention import flex_attention as _raw_flex_attn
            # Bump dynamo cache_size_limit. Gemma 4 has 2 KV-head variants
            # (sliding: kv=16, global: kv=4), so even with dynamic shapes flex
            # recompiles per variant; with sliding_window vs global mask shapes
            # and fwd/bwd, we easily exceed the default cache=8 limit. After
            # cache exhaustion dynamo falls back to eager math_attention which
            # materializes O(L²) scores -> OOM at our seq lengths.
            # 2026-04-30: bumped from 256 to allow longer single-shape-per-sample
            # runs without cache exhaustion. With dynamic=False each unique seq_len
            # × {sliding,global} × {fwd,bwd} consumes 4 entries; 1000 unique seqs
            # = 4000 entries. Default of 256 fills around step 32 on 250-sample
            # runs and falls back to math_attention -> OOM.
            _flex_cache_limit = int(os.environ.get("LOFT_FLEX_CACHE_LIMIT", "8192"))
            torch._dynamo.config.cache_size_limit = _flex_cache_limit
            torch._dynamo.config.accumulated_cache_size_limit = max(_flex_cache_limit * 4, 1024)
            # 2026-04-30: env-var override for dynamic= flag. dynamic=True collapses
            # all seq_len variants into a single compile (huge win), but the
            # 2026-04-29 mixed marvin+instruct data triggered a sympy guard.
            # Pure-marvin runs don't have that mix, so dynamic=True is safe and
            # is the proper fix to cache exhaustion. Set LOFT_FLEX_DYNAMIC=1 to
            # opt in; default kept at False to preserve mixed-data behavior.
            _flex_dynamic_env = os.environ.get("LOFT_FLEX_DYNAMIC", "").strip().lower()
            _flex_dynamic = _flex_dynamic_env in ("1", "true", "yes", "on")
            _orig_init = _hf_flex_int.WrappedFlexAttention.__init__
            @torch.compiler.disable(recursive=False)
            def _patched_init(self, training):
                if not getattr(self, "_is_flex_compiled", False) or training != getattr(self, "training", None):
                    self.training = training
                    self._compiled_flex_attention = torch.compile(_raw_flex_attn, dynamic=_flex_dynamic)
                    self._is_flex_compiled = True
            _hf_flex_int.WrappedFlexAttention.__init__ = _patched_init
            _hf_flex_int.WrappedFlexAttention._instance = None
            _hf_flex_int.WrappedFlexAttention._is_flex_compiled = False
            logger.info(
                "Patched HF WrappedFlexAttention: skip torch 2.6 max-autotune "
                "workaround, use dynamic=%s. Bumped dynamo cache_size_limit=%d.",
                _flex_dynamic, _flex_cache_limit,
            )

    if _vl_arch:
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    # Text-only training on a multimodal-config model (e.g. Gemma 4 31B-it):
    # bypass the multimodal wrapper's forward so we never enter the image_mask
    # / audio_mask code path. Under model_parallel that path triggers a
    # device-mismatch (input_ids on cuda:1 while embed_tokens lives on cuda:0)
    # since `self.get_input_embeddings()(llm_input_ids)` runs ahead of the
    # accelerate hook on the embedding module. Free vision/audio submodules
    # too — they're loaded on GPU but we'll never use them.
    if _vl_arch and hasattr(model, "model") and hasattr(model.model, "language_model"):
        import types
        from dataclasses import dataclass
        from typing import Optional, Tuple
        from transformers.cache_utils import Cache

        _inner = model.model
        _lang = _inner.language_model

        # Free unused multimodal submodules.
        for _attr in ("vision_tower", "audio_tower", "embed_vision", "embed_audio", "multi_modal_projector"):
            if hasattr(_inner, _attr) and getattr(_inner, _attr) is not None:
                setattr(_inner, _attr, None)
        import torch as _torch
        _torch.cuda.empty_cache()

        # Wrapper that exposes the multimodal model's expected output shape so
        # CCE's gemma4 patch (which reads outputs.image_hidden_states /
        # audio_hidden_states / past_key_values / hidden_states / attentions)
        # doesn't AttributeError on a vanilla BaseModelOutputWithPast.
        class _TextOnlyMMOutput:
            __slots__ = ("last_hidden_state", "past_key_values", "hidden_states",
                         "attentions", "image_hidden_states", "audio_hidden_states")

            def __init__(self, base):
                self.last_hidden_state = base.last_hidden_state
                self.past_key_values = getattr(base, "past_key_values", None)
                self.hidden_states = getattr(base, "hidden_states", None)
                self.attentions = getattr(base, "attentions", None)
                self.image_hidden_states = None
                self.audio_hidden_states = None

        _DROP_KWARGS = {
            "pixel_values", "pixel_values_videos", "input_features",
            "input_features_mask", "image_position_ids", "video_position_ids",
            "mm_token_type_ids", "labels",
        }

        def _text_only_inner_forward(self, input_ids=None, attention_mask=None,
                                      position_ids=None, past_key_values=None,
                                      inputs_embeds=None, use_cache=None, **kwargs):
            kwargs = {k: v for k, v in kwargs.items() if k not in _DROP_KWARGS}
            # Force input_ids onto the actual embed_tokens.weight device.
            # accelerate's per-module hook moves args to the device the
            # device_map thinks the module is on, but for VL composite
            # configs the weight can end up elsewhere — verify against the
            # storage tensor itself, not the hook's belief.
            _embed_dev = _lang.embed_tokens.weight.device
            if input_ids is not None and input_ids.device != _embed_dev:
                input_ids = input_ids.to(_embed_dev)
            if attention_mask is not None and attention_mask.device != _embed_dev:
                attention_mask = attention_mask.to(_embed_dev)
            if position_ids is not None and position_ids.device != _embed_dev:
                position_ids = position_ids.to(_embed_dev)
            base = _lang(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                **kwargs,
            )
            return _TextOnlyMMOutput(base)

        model.model.forward = types.MethodType(_text_only_inner_forward, model.model)
        logger.info(
            "Bypassed multimodal Gemma4Model.forward — routing directly to "
            "language_model. Vision/audio submodules freed."
        )

    # Optional: swap to Gemma4ForCausalLM container so the model is text-only at
    # the architecture level (no multimodal wrapper, no _TextOnlyMMOutput shim,
    # simpler `model.layers.X` paths instead of `model.language_model.layers.X`).
    # The swap moves loaded module references — bnb-4bit weights and accelerate
    # device hooks travel with the modules; only the top-level wrapper changes.
    if (
        getattr(model_args, "force_text_only_causal_lm", False)
        and _vl_arch
        and hasattr(model, "model")
        and hasattr(model.model, "language_model")
    ):
        from transformers import Gemma4ForCausalLM, Gemma4TextConfig
        import torch as _torch

        # Resolve text_config (Gemma4TextConfig). Already on the multimodal config.
        _text_cfg = getattr(config, "text_config", None)
        if _text_cfg is None or not isinstance(_text_cfg, Gemma4TextConfig):
            logger.warning(
                "force_text_only_causal_lm set but config has no text_config "
                "(or wrong type) — leaving model as Gemma4ForConditionalGeneration."
            )
        else:
            # Save references to the loaded text-only modules and the top-level
            # accelerate hook for re-installation on the new container.
            _orig_inner_lang = model.model.language_model
            _orig_lm_head = model.lm_head
            _orig_top_hook = getattr(model, "_hf_hook", None)

            # Build a Gemma4ForCausalLM shell on meta to avoid CPU param allocation
            # (we'll replace its empty modules with the real ones).
            from accelerate import init_empty_weights as _init_empty_weights
            with _init_empty_weights():
                _new_model = Gemma4ForCausalLM(_text_cfg)

            # Carry over name_or_path so SFTTrainer's AutoTokenizer/AutoProcessor
            # auto-load can find the original checkpoint dir/repo.
            _src_path = getattr(model.config, "_name_or_path", None) or getattr(model_args, "model_name_or_path", None)
            if _src_path:
                _new_model.config._name_or_path = _src_path
                if hasattr(_new_model, "name_or_path"):
                    _new_model.name_or_path = _src_path

            # Move the loaded modules into the new container. These are real
            # bnb-4bit-quantized modules with their own device hooks already attached.
            _new_model.model = _orig_inner_lang
            _new_model.lm_head = _orig_lm_head

            # Re-attach the top-level accelerate hook (controls io_same_device
            # behavior). Without this, accelerate's add_hook_to_module would
            # need to be re-run — instead we just steal the hook from the old
            # top-level model.
            if _orig_top_hook is not None:
                _new_model._hf_hook = _orig_top_hook

            # Free the original wrapper (its empty model.vision_tower etc. were
            # already None'd above; only the wrapper container remains).
            del model
            _torch.cuda.empty_cache()

            model = _new_model
            # Stop the multimodal codepath from being detected by downstream code.
            _vl_arch = False
            logger.info(
                "Swapped Gemma4ForConditionalGeneration -> Gemma4ForCausalLM "
                "(force_text_only_causal_lm=True). Module references preserved; "
                "bnb-4bit + accelerate hooks travel with the modules."
            )

            # Register a CCE patch function for `gemma4_text` model_type so that
            # cce_patch(model) succeeds when training_args.use_cce is True.
            # The Gemma4ForCausalLM forward signature matches Gemma3ForCausalLM's
            # text-only `cce_forward` (input_ids/attention_mask/position_ids/...
            # — no pixel_values), so we reuse that implementation directly.
            try:
                import cut_cross_entropy.transformers.patch as _cce_patch_mod
                import cut_cross_entropy.transformers.gemma3 as _g3_cce
                from cut_cross_entropy.transformers.gemma3 import cce_forward as _g3_cce_forward
                from types import MethodType as _MethodType

                def _patch_gemma4_text(maybe_model, patch_options, remote_model_id=None):
                    # Mirror gemma3.patch_gemma3_text: install gemma3's cce_forward
                    # (which has the matching text-only signature) as the model's
                    # forward and stash patch options on the gemma3 module's
                    # _PATCH_OPTS global (cce_forward reads from there).
                    _g3_cce._PATCH_OPTS = patch_options
                    if remote_model_id is None and hasattr(maybe_model, "forward"):
                        maybe_model.forward = _MethodType(_g3_cce_forward, maybe_model)
                        return maybe_model
                    return None

                _cce_patch_mod.PATCH_FNS["gemma4_text"] = _patch_gemma4_text
                logger.info(
                    "Registered CCE patch for `gemma4_text` (reusing gemma3 cce_forward)."
                )
            except ImportError:
                logger.info(
                    "cut_cross_entropy not importable — skipping gemma4_text "
                    "CCE registration. (use_cce path will fail if enabled.)"
                )

            # Device alignment: under model_parallel, the trainer's input_ids /
            # attention_mask / position_ids can arrive on a different GPU than
            # embed_tokens.weight. Wrap Gemma4TextModel.forward so it re-aligns
            # to the embed weight device on every call, regardless of what the
            # outer cce_forward or accelerate hook chain did. This mirrors the
            # `_text_only_inner_forward` fix that the multimodal wrapper used.
            from types import MethodType as _MethodType2
            _orig_text_model_fwd = type(model.model).forward

            def _aligned_text_model_forward(self, input_ids=None,
                                            attention_mask=None,
                                            position_ids=None,
                                            **kwargs):
                _embed_dev = self.embed_tokens.weight.device
                if input_ids is not None and input_ids.device != _embed_dev:
                    input_ids = input_ids.to(_embed_dev)
                if attention_mask is not None and attention_mask.device != _embed_dev:
                    attention_mask = attention_mask.to(_embed_dev)
                if position_ids is not None and position_ids.device != _embed_dev:
                    position_ids = position_ids.to(_embed_dev)
                return _orig_text_model_fwd(
                    self,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **kwargs,
                )

            model.model.forward = _MethodType2(_aligned_text_model_forward, model.model)
            logger.info(
                "Wrapped Gemma4TextModel.forward with device alignment for "
                "model_parallel."
            )

            # Force-align all submodule hooks' execution_device to match their
            # actual weight device. The post-swap module references retained
            # hooks from the original Gemma4ForConditionalGeneration context,
            # where execution_device was set for the wrapping container — those
            # values may not match the new (raw) module weight device anymore.
            # Without this fix, accelerate's pre_forward moves input_ids to the
            # WRONG device just before it hits embed_tokens. Also align
            # io_same_device on the top-level hook.
            from accelerate.hooks import remove_hook_from_module as _remove_hook
            _fixed_hooks = 0
            _removed_hooks = 0
            for _n, _m in model.named_modules():
                _p = next(_m.parameters(recurse=False), None)
                if _p is None or _p.device.type != "cuda":
                    continue
                _hook = getattr(_m, "_hf_hook", None)
                if _hook is None:
                    continue
                _hook.execution_device = _p.device.index
                _fixed_hooks += 1
                # io_same_device on inner hooks would move output back to input
                # device — disable so logits stay on lm_head's GPU.
                if hasattr(_hook, "io_same_device"):
                    _hook.io_same_device = False

            # Print embed_tokens device info to confirm alignment.
            _et = model.model.embed_tokens
            _et_hook = getattr(_et, "_hf_hook", None)
            logger.info(
                f"embed_tokens.weight on {_et.weight.device} | "
                f"hook.execution_device={getattr(_et_hook, 'execution_device', None) if _et_hook else 'none'} | "
                f"aligned {_fixed_hooks} hooks total."
            )

            # Auto-rewrite lora_target_modules regex to drop the `language_model.`
            # prefix that the multimodal layout required. The user's regex now
            # needs to match `model.layers.X` instead of `language_model.layers.X`.
            _ltm = getattr(model_args, "lora_target_modules", None)
            if isinstance(_ltm, str) and "language_model" in _ltm:
                _new_ltm = _ltm.replace("language_model\\.", "").replace("language_model.", "")
                model_args.lora_target_modules = _new_ltm
                logger.info(
                    f"Auto-rewrote lora_target_modules: '{_ltm}' -> '{_new_ltm}'"
                )
            elif isinstance(_ltm, (list, tuple)):
                _new_ltm = [
                    s.replace("language_model\\.", "").replace("language_model.", "")
                    if isinstance(s, str) else s
                    for s in _ltm
                ]
                if _new_ltm != list(_ltm):
                    model_args.lora_target_modules = _new_ltm
                    logger.info(
                        f"Auto-rewrote lora_target_modules: {_ltm} -> {_new_ltm}"
                    )

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

    # Sync per-module hook execution_device + non-persistent buffers to actual
    # weight device. When a device_map target GPU runs out of room mid-load,
    # accelerate places the weight on a different device but (a) leaves the
    # hook's execution_device pointing at the original target, and (b) leaves
    # non-persistent buffers (e.g. Gemma4TextScaledWordEmbedding.embed_scale)
    # on the original target since they're allocated at module init from
    # config rather than loaded from the checkpoint. The hook then moves
    # inputs to the wrong GPU and arithmetic against the misplaced buffer
    # errors with "tensors on different devices". Walk the module tree and
    # re-align both.
    if model_args.model_parallel:
        import torch as _torch
        _aligned_hooks = 0
        _moved_buffers = 0
        for _name, _module in model.named_modules():
            # Determine the module's "actual" device from its first parameter.
            _primary_param = None
            for _p in _module.parameters(recurse=False):
                _primary_param = _p
                break
            if _primary_param is None or _primary_param.device.type != "cuda":
                continue
            _actual_dev = _primary_param.device

            # Move any buffers that drifted off the param device.
            for _bname, _buf in _module.named_buffers(recurse=False):
                if _buf.device != _actual_dev:
                    _module._buffers[_bname] = _buf.to(_actual_dev)
                    _moved_buffers += 1

            # Re-align hook execution_device.
            _hook = getattr(_module, "_hf_hook", None)
            if _hook is not None:
                _expected = getattr(_hook, "execution_device", None)
                _expected_idx = (
                    _expected.index if hasattr(_expected, "index") and _expected.index is not None else _expected
                )
                if _expected_idx is not None and _expected_idx != _actual_dev.index:
                    _hook.execution_device = _actual_dev.index
                    _aligned_hooks += 1
        if _aligned_hooks or _moved_buffers:
            logger.info(
                f"Post-load device sync: re-aligned {_aligned_hooks} hooks, "
                f"moved {_moved_buffers} drifted buffers to match param device"
            )

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

        # CCE dispatches by config.model_type, but composite VL models (e.g. Qwen3.5's
        # ForConditionalGeneration) need the "_vl" patch variant.  Detect this and
        # temporarily override model_type so cce_patch routes to the correct function.
        _original_model_type = getattr(model.config, "model_type", None)
        _cce_needs_vl = False
        if _original_model_type and "ForConditionalGeneration" in type(model).__name__:
            _vl_key = _original_model_type + "_vl"
            from cut_cross_entropy.transformers.patch import PATCH_FNS
            if _vl_key in PATCH_FNS:
                model.config.model_type = _vl_key
                _cce_needs_vl = True
        try:
            model = cce_patch(model)
        finally:
            if _cce_needs_vl:
                model.config.model_type = _original_model_type
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
                # Move num_items_in_batch to target device (used for loss / num_items_in_batch)
                if "num_items_in_batch" in loss_kwargs and hasattr(loss_kwargs["num_items_in_batch"], "to"):
                    loss_kwargs["num_items_in_batch"] = loss_kwargs["num_items_in_batch"].to(target_device)
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

    # Apply Liger kernel patches for model types not yet in Liger's registry.
    # Qwen3.5 uses the same RMSNorm pattern as Qwen3Next (zero-init, offset=1.0)
    # and standard SwiGLU MLPs.  Linear attention layers are left as-is.
    model_type = getattr(config, "model_type", None)
    if training_args.use_liger_kernel and model_type in ("qwen3_5", "qwen3_5_moe"):
        try:
            from liger_kernel.transformers.monkey_patch import MODEL_TYPE_TO_APPLY_LIGER_FN
            from liger_kernel.transformers.rms_norm import LigerRMSNorm
            from liger_kernel.transformers.swiglu import LigerSwiGLUMLP

            # Register a no-op so HF Trainer doesn't warn about unsupported model type
            if model_type not in MODEL_TYPE_TO_APPLY_LIGER_FN:
                MODEL_TYPE_TO_APPLY_LIGER_FN[model_type] = lambda **kwargs: None

            def _hook_aware_bind(module, method_name, new_method):
                """Bind a method to a module, respecting accelerate dispatch hooks.

                When accelerate's dispatch_model installs hooks, it saves the original
                forward as module._old_forward and replaces forward with a wrapper that
                handles device placement.  If we naively replace forward, we lose the
                hook.  Instead, patch _old_forward so the hook wrapper calls our new
                method with proper device context.
                """
                bound = new_method.__get__(module, module.__class__)
                if method_name == "forward" and hasattr(module, "_old_forward"):
                    module._old_forward = bound
                else:
                    module.__dict__[method_name] = bound

            # When using model_parallel, Triton kernels launch on torch.cuda.current_device()
            # but accelerate hooks only move tensors — they don't switch the CUDA context.
            # We wrap the Liger forward to set the correct CUDA device before the kernel runs.
            _needs_device_ctx = model_args.model_parallel

            _liger_rms_forward = LigerRMSNorm.forward
            _liger_swiglu_forward = LigerSwiGLUMLP.forward

            if _needs_device_ctx:
                import torch as _torch_liger

                def _device_ctx_rms_forward(self, hidden_states):
                    prev = _torch_liger.cuda.current_device()
                    target = hidden_states.device
                    if target.type == "cuda" and target.index != prev:
                        _torch_liger.cuda.set_device(target)
                    try:
                        return _liger_rms_forward(self, hidden_states)
                    finally:
                        if target.type == "cuda" and target.index != prev:
                            _torch_liger.cuda.set_device(prev)

                def _device_ctx_swiglu_forward(self, x):
                    prev = _torch_liger.cuda.current_device()
                    target = x.device
                    if target.type == "cuda" and target.index != prev:
                        _torch_liger.cuda.set_device(target)
                    try:
                        return _liger_swiglu_forward(self, x)
                    finally:
                        if target.type == "cuda" and target.index != prev:
                            _torch_liger.cuda.set_device(prev)

                _rms_forward_to_use = _device_ctx_rms_forward
                _swiglu_forward_to_use = _device_ctx_swiglu_forward
            else:
                _rms_forward_to_use = _liger_rms_forward
                _swiglu_forward_to_use = _liger_swiglu_forward

            def _patch_norm(module):
                """Patch a Qwen3.5 RMSNorm module with Liger's Triton kernel."""
                module.offset = 1.0
                module.casting_mode = "gemma"
                module.variance_epsilon = getattr(module, "variance_epsilon", None) or getattr(module, "eps", None) or 1e-6
                module.in_place = False  # in_place=False is safer with gradient checkpointing
                module.row_mode = None
                _hook_aware_bind(module, "forward", _rms_forward_to_use)

            # Navigate to the text model inside the composite wrapper
            base = model
            if hasattr(base, "model") and hasattr(base.model, "language_model"):
                base = base.model.language_model  # Qwen3_5ForConditionalGeneration → text model
            elif hasattr(base, "model"):
                base = base.model

            # Patch final norm
            if hasattr(base, "norm"):
                _patch_norm(base.norm)

            # Patch decoder layers: RMSNorm and SwiGLU MLP (skip linear attention modules)
            n_patched_norm = 0
            n_patched_mlp = 0
            for layer in getattr(base, "layers", []):
                if hasattr(layer, "input_layernorm"):
                    _patch_norm(layer.input_layernorm)
                    n_patched_norm += 1
                if hasattr(layer, "post_attention_layernorm"):
                    _patch_norm(layer.post_attention_layernorm)
                    n_patched_norm += 1
                if hasattr(layer, "mlp"):
                    _hook_aware_bind(layer.mlp, "forward", _swiglu_forward_to_use)
                    n_patched_mlp += 1

            logger.info(
                f"Applied Liger kernel patches to {model_type}: "
                f"{n_patched_norm} RMSNorm, {n_patched_mlp} SwiGLU MLP modules patched. "
                f"Linear attention layers left as-is."
            )
        except ImportError:
            logger.warning("Liger kernel requested but import failed — skipping Qwen3.5 patches.")

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

    # Apply chunked MLP for memory-efficient long-context training.
    # Patches gated MLP modules to process the sequence dimension in chunks,
    # reducing peak intermediate activation memory by ~num_chunks×.
    # Must be applied after model loading (and after any Liger patches) but
    # before trainer init, since PeftModel wrapping happens in the trainer.
    if training_args.chunked_mlp:
        from loft.trainer.chunked_mlp import patch_mlp_chunking
        n_patched = patch_mlp_chunking(model, num_chunks=training_args.chunked_mlp_chunks)
        logger.info(
            f"Applied chunked MLP: {n_patched} modules patched with "
            f"{training_args.chunked_mlp_chunks} chunks"
        )

    # Apply ScatterMoE kernel patches for accelerated MoE training.
    # Replaces MoE expert forward with fused Triton scatter2scatter kernels.
    # Must be applied after model loading but before trainer init.
    # Adapted from axolotl (Apache 2.0, https://github.com/axolotl-ai-collective/axolotl)
    if training_args.use_scattermoe:
        from loft.kernels.scattermoe.patch import patch_scattermoe
        success = patch_scattermoe(model)
        if success:
            logger.info("Applied ScatterMoE kernel patches")
        else:
            logger.warning("ScatterMoE patching failed — continuing without acceleration")

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

    # For multimodal-config models (e.g. Gemma 4 31B-it = Gemma4ForConditionalGeneration)
    # AutoProcessor returns a multimodal Processor in SFTTrainer's __init__, which sets
    # _is_vlm=True and triggers a hard error against assistant_only_loss. When the dataset
    # is text-only / pre-tokenized, force the tokenizer-only path by pre-loading AutoTokenizer
    # and passing it explicitly as processing_class.
    _processing_class = None
    _is_pretokenized = getattr(training_args, "_pretokenized", False)
    _has_vision_config = "ForConditionalGeneration" in type(model).__name__
    # When force_text_only_causal_lm swapped the wrapper, the model class is no
    # longer ForConditionalGeneration but the underlying checkpoint dir still
    # has multimodal Processor configs. Without an explicit AutoTokenizer pass-in,
    # SFTTrainer's auto-init loads the multimodal Processor and refuses
    # assistant_only_loss. Force the tokenizer-only path here too.
    _forced_text_only = getattr(model_args, "force_text_only_causal_lm", False)
    if (_is_pretokenized and _has_vision_config) or _forced_text_only:
        from transformers import AutoTokenizer
        _trust_remote_code = getattr(model.config, "auto_map", None) is not None
        _processing_class = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=_trust_remote_code
        )
        logger.info(
            "Forcing AutoTokenizer as processing_class to bypass multimodal "
            "Processor / VLM detection (text-only training)."
        )

    # Initialize the SFT trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args),
        **({"processing_class": _processing_class} if _processing_class is not None else {}),
    )

    # Optionally compile each decoder layer individually with torch.compile.
    # Per-layer compile fuses intermediate buffers in attention forward/backward
    # without conflicting with CCE's outer forward wrapper.  Particularly useful
    # for Gemma4 where SDPA on global head_dim=512 layers allocates large
    # intermediate buffers that fragmentation makes hard to fit at high step
    # counts. Gated by env var since first-step compile cost is non-trivial.
    # Applied AFTER trainer init so LoRA target_modules regex matches first.
    # NOTE: LOFT_COMPILE_LAYERS env var was attempted (per-layer torch.compile)
    # but breaks accelerate's unwrap_model + PEFT interaction.
    # Full-model torch_compile=True via TrainingArguments breaks CCE's forward
    # patch (NoneType hidden_states). Both paths abandoned for now.

    # Train the model — resume from checkpoint if available and requested
    resume_ckpt = getattr(training_args, "resume_from_checkpoint", None)
    # Handle string "true"/"false" from CLI/YAML
    if isinstance(resume_ckpt, str) and resume_ckpt.lower() in ("true", "yes", "1"):
        resume_ckpt = True
    elif isinstance(resume_ckpt, str) and resume_ckpt.lower() in ("false", "no", "0", "none"):
        resume_ckpt = None
    if resume_ckpt is True:
        # Auto-detect: find the latest checkpoint in output_dir
        import glob
        checkpoints = sorted(glob.glob(os.path.join(training_args.output_dir, "checkpoint-*")))
        if checkpoints:
            resume_ckpt = checkpoints[-1]
            logger.info(f"Resuming from checkpoint: {resume_ckpt}")
        else:
            logger.info("No checkpoint found in output_dir, starting fresh")
            resume_ckpt = None
    elif resume_ckpt and isinstance(resume_ckpt, str):
        logger.info(f"Resuming from specified checkpoint: {resume_ckpt}")

    trainer.train(resume_from_checkpoint=resume_ckpt)

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
