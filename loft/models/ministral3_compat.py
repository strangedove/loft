"""
Ministral3 compatibility shim for transformers 4.57.

Ministral3ForCausalLM is a model type added in transformers 5.x. It is
architecturally identical to MistralForCausalLM except for one thing:
after applying RoPE, it scales query states by a position-dependent factor
borrowed from Llama 4 ("llama_4_scaling_beta").

This module provides:
  - ``patch_mistral_for_ministral3``: monkey-patches MistralAttention.forward
    to apply the Llama 4 attention scaling when the config carries the
    ``_ministral3_llama4_beta`` marker set by the config remap script.

The scaling function is:
    scale = 1 + beta * log(1 + floor(pos / original_max_position_embeddings))
For positions below original_max_position_embeddings (default 16384) the
scale is exactly 1.0 — it only activates for extended-context inference.
"""

import logging

import torch
from transformers.models.mistral.modeling_mistral import MistralAttention

logger = logging.getLogger(__name__)

# Stash the original forward so we can call through to it.
_original_mistral_attention_forward = MistralAttention.forward.__wrapped__ if hasattr(MistralAttention.forward, '__wrapped__') else MistralAttention.forward


def _get_llama_4_attn_scale(
    position_ids: torch.Tensor,
    beta: float,
    max_position_embeddings: int,
) -> torch.Tensor:
    """Position-dependent query scaling from Llama 4, used by Ministral3."""
    scaling = 1 + beta * torch.log(1 + torch.floor(position_ids / max_position_embeddings))
    return scaling.unsqueeze(-1)


def _patched_mistral_attention_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, cache_position=None, **kwargs):
    """MistralAttention.forward with Llama 4 query scaling injected."""
    from transformers.models.mistral.modeling_mistral import (
        apply_rotary_pos_emb,
        eager_attention_forward,
        repeat_kv,
    )
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # --- Ministral3 addition: Llama 4 attention scaling ---
    beta = getattr(self.config, "_ministral3_llama4_beta", None)
    orig_max_pos = getattr(self.config, "_ministral3_orig_max_pos", None)
    if beta is not None and orig_max_pos is not None and cache_position is not None:
        scale = _get_llama_4_attn_scale(cache_position, beta, orig_max_pos)
        query_states = query_states * scale.to(query_states.dtype)
    # --- end Ministral3 addition ---

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    attention_interface = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=getattr(self.config, "sliding_window", None),
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def patch_mistral_for_ministral3(model):
    """
    Apply the Ministral3 Llama-4 attention scaling monkey-patch.

    Only activates if the model config has ``_ministral3_llama4_beta`` set
    (which the remap script adds). Safe to call on any MistralForCausalLM —
    it's a no-op when the marker is absent.

    Args:
        model: A loaded PreTrainedModel (typically MistralForCausalLM).

    Returns:
        The same model (patched in-place).
    """
    beta = getattr(model.config, "_ministral3_llama4_beta", None)
    if beta is None:
        return model

    MistralAttention.forward = _patched_mistral_attention_forward
    logger.info(
        f"Patched MistralAttention with Ministral3 Llama-4 query scaling "
        f"(beta={beta}, orig_max_pos={model.config._ministral3_orig_max_pos})"
    )
    return model
