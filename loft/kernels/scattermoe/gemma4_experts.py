"""
ScatterMoE-accelerated experts forward for Gemma4.

Gemma4 has no separate SparseMoeBlock — MoE is embedded in the decoder layer.
The decoder layer handles routing (Gemma4TextRouter) and calls
``experts(hidden_states, top_k_index, top_k_weights)`` directly.

This module registers a ``"scattermoe"`` implementation in the transformers
``ExpertsInterface``, which the ``@use_experts_implementation`` decorator
dispatches to when ``config._experts_implementation == "scattermoe"``.

This is the clean way to hook into transformers' MoE dispatch — no
monkeypatching required.  Works for Gemma4 and any future model that uses
``@use_experts_implementation`` with the standard forward signature
``(hidden_states, top_k_index, top_k_weights) -> Tensor``.
"""

import torch

from .parallel_experts import flatten_sort_count, parallel_linear
from .parallel_linear_lora import get_lora_params_from_wrapper, parallel_linear_lora


def _has_peft_wrapper(module):
    """Check if a module's parameter has been wrapped by PEFT ParamWrapper."""
    try:
        from peft.tuners.param_wrapper import ParamWrapper

        for attr in ("gate_up_proj", "down_proj"):
            param = getattr(module, attr, None)
            if isinstance(param, ParamWrapper):
                return True
    except ImportError:
        pass
    return False


def _unwrap_experts_lora(experts):
    """Extract base weights and LoRA params from a PEFT-wrapped Experts module.

    Returns:
        (base_experts, gup_lora, down_lora) where each lora is
        (lora_A, lora_B, scaling) or None.
    """
    try:
        from peft.tuners.param_wrapper import ParamWrapper
    except ImportError:
        return experts, None, None

    if not isinstance(getattr(experts, "gate_up_proj", None), ParamWrapper):
        return experts, None, None

    base_experts = experts
    gup_lora = None
    down_lora = None

    gup_param = experts.gate_up_proj
    if isinstance(gup_param, ParamWrapper):
        lora_A, lora_B, scaling = get_lora_params_from_wrapper(gup_param)
        if lora_A is not None:
            num_experts = experts.num_experts
            rank = lora_A.shape[0] // num_experts
            from .layers import peft_lora_to_scattermoe

            sm_A, sm_B = peft_lora_to_scattermoe(lora_A, lora_B, num_experts, rank)
            gup_lora = (sm_A, sm_B, scaling)

    down_param = experts.down_proj
    if isinstance(down_param, ParamWrapper):
        lora_A, lora_B, scaling = get_lora_params_from_wrapper(down_param)
        if lora_A is not None:
            num_experts = experts.num_experts
            rank = lora_A.shape[0] // num_experts
            from .layers import peft_lora_to_scattermoe

            sm_A, sm_B = peft_lora_to_scattermoe(lora_A, lora_B, num_experts, rank)
            down_lora = (sm_A, sm_B, scaling)

    return base_experts, gup_lora, down_lora


def _get_base_param(param):
    """Get the base tensor from a PEFT ParamWrapper or regular Parameter."""
    try:
        from peft.tuners.param_wrapper import ParamWrapper

        while isinstance(param, ParamWrapper):
            param = param.original_parameter
    except ImportError:
        pass
    return param


def _parallel_linear_maybe_lora(
    x,
    weight,
    top_k,
    sorted_expert_idxs,
    sorted_scattered_idxs,
    expert_offsets,
    lora_tuple,
    grouped_in,
    grouped_out,
    gates=None,
):
    """Call parallel_linear or parallel_linear_lora depending on whether LoRA is active."""
    if lora_tuple is not None:
        lora_A, lora_B, scaling = lora_tuple
        return parallel_linear_lora(
            x,
            weight,
            top_k,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            expert_offsets,
            lora_A,
            lora_B,
            scaling,
            grouped_in=grouped_in,
            grouped_out=grouped_out,
            gates=gates,
        )
    return parallel_linear(
        x,
        weight,
        top_k,
        sorted_expert_idxs,
        sorted_scattered_idxs,
        expert_offsets,
        grouped_in=grouped_in,
        grouped_out=grouped_out,
        gates=gates,
    )


def scattermoe_experts_forward(
    self,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """ScatterMoE-accelerated experts forward.

    Drop-in replacement for the standard Experts forward signature used by
    ``@use_experts_implementation``-decorated classes (Gemma4, Mixtral, etc.):
    ``(hidden_states [T, H], top_k_index [T, K], top_k_weights [T, K]) -> [T, H]``
    """
    K = top_k_index.shape[1]

    routing_weights = top_k_weights.to(hidden_states.dtype)
    sorted_expert_idxs, sorted_scattered_idxs, expert_offsets = flatten_sort_count(
        top_k_index, num_experts=self.num_experts
    )

    # Detect bnb NF4 parametrization (applied via bitsandbytes.nn.parametrize
    # replace_parameter_4bit on the 3D gate_up_proj / down_proj).
    # Set LOFT_SCATTERMOE_FULL_DEQUANT=1 to force full dequant (simpler,
    # slower because all 128 experts are dequantized each forward instead of
    # only the routed ones).
    import os as _os
    use_selective = (
        hasattr(self, "parametrizations")
        and (
            "gate_up_proj" in getattr(self, "parametrizations", {})
            or "down_proj" in getattr(self, "parametrizations", {})
        )
        and not _os.environ.get("LOFT_SCATTERMOE_FULL_DEQUANT")
    )

    # Extract LoRA params. Priority:
    #  1. Custom scattermoe-native attrs attached by
    #     ``loft.kernels.scattermoe.custom_lora.attach_scattermoe_lora``
    #     (bypasses PEFT materialization — recommended path for Gemma4 NF4).
    #     Respects the ``_scmoe_lora_disabled`` flag so DPO's reference
    #     forward (run inside ``disable_scmoe_lora(model)``) sees the base
    #     experts only.
    #  2. PEFT ParamWrapper at the PARAMETER level (legacy layout, unused
    #     with PEFT 0.18.1 since it wraps at the MODULE level instead).
    gup_lora, down_lora = None, None
    _scmoe_disabled = getattr(self, "_scmoe_lora_disabled", False)
    if hasattr(self, "_scmoe_lora_A_gate_up") and not _scmoe_disabled:
        gup_lora = (
            self._scmoe_lora_A_gate_up,
            self._scmoe_lora_B_gate_up,
            float(self._scmoe_lora_scaling_gate_up.item()),
        )
        down_lora = (
            self._scmoe_lora_A_down,
            self._scmoe_lora_B_down,
            float(self._scmoe_lora_scaling_down.item()),
        )
    elif _has_peft_wrapper(self):
        _, gup_lora, down_lora = _unwrap_experts_lora(self)

    if use_selective:
        # NF4 path: only dequant + use experts routed to by this batch
        from .selective_dequant import (
            get_active_experts,
            remap_expert_indices,
            selective_expert_weights,
            selective_lora_weights,
        )

        active = get_active_experts(sorted_expert_idxs, self.num_experts)
        remapped_idxs, compact_offsets = remap_expert_indices(
            sorted_expert_idxs, expert_offsets, active, self.num_experts
        )

        # Memory optimization: dequant gate_up_proj first, use it, free it,
        # THEN dequant down_proj. Holding both simultaneously doubles peak
        # scratch memory per layer (~1 GB each at seq=4096, 128 active experts).
        gate_up_weight = selective_expert_weights(self, "gate_up_proj", active).transpose(2, 1)

        # Slice LoRA to active experts so indices align with the compact weight
        if gup_lora is not None:
            A, B, scaling = gup_lora
            A, B = selective_lora_weights(A, B, active, self.num_experts)
            gup_lora = (A, B, scaling)

        sei = remapped_idxs
        eo = compact_offsets
        _selective_down_pending = True
    else:
        # Dense path: full 3D weight + original indexing
        gate_up_weight = _get_base_param(self.gate_up_proj).transpose(2, 1)
        down_weight = _get_base_param(self.down_proj).transpose(2, 1)
        sei = sorted_expert_idxs
        eo = expert_offsets
        _selective_down_pending = False

    # Gate-up projection (with optional LoRA)
    gates_h = _parallel_linear_maybe_lora(
        hidden_states,
        gate_up_weight,
        K,
        sei,
        sorted_scattered_idxs,
        eo,
        gup_lora,
        grouped_in=False,
        grouped_out=True,
    )
    gates, h = gates_h.chunk(2, dim=-1)
    h = self.act_fn(gates) * h

    # Free gate_up scratch BEFORE dequantizing down (selective NF4 path only).
    # These refs hold the full dequantized [E_active, 2*I, H] tensor alive;
    # without explicit del, peak memory = gate_up + down simultaneously.
    del gate_up_weight, gates, gates_h
    if _selective_down_pending:
        if gup_lora is not None:
            del gup_lora
        # Now safe to dequant down_proj
        down_weight = selective_expert_weights(self, "down_proj", active).transpose(2, 1)
        if down_lora is not None:
            A, B, scaling = down_lora
            A, B = selective_lora_weights(A, B, active, self.num_experts)
            down_lora = (A, B, scaling)

    # Down projection (with optional LoRA + routing weights)
    output = _parallel_linear_maybe_lora(
        h,
        down_weight,
        1,
        sei,
        sorted_scattered_idxs,
        eo,
        down_lora,
        grouped_in=True,
        grouped_out=False,
        gates=routing_weights,
    )

    return output


def register_scattermoe_experts():
    """Register ``"scattermoe"`` in the transformers ExpertsInterface.

    After calling this, any model with ``@use_experts_implementation`` will
    dispatch to ScatterMoE when ``config._experts_implementation == "scattermoe"``.

    Also patches ``get_correct_experts_implementation`` to accept ``"scattermoe"``
    as a valid value (transformers hardcodes an allowlist).
    """
    from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS
    from transformers.modeling_utils import PreTrainedModel

    # 1. Register the forward function in the global interface
    ALL_EXPERTS_FUNCTIONS.register("scattermoe", scattermoe_experts_forward)

    # 2. Patch the validation to accept "scattermoe"
    _original_get_correct = PreTrainedModel.get_correct_experts_implementation

    def _patched_get_correct(self_model, requested_experts: str | None) -> str:
        if requested_experts == "scattermoe":
            return "scattermoe"
        return _original_get_correct(self_model, requested_experts)

    PreTrainedModel.get_correct_experts_implementation = _patched_get_correct


# Legacy monkeypatch approach (kept for backward compat with existing tests)
def patch_gemma4_scattermoe():
    """Monkeypatch Gemma4TextExperts.forward with ScatterMoE kernel."""
    # axolotl import removed - use transformers directly

    experts_cls = resolve_experts_class("gemma4_text")
    if experts_cls is None:
        raise ValueError("Could not resolve Gemma4TextExperts class")

    if hasattr(experts_cls, "_original_forward"):
        return  # already patched

    experts_cls._original_forward = experts_cls.forward
    experts_cls.forward = scattermoe_experts_forward
