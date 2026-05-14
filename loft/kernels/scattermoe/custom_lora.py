"""Custom scattermoe-native LoRA for MoE experts modules (Gemma4 and similar).

Why this exists:

PEFT 0.18.1's ``target_parameters`` wraps each expert module in a ParamWrapper
and materializes the full 3D delta weight via einsum in ``_activate_lora`` on
every forward. For Gemma4 26B-A4B that is ~1.89 GiB per layer × 60 layer-params,
which blows past 2×24 GiB. Loft's scattermoe kernels already support a LoRA
path via ``parallel_linear_lora``; this module attaches the LoRA as plain
``nn.Parameter`` attributes (``_scmoe_lora_*``) on each experts module so the
kernel picks them up directly without any PEFT materialization.

Layout (matches ``ScatterMoELoRA`` / ``_compute_lora_input_grad``):
    lora_A: [R*E, K]   K = input dim
    lora_B: [N, R*E]   N = output dim

For Gemma4TextExperts:
    gate_up_proj weight: [E, 2*intermediate_dim, hidden_dim]
      kernel sees it as [E, K=hidden_dim, N=2*intermediate_dim] (after .transpose(2,1))
      → A_gu [R*E, H],  B_gu [2I, R*E]
    down_proj weight:    [E, hidden_dim, intermediate_dim]
      kernel sees [E, K=intermediate_dim, N=hidden_dim]
      → A_dn [R*E, I],  B_dn [H, R*E]

## DPO integration

For DPO training, the reference forward must see the model WITHOUT the LoRA
delta. PEFT supplies ``PeftModel.disable_adapter()`` for its own adapters, but
knows nothing about these plain-parameter attachments. This module adds:

- A per-module boolean flag ``_scmoe_lora_disabled`` read by
  ``scattermoe_experts_forward`` to skip the LoRA branch.
- ``disable_scmoe_lora(model)`` contextmanager that sets the flag on every
  experts module with attached scmoe_lora, clears it on exit.
- ``set_scmoe_lora_disabled(model, bool)`` helper for explicit control.

Typical DPO wiring (see ``loft/scripts/dpo.py``): patch ``DPOTrainer.null_ref_context``
to compose ``PeftModel.disable_adapter()`` with ``disable_scmoe_lora(model)``.
"""

from __future__ import annotations

import math
from contextlib import contextmanager

import torch
import torch.nn as nn


_LORA_ATTRS = [
    "_scmoe_lora_A_gate_up",
    "_scmoe_lora_B_gate_up",
    "_scmoe_lora_A_down",
    "_scmoe_lora_B_down",
]

_SCALING_ATTRS = [
    "_scmoe_lora_scaling_gate_up",
    "_scmoe_lora_scaling_down",
]

_DISABLE_FLAG = "_scmoe_lora_disabled"


def _experts_device(experts_mod: nn.Module) -> torch.device:
    """Return the device holding this experts module's weights.

    Works for both plain nn.Parameter (``experts.gate_up_proj``) and the bnb
    NF4 parametrization case where the true param lives at
    ``experts.parametrizations.gate_up_proj.original``.
    """
    if hasattr(experts_mod, "parametrizations") and "gate_up_proj" in experts_mod.parametrizations:
        return experts_mod.parametrizations["gate_up_proj"].original.device
    return experts_mod.gate_up_proj.device


def _is_experts_module(name: str, module: nn.Module) -> bool:
    """Duck-type test for an MoE experts module."""
    if not (
        hasattr(module, "num_experts")
        and hasattr(module, "hidden_dim")
        and hasattr(module, "intermediate_dim")
        and hasattr(module, "gate_up_proj")
        and hasattr(module, "down_proj")
    ):
        return False
    # Skip anything not actually called "experts" in the attribute path.
    if "experts" not in name.rsplit(".", 1)[-1].lower() and not name.endswith(".experts"):
        return False
    return True


def iter_scmoe_experts(model: nn.Module):
    """Yield (name, module) for every experts module the scmoe LoRA is/can be attached to."""
    for name, module in model.named_modules():
        if _is_experts_module(name, module):
            yield name, module


def attach_scattermoe_lora(
    model: nn.Module,
    rank: int = 16,
    alpha: float = 16.0,
    use_rslora: bool = True,
    dtype: torch.dtype = torch.bfloat16,
    verbose: bool = True,
) -> list[nn.Parameter]:
    """Walk the model, attach scattermoe-layout LoRA params to every experts
    module, return the list of new trainable parameters.

    Does not touch existing parameters' requires_grad — the caller should
    freeze everything else and leave the returned params trainable.
    """
    new_params: list[nn.Parameter] = []
    n_attached = 0

    for name, module in iter_scmoe_experts(model):
        E = module.num_experts
        H = module.hidden_dim
        I = module.intermediate_dim  # noqa: E741
        dev = _experts_device(module)

        if use_rslora:
            scaling = alpha / math.sqrt(rank)
        else:
            scaling = alpha / rank

        # gate_up_proj: K=H, N=2I
        A_gu = nn.Parameter(
            torch.empty(rank * E, H, device=dev, dtype=dtype), requires_grad=True
        )
        B_gu = nn.Parameter(
            torch.zeros(2 * I, rank * E, device=dev, dtype=dtype), requires_grad=True
        )
        # down_proj: K=I, N=H
        A_dn = nn.Parameter(
            torch.empty(rank * E, I, device=dev, dtype=dtype), requires_grad=True
        )
        B_dn = nn.Parameter(
            torch.zeros(H, rank * E, device=dev, dtype=dtype), requires_grad=True
        )

        # Kaiming init for A (PEFT convention: a=sqrt(5), matches nn.Linear init)
        nn.init.kaiming_uniform_(A_gu, a=math.sqrt(5))
        nn.init.kaiming_uniform_(A_dn, a=math.sqrt(5))
        # B remains zero → initial delta contribution is zero.

        module.register_parameter("_scmoe_lora_A_gate_up", A_gu)
        module.register_parameter("_scmoe_lora_B_gate_up", B_gu)
        module.register_parameter("_scmoe_lora_A_down", A_dn)
        module.register_parameter("_scmoe_lora_B_down", B_dn)
        module.register_buffer(
            "_scmoe_lora_scaling_gate_up",
            torch.tensor(float(scaling), device=dev, dtype=torch.float32),
            persistent=True,
        )
        module.register_buffer(
            "_scmoe_lora_scaling_down",
            torch.tensor(float(scaling), device=dev, dtype=torch.float32),
            persistent=True,
        )
        # Disabled flag — not a Parameter/Buffer (don't want it in state_dict);
        # pure python attribute read at forward time.
        module._scmoe_lora_disabled = False

        new_params.extend([A_gu, B_gu, A_dn, B_dn])
        n_attached += 1

    if verbose:
        print(
            f"  [scmoe-lora] attached LoRA (r={rank}, α={alpha}, rslora={use_rslora}) "
            f"to {n_attached} experts modules"
        )
        print(
            f"  [scmoe-lora] {len(new_params)} new Parameters, "
            f"{sum(p.numel() for p in new_params):,} total elements"
        )
    return new_params


def has_scmoe_lora(model: nn.Module) -> bool:
    """Return True if ANY experts module in ``model`` has scmoe_lora attached."""
    for _, module in iter_scmoe_experts(model):
        if hasattr(module, "_scmoe_lora_A_gate_up"):
            return True
    return False


def set_scmoe_lora_disabled(model: nn.Module, disabled: bool) -> int:
    """Set the disable flag on every scmoe-lora-bearing experts module.

    Returns the number of modules touched.
    """
    n = 0
    for _, module in iter_scmoe_experts(model):
        if hasattr(module, "_scmoe_lora_A_gate_up"):
            module._scmoe_lora_disabled = bool(disabled)
            n += 1
    return n


@contextmanager
def disable_scmoe_lora(model: nn.Module):
    """Contextmanager that disables scmoe_lora for every experts module inside.

    Nests cleanly — inner flag state is saved and restored per-module on exit.
    Safe to call even if the model has no scmoe_lora attached (no-op).
    """
    saved: list[tuple[nn.Module, bool]] = []
    for _, module in iter_scmoe_experts(model):
        if hasattr(module, "_scmoe_lora_A_gate_up"):
            saved.append((module, getattr(module, _DISABLE_FLAG, False)))
            module._scmoe_lora_disabled = True
    try:
        yield
    finally:
        for module, prev in saved:
            module._scmoe_lora_disabled = prev


def extract_scattermoe_lora_tuples(experts_mod: nn.Module):
    """Return (gup_lora, down_lora) tuples if attached and not disabled, else (None, None).

    Each tuple is (lora_A, lora_B, scaling) in scattermoe layout.
    """
    if not hasattr(experts_mod, "_scmoe_lora_A_gate_up"):
        return None, None
    if getattr(experts_mod, _DISABLE_FLAG, False):
        return None, None
    gup = (
        experts_mod._scmoe_lora_A_gate_up,
        experts_mod._scmoe_lora_B_gate_up,
        float(experts_mod._scmoe_lora_scaling_gate_up.item()),
    )
    dn = (
        experts_mod._scmoe_lora_A_down,
        experts_mod._scmoe_lora_B_down,
        float(experts_mod._scmoe_lora_scaling_down.item()),
    )
    return gup, dn


def save_scmoe_lora(model: nn.Module, path: str) -> int:
    """Save all ``_scmoe_lora_*`` params + scaling buffers to a safetensors file.

    Companion to PEFT's ``adapter_model.safetensors`` — PEFT doesn't know
    about these plain-nn.Parameter attachments, so Trainer's ``save_model``
    drops them silently. Call this manually after training / from a
    checkpoint callback.
    """
    from safetensors.torch import save_file

    state: dict[str, torch.Tensor] = {}
    for mod_name, mod in model.named_modules():
        for attr in _LORA_ATTRS:
            if hasattr(mod, attr):
                state[f"{mod_name}.{attr}"] = getattr(mod, attr).detach().cpu()
        for attr in _SCALING_ATTRS:
            if hasattr(mod, attr):
                state[f"{mod_name}.{attr}"] = getattr(mod, attr).detach().cpu()
    save_file(state, path)
    print(f"  [scmoe-lora] saved {len(state)} tensors to {path}")
    return len(state)


def load_scmoe_lora(model: nn.Module, path: str) -> int:
    """Load scmoe-lora tensors previously saved by ``save_scmoe_lora``.

    Assumes ``attach_scattermoe_lora`` has been called first so the target
    params exist. Copies values in-place. Returns the number of tensors
    actually applied.
    """
    from safetensors.torch import load_file

    state = load_file(path)
    mod_by_name = dict(model.named_modules())
    n = 0
    for k, v in state.items():
        mod_name, _, attr = k.rpartition(".")
        mod = mod_by_name.get(mod_name)
        if mod is None or not hasattr(mod, attr):
            print(f"  [scmoe-lora] skipping {k}: no target")
            continue
        target = getattr(mod, attr)
        target.data.copy_(v.to(target.device, target.dtype))
        n += 1
    print(f"  [scmoe-lora] loaded {n} / {len(state)} tensors from {path}")
    return n


def freeze_everything_except_scmoe_lora_and_peft(model: nn.Module) -> int:
    """Freeze all params except our custom scmoe-lora params AND PEFT lora params.

    PEFT's attention LoRA params are identified by the presence of ``lora_``
    in their name (PEFT's prefix convention). Our scmoe params start with
    ``_scmoe_lora_``. Returns the count of params that remain trainable.
    """
    n_trainable = 0
    for pname, p in model.named_parameters():
        if "_scmoe_lora_" in pname or ".lora_" in pname or pname.startswith("lora_"):
            p.requires_grad = True
            n_trainable += 1
        else:
            p.requires_grad = False
    return n_trainable
