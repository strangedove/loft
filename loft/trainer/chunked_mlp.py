"""
Chunked MLP forward pass — reduces peak activation memory by processing
the sequence dimension in chunks.

The standard MLP forward computes:
    out = down_proj(act_fn(gate_proj(x)) * up_proj(x))

This materializes a [batch, seq, intermediate] tensor (e.g. 17408-wide for
Qwen3.5-27B). With long sequences this dominates GPU memory during gradient
checkpointing recompute.

Chunked MLP splits the sequence into `num_chunks` pieces, computes each
independently (MLP is position-independent), and concatenates the results.
Peak intermediate memory drops by ~num_chunks× with zero change to outputs.

Usage:
    from loft.trainer.chunked_mlp import patch_mlp_chunking, unpatch_mlp_chunking
    patch_mlp_chunking(model, num_chunks=4)
    # ... train ...
    unpatch_mlp_chunking(model)  # restore originals
"""

import torch
import torch.nn as nn
from typing import Optional


_ORIGINAL_FORWARDS = {}  # id(module) -> original forward


class ChunkedMLPForward:
    """Replacement forward that chunks along the sequence dimension."""

    def __init__(self, mlp_module: nn.Module, num_chunks: int = 4):
        self.mlp = mlp_module
        self.num_chunks = num_chunks

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]

        # Don't chunk if sequence is already short
        if seq_len <= 1024 or self.num_chunks <= 1:
            return self._original_forward(x)

        # Split along sequence dimension
        chunks = x.chunk(self.num_chunks, dim=1)
        out_chunks = []
        for chunk in chunks:
            out_chunks.append(self._original_forward(chunk))
        return torch.cat(out_chunks, dim=1)

    def _original_forward(self, x: torch.Tensor) -> torch.Tensor:
        mlp = self.mlp
        return mlp.down_proj(mlp.act_fn(mlp.gate_proj(x)) * mlp.up_proj(x))


def _is_gated_mlp(module: nn.Module) -> bool:
    """Check if a module looks like a standard gated MLP (gate_proj + up_proj + down_proj)."""
    return (
        hasattr(module, "gate_proj")
        and hasattr(module, "up_proj")
        and hasattr(module, "down_proj")
        and hasattr(module, "act_fn")
    )


def patch_mlp_chunking(model: nn.Module, num_chunks: int = 4) -> int:
    """
    Patch all gated MLP modules in the model to use chunked forward.

    Args:
        model: The model to patch (can be a PeftModel).
        num_chunks: Number of chunks to split the sequence into.

    Returns:
        Number of MLP modules patched.
    """
    count = 0
    for name, module in model.named_modules():
        if _is_gated_mlp(module) and id(module) not in _ORIGINAL_FORWARDS:
            _ORIGINAL_FORWARDS[id(module)] = module.forward
            chunked = ChunkedMLPForward(module, num_chunks=num_chunks)
            module.forward = chunked
            count += 1
    return count


def unpatch_mlp_chunking(model: nn.Module) -> int:
    """
    Restore original MLP forwards.

    Returns:
        Number of MLP modules unpatched.
    """
    count = 0
    for name, module in model.named_modules():
        if id(module) in _ORIGINAL_FORWARDS:
            module.forward = _ORIGINAL_FORWARDS.pop(id(module))
            count += 1
    return count
