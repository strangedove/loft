"""
Triton-fused kernels for RL/DPO training.

Adapted from Axolotl v0.16.0 (Apache 2.0 License).
Original: https://github.com/axolotl-ai-cloud/axolotl/blob/v0.16.0/src/axolotl/monkeypatch/trainer/utils.py

Two kernels:
1. entropy_from_logits: Single-pass online entropy, 5.2x faster than chunked softmax
2. selective_log_softmax: Fused gather + logsumexp with custom backward, 2.9x faster

Both avoid materializing the full [batch, seq, vocab_size] softmax tensor.
"""

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ─── Entropy from logits ──────────────────────────────────────────

if HAS_TRITON:
    @triton.jit
    def _entropy_online_kernel(
        logits_ptr,
        output_ptr,
        stride_row,
        V: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        """Online entropy: single pass with running max correction."""
        row = tl.program_id(0)
        row_ptr = logits_ptr + tl.cast(row, tl.int64) * stride_row

        running_max = tl.full([], float("-inf"), dtype=tl.float32)
        running_sum_exp = tl.full([], 0.0, dtype=tl.float32)
        running_weighted = tl.full([], 0.0, dtype=tl.float32)

        for v_start in range(0, V, BLOCK_V):
            offs = v_start + tl.arange(0, BLOCK_V)
            mask = offs < V
            x = tl.load(row_ptr + offs, mask=mask, other=float("-inf")).to(tl.float32)

            block_max = tl.max(x, axis=0)
            new_max = tl.maximum(running_max, block_max)

            correction = tl.exp(running_max - new_max)
            running_sum_exp = running_sum_exp * correction
            running_weighted = running_weighted * correction

            exp_x = tl.exp(x - new_max)
            exp_x = tl.where(mask, exp_x, 0.0)
            x = tl.where(mask, x, 0.0)
            running_sum_exp += tl.sum(exp_x, axis=0)
            running_weighted += tl.sum(exp_x * x, axis=0)

            running_max = new_max

        entropy = tl.log(running_sum_exp) + running_max - running_weighted / running_sum_exp
        tl.store(output_ptr + row, entropy)


def entropy_from_logits(logits: torch.Tensor, chunk_size: int = 128) -> torch.Tensor:
    """Compute entropy from logits. Uses Triton if available, falls back to chunked."""
    original_shape = logits.shape[:-1]
    V = logits.shape[-1]
    N = 1
    for s in original_shape:
        N *= s

    if not logits.is_cuda or not HAS_TRITON:
        # CPU/no-Triton fallback
        logp = F.log_softmax(logits.float(), dim=-1)
        ent = -(logp.exp() * logp).sum(dim=-1)
        return ent.to(logits.dtype).reshape(original_shape)

    flat = logits.reshape(N, V)
    if not flat.is_contiguous():
        flat = flat.contiguous()

    output = torch.empty(N, device=logits.device, dtype=torch.float32)
    BLOCK_V = 4096
    MAX_GRID = 8192

    for start in range(0, N, MAX_GRID):
        end = min(start + MAX_GRID, N)
        chunk_n = end - start
        _entropy_online_kernel[(chunk_n,)](
            flat[start:end],
            output[start:end],
            flat.stride(0),
            V=V,
            BLOCK_V=BLOCK_V,
        )

    return output.to(logits.dtype).reshape(original_shape)


# ─── Selective log softmax ────────────────────────────────────────

if HAS_TRITON:
    @triton.jit
    def _selective_logsoftmax_fwd_kernel(
        logits_ptr,
        index_ptr,
        output_ptr,
        logsumexp_ptr,
        stride_logits_row,
        stride_index_row,
        stride_output_row,
        actual_K,
        K_BLOCK: tl.constexpr,
        V: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        """Forward: online logsumexp + gather. Saves logsumexp for backward."""
        row = tl.program_id(0)
        logits_row_ptr = logits_ptr + tl.cast(row, tl.int64) * stride_logits_row

        # Online logsumexp
        running_max = tl.full([], float("-inf"), dtype=tl.float32)
        running_sum_exp = tl.full([], 0.0, dtype=tl.float32)

        for v_start in range(0, V, BLOCK_V):
            offs = v_start + tl.arange(0, BLOCK_V)
            mask = offs < V
            x = tl.load(logits_row_ptr + offs, mask=mask, other=float("-inf")).to(tl.float32)

            block_max = tl.max(x, axis=0)
            new_max = tl.maximum(running_max, block_max)
            running_sum_exp = running_sum_exp * tl.exp(running_max - new_max)

            exp_x = tl.exp(x - new_max)
            exp_x = tl.where(mask, exp_x, 0.0)
            running_sum_exp += tl.sum(exp_x, axis=0)
            running_max = new_max

        lse = tl.log(running_sum_exp) + running_max
        tl.store(logsumexp_ptr + row, lse)

        # Gather and subtract
        index_row_ptr = index_ptr + tl.cast(row, tl.int64) * stride_index_row
        output_row_ptr = output_ptr + tl.cast(row, tl.int64) * stride_output_row

        k_offs = tl.arange(0, K_BLOCK)
        k_mask = k_offs < actual_K
        indices = tl.load(index_row_ptr + k_offs, mask=k_mask, other=0).to(tl.int64)
        valid_mask = k_mask & (indices >= 0) & (indices < V)
        safe_indices = tl.where(valid_mask, indices, 0)
        selected = tl.load(logits_row_ptr + safe_indices, mask=valid_mask, other=0.0).to(tl.float32)
        tl.store(output_row_ptr + k_offs, selected - lse, mask=valid_mask)

    @triton.jit
    def _selective_logsoftmax_bwd_kernel(
        grad_output_ptr,
        logsumexp_ptr,
        logits_ptr,
        index_ptr,
        grad_logits_ptr,
        stride_grad_output_row,
        stride_logits_row,
        stride_index_row,
        stride_grad_logits_row,
        actual_K,
        K_BLOCK: tl.constexpr,
        V: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        """Backward: compute grad_logits from grad_output and saved logsumexp."""
        row = tl.program_id(0)

        lse = tl.load(logsumexp_ptr + row)
        logits_row = logits_ptr + tl.cast(row, tl.int64) * stride_logits_row
        grad_out_row = grad_output_ptr + tl.cast(row, tl.int64) * stride_grad_output_row
        index_row = index_ptr + tl.cast(row, tl.int64) * stride_index_row
        grad_row = grad_logits_ptr + tl.cast(row, tl.int64) * stride_grad_logits_row

        # Sum of upstream gradients
        k_offs = tl.arange(0, K_BLOCK)
        k_mask = k_offs < actual_K
        go = tl.load(grad_out_row + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        go_sum = tl.sum(go, axis=0)

        # softmax(x_i) * go_sum  for all vocab
        for v_start in range(0, V, BLOCK_V):
            offs = v_start + tl.arange(0, BLOCK_V)
            v_mask = offs < V
            x = tl.load(logits_row + offs, mask=v_mask, other=0.0).to(tl.float32)
            soft = tl.exp(x - lse)
            grad_val = -soft * go_sum
            tl.store(grad_row + offs, grad_val, mask=v_mask)

        # Add go for selected indices
        indices = tl.load(index_row + k_offs, mask=k_mask, other=0).to(tl.int64)
        valid = k_mask & (indices >= 0) & (indices < V)
        safe_idx = tl.where(valid, indices, 0)
        existing = tl.load(grad_row + safe_idx, mask=valid, other=0.0)
        tl.store(grad_row + safe_idx, existing + go, mask=valid)


    class _SelectiveLogSoftmaxTriton(torch.autograd.Function):
        @staticmethod
        def forward(ctx, logits, index, K, K_BLOCK, V, BLOCK_V, MAX_GRID):
            N = logits.shape[0]
            output = torch.empty(N, K_BLOCK, device=logits.device, dtype=torch.float32)
            logsumexp = torch.empty(N, device=logits.device, dtype=torch.float32)

            for start in range(0, N, MAX_GRID):
                end = min(start + MAX_GRID, N)
                chunk_n = end - start
                _selective_logsoftmax_fwd_kernel[(chunk_n,)](
                    logits[start:end], index[start:end], output[start:end],
                    logsumexp[start:end],
                    logits.stride(0), index.stride(0), output.stride(0),
                    actual_K=K, K_BLOCK=K_BLOCK, V=V, BLOCK_V=BLOCK_V,
                )

            ctx.save_for_backward(logits, index, logsumexp)
            ctx.K = K
            ctx.K_BLOCK = K_BLOCK
            ctx.V = V
            ctx.BLOCK_V = BLOCK_V
            ctx.MAX_GRID = MAX_GRID
            return output

        @staticmethod
        def backward(ctx, grad_output):
            logits, index, logsumexp = ctx.saved_tensors
            N = logits.shape[0]
            grad_logits = torch.zeros_like(logits)
            grad_output = grad_output.contiguous()

            for start in range(0, N, ctx.MAX_GRID):
                end = min(start + N, N)
                chunk_n = end - start
                _selective_logsoftmax_bwd_kernel[(chunk_n,)](
                    grad_output[start:end], logsumexp[start:end],
                    logits[start:end], index[start:end],
                    grad_logits[start:end],
                    grad_output.stride(0), logits.stride(0),
                    index.stride(0), grad_logits.stride(0),
                    actual_K=ctx.K, K_BLOCK=ctx.K_BLOCK,
                    V=ctx.V, BLOCK_V=ctx.BLOCK_V,
                )

            return grad_logits, None, None, None, None, None, None


def selective_log_softmax(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Fused selective_log_softmax: gather(log_softmax(logits), index).

    Uses Triton kernels if available, otherwise falls back to PyTorch.
    Much faster and more memory-efficient than materializing the full softmax.
    """
    squeeze = index.ndim == logits.ndim - 1
    if squeeze:
        index = index.unsqueeze(-1)

    if not logits.is_cuda or not HAS_TRITON:
        # Fallback
        if logits.dtype in [torch.float32, torch.float64]:
            selected = torch.gather(logits, dim=-1, index=index)
            lse = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
            per_token_logps = selected - lse.unsqueeze(-1)
        else:
            per_token_logps = []
            for row_logits, row_labels in zip(logits, index):
                row_logps = F.log_softmax(row_logits, dim=-1)
                per_token_logps.append(row_logps.gather(dim=-1, index=row_labels))
            per_token_logps = torch.stack(per_token_logps)
        if squeeze:
            per_token_logps = per_token_logps.squeeze(-1)
        return per_token_logps

    V = logits.shape[-1]
    K = index.shape[-1]
    original_index_shape = index.shape

    flat_logits = logits.reshape(-1, V).contiguous()
    flat_index = index.reshape(-1, K).contiguous()

    BLOCK_V = 4096
    MAX_GRID = 8192
    K_BLOCK = max(1, triton.next_power_of_2(K))

    output = _SelectiveLogSoftmaxTriton.apply(
        flat_logits, flat_index, K, K_BLOCK, V, BLOCK_V, MAX_GRID
    )

    if K_BLOCK != K:
        output = output[:, :K]

    per_token_logps = output.to(logits.dtype).reshape(original_index_shape)

    if squeeze:
        per_token_logps = per_token_logps.squeeze(-1)

    return per_token_logps
