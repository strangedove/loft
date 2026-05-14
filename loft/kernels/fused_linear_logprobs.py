"""Fused ``hidden @ lm_head + log_softmax + gather(labels)`` for DPO.

TRL's ``DPOTrainer.concatenated_forward`` needs per-token log-probabilities
at the label positions. The reference path materializes
``logits = h @ W_lm.T`` as a ``[N, V]`` tensor and calls
``F.log_softmax(logits, -1).gather(-1, labels)``. For 262K-vocab models
(Gemma4) at 3K sequence, ``[N, V]`` alone is 2-3 GiB in bf16 and its fp32
grad is another 6 GiB — enough to OOM a 24 GiB 3090 on top of scattermoe
activations.

``loft.trainer.chunked_logprobs`` already chunks this along the vocab
dimension, but still goes through an explicit ``h @ W_chunk.T`` materialize
per chunk. This module provides a tighter autograd.Function that:

1. **Never materializes the full ``[N, V]`` tensor.** Forward streams a
   vocab chunk at a time, keeps a running ``M`` (max) and ``S`` (sum of
   shifted exponentials) in fp32 per row, and pulls the label's logit
   aside as the chunk goes by. Output is ``[N]`` logprobs + ``[N]`` saved
   logsumexp values (for backward).
2. **Backward recomputes per chunk.** The grad of ``lp_i`` w.r.t. a
   logit row is ``(onehot(label_i) - softmax_i) * grad_lp_i``. We rebuild
   each chunk's logits from saved ``hidden`` and ``W_chunk``, compute
   softmax with the saved logsumexp, and accumulate
   ``grad_hidden += grad_logits_chunk @ W_chunk`` plus
   ``grad_W[V_slice] += grad_logits_chunk.T @ hidden``. Peak extra memory
   stays ``O(N * V_chunk)`` not ``O(N * V)``.
3. **Optional softcap** (``tanh(logits/cap) * cap``) applied inline in
   both forward and backward. Gemma4's ``final_logit_softcapping=30.0``
   needs this to produce matching logprobs.
4. **Optional max-logit export** for ``aux_loss_top_prob_weight``: the
   running-max tracker already computes ``max_logit`` per row for free;
   we return it as a second tensor the caller can use to compute
   ``-log(max_softmax)`` without a separate pass.

Status: Phase-1 prototype. Hybrid triton + torch — matmuls stay on cuBLAS
(no win from rewriting those), triton handles the online logsumexp +
softmax recompute which is where the memory pressure lives.

Drop-in for DPO:

    from loft.kernels.fused_linear_logprobs import fused_linear_logprobs

    # Replace:
    #     logits = model.lm_head(hidden_states)      # [N, V] — OOM
    #     lps = F.log_softmax(logits, -1).gather(-1, labels.unsqueeze(-1))
    # with:
    lps, max_logit = fused_linear_logprobs(
        hidden_states.reshape(-1, H),   # [N, H]
        model.lm_head.weight,           # [V, H]
        labels.reshape(-1),             # [N]
        softcap=30.0,                   # Gemma4; None for Qwen3.x
        chunk_size=4096,
        return_max_logit=True,
    )
    lps = lps.reshape(labels.shape)

Testing checklist (GPU):
  [ ] forward matches reference F.log_softmax+gather within 1e-5 bf16
  [ ] backward grad_hidden matches within 1e-4
  [ ] backward grad_W matches within 1e-4
  [ ] softcap path matches reference for Gemma4
  [ ] peak memory is ≈ N * chunk_size * 4 bytes (not N * V)
  [ ] speed within 5% of Liger fused_linear_cross_entropy (no-aux case)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:  # pragma: no cover
    HAS_TRITON = False


# ---------------------------------------------------------------------------
# Triton kernels — online logsumexp and softmax recompute.
# ---------------------------------------------------------------------------

if HAS_TRITON:

    @triton.jit
    def _online_lse_update_kernel(
        logits_chunk_ptr,   # [N, V_chunk] fp32 or bf16
        M_ptr,              # [N] fp32 running max (in/out)
        S_ptr,              # [N] fp32 running sum-exp (in/out)
        N,
        V_chunk,
        stride_logits_n,
        stride_logits_v,
        BLOCK_V: tl.constexpr,
    ):
        """Update running (M, S) given a vocab chunk's logits.

        For each row:
            M_new = max(M_old, row_max)
            S_new = exp(M_old - M_new) * S_old + sum(exp(row - M_new))

        Launched as a 1D grid over rows (N).
        """
        row = tl.program_id(0)
        if row >= N:
            return

        m_old = tl.load(M_ptr + row)
        s_old = tl.load(S_ptr + row)

        # First pass: find row max across the chunk
        row_max = tl.full([], -float("inf"), tl.float32)
        for off in range(0, V_chunk, BLOCK_V):
            v_idx = off + tl.arange(0, BLOCK_V)
            mask = v_idx < V_chunk
            logits = tl.load(
                logits_chunk_ptr + row * stride_logits_n + v_idx * stride_logits_v,
                mask=mask,
                other=-float("inf"),
            ).to(tl.float32)
            row_max = tl.maximum(row_max, tl.max(logits, axis=0))

        m_new = tl.maximum(m_old, row_max)

        # Second pass: accumulate shifted exp sum at m_new
        row_sum = tl.zeros([], tl.float32)
        for off in range(0, V_chunk, BLOCK_V):
            v_idx = off + tl.arange(0, BLOCK_V)
            mask = v_idx < V_chunk
            logits = tl.load(
                logits_chunk_ptr + row * stride_logits_n + v_idx * stride_logits_v,
                mask=mask,
                other=-float("inf"),
            ).to(tl.float32)
            row_sum += tl.sum(tl.exp(logits - m_new) * mask, axis=0)

        s_new = tl.exp(m_old - m_new) * s_old + row_sum

        tl.store(M_ptr + row, m_new)
        tl.store(S_ptr + row, s_new)


    @triton.jit
    def _softmax_from_saved_lse_kernel(
        logits_chunk_ptr,   # [N, V_chunk]  in/out (overwritten with softmax)
        lse_ptr,            # [N] fp32 (final M + log(S))
        N,
        V_chunk,
        stride_n,
        stride_v,
        BLOCK_V: tl.constexpr,
    ):
        """softmax = exp(logits - lse). In-place over the passed chunk."""
        row = tl.program_id(0)
        if row >= N:
            return
        lse = tl.load(lse_ptr + row)
        for off in range(0, V_chunk, BLOCK_V):
            v_idx = off + tl.arange(0, BLOCK_V)
            mask = v_idx < V_chunk
            p = (
                logits_chunk_ptr + row * stride_n + v_idx * stride_v
            )
            logits = tl.load(p, mask=mask, other=0.0).to(tl.float32)
            sm = tl.exp(logits - lse) * mask
            tl.store(p, sm.to(tl.load(p, mask=mask, other=0.0).dtype), mask=mask)


# ---------------------------------------------------------------------------
# Autograd function
# ---------------------------------------------------------------------------


class _FusedLinearLogprobs(torch.autograd.Function):
    """Fused lm_head + log_softmax + gather(labels).

    Forward I/O:
        hidden:   [N, H]       (possibly autograd-tracked activations)
        W_lm:     [V, H]       (lm_head weight; may be tied to embed_tokens)
        labels:   [N]          (int64, ignore_index sentinel: -100)
        softcap:  Optional[float]
        chunk_size: int (vocab chunk size)

    Forward returns:
        lps:        [N] fp32 (log-probs at labels; 0.0 where label == -100)
        max_logit:  [N] fp32 (optional; same as saved M, for top-prob aux)

    Backward:
        grads with respect to ``hidden`` and ``W_lm`` only (softcap / chunk
        / labels are not differentiable).
    """

    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,
        W_lm: torch.Tensor,
        labels: torch.Tensor,
        softcap: float | None = None,
        chunk_size: int = 4096,
        ignore_index: int = -100,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert hidden.dim() == 2 and W_lm.dim() == 2
        assert hidden.shape[1] == W_lm.shape[1]
        assert labels.shape[0] == hidden.shape[0]

        N, H = hidden.shape
        V = W_lm.shape[0]

        device = hidden.device
        acc_dtype = torch.float32

        # Running logsumexp accumulators
        M = torch.full((N,), -float("inf"), device=device, dtype=acc_dtype)
        S = torch.zeros((N,), device=device, dtype=acc_dtype)

        # Label logit (collected as chunks pass the label's vocab column)
        label_logit = torch.zeros((N,), device=device, dtype=acc_dtype)
        valid_mask = labels != ignore_index  # ignored rows: we'll zero their lp

        # Forward: stream vocab chunks
        for v_start in range(0, V, chunk_size):
            v_end = min(v_start + chunk_size, V)
            W_chunk = W_lm[v_start:v_end]              # [V_chunk, H]
            logits_chunk = hidden @ W_chunk.T          # [N, V_chunk]

            if softcap is not None:
                logits_chunk = torch.tanh(logits_chunk / softcap) * softcap

            # Update running (M, S). Torch fallback — triton kernel above is
            # a future drop-in; benchmark before wiring in.
            chunk_max = logits_chunk.amax(dim=-1).to(acc_dtype)  # [N]
            new_M = torch.maximum(M, chunk_max)
            # Protect against -inf -> 0 exponent underflow (first chunk)
            scale_old = torch.exp(M - new_M)
            scale_old[M == -float("inf")] = 0.0
            chunk_sum = torch.exp(
                logits_chunk.to(acc_dtype) - new_M.unsqueeze(-1)
            ).sum(dim=-1)
            S = scale_old * S + chunk_sum
            M = new_M

            # Collect label logits that fell in this chunk
            in_chunk = (labels >= v_start) & (labels < v_end) & valid_mask
            if in_chunk.any():
                rows = in_chunk.nonzero(as_tuple=True)[0]
                cols = labels[rows] - v_start
                label_logit[rows] = logits_chunk[rows, cols].to(acc_dtype)

        lse = M + torch.log(S)                          # [N]
        lps = label_logit - lse                         # [N]
        lps = torch.where(valid_mask, lps, torch.zeros_like(lps))

        ctx.save_for_backward(hidden, W_lm, labels, lse, M)
        ctx.softcap = softcap
        ctx.chunk_size = chunk_size
        ctx.ignore_index = ignore_index
        ctx.V = V
        # Remember whether we need to compute grad_W at all. In the common
        # training setup (QLoRA + frozen base + LoRA on attn only), lm_head
        # is frozen — skipping grad_W avoids the [V, H] fp32 accumulator
        # (multi-GiB on 262K-vocab models) entirely.
        ctx.needs_grad_W = W_lm.requires_grad
        ctx.needs_grad_hidden = hidden.requires_grad

        return lps, M, label_logit

    @staticmethod
    def backward(ctx, grad_lps: torch.Tensor, grad_max_logit: torch.Tensor, grad_label_logit: torch.Tensor):
        """Recompute softmax per vocab chunk and accumulate grads.

        grad_logits[i, v] = (onehot(label_i)[v] - softmax[i, v]) * grad_lp_i
                            for rows with valid labels; zero otherwise.

        grad_hidden += grad_logits_chunk @ W_chunk         (over chunks)
        grad_W[V_slice] = grad_logits_chunk.T @ hidden     (per chunk, unique
                                                            slice so no accum)
        """
        hidden, W_lm, labels, lse, _M = ctx.saved_tensors
        softcap = ctx.softcap
        chunk_size = ctx.chunk_size
        ignore_index = ctx.ignore_index
        V = ctx.V

        N, H = hidden.shape
        valid_mask = labels != ignore_index
        grad_lp_eff = torch.where(
            valid_mask, grad_lps, torch.zeros_like(grad_lps)
        ).to(torch.float32)  # [N]

        # Only allocate the outputs we actually need. The full [V, H] fp32
        # grad_W accumulator is the dominant cost on 262K-vocab models
        # (~4.6 GiB for Gemma4) — skip it when lm_head is frozen.
        grad_hidden = (
            torch.zeros_like(hidden, dtype=torch.float32)
            if ctx.needs_grad_hidden
            else None
        )
        grad_W = (
            torch.zeros_like(W_lm)  # accumulate in W_lm's dtype, not fp32
            if ctx.needs_grad_W
            else None
        )
        # Cast hidden to fp32 once rather than per-chunk (same alloc, reused).
        hidden_fp32 = hidden.to(torch.float32) if ctx.needs_grad_W else None

        for v_start in range(0, V, chunk_size):
            v_end = min(v_start + chunk_size, V)
            W_chunk = W_lm[v_start:v_end]                 # [V_chunk, H] view

            # Recompute logits + softmax for this chunk (bf16 matmul, fp32 softmax)
            logits_chunk = hidden @ W_chunk.T             # [N, V_chunk]
            if softcap is not None:
                logits_chunk = torch.tanh(logits_chunk / softcap) * softcap

            softmax_chunk = torch.exp(
                logits_chunk.to(torch.float32) - lse.unsqueeze(-1)
            )  # [N, V_chunk] fp32

            grad_logits_chunk = -softmax_chunk * grad_lp_eff.unsqueeze(-1)

            in_chunk = (labels >= v_start) & (labels < v_end) & valid_mask
            if in_chunk.any():
                rows = in_chunk.nonzero(as_tuple=True)[0]
                cols = labels[rows] - v_start
                grad_logits_chunk[rows, cols] += grad_lp_eff[rows]

            if softcap is not None:
                # softcap chain rule: d/dx(tanh(x/c)*c) = 1 - tanh²(x/c).
                # logits_chunk here is already tanh(·)*softcap, so tanh(·) = logits_chunk/softcap.
                tanh_val = (logits_chunk / softcap).to(torch.float32)
                grad_logits_chunk = grad_logits_chunk * (1.0 - tanh_val * tanh_val)

            if grad_hidden is not None:
                # [N, V_chunk] fp32 × [V_chunk, H] bf16 → [N, H] fp32
                grad_hidden += grad_logits_chunk @ W_chunk.to(torch.float32)

            if grad_W is not None:
                # [V_chunk, N] fp32 × [N, H] fp32 → [V_chunk, H] fp32,
                # then cast to W_lm dtype for write-back.
                grad_W[v_start:v_end] = (
                    grad_logits_chunk.T @ hidden_fp32
                ).to(W_lm.dtype)

            # Free per-chunk temporaries explicitly (helps allocator reuse).
            del logits_chunk, softmax_chunk, grad_logits_chunk

        return (
            grad_hidden.to(hidden.dtype) if grad_hidden is not None else None,
            grad_W,
            None, None, None, None,
        )


def fused_linear_logprobs(
    hidden: torch.Tensor,
    W_lm: torch.Tensor,
    labels: torch.Tensor,
    softcap: float | None = None,
    chunk_size: int = 4096,
    ignore_index: int = -100,
    return_max_logit: bool = False,
    return_label_logit: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Compute per-token log-probabilities at ``labels`` without materializing
    the full ``[N, V]`` logits tensor.

    Args:
        hidden:     [N, H] activation tensor. Autograd-tracked.
        W_lm:       [V, H] lm_head weight. Autograd-tracked if trainable.
        labels:     [N] int64 label indices. ``ignore_index`` entries get
                    logprob 0.0 (and zero gradient).
        softcap:    Optional ``tanh(logits/cap)*cap`` clamp (Gemma4: 30.0).
        chunk_size: Vocab slice size. Higher = faster, more memory.
        ignore_index: Label sentinel for positions to skip (default -100).
        return_max_logit: If True, also returns per-row max logit (for the
                    ``aux_loss_top_prob_weight`` penalty, which needs
                    ``max(softmax)``; compute from this as
                    ``exp(max_logit - lse)``).

    Returns:
        ``lps`` with shape ``[N]``, or ``(lps, max_logit)`` if
        ``return_max_logit=True``.
    """
    lps, max_logit, label_logit = _FusedLinearLogprobs.apply(
        hidden, W_lm, labels, softcap, chunk_size, ignore_index
    )
    if return_max_logit and return_label_logit:
        return lps, max_logit, label_logit
    if return_max_logit:
        return lps, max_logit
    if return_label_logit:
        return lps, label_logit
    return lps


# ---------------------------------------------------------------------------
# Reference implementation (non-fused) — for unit-test parity checks.
# ---------------------------------------------------------------------------


def reference_linear_logprobs(
    hidden: torch.Tensor,
    W_lm: torch.Tensor,
    labels: torch.Tensor,
    softcap: float | None = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """The thing the fused kernel replaces. Materializes [N, V]."""
    logits = hidden @ W_lm.T
    if softcap is not None:
        logits = torch.tanh(logits / softcap) * softcap
    log_probs = F.log_softmax(logits.float(), dim=-1)
    valid_mask = labels != ignore_index
    safe_labels = torch.where(valid_mask, labels, torch.zeros_like(labels))
    lps = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return torch.where(valid_mask, lps, torch.zeros_like(lps))
