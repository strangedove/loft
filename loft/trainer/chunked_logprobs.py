"""Vocab-chunked per-token log probability computation for DPO training.

Computes per-token log probs without materializing the full [batch, seq, vocab]
logits tensor.  For 248K-vocab models, this reduces peak backward memory from
~8.6 GiB (logits + fp32 gradient) to ~140 MiB (one chunk at a time).

The tradeoff is speed: ~60 smaller matmuls instead of one large one, plus
recomputation during backward.  Typically 10-20% slower per training step.

Usage:
    from loft.trainer.chunked_logprobs import patch_dpo_chunked_logprobs
    patch_dpo_chunked_logprobs(chunk_size=4096)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import logging

logger = logging.get_logger(__name__)


class ChunkedPerTokenLogProbs(torch.autograd.Function):
    """Compute per-token log P(label|context) by chunking the vocab dimension.

    Forward:  for each vocab chunk, compute logits = hidden @ weight_chunk.T,
              accumulate logsumexp incrementally, gather label logits.
    Backward: recompute logits chunks, compute softmax gradient from saved
              logsumexp, accumulate grad_hidden.  Never allocates [batch, seq, vocab].
    """

    @staticmethod
    def forward(ctx, hidden_states, weight, labels, chunk_size, ignore_index):
        """
        Args:
            hidden_states: [batch, seq, hidden_dim]
            weight: [vocab_size, hidden_dim] — lm_head weight (may be on different device)
            labels: [batch, seq] — target token ids
            chunk_size: vocab chunk size
            ignore_index: label value to ignore (typically 0 for masked positions)

        Returns:
            per_token_logps: [batch, seq] — log P(label_t | context) per position
            gathered_logits: [batch, seq] — raw logit at label position (for logging)
        """
        device = weight.device
        dtype = hidden_states.dtype
        hidden_local = hidden_states.to(device)
        labels_local = labels.to(device)

        batch, seq = labels_local.shape
        vocab_size = weight.shape[0]

        # Accumulators for incremental logsumexp
        max_logit = torch.full((batch, seq), float('-inf'), device=device, dtype=torch.float32)
        sum_exp = torch.zeros((batch, seq), device=device, dtype=torch.float32)
        gathered_logits = torch.zeros((batch, seq), device=device, dtype=torch.float32)

        for v_start in range(0, vocab_size, chunk_size):
            v_end = min(v_start + chunk_size, vocab_size)
            logits_chunk = hidden_local @ weight[v_start:v_end].t()
            logits_f32 = logits_chunk.float()

            # Gather label logits in this vocab range
            mask = (labels_local >= v_start) & (labels_local < v_end) & (labels_local != ignore_index)
            if mask.any():
                local_labels = (labels_local[mask] - v_start).unsqueeze(-1)
                gathered_logits[mask] = logits_f32[mask].gather(-1, local_labels).squeeze(-1)

            # Incremental numerically-stable logsumexp
            chunk_max = logits_f32.max(dim=-1).values
            new_max = torch.maximum(max_logit, chunk_max)
            sum_exp = (
                sum_exp * (max_logit - new_max).exp()
                + (logits_f32 - new_max.unsqueeze(-1)).exp().sum(dim=-1)
            )
            max_logit = new_max

            del logits_chunk, logits_f32

        logsumexp = max_logit + sum_exp.log()
        per_token_logps = gathered_logits - logsumexp

        # Zero out ignored positions
        per_token_logps[labels_local == ignore_index] = 0.0
        gathered_logits[labels_local == ignore_index] = 0.0

        # Save for backward — NOT the full logits tensor
        # Track the output device explicitly — activation offloading may move
        # hidden_states to CPU between forward and backward, but autograd
        # expects gradients on the same device as the forward outputs.
        output_device = hidden_states.device
        ctx.save_for_backward(hidden_states, weight, labels, logsumexp.to(output_device))
        ctx.chunk_size = chunk_size
        ctx.ignore_index = ignore_index
        ctx.output_device = output_device

        return per_token_logps.to(output_device), gathered_logits.to(output_device)

    @staticmethod
    def backward(ctx, grad_output, _grad_gathered):
        """Gradient flows through per_token_logps only (gathered_logits is detached)."""
        hidden_states, weight, labels, logsumexp = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        ignore_index = ctx.ignore_index

        device = weight.device
        dtype = hidden_states.dtype
        hidden_local = hidden_states.to(device)
        labels_local = labels.to(device)
        logsumexp_local = logsumexp.to(device)
        grad_local = grad_output.to(device)

        # Zero gradient for ignored positions
        grad_local = grad_local * (labels_local != ignore_index).float()

        vocab_size = weight.shape[0]
        grad_hidden = torch.zeros_like(hidden_local)

        for v_start in range(0, vocab_size, chunk_size):
            v_end = min(v_start + chunk_size, vocab_size)
            weight_chunk = weight[v_start:v_end]

            # Recompute logits chunk (not saved from forward)
            logits_chunk = hidden_local @ weight_chunk.t()

            # Softmax for this chunk via saved logsumexp
            softmax_chunk = (logits_chunk.float() - logsumexp_local.unsqueeze(-1)).exp()

            # d(logps)/d(logits) = one_hot(label) - softmax
            # So grad_logits = grad_output * (one_hot(label) - softmax)
            #                = -grad_output * softmax  +  grad_output * one_hot(label)
            grad_logits = -grad_local.unsqueeze(-1) * softmax_chunk

            # Add +grad_output at label positions within this chunk
            mask = (labels_local >= v_start) & (labels_local < v_end) & (labels_local != ignore_index)
            if mask.any():
                local_labels = (labels_local[mask] - v_start).unsqueeze(-1)
                grad_logits[mask] = grad_logits[mask].scatter_add(
                    -1, local_labels, grad_local[mask].unsqueeze(-1).float()
                )

            # Accumulate gradient for hidden_states
            grad_hidden += grad_logits.to(dtype) @ weight_chunk

            del logits_chunk, softmax_chunk, grad_logits

        # 5 inputs to forward: hidden_states, weight, labels, chunk_size, ignore_index
        # Use the saved output_device — hidden_states.device may differ after
        # activation offloading moves tensors to CPU and back.
        return grad_hidden.to(ctx.output_device), None, None, None, None


def chunked_per_token_logps(hidden_states, weight, labels, chunk_size=4096, ignore_index=0):
    """Convenience wrapper. Returns (per_token_logps, gathered_logits)."""
    return ChunkedPerTokenLogProbs.apply(hidden_states, weight, labels, chunk_size, ignore_index)


def _get_lm_head(model):
    """Navigate PEFT/accelerate wrappers to find the lm_head module and its weight.

    Returns (base_module, lm_head_module, lm_head_weight) where base_module is
    the object on which .lm_head can be swapped.
    """
    m = model
    # Accelerate wrapping
    while hasattr(m, 'module'):
        m = m.module
    # PEFT wrapping: PeftModel → LoraModel → original CausalLM
    if hasattr(m, 'base_model'):
        if hasattr(m.base_model, 'model'):
            base = m.base_model.model
        else:
            base = m.base_model
    else:
        base = m
    lm_head = base.lm_head
    return base, lm_head, lm_head.weight


def patch_dpo_chunked_logprobs(chunk_size=4096):
    """Monkey-patch DPOTrainer.concatenated_forward to use vocab-chunked log probs.

    Instead of computing full [batch, seq, vocab] logits, this:
    1. Bypasses the lm_head during model forward (returns hidden_states directly)
    2. Computes per-token log probs by chunking the vocab-dim matmul
    3. Never allocates more than [batch, seq, chunk_size] at a time

    Args:
        chunk_size: Number of vocab entries per chunk. 4096 is a good default.
    """
    import trl.trainer.dpo_trainer as _dpo_mod

    logger.info(f"Patching DPOTrainer.concatenated_forward for vocab-chunked log probs (chunk_size={chunk_size})")

    def _chunked_concatenated_forward(self, model, batch):
        """concatenated_forward with vocab-chunked log prob computation.

        Based on TRL's DPOTrainer.concatenated_forward but replaces the lm_head
        forward + cross_entropy log prob computation with a chunked version that
        never materializes [batch, seq, vocab_size].
        """
        num_examples = batch["prompt_input_ids"].shape[0]
        concatenated_batch = self.concatenated_inputs(batch, padding_value=self.padding_value)

        model_kwargs = {}
        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True
        if "pixel_values" in concatenated_batch:
            model_kwargs["pixel_values"] = concatenated_batch["pixel_values"]
        if "pixel_attention_mask" in concatenated_batch:
            model_kwargs["pixel_attention_mask"] = concatenated_batch["pixel_attention_mask"]

        prompt_input_ids = concatenated_batch["prompt_input_ids"]
        prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
        completion_input_ids = concatenated_batch["completion_input_ids"]
        completion_attention_mask = concatenated_batch["completion_attention_mask"]

        if self.is_encoder_decoder:
            raise NotImplementedError("Chunked log probs not supported for encoder-decoder models")

        # ── Input preparation (identical to original) ──
        input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
        attention_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim=1)
        loss_mask = torch.cat(
            (torch.zeros_like(prompt_attention_mask), completion_attention_mask), dim=1,
        )

        # Flush left to reduce memory
        for i in range(attention_mask.size(0)):
            first_one_idx = torch.nonzero(attention_mask[i])[0].item()
            input_ids[i] = torch.roll(input_ids[i], shifts=-first_one_idx)
            attention_mask[i] = torch.roll(attention_mask[i], shifts=-first_one_idx)
            loss_mask[i] = torch.roll(loss_mask[i], shifts=-first_one_idx)

        empty_cols = torch.sum(attention_mask, dim=0) == 0
        first_empty_col = torch.nonzero(empty_cols)[0].item() if empty_cols.any() else attention_mask.size(1) + 1
        input_ids = input_ids[:, : first_empty_col - 1]
        attention_mask = attention_mask[:, : first_empty_col - 1]
        loss_mask = loss_mask[:, : first_empty_col - 1]

        if self.args.max_length is not None:
            input_ids = input_ids[:, : self.args.max_length]
            attention_mask = attention_mask[:, : self.args.max_length]
            loss_mask = loss_mask[:, : self.args.max_length]

        # ── Model forward with lm_head bypassed ──
        base, lm_head, lm_head_weight = _get_lm_head(model)
        saved_lm_head = base.lm_head
        base.lm_head = nn.Identity()
        try:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, **model_kwargs)
            # With Identity lm_head, "logits" is actually the hidden_states
            hidden_states = outputs.logits
        finally:
            base.lm_head = saved_lm_head

        # Offset by one (standard causal LM alignment)
        hidden_states = hidden_states[:, :-1, :]
        labels = input_ids[:, 1:].clone()
        loss_mask = loss_mask[:, 1:].bool()

        # Handle LLaVA-style models where hidden_states includes image tokens
        if hidden_states.shape[1] != labels.shape[1]:
            seq_len = labels.shape[1]
            hidden_states = hidden_states[:, -seq_len:]

        # ── Chunked log prob computation ──
        labels[~loss_mask] = 0  # dummy token for masked positions
        per_token_logps, gathered_logits = chunked_per_token_logps(
            hidden_states, lm_head_weight, labels, chunk_size=chunk_size, ignore_index=0,
        )

        # Move to same device if needed (model_parallel)
        if per_token_logps.device != loss_mask.device:
            per_token_logps = per_token_logps.to(loss_mask.device)
            gathered_logits = gathered_logits.to(loss_mask.device)

        # Mean logits for logging (from gathered logits at label positions)
        _mean_chosen_logits = gathered_logits[:num_examples][loss_mask[:num_examples]].mean().detach()
        _mean_rejected_logits = gathered_logits[num_examples:][loss_mask[num_examples:]].mean().detach()
        del gathered_logits, hidden_states

        per_token_logps[~loss_mask] = 0
        all_logps = per_token_logps.sum(-1)

        # ── Build output dict ──
        output = {}

        if self.use_weighting:
            raise NotImplementedError(
                "WPO weighting requires full logits and is incompatible with chunked log probs. "
                "Disable use_chunked_dpo or use_weighting."
            )
        if self.args.rpo_alpha is not None:
            raise NotImplementedError(
                "RPO loss requires full logits and is incompatible with chunked log probs. "
                "Disable use_chunked_dpo or set rpo_alpha=None."
            )

        if self.loss_type == "ipo":
            all_logps = all_logps / loss_mask.sum(-1)

        output["chosen_logps"] = all_logps[:num_examples]
        output["rejected_logps"] = all_logps[num_examples:]
        output["mean_chosen_logits"] = _mean_chosen_logits
        output["mean_rejected_logits"] = _mean_rejected_logits

        if self.aux_loss_enabled:
            output["aux_loss"] = outputs.aux_loss

        return output

    _dpo_mod.DPOTrainer.concatenated_forward = _chunked_concatenated_forward
    logger.info("DPOTrainer.concatenated_forward patched for chunked log probs")
