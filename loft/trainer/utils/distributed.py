import logging

import torch
from accelerate import PartialState
from transformers.utils.import_utils import is_torch_xpu_available

logger = logging.getLogger(__name__)


def get_kbit_device_map(model_parallel: bool = False) -> dict[str, int] | str | None:
    if torch.cuda.is_available() or is_torch_xpu_available():
        if model_parallel:
            return "balanced"
        return {"": PartialState().local_process_index}
    else:
        return None


def compute_balanced_device_map(
    meta_model,
    model_config,
    is_quantized_4bit: bool = False,
    is_quantized_8bit: bool = False,
    max_memory: dict | None = None,
    batch_size: int = 1,
    max_length: int = 2048,
    use_cce: bool = False,
    dtype_bytes: int = 2,
    use_lora: bool = False,
    lora_r: int = 16,
) -> dict[str, int]:
    """Compute a quantization-aware, training-balanced device map.

    ``from_pretrained`` with ``device_map="auto"`` computes module sizes using the
    model's original dtype (e.g. bf16), even when 4-bit quantization will dramatically
    reduce actual memory usage.  This leads to heavily unbalanced GPU distributions
    (e.g. 14 layers on GPU 0 vs 50 layers on GPU 1 for Qwen3.5-27B in 4-bit).

    This function fixes three issues:
    1. Uses ``torch.int8`` dtype for size estimation when 4-bit quantization is enabled,
       matching the actual storage format used by bitsandbytes.
    2. Sets per-GPU budgets based on the model's estimated size (to force balanced
       distribution), not physical GPU memory (which is too generous for quantized models
       and causes everything to land on GPU 0).
    3. Reserves headroom for training overhead not present at load time: logits/gradients
       on the last GPU, and LoRA adapter + optimizer states on all GPUs.

    Args:
        meta_model: Model on meta device (from ``init_empty_weights()``).
        model_config: HuggingFace model config.
        is_quantized_4bit: Whether 4-bit quantization is enabled.
        is_quantized_8bit: Whether 8-bit quantization is enabled.
        max_memory: Optional user-specified per-GPU memory budgets.
            If None, computed automatically with training overhead awareness.
        batch_size: Per-device training batch size.
        max_length: Maximum sequence length.
        use_cce: Whether CCE is enabled (reduces logits memory).
        dtype_bytes: Bytes per element for the model dtype (2 for bf16/fp16).
        use_lora: Whether LoRA/QLoRA training is enabled.
        lora_r: LoRA rank (used to estimate adapter + optimizer memory).

    Returns:
        Explicit device map dict mapping module names to GPU indices.
    """
    from accelerate import infer_auto_device_map

    num_devices = torch.cuda.device_count()
    if num_devices < 2:
        return None

    # Use int8 for 4-bit models (bitsandbytes stores 4-bit weights as packed int8),
    # int8 for 8-bit models, otherwise use the model's dtype.
    if is_quantized_4bit or is_quantized_8bit:
        map_dtype = torch.int8
    else:
        map_dtype = None  # let accelerate infer from model

    # Compute max_memory if not user-specified.
    if max_memory is None:
        if is_quantized_4bit:
            est_to_real = 0.55
        elif is_quantized_8bit:
            # LLM.int8() keeps outlier features in fp16 and stores quantization
            # state (scales, absmax) alongside int8 weights. ~10-15% overhead
            # over a pure int8 estimate.
            est_to_real = 1.15
        else:
            est_to_real = 1.0

        max_memory = _find_optimal_split(
            meta_model=meta_model,
            model_config=model_config,
            num_devices=num_devices,
            est_to_real=est_to_real,
            map_dtype=map_dtype,
            batch_size=batch_size,
            max_length=max_length,
            use_cce=use_cce,
            dtype_bytes=dtype_bytes,
            use_lora=use_lora,
            lora_r=lora_r,
        )
    else:
        # Normalize YAML string keys to int for GPU indices.
        max_memory = {
            int(k) if isinstance(k, str) and k.isdigit() else k: v
            for k, v in max_memory.items()
        }

    no_split = getattr(meta_model, "_no_split_modules", None) or []

    kwargs = {}
    if map_dtype is not None:
        kwargs["dtype"] = map_dtype
    if max_memory is not None:
        kwargs["max_memory"] = max_memory

    device_map = infer_auto_device_map(
        meta_model,
        no_split_module_classes=no_split,
        **kwargs,
    )

    # Fix layers that got split across GPUs despite no_split_module_classes.
    # This can happen when budgets are tight and accelerate can't fit a whole
    # layer on the current GPU. We move all sub-modules of a split layer to
    # the next GPU in the pipeline (the earlier GPU was full).
    device_map = _fix_split_layers(device_map)

    # Log summary.
    devs: dict[str, list[str]] = {}
    for mod, dev in device_map.items():
        devs.setdefault(str(dev), []).append(mod)
    parts = []
    for d in sorted(devs.keys()):
        mods = devs[d]
        layer_ids = set()
        for m in mods:
            ps = m.split(".")
            for j, p in enumerate(ps):
                if p == "layers" and j + 1 < len(ps) and ps[j + 1].isdigit():
                    layer_ids.add(int(ps[j + 1]))
        if layer_ids:
            parts.append(f"GPU {d}: layers {min(layer_ids)}-{max(layer_ids)} ({len(layer_ids)} layers)")
        else:
            parts.append(f"GPU {d}: {len(mods)} modules")
    logger.info(f"Computed balanced device_map: {' | '.join(parts)}")
    if max_memory is not None:
        logger.info(f"  max_memory={max_memory}")

    return device_map


def _get_config_value(config, attr, fallback=None):
    """Get attribute from config, falling through to text_config for composite models."""
    val = getattr(config, attr, None)
    if val is None and hasattr(config, "text_config"):
        val = getattr(config.text_config, attr, None)
    return val if val is not None else fallback


def _estimate_lora_training_overhead(
    model_config,
    lora_r: int = 16,
    dtype_bytes: int = 2,
) -> int:
    """Estimate total real bytes for LoRA adapters + AdamW optimizer states + gradients.

    Assumes LoRA targets all 7 standard linear layers per transformer block:
    q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj.

    Each LoRA pair (A, B) adds ``(in_features + out_features) * r`` trainable params.
    During training each param needs: weight (dtype_bytes) + Adam momentum (fp32) +
    Adam variance (fp32) + gradient (fp32) = dtype_bytes + 12 bytes.
    """
    hidden = _get_config_value(model_config, "hidden_size", 4096)
    intermediate = _get_config_value(model_config, "intermediate_size", hidden * 4)
    num_layers = _get_config_value(model_config, "num_hidden_layers", 32)
    num_kv_heads = _get_config_value(model_config, "num_key_value_heads")
    head_dim = _get_config_value(model_config, "head_dim")

    # KV projection dimension (accounts for GQA where kv_dim < hidden)
    if num_kv_heads is not None and head_dim is not None:
        kv_dim = num_kv_heads * head_dim
    else:
        kv_dim = hidden  # assume MHA

    # Attention LoRA params per layer:
    #   q_proj: (hidden + hidden) * r,  o_proj: (hidden + hidden) * r
    #   k_proj: (hidden + kv_dim) * r,  v_proj: (hidden + kv_dim) * r
    attn_params_per_layer = (4 * hidden + 2 * kv_dim) * lora_r

    # MLP LoRA params per layer:
    #   gate_proj, up_proj, down_proj: each (hidden + intermediate) * r
    mlp_params_per_layer = 3 * (hidden + intermediate) * lora_r

    total_params = (attn_params_per_layer + mlp_params_per_layer) * num_layers

    # Per param: weight + optimizer momentum + optimizer variance + gradient
    bytes_per_param = dtype_bytes + 4 + 4 + 4
    total_bytes = total_params * bytes_per_param

    logger.info(
        f"Estimated LoRA training overhead: {total_bytes / (1024**3):.2f} GiB "
        f"({total_params / 1e6:.1f}M params × {bytes_per_param}B, "
        f"r={lora_r}, layers={num_layers}, hidden={hidden}, kv_dim={kv_dim}, "
        f"intermediate={intermediate})"
    )
    return total_bytes


def _fix_split_layers(device_map: dict) -> dict:
    """Ensure no transformer layer is split across multiple GPUs.

    ``infer_auto_device_map`` may split a layer's sub-modules across GPUs when
    the per-GPU budget is tight (e.g. vision encoder + embeddings consume more
    space than expected).  This post-processes the map to keep each layer whole
    by replacing sub-module entries with a single layer-level entry pointing to
    the next GPU in the pipeline (the earlier GPU ran out of budget space).

    Using a layer-level entry (instead of per-sub-module entries) is critical:
    accelerate places dispatch hooks at the granularity of device_map keys.
    A layer-level entry ensures the hook moves the *entire input* to the target
    GPU before the layer's forward runs, which is necessary for residual/skip
    connections that reference the input before any sub-module call.
    """
    import re

    # Collect per-layer device assignments.
    # Key: (prefix_up_to_layers, layer_idx) → {device_str: [module_names]}
    layer_pattern = re.compile(r"^(.+\.layers)\.(\d+)")
    layer_devices: dict[tuple, dict[str, list[str]]] = {}
    for mod, dev in device_map.items():
        m = layer_pattern.match(mod)
        if m:
            key = (m.group(1), int(m.group(2)))
            layer_devices.setdefault(key, {}).setdefault(str(dev), []).append(mod)

    # Find layers whose sub-modules span multiple devices.
    split_layers = {k: v for k, v in layer_devices.items() if len(v) > 1}
    if not split_layers:
        return device_map

    fixed = dict(device_map)
    for (prefix, layer_idx), dev_mods in split_layers.items():
        # Move the whole layer to the numerically highest (later) GPU,
        # since the earlier GPU ran out of budget space.
        gpu_devs = [int(d) for d in dev_mods if d.isdigit()]
        if not gpu_devs:
            continue
        target = max(gpu_devs)
        layer_key = f"{prefix}.{layer_idx}"
        logger.info(
            f"Fixing split layer {layer_key}: "
            f"collapsing to GPU {target} (was on {sorted(dev_mods.keys())})"
        )
        # Remove all sub-module entries for this layer.
        for mods in dev_mods.values():
            for mod in mods:
                del fixed[mod]
        # Add a single layer-level entry so accelerate hooks the layer itself.
        fixed[layer_key] = target

    return fixed


def _find_optimal_split(
    meta_model,
    model_config,
    num_devices: int,
    est_to_real: float = 1.0,
    map_dtype=None,
    batch_size: int = 1,
    max_length: int = 2048,
    use_cce: bool = False,
    dtype_bytes: int = 2,
    use_lora: bool = False,
    lora_r: int = 16,
) -> dict[int, int]:
    """Compute per-GPU max_memory budgets by finding the optimal layer split in real bytes.

    The previous approach (``_compute_training_aware_max_memory``) tried to subtract
    runtime training overhead (logits, gradients) from estimation-unit budgets.  For
    4-bit quantized models with large vocabularies, the ``est_to_real`` conversion
    amplifies the overhead beyond the per-GPU model budget, making it go negative.

    This function avoids that by working entirely in real bytes:

    1. Gets exact per-module sizes via ``compute_module_sizes`` (estimation units).
    2. Discovers model structure (layers, prefix, suffix) automatically.
    3. Computes per-layer real runtime cost (weight + LoRA + activations).
    4. Computes fixed per-GPU overhead in real bytes (embeddings, logits, CUDA).
    5. Scans all possible contiguous splits to maximize min(headroom) across GPUs.
    6. Converts the optimal split back to estimation-unit budgets.

    Returns:
        Dict mapping GPU index to max_memory budget in estimation bytes (int).
    """
    import re
    from accelerate.utils.modeling import compute_module_sizes

    # --- Step 1: Get exact module sizes in estimation units ---
    module_sizes = compute_module_sizes(meta_model, dtype=map_dtype)
    total_est = module_sizes[""]

    # --- Step 2: Discover model structure ---
    # Find the transformer layer group (e.g. "model.layers")
    layer_pattern = re.compile(r"^(.+\.layers)\.(\d+)$")
    layer_groups: dict[str, list[tuple[int, int]]] = {}
    for name, size in module_sizes.items():
        m = layer_pattern.match(name)
        if m:
            prefix = m.group(1)
            idx = int(m.group(2))
            layer_groups.setdefault(prefix, []).append((idx, size))

    if not layer_groups:
        # No layer structure found — fall back to even split.
        per_gpu = int(total_est * 1.12 / num_devices)
        logger.warning(
            f"No transformer layers found in model — falling back to even split: "
            f"{per_gpu / (1024**3):.1f} GiB per GPU"
        )
        return {i: per_gpu for i in range(num_devices)}

    # Use the largest layer group (the main transformer stack).
    main_prefix = max(layer_groups, key=lambda k: len(layer_groups[k]))
    layers = sorted(layer_groups[main_prefix])  # [(idx, est_size), ...]
    num_layers = len(layers)

    # Suffix: lm_head (always last in the sequential pipeline).
    # Use module_sizes for exact value; fall back to 0 if not found.
    suffix_est = module_sizes.get("lm_head", 0)

    # Prefix: everything that isn't layers or lm_head.
    layers_est = sum(s for _, s in layers)
    prefix_est = total_est - layers_est - suffix_est

    # --- Step 3: Compute per-layer real runtime cost ---
    hidden = _get_config_value(model_config, "hidden_size", 4096)
    intermediate = _get_config_value(model_config, "intermediate_size", hidden * 4)
    vocab_size = _get_config_value(model_config, "vocab_size", 32000)

    # Use mean layer estimation size (handles non-uniform layers like GDN vs attention).
    mean_layer_est = layers_est / num_layers
    per_layer_weight_real = mean_layer_est * est_to_real

    # LoRA adapter + optimizer + gradient overhead per layer.
    if use_lora:
        num_kv_heads = _get_config_value(model_config, "num_key_value_heads")
        head_dim = _get_config_value(model_config, "head_dim")
        kv_dim = (num_kv_heads * head_dim) if (num_kv_heads and head_dim) else hidden
        attn_lora = (4 * hidden + 2 * kv_dim) * lora_r
        mlp_lora = 3 * (hidden + intermediate) * lora_r
        bytes_per_param = dtype_bytes + 12  # weight + adam_m + adam_v + grad
        per_layer_lora_real = (attn_lora + mlp_lora) * bytes_per_param
    else:
        per_layer_lora_real = 0

    # Activation memory per layer during training with gradient checkpointing.
    # GC stores only the input hidden state between layers, but during backward it
    # recomputes each layer's full forward pass. The peak transient memory includes:
    #   - Stored input: batch * seq * hidden * dtype_bytes
    #   - MLP intermediate: batch * seq * intermediate * dtype_bytes (gate/up before down_proj)
    #   - Attention working memory (flash attention keeps this small)
    #   - Gradient tensors for the layer's output
    # The MLP intermediate is typically the dominant peak allocation.
    per_layer_act_stored = batch_size * max_length * hidden * dtype_bytes
    per_layer_act_peak = batch_size * max_length * intermediate * dtype_bytes
    per_layer_act_real = per_layer_act_stored + per_layer_act_peak

    per_layer_total_real = per_layer_weight_real + per_layer_lora_real + per_layer_act_real

    # --- Step 4: Compute fixed per-GPU overhead in real bytes ---
    CUDA_RESERVE = 2.0 * (1024 ** 3)

    prefix_real = prefix_est * est_to_real  # embeddings, vision encoder, etc.
    suffix_real = suffix_est * est_to_real   # lm_head weights

    effective_seq = 128 if use_cce else max_length
    logits_real = batch_size * effective_seq * vocab_size * dtype_bytes   # bf16/fp16
    grad_logits_real = batch_size * effective_seq * vocab_size * 4        # fp32 gradient
    logits_overhead_real = logits_real + grad_logits_real

    # --- Step 5: Find optimal contiguous split ---
    gpu_mem = [torch.cuda.get_device_properties(i).total_memory for i in range(num_devices)]
    last_gpu = num_devices - 1

    # Cap: last GPU must get at least 1 layer so infer_auto_device_map produces
    # a multi-GPU split.  Without this, the budget for GPU 0 would equal the total
    # model, causing everything (including lm_head) to land on GPU 0.
    max_layers_gpu0 = num_layers - 1

    if num_devices == 2:
        # Linear scan over all possible split points.
        # For model_parallel (sequential execution), we prefer maximizing layers
        # on GPU 0 to minimize PCIe transfers. Among all splits with positive
        # headroom, pick the one with the most layers on GPU 0.
        best_split = num_layers // 2
        best_min_headroom = -float("inf")
        feasible_splits = []

        for s in range(max_layers_gpu0 + 1):
            g0_usage = CUDA_RESERVE + prefix_real + s * per_layer_total_real
            g1_usage = (
                CUDA_RESERVE + suffix_real + logits_overhead_real
                + (num_layers - s) * per_layer_total_real
            )
            min_hr = min(gpu_mem[0] - g0_usage, gpu_mem[1] - g1_usage)
            if min_hr > best_min_headroom:
                best_min_headroom = min_hr
                best_split = s
            if min_hr > 0:
                feasible_splits.append((s, min_hr))

        # Among feasible splits (positive headroom on both GPUs), prefer
        # maximizing layers on GPU 0 — each cross-GPU transfer in sequential
        # model_parallel adds PCIe latency, so fewer layers on GPU 1 is faster.
        if feasible_splits:
            best_split = max(feasible_splits, key=lambda x: x[0])[0]

        layer_counts = [best_split, num_layers - best_split]
    else:
        # N > 2 GPUs: equalize headroom across GPUs.
        # Compute available real memory per GPU (after fixed overhead).
        available = [gpu_mem[i] - CUDA_RESERVE for i in range(num_devices)]
        available[0] -= prefix_real
        available[last_gpu] -= suffix_real + logits_overhead_real

        # Target: each GPU should have the same headroom after layer assignment.
        total_available = sum(available)
        total_layer_cost = num_layers * per_layer_total_real
        target_headroom = (total_available - total_layer_cost) / num_devices

        layer_counts = []
        remaining_layers = num_layers
        for i in range(num_devices):
            if i == num_devices - 1:
                layer_counts.append(remaining_layers)
            else:
                target = (available[i] - target_headroom) / per_layer_total_real
                actual = max(0, min(remaining_layers, round(target)))
                layer_counts.append(actual)
                remaining_layers -= actual

        # Ensure last GPU has at least 1 layer.
        if layer_counts[-1] == 0 and layer_counts[-2] > 0:
            layer_counts[-2] -= 1
            layer_counts[-1] = 1

    # --- Step 6: Convert to estimation-unit budgets ---
    # infer_auto_device_map reserves space for the largest remaining unplaced
    # module at each GPU boundary.  For large-vocab models, this is the lm_head
    # (~970 MiB for 248K vocab).  Each non-last GPU's budget must include this
    # reservation on top of the assigned module sizes.
    max_single_module_est = max(
        max((s for _, s in layers), default=0),  # largest layer
        suffix_est,  # lm_head
    )

    cumulative = 0
    max_memory = {}
    MARGIN = 1.02  # 2% margin for rounding and tied-param bookkeeping

    for i in range(num_devices):
        start = cumulative
        end = cumulative + layer_counts[i]
        gpu_est = sum(layers[j][1] for j in range(start, min(end, num_layers)))
        if i == 0:
            gpu_est += prefix_est
        if i == last_gpu:
            # Last GPU: generous budget — everything that doesn't fit elsewhere
            # lands here.  Use remaining model size with ample margin.
            gpu_est += suffix_est
            gpu_est = int(gpu_est * 2.0)  # 2× for headroom
        else:
            # Non-last GPUs: add reservation for the largest unplaced module
            # (infer_auto_device_map won't place a module if the remaining budget
            # can't hold the largest remaining module as a safety margin).
            gpu_est += max_single_module_est
            gpu_est = int(gpu_est * MARGIN)
        max_memory[i] = gpu_est
        cumulative = end

    # Log the decision.
    split_desc = " | ".join(
        f"GPU {i}: {layer_counts[i]} layers" for i in range(num_devices)
    )
    headroom_desc = []
    cumulative = 0
    for i in range(num_devices):
        usage = CUDA_RESERVE
        if i == 0:
            usage += prefix_real
        if i == last_gpu:
            usage += suffix_real + logits_overhead_real
        usage += layer_counts[i] * per_layer_total_real
        headroom = gpu_mem[i] - usage
        headroom_desc.append(f"GPU {i}: {headroom / (1024**3):.1f}G free")
        cumulative += layer_counts[i]

    split_msg = (
        f"Optimal layer split for {num_devices} GPUs: {split_desc} "
        f"(per_layer_real={per_layer_total_real / (1024**2):.0f}MiB, "
        f"logits_overhead={logits_overhead_real / (1024**3):.2f}GiB, "
        f"headroom: {', '.join(headroom_desc)})"
    )
    budget_msg = f"  est budgets: {max_memory}"
    logger.info(split_msg)
    logger.info(budget_msg)
    print(split_msg)
    print(budget_msg)

    return max_memory


@torch.no_grad()
def get_global_statistics(
    accelerator, xs: torch.Tensor, mask=None, device="cpu"
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Computes element-wise mean and variance of the tensor across processes. Reference:
    https://github.com/OpenLMLab/MOSS-RLHF/blob/40b91eb2f2b71b16919addede0341d2bef70825d/utils.py#L57C1-L73C75
    """
    xs = xs.to(accelerator.device)
    sum_and_count = torch.tensor([xs.sum(), (xs.numel() if mask is None else mask.sum())], device=xs.device)
    sum_and_count = accelerator.reduce(sum_and_count)
    global_sum, count = sum_and_count
    global_mean = global_sum / count

    sum_var = torch.sum(((xs - global_mean) ** 2).mul(1 if mask is None else mask))
    sum_var = accelerator.reduce(sum_var)
    global_var = sum_var / count

    return global_mean.to(device), global_var.to(device), count.item()
