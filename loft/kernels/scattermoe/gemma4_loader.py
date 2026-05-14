"""Load Gemma4 26B-A4B with fused 3D expert tensors NF4-quantized via bnb parametrize.

Keeps the stock architecture (NOT unfused), so scattermoe can operate on the
fused 3D expert tensors directly. The 3D gate_up_proj + down_proj on each
Gemma4TextExperts module are quantized in-place via bitsandbytes'
parametrize API, which transformers does not invoke automatically.

Produces:
  - model with parametrized experts (NF4)
  - scattermoe ExpertsInterface registered
  - ready for PEFT LoRA on q/k/v/o_proj (attention) + gate_up_proj/down_proj (experts)

Environment variables:
  LOFT_GEMMA4_NF4_CACHE   — path to pre-quantized fused cache directory
                             (default: ~/models/gemma-4-26B-A4B-it-nf4-fused-scattermoe)
  GEMMA4_ATTN_IMPL        — attention implementation for cached loads (default: sdpa)
  LOFT_SINGLE_GPU         — set to 1 to keep everything on cuda:0
  LOFT_FIRST_TO_GPU0      — number of decoder layers on cuda:0 in 2-GPU split (default: 10)
"""
from __future__ import annotations
import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


_DEFAULT_CACHE_DIR = os.path.expanduser("~/models/gemma-4-26B-A4B-it-nf4-fused-scattermoe")
CACHE_DIR = os.environ.get("LOFT_GEMMA4_NF4_CACHE", _DEFAULT_CACHE_DIR)


def register_scattermoe():
    """Register scattermoe ExpertsInterface for Gemma4."""
    from loft.kernels.scattermoe.gemma4_experts import register_scattermoe_experts
    register_scattermoe_experts()


def save_quantized_cache(model, cache_dir: str = CACHE_DIR):
    """Save the NF4-quantized model + quant_state to a persistent cache.

    Storage format: safetensors for weights + a pickled dict of quant_states
    keyed by `{module_path}.{param_name}`. On load, we rebuild the model
    skeleton, then call replace_parameter_4bit_prequantized for each entry.
    """
    import pickle
    from safetensors.torch import save_file

    os.makedirs(cache_dir, exist_ok=True)

    # Collect quantized packed data + quant_states
    packed_tensors = {}  # {module_path.param_name: packed_uint8}
    quant_states = {}    # {module_path.param_name: qs.as_dict(packed=True)}
    other_state = {}     # everything else from state_dict

    # First, iterate parametrized modules to grab packed data + qs
    param_keys = set()
    for mod_name, module in model.named_modules():
        if not hasattr(module, "parametrizations"):
            continue
        for pname in list(module.parametrizations.keys()):
            param_list = module.parametrizations[pname]
            parametrization = param_list[0]
            qs = getattr(parametrization, "quant_state", None)
            if qs is None:
                continue
            packed_key = f"{mod_name}.{pname}"
            packed_tensors[packed_key] = param_list.original.detach().cpu()
            # QuantState.as_dict(packed=True) returns a dict w/ quant_state.*
            # prefix — we'll store without prefix and re-prefix on load
            quant_states[packed_key] = qs.as_dict(packed=True)
            param_keys.add(packed_key)

    # Other state_dict entries — dedupe tied tensors (e.g. lm_head.weight ties
    # to embed_tokens.weight), and skip anything that relates to a parametrized
    # expert param (bnb's state_dict post-hook renames parametrizations.X.original
    # back to X and adds X.quant_state.* entries — we handle those separately).
    sd = model.state_dict()
    seen_storage = set()
    tied_pairs = {}  # {dropped_key: kept_key}
    # Build exclusion set: for each captured packed key "mod.gate_up_proj",
    # exclude the key itself + any "mod.gate_up_proj.*" suffix.
    excluded_prefixes = set()
    for k in param_keys:
        excluded_prefixes.add(k)       # exact match
        excluded_prefixes.add(k + ".")  # suffix match

    def _is_excluded(key: str) -> bool:
        if ".parametrizations." in key:
            return True
        for prefix in excluded_prefixes:
            if key == prefix.rstrip(".") or key.startswith(prefix):
                return True
        return False

    for k, v in sd.items():
        if _is_excluded(k):
            continue
        if isinstance(v, torch.Tensor):
            sid = (v.data_ptr(), v.device)
            if sid in seen_storage:
                for kept_k, kept_v in other_state.items():
                    if isinstance(kept_v, torch.Tensor) and (kept_v.data_ptr(), kept_v.device) == sid:
                        tied_pairs[k] = kept_k
                        break
                continue
            seen_storage.add(sid)
            other_state[k] = v.detach().cpu()
        else:
            other_state[k] = v

    # Record tied pairs so load can re-tie after state_dict load
    with open(os.path.join(cache_dir, "tied_weights.pkl"), "wb") as f:
        pickle.dump(tied_pairs, f)
    if tied_pairs:
        print(f"  tied weights: {len(tied_pairs)} pairs ({list(tied_pairs.items())[:3]}...)")

    # Write to disk
    save_file(other_state, os.path.join(cache_dir, "base.safetensors"))
    save_file(packed_tensors, os.path.join(cache_dir, "experts_nf4_packed.safetensors"))
    with open(os.path.join(cache_dir, "quant_states.pkl"), "wb") as f:
        pickle.dump({"quant_states": quant_states}, f)

    # Also save config + tokenizer for standalone use
    model.config.save_pretrained(cache_dir)
    print(f"saved NF4 cache → {cache_dir}  "
          f"(base: {len(other_state)} tensors, experts: {len(packed_tensors)} packed params)")


def _reinit_non_persistent_buffers(model):
    """Reinitialize known non-persistent buffers on a model built via
    `to_empty()`. These are not restored by `load_state_dict` since they are
    not in the state_dict.

    Covers Gemma4-family modules:
      - Gemma4TextScaledWordEmbedding.embed_scale (= scalar_embed_scale)
      - Attention softcap tensors
      - Rotary embedding per-layer-type inv_freq / original_inv_freq — the
        owning module stores its own rope_init_fns dict, re-invoke them.
    """
    fixed = 0
    for name, m in model.named_modules():
        # Scaled word embeddings — the silent-killer buffer
        if hasattr(m, "scalar_embed_scale") and hasattr(m, "embed_scale"):
            m.embed_scale.fill_(float(m.scalar_embed_scale))
            fixed += 1

        # Attention softcap. `attention_logits_soft_cap` attr carries the
        # intended scalar.
        if hasattr(m, "attention_logits_soft_cap") and hasattr(m, "softcap"):
            soft = m.attention_logits_soft_cap
            if soft is not None:
                try:
                    m.softcap.fill_(float(soft))
                    fixed += 1
                except Exception:
                    pass

        # Rotary embedding: owning module has `rope_init_fns` keyed by
        # layer_type and buffers named f"{layer_type}_inv_freq" /
        # f"{layer_type}_original_inv_freq". Re-invoke each init fn.
        if hasattr(m, "rope_init_fns") and isinstance(m.rope_init_fns, dict):
            cfg = getattr(m, "config", None)
            if cfg is not None:
                for layer_type, init_fn in m.rope_init_fns.items():
                    try:
                        kwargs = {"device": torch.device("cpu"), "layer_type": layer_type}
                        if layer_type == "full_attention" and m.rope_type.get(layer_type) == "proportional":
                            kwargs["head_dim_key"] = "global_head_dim"
                        inv_freq, _ = init_fn(cfg, **kwargs)
                        buf = f"{layer_type}_inv_freq"
                        orig = f"{layer_type}_original_inv_freq"
                        if hasattr(m, buf):
                            getattr(m, buf).copy_(inv_freq.to(getattr(m, buf).dtype))
                            fixed += 1
                        if hasattr(m, orig):
                            getattr(m, orig).copy_(inv_freq.to(getattr(m, orig).dtype))
                            fixed += 1
                    except Exception as e:
                        print(f"    [reinit-warn] {name}.{layer_type}_inv_freq: {e}")

        # Also handle the older pattern (single inv_freq without per-type)
        if hasattr(m, "inv_freq") and not hasattr(m, "rope_init_fns"):
            cfg = getattr(m, "config", None)
            if cfg is not None:
                try:
                    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
                    base = getattr(cfg, "rope_theta", 10000.0)
                    new_inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
                    if m.inv_freq.shape == new_inv.shape:
                        m.inv_freq.copy_(new_inv.to(m.inv_freq.dtype))
                        if hasattr(m, "original_inv_freq"):
                            m.original_inv_freq.copy_(new_inv.to(m.original_inv_freq.dtype))
                        fixed += 2
                except Exception:
                    pass

    print(f"  [reinit] restored {fixed} non-persistent buffers")
    return fixed


def load_quantized_cache(
    cache_dir: str = CACHE_DIR,
    dtype: torch.dtype = torch.bfloat16,
):
    """Load from the NF4 cache — avoids the 10-minute per-param quantize.

    Flow:
      1. Build empty model skeleton from cached config
      2. Load "other" state_dict (attention, embed, lm_head, routers, norms)
      3. For each experts module, call replace_parameter_4bit_prequantized
         with the cached packed data + quant_state dict
    """
    import pickle
    import time
    from safetensors.torch import load_file
    from bitsandbytes.nn.parametrize import replace_parameter_4bit_prequantized

    t0 = time.time()
    config = AutoConfig.from_pretrained(cache_dir)
    register_scattermoe()
    config._experts_implementation = "scattermoe"
    tc = getattr(config, "text_config", None)
    if tc is not None:
        tc._experts_implementation = "scattermoe"

    # Gemma4 26B-A4B has head_dim > 256 which FA2 rejects; fall back to sdpa.
    # Set GEMMA4_ATTN_IMPL=flash_attention_2 to try anyway.
    _attn_impl = os.environ.get("GEMMA4_ATTN_IMPL", "sdpa")
    try:
        config._attn_implementation = _attn_impl
        if tc is not None:
            tc._attn_implementation = _attn_impl
        print(f"  attn_implementation set to '{_attn_impl}' (override via GEMMA4_ATTN_IMPL)")
    except Exception as e:
        print(f"  [warn] could not set attn_implementation to {_attn_impl}: {e}")

    # Build empty model
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype)
    model.to_empty(device="cpu")

    # to_empty() allocates buffers but leaves them UNINITIALIZED (garbage
    # bits). Non-persistent buffers (persistent=False) are NOT in state_dict,
    # so the subsequent load_state_dict() will NOT fix them. For Gemma4:
    #   - Gemma4TextScaledWordEmbedding.embed_scale == sqrt(hidden_size) — if
    #     this ends up as a random near-zero float, all embeddings get
    #     multiplied by ~0 and the entire model silently produces
    #     uniform-distribution outputs. This was the root cause of the
    #     "uniform logits / garbage generation" bug before this fix.
    #   - Rotary emb inv_freq, attention softcap tensors, etc. are also
    #     non-persistent and may read as garbage; most are re-derived at
    #     forward time from config, but anything that isn't will misbehave.
    _reinit_non_persistent_buffers(model)

    print(f"  skeleton ready in {time.time()-t0:.1f}s")

    # Load base state
    t0 = time.time()
    base_sd = load_file(os.path.join(cache_dir, "base.safetensors"))
    missing, unexpected = model.load_state_dict(base_sd, strict=False)
    # Restore tied weights that were deduped on save
    tied_pairs_path = os.path.join(cache_dir, "tied_weights.pkl")
    if os.path.isfile(tied_pairs_path):
        with open(tied_pairs_path, "rb") as f:
            tied_pairs = pickle.load(f)
        mod_by_name_ = dict(model.named_modules())
        for dropped_key, kept_key in tied_pairs.items():
            # Parse "module.path.param_name"
            drop_mod, _, drop_p = dropped_key.rpartition(".")
            keep_mod, _, keep_p = kept_key.rpartition(".")
            if drop_mod in mod_by_name_ and keep_mod in mod_by_name_:
                kept_param = getattr(mod_by_name_[keep_mod], keep_p)
                setattr(mod_by_name_[drop_mod], drop_p, kept_param)
    print(f"  base weights loaded in {time.time()-t0:.1f}s "
          f"(missing={len(missing)} unexpected={len(unexpected)})")

    # Load packed experts + quant_states, reinstall parametrizations
    t0 = time.time()
    packed = load_file(os.path.join(cache_dir, "experts_nf4_packed.safetensors"))
    with open(os.path.join(cache_dir, "quant_states.pkl"), "rb") as f:
        qs_data = pickle.load(f)["quant_states"]

    # Resolve each key like "model.language_model.layers.0.experts.gate_up_proj"
    # → find the Gemma4TextExperts module, set param, install parametrization
    mod_by_name = dict(model.named_modules())
    for key, packed_tensor in packed.items():
        module_path, _, param_name = key.rpartition(".")
        module = mod_by_name[module_path]
        # Set the packed data as the param (bf16→uint8 packing)
        setattr(module, param_name, torch.nn.Parameter(packed_tensor, requires_grad=False))
        # Apply parametrization with cached quant_state
        qs_dict = qs_data[key]
        replace_parameter_4bit_prequantized(
            module, param_name, qs_dict, device=torch.device("cpu")
        )
    print(f"  experts reinstalled in {time.time()-t0:.1f}s "
          f"({len(packed)} params)")

    return model


def quantize_gemma4_experts_nf4(model, verbose=True) -> int:
    """Walk model, apply bnb NF4 parametrization to each Gemma4TextExperts'
    `gate_up_proj` and `down_proj` 3D parameters in-place.

    Returns the number of 3D parameters quantized.
    """
    from bitsandbytes.nn.parametrize import replace_parameter_4bit

    count = 0
    total_bytes_before = 0
    total_bytes_after = 0
    for name, module in model.named_modules():
        cls_name = module.__class__.__name__
        if cls_name != "Gemma4TextExperts":
            continue

        for pname in ("gate_up_proj", "down_proj"):
            if not hasattr(module, pname):
                continue
            p = getattr(module, pname)
            if not isinstance(p, torch.nn.Parameter):
                continue
            before = p.numel() * p.element_size()
            total_bytes_before += before
            replace_parameter_4bit(
                module,
                pname,
                compress_statistics=True,  # nested quant, as is standard NF4
                quant_type="nf4",
                blocksize=64,
            )
            # after: the `original` parameter on parametrizations is packed 4-bit
            new_p = module.parametrizations[pname].original
            after = new_p.numel() * new_p.element_size()
            total_bytes_after += after
            count += 1
            if verbose:
                print(f"   {name}.{pname}: {before/1e6:.0f}MB → {after/1e6:.0f}MB")

    if verbose:
        print(f"\nquantized {count} 3D expert params "
              f"({total_bytes_before/1e9:.1f}GB → {total_bytes_after/1e9:.1f}GB)")
    return count


def _move_quant_states_to_device(model):
    """After moving modules to GPU, walk parametrizations and move each
    Bnb4bitParametrization's quant_state tensors to the module's device.

    nn.Module.to() only moves registered Parameters and Buffers — but
    quant_state stores absmax/code as plain attributes, so they stay on CPU.
    """
    for mod_name, module in model.named_modules():
        if not hasattr(module, "parametrizations"):
            continue
        for pname in list(module.parametrizations.keys()):
            param_list = module.parametrizations[pname]
            # The actual quantized data lives on the ParametrizationList as
            # `.original` — that's what moves with .to(). Its device is truth.
            orig = getattr(param_list, "original", None)
            if orig is None:
                continue
            target_device = orig.device
            for parametrization in param_list:
                qs = getattr(parametrization, "quant_state", None)
                if qs is None:
                    continue
                for attr in ("absmax", "code", "offset"):
                    t = getattr(qs, attr, None)
                    if isinstance(t, torch.Tensor) and t.device != target_device:
                        setattr(qs, attr, t.to(target_device))
                state2 = getattr(qs, "state2", None)
                if state2 is not None:
                    for attr in ("absmax", "code", "offset"):
                        t = getattr(state2, attr, None)
                        if isinstance(t, torch.Tensor) and t.device != target_device:
                            setattr(state2, attr, t.to(target_device))


def _manual_two_gpu_split(model, first_to_gpu0: int = 10):
    """Distribute Gemma4 across 2x3090 via explicit .to() moves + a forward
    pre-hook at the GPU-split boundary to transfer hidden states.

    first_to_gpu0: first N decoder layers on cuda:0 along with embed +
    lm_head + language-model final norm. Remaining layers on cuda:1.
    Non-LM modules (vision tower, mm projector) parked on cuda:1.
    """
    # Find decoder layers ModuleList
    layers = None
    layers_name = None
    for name, sub in model.named_modules():
        if name.endswith(".layers") and hasattr(sub, "__len__"):
            layers = sub
            layers_name = name
            break
    if layers is None:
        raise RuntimeError("Could not find decoder layers ModuleList")
    n_layers = len(layers)
    print(f"  layers at {layers_name}: {n_layers} total, "
          f"first {first_to_gpu0} → cuda:0, rest → cuda:1")

    # Step 1: default all unplaced params to cuda:1 via the top-level .to()
    # then override specific subtrees. .to() moves registered Parameters +
    # Buffers recursively, including parametrize.original (which lives on a
    # child ParametrizationList).
    model.to("cuda:1")

    # Step 2: move specific subtrees to cuda:0
    # 2a. first N decoder layers
    for i in range(first_to_gpu0):
        layers[i].to("cuda:0")

    # 2b. embed, lm_head, language_model final norm
    for name, module in model.named_modules():
        if name == layers_name or name.startswith(f"{layers_name}."):
            continue
        # embed_tokens
        if name.endswith("embed_tokens"):
            module.to("cuda:0")
        # lm_head
        elif name == "lm_head":
            module.to("cuda:0")
        # Language-model final norm (not inside a layer)
        elif (("language_model" in name or "model." in name)
              and name.endswith(".norm") and "layers." not in name):
            module.to("cuda:0")

    # Step 3: move quant_state tensors to match each parametrized module's device
    _move_quant_states_to_device(model)

    # Sanity: walk EVERY parametrization and flag any CPU/mismatch
    cpu_flagged = 0
    for name, module in model.named_modules():
        if not hasattr(module, "parametrizations"):
            continue
        for pname in list(module.parametrizations.keys()):
            pl = module.parametrizations[pname]
            orig_dev = pl.original.device
            qs = pl[0].quant_state
            am_dev = qs.absmax.device
            s2_dev = qs.state2.absmax.device if qs.state2 is not None else am_dev
            if orig_dev.type == "cpu" or am_dev != orig_dev or s2_dev != orig_dev:
                print(f"    [MISMATCH] {name}.{pname}: orig={orig_dev} absmax={am_dev} s2={s2_dev}")
                cpu_flagged += 1
    print(f"    [device sweep] {cpu_flagged} mismatches across all parametrizations")

    # Keep the sampled layer print for context
    for layer_idx in [0, 5, 6, 10, 15, 20, 25, 29]:
        if layer_idx >= n_layers:
            continue
        experts_mod_path = f"{layers_name}.{layer_idx}.experts"
        experts_mod = dict(model.named_modules()).get(experts_mod_path)
        if experts_mod is None or not hasattr(experts_mod, "parametrizations"):
            continue
        if "gate_up_proj" not in experts_mod.parametrizations:
            continue
        pl = experts_mod.parametrizations["gate_up_proj"]
        qs = pl[0].quant_state
        print(f"    [devices] layer {layer_idx}: orig={pl.original.device} "
              f"absmax={qs.absmax.device} state2_absmax={qs.state2.absmax.device if qs.state2 else '(none)'}")

    # Step 4: install per-layer pre-hooks that move ALL tensor args/kwargs
    # onto the layer's device. This handles rotary cos/sin, attention_mask,
    # position_ids, etc. — any tensor computed upstream on a different GPU.
    def _make_pre_hook(target_device):
        def _hook(module, args, kwargs):
            # Critical for triton: set current cuda device BEFORE the layer's
            # kernels launch. Triton picks up torch.cuda.current_device() when
            # it launches a kernel — if the layer's tensors are on cuda:1 but
            # current_device is 0, we get "Pointer cannot be accessed from
            # Triton".
            if target_device.type == "cuda":
                torch.cuda.set_device(target_device)
            new_args = tuple(
                a.to(target_device) if isinstance(a, torch.Tensor) and a.device != target_device else a
                for a in args
            )
            new_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor) and v.device != target_device:
                    new_kwargs[k] = v.to(target_device)
                elif isinstance(v, tuple):
                    new_kwargs[k] = tuple(
                        t.to(target_device) if isinstance(t, torch.Tensor) and t.device != target_device else t
                        for t in v
                    )
                else:
                    new_kwargs[k] = v
            return new_args, new_kwargs
        return _hook

    for i in range(n_layers):
        dev = torch.device("cuda:0") if i < first_to_gpu0 else torch.device("cuda:1")
        layers[i].register_forward_pre_hook(_make_pre_hook(dev), with_kwargs=True)

    # After the last layer, move hidden state back to cuda:0 for final norm + lm_head
    if first_to_gpu0 < n_layers:
        last_layer = layers[n_layers - 1]

        def _transfer_back_hook(module, args, output):
            target = torch.device("cuda:0")
            if isinstance(output, tuple):
                return tuple(
                    o.to(target) if isinstance(o, torch.Tensor) and o.device != target else o
                    for o in output
                )
            if isinstance(output, torch.Tensor) and output.device != target:
                return output.to(target)
            return output

        last_layer.register_forward_hook(_transfer_back_hook)

    # Diagnostic
    g0 = g1 = cpu = 0
    for p in model.parameters():
        b = p.numel() * p.element_size()
        if p.device.type == "cuda" and p.device.index == 0:
            g0 += b
        elif p.device.type == "cuda" and p.device.index == 1:
            g1 += b
        else:
            cpu += b
    print(f"  post-split: cuda:0={g0/1e9:.1f}GB  cuda:1={g1/1e9:.1f}GB  cpu={cpu/1e9:.1f}GB")


def load_fused_quantized_gemma4(
    model_id: str = "google/gemma-4-26B-A4B-it",
    device_map: dict | str | None = None,
    dtype: torch.dtype = torch.bfloat16,
    register_smoe: bool = True,
    quantize: bool = True,
    use_cache: bool = True,
    save_cache: bool = False,
):
    """Load stock-fused Gemma4 with 3D experts NF4-quantized.

    Strategy:
      1. Register scattermoe ExpertsInterface BEFORE load, set
         `config._experts_implementation="scattermoe"` so experts_forward
         dispatches to the scattermoe kernel.
      2. Load full bf16 on CPU (need ~52GB RAM, host has 172GB).
      3. Walk the model, quantize each experts module's gate_up_proj + down_proj
         (3D nn.Parameter) via bnb replace_parameter_4bit.
      4. Manually distribute layers across GPUs (stock `device_map="auto"` moves
         experts to CPU in hook mode, which fights our in-place parametrization).
    """
    # Prefer cached quantized model if available — saves 10 min per load
    if use_cache and os.path.isdir(CACHE_DIR) and os.path.isfile(os.path.join(CACHE_DIR, "experts_nf4_packed.safetensors")):
        print(f"loading from NF4 cache: {CACHE_DIR}")
        model = load_quantized_cache(CACHE_DIR, dtype=dtype)
        quantize = False  # already quantized
    else:
        print(f"loading {model_id} ...")
        t0 = time.time()

        config = AutoConfig.from_pretrained(model_id)
        if register_smoe:
            register_scattermoe()
            config._experts_implementation = "scattermoe"
            tc = getattr(config, "text_config", None)
            if tc is not None:
                tc._experts_implementation = "scattermoe"

        # Load in bf16 on CPU first — parametrize then distribute
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map={"": "cpu"},
        )
        print(f"  loaded to CPU in {time.time()-t0:.1f}s")

        if quantize:
            print("\nquantizing expert 3D tensors to NF4 ...")
            t0 = time.time()
            n = quantize_gemma4_experts_nf4(model, verbose=True)
            print(f"quantization done in {time.time()-t0:.1f}s ({n} tensors)")

            if save_cache:
                print(f"\nsaving NF4 cache → {CACHE_DIR} ...")
                t0 = time.time()
                save_quantized_cache(model, CACHE_DIR)
                print(f"cache saved in {time.time()-t0:.1f}s")

    if device_map is None:
        # Default: put the first third of decoder layers on GPU 0, rest on GPU 1,
        # embeddings + lm_head on GPU 0 (where CE loss runs).
        # Set LOFT_SINGLE_GPU=1 to keep everything on cuda:0 (simpler,
        # 17GB model; still leaves ~7GB for activations + lora on a 3090).
        if os.environ.get("LOFT_SINGLE_GPU"):
            print("\nmoving model to cuda:0 (single-GPU mode) ...")
            t0 = time.time()
            model.to("cuda:0")
            _move_quant_states_to_device(model)
            print(f"moved in {time.time()-t0:.1f}s")
        else:
            print("\ndistributing to GPUs (manual split) ...")
            t0 = time.time()
            _split_n = int(os.environ.get("LOFT_FIRST_TO_GPU0", "10"))
            _manual_two_gpu_split(model, first_to_gpu0=_split_n)
            print(f"distributed in {time.time()-t0:.1f}s")
    else:
        # Let transformers dispatch
        print(f"\ndispatching with device_map={device_map!r} ...")
        from accelerate import dispatch_model
        model = dispatch_model(model, device_map=device_map)

    # Sanity print
    gpu0_bytes = gpu1_bytes = 0
    cpu_bytes = 0
    for p in model.parameters():
        b = p.numel() * p.element_size()
        if p.device.type == "cuda":
            if p.device.index == 0:
                gpu0_bytes += b
            else:
                gpu1_bytes += b
        else:
            cpu_bytes += b
    print(f"\nmemory: cuda:0={gpu0_bytes/1e9:.1f}GB  cuda:1={gpu1_bytes/1e9:.1f}GB  "
          f"cpu={cpu_bytes/1e9:.1f}GB")

    tok = AutoTokenizer.from_pretrained(model_id)
    return model, tok


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-cache", action="store_true", help="Quantize fresh and save cache")
    ap.add_argument("--no-cache", action="store_true", help="Skip loading from cache")
    args = ap.parse_args()
    model, tok = load_fused_quantized_gemma4(
        use_cache=not args.no_cache,
        save_cache=args.save_cache,
    )
    print("\nload OK — running single forward for sanity ...")
    with torch.inference_mode():
        inputs = tok("The quick brown fox", return_tensors="pt")
        inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
        inputs["mm_token_type_ids"] = torch.zeros_like(inputs["input_ids"])
        out = model(**inputs)
        logits = out.logits
        print(f"logits shape: {logits.shape}, dtype: {logits.dtype}, device: {logits.device}")
        # argmax greedy next tok
        nxt = logits[0, -1].argmax().item()
        print(f"greedy next token id: {nxt} → {tok.decode([nxt])!r}")
    print("DONE")
