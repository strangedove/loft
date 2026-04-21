# SPDX-License-Identifier: Apache-2.0
# ScatterMoE integration for loft training framework.
# Adapted from axolotl (https://github.com/axolotl-ai-collective/axolotl)
# Copyright (c) axolotl-ai-collective contributors
# Licensed under the Apache License, Version 2.0

"""
ScatterMoE kernel patching for MoE model training acceleration.

Supports two integration paths:
1. ExpertsInterface (Gemma4, Mixtral with transformers >= 5.x)
   - Registers "scattermoe" implementation via transformers.integrations.moe
   - Set experts_implementation="scattermoe" in model config

2. SparseMoeBlock monkeypatch (Qwen3-MoE, OLMoE, etc.)
   - Replaces forward method on the SparseMoeBlock class
   - Applied after model loading

Usage in loft config:
    use_scattermoe: true
"""

import logging
from typing import Optional

import torch.nn as nn

logger = logging.getLogger(__name__)

# Model type -> SparseMoeBlock class name mapping
SPARSE_MOE_BLOCK = {
    "qwen2_moe": "Qwen2MoeSparseMoeBlock",
    "qwen3_moe": "Qwen3MoeSparseMoeBlock",
    "olmoe": "OlmoeSparseMoeBlock",
    "mixtral": "MixtralSparseMoeBlock",
    "deepseek_v3": "DeepseekV3MoE",
}

# Model types that use ExpertsInterface (no SparseMoeBlock)
EXPERTS_INTERFACE_MODELS = {
    "gemma4_text",
    "gemma4",
}


def _resolve_model_type(model: nn.Module) -> Optional[str]:
    """Get the model_type from a model's config, checking text_config first."""
    config = getattr(model, "config", None)
    if config is None:
        return None

    # VLMs have text_config with the actual LM model_type
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        return getattr(text_config, "model_type", None)

    return getattr(config, "model_type", None)


def _resolve_sparse_moe_block(model_type: str):
    """Dynamically import the SparseMoeBlock class for a given model type."""
    import importlib

    cls_name = SPARSE_MOE_BLOCK.get(model_type)
    if cls_name is None:
        return None

    try:
        module = importlib.import_module(
            f"transformers.models.{model_type}.modeling_{model_type}"
        )
        return getattr(module, cls_name, None)
    except (ImportError, AttributeError) as e:
        logger.warning(f"Could not resolve SparseMoeBlock for {model_type}: {e}")
        return None


def patch_scattermoe(model: nn.Module) -> bool:
    """Apply ScatterMoE kernel patches to a model.

    Detects the model type and applies the appropriate patching strategy:
    - ExpertsInterface registration for Gemma4-style models
    - SparseMoeBlock forward replacement for Qwen/OLMoE-style models

    Args:
        model: The model to patch (can be a PeftModel wrapping the base).

    Returns:
        True if patching was applied, False if the model type is not supported.
    """
    # Unwrap PeftModel to get base model config
    base_model = model
    while hasattr(base_model, "model"):
        base_model = base_model.model

    model_type = _resolve_model_type(base_model)
    if model_type is None:
        logger.warning("Could not determine model_type for ScatterMoE patching")
        return False

    logger.info(f"ScatterMoE: detected model_type={model_type}")

    # Path 1: ExpertsInterface (Gemma4)
    if model_type in EXPERTS_INTERFACE_MODELS:
        return _patch_experts_interface(base_model, model_type)

    # Path 2: SparseMoeBlock monkeypatch
    if model_type in SPARSE_MOE_BLOCK:
        return _patch_sparse_moe_block(model_type)

    logger.warning(
        f"ScatterMoE: model_type '{model_type}' not supported. "
        f"Supported: {list(EXPERTS_INTERFACE_MODELS | set(SPARSE_MOE_BLOCK.keys()))}"
    )
    return False


def _patch_experts_interface(model: nn.Module, model_type: str) -> bool:
    """Register ScatterMoE via transformers ExpertsInterface."""
    try:
        from .gemma4_experts import register_scattermoe_experts

        register_scattermoe_experts()

        # Set the implementation on the model config
        config = getattr(model, "config", None)
        if config is not None:
            config._experts_implementation = "scattermoe"
            # Also set on text_config if it exists
            text_config = getattr(config, "text_config", None)
            if text_config is not None:
                text_config._experts_implementation = "scattermoe"

        logger.info(
            f"ScatterMoE: registered ExpertsInterface for {model_type}"
        )
        return True
    except Exception as e:
        logger.error(f"ScatterMoE ExpertsInterface registration failed: {e}")
        return False


def _patch_sparse_moe_block(model_type: str) -> bool:
    """Replace SparseMoeBlock.forward with ScatterMoE kernel."""
    moe_cls = _resolve_sparse_moe_block(model_type)
    if moe_cls is None:
        logger.warning(f"ScatterMoE: could not find SparseMoeBlock for {model_type}")
        return False

    if hasattr(moe_cls, "_original_forward"):
        logger.info(f"ScatterMoE: {moe_cls.__name__} already patched")
        return True

    try:
        from .layers import HFScatterMoEGatedMLP

        moe_cls._original_forward = moe_cls.forward
        moe_cls.forward = HFScatterMoEGatedMLP.forward
        logger.info(
            f"ScatterMoE: patched {moe_cls.__name__}.forward with ScatterMoE kernel"
        )
        return True
    except Exception as e:
        logger.error(f"ScatterMoE forward patch failed: {e}")
        return False
