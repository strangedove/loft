"""TRL / llm_blender <-> Transformers 5.x compatibility shim.

Several third-party modules that TRL pulls in still reference names that
transformers 5.x removed / renamed. This shim reinjects the old aliases
before TRL imports so ``DPOConfig`` / ``DPOTrainer`` load cleanly on
transformers 5.2 / 5.5.

Patches installed (each is a no-op if already present):
  - ``transformers.models.auto.modeling_auto.MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES``
    → alias for ``MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES`` (needed by
    older TRL's ``dpo_trainer`` import).
  - ``transformers.utils.hub.TRANSFORMERS_CACHE``
    → alias for ``transformers.utils.constants.HF_HUB_CACHE`` (needed by
    ``llm_blender`` which TRL eagerly imports via the judges module).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def patch_trl_vision_mapping_import() -> bool:
    """Inject ``MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES`` alias if missing.

    Returns True if the shim was installed, False if it wasn't needed.
    """
    try:
        import transformers.models.auto.modeling_auto as _ma
    except Exception:  # pragma: no cover
        return False

    if hasattr(_ma, "MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES"):
        return False

    # Transformers 5.x name
    new_name = "MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES"
    if not hasattr(_ma, new_name):
        # Neither name present — nothing to alias. Let TRL fail loudly.
        return False

    _ma.MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES = getattr(_ma, new_name)
    logger.info(
        "Aliased MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES → "
        "MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES for TRL 0.12 compatibility"
    )
    return True


def patch_transformers_cache_alias() -> bool:
    """Inject ``transformers.utils.hub.TRANSFORMERS_CACHE`` alias if missing.

    ``llm_blender`` (pulled in by TRL's ``judges`` module) imports
    ``TRANSFORMERS_CACHE`` from ``transformers.utils.hub``, which
    transformers 5.x moved to ``transformers.utils.constants.HF_HUB_CACHE``.

    Returns True if the shim was installed, False if it wasn't needed.
    """
    try:
        import transformers.utils.hub as _hub
    except Exception:  # pragma: no cover
        return False

    if hasattr(_hub, "TRANSFORMERS_CACHE"):
        return False

    cache_val = None
    try:
        from transformers.utils import constants as _consts

        cache_val = getattr(_consts, "HF_HUB_CACHE", None)
    except Exception:
        cache_val = None

    if cache_val is None:
        try:
            from huggingface_hub.constants import HF_HUB_CACHE as cache_val  # type: ignore
        except Exception:
            return False

    _hub.TRANSFORMERS_CACHE = cache_val
    logger.info(
        "Aliased transformers.utils.hub.TRANSFORMERS_CACHE → HF_HUB_CACHE "
        "for llm_blender / TRL import compatibility"
    )
    return True


def patch_trl_imports() -> None:
    """Apply every TRL-related import shim in order."""
    patch_trl_vision_mapping_import()
    patch_transformers_cache_alias()
