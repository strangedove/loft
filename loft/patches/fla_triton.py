"""Patch flash-linear-attention Triton kernel compatibility issues.

fla 0.4.1's gated_delta_rule backward kernel (wy_fast.py) has a Triton
compilation bug: ``tl.trans(b_k)`` produces a tensor with a transposed
memory layout, and the subsequent element-wise multiply with a broadcast
(``b_kt * b_b[None, :]``) fails because Triton's ``arith.mulf`` requires
matching encodings for all operands.

Error: ``'arith.mulf' op requires the same encoding for all operands and results``

The fix rewrites the expression to perform the multiply before the
transpose, avoiding the layout mismatch entirely:
  ``b_ktb = b_kt * b_b[None, :]``  ->  ``b_ktb = tl.trans(b_k * b_b[:, None])``

These are mathematically equivalent (transpose distributes over
element-wise products when the broadcast dimensions are swapped).

This patch modifies the installed fla source file in site-packages and
clears the Triton compilation cache for the affected module so the fixed
kernel is recompiled on next use.
"""

import logging
import shutil
from pathlib import Path


logger = logging.getLogger(__name__)

# The buggy pattern and its replacement
_BUG_PATTERN = "b_ktb = b_kt * b_b[None, :]"
_FIX_PATTERN = "b_ktb = tl.trans(b_k * b_b[:, None])"


def _find_wy_fast() -> Path | None:
    """Locate the fla gated_delta_rule wy_fast.py file."""
    try:
        import fla.ops.gated_delta_rule.wy_fast as mod

        return Path(mod.__file__)
    except (ImportError, AttributeError):
        return None


def patch_fla_wy_fast(*, dry_run: bool = False) -> bool:
    """Patch the fla wy_fast.py Triton kernel if the bug is present.

    Returns True if the patch was applied (or was already applied).
    Returns False if fla is not installed or the file couldn't be found.
    """
    path = _find_wy_fast()
    if path is None:
        logger.debug("flash-linear-attention not installed — skipping wy_fast patch")
        return False

    source = path.read_text()

    if _FIX_PATTERN in source:
        logger.debug("fla wy_fast.py already patched")
        return True

    if _BUG_PATTERN not in source:
        logger.debug(
            "fla wy_fast.py does not contain the known bug pattern — "
            "may be a newer version with the fix upstream"
        )
        return True

    if dry_run:
        logger.info(f"Would patch {path}")
        return True

    # Apply the fix
    patched = source.replace(_BUG_PATTERN, _FIX_PATTERN)

    try:
        path.write_text(patched)
    except PermissionError:
        logger.warning(
            f"Cannot write to {path} (permission denied). "
            "Run with appropriate permissions or apply the patch manually:\n"
            f"  Replace: {_BUG_PATTERN}\n"
            f"  With:    {_FIX_PATTERN}"
        )
        return False

    # Clear any cached Triton compilations for this module so the
    # fixed kernel is recompiled on next use
    _clear_triton_cache(path)

    logger.info(f"Patched fla wy_fast.py Triton kernel bug at {path}")
    return True


def _clear_triton_cache(patched_file: Path) -> None:
    """Remove Triton's compilation cache so patched kernels are recompiled."""
    try:
        import triton

        cache_dir = Path(triton.runtime.cache.default_cache_dir())
        if cache_dir.exists():
            # Triton caches by function hash, so clearing the whole cache
            # is the safest approach.  It only costs one recompilation.
            shutil.rmtree(cache_dir, ignore_errors=True)
            logger.debug(f"Cleared Triton cache at {cache_dir}")
    except Exception:
        # Non-fatal — worst case, the old cached kernel is used once
        # and then the cache entry is replaced on the next change
        pass
