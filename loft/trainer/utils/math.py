from typing import TypeVar

import torch

def get_exp_cap(value: torch.Tensor, decimal: int = 4) -> torch.Tensor:
    """
    Get the exponent cap of a value. This is used to cap the exponent of a value to avoid overflow. The formula is :
    log(value.dtype.max) E.g. for float32 data type, the maximum exponent value is 88.7228 to 4 decimal points.

    Args:
        value (`torch.Tensor`):
            The input tensor to obtain the data type
        decimal (`int`):
            The number of decimal points of the output exponent cap. eg: direct calling exp(log(torch.float32.max))
            will result in inf so we cap the exponent to 88.7228 to avoid overflow.
    """
    vdtype_max = torch.zeros([1]).to(value.dtype) + torch.finfo(value.dtype).max
    vdtype_log_max = torch.log(vdtype_max).to(value.device)
    return torch.floor(vdtype_log_max * 10**decimal) / 10**decimal if decimal > 0 else vdtype_log_max  # pyright: ignore[reportAny]


def cap_exp(value: torch.Tensor, cap: int = -1) -> torch.Tensor:
    # Cap the exponent value below the upper-bound to avoid overflow, before calling torch.exp
    cap = get_exp_cap(value) if cap < 0 else cap  # pyright: ignore[reportAssignmentType]
    return torch.exp(torch.clamp(value, max=cap))