# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""AdamW factory that picks the bf16-safe optimizer from parameter dtypes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from .adamw_bf16_kahan import AdamWBF16Kahan

if TYPE_CHECKING:
    from torch.optim.optimizer import ParamsT


def build_adamw_auto_precision(params: ParamsT, **kwargs: Any) -> torch.optim.Optimizer:  # noqa: ANN401
    """Build AdamW, with Kahan-compensated updates when any parameter is bf16.

    Plain AdamW on bf16 weights rounds updates below bf16 resolution to zero, so small
    learning rates leave most weights unchanged. ``AdamWBF16Kahan`` keeps the rounding residual
    in a bf16 buffer instead of an fp32 copy of the weights. Other parameter dtypes use
    ``torch.optim.AdamW``.

    Args:
        params: Parameters or parameter groups, as for ``torch.optim.AdamW``.
        **kwargs: ``torch.optim.AdamW`` options. With bf16 parameters, ``fused`` and ``foreach``
            are ignored and ``amsgrad``, ``maximize``, ``capturable`` and ``differentiable`` must
            be left at their defaults.

    Returns:
        The optimizer.

    Raises:
        ValueError: If an unsupported option is enabled and a parameter is bf16.
    """
    # Materialize generator groups (e.g. ``model.parameters()``) so the dtype scan doesn't consume them.
    param_list: list[Any] = [
        {**item, "params": _as_list(item["params"])} if isinstance(item, dict) else item for item in params
    ]
    groups = (item["params"] if isinstance(item, dict) else [item] for item in param_list)
    if not any(getattr(p, "dtype", None) == torch.bfloat16 for group in groups for p in group):  # skips names
        return torch.optim.AdamW(param_list, **kwargs)
    kwargs.pop("fused", None)
    kwargs.pop("foreach", None)
    enabled = [key for key in ("amsgrad", "maximize", "capturable", "differentiable") if kwargs.pop(key, False)]
    if enabled:
        msg = f"AdamWBF16Kahan does not support {enabled}; set them to False or train with non-bf16 weights."
        raise ValueError(msg)
    return AdamWBF16Kahan(param_list, **kwargs)


def _as_list(params: Any) -> list[Any]:  # noqa: ANN401
    return [params] if isinstance(params, torch.Tensor) else list(params)
