# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Optimizer factories shared by policies."""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from torch.optim.optimizer import ParamsT


def adamw(params: ParamsT, **kwargs: Any) -> torch.optim.Optimizer:  # noqa: ANN401
    """Build AdamW, with bf16 stochastic rounding when any parameter is bf16.

    Plain AdamW on bf16 weights rounds updates below bf16 resolution to zero, so small
    learning rates leave most weights unchanged. torchao's AdamW computes the update in
    fp32 and rounds it stochastically, keeping the expected update without fp32 master
    weights. Other parameter dtypes use ``torch.optim.AdamW``.

    Args:
        params: Parameters or parameter groups, as for ``torch.optim.AdamW``.
        **kwargs: ``torch.optim.AdamW`` options. ``fused`` is ignored for bf16 parameters.

    Returns:
        The optimizer.
    """
    param_list: list[Any] = list(params)
    groups = (item["params"] if isinstance(item, dict) else item for item in param_list)
    tensors = (p for group in groups for p in ([group] if isinstance(group, torch.Tensor) else group))
    if not any(getattr(p, "dtype", None) == torch.bfloat16 for p in tensors):  # skips names in named params
        return torch.optim.AdamW(param_list, **kwargs)
    kwargs.pop("fused", None)
    return _bf16_adamw_class()(param_list, **kwargs)


@cache
def _bf16_adamw_class() -> type[torch.optim.Optimizer]:
    # Lazy: importing torchao takes seconds and only bf16 training needs it.
    from torchao.optim import _AdamW  # noqa: PLC0415, PLC2701

    class BF16AdamW(_AdamW):
        """torchao AdamW with bf16 stochastic rounding."""

        def __init__(self, params: Any, **kwargs: Any) -> None:  # noqa: ANN401
            super().__init__(params, bf16_stochastic_round=True, **kwargs)

        def load_state_dict(self, state_dict: dict[str, Any]) -> None:
            """Load state, converting float learning rates from torch AdamW checkpoints to tensors."""
            super().load_state_dict(state_dict)
            for group in self.param_groups:
                if not isinstance(group["lr"], torch.Tensor):
                    group["lr"] = torch.tensor(group["lr"], dtype=torch.float32)

    return BF16AdamW
