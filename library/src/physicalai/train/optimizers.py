# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Optimizer factories shared by policies."""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Any

import torch
from torch.optim.adam import adam

if TYPE_CHECKING:
    from collections.abc import Callable

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
        """torchao AdamW with bf16 stochastic rounding; other dtypes use torch's foreach AdamW."""

        def __init__(self, params: Any, **kwargs: Any) -> None:  # noqa: ANN401
            super().__init__(params, bf16_stochastic_round=True, **kwargs)

        @torch.no_grad()
        def step(self, closure: Callable[[], float] | None = None) -> float | None:
            """Update bf16 parameters with torchao and the rest with one foreach AdamW call per group.

            torchao launches a compiled step per tensor; on Pi05 the 562 fp32 tensors made that
            overhead larger than the whole torch AdamW step.

            Returns:
                The optional closure loss.
            """
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
            all_params = [group["params"] for group in self.param_groups]
            try:
                for group, params in zip(self.param_groups, all_params, strict=True):
                    group["params"] = [p for p in params if p.dtype == torch.bfloat16]
                super().step()
            finally:
                for group, params in zip(self.param_groups, all_params, strict=True):
                    group["params"] = params
            for group in self.param_groups:
                self._foreach_step(group)
            return loss

        def _foreach_step(self, group: dict[str, Any]) -> None:
            params = [p for p in group["params"] if p.grad is not None and p.dtype != torch.bfloat16]
            for p in params:
                state = self.state[p]
                if not state:
                    state["step"] = torch.tensor(0.0)
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    if group["amsgrad"]:
                        state["max_exp_avg_sq"] = torch.zeros_like(p)
            if not params:
                return
            states = [self.state[p] for p in params]
            beta1, beta2 = group["betas"]
            adam(
                params,
                [p.grad for p in params],
                [s["exp_avg"] for s in states],
                [s["exp_avg_sq"] for s in states],
                [s["max_exp_avg_sq"] for s in states] if group["amsgrad"] else [],
                [s["step"] for s in states],
                foreach=True,
                decoupled_weight_decay=True,
                amsgrad=group["amsgrad"],
                beta1=beta1,
                beta2=beta2,
                lr=float(group["lr"]),
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=False,
            )

        def load_state_dict(self, state_dict: dict[str, Any]) -> None:
            """Load state, converting float learning rates from torch AdamW checkpoints to tensors."""
            super().load_state_dict(state_dict)
            for group in self.param_groups:
                if not isinstance(group["lr"], torch.Tensor):
                    group["lr"] = torch.tensor(group["lr"], dtype=torch.float32)

    return BF16AdamW
