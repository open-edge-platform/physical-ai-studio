# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Optimizer factories shared by policies."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch
from torch.optim.adam import adam

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from torch.optim.optimizer import ParamsT

# Elements per chunk for the bf16 update; bounds the fp32 temporaries for large embeddings.
_CHUNK_NUMEL = 1 << 24


def adamw(params: ParamsT, **kwargs: Any) -> torch.optim.Optimizer:  # noqa: ANN401
    """Build AdamW, with Kahan-compensated updates when any parameter is bf16.

    Plain AdamW on bf16 weights rounds updates below bf16 resolution to zero, so small
    learning rates leave most weights unchanged. ``BF16AdamW`` keeps the rounding residual
    in a bf16 buffer instead of an fp32 copy of the weights. Other parameter dtypes use
    ``torch.optim.AdamW``.

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
    return BF16AdamW(param_list, **kwargs)


class BF16AdamW(torch.optim.Optimizer):
    """AdamW with Kahan-compensated updates for bf16 parameters.

    Each bf16 update is computed in fp32 on ``weight + compensation``, rounded to bf16, and
    the rounding residual is stored in a bf16 ``compensation`` buffer for the next step. The
    weights follow the fp32-weight trajectory at 2 extra bytes per bf16 parameter. Other
    parameters use torch's foreach AdamW. Optimizer state stays in the parameter dtype.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ) -> None:
        """Initialize the optimizer with ``torch.optim.AdamW`` defaults."""
        super().__init__(params, {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay})

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:  # pyrefly: ignore[bad-override]
        """Apply one AdamW update.

        Returns:
            The optional closure loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            for p in params:
                state = self.state[p]
                if not state:
                    state["step"] = torch.tensor(0.0)
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                if p.dtype == torch.bfloat16 and "compensation" not in state:
                    state["compensation"] = torch.zeros_like(p)
            for p in params:
                if p.dtype == torch.bfloat16:
                    self._step_bf16(p, group)
            others = [p for p in params if p.dtype != torch.bfloat16]
            if others:
                states = [self.state[p] for p in others]
                beta1, beta2 = group["betas"]
                adam(
                    others,
                    [p.grad for p in others],
                    [s["exp_avg"] for s in states],
                    [s["exp_avg_sq"] for s in states],
                    [],
                    [s["step"] for s in states],
                    foreach=True,
                    decoupled_weight_decay=True,
                    amsgrad=False,
                    beta1=beta1,
                    beta2=beta2,
                    lr=float(group["lr"]),
                    weight_decay=group["weight_decay"],
                    eps=group["eps"],
                    maximize=False,
                )
        return loss

    def _step_bf16(self, p: torch.Tensor, group: dict[str, Any]) -> None:  # noqa: PLR0914
        grad = p.grad
        if grad is None:
            return
        state = self.state[p]
        state["step"] += 1
        step = state["step"].item()
        beta1, beta2 = group["betas"]
        lr = float(group["lr"])
        step_size = lr / (1 - beta1**step)
        bias_correction2_sqrt = math.sqrt(1 - beta2**step)
        decay = 1 - lr * group["weight_decay"]
        tensors = (p, grad, state["compensation"], state["exp_avg"], state["exp_avg_sq"])
        chunks: Iterable[tuple[torch.Tensor, ...]]
        if all(t.is_contiguous() for t in tensors):
            chunks = zip(*(t.view(-1).split(_CHUNK_NUMEL) for t in tensors), strict=True)
        else:
            chunks = iter([tensors])
        for weight, grad_chunk, compensation, exp_avg, exp_avg_sq in chunks:
            grad32 = grad_chunk.float()
            exp_avg32 = exp_avg.float().lerp_(grad32, 1 - beta1)
            exp_avg_sq32 = exp_avg_sq.float().lerp_(grad32.square(), 1 - beta2)
            exp_avg.copy_(exp_avg32)
            exp_avg_sq.copy_(exp_avg_sq32)
            denom = exp_avg_sq32.sqrt_().div_(bias_correction2_sqrt).add_(group["eps"])
            weight32 = weight.float().add_(compensation.float()).mul_(decay)
            weight32.addcdiv_(exp_avg32, denom, value=-step_size)
            weight.copy_(weight32)
            compensation.copy_(weight32.sub_(weight.float()))
