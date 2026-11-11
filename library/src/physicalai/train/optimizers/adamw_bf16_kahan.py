# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""AdamW with Kahan-compensated updates for bf16 parameters."""

from __future__ import annotations

import math
from functools import cache
from typing import TYPE_CHECKING, Any

import torch
from torch.optim.adam import adam

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from torch.optim.optimizer import ParamsT

# Elements per chunk for the bf16 update; bounds the fp32 temporaries for large embeddings.
_CHUNK_NUMEL = 1 << 24
# Moments with an EMA weight (1 - beta) below this stay fp32: bf16's 8-bit mantissa rounds their updates away.
_MIN_BF16_EMA_WEIGHT = 2**-5


class AdamWBF16Kahan(torch.optim.Optimizer):
    """AdamW with Kahan-compensated updates for bf16 parameters.

    Each bf16 update is computed in fp32 on ``weight + compensation``, rounded to bf16, and
    the rounding residual is stored in a bf16 ``compensation`` buffer for the next step. The
    weights follow the fp32-weight trajectory at 2 extra bytes per bf16 parameter. On CUDA and XPU the
    bf16 update is one fused Triton kernel per tensor; without Triton, or for non-contiguous
    tensors, it falls back to chunked eager ops. Other parameters use torch's foreach AdamW.

    Moments of bf16 parameters are bf16 unless their beta is too close to 1 for bf16 to hold
    the EMA update (e.g. the default ``beta2=0.999``); those moments are kept in fp32.
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

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state, keeping fp32 moments that torch would cast to the bf16 parameter dtype."""
        super().load_state_dict(state_dict)
        saved_ids = (i for group in state_dict["param_groups"] for i in group["params"])
        params = (p for group in self.param_groups for p in group["params"])
        for param_id, p in zip(saved_ids, params, strict=True):
            saved = state_dict["state"].get(param_id, {})
            for key in ("exp_avg", "exp_avg_sq"):
                value = saved.get(key)
                if isinstance(value, torch.Tensor) and value.dtype == torch.float32 and p.dtype != torch.float32:
                    self.state[p][key] = value.to(device=p.device, copy=True)

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
                    state["exp_avg"] = torch.zeros_like(p, dtype=_moment_dtype(p, group["betas"][0]))
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=_moment_dtype(p, group["betas"][1]))
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
        contiguous = all(t.is_contiguous() for t in tensors)
        if p.device.type in {"cuda", "xpu"} and contiguous and (fused := _fused_update()) is not None:
            fused(*tensors, beta1, beta2, step_size, bias_correction2_sqrt, group["eps"], decay)
            return
        chunks: Iterable[tuple[torch.Tensor, ...]] = (
            zip(*(t.view(-1).split(_CHUNK_NUMEL) for t in tensors), strict=True) if contiguous else [tensors]
        )
        for weight, grad_chunk, compensation, exp_avg, exp_avg_sq in chunks:
            grad32 = grad_chunk.float()
            # fp32 moments are updated in place (``.float()`` returns them); bf16 ones via the copies.
            exp_avg32 = exp_avg.float().lerp_(grad32, 1 - beta1)
            exp_avg_sq32 = exp_avg_sq.float().lerp_(grad32.square(), 1 - beta2)
            exp_avg.copy_(exp_avg32)
            exp_avg_sq.copy_(exp_avg_sq32)
            denom = exp_avg_sq32.sqrt().div_(bias_correction2_sqrt).add_(group["eps"])
            weight32 = weight.float().add_(compensation.float()).mul_(decay)
            weight32.addcdiv_(exp_avg32, denom, value=-step_size)
            weight.copy_(weight32)
            compensation.copy_(weight32.sub_(weight.float()))


@cache
def _fused_update() -> Callable[..., None] | None:
    """Return the Triton update kernel, or ``None`` when Triton is unavailable."""
    try:
        from ._kahan_triton import kahan_adamw_update  # noqa: PLC0415
    except ImportError:
        return None
    return kahan_adamw_update


def _moment_dtype(p: torch.Tensor, beta: float) -> torch.dtype:
    if p.dtype == torch.bfloat16 and 1 - beta < _MIN_BF16_EMA_WEIGHT:
        return torch.float32
    return p.dtype
