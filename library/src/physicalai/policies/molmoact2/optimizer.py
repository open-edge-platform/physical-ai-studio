# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MolmoAct2 optimizer with component-wise clipping and BF16 compensation."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


def molmoact2_cosine_with_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    peak_lr: float,
    decay_lr: float,
    num_warmup_steps: int,
    num_decay_steps: int,
    num_training_steps: int,
) -> LambdaLR:
    """Build the step-based warmup and cosine schedule used by MolmoAct2.

    When training has fewer optimizer updates than the configured decay
    horizon, warmup and decay are scaled proportionally to fit the run.

    Returns:
        LambdaLR: The learning rate scheduler instance.

    Raises:
        ValueError: If any of the input arguments are invalid.
    """
    if num_warmup_steps < 0:
        msg = f"num_warmup_steps must be >= 0, got {num_warmup_steps}."
        raise ValueError(msg)
    if num_decay_steps < 1:
        msg = f"num_decay_steps must be >= 1, got {num_decay_steps}."
        raise ValueError(msg)
    if num_training_steps < 1:
        msg = f"num_training_steps must be >= 1, got {num_training_steps}."
        raise ValueError(msg)
    if peak_lr <= 0:
        msg = f"peak_lr must be > 0, got {peak_lr}."
        raise ValueError(msg)
    if not 0 <= decay_lr < peak_lr:
        msg = f"decay_lr must be in [0, peak_lr), got decay_lr={decay_lr}, peak_lr={peak_lr}."
        raise ValueError(msg)

    decay_steps = min(num_decay_steps, num_training_steps)
    warmup_steps = num_warmup_steps
    if num_training_steps < num_decay_steps:
        warmup_steps = int(num_warmup_steps * num_training_steps / num_decay_steps)
    warmup_steps = min(warmup_steps, max(decay_steps - 1, 0))
    alpha = decay_lr / peak_lr

    def lr_lambda(current_step: int) -> float:
        step = max(current_step + 1, 0)
        if warmup_steps > 0 and step < warmup_steps:
            return step / warmup_steps
        if step >= decay_steps:
            return alpha
        cosine_span = decay_steps - warmup_steps
        cosine_step = step - warmup_steps
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * cosine_step / cosine_span))
        return alpha + (1.0 - alpha) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


class MolmoAct2AdamW(torch.optim.AdamW):
    """AdamW with independent group clipping and compensated BF16 updates."""

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        *,
        group_grad_clip_norm: float,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize the optimizer.

        Raises:
            ValueError: If the group clipping norm is not positive.
        """
        if group_grad_clip_norm <= 0:
            msg = f"group_grad_clip_norm must be positive, got {group_grad_clip_norm}."
            raise ValueError(msg)
        super().__init__(params, **kwargs)
        self.group_grad_clip_norm = float(group_grad_clip_norm)

    def _clip_grad_groups(self) -> tuple[Tensor, ...]:
        norms = []
        for group in self.param_groups:
            parameters = [parameter for parameter in group["params"] if parameter.grad is not None]
            if parameters:
                norms.append(torch.nn.utils.clip_grad_norm_(parameters, self.group_grad_clip_norm))
        return tuple(norms)

    @staticmethod
    def _has_nonfinite_gradient(norms: tuple[Tensor, ...]) -> bool:
        nonfinite = any(not bool(torch.isfinite(norm).all()) for norm in norms)
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return nonfinite

        device = (
            torch.device("cuda", torch.cuda.current_device())
            if "nccl"
            in str(
                torch.distributed.get_backend(),
            ).lower()
            else torch.device("cpu")
        )
        flag = torch.tensor(nonfinite, device=device, dtype=torch.int32)
        torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
        return bool(flag.item())

    def _step_non_bfloat16(self) -> None:
        original_groups: list[list[Tensor]] = []
        try:
            for group in self.param_groups:
                parameters = group["params"]
                original_groups.append(parameters)
                group["params"] = [parameter for parameter in parameters if parameter.dtype != torch.bfloat16]
            super().step()
        finally:
            for group, parameters in zip(self.param_groups, original_groups, strict=True):
                group["params"] = parameters

    def _step_bfloat16(self) -> None:  # noqa: PLR0914
        for group in self.param_groups:
            bf16_group = dict(group)
            bf16_group["params"] = [parameter for parameter in group["params"] if parameter.dtype == torch.bfloat16]
            parameters: list[Tensor] = []
            gradients: list[Tensor] = []
            averages: list[Tensor] = []
            squared_averages: list[Tensor] = []
            max_squared_averages: list[Tensor] = []
            steps: list[Tensor] = []
            self._init_group(
                bf16_group,
                parameters,
                gradients,
                averages,
                squared_averages,
                max_squared_averages,
                steps,
            )

            beta1, beta2 = group["betas"]
            learning_rate = group["lr"]
            if torch.is_tensor(learning_rate):
                learning_rate = learning_rate.item()
            learning_rate = float(learning_rate)

            for index, parameter in enumerate(parameters):
                gradient = -gradients[index] if group["maximize"] else gradients[index]
                state = self.state[parameter]
                compensation = state.setdefault("compensation", torch.zeros_like(parameter))
                steps[index].add_(1)
                step = steps[index].item()
                averages[index].lerp_(gradient, 1 - beta1)
                squared_averages[index].mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)

                squared_average = squared_averages[index]
                if group["amsgrad"]:
                    torch.maximum(max_squared_averages[index], squared_average, out=max_squared_averages[index])
                    squared_average = max_squared_averages[index]
                denominator = (squared_average.sqrt() / (1 - beta2**step) ** 0.5).add_(group["eps"])
                effective = parameter.float().add_(compensation)
                if group["weight_decay"]:
                    effective.mul_(1 - learning_rate * group["weight_decay"])
                effective.addcdiv_(averages[index], denominator, value=-learning_rate / (1 - beta1**step))
                parameter.copy_(effective)
                compensation.copy_(effective.sub_(parameter))

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Clip gradients and apply one optimizer update.

        Returns:
            The optional closure loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        norms = self._clip_grad_groups()
        if self._has_nonfinite_gradient(norms):
            self.zero_grad(set_to_none=True)
            return loss
        self._step_non_bfloat16()
        self._step_bfloat16()
        return loss
