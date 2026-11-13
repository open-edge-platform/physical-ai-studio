# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Fused Triton kernel for the Kahan-compensated bf16 AdamW update."""

# Triton reads argument annotations, so the kernel arguments stay unannotated.
# ruff: noqa: ANN001

from __future__ import annotations

from typing import TYPE_CHECKING

import triton
import triton.language as tl

if TYPE_CHECKING:
    import torch

_BLOCK_SIZE = 2048


@triton.jit
def _kahan_adamw_kernel(
    weight_ptr,
    grad_ptr,
    compensation_ptr,
    exp_avg_ptr,
    exp_avg_sq_ptr,
    n_elements,
    beta1,
    beta2,
    step_size,
    bias_correction2_sqrt,
    eps,
    decay,
    BLOCK_SIZE: tl.constexpr,  # noqa: N803
) -> None:
    offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    weight = tl.load(weight_ptr + offsets, mask=mask).to(tl.float32)
    grad = tl.load(grad_ptr + offsets, mask=mask).to(tl.float32)
    compensation = tl.load(compensation_ptr + offsets, mask=mask).to(tl.float32)
    exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask).to(tl.float32)
    exp_avg_sq = tl.load(exp_avg_sq_ptr + offsets, mask=mask).to(tl.float32)
    exp_avg += (1.0 - beta1) * (grad - exp_avg)
    exp_avg_sq += (1.0 - beta2) * (grad * grad - exp_avg_sq)
    updated = (weight + compensation) * decay - step_size * exp_avg / (
        tl.sqrt(exp_avg_sq) / bias_correction2_sqrt + eps
    )
    rounded = updated.to(weight_ptr.dtype.element_ty)
    tl.store(weight_ptr + offsets, rounded, mask=mask)
    tl.store(
        compensation_ptr + offsets,
        (updated - rounded.to(tl.float32)).to(compensation_ptr.dtype.element_ty),
        mask=mask,
    )
    tl.store(exp_avg_ptr + offsets, exp_avg.to(exp_avg_ptr.dtype.element_ty), mask=mask)
    tl.store(exp_avg_sq_ptr + offsets, exp_avg_sq.to(exp_avg_sq_ptr.dtype.element_ty), mask=mask)


def kahan_adamw_update(
    weight: torch.Tensor,
    grad: torch.Tensor,
    compensation: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    beta1: float,
    beta2: float,
    step_size: float,
    bias_correction2_sqrt: float,
    eps: float,
    decay: float,
) -> None:
    """Apply one Kahan-compensated AdamW update in place. All tensors must be contiguous and on one GPU."""
    n_elements = weight.numel()
    grid = (triton.cdiv(n_elements, _BLOCK_SIZE),)
    _kahan_adamw_kernel[grid](
        weight,
        grad,
        compensation,
        exp_avg,
        exp_avg_sq,
        n_elements,
        beta1,
        beta2,
        step_size,
        bias_correction2_sqrt,
        eps,
        decay,
        BLOCK_SIZE=_BLOCK_SIZE,  # pyrefly: ignore[bad-argument-type]
        num_warps=8,  # pyrefly: ignore[unexpected-keyword]
    )
