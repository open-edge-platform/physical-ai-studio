# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""LoRA utilities for Pi0/Pi0.5 models."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch import nn


def apply_lora(
    model: nn.Module,
    *,
    rank: int,
    alpha: int,
    dropout: float,
    target_modules: Iterable[str],
) -> nn.Module:
    """Apply LoRA adapters to target modules in a model.

    Args:
        model: The model to apply LoRA to.
        rank: LoRA rank (dimension of low-rank matrices).
        alpha: LoRA alpha scaling factor.
        dropout: Dropout rate for LoRA layers.
        target_modules: Module names to apply LoRA to.

    Returns:
        Model wrapped with LoRA layers.

    Raises:
        ImportError: If peft library is not installed.
    """
    try:
        from peft import LoraConfig, get_peft_model  # noqa: PLC0415
    except ImportError as e:
        msg = "LoRA requires peft. Install with: pip install peft"
        raise ImportError(msg) from e

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(target_modules),
        bias="none",
    )
    return get_peft_model(model, lora_config)
