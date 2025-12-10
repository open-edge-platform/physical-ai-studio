# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""LoRA (Low-Rank Adaptation) utilities for Pi0/Pi0.5 models.

This module provides LoRA integration for efficient fine-tuning of
PaliGemma and action expert models.

Based on the PEFT library with simplified configuration.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import nn

logger = logging.getLogger(__name__)


def apply_lora(
    model: nn.Module,
    *,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.1,
    target_modules: tuple[str, ...] = ("q_proj", "v_proj", "k_proj", "o_proj"),
    bias: str = "none",
    modules_to_save: list[str] | None = None,
) -> nn.Module:
    """Apply LoRA adaptation to a model using PEFT.

    This function wraps the model with LoRA layers for efficient fine-tuning.
    Only the LoRA parameters will be trainable.

    Args:
        model: The model to apply LoRA to.
        rank: LoRA rank (dimension of low-rank matrices).
        alpha: LoRA alpha scaling factor.
        dropout: Dropout rate for LoRA layers.
        target_modules: Module names to apply LoRA to.
        bias: Whether to train bias parameters ("none", "all", "lora_only").
        modules_to_save: Additional modules to keep trainable (not LoRA-adapted).

    Returns:
        Model wrapped with LoRA layers.

    Raises:
        ImportError: If peft library is not installed.

    Example:
        >>> from transformers import AutoModel
        >>> model = AutoModel.from_pretrained("google/paligemma-3b-pt-224")
        >>> model = apply_lora(model, rank=16, alpha=32)
    """
    try:
        from peft import LoraConfig as PeftLoraConfig  # noqa: PLC0415
        from peft import get_peft_model  # noqa: PLC0415
    except ImportError as e:
        msg = "LoRA requires the peft library. Install with: uv pip install peft"
        raise ImportError(msg) from e

    # Create PEFT LoRA config
    peft_config = PeftLoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(target_modules),
        bias=bias,
        task_type="CAUSAL_LM",
        modules_to_save=modules_to_save,
    )

    # Apply LoRA
    model = get_peft_model(model, peft_config)

    # Log trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "LoRA applied: %d trainable params (%.2f%% of %d total)",
        trainable_params,
        100 * trainable_params / total_params,
        total_params,
    )

    return model


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """Merge LoRA weights into the base model.

    After training, this merges the LoRA adaptations into the base weights
    for efficient inference without the LoRA overhead.

    Args:
        model: Model with LoRA layers.

    Returns:
        Model with merged weights (LoRA removed).

    Raises:
        ImportError: If peft library is not installed.
    """
    try:
        from peft import PeftModel  # noqa: PLC0415
    except ImportError as e:
        msg = "LoRA requires the peft library. Install with: pip install peft"
        raise ImportError(msg) from e

    if isinstance(model, PeftModel):
        logger.info("Merging LoRA weights into base model")
        model = model.merge_and_unload()

    return model


def get_lora_state_dict(model: nn.Module) -> dict[str, Any]:
    """Extract only LoRA parameters from model state dict.

    Useful for saving only the LoRA weights without the full model.

    Args:
        model: Model with LoRA layers.

    Returns:
        State dict containing only LoRA parameters.
    """
    state_dict = model.state_dict()
    return {k: v for k, v in state_dict.items() if "lora" in k.lower()}
