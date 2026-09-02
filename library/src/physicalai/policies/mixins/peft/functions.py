# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared PEFT (LoRA/DoRA) utilities for Studio policies.

This module wraps ``peft.inject_adapter_in_model`` rather than ``peft.get_peft_model``
so that adapters are injected *in place*: the wrapped module keeps its original type and
state-dict key names (targeted ``nn.Linear`` submodules simply become
``peft.tuners.lora.Linear`` instances that still respond to attribute lookups like
``.weight`` via the underlying base layer). This preserves compatibility with:

- Raw ``state_dict()`` based checkpointing/export (no ``base_model.model.`` prefix).
- Direct submodule dtype probes elsewhere in the codebase (e.g. Pi05's joint attention
  fast path), since ``BaseTunerLayer`` proxies ``.weight`` to the base layer.
- ``torch.jit`` / ``torch.onnx`` / ``torch.export`` tracing of the plain module tree.

DoRA (Weight-Decomposed Low-Rank Adaptation) is also supported via
``LoraConfig(use_dora=True)``; see ``build_lora_config``'s ``use_dora`` argument.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import peft
import torch
from peft.tuners.tuners_utils import BaseTunerLayer

if TYPE_CHECKING:
    from torch import nn

logger = logging.getLogger(__name__)


def build_lora_config(
    *,
    rank: int,
    alpha: int,
    dropout: float,
    target_modules: str | list[str] | tuple[str, ...],
    init_lora_weights: bool
    | Literal[
        "gaussian",
        "eva",
        "olora",
        "pissa",
        "pissa_niter_[number of iters]",
        "corda",
        "loftq",
        "orthogonal",
        "mica",
    ] = True,
    use_dora: bool = False,
) -> peft.LoraConfig:
    """Build a ``peft.LoraConfig`` for injection into a model.

    Args:
        rank: LoRA rank (dimension of the low-rank decomposition). Must be > 0.
        alpha: LoRA alpha scaling factor (``scaling = alpha / rank``).
        dropout: Dropout probability applied to LoRA inputs.
        target_modules: Either a regex string or a list/tuple of module name suffixes
            matched against submodule names, following PEFT's `target_modules` semantics.
        init_lora_weights: PEFT's adapter initialization strategy. Defaults to ``True``
            (standard Kaiming-uniform A / zero B init).
        use_dora: Enable DoRA (Weight-Decomposed Low-Rank Adaptation), which
            additionally learns a per-column magnitude vector on top of the LoRA
            direction update. Slightly more compute/memory than plain LoRA but
            typically improves quality at low ranks. See arxiv.org/abs/2402.09353.

    Returns:
        A configured ``peft.LoraConfig`` instance.
    """
    target = list(target_modules) if isinstance(target_modules, tuple) else target_modules
    return peft.LoraConfig(  # pyrefly: ignore[missing-attribute]
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target,
        bias="none",
        init_lora_weights=init_lora_weights,
        use_dora=use_dora,
    )


def inject_lora(
    module: nn.Module,
    lora_config: peft.LoraConfig,
    *,
    adapter_dtype: torch.dtype | None = torch.float32,
) -> nn.Module:
    """Inject LoRA adapters into ``module`` in place, freezing all base parameters.

    Unlike ``peft.get_peft_model``, this does not wrap ``module`` in a ``PeftModel``:
    the original module tree, type, and state-dict key names are preserved. Targeted
    ``nn.Linear`` submodules become ``peft.tuners.lora.Linear`` instances in place.

    Args:
        module: The model (or submodule) to inject adapters into.
        lora_config: A ``peft.LoraConfig`` (see :func:`build_lora_config`).
        adapter_dtype: If set, cast newly created LoRA parameters to this dtype
            (recommended when the base model is bfloat16, to avoid low-precision
            adapter training). If ``None``, adapters inherit the base layer's dtype.

    Returns:
        The same ``module`` instance, mutated in place.

    Raises:
        RuntimeError: If no target modules matched (target_modules matched nothing).
    """
    # Freeze all existing parameters; only newly injected adapter params should train.
    for param in module.parameters():
        param.requires_grad = False

    try:
        peft.inject_adapter_in_model(lora_config, module)  # pyrefly: ignore[missing-attribute]
    except ValueError as e:
        msg = (
            "LoRA injection matched zero target modules. Check that "
            "`lora_target_modules` matches at least one Linear submodule name."
        )
        raise RuntimeError(msg) from e

    if adapter_dtype is not None:
        for name, param in module.named_parameters():
            if "lora_" in name:
                param.data = param.data.to(dtype=adapter_dtype)

    num_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    if num_trainable == 0:
        msg = (
            "LoRA injection matched zero trainable parameters. Check that "
            "`lora_target_modules` matches at least one Linear submodule name."
        )
        raise RuntimeError(msg)

    log_trainable_parameters(module)
    return module


def log_trainable_parameters(module: nn.Module) -> None:
    """Log the number and percentage of trainable parameters in ``module``."""
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total = sum(p.numel() for p in module.parameters())
    pct = 100 * trainable / total if total else 0.0
    logger.info("Trainable parameters: %d / %d (%.4f%%)", trainable, total, pct)


def is_lora_injected(module: nn.Module) -> bool:
    """Return True if any submodule of ``module`` is a PEFT tuner layer.

    Returns:
        True if LoRA (or another PEFT method) has been injected into ``module``.
    """
    return any(isinstance(m, BaseTunerLayer) for m in module.modules())


def merge_lora_(module: nn.Module) -> None:
    """Merge LoRA adapters into their base layers in place, replacing tuner wrappers.

    After calling this, ``module`` behaves as a plain (adapter-free) network with the
    adaptation folded into the base layer weights. Intended for use on a disposable
    copy of a model prior to export, since merging is lossy under low precision and
    is not reversible on the mutated instance without keeping a copy of the adapters.

    Args:
        module: The model to merge adapters into, mutated in place.
    """
    for parent in list(module.modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, BaseTunerLayer):
                child.merge(safe_merge=False)
                base_layer = child.get_base_layer()
                setattr(parent, child_name, base_layer)
