# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Model surgery helpers on top of diffusers Cosmos3 transformer.

Provides utilities for making the omni transformer trainable as a policy:
attaching LoRA/DoRA adapters (peft) or unfreezing the generation tower (full),
initializing a domain's action head, and splitting trainable parameters into
head vs. base groups for differential learning rates.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from peft import LoraConfig
from safetensors.torch import load_file

if TYPE_CHECKING:
    from torch import nn

    from .pipeline import PolicyPipelineWithState

logger = logging.getLogger(__name__)

# nn.Linear projections inside every MoT attention block that LoRA adapts.
LORA_TARGETS = [
    "to_q",
    "to_k",
    "to_v",
    "to_out",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_add_out",
]

# Domain-aware action head, trained full-rank (DomainAwareLinear is not nn.Linear).
HEAD_KEYS = ("action_proj_in", "action_proj_out", "action_modality_embed")

# Generation tower (cosmos-framework "moe_gen" branch) + vision I/O + time embed:
# parameter-name substrings the canonical full fine-tuning recipe targets.
GEN_TOWER_KEYS = (
    "moe_gen",  # gen MLP experts + moe_gen norms
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_add_out",  # gen-branch attention
    "norm_added_q",
    "norm_added_k",  # gen-branch qk norms
    "time_embedder",
    "proj_in",
    "proj_out",  # time embed + vision I/O
)


def init_domain_action_head(tf: nn.Module, domain_id: int) -> None:
    """Initialize domain action head weights for the given domain id.

    Args:
        tf: Cosmos3 transformer model.
        domain_id: Integer embodiment domain identifier.
    """
    for proj_name in ("action_proj_in", "action_proj_out"):
        proj = getattr(tf, proj_name)
        weight_view = proj.fc.weight.data.view(proj.num_domains, proj.input_size, proj.output_size)
        torch.nn.init.xavier_uniform_(weight_view[domain_id])
        proj.bias.weight.data[domain_id].zero_()
    embed = getattr(tf, "action_modality_embed", None)
    if embed is not None and hasattr(embed, "data"):
        embed.data.zero_()


def configure_trainable(
    tf: nn.Module,
    mode: str,
    *,
    rank: int = 32,
    alpha_scale: float = 1.0,
    dora: bool = False,
) -> None:
    """Freeze the transformer backbone and enable trainable weights for the chosen mode.

    Args:
        tf: Cosmos3 transformer model.
        mode: Training mode ("peft" or "full").
        rank: LoRA/DoRA rank.
        alpha_scale: Scaling factor for LoRA alpha.
        dora: Whether to use Weight-Decomposed Low-Rank Adaptation (DoRA).
    """
    tf.requires_grad_(requires_grad=False)
    if mode == "full":
        for name, param in tf.named_parameters():
            if any(key in name for key in GEN_TOWER_KEYS + HEAD_KEYS):
                param.requires_grad_(requires_grad=True)
    else:
        lora_cfg = LoraConfig(
            r=rank,
            lora_alpha=round(alpha_scale * rank),
            target_modules=LORA_TARGETS,
            use_dora=dora,
        )
        add_adapter_fn = getattr(tf, "add_adapter", None)
        if callable(add_adapter_fn):
            add_adapter_fn(lora_cfg)
        for name, param in tf.named_parameters():
            if any(key in name for key in HEAD_KEYS):
                param.requires_grad_(requires_grad=True)


def split_trainable_params(tf: nn.Module) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Split trainable parameters into (base, head) groups for per-group learning rates.

    Args:
        tf: Cosmos3 transformer model.

    Returns:
        Tuple of (base_parameters_list, head_parameters_list).
    """
    head_params: list[torch.nn.Parameter] = []
    base_params: list[torch.nn.Parameter] = []
    for name, param in tf.named_parameters():
        if param.requires_grad:
            if any(k in name for k in HEAD_KEYS):
                head_params.append(param)
            else:
                base_params.append(param)
    return base_params, head_params


def load_finetuned(
    pipe: PolicyPipelineWithState,
    adapter: str | Path,
    domain: str,
    head: str | Path | None = None,
) -> dict[str, Any]:
    """Restore fine-tuned weights onto pipe.transformer.

    Args:
        pipe: Cosmos3 pipeline instance.
        adapter: Path to adapter directory.
        domain: Embodiment domain identifier.
        head: Optional explicit path to the domain head checkpoint file.

    Returns:
        Head checkpoint dictionary containing metadata and normalization bounds.
    """
    adapter_path = Path(adapter)
    tf = pipe.transformer
    full_path = adapter_path / "transformer_full.pt"
    if full_path.exists():
        # Security rule #10: weights_only=True
        full_weights = torch.load(full_path, map_location="cpu", weights_only=True)
        tf.load_state_dict(full_weights, strict=False)
    else:
        lora_safetensors = adapter_path / "pytorch_lora_weights.safetensors"
        if lora_safetensors.exists():
            load_lora_adapter_fn = getattr(tf, "load_lora_adapter", None)
            if callable(load_lora_adapter_fn):
                load_lora_adapter_fn(load_file(str(lora_safetensors)), prefix=None)

    head_path = Path(head) if head is not None else adapter_path / f"{domain}_head.pt"
    ckpt: dict[str, Any] = torch.load(head_path, map_location="cpu", weights_only=True)
    tf.load_state_dict(ckpt["head"], strict=False)
    return ckpt
