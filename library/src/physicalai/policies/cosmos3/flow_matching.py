# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Rectified flow matching sequence packing and training step for Cosmos3."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch.nn import functional

from .pipeline import state_action_mrope_ids

if TYPE_CHECKING:
    from torch import nn

    from .pipeline import PolicyPipelineWithState


def build_action_tokens(
    paradigm: str,
    action: torch.Tensor,
    state: torch.Tensor,
    chunk: int,
    action_dim: int,
    raw_dim: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Construct clean action-stream x0 tensor for the specified training paradigm.

    For 'policy', prepends a clean current-state token (use_state, chunk + 1 tokens).
    For 'fd' and 'id', uses bare chunk length tokens.

    Args:
        paradigm: Denoising paradigm ("policy", "fd", or "id").
        action: Normalized action sequence tensor of shape (chunk, raw_dim).
        state: Normalized initial state tensor.
        chunk: Action prediction chunk length.
        action_dim: Padded action dimension.
        raw_dim: Physical embodiment action dimension.
        device: Target device for the constructed tensor.

    Returns:
        Tensor of shape (chunk + 1, action_dim) for policy, or (chunk, action_dim) for fd/id.
    """
    if paradigm in {"fd", "id"}:
        x0 = torch.zeros(chunk, action_dim, device=device, dtype=torch.float32)
        x0[:, :raw_dim] = action
        return x0

    x0 = torch.zeros(chunk + 1, action_dim, device=device, dtype=torch.float32)
    x0[0, : state.shape[0]] = state
    x0[1:, :raw_dim] = action
    return x0


def build_pack(
    pipe: PolicyPipelineWithState,
    paradigm: str,
    x0_vision: torch.Tensor,
    prompt: str,
    chunk: int,
    height: int,
    width: int,
    fps: int,
    action_dim: int,
    device: torch.device | str,
) -> dict[str, Any]:
    """Construct fixed-shape sequence pack dictionary for one training paradigm.

    Args:
        pipe: PolicyPipelineWithState instance.
        paradigm: Training objective name ("policy", "fd", or "id").
        x0_vision: Representative clean vision latent tensor used for temporal dimensions.
        prompt: Task instruction conditioning string.
        chunk: Action chunk length.
        height: Image frame height.
        width: Image frame width.
        fps: Video and action frame rate.
        action_dim: Padded action dimension.
        device: Target execution device.

    Returns:
        Dictionary containing sequence segments, position IDs, sequence lengths, and masks.
    """
    latent_t = x0_vision.shape[2]
    n_action = chunk if paradigm in {"fd", "id"} else chunk + 1
    vision_cond = list(range(latent_t)) if paradigm == "id" else [0]
    action_cond = list(range(chunk)) if paradigm == "fd" else ([0] if paradigm == "policy" else [])

    cond_ids, _ = pipe.tokenize_prompt(
        prompt,
        None,
        num_frames=chunk + 1,
        height=height,
        width=width,
        fps=fps,
        action_mode="policy",
    )
    # Access pipeline segment preparation methods
    text = pipe._prepare_text_segment(cond_ids, device=device)  # ruff: ignore[private-member-access]
    vis = pipe._prepare_vision_segment(  # ruff: ignore[private-member-access]
        input_vision_tokens=x0_vision,
        has_image_condition=True,
        mrope_offset=text["vision_start_temporal_offset"],
        vision_fps=float(fps),
        curr=text["und_len"],
        device=device,
        condition_frame_indexes=vision_cond,
    )
    act = pipe._prepare_action_segment(  # ruff: ignore[private-member-access]
        input_action_tokens=torch.zeros(n_action, action_dim, device=device),
        condition_frame_indexes=action_cond,
        mrope_offset=text["vision_start_temporal_offset"],
        action_fps=float(fps),
        curr=text["und_len"] + vis["num_vision_tokens"],
        device=device,
    )

    if paradigm == "policy":
        # Case B parity: state token aligns to frame 0, actions land at 1..chunk
        act["action_mrope_ids"] = state_action_mrope_ids(
            pipe,
            act["action_len"],
            text["vision_start_temporal_offset"],
            float(fps),
            device,
        )

    vision_keep = torch.zeros((latent_t, 1, 1), device=device, dtype=torch.float32)
    for i in vision_cond:
        vision_keep[i, 0, 0] = 1.0

    action_keep = torch.zeros((n_action, 1), device=device, dtype=torch.float32)
    for i in action_cond:
        action_keep[i, 0] = 1.0

    position_ids = torch.cat(
        [text["text_mrope_ids"], vis["vision_mrope_ids"], act["action_mrope_ids"]],
        dim=1,
    )
    seq_len = text["und_len"] + vis["num_vision_tokens"] + act["action_len"]

    return {
        "text": text,
        "vis": vis,
        "act": act,
        "position_ids": position_ids,
        "seq_len": seq_len,
        "vision_keep": vision_keep,
        "action_keep": action_keep,
        "train_video": paradigm != "id",
        "train_action": paradigm != "fd",
    }


def flow_matching_step(  # ruff: ignore[too-many-locals]
    tf: nn.Module,
    pack: dict[str, Any],
    x0_vision: torch.Tensor,
    x0_action: torch.Tensor,
    domain_id: torch.Tensor,
    dtype: torch.dtype,
    raw_dim: int,
    action_weight: float,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute one rectified flow matching step on vision and action sequences.

    Args:
        tf: Cosmos3 transformer model.
        pack: Sequence pack dictionary produced by `build_pack`.
        x0_vision: Clean vision latents.
        x0_action: Clean action tokens.
        domain_id: Embodiment domain identifier tensor.
        dtype: Computation precision.
        raw_dim: Physical embodiment action dimension.
        action_weight: Loss weight multiplier for action denoising.
        device: Target execution device.

    Returns:
        Tuple of (total_loss, loss_vision, loss_action).
    """
    text, vis, act = pack["text"], pack["vis"], pack["act"]
    vision_keep, action_keep = pack["vision_keep"], pack["action_keep"]

    sigma = torch.rand(1, device=device)
    noise_v = torch.randn_like(x0_vision)
    xt_vision = vision_keep * x0_vision + (1.0 - vision_keep) * ((1.0 - sigma) * x0_vision + sigma * noise_v)

    noise_a = torch.randn_like(x0_action)
    xt_action = action_keep * x0_action + (1.0 - action_keep) * ((1.0 - sigma) * x0_action + sigma * noise_a)
    xt_action[:, raw_dim:] = 0
    timestep = float(sigma.item() * 1000.0)

    preds_vision, _, preds_action = tf(
        input_ids=text["input_ids"],
        text_indexes=text["text_indexes"],
        position_ids=pack["position_ids"],
        und_len=text["und_len"],
        sequence_length=pack["seq_len"],
        vision_tokens=[xt_vision.to(dtype)],
        vision_token_shapes=vis["vision_token_shapes"],
        vision_sequence_indexes=vis["vision_sequence_indexes"],
        vision_mse_loss_indexes=vis["vision_mse_loss_indexes"],
        vision_timesteps=torch.full((vis["num_noisy_vision_tokens"],), timestep, device=device),
        vision_noisy_frame_indexes=vis["vision_noisy_frame_indexes"],
        action_tokens=[xt_action.to(dtype)],
        action_token_shapes=act["action_token_shapes"],
        action_sequence_indexes=act["action_sequence_indexes"],
        action_mse_loss_indexes=act["action_mse_loss_indexes"],
        action_timesteps=torch.full((act["num_noisy_action_tokens"],), timestep, device=device),
        action_noisy_frame_indexes=act["action_noisy_frame_indexes"],
        action_domain_ids=[domain_id],
        return_dict=False,
    )

    loss_vision = x0_vision.new_zeros(())
    loss_action = x0_vision.new_zeros(())
    if pack["train_video"]:
        target_vision = (noise_v - x0_vision) * (1.0 - vision_keep)
        loss_vision = functional.mse_loss(preds_vision[0].float(), target_vision)
    if pack["train_action"]:
        target_action = ((noise_a - x0_action) * (1.0 - action_keep))[:, :raw_dim]
        loss_action = functional.mse_loss(preds_action[0][:, :raw_dim].float(), target_action)

    loss = loss_vision + action_weight * loss_action
    return loss, loss_vision, loss_action
