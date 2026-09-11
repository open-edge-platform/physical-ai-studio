# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Proprioceptive-state (use_state) policy pipeline on top of diffusers.

Keeps checkpoints byte-compatible with the official cosmos-framework "use_state"
layout so they are swappable in either direction. The state token is prepended
as a clean current-state row, giving an action stream of length ``chunk + 1``
(token 0 = state, tokens 1..chunk = the predicted chunk).

The only piece that diverges from the vanilla policy path is the action-segment
mRoPE. diffusers' ``_prepare_action_segment`` hardcodes ``start_frame_offset=1``
(cosmos-framework Case A, no state). The use_state layout is Case B, which uses
``start_frame_offset=0`` so the clean state token aligns with video frame 0 and
the predicted actions land at temporal positions 1..chunk -- identical to the
canonical policy path, just with the extra leading state token.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import get_3d_mrope_ids_vae_tokens

if TYPE_CHECKING:
    from torch import nn

logger = logging.getLogger(__name__)

DEFAULT_MIN_XPU_DRIVER = "1.15.38646"


def check_xpu_driver(
    device: torch.device | str | int | None = None,
    min_xpu_driver: str = DEFAULT_MIN_XPU_DRIVER,
) -> None:
    """Validate that the Intel compute-runtime driver satisfies ``min_xpu_driver``.

    Args:
        device: Target device or device index.
        min_xpu_driver: Minimum supported driver version string.

    Raises:
        RuntimeError: If the driver cannot be queried or is older than ``min_xpu_driver``.
    """
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        return

    dev_idx = 0
    if isinstance(device, torch.device):
        if device.type != "xpu":
            return
        dev_idx = device.index if device.index is not None else 0
    elif isinstance(device, str):
        if not device.startswith("xpu"):
            return
        parts = device.split(":")
        if len(parts) > 1 and parts[1].isdigit():
            dev_idx = int(parts[1])
    elif isinstance(device, int):
        dev_idx = device

    try:
        driver = torch.xpu.get_device_properties(dev_idx).driver_version
    except Exception as exc:
        msg = f"[xpu] could not query XPU driver version on device {dev_idx}: {exc}"
        raise RuntimeError(msg) from exc

    def version_tuple(ver: str) -> tuple[int, ...]:
        return tuple(int(n) for n in re.findall(r"\d+", ver))

    if version_tuple(driver) < version_tuple(min_xpu_driver):
        msg = (
            f"[xpu] driver {driver} is too old; Cosmos3 requires >= {min_xpu_driver} "
            "(older drivers ship broken nonzero/VAE kernels that corrupt output). "
            "Please update the Intel compute-runtime driver."
        )
        raise RuntimeError(msg)


require_xpu_driver = check_xpu_driver


def state_action_mrope_ids(
    pipe: Cosmos3OmniPipeline,
    action_len: int,
    mrope_offset: int,
    action_fps: float,
    device: torch.device | str,
) -> torch.Tensor:
    """Calculate mRoPE ids for the state-prepended action segment (cosmos-framework Case B).

    Same call as diffusers' ``_prepare_action_segment`` but with
    ``start_frame_offset=0`` instead of ``1`` so the state token (index 0) lands
    on video frame 0 and the predicted actions keep the canonical 1..chunk
    positions.

    Args:
        pipe: Cosmos3 pipeline instance with transformer and vae config.
        action_len: Total length of action sequence tokens.
        mrope_offset: Temporal offset from preceding modalities.
        action_fps: Effective action frame rate.
        device: Device to place the calculated IDs on.

    Returns:
        Tensor of 3D mRoPE position IDs.
    """
    config = pipe.transformer.config
    effective_fps = action_fps if getattr(config, "enable_fps_modulation", False) else None
    ids, _ = get_3d_mrope_ids_vae_tokens(
        grid_t=action_len,
        grid_h=1,
        grid_w=1,
        temporal_offset=mrope_offset,
        reset_spatial_indices=getattr(config, "unified_3d_mrope_reset_spatial_ids", False),
        fps=effective_fps,
        base_fps=float(getattr(config, "base_fps", 24.0)),
        temporal_compression_factor=1,
        base_temporal_compression_factor=pipe.vae_scale_factor_temporal,
        start_frame_offset=0,
    )
    return ids.to(device)


class PolicyPipelineWithState(Cosmos3OmniPipeline):
    """Policy pipeline with optional proprioceptive-state conditioning (cosmos-framework `use_state`).

    With ``current_state`` unset the pipeline operates in standard policy mode: image + text in,
    the full action chunk denoised from noise. With ``current_state`` set (already normalized to
    the model's action space) a clean current-state token is prepended at action index 0 and marked
    conditioning, so the chunk rolls out from a known state and the predicted actions are tokens 1..chunk.
    """

    min_xpu_driver: str = DEFAULT_MIN_XPU_DRIVER
    current_state: torch.Tensor | None = None

    def __init__(
        self,
        transformer: nn.Module,
        text_tokenizer: object,
        vae: nn.Module,
        scheduler: object,
        sound_tokenizer: object = None,
        safety_checker: object = None,
        *,
        enable_safety_checker: bool = False,
        default_use_system_prompt: bool = True,
        use_native_flow_schedule: bool = False,
        min_xpu_driver: str = DEFAULT_MIN_XPU_DRIVER,
    ) -> None:
        """Initialize PolicyPipelineWithState.

        Args:
            transformer: Transformer backbone module.
            text_tokenizer: Prompt tokenizer.
            vae: VAE image/video autoencoder module.
            scheduler: UniPC multistep diffusion scheduler.
            sound_tokenizer: Optional sound tokenizer.
            safety_checker: Optional safety checker module.
            enable_safety_checker: Whether safety checking is enabled.
            default_use_system_prompt: Whether to append default system prompt.
            use_native_flow_schedule: Whether to use native flow schedule.
            min_xpu_driver: Minimum required Intel compute-runtime driver string.
        """
        super().__init__(
            transformer=transformer,
            text_tokenizer=text_tokenizer,
            vae=vae,
            scheduler=scheduler,
            sound_tokenizer=sound_tokenizer,
            safety_checker=safety_checker,
            enable_safety_checker=enable_safety_checker,
            default_use_system_prompt=default_use_system_prompt,
            use_native_flow_schedule=use_native_flow_schedule,
        )
        self.min_xpu_driver = min_xpu_driver
        dev = getattr(self, "_execution_device", None) or getattr(self, "device", None)
        if dev is not None and getattr(dev, "type", None) == "xpu":
            check_xpu_driver(device=dev, min_xpu_driver=self.min_xpu_driver)

    def to(self, *args: object, **kwargs: object) -> PolicyPipelineWithState:
        """Move pipeline components to target device, checking XPU driver compatibility if applicable.

        Args:
            *args: Positional device arguments.
            **kwargs: Keyword device arguments.

        Returns:
            Pipeline on destination device.
        """
        target_device = args[0] if args else kwargs.get("device")
        if target_device is not None:
            dev_str = str(target_device)
            if "xpu" in dev_str:
                check_xpu_driver(
                    device=target_device,  # type: ignore[arg-type]
                    min_xpu_driver=getattr(self, "min_xpu_driver", DEFAULT_MIN_XPU_DRIVER),
                )

        pipe = super().to(*args, **kwargs)
        dev = getattr(pipe, "_execution_device", None) or getattr(pipe, "device", None)
        if dev is not None and getattr(dev, "type", None) == "xpu":
            check_xpu_driver(
                device=dev,
                min_xpu_driver=getattr(pipe, "min_xpu_driver", DEFAULT_MIN_XPU_DRIVER),
            )
        return pipe

    def check_xpu_driver(self, device: torch.device | str | int | None = None) -> None:
        """Validate that the active or provided XPU device satisfies ``min_xpu_driver``."""
        target_device = device or getattr(self, "_execution_device", None) or getattr(self, "device", None)
        check_xpu_driver(device=target_device, min_xpu_driver=self.min_xpu_driver)

    def prepare_latents(self, *args: object, **kwargs: object) -> tuple[Any, ...]:
        """Prepare latents, prepending current_state row to action latents when configured.

        Args:
            *args: Variable positional arguments forwarded to super().prepare_latents.
            **kwargs: Variable keyword arguments forwarded to super().prepare_latents.

        Returns:
            Tuple of prepared latent tensors and conditioning masks.
        """
        out = list(super().prepare_latents(*args, **kwargs))
        action_latents, action_condition_mask = out[2], out[7]
        if self.current_state is not None and action_latents is not None and action_condition_mask is not None:
            # Prepend a clean current-state token; tokens 1..chunk stay noisy and are the predicted chunk.
            state_row = torch.zeros_like(action_latents[:1])
            state_row[0, : self.current_state.shape[0]] = self.current_state.to(state_row)
            out[2] = torch.cat([state_row, action_latents], dim=0)
            out[7] = torch.cat(
                [action_condition_mask.new_ones(1, action_condition_mask.shape[1]), action_condition_mask],
                dim=0,
            )
            out[11] = [0]
        return tuple(out)

    def _prepare_action_segment(self, *args: object, **kwargs: object) -> dict[str, Any]:
        """Prepare action segment dictionary, aligning mRoPE IDs when using state conditioning.

        Args:
            *args: Variable positional arguments forwarded to super()._prepare_action_segment.
            **kwargs: Variable keyword arguments forwarded to super()._prepare_action_segment.

        Returns:
            Dictionary defining the action sequence segment.
        """
        seg = super()._prepare_action_segment(*args, **kwargs)
        if self.current_state is not None:
            # Realign state-prepended segment to cosmos-framework Case B (start_frame_offset=0)
            seg["action_mrope_ids"] = state_action_mrope_ids(
                self,
                int(seg["action_len"]),
                int(kwargs["mrope_offset"]),  # type: ignore[arg-type]
                float(kwargs["action_fps"]),  # type: ignore[arg-type]
                kwargs["device"],  # type: ignore[arg-type]
            )
        return seg
