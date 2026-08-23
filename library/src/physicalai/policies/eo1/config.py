# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Copyright 2026 The HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the EO-1 model.

EO-1 pairs a Qwen2.5-VL vision-language backbone with a continuous flow-matching action head.
Each robot-control sample is formatted as a multimodal conversation: camera frames go to
Qwen2.5-VL, the robot state occupies a state placeholder token and the future action chunk
occupies `chunk_size` action placeholder tokens that the flow head denoises.

For CLI usage, use the YAML config in `configs/physicalai/eo1.yaml`:

    physicalai fit --config configs/physicalai/eo1.yaml

Example (API):
    >>> from physicalai.policies.eo1 import EO1Config
    >>> config = EO1Config(chunk_size=16, n_action_steps=8)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from physicalai.config import Config

DTYPE_CHOICES = ("auto", "bfloat16", "float32")


@dataclass(frozen=True)
class EO1Config(Config):
    """Configuration for the EO-1 vision-language-action model.

    Attributes:
        n_obs_steps: Number of observation steps to use. Defaults to 1.
        chunk_size: Size of the predicted action chunk. Also the number of action placeholder
            tokens inserted into the prompt. Defaults to 8.
        n_action_steps: Number of action steps executed per model invocation. Defaults to 8.
        vlm_base: Name or path of the Qwen2.5-VL backbone.
            Defaults to "Qwen/Qwen2.5-VL-3B-Instruct".
        vlm_config: Serialized `Qwen2_5_VLConfig` read from a published EO-1 checkpoint. When set,
            the backbone is built from it with random weights (the checkpoint then fills them in)
            instead of downloading the `vlm_base` weights. Defaults to None.
        attn_implementation: Attention backend forwarded to the Qwen backbone, e.g. "eager",
            "sdpa", "flash_attention_2". None keeps the backbone default. Defaults to None.
        dtype: Compute dtype requested for the Qwen backbone: "auto" follows the backbone config
            (bf16 for Qwen2.5-VL), "bfloat16" and "float32" force it. The flow head always keeps
            its own parameters in fp32. Defaults to "auto".
        force_fp32_autocast: Whether the flow head runs with autocast disabled so its projections
            stay in fp32 even under mixed precision. Defaults to True.
        gradient_checkpointing: Whether to enable gradient checkpointing on the backbone and the
            flow-head computations. Defaults to False.
        image_min_pixels: Lower bound on the pixel budget the Qwen image processor resizes each
            camera frame to. None leaves the processor default. Defaults to 64 * 28 * 28.
        image_max_pixels: Upper bound on that pixel budget. None leaves the processor default.
            Defaults to 128 * 28 * 28.
        use_fast_processor: Whether to load the torchvision-backed fast image processor.
            Defaults to False.
        max_state_dim: Dimension the robot state is zero-padded to before the flow head. Baked into
            `state_proj`'s weight shape. Defaults to 32.
        max_action_dim: Dimension actions are zero-padded to before the flow head. Baked into
            `action_in_proj` / `action_out_proj` weight shapes. Defaults to 32.
        action_dim: True action dimensionality, resolved from dataset stats during setup. Predicted
            chunks are cropped back to it. Defaults to 7.
        state_dim: True state dimensionality, resolved from dataset stats during setup.
            Defaults to 8.
        num_denoise_steps: Forward-Euler steps used to integrate the velocity field at inference.
            Defaults to 10.
        num_action_layers: Number of linear layers in the action output projector. Defaults to 2.
        action_act: Activation between those layers, resolved through `transformers.ACT2FN`.
            Defaults to "linear".
        time_sampling_beta_alpha: Alpha of the Beta distribution used for timestep sampling.
            Defaults to 1.5.
        time_sampling_beta_beta: Beta of that distribution. Defaults to 1.0.
        time_sampling_scale: Scaling applied to the sampled timesteps. Defaults to 0.999.
        time_sampling_offset: Offset added to the sampled timesteps. Defaults to 0.001.
        min_period: Minimum period of the sine-cosine timestep embedding. Defaults to 4e-3.
        max_period: Maximum period of that embedding. Defaults to 4.0.
        supervise_padding_action_dims: Whether the zero-padded action dimensions between
            `action_dim` and `max_action_dim` contribute to the loss. Defaults to True.
        supervise_padding_actions: Whether padded action rows (past the end of an episode)
            contribute to the loss. When False they are also masked out of backbone attention.
            Defaults to True.
        state_normalization: Normalization applied to the robot state. Defaults to "MEAN_STD".
        action_normalization: Normalization applied to actions. Defaults to "MEAN_STD".
        optimizer_lr: Learning rate for the optimizer. Defaults to 1e-4.
        optimizer_betas: Beta coefficients for AdamW. Defaults to (0.9, 0.999).
        optimizer_eps: Epsilon for numerical stability. Defaults to 1e-8.
        optimizer_weight_decay: Weight decay coefficient. Defaults to 0.1.
        optimizer_grad_clip_norm: Maximum gradient norm. Defaults to 1.0.
        scheduler_warmup_steps: Warmup steps for the scheduler. Defaults to 900.
        scheduler_decay_steps: Decay steps for the scheduler. Defaults to 30000.
        scheduler_decay_lr: Final learning rate after decay. Defaults to 0.0.
    """

    n_obs_steps: int = 1
    chunk_size: int = 8
    n_action_steps: int = 8

    vlm_base: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    vlm_config: dict[str, Any] | None = None
    attn_implementation: str | None = None
    dtype: str = "auto"
    force_fp32_autocast: bool = True
    gradient_checkpointing: bool = False

    image_min_pixels: int | None = 64 * 28 * 28
    image_max_pixels: int | None = 128 * 28 * 28
    use_fast_processor: bool = False

    max_state_dim: int = 32
    max_action_dim: int = 32
    action_dim: int = 7
    state_dim: int = 8

    num_denoise_steps: int = 10
    num_action_layers: int = 2
    action_act: str = "linear"
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0
    supervise_padding_action_dims: bool = True
    supervise_padding_actions: bool = True

    state_normalization: str = "MEAN_STD"
    action_normalization: str = "MEAN_STD"

    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.1
    optimizer_grad_clip_norm: float = 1.0

    scheduler_warmup_steps: int = 900
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 0.0

    def __post_init__(self) -> None:
        """Validate the configuration.

        Raises:
            ValueError: If `n_action_steps` exceeds `chunk_size`, if `dtype` is not one of the
                supported dtype choices, or if the true feature dimensions exceed the padded ones
                the flow head is built for.
        """
        if self.n_action_steps > self.chunk_size:
            msg = (
                f"The chunk size is the upper bound for the number of action steps per model "
                f"invocation. Got {self.n_action_steps} for `n_action_steps` and {self.chunk_size} "
                f"for `chunk_size`."
            )
            raise ValueError(msg)
        if self.dtype not in DTYPE_CHOICES:
            msg = f"Unknown dtype '{self.dtype}'. Supported: {', '.join(DTYPE_CHOICES)}."
            raise ValueError(msg)
        if self.action_dim > self.max_action_dim:
            msg = (
                f"`action_dim` ({self.action_dim}) exceeds `max_action_dim` ({self.max_action_dim}), "
                f"which is the width the flow head is built for. Raise `max_action_dim`."
            )
            raise ValueError(msg)
        if self.state_dim > self.max_state_dim:
            msg = (
                f"`state_dim` ({self.state_dim}) exceeds `max_state_dim` ({self.max_state_dim}), "
                f"which is the width the state projection is built for. Raise `max_state_dim`."
            )
            raise ValueError(msg)

    @property
    def observation_delta_indices(self) -> None:
        """Observation indices relative to the current timestep.

        Returns:
            None: EO-1 conditions on the current frame only.
        """
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        """Action indices relative to the current timestep.

        Returns:
            One index per step of the predicted chunk.
        """
        return list(range(self.chunk_size))
