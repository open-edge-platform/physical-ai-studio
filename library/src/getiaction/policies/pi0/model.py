# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2025 Physical Intelligence
# SPDX-License-Identifier: Apache-2.0

"""Pi0/Pi0.5 Model - Core PyTorch implementation.

This module provides the core Pi0/Pi0.5 flow matching model as a pure
PyTorch nn.Module. It handles:

1. Image encoding via SigLIP (through PaliGemma)
2. Language encoding via Gemma
3. Action prediction via flow matching with a Gemma action expert

The model supports both Pi0 (continuous state, MLP timestep) and Pi0.5
(discrete state, AdaRMSNorm timestep) variants.

Based on OpenPI's pi0_pytorch.py implementation.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, Literal

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from .components.attention import make_attention_mask_2d, prepare_4d_attention_mask
from .components.gemma import GemmaVariant, PaliGemmaWithExpert

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

logger = logging.getLogger(__name__)


def create_sinusoidal_pos_embedding(
    time: torch.Tensor,
    dimension: int,
    min_period: float,
    max_period: float,
) -> torch.Tensor:
    """Compute sine-cosine positional embeddings for scalar positions.

    Used for encoding flow matching timestep into the model.

    Args:
        time: Timestep tensor of shape (batch_size,).
        dimension: Embedding dimension (must be even).
        min_period: Minimum period for sinusoidal encoding.
        max_period: Maximum period for sinusoidal encoding.

    Returns:
        Positional embeddings of shape (batch_size, dimension).

    Raises:
        ValueError: If dimension is not even or time is not 1D.
    """
    if dimension % 2 != 0:
        msg = f"dimension ({dimension}) must be divisible by 2"
        raise ValueError(msg)

    if time.ndim != 1:
        msg = "The time tensor is expected to be of shape (batch_size,)"
        raise ValueError(msg)

    device = time.device
    # Use float64 for precision, but fall back to float32 if bfloat16 on CPU (unsupported)
    dtype = torch.float64

    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]

    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha: float, beta: float, size: int, device: torch.device) -> torch.Tensor:
    """Sample from Beta distribution.

    Used for sampling flow matching timesteps during training.

    Args:
        alpha: Alpha parameter of Beta distribution.
        beta: Beta parameter of Beta distribution.
        size: Number of samples.
        device: Device for output tensor.

    Returns:
        Samples of shape (size,).
    """
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((size,))


class Pi0Model(nn.Module):
    """Pi0/Pi0.5 Flow Matching Vision-Language-Action Model.

    Pure PyTorch implementation of the Pi0/Pi0.5 model for robot control.
    Uses flow matching to generate continuous action trajectories conditioned
    on visual observations and language instructions.

    Architecture:
        - PaliGemma: SigLIP vision encoder + Gemma language model
        - Action Expert: Smaller Gemma model with optional AdaRMSNorm (Pi0.5)
        - Flow Matching: Euler integration for action generation

    Args:
        variant: Model variant ("pi0" or "pi05").
        paligemma_variant: PaliGemma backbone size.
        action_expert_variant: Action expert size.
        max_action_dim: Maximum action dimension for padding.
        max_state_dim: Maximum state dimension for padding (Pi0 only).
        action_horizon: Number of action steps to predict.
        num_inference_steps: Number of Euler steps for inference.
        dtype: Compute dtype ("bfloat16" or "float32").
        time_beta_alpha: Alpha parameter for timestep Beta distribution.
        time_beta_beta: Beta parameter for timestep Beta distribution.
        time_scale: Scale for timestep sampling.
        time_offset: Offset for timestep sampling.
        time_min_period: Minimum period for sinusoidal timestep encoding.
        time_max_period: Maximum period for sinusoidal timestep encoding.

    Example:
        >>> model = Pi0Model(variant="pi0", max_action_dim=14, action_horizon=50)

        >>> # Training
        >>> loss = model.forward(observation, actions)

        >>> # Inference
        >>> actions = model.sample_actions(device, observation)

        >>> # Create with explicit args
        >>> model = Pi0Model(variant="pi0", max_action_dim=14, action_horizon=50)
    """

    def __init__(
        self,
        variant: Literal["pi0", "pi05"] = "pi0",
        paligemma_variant: GemmaVariant = "gemma_2b",
        action_expert_variant: GemmaVariant = "gemma_300m",
        max_action_dim: int = 32,
        max_state_dim: int = 32,
        action_horizon: int = 50,
        num_inference_steps: int = 10,
        dtype: str = "bfloat16",
        # Flow matching parameters
        time_beta_alpha: float = 1.5,
        time_beta_beta: float = 1.0,
        time_scale: float = 0.999,
        time_offset: float = 0.001,
        time_min_period: float = 4e-3,
        time_max_period: float = 4.0,
    ) -> None:
        """Initialize Pi0 model with explicit parameters.

        Args:
            variant: Model variant ("pi0" or "pi05").
            paligemma_variant: PaliGemma backbone size.
            action_expert_variant: Action expert size.
            max_action_dim: Maximum action dimension for padding.
            max_state_dim: Maximum state dimension for padding (Pi0 only).
            action_horizon: Number of action steps to predict.
            num_inference_steps: Number of Euler steps for inference.
            dtype: Compute dtype ("bfloat16" or "float32").
            time_beta_alpha: Alpha parameter for timestep Beta distribution.
            time_beta_beta: Beta parameter for timestep Beta distribution.
            time_scale: Scale for timestep sampling.
            time_offset: Offset for timestep sampling.
            time_min_period: Minimum period for sinusoidal timestep encoding.
            time_max_period: Maximum period for sinusoidal timestep encoding.
        """
        super().__init__()

        # Store all parameters as instance variables (no config dependency)
        self.variant = variant
        self.paligemma_variant = paligemma_variant
        self.action_expert_variant = action_expert_variant
        self.max_action_dim = max_action_dim
        self.max_state_dim = max_state_dim
        self.action_horizon = action_horizon
        self.num_inference_steps = num_inference_steps
        self.dtype = dtype
        self.time_beta_alpha = time_beta_alpha
        self.time_beta_beta = time_beta_beta
        self.time_scale = time_scale
        self.time_offset = time_offset
        self.time_min_period = time_min_period
        self.time_max_period = time_max_period

        # Derived properties
        self.is_pi05 = variant == "pi05"
        self.use_adarms = self.is_pi05

        # PaliGemma + Action Expert backbone
        self.paligemma_with_expert = PaliGemmaWithExpert(
            paligemma_variant=paligemma_variant,
            action_expert_variant=action_expert_variant,
            use_adarms=self.use_adarms,
            dtype=dtype,
        )

        # Action projection layers (use explicit hparam attribute)
        action_expert_width = self.paligemma_with_expert.action_expert_hidden_size
        self.action_in_proj = nn.Linear(max_action_dim, action_expert_width)
        self.action_out_proj = nn.Linear(action_expert_width, max_action_dim)

        # Timestep conditioning layers
        if self.is_pi05:
            # Pi0.5: MLP for timestep -> AdaRMSNorm conditioning
            self.time_mlp_in = nn.Linear(action_expert_width, action_expert_width)
            self.time_mlp_out = nn.Linear(action_expert_width, action_expert_width)
        else:
            # Pi0: State projection + MLP for action+time fusion
            self.state_proj = nn.Linear(max_state_dim, action_expert_width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_width, action_expert_width)
            self.action_time_mlp_out = nn.Linear(action_expert_width, action_expert_width)

        # Gradient checkpointing flag (runtime state, not in config)
        self._gradient_checkpointing_enabled = False

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        *,
        device: str | torch.device = "cpu",
    ) -> Pi0Model:
        """Load Pi0 model from checkpoint.

        Supports loading from:
        - OpenPI checkpoints (model.safetensors)
        - Native PyTorch checkpoints (.pt, .pth)

        Model architecture is loaded from checkpoint's config.json if available,
        otherwise model defaults are used. This ensures weights match the architecture.

        Args:
            checkpoint_path: Path to checkpoint directory or file.
            device: Device to load model to.

        Returns:
            Loaded Pi0Model with weights restored.

        Examples:
            Load from checkpoint (auto-loads config.json):

            >>> model = Pi0Model.from_pretrained("./checkpoint")

            Load to specific device:

            >>> model = Pi0Model.from_pretrained("./checkpoint", device="cuda:0")

        Note:
            If you need different architecture parameters, create a new model
            with desired params and load weights manually:

            >>> model = Pi0Model(variant="pi0", max_action_dim=14)
            >>> state_dict = torch.load("checkpoint/model.safetensors")
            >>> model.load_state_dict(state_dict, strict=False)
        """
        import json  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        checkpoint_path_obj = Path(checkpoint_path)

        # Load config from checkpoint if available
        config_path = checkpoint_path_obj / "config.json"
        if config_path.exists():
            with Path(config_path).open(encoding="utf-8") as f:
                config_dict = json.load(f)
            # Create model from saved config (ensures weights match architecture)
            model = cls(
                variant=config_dict.get("variant", "pi0"),
                paligemma_variant=config_dict.get("paligemma_variant", "gemma_2b"),
                action_expert_variant=config_dict.get("action_expert_variant", "gemma_300m"),
                max_action_dim=config_dict.get("max_action_dim", 32),
                max_state_dim=config_dict.get("max_state_dim", 32),
                action_horizon=config_dict.get("action_horizon", 50),
                num_inference_steps=config_dict.get("num_inference_steps", 10),
                dtype=config_dict.get("dtype", "bfloat16"),
                time_beta_alpha=config_dict.get("time_beta_alpha", 1.5),
                time_beta_beta=config_dict.get("time_beta_beta", 1.0),
                time_scale=config_dict.get("time_scale", 0.999),
                time_offset=config_dict.get("time_offset", 0.001),
                time_min_period=config_dict.get("time_min_period", 4e-3),
                time_max_period=config_dict.get("time_max_period", 4.0),
            )
        else:
            logger.warning("No config.json found, using model defaults")
            # Use model defaults (weights may not match if checkpoint was trained with different config)
            model = cls()

        # Load weights
        weights_path = checkpoint_path_obj / "model.safetensors"
        if weights_path.exists():
            try:
                from safetensors.torch import load_file  # noqa: PLC0415

                state_dict = load_file(weights_path)
                model.load_state_dict(state_dict, strict=False)
                logger.info("Loaded weights from %s", weights_path)
            except ImportError:
                logger.warning("safetensors not installed, trying torch.load")
        else:
            # Try .pt format
            pt_path = checkpoint_path_obj / "model.pt"
            if pt_path.exists():
                # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
                state_dict = torch.load(pt_path, map_location="cpu", weights_only=True)
                model.load_state_dict(state_dict, strict=False)
                logger.info("Loaded weights from %s", pt_path)

        return model.to(device)

    def gradient_checkpointing_enable(self) -> None:
        """Enable gradient checkpointing for memory optimization."""
        self._gradient_checkpointing_enabled = True
        logger.info("Enabled gradient checkpointing")

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing."""
        self._gradient_checkpointing_enabled = False
        logger.info("Disabled gradient checkpointing")

    def set_trainable_parameters(
        self,
        *,
        tune_paligemma: bool = False,
        tune_action_expert: bool = True,
        tune_vision_encoder: bool = False,
        tune_projection_heads: bool = True,
    ) -> None:
        """Configure which parameters are trainable for memory-efficient training.

        This enables training on smaller GPUs by freezing large models and only
        training lightweight heads.

        Args:
            tune_paligemma: Train PaliGemma language model backbone.
            tune_action_expert: Train Gemma action expert.
            tune_vision_encoder: Train SigLIP vision encoder.
            tune_projection_heads: Train action/state projection heads (recommended).

        Example:
            For smallest memory footprint (8-10GB VRAM):

            >>> model.set_trainable_parameters(
            ...     tune_paligemma=False,
            ...     tune_action_expert=False,
            ...     tune_vision_encoder=False,
            ...     tune_projection_heads=True,  # Only train heads
            ... )

            For moderate memory (12-16GB VRAM):

            >>> model.set_trainable_parameters(
            ...     tune_paligemma=False,
            ...     tune_action_expert=True,  # Train action expert
            ...     tune_vision_encoder=False,
            ...     tune_projection_heads=True,
            ... )
        """
        # Set backbone trainability
        self.paligemma_with_expert.set_trainable_parameters(
            tune_paligemma=tune_paligemma,
            tune_action_expert=tune_action_expert,
            tune_vision_encoder=tune_vision_encoder,
        )

        # Set projection head trainability
        # These are lightweight and should almost always be trained
        for param in self.action_in_proj.parameters():
            param.requires_grad = tune_projection_heads
        for param in self.action_out_proj.parameters():
            param.requires_grad = tune_projection_heads

        if self.is_pi05:
            # Pi0.5: time MLPs
            for param in self.time_mlp_in.parameters():
                param.requires_grad = tune_projection_heads
            for param in self.time_mlp_out.parameters():
                param.requires_grad = tune_projection_heads
        else:
            # Pi0: state projection + action-time MLPs
            for param in self.state_proj.parameters():
                param.requires_grad = tune_projection_heads
            for param in self.action_time_mlp_in.parameters():
                param.requires_grad = tune_projection_heads
            for param in self.action_time_mlp_out.parameters():
                param.requires_grad = tune_projection_heads

        # Log summary
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(
            "Trainable: %d / %d params (%.2f%%) | VLM=%s Expert=%s Vision=%s Heads=%s",
            trainable,
            total,
            100 * trainable / total,
            "🔥" if tune_paligemma else "❄️",
            "🔥" if tune_action_expert else "❄️",
            "🔥" if tune_vision_encoder else "❄️",
            "🔥" if tune_projection_heads else "❄️",
        )

    def get_trainable_param_count(self) -> tuple[int, int]:
        """Get count of trainable vs total parameters.

        Returns:
            Tuple of (trainable_params, total_params).
        """
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total

    def _apply_checkpoint(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """Apply gradient checkpointing if enabled.

        Args:
            func: Function to checkpoint.
            *args: Positional arguments for func.
            **kwargs: Keyword arguments for func.

        Returns:
            Result of calling func with the given arguments.
        """
        if self._gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(func, *args, use_reentrant=False, **kwargs)
        return func(*args, **kwargs)

    @staticmethod
    def _sample_noise(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        """Sample Gaussian noise for flow matching.

        Args:
            shape: Shape of noise tensor.
            device: Device for tensor.

        Returns:
            Noise tensor.
        """
        return torch.randn(shape, dtype=torch.float32, device=device)

    def _sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample timesteps from Beta distribution.

        Args:
            batch_size: Number of samples.
            device: Device for tensor.

        Returns:
            Timesteps in [0.001, 1.0].
        """
        time_beta = sample_beta(
            self.time_beta_alpha,
            self.time_beta_beta,
            batch_size,
            device,
        )
        time = time_beta * self.time_scale + self.time_offset
        return time.to(dtype=torch.float32)

    def embed_prefix(
        self,
        images: list[torch.Tensor],
        image_masks: list[torch.Tensor],
        language_tokens: torch.Tensor,
        language_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images and language tokens for prefix.

        Args:
            images: List of image tensors, each (batch, channels, height, width).
            image_masks: List of image validity masks, each (batch,).
            language_tokens: Token IDs of shape (batch, seq_len).
            language_masks: Token validity masks of shape (batch, seq_len).

        Returns:
            Tuple of:
                - embeddings: Combined embeddings (batch, total_seq, hidden)
                - pad_masks: Padding masks (batch, total_seq)
                - att_masks: Attention pattern masks (batch, total_seq)
        """
        embeddings = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, image_masks, strict=True):

            def _embed_image(img: torch.Tensor) -> torch.Tensor:
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(_embed_image, img)
            batch_size, num_patches = img_emb.shape[:2]

            embeddings.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(batch_size, num_patches))
            # Image tokens attend bidirectionally to each other
            att_masks.extend([0] * num_patches)

        # Process language tokens
        def _embed_language(tokens: torch.Tensor) -> torch.Tensor:
            emb = self.paligemma_with_expert.embed_language_tokens(tokens)
            # Scale by sqrt(dim) as in standard transformer
            return emb * math.sqrt(emb.shape[-1])

        lang_emb = self._apply_checkpoint(_embed_language, language_tokens)
        embeddings.append(lang_emb)
        pad_masks.append(language_masks)
        # Language tokens attend bidirectionally to images and each other
        att_masks.extend([0] * lang_emb.shape[1])

        # Concatenate
        embeddings = torch.cat(embeddings, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)

        batch_size = pad_masks.shape[0]
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(batch_size, -1)

        return embeddings, pad_masks, att_masks

    def embed_suffix(
        self,
        state: torch.Tensor,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Embed state, noisy actions, and timestep for suffix.

        For Pi0: state is embedded as continuous token, timestep fused with actions via MLP.
        For Pi0.5: state is in language tokens, timestep conditions via AdaRMSNorm.

        Args:
            state: State tensor of shape (batch, state_dim).
            noisy_actions: Noisy action trajectory (batch, horizon, action_dim).
            timestep: Flow matching timestep (batch,).

        Returns:
            Tuple of:
                - embeddings: Suffix embeddings (batch, suffix_seq, hidden)
                - pad_masks: Padding masks (batch, suffix_seq)
                - att_masks: Attention pattern masks (batch, suffix_seq)
                - adarms_cond: AdaRMSNorm conditioning (Pi0.5 only)
        """
        embeddings = []
        pad_masks = []
        att_masks = []
        adarms_cond = None

        device = noisy_actions.device
        batch_size = noisy_actions.shape[0]

        if not self.is_pi05:
            # Pi0: Embed state as single token
            state_emb = self.state_proj(state.float())
            embeddings.append(state_emb[:, None, :])
            pad_masks.append(torch.ones(batch_size, 1, dtype=torch.bool, device=device))
            # State breaks bidirectional attention - starts causal section
            att_masks.append(1)

        # Compute timestep embedding
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=self.time_min_period,
            max_period=self.time_max_period,
        )
        time_emb = time_emb.to(dtype=noisy_actions.dtype)

        # Project actions
        action_emb = self.action_in_proj(noisy_actions)

        if self.is_pi05:
            # Pi0.5: Time conditions via AdaRMSNorm
            time_emb = self.time_mlp_in(time_emb)
            time_emb = F.silu(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = F.silu(time_emb)
            adarms_cond = time_emb
            action_time_emb = action_emb
        else:
            # Pi0: Fuse timestep + action via MLP
            time_emb_expanded = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb_expanded], dim=-1)
            action_time_emb = self.action_time_mlp_in(action_time_emb)
            action_time_emb = F.silu(action_time_emb)
            action_time_emb = self.action_time_mlp_out(action_time_emb)

        embeddings.append(action_time_emb)
        action_seq_len = action_time_emb.shape[1]
        pad_masks.append(torch.ones(batch_size, action_seq_len, dtype=torch.bool, device=device))
        # First action token breaks attention, rest attend to previous
        att_masks.extend([1] + [0] * (self.action_horizon - 1))

        # Concatenate
        embeddings = torch.cat(embeddings, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embeddings.dtype, device=device)
        att_masks = att_masks[None, :].expand(batch_size, -1)

        return embeddings, pad_masks, att_masks, adarms_cond

    def forward(self, batch: Mapping[str, Any], *, use_bf16: bool = True) -> tuple[torch.Tensor, dict[str, float]]:
        """Training forward pass.

        Args:
            batch: Input batch with observations and actions.
            use_bf16: Whether to use bfloat16 autocasting.

        Returns:
            Tuple of (loss, loss_dict).
        """
        device = next(self.parameters()).device

        observation = {
            "images": batch["images"],
            "image_masks": batch["image_masks"],
            "state": batch["state"],
            "tokenized_prompt": batch["tokenized_prompt"].to(device),
            "tokenized_prompt_mask": batch["tokenized_prompt_mask"].to(device),
        }
        actions = batch["actions"]

        # Forward pass
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            loss_per_sample = self._forward_train(observation, actions)

        # Average loss
        loss = loss_per_sample.mean()
        loss_dict = {"loss": loss.item()}

        return loss, loss_dict

    def predict_action_chunk(self, batch: Mapping[str, Any], *, use_bf16: bool = True) -> torch.Tensor:
        """Predict a chunk of actions from input batch.

        This method processes the input batch, prepares images, state, and language tokens,
        then uses the model to sample actions.

        Args:
            batch: A dictionary containing input tensors including images, state information,
                and tokenized prompts with their masks.
            use_bf16: Whether to use bfloat16 autocasting.

        Returns:
            torch.Tensor: A tensor of predicted actions with shape matching the original
                action dimensions from the dataset statistics.
        """
        device = next(self.parameters()).device
        self.eval()

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            return self.sample_actions(device, batch)

    @property
    def extra_export_args(self) -> dict:
        """Additional export arguments for model conversion.

        This property provides extra configuration parameters needed when exporting
        the model to different formats, particularly ONNX format.

        Returns:
            dict: A dictionary containing format-specific export arguments.

        Example:
            >>> extra_args = model.extra_export_args()
            >>> print(extra_args)
            {'onnx': {'output_names': ['action']}}
        """
        extra_args = {}
        extra_args["onnx"] = {
            "output_names": ["action"],
        }
        extra_args["openvino"] = {
            "output": ["action"],
        }
        extra_args["torch_export_ir"] = {}
        extra_args["torch"] = {
            "input_names": ["Observation"],
            "output_names": ["action"],
        }

        return extra_args

    @property
    def reward_delta_indices(self) -> None:
        """Return reward indices.

        Currently returns `None` as rewards are not implemented.

        Returns:
            None
        """
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        """Get indices of actions relative to the current timestep.

        Returns:
            list[int]: A list of relative action indices.
        """
        return list(range(self.action_horizon))

    @property
    def observation_delta_indices(self) -> None:
        """Get indices of observations relative to the current timestep.

        Returns:
            list[int]: A list of relative observation indices.
        """
        return None

    def _forward_train(  # noqa: PLR0914
        self,
        observation: Mapping[str, Any],
        actions: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Training forward pass - compute flow matching loss.

        Args:
            observation: Observation dict containing:
                - images: dict of image tensors
                - image_masks: dict of image validity masks
                - state: state tensor
                - tokenized_prompt: language token IDs
                - tokenized_prompt_mask: language token masks
            actions: Ground truth actions (batch, horizon, action_dim).
            noise: Optional pre-sampled noise for deterministic training.
            time: Optional pre-sampled timesteps for deterministic training.

        Returns:
            Per-sample MSE loss of shape (batch, horizon, action_dim).
        """
        # Preprocess observation
        images, image_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation)

        device = actions.device
        batch_size = actions.shape[0]

        # Sample noise and time if not provided
        if noise is None:
            noise = self._sample_noise(actions.shape, device)
        if time is None:
            time = self._sample_time(batch_size, device)

        # Create noisy actions via flow matching interpolation: x_t = t*noise + (1-t)*actions
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions

        # Target: noise - actions (the flow direction)
        u_t = noise - actions

        # Embed prefix (images + language)
        prefix_emb, prefix_pad, prefix_att = self.embed_prefix(images, image_masks, lang_tokens, lang_masks)

        # Embed suffix (state + noisy actions + time)
        suffix_emb, suffix_pad, suffix_att, adarms_cond = self.embed_suffix(state, x_t, time)

        # Prepare combined attention mask
        pad_masks = torch.cat([prefix_pad, suffix_pad], dim=1)
        att_masks = torch.cat([prefix_att, suffix_att], dim=1)
        att_2d_mask = make_attention_mask_2d(pad_masks, att_masks)
        att_4d_mask = prepare_4d_attention_mask(att_2d_mask, dtype=prefix_emb.dtype)

        # Compute position IDs
        position_ids = torch.cumsum(pad_masks.long(), dim=1) - 1

        # Forward through backbone
        (_prefix_out, suffix_out), _ = self.paligemma_with_expert(
            inputs_embeds=[prefix_emb, suffix_emb],
            attention_mask=att_4d_mask,
            position_ids=position_ids,
            adarms_cond=[None, adarms_cond],
            use_cache=False,
        )

        # Extract action predictions from suffix output
        suffix_out = suffix_out[:, -self.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        # Project to action space
        v_t = self.action_out_proj(suffix_out)

        # MSE loss
        return F.mse_loss(u_t, v_t, reduction="none")

    @torch.no_grad()
    def sample_actions(  # noqa: PLR0914
        self,
        device: str | torch.device,
        observation: Mapping[str, Any],
        *,
        noise: torch.Tensor | None = None,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        """Inference - sample actions via Euler integration.

        Args:
            device: Device for computation.
            observation: Observation dict (same format as forward).
            noise: Optional starting noise. If None, sampled randomly.
            num_steps: Number of Euler steps. If None, uses default.

        Returns:
            Sampled actions of shape (batch, horizon, action_dim).
        """
        if num_steps is None:
            num_steps = self.num_inference_steps

        # Preprocess observation
        images, image_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation)

        batch_size = state.shape[0]
        action_shape = (batch_size, self.action_horizon, self.max_action_dim)

        # Initialize from noise
        if noise is None:
            noise = self._sample_noise(action_shape, device)
        x_t = noise

        # Embed prefix (cached for all steps)
        prefix_emb, prefix_pad, prefix_att = self.embed_prefix(images, image_masks, lang_tokens, lang_masks)

        # Compute prefix KV cache
        prefix_att_2d = make_attention_mask_2d(prefix_pad, prefix_att)
        prefix_att_4d = prepare_4d_attention_mask(prefix_att_2d, dtype=prefix_emb.dtype)
        prefix_position_ids = torch.cumsum(prefix_pad.long(), dim=1) - 1

        # Get cached prefix
        (_, _), past_key_values = self.paligemma_with_expert(
            inputs_embeds=[prefix_emb, None],
            attention_mask=prefix_att_4d,
            position_ids=prefix_position_ids,
            use_cache=True,
        )

        # Euler integration: t goes from 1 (noise) to 0 (target)
        # Using for loop instead of while for ONNX/OpenVINO export compatibility
        dt = -1.0 / num_steps

        for step in range(num_steps):
            # Compute time for this step: starts at 1.0, ends near 0.0
            time = 1.0 + step * dt
            timestep = torch.full((batch_size,), time, dtype=torch.float32, device=device)

            # Embed suffix for current x_t
            suffix_emb, suffix_pad, suffix_att, adarms_cond = self.embed_suffix(state, x_t, timestep)

            # Create attention mask for suffix attending to prefix + suffix
            suffix_len = suffix_pad.shape[1]
            prefix_len = prefix_pad.shape[1]

            # Suffix can attend to all prefix tokens
            prefix_2d = prefix_pad[:, None, :].expand(batch_size, suffix_len, prefix_len)
            suffix_2d = make_attention_mask_2d(suffix_pad, suffix_att)
            full_2d = torch.cat([prefix_2d, suffix_2d], dim=-1)
            full_4d = prepare_4d_attention_mask(full_2d, dtype=suffix_emb.dtype)

            # Suffix position IDs continue from prefix
            prefix_offsets = prefix_pad.sum(dim=-1)[:, None]
            suffix_position_ids = prefix_offsets + torch.cumsum(suffix_pad.long(), dim=1) - 1

            # Forward (suffix only, using cached prefix KV)
            (_, suffix_out), _ = self.paligemma_with_expert(
                inputs_embeds=[None, suffix_emb],
                attention_mask=full_4d,
                position_ids=suffix_position_ids,
                past_key_values=past_key_values,
                adarms_cond=[None, adarms_cond],
                use_cache=False,
            )

            # Extract velocity prediction
            suffix_out = suffix_out[:, -self.action_horizon :]
            suffix_out = suffix_out.to(dtype=torch.float32)
            v_t = self.action_out_proj(suffix_out)

            # Euler step: x_t = x_t + dt * v_t
            # Note: Using explicit assignment (not +=) for ONNX export compatibility
            x_t = x_t + dt * v_t  # noqa: PLR6104

        return x_t

    @staticmethod
    def _preprocess_observation(
        observation: Mapping[str, Any],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract and preprocess observation components.

        Args:
            observation: Raw observation dict.

        Returns:
            Tuple of (images, image_masks, lang_tokens, lang_masks, state).
        """
        # Extract images and masks
        images = list(observation.get("images", {}).values())
        image_masks = list(observation.get("image_masks", {}).values())

        # Ensure masks are tensors
        image_masks = [m if isinstance(m, torch.Tensor) else torch.tensor(m, dtype=torch.bool) for m in image_masks]

        # Extract language
        lang_tokens = observation.get("tokenized_prompt")
        lang_masks = observation.get("tokenized_prompt_mask")

        # Extract state
        state = observation.get("state")

        return images, image_masks, lang_tokens, lang_masks, state
