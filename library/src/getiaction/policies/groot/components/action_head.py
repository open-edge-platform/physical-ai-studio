# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Based on NVIDIA's GR00T implementation (Apache-2.0 licensed)
# Original source: https://github.com/NVIDIA/Isaac-GR00T

"""Flow Matching Action Head for diffusion-based action generation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from torch.distributions import Beta

from getiaction.config.mixin import FromConfig

from .nn import CategorySpecificMLP, MultiEmbodimentActionEncoder
from .transformer import get_dit_class, get_self_attention_transformer_class

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)


@dataclass
class FlowMatchingActionHeadConfig:
    """Configuration for FlowMatchingActionHead (for FromConfig mixin).

    This dataclass mirrors the explicit constructor arguments for config-based
    instantiation via the FromConfig mixin.

    Attributes:
        action_dim: Action vector dimension.
        action_horizon: Number of action steps to predict.
        add_pos_embed: Whether to add positional embedding.
        input_embedding_dim: Input embedding dimension.
        backbone_embedding_dim: Backbone (Eagle) embedding dimension.
        hidden_size: Hidden dimension for encoders/decoders.
        max_seq_len: Maximum sequence length.
        noise_beta_alpha: Beta distribution alpha parameter.
        noise_beta_beta: Beta distribution beta parameter.
        noise_s: Flow matching noise scale.
        num_timestep_buckets: Number of timestep discretization buckets.
        num_inference_timesteps: Number of denoising steps at inference.
        max_num_embodiments: Maximum number of embodiment categories.
        max_state_dim: Maximum state dimension.
        tune_projector: Whether to tune encoder/decoder projectors.
        tune_diffusion_model: Whether to tune the DiT model.
        use_vlln: Whether to use VL layer norm and self-attention.
        num_target_vision_tokens: Number of future vision tokens.
        diffusion_model_cfg: DiT configuration dictionary.
        vl_self_attention_cfg: Config for VL self-attention transformer.
    """

    # Required
    action_dim: int
    action_horizon: int
    # Core architecture
    add_pos_embed: bool = True
    input_embedding_dim: int = 1536
    backbone_embedding_dim: int = 1536
    hidden_size: int = 1024
    max_seq_len: int = 1024
    # Flow matching
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    num_timestep_buckets: int = 1000
    num_inference_timesteps: int = 10
    # Multi-embodiment
    max_num_embodiments: int = 32
    max_state_dim: int = 64
    # Training
    tune_projector: bool = True
    tune_diffusion_model: bool = True
    # VL processing
    use_vlln: bool = True
    num_target_vision_tokens: int = 32
    # Nested configs (dicts for transformer modules)
    diffusion_model_cfg: dict[str, Any] | None = None
    vl_self_attention_cfg: dict[str, Any] | None = None


class FlowMatchingActionHead(nn.Module, FromConfig):
    """Flow Matching Action Head for diffusion-based action generation.

    Uses a Diffusion Transformer (DiT) conditioned on vision-language features
    to generate action trajectories via flow matching.

    All constructor arguments are explicit for clarity and testability.
    Supports config-based instantiation via the FromConfig mixin.

    Args:
        action_dim: Action vector dimension.
        action_horizon: Number of action steps to predict.
        add_pos_embed: Whether to add positional embedding.
        input_embedding_dim: Input embedding dimension.
        backbone_embedding_dim: Backbone (Eagle) embedding dimension.
        hidden_size: Hidden dimension for encoders/decoders.
        max_seq_len: Maximum sequence length.
        noise_beta_alpha: Beta distribution alpha parameter.
        noise_beta_beta: Beta distribution beta parameter.
        noise_s: Flow matching noise scale.
        num_timestep_buckets: Number of timestep discretization buckets.
        num_inference_timesteps: Number of denoising steps at inference.
        max_num_embodiments: Maximum number of embodiment categories.
        max_state_dim: Maximum state dimension.
        tune_projector: Whether to tune encoder/decoder projectors.
        tune_diffusion_model: Whether to tune the DiT model.
        use_vlln: Whether to use VL layer norm and self-attention.
        num_target_vision_tokens: Number of future vision tokens.
        diffusion_model_cfg: DiT configuration dictionary.
        vl_self_attention_cfg: Config for VL self-attention transformer.

    Examples:
        Direct instantiation with explicit args:

        >>> action_head = FlowMatchingActionHead(
        ...     action_dim=32,
        ...     action_horizon=50,
        ...     num_inference_timesteps=10,
        ... )

        From config dataclass:

        >>> config = FlowMatchingActionHeadConfig(action_dim=32, action_horizon=50)
        >>> action_head = FlowMatchingActionHead.from_config(config)
    """

    def __init__(  # noqa: PLR0913
        self,
        action_dim: int,
        action_horizon: int,
        *,
        # Core architecture
        add_pos_embed: bool = True,
        input_embedding_dim: int = 1536,
        backbone_embedding_dim: int = 1536,
        hidden_size: int = 1024,
        max_seq_len: int = 1024,
        # Flow matching
        noise_beta_alpha: float = 1.5,
        noise_beta_beta: float = 1.0,
        noise_s: float = 0.999,
        num_timestep_buckets: int = 1000,
        num_inference_timesteps: int = 10,
        # Multi-embodiment
        max_num_embodiments: int = 32,
        max_state_dim: int = 64,
        # Training
        tune_projector: bool = True,
        tune_diffusion_model: bool = True,
        # VL processing
        use_vlln: bool = True,
        num_target_vision_tokens: int = 32,
        # Nested configs (dicts for transformer modules)
        diffusion_model_cfg: dict[str, Any] | None = None,
        vl_self_attention_cfg: dict[str, Any] | None = None,
    ) -> None:
        """Initialize FlowMatchingActionHead with explicit arguments."""
        super().__init__()

        # Store core dimensions
        self.hidden_size = hidden_size
        self.input_embedding_dim = input_embedding_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.num_inference_timesteps = num_inference_timesteps
        self.noise_s = noise_s

        # Get DiT class (lazy loaded)
        dit_cls = get_dit_class()

        # Initialize DiT model
        diffusion_cfg = diffusion_model_cfg or {}
        self.model = dit_cls(**diffusion_cfg)

        # State encoder: state -> embedding
        self.state_encoder = CategorySpecificMLP(
            num_categories=max_num_embodiments,
            input_dim=max_state_dim,
            hidden_dim=hidden_size,
            output_dim=input_embedding_dim,
        )

        # Action encoder: noisy action + timestep -> embedding
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=action_dim,
            hidden_size=input_embedding_dim,
            num_embodiments=max_num_embodiments,
        )

        # Action decoder: embedding -> action
        self.action_decoder = CategorySpecificMLP(
            num_categories=max_num_embodiments,
            input_dim=hidden_size,
            hidden_dim=hidden_size,
            output_dim=action_dim,
        )

        # Future tokens for conditioning
        self.future_tokens = nn.Embedding(num_target_vision_tokens, input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        # VL processing
        if use_vlln:
            self.vlln = nn.LayerNorm(backbone_embedding_dim)
            self_attn_cls = get_self_attention_transformer_class()
            vl_cfg = vl_self_attention_cfg or {}
            self.vl_self_attention = self_attn_cls(**vl_cfg)
        else:
            self.vlln = nn.Identity()
            self.vl_self_attention = nn.Identity()

        # Position embedding
        if add_pos_embed:
            self.position_embedding = nn.Embedding(max_seq_len, input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        else:
            self.position_embedding = None

        # Flow matching noise distribution
        self.beta_dist = Beta(noise_beta_alpha, noise_beta_beta)
        self.num_timestep_buckets = num_timestep_buckets

        # Set trainable parameters
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        self._set_trainable_parameters()

    def _set_trainable_parameters(self) -> None:
        """Configure which parameters are trainable."""
        for p in self.parameters():
            p.requires_grad = True

        if not self.tune_projector:
            self.state_encoder.requires_grad_(requires_grad=False)
            self.action_encoder.requires_grad_(requires_grad=False)
            self.action_decoder.requires_grad_(requires_grad=False)
            if self.position_embedding is not None:
                self.position_embedding.requires_grad_(requires_grad=False)

        if not self.tune_diffusion_model:
            self.model.requires_grad_(requires_grad=False)

        logger.info("Tune action head projector: %s", self.tune_projector)
        logger.info("Tune action head diffusion model: %s", self.tune_diffusion_model)

    def set_trainable_parameters(self, *, tune_projector: bool, tune_diffusion_model: bool) -> None:
        """Public method to configure trainable parameters.

        Args:
            tune_projector: Whether to tune encoder/decoder projectors.
            tune_diffusion_model: Whether to tune the DiT model.
        """
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        self._set_trainable_parameters()

    def _set_frozen_modules_to_eval_mode(self) -> None:
        """Set frozen modules to eval mode during training."""
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.position_embedding is not None:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()

    def sample_time(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Sample timesteps from beta distribution.

        Args:
            batch_size: Batch size.
            device: Target device.
            dtype: Target dtype.

        Returns:
            Sampled timesteps.
        """
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.noise_s - sample) / self.noise_s

    def _process_backbone_output(
        self,
        backbone_output: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Process backbone features through VL layers.

        Args:
            backbone_output: Dict with backbone_features and attention_mask.

        Returns:
            Processed backbone output.
        """
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        backbone_features = self.vl_self_attention(backbone_features)
        return {
            "backbone_features": backbone_features,
            "backbone_attention_mask": backbone_output["backbone_attention_mask"],
        }

    def forward(  # noqa: PLR0914
        self,
        backbone_output: Mapping[str, torch.Tensor],
        action_input: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Training forward pass - compute flow matching loss.

        Args:
            backbone_output: Dict with backbone_features and attention_mask.
            action_input: Dict with state, action, action_mask, embodiment_id.

        Returns:
            Dict with 'loss' key.
        """
        self._set_frozen_modules_to_eval_mode()

        backbone_output = self._process_backbone_output(backbone_output)

        # Get vision and language embeddings
        vl_embs = backbone_output["backbone_features"]
        device = vl_embs.device

        # Get embodiment ID
        embodiment_id = action_input["embodiment_id"]

        # Embed state
        state_features = self.state_encoder(action_input["state"], embodiment_id)

        # Flow matching: add noise to actions
        actions = action_input["action"]
        noise = torch.randn(actions.shape, device=device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=device, dtype=actions.dtype)
        t = t[:, None, None]  # (B, 1, 1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Discretize timestep
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        # Add position embedding
        if self.position_embedding is not None:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features += pos_embs

        # Concatenate state, future tokens, and action features
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
        sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

        # Run through DiT
        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=backbone_output.get("backbone_attention_mask"),
            timestep=t_discretized,
            return_all_hidden_states=False,
        )

        # Decode actions
        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1] :]

        # Compute loss
        action_mask = action_input["action_mask"]
        loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        loss = loss.sum() / action_mask.sum()

        return {"loss": loss}

    @torch.no_grad()
    def get_action(  # noqa: PLR0914
        self,
        backbone_output: Mapping[str, torch.Tensor],
        action_input: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Inference - generate actions via iterative denoising.

        Args:
            backbone_output: Dict with backbone_features and attention_mask.
            action_input: Dict with state, embodiment_id.

        Returns:
            Dict with 'action_pred' key containing predicted actions.
        """
        backbone_output = self._process_backbone_output(backbone_output)

        # Get vision and language embeddings
        vl_embs = backbone_output["backbone_features"]
        device = vl_embs.device
        embodiment_id = action_input["embodiment_id"]

        # Embed state
        state_features = self.state_encoder(action_input["state"], embodiment_id)

        # Start from noise
        batch_size = vl_embs.shape[0]
        actions = torch.randn(
            size=(batch_size, self.action_horizon, self.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        # Iterative denoising
        for t in range(num_steps):
            t_cont = t / float(num_steps)
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Encode current noisy actions
            timesteps_tensor = torch.full(
                size=(batch_size,),
                fill_value=t_discretized,
                device=device,
            )
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)

            # Add position embedding
            if self.position_embedding is not None:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features += pos_embs

            # Concatenate features
            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)
            sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

            # Run DiT
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps_tensor,
            )

            # Decode velocity
            pred = self.action_decoder(model_output, embodiment_id)
            pred_velocity = pred[:, -self.action_horizon :]

            # Euler integration
            actions += dt * pred_velocity

        return {"action_pred": actions}

    @property
    def device(self) -> torch.device:
        """Get device of model parameters."""
        return next(iter(self.parameters())).device

    @property
    def dtype(self) -> torch.dtype:
        """Get dtype of model parameters."""
        return next(iter(self.parameters())).dtype
