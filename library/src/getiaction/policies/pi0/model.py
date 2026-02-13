# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2025 Physical Intelligence
# SPDX-License-Identifier: Apache-2.0

"""Pi0/Pi0.5 Model - Core PyTorch implementation."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, Literal, cast

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from getiaction.data.observation import ACTION, Observation

from .components.attention import make_attention_mask_2d, prepare_4d_attention_mask
from .components.gemma import GemmaVariant, PaliGemmaWithExpert

__all__ = ["GemmaVariant", "Pi0Model", "create_sinusoidal_pos_embedding", "sample_beta"]

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from .preprocessor import Pi0Postprocessor, Pi0Preprocessor

logger = logging.getLogger(__name__)


def create_sinusoidal_pos_embedding(
    time: torch.Tensor,
    dimension: int,
    min_period: float,
    max_period: float,
) -> torch.Tensor:
    """Create sinusoidal positional embeddings for timestep encoding."""  # noqa: DOC201, DOC501
    if dimension % 2 != 0:
        msg = f"dimension ({dimension}) must be divisible by 2"
        raise ValueError(msg)

    if time.ndim != 1:
        msg = "The time tensor is expected to be of shape (batch_size,)"
        raise ValueError(msg)

    device = time.device
    dtype = torch.float64

    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]

    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha: float, beta: float, size: int, device: torch.device) -> torch.Tensor:
    """Sample from a Beta distribution for timestep sampling."""  # noqa: DOC201
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((size,))


class Pi0Model(nn.Module):
    """Pi0/Pi0.5 Flow Matching Vision-Language-Action Model."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        variant: Literal["pi0", "pi05"] = "pi0",
        paligemma_variant: GemmaVariant = "gemma_2b",
        action_expert_variant: GemmaVariant = "gemma_300m",
        max_action_dim: int = 32,
        max_state_dim: int = 32,
        chunk_size: int = 50,
        num_inference_steps: int = 10,
        dtype: str = "float32",
        time_beta_alpha: float = 1.5,
        time_beta_beta: float = 1.0,
        time_scale: float = 0.999,
        time_offset: float = 0.001,
        time_min_period: float = 4e-3,
        time_max_period: float = 4.0,
        preprocessor: Pi0Preprocessor | None = None,
        postprocessor: Pi0Postprocessor | None = None,
    ) -> None:
        """Initialize Pi0Model."""
        super().__init__()

        self.variant = variant
        self.paligemma_variant = paligemma_variant
        self.action_expert_variant = action_expert_variant
        self.max_action_dim = max_action_dim
        self.max_state_dim = max_state_dim
        self.chunk_size = chunk_size
        self.num_inference_steps = num_inference_steps
        self.dtype = dtype
        self.time_beta_alpha = time_beta_alpha
        self.time_beta_beta = time_beta_beta
        self.time_scale = time_scale
        self.time_offset = time_offset
        self.time_min_period = time_min_period
        self.time_max_period = time_max_period

        self.is_pi05 = variant == "pi05"
        self.use_adarms = self.is_pi05

        self.paligemma_with_expert = PaliGemmaWithExpert(
            paligemma_variant=paligemma_variant,
            action_expert_variant=action_expert_variant,
            use_adarms=self.use_adarms,
            dtype=dtype,
        )

        action_expert_width = self.paligemma_with_expert.action_expert_hidden_size
        self.action_in_proj = nn.Linear(max_action_dim, action_expert_width)
        self.action_out_proj = nn.Linear(action_expert_width, max_action_dim)

        if self.is_pi05:
            self.time_mlp_in = nn.Linear(action_expert_width, action_expert_width)
            self.time_mlp_out = nn.Linear(action_expert_width, action_expert_width)
        else:
            self.state_proj = nn.Linear(max_state_dim, action_expert_width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_width, action_expert_width)
            self.action_time_mlp_out = nn.Linear(action_expert_width, action_expert_width)

        self._gradient_checkpointing_enabled = False

        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

    def gradient_checkpointing_enable(self) -> None:
        """Enable gradient checkpointing for memory efficiency."""
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
        """Configure which model parameters are trainable."""
        self.paligemma_with_expert.set_trainable_parameters(
            tune_paligemma=tune_paligemma,
            tune_action_expert=tune_action_expert,
            tune_vision_encoder=tune_vision_encoder,
        )

        for param in self.action_in_proj.parameters():
            param.requires_grad = tune_projection_heads
        for param in self.action_out_proj.parameters():
            param.requires_grad = tune_projection_heads

        if self.is_pi05:
            for param in self.time_mlp_in.parameters():
                param.requires_grad = tune_projection_heads
            for param in self.time_mlp_out.parameters():
                param.requires_grad = tune_projection_heads
        else:
            for param in self.state_proj.parameters():
                param.requires_grad = tune_projection_heads
            for param in self.action_time_mlp_in.parameters():
                param.requires_grad = tune_projection_heads
            for param in self.action_time_mlp_out.parameters():
                param.requires_grad = tune_projection_heads

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(
            "Trainable: %d / %d params (%.2f%%)",
            trainable,
            total,
            100 * trainable / total,
        )

    def _apply_checkpoint(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        if self._gradient_checkpointing_enabled and self.training:
            from torch.utils import checkpoint as torch_checkpoint  # noqa: PLC0415

            return torch_checkpoint.checkpoint(func, *args, use_reentrant=False, **kwargs)
        return func(*args, **kwargs)

    @staticmethod
    def _sample_noise(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        return torch.randn(shape, dtype=torch.float32, device=device)

    def _sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
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
        """Embed image and language inputs into prefix embeddings."""  # noqa: DOC201
        embeddings = []
        pad_masks = []
        att_masks = []

        for img, img_mask in zip(images, image_masks, strict=True):

            def _embed_image(img: torch.Tensor) -> torch.Tensor:
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(_embed_image, img)
            batch_size, num_patches = img_emb.shape[:2]

            embeddings.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(batch_size, num_patches))
            att_masks.extend([0] * num_patches)

        def _embed_language(tokens: torch.Tensor) -> torch.Tensor:
            emb = self.paligemma_with_expert.embed_language_tokens(tokens)
            return emb * math.sqrt(emb.shape[-1])

        lang_emb = self._apply_checkpoint(_embed_language, language_tokens)
        embeddings.append(lang_emb)
        pad_masks.append(language_masks)
        att_masks.extend([0] * lang_emb.shape[1])

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
        """Embed state, actions, and timestep into suffix embeddings."""  # noqa: DOC201
        embeddings = []
        pad_masks = []
        att_masks = []
        adarms_cond = None

        device = noisy_actions.device
        batch_size = noisy_actions.shape[0]

        if not self.is_pi05:
            state_emb = self.state_proj(state.to(dtype=self.state_proj.weight.dtype))
            embeddings.append(state_emb[:, None, :])
            pad_masks.append(torch.ones(batch_size, 1, dtype=torch.bool, device=device))
            att_masks.append(1)

        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=self.time_min_period,
            max_period=self.time_max_period,
        )
        time_emb = time_emb.to(dtype=noisy_actions.dtype)

        action_emb = self.action_in_proj(noisy_actions.to(dtype=self.action_in_proj.weight.dtype))

        if self.is_pi05:
            time_emb = self.time_mlp_in(time_emb)
            time_emb = F.silu(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = F.silu(time_emb)
            adarms_cond = time_emb
            action_time_emb = action_emb
        else:
            time_emb_expanded = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb_expanded], dim=-1)
            action_time_emb = self.action_time_mlp_in(action_time_emb)
            action_time_emb = F.silu(action_time_emb)
            action_time_emb = self.action_time_mlp_out(action_time_emb)

        embeddings.append(action_time_emb)
        action_seq_len = action_time_emb.shape[1]
        pad_masks.append(torch.ones(batch_size, action_seq_len, dtype=torch.bool, device=device))
        att_masks.extend([1] + [0] * (self.chunk_size - 1))

        embeddings = torch.cat(embeddings, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embeddings.dtype, device=device)
        att_masks = att_masks[None, :].expand(batch_size, -1)

        return embeddings, pad_masks, att_masks, adarms_cond

    def forward(
        self,
        batch: Mapping[str, Any] | Observation,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        """Forward pass: compute loss during training or predict actions during inference."""  # noqa: DOC201
        if self.training:
            processed = self._preprocess_batch(batch, require_actions=True)
            return self._forward_loss(processed)
        return self.predict_action_chunk(batch)

    def predict_action_chunk(self, batch: Mapping[str, Any] | Observation) -> torch.Tensor:
        """Generate predicted actions for a batch of observations.

        This method preprocesses the input batch, samples actions from the model,
        and applies postprocessing to return the final action predictions.

        Args:
            batch: A mapping of input features or an Observation object containing
                the data needed to generate action predictions.

        Returns:
            torch.Tensor: The postprocessed predicted actions.

        Raises:
            ValueError: If the postprocessor is not initialized.
        """
        device = next(self.parameters()).device
        self.eval()

        processed = self._preprocess_batch(batch, require_actions=False)
        actions = self.sample_actions(device, processed)

        if self.postprocessor is None:
            msg = "Pi0Model postprocessor is not initialized"
            raise ValueError(msg)

        return self.postprocessor({ACTION: actions})[ACTION]

    @property
    def extra_export_args(self) -> dict:
        """Return extra export arguments for different export formats."""
        return {
            "onnx": {"output_names": ["action"]},
            "openvino": {"output": ["action"]},
            "torch_export_ir": {},
            "torch": {"input_names": ["observation"], "output_names": ["action"]},
        }

    @property
    def reward_delta_indices(self) -> None:
        """Return reward delta indices (not used in Pi0)."""
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        """Return action delta indices for the full action chunk."""
        return list(range(self.chunk_size))

    @property
    def observation_delta_indices(self) -> None:
        """Return observation delta indices (not used in Pi0)."""
        return None

    def _compute_loss(  # noqa: PLR0914
        self,
        observation: Mapping[str, Any],
        actions: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        images, image_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation)

        device = actions.device
        batch_size = actions.shape[0]

        if noise is None:
            noise = self._sample_noise(actions.shape, device)
        if time is None:
            time = self._sample_time(batch_size, device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions

        u_t = noise - actions

        prefix_emb, prefix_pad, prefix_att = self.embed_prefix(images, image_masks, lang_tokens, lang_masks)
        suffix_emb, suffix_pad, suffix_att, adarms_cond = self.embed_suffix(state, x_t, time)

        pad_masks = torch.cat([prefix_pad, suffix_pad], dim=1)
        att_masks = torch.cat([prefix_att, suffix_att], dim=1)
        att_2d_mask = make_attention_mask_2d(pad_masks, att_masks)
        att_4d_mask = prepare_4d_attention_mask(att_2d_mask, dtype=prefix_emb.dtype)

        position_ids = torch.cumsum(pad_masks.long(), dim=1) - 1

        (_prefix_out, suffix_out), _ = self.paligemma_with_expert(
            inputs_embeds=[prefix_emb, suffix_emb],
            attention_mask=att_4d_mask,
            position_ids=position_ids,
            adarms_cond=[None, adarms_cond],
            use_cache=False,
        )

        suffix_out = suffix_out[:, -self.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        v_t = self.action_out_proj(suffix_out)

        return F.mse_loss(u_t, v_t, reduction="none")

    def _forward_loss(self, batch: Mapping[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        device = next(self.parameters()).device

        observation = {
            "images": batch["images"],
            "image_masks": batch["image_masks"],
            "state": batch["state"],
            "tokenized_prompt": batch["tokenized_prompt"].to(device),
            "tokenized_prompt_mask": batch["tokenized_prompt_mask"].to(device),
        }
        actions = batch["actions"]

        loss_per_sample = self._compute_loss(observation, actions)

        loss = loss_per_sample.mean()
        loss_dict = {
            "loss": loss.item(),
            "loss_per_dim": loss_per_sample.mean(dim=(0, 1)).detach().cpu().tolist(),
        }

        return loss, loss_dict

    @torch.no_grad()
    def sample_actions(  # noqa: PLR0914
        self,
        device: str | torch.device,
        observation: Mapping[str, Any],
        *,
        noise: torch.Tensor | None = None,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        """Sample actions using flow matching iterative denoising."""  # noqa: DOC201
        if num_steps is None:
            num_steps = self.num_inference_steps

        images, image_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation)

        batch_size = state.shape[0]
        action_shape = (batch_size, self.chunk_size, self.max_action_dim)

        if noise is None:
            resolved_device = torch.device(device) if isinstance(device, str) else device
            noise = self._sample_noise(action_shape, resolved_device)
        else:
            resolved_device = noise.device
        x_t = noise

        prefix_emb, prefix_pad, prefix_att = self.embed_prefix(images, image_masks, lang_tokens, lang_masks)

        prefix_att_2d = make_attention_mask_2d(prefix_pad, prefix_att)
        prefix_att_4d = prepare_4d_attention_mask(prefix_att_2d, dtype=prefix_emb.dtype)
        prefix_position_ids = torch.cumsum(prefix_pad.long(), dim=1) - 1

        (_, _), past_key_values = self.paligemma_with_expert(
            inputs_embeds=[prefix_emb, None],
            attention_mask=prefix_att_4d,
            position_ids=prefix_position_ids,
            use_cache=True,
        )

        dt = -1.0 / num_steps

        for step in range(num_steps):
            time = 1.0 + step * dt
            timestep = torch.full((batch_size,), time, dtype=torch.float32, device=resolved_device)

            suffix_emb, suffix_pad, suffix_att, adarms_cond = self.embed_suffix(state, x_t, timestep)

            suffix_len = suffix_pad.shape[1]
            prefix_len = prefix_pad.shape[1]

            prefix_2d = prefix_pad[:, None, :].expand(batch_size, suffix_len, prefix_len)
            suffix_2d = make_attention_mask_2d(suffix_pad, suffix_att)
            full_2d = torch.cat([prefix_2d, suffix_2d], dim=-1)
            full_4d = prepare_4d_attention_mask(full_2d, dtype=suffix_emb.dtype)

            prefix_offsets = prefix_pad.sum(dim=-1)[:, None]
            suffix_position_ids = prefix_offsets + torch.cumsum(suffix_pad.long(), dim=1) - 1

            (_, suffix_out), _ = self.paligemma_with_expert(
                inputs_embeds=[None, suffix_emb],
                attention_mask=full_4d,
                position_ids=suffix_position_ids,
                past_key_values=past_key_values,
                adarms_cond=[None, adarms_cond],
                use_cache=False,
            )

            suffix_out = suffix_out[:, -self.chunk_size :]
            suffix_out = suffix_out.to(dtype=torch.float32)
            v_t = self.action_out_proj(suffix_out)

            x_t = x_t + dt * v_t  # noqa: PLR6104

        return x_t

    def _preprocess_batch(
        self,
        batch: Mapping[str, Any] | Observation,
        *,
        require_actions: bool,
    ) -> Mapping[str, Any]:
        device = next(self.parameters()).device

        if self.preprocessor is None:
            msg = "Pi0Model preprocessor is not initialized"
            raise ValueError(msg)

        if isinstance(batch, Observation):
            batch = batch.to(device)

        processed = self.preprocessor(batch)
        processed = self._move_to_device(processed, device)

        if require_actions and "actions" not in processed:
            msg = "Processed batch is missing 'actions' for training"
            raise ValueError(msg)

        return processed

    @staticmethod
    def _move_to_device(batch: Mapping[str, Any], device: torch.device) -> Mapping[str, Any]:
        result: dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.to(device)
            elif isinstance(value, dict):
                result[key] = {
                    inner_key: inner_value.to(device) if isinstance(inner_value, torch.Tensor) else inner_value
                    for inner_key, inner_value in value.items()
                }
            else:
                result[key] = value
        return result

    @staticmethod
    def _preprocess_observation(
        observation: Mapping[str, Any],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        images = list(observation.get("images", {}).values())
        image_masks = list(observation.get("image_masks", {}).values())

        image_masks = [m if isinstance(m, torch.Tensor) else torch.tensor(m, dtype=torch.bool) for m in image_masks]

        lang_tokens = observation.get("tokenized_prompt")
        lang_masks = observation.get("tokenized_prompt_mask")

        state = observation.get("state")

        if lang_tokens is None or lang_masks is None or state is None:
            msg = "Observation is missing required fields for Pi0Model"
            raise ValueError(msg)

        return (
            images,
            image_masks,
            cast("torch.Tensor", lang_tokens),
            cast("torch.Tensor", lang_masks),
            cast(
                "torch.Tensor",
                state,
            ),
        )
