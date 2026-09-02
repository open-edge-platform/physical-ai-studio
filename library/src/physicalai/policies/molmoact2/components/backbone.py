# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MolmoAct2 assembly and continuous action generation.

``MolmoAct2ForConditionalGeneration`` is the checkpoint root: it owns the
``model`` backbone (text + vision + action expert) and the ``lm_head``. Weight
keys are ``model.transformer.*``, ``model.vision_backbone.*``,
``model.action_expert.*`` and ``lm_head.weight``.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING, cast

import torch
from torch import nn
from torch.distributions import Beta

if TYPE_CHECKING:
    from .action_expert import ActionExpert, ActionExpertBlock, ActionExpertContext
    from .text import KVState, MolmoAct2TextModel
    from .vision import MolmoAct2VisionBackbone


def _sample_beta_timesteps(
    *,
    batch_size: int,
    device: torch.device,
    cutoff: float,
    time_offset: float,
    time_scale: float,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    """Sample flow-matching timesteps from a scaled Beta distribution.

    Returns:
        Timesteps ``(batch_size,)`` in ``[time_offset, min(cutoff, time_offset + time_scale)]``.
    """
    upper = min(cutoff, time_offset + time_scale)
    samples = Beta(torch.tensor(alpha, device=device), torch.tensor(beta, device=device)).sample((batch_size,))
    scale = upper - time_offset
    if scale == 0:
        return torch.full((batch_size,), time_offset, device=device, dtype=samples.dtype)
    return time_offset + scale * samples


def _merge_image_features(
    embeddings: torch.Tensor,
    image_features: torch.Tensor,
    is_image_patch: torch.Tensor,
) -> torch.Tensor:
    """Add dense per-example image features to their ordered token positions.

    Returns:
        Embeddings with image features added at image-patch positions.
    """
    feature_indices = is_image_patch.to(dtype=torch.int64).cumsum(1) - 1
    feature_indices = feature_indices.clamp_min(0)
    gather_indices = feature_indices[..., None].expand(-1, -1, image_features.shape[-1])
    aligned_features = torch.gather(image_features, 1, gather_indices)
    image_delta = torch.where(is_image_patch[..., None], aligned_features, torch.zeros_like(aligned_features))
    return embeddings + image_delta


class MolmoAct2Backbone(nn.Module):
    """Text + vision + action expert. Checkpoint prefix: ``model.*``."""

    def __init__(
        self,
        *,
        transformer: MolmoAct2TextModel,
        vision_backbone: MolmoAct2VisionBackbone,
        action_expert: ActionExpert | None,
        image_patch_id: int,
        mask_action_dim_padding: bool,
        flow_matching_num_steps: int,
        max_action_dim: int,
        num_flow_timesteps: int,
        flow_matching_cutoff: float,
        flow_matching_time_offset: float,
        flow_matching_time_scale: float,
        flow_matching_beta_alpha: float,
        flow_matching_beta_beta: float,
    ) -> None:
        """Assemble already constructed model components and rollout settings."""
        super().__init__()
        self.transformer = transformer
        self.vision_backbone = vision_backbone
        self.action_expert = action_expert
        self.image_patch_id = image_patch_id
        self.mask_action_dim_padding = mask_action_dim_padding
        self.flow_matching_num_steps = flow_matching_num_steps
        self.max_action_dim = max_action_dim
        self.num_flow_timesteps = num_flow_timesteps
        self.flow_matching_cutoff = flow_matching_cutoff
        self.flow_matching_time_offset = flow_matching_time_offset
        self.flow_matching_time_scale = flow_matching_time_scale
        self.flow_matching_beta_alpha = flow_matching_beta_alpha
        self.flow_matching_beta_beta = flow_matching_beta_beta

    def _require_action_expert(self) -> ActionExpert:
        """Return the action expert.

        Returns:
            The action expert module.

        Raises:
            RuntimeError: If the checkpoint has no action expert.
        """
        if self.action_expert is None:
            msg = "This MolmoAct2 checkpoint does not include an action expert."
            raise RuntimeError(msg)
        return self.action_expert

    def build_input_embeddings(
        self,
        input_ids: torch.Tensor,
        images: torch.Tensor | None,
        token_pooling: torch.Tensor | None,
    ) -> torch.Tensor:
        """Embed tokens and add projected image features at image-patch positions.

        Returns:
            Token embeddings ``(batch, seq_len, hidden)``.
        """
        token_ids = input_ids * (input_ids != -1).to(input_ids.dtype)
        embeddings = self.transformer.wte(token_ids)
        if images is None:
            return embeddings

        # Keep the dtype conversion off the input boundary so OpenVINO retains the semantic ``images`` input name.
        images = images.reshape(images.shape)
        image_features = self.vision_backbone(images.to(self.vision_backbone.dtype), token_pooling).to(embeddings.dtype)
        is_image_patch = token_ids == self.image_patch_id
        return _merge_image_features(embeddings, image_features, is_image_patch)

    @staticmethod
    def _build_attention_bias(
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        token_type_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        """Build an additive causal bias; image tokens attend bidirectionally.

        Returns:
            Additive attention bias ``(batch, 1, seq_len, seq_len)``.
        """
        batch_size, seq_len = inputs_embeds.shape[:2]
        device, dtype = inputs_embeds.device, inputs_embeds.dtype

        valid = (
            torch.ones(batch_size, seq_len, device=device, dtype=torch.bool)
            if attention_mask is None
            else attention_mask.to(device=device, dtype=torch.bool)
        )
        causal = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        causal = causal[None, None].expand(batch_size, 1, -1, -1)
        if token_type_ids is not None:
            image_mask = token_type_ids.to(device=device, dtype=torch.bool)
            can_attend_back = image_mask[:, None, :, None] & image_mask[:, None, None, :]
            causal = causal | can_attend_back  # noqa: PLR6104  (causal is an expand view; in-place is unsafe)
        allowed = valid[:, None, None, :] & causal
        return torch.where(allowed, 0.0, torch.finfo(dtype).min).to(dtype)

    @staticmethod
    def _encoder_attention_mask(
        input_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Compute the text positions the action expert may cross-attend to.

        Returns:
            Boolean mask ``(batch, seq_len)`` or ``None``.
        """
        if attention_mask is not None:
            return attention_mask.to(dtype=torch.bool)
        if input_ids is not None:
            return input_ids != -1
        return None

    @staticmethod
    def _kv_to_sequence(cache: torch.Tensor) -> torch.Tensor:
        """Flatten KV heads into the feature dimension.

        Returns:
            Tensor ``(batch, seq, heads * head_dim)``.
        """
        batch, heads, seq_len, head_dim = cache.shape
        return cache.permute(0, 2, 1, 3).reshape(batch, seq_len, heads * head_dim)

    def _mask_action_dims(self, tensor: torch.Tensor, action_dim_is_pad: torch.Tensor | None) -> torch.Tensor:
        """Zero out padded action dimensions when configured to do so.

        Returns:
            The masked tensor (or the input unchanged when masking is off).
        """
        if not self.mask_action_dim_padding or action_dim_is_pad is None:
            return tensor
        valid = (~action_dim_is_pad)[:, None, :]
        return tensor * valid

    def encode(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        token_type_ids: torch.Tensor | None,
        images: torch.Tensor | None,
        token_pooling: torch.Tensor | None,
    ) -> list[KVState]:
        """Run the vision+text encoder.

        Returns:
            Per-layer ``(key, value)`` states from the text decoder.
        """
        inputs_embeds = self.build_input_embeddings(input_ids, images, token_pooling)
        attention_bias = self._build_attention_bias(inputs_embeds, attention_mask, token_type_ids)
        _, kv_states = self.transformer(inputs_embeds, attention_bias=attention_bias)
        return kv_states

    @torch.no_grad()
    def generate_actions_from_inputs(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        images: torch.Tensor | None = None,
        token_pooling: torch.Tensor | None = None,
        action_dim_is_pad: torch.Tensor | None = None,
        action_horizon: int,
        num_steps: int | None = None,
        sample_noise: bool = False,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Continuous flow-matching action generation.

        Encodes text+vision, collects per-layer KV, then Euler-integrates the
        action expert's velocity field. Integration starts from zeros by default
        (deterministic, export-friendly); set ``sample_noise`` to start from a
        sampled Gaussian vector instead.

        Returns:
            Action trajectory ``(batch, action_horizon, max_action_dim)``.

        Raises:
            ValueError: If ``num_steps`` is not positive.
        """
        action_expert = self._require_action_expert()
        steps = int(num_steps or self.flow_matching_num_steps)
        if steps <= 0:
            msg = f"num_steps must be >= 1, got {steps}."
            raise ValueError(msg)

        batch_size = input_ids.shape[0]
        device = action_expert.action_embed.weight.device
        dtype = action_expert.action_embed.weight.dtype
        context = self._encode_action_context(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            images=images,
            token_pooling=token_pooling,
            seq_len=action_horizon,
            device=device,
            dtype=dtype,
        )

        shape = (batch_size, action_horizon, self.max_action_dim)
        if sample_noise:
            noise = torch.randn(*shape, device=device, dtype=dtype, generator=generator)
            trajectory = self._mask_action_dims(noise, action_dim_is_pad)
        else:
            trajectory = torch.zeros(*shape, device=device, dtype=dtype)

        dt = 1.0 / steps
        for step in range(steps):
            timestep = torch.full((batch_size,), step / steps, device=device, dtype=dtype)
            velocity = action_expert.forward_with_context(trajectory, timestep, context=context)
            velocity = self._mask_action_dims(velocity, action_dim_is_pad)
            trajectory = self._mask_action_dims(trajectory + dt * velocity, action_dim_is_pad)
        return trajectory

    def _encode_action_context(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        token_type_ids: torch.Tensor | None,
        images: torch.Tensor | None,
        token_pooling: torch.Tensor | None,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        freeze_encoder: bool = False,
    ) -> ActionExpertContext:
        """Encode text+vision and build the action expert's cross-attention context.

        Returns:
            The prepared :class:`ActionExpertContext`.
        """
        with torch.no_grad() if freeze_encoder else nullcontext():
            kv_states = self.encode(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                images=images,
                token_pooling=token_pooling,
            )
        encoder_kv = [(self._kv_to_sequence(k), self._kv_to_sequence(v)) for k, v in kv_states]
        encoder_mask = self._encoder_attention_mask(input_ids, attention_mask)
        return self._require_action_expert().prepare_context(
            encoder_kv_states=encoder_kv,
            encoder_attention_mask=encoder_mask,
            seq_len=seq_len,
            device=device,
            dtype=dtype,
        )

    def _flow_interpolation(
        self,
        actions: torch.Tensor,
        action_dim_is_pad: torch.Tensor | None,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Interpolate ``x_t`` between noise and actions at ``num_flow_timesteps`` sampled timesteps.

        Draws ``config.num_flow_timesteps`` independent (timestep, noise)
        samples per example (matching the reference MolmoAct2 training
        recipe) to reduce the flow-matching loss's per-step variance; the
        encoder only runs once per example regardless of this count (see
        :meth:`predict_flow_velocity`).

        Returns:
            ``(x_t, timesteps, target_velocity)``, each with a leading
            ``batch_size * num_flow_timesteps`` dimension.
        """
        batch_size, horizon, action_dim = actions.shape
        num_flow_timesteps = max(1, int(self.num_flow_timesteps))
        flat_batch = batch_size * num_flow_timesteps
        timesteps = _sample_beta_timesteps(
            batch_size=flat_batch,
            device=actions.device,
            cutoff=self.flow_matching_cutoff,
            time_offset=self.flow_matching_time_offset,
            time_scale=self.flow_matching_time_scale,
            alpha=self.flow_matching_beta_alpha,
            beta=self.flow_matching_beta_beta,
        ).to(dtype)
        expanded_action_dim_is_pad = (
            action_dim_is_pad.repeat_interleave(num_flow_timesteps, dim=0) if action_dim_is_pad is not None else None
        )
        noise = self._mask_action_dims(
            torch.randn(flat_batch, horizon, action_dim, device=actions.device, dtype=dtype),
            expanded_action_dim_is_pad,
        )
        actions_expanded = actions.repeat_interleave(num_flow_timesteps, dim=0)
        t = timesteps.view(flat_batch, 1, 1)
        x_t = (1.0 - t) * noise + t * actions_expanded
        return x_t, timesteps, actions_expanded - noise

    def _predict_flow_velocity_per_layer(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        token_type_ids: torch.Tensor | None,
        images: torch.Tensor | None,
        token_pooling: torch.Tensor | None,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        action_horizon: int,
        num_flow_timesteps: int,
        freeze_encoder: bool,
    ) -> torch.Tensor:
        """Stream each text layer's KV directly through its matching action block.

        Returns:
            Predicted flow velocity for the flattened flow-timestep batch.
        """
        action_expert = self._require_action_expert()
        dtype = action_expert.action_embed.weight.dtype
        with torch.no_grad() if freeze_encoder else nullcontext():
            hidden_states = self.build_input_embeddings(input_ids, images, token_pooling)
            attention_bias = self._build_attention_bias(hidden_states, attention_mask, token_type_ids)
            position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
            position_embeddings = self.transformer.rotary_emb(hidden_states, position_ids)

        encoder_mask = self._encoder_attention_mask(input_ids, attention_mask)
        cross_mask, self_mask, rope_cache = action_expert.prepare_context_metadata(
            encoder_attention_mask=encoder_mask,
            seq_len=action_horizon,
            device=x_t.device,
            dtype=dtype,
        )
        if cross_mask is not None and num_flow_timesteps != 1:
            cross_mask = cross_mask.repeat_interleave(num_flow_timesteps, dim=0)

        conditioning = action_expert.time_conditioning(timesteps)
        action_hidden = action_expert.action_embed(x_t)
        use_gradient_checkpointing = (
            self.transformer.gradient_checkpointing
            and action_expert.gradient_checkpointing
            and self.training
            and torch.is_grad_enabled()
        )

        def run_layer(
            layer_idx: int,
            layer_hidden: torch.Tensor,
            layer_action_hidden: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            decoder_block = self.transformer.blocks[layer_idx]
            action_block = cast("ActionExpertBlock", action_expert.blocks[layer_idx])
            with torch.no_grad() if freeze_encoder else nullcontext():
                next_hidden, (key_states, value_states) = decoder_block(
                    layer_hidden,
                    position_embeddings,
                    attention_bias,
                )
            key_sequence = self._kv_to_sequence(key_states)
            value_sequence = self._kv_to_sequence(value_states)
            key_context, value_context = action_expert.project_kv_context(
                action_block,
                key_sequence,
                value_sequence,
            )
            if num_flow_timesteps != 1:
                key_context = key_context.repeat_interleave(num_flow_timesteps, dim=0)
                value_context = value_context.repeat_interleave(num_flow_timesteps, dim=0)
            next_action_hidden = action_block(
                layer_action_hidden,
                conditioning,
                cross_kv=(key_context, value_context),
                self_attn_mask=self_mask,
                cross_attn_mask=cross_mask,
                is_causal=action_expert.causal_attn,
                rope_cache=rope_cache,
            )
            return next_hidden, next_action_hidden

        for layer_idx in range(len(self.transformer.blocks)):
            if use_gradient_checkpointing:
                hidden_states, action_hidden = torch.utils.checkpoint.checkpoint(  # pyrefly: ignore[not-iterable]
                    lambda layer_hidden, layer_action_hidden, idx=layer_idx: run_layer(
                        idx,
                        layer_hidden,
                        layer_action_hidden,
                    ),
                    hidden_states,
                    action_hidden,
                    use_reentrant=False,
                )
            else:
                hidden_states, action_hidden = run_layer(layer_idx, hidden_states, action_hidden)
        return action_expert.final_layer(action_hidden, conditioning)

    def predict_flow_velocity(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        token_type_ids: torch.Tensor | None,
        images: torch.Tensor | None,
        token_pooling: torch.Tensor | None,
        actions: torch.Tensor,
        action_dim_is_pad: torch.Tensor | None,
        freeze_encoder: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Flow-matching training forward: predict velocity and its target.

        Samples ``config.num_flow_timesteps`` per-example (timestep, noise)
        pairs, interpolates ``x_t`` between noise and the ground-truth
        ``actions`` for each, then predicts the velocity that the action
        expert should produce. The vision+text encoder runs exactly once per
        example (its output context is repeated across the
        ``num_flow_timesteps`` samples), matching the reference MolmoAct2
        training recipe's variance-reduction trick. When ``freeze_encoder``
        is set the encoder runs under ``no_grad`` (action-expert-only
        training).

        Returns:
            ``(predicted_velocity, target_velocity)``, both
            ``(batch, num_flow_timesteps, horizon, max_action_dim)``.
        """
        action_expert = self._require_action_expert()
        dtype = action_expert.action_embed.weight.dtype
        actions = self._mask_action_dims(actions.to(dtype), action_dim_is_pad)
        batch_size, horizon, action_dim = actions.shape
        num_flow_timesteps = max(1, int(self.num_flow_timesteps))
        x_t, timesteps, target_velocity = self._flow_interpolation(actions, action_dim_is_pad, dtype)
        predicted_velocity = self._predict_flow_velocity_per_layer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            images=images,
            token_pooling=token_pooling,
            x_t=x_t,
            timesteps=timesteps,
            action_horizon=horizon,
            num_flow_timesteps=num_flow_timesteps,
            freeze_encoder=freeze_encoder,
        )
        predicted_velocity = predicted_velocity.view(batch_size, num_flow_timesteps, horizon, action_dim)
        target_velocity = target_velocity.view(batch_size, num_flow_timesteps, horizon, action_dim)
        return predicted_velocity, target_velocity


class MolmoAct2ForConditionalGeneration(nn.Module):
    """Checkpoint root module: ``model`` backbone + ``lm_head``."""

    def __init__(self, *, model: MolmoAct2Backbone, hidden_size: int, vocab_size: int) -> None:
        """Build the backbone and the language-model head."""
        super().__init__()
        self.model = model
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
