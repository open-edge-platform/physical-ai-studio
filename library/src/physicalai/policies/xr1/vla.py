# Copyright (C) 2026 Xiaomi Corporation.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""XR1 vision-language-action model.

The backbone encodes images and the instruction once; the DiT action expert then
denoises an action chunk while attending over the backbone's key/value cache, layer
by layer. Its query sequence is ``[sink, state, actions]``: a learned sink token
that gives attention somewhere neutral to point, the projected robot state, and the
action tokens being predicted.

Two pieces of the reference recipe are training-only and configurable here:

* ``async_train`` conditions on an executed action prefix taken from the head of the
  target chunk, and weights the loss by the error of a full rollout;
* ``enable_choice_head`` trains the auxiliary Choice Policy head - ``n_choices``
  candidate chunks plus a score per candidate, supervised winner-takes-all. Its query
  tokens are appended to the language sequence *after* the action expert's cache has
  been snapshotted, so they can never leak into the action expert. Inference does not
  run the head at all.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import nn

from physicalai.policies.base.model import Model
from physicalai.policies.xr1.io import (
    build_dit_attention_mask,
    continue_position_ids,
    continue_text_position_ids,
)
from physicalai.policies.xr1.model import XR1FlowModel
from physicalai.policies.xr1.qwen3vl_dit import DiT, MLPProjector, TimestepEmbedder

if TYPE_CHECKING:
    from transformers.cache_utils import Cache

    from physicalai.policies.xr1.config import XR1Config
    from physicalai.policies.xr1.vlm import XR1Qwen3VL, XR1VLMOutput

MODULATION_TERMS = 6
TIMESTEP_SCALE = 1000
SUFFIX_POSITION_OFFSET = 10
MAX_PREFIX_LENGTH = 6
PROJECTOR_DEPTH = 2
CHOICE_PROJECTOR_DEPTH = 4


class XR1Model(Model):
    """Qwen3-VL backbone plus DiT action expert, trained by flow matching."""

    def __init__(self, config: XR1Config, vlm: XR1Qwen3VL | None = None) -> None:
        """Build the model.

        Args:
            config: Model configuration.
            vlm: Pre-built backbone. When ``None`` the backbone is instantiated
                from ``config.vlm_model_id``. Tests inject a small random backbone
                to avoid a multi-gigabyte download.
        """
        super().__init__()
        self.config = config
        self.vlm = vlm if vlm is not None else self._build_vlm(config)

        text_config = cast("Any", self.vlm.config).text_config
        self._validate_against_backbone(config, text_config)

        dit_hidden = config.dit_hidden_size
        self.dit = DiT(
            hidden_size=dit_hidden,
            layer_num=config.dit_num_layers,
            head_dim=config.dit_head_dim,
            kv_heads=config.dit_kv_heads,
        )
        self.state_projector = MLPProjector(config.max_state_dim, dit_hidden, num_layers=PROJECTOR_DEPTH)
        self.action_projector = MLPProjector(config.max_action_dim, dit_hidden, num_layers=PROJECTOR_DEPTH)
        self.action_output_layer = MLPProjector(
            dit_hidden,
            config.max_action_dim,
            inter_dim=dit_hidden,
            num_layers=PROJECTOR_DEPTH,
        )
        self.t_embedder = TimestepEmbedder(dit_hidden)
        self.t_projector = MLPProjector(dit_hidden, MODULATION_TERMS * dit_hidden, bias=True)
        self.sink = nn.Embedding(1, dit_hidden)

        from transformers.models.qwen3_vl.modeling_qwen3_vl import (  # noqa: PLC0415  # heavy import, deferred
            Qwen3VLTextRotaryEmbedding,
        )

        self.rotary_emb = Qwen3VLTextRotaryEmbedding(text_config)  # type: ignore[arg-type]

        self.flow = XR1FlowModel(
            num_inference_steps=config.num_inference_steps,
            flow_sampling=config.flow_sampling,
            beta_alpha=config.beta_alpha,
            beta_beta=config.beta_beta,
            enable_freq=config.enable_freq,
            freq_coefficient=config.freq_coefficient,
            freq_excluded_dims=config.freq_excluded_dims,
        )

        if config.enable_choice_head:
            self._build_choice_head(config, text_config.hidden_size)

        self._apply_training_flags(config)

    @staticmethod
    def _build_vlm(config: XR1Config) -> XR1Qwen3VL:
        """Instantiate the Qwen3-VL backbone.

        With ``vlm_pretrained`` set the checkpoint weights are downloaded; otherwise
        only the backbone *config* is fetched and the weights are randomly
        initialized, optionally shrunk by ``vlm_config_overrides``. The random path
        keeps smoke runs and tests off a multi-gigabyte download.

        Args:
            config: Model configuration.

        Returns:
            The backbone, in the configured dtype and attention implementation.
        """
        from transformers import Qwen3VLConfig  # noqa: PLC0415  # heavy import, deferred

        # Deferred to avoid a circular import with the policy module.
        from physicalai.policies.xr1.vlm import XR1Qwen3VL  # noqa: PLC0415

        dtype = getattr(torch, config.dtype)
        if config.vlm_pretrained:
            return XR1Qwen3VL.from_pretrained(
                config.vlm_model_id,
                attn_implementation=config.vlm_attn_implementation,
                dtype=dtype,
            )

        vlm_config = Qwen3VLConfig.from_pretrained(config.vlm_model_id)
        apply_config_overrides(vlm_config, config.vlm_config_overrides or {})
        # ``transformers`` exposes no public "build from config" entry point on a
        # concrete model class; ``_from_config`` is what the reference
        # implementation and the library's own Auto classes use.
        return XR1Qwen3VL._from_config(  # noqa: SLF001
            vlm_config,
            attn_implementation=config.vlm_attn_implementation,
            dtype=dtype,
        )

    @staticmethod
    def _validate_against_backbone(config: XR1Config, text_config: Any) -> None:  # noqa: ANN401
        """Check the DiT can read the backbone's cache.

        These cannot be checked in ``XR1Config.__post_init__`` because they depend
        on the backbone's own config, but they must be checked before the first
        forward pass rather than surfacing as a shape error deep inside attention.

        Args:
            config: Model configuration.
            text_config: The backbone's text configuration.

        Raises:
            ValueError: If depth, head dim or kv heads disagree.
        """
        vlm_layers = text_config.num_hidden_layers
        if config.dit_num_layers > vlm_layers:
            msg = (
                f"dit_num_layers ({config.dit_num_layers}) exceeds the backbone's "
                f"{vlm_layers} layers; each DiT layer attends over one cached VLM layer"
            )
            raise ValueError(msg)

        vlm_head_dim = getattr(text_config, "head_dim", None) or (
            text_config.hidden_size // text_config.num_attention_heads
        )
        if config.dit_head_dim != vlm_head_dim:
            msg = (
                f"dit_head_dim ({config.dit_head_dim}) must match the backbone head dim ({vlm_head_dim}); "
                "the DiT applies rotary embeddings to cached VLM keys"
            )
            raise ValueError(msg)

        if config.dit_kv_heads != text_config.num_key_value_heads:
            msg = (
                f"dit_kv_heads ({config.dit_kv_heads}) must match the backbone's "
                f"num_key_value_heads ({text_config.num_key_value_heads})"
            )
            raise ValueError(msg)

    def _build_choice_head(self, config: XR1Config, vlm_hidden: int) -> None:
        """Build the Choice Policy head.

        The head reads the backbone at dedicated query positions: one state token,
        one token per action step, and one score token. The reference implementation
        gets those positions by adding ``<state>``, ``<a_i>`` and ``<score>`` to the
        tokenizer and scattering learned embeddings over them. Adding vocabulary
        entries would fork the released tokenizer, so the same embeddings are appended
        directly instead - the tokens have no text meaning, only a learned embedding,
        so nothing is lost.

        Args:
            config: Model configuration.
            vlm_hidden: Backbone hidden width.
        """
        self.state_projector_choice = MLPProjector(
            config.max_state_dim,
            vlm_hidden,
            num_layers=PROJECTOR_DEPTH,
        )
        # One learned query per action step, so a step's candidates can depend on
        # where in the chunk it sits, plus one for the score.
        self.action_query_embed = nn.Embedding(config.chunk_size, vlm_hidden)
        self.score_query_embed = nn.Embedding(1, vlm_hidden)
        self.action_projector_choice = nn.Sequential(
            MLPProjector(vlm_hidden, vlm_hidden, num_layers=CHOICE_PROJECTOR_DEPTH),
            MLPProjector(vlm_hidden, config.max_action_dim * config.n_choices),
        )
        self.score_projector_choice = nn.Sequential(
            MLPProjector(vlm_hidden, vlm_hidden, num_layers=CHOICE_PROJECTOR_DEPTH),
            MLPProjector(vlm_hidden, config.n_choices),
        )

    def _apply_training_flags(self, config: XR1Config) -> None:
        """Apply freezing and gradient-checkpointing options.

        Args:
            config: Model configuration.
        """
        if config.freeze_vlm:
            self.vlm.requires_grad_(requires_grad=False)
        elif config.freeze_vision_encoder:
            self.vlm.model.visual.requires_grad_(requires_grad=False)

        if config.gradient_checkpointing and not config.freeze_vlm:
            self.vlm.gradient_checkpointing_enable()

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the action expert.

        Returns:
            Dtype of the DiT parameters.
        """
        weight = cast("torch.Tensor", self.action_projector.layers[0].weight)
        return weight.dtype

    def encode_prompt(self, batch: dict[str, Any]) -> XR1VLMOutput:
        """Run the backbone over images and the instruction.

        Args:
            batch: Preprocessed batch carrying ``input_ids`` and, when images are
                present, ``pixel_values`` and ``image_grid_thw``.

        Returns:
            Cache, position grid and padding mask for the action expert.
        """
        return self.vlm.encode(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            pixel_values=batch.get("pixel_values"),
            image_grid_thw=batch.get("image_grid_thw"),
            mm_token_type_ids=batch.get("mm_token_type_ids"),
        )

    def dit_forward(
        self,
        noisy_action: torch.Tensor,
        timestep: torch.Tensor,
        action_mask: torch.Tensor,
        state_embed: torch.Tensor,
        position_embeds: tuple[torch.Tensor, torch.Tensor],
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]],
        attn_mask: torch.Tensor | None,
        prefix_length: int = 0,
    ) -> torch.Tensor:
        """Predict the velocity field for one denoising step.

        Args:
            noisy_action: Noisy action chunk of shape ``(batch, horizon, action_dim)``.
            timestep: Timestep of shape ``(batch, 1, 1)``.
            action_mask: Mask marking supervised action entries.
            state_embed: Projected state of shape ``(batch, state_len, hidden)``.
            position_embeds: ``(cos, sin)`` for the query sequence.
            past_key_values: Backbone cache.
            attn_mask: Boolean attention mask over cache and query.
            prefix_length: Number of leading action steps supplied as a prefix;
                their predictions are zeroed.

        Returns:
            Predicted velocity of shape ``(batch, horizon, action_dim)``.
        """
        hidden = self.config.dit_hidden_size
        conditioning = self.t_projector(self.t_embedder(timestep[:, 0, 0] * TIMESTEP_SCALE))
        conditioning = conditioning.view(-1, MODULATION_TERMS, hidden)

        action_tokens = self.action_projector(noisy_action * action_mask)
        sink = self.sink.weight[None].expand(state_embed.shape[0], -1, -1)
        hidden_states = torch.cat([sink, state_embed, action_tokens], dim=1).contiguous()

        hidden_states = self.dit(hidden_states, past_key_values, attn_mask, position_embeds, conditioning)
        output = self.action_output_layer(hidden_states[:, -action_tokens.shape[1] :])
        if prefix_length:
            output = torch.cat([torch.zeros_like(output[:, :prefix_length]), output[:, prefix_length:]], dim=1)
        return output

    def _repeat(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Repeat a tensor ``training_repeat`` times while training.

        Each sample is denoised at several timesteps per optimizer step, which the
        reference implementation uses to cut the cost of the (shared) backbone pass.

        Args:
            tensor: Tensor to repeat.
            dim: Dimension to repeat along.

        Returns:
            The repeated tensor, or the input unchanged in eval mode.
        """
        if not self.training or self.config.training_repeat == 1:
            return tensor
        return tensor.repeat_interleave(self.config.training_repeat, dim=dim)

    def _prepare_dit_inputs(
        self,
        batch: dict[str, Any],
        vlm_outputs: XR1VLMOutput,
        action: torch.Tensor,
        action_mask: torch.Tensor,
        prefix_length: int,
    ) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
        """Assemble every tensor the action expert needs.

        Args:
            batch: Preprocessed batch carrying ``state``.
            vlm_outputs: Backbone outputs.
            action: Target (or placeholder) action chunk.
            action_mask: Mask marking supervised action entries.
            prefix_length: Number of leading action steps supplied as a prefix.

        Returns:
            ``(dit_kwargs, action, action_mask)`` where the action tensors have been
            repeated to match the query batch.
        """
        state = batch["state"].to(self.dtype)
        batch_size, action_length, _ = action.shape
        state_length = state.shape[1]
        query_length = 1 + state_length + action_length

        suffix_length = max(0, action_length - prefix_length)
        position_ids = continue_position_ids(
            vlm_outputs.position_ids,
            query_length,
            batch_size=batch_size,
            suffix_offset=SUFFIX_POSITION_OFFSET,
            suffix_length=suffix_length,
        )
        attn_mask = build_dit_attention_mask(
            vlm_outputs.attention_mask,
            query_length,
            prefix_length=prefix_length,
            prefix_mask_prob=self.config.prefix_mask_prob if self.training else 0.0,
            state_length=state_length,
        )

        state_embed = self.state_projector(state)
        action = self._repeat(action)
        action_mask = self._repeat(action_mask)
        state_embed = self._repeat(state_embed)
        attn_mask = self._repeat(attn_mask)
        position_ids = self._repeat(position_ids, dim=1)

        position_embeds = self.rotary_emb(action, position_ids)
        dit_kwargs = {
            "action_mask": action_mask,
            "state_embed": state_embed,
            "position_embeds": position_embeds,
            "past_key_values": vlm_outputs.past_key_values,
            "attn_mask": attn_mask,
            "prefix_length": prefix_length,
        }
        return dit_kwargs, action, action_mask

    def _sample_prefix_length(self, action_length: int) -> int:
        """Draw the training-time action prefix length.

        Args:
            action_length: Length of the action chunk.

        Returns:
            A prefix length in ``1 .. MAX_PREFIX_LENGTH`` on roughly half the steps
            when ``async_train`` is set, otherwise 0.
        """
        if self.config.async_train and random.random() < 0.5:  # noqa: S311, PLR2004 - schedule jitter, not crypto
            return min(random.randint(1, MAX_PREFIX_LENGTH), action_length)  # noqa: S311
        return 0

    @staticmethod
    def _requested_prefix_length(batch: dict[str, Any], action_length: int) -> int:
        """Read the caller's action prefix length for inference.

        Asynchronous execution is a property of the request, not of the module's
        train/eval flag, so this never consults ``self.training``: calling
        :meth:`predict_action_chunk` on a module left in train mode must not start
        demanding an ``action_prefix`` the caller never supplied.

        Args:
            batch: Preprocessed batch; may carry an explicit ``prefix_length``.
            action_length: Length of the action chunk.

        Returns:
            The requested prefix length, clamped to the chunk length.
        """
        return min(int(batch.get("prefix_length", 0) or 0), action_length)

    def compute_loss(  # noqa: PLR0914 - the flow-matching step needs its intermediates named
        self,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        """Compute the flow-matching training loss.

        Args:
            batch: Preprocessed batch with ``input_ids``, ``state``, ``action`` and
                ``action_mask``.

        Returns:
            ``(loss, metrics)``; metrics carry the individual loss terms as
            detached tensors.
        """
        action = batch["action"].to(self.dtype)
        action_mask = batch["action_mask"].to(self.dtype)
        prefix_length = self._sample_prefix_length(action.shape[1]) if self.training else 0

        vlm_outputs = self.encode_prompt(batch)
        choice_losses = self._choice_loss(batch, vlm_outputs) if self.config.enable_choice_head else None
        dit_kwargs, action, action_mask = self._prepare_dit_inputs(
            batch,
            vlm_outputs,
            action,
            action_mask,
            prefix_length,
        )

        noise = torch.randn_like(action)
        timestep = self.flow.sample_timestep(action.shape[0], action.device, action.dtype)
        noisy_action = self.flow.interpolate(noise, action, timestep)
        target = self.flow.velocity_target(noise, action)

        prefix = action[:, :prefix_length]
        pred = self.dit_forward(
            torch.cat([prefix, noisy_action[:, prefix_length:]], dim=1),
            timestep,
            **dit_kwargs,
        )[:, prefix_length:]
        target = target[:, prefix_length:]

        if prefix_length:
            # Weight hard samples higher, measured by the error of a full rollout.
            rollout = self.flow.generate(
                torch.cat([prefix, noise[:, prefix_length:]], dim=1),
                lambda sample, step: self.dit_forward(sample, step, **dit_kwargs),
            )
            weight = (rollout[:, prefix_length:] - action[:, prefix_length:]).abs()
        else:
            weight = torch.ones_like(pred)

        loss_mse, loss_freq = self.flow.flow_loss(pred, target, action_mask[:, prefix_length:], weight)
        loss = 0.5 * loss_mse + self.config.freq_coefficient * loss_freq
        metrics: dict[str, torch.Tensor | float] = {
            "loss_mse": loss_mse.detach(),
            "loss_freq": loss_freq.detach(),
        }

        if choice_losses is not None:
            loss_choice, loss_score = choice_losses
            loss = loss + 0.5 * loss_choice + 0.5 * loss_score
            metrics["loss_choice"] = loss_choice.detach()
            metrics["loss_score"] = loss_score.detach()

        metrics["loss"] = loss.detach()
        return loss, metrics

    def _choice_queries(self, state: torch.Tensor, horizon: int) -> torch.Tensor:
        """Build the choice turn's query embeddings.

        The layout mirrors the reference implementation's appended chat turn:
        ``"Robot state: <state>"`` followed by ``<a_0>...<a_{T-1}><score>``.

        Args:
            state: Padded state of shape ``(batch, state_len, max_state_dim)``.
            horizon: Number of action steps, one query token each.

        Returns:
            Embeddings of shape ``(batch, state_len + horizon + 1, vlm_hidden)``.

        Raises:
            ValueError: If ``horizon`` exceeds the configured ``chunk_size``, which
                would index past the learned per-step query embeddings.
        """
        if horizon > self.config.chunk_size:
            msg = f"action horizon ({horizon}) exceeds chunk_size ({self.config.chunk_size})"
            raise ValueError(msg)

        batch_size = state.shape[0]
        state_tokens = self.state_projector_choice(state)
        steps = torch.arange(horizon, device=state.device)
        action_tokens = self.action_query_embed(steps)[None].expand(batch_size, -1, -1)
        score_token = self.score_query_embed.weight[None].expand(batch_size, -1, -1)
        return torch.cat([state_tokens, action_tokens, score_token], dim=1).to(state_tokens.dtype)

    def _choice_predictions(
        self,
        hidden: torch.Tensor,
        state_length: int,
        horizon: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read candidate chunks and their scores off the query positions.

        Args:
            hidden: Hidden states over the choice turn, shape
                ``(batch, state_len + horizon + 1, vlm_hidden)``.
            state_length: Number of leading state tokens to skip.
            horizon: Number of action-query tokens.

        Returns:
            ``(candidates, scores)`` of shapes
            ``(batch, horizon, n_choices, max_action_dim)`` and ``(batch, n_choices)``.
        """
        action_hidden = hidden[:, state_length : state_length + horizon]
        candidates = self.action_projector_choice(action_hidden).view(
            hidden.shape[0],
            horizon,
            self.config.n_choices,
            self.config.max_action_dim,
        )
        return candidates, self.score_projector_choice(hidden[:, -1])

    @staticmethod
    def _choice_cache(batch: dict[str, Any], vlm_outputs: XR1VLMOutput) -> Cache:
        """Check the batch can supervise the choice head and return the prompt cache.

        Args:
            batch: Preprocessed batch.
            vlm_outputs: Backbone outputs for the prompt.

        Returns:
            The cache the choice turn continues from.

        Raises:
            KeyError: If the batch carries no action supervision.
            ValueError: If the backbone returned no cache to continue from.
        """
        if "action" not in batch or "action_mask" not in batch:
            msg = "The choice head needs 'action' and 'action_mask' supervision in the batch"
            raise KeyError(msg)
        if vlm_outputs.cache is None:
            msg = "The choice head needs the backbone cache; encode() must run with use_cache=True"
            raise ValueError(msg)
        return vlm_outputs.cache

    def _choice_loss(
        self,
        batch: dict[str, Any],
        vlm_outputs: XR1VLMOutput,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the Choice Policy losses.

        The head emits ``n_choices`` candidate chunks and one score per candidate.
        Only the candidate closest to the target is trained (winner-takes-all on the
        masked L1 error), so the candidates are free to spread over the distinct ways
        of solving the task instead of collapsing onto their average. The scores
        regress onto those same L1 errors, detached, which is what makes them usable
        as a selection signal at deployment.

        The target is the action chunk the flow branch already trains on, so no extra
        dataset fields are needed. Steps that run past the end of an episode carry a
        zero mask and contribute nothing.

        Args:
            batch: Preprocessed batch carrying ``state``, ``action`` and
                ``action_mask``.
            vlm_outputs: Backbone outputs for the prompt.

        Returns:
            ``(candidate_loss, score_loss)``.
        """
        cache = self._choice_cache(batch, vlm_outputs)

        state = batch["state"].to(self.dtype)
        target = batch["action"].to(self.dtype)
        mask = batch["action_mask"].to(self.dtype)
        horizon = target.shape[1]

        queries = self._choice_queries(state, horizon)
        hidden = self.vlm.continue_sequence(
            queries,
            cache=cache,
            prompt_attention_mask=vlm_outputs.attention_mask,
            position_ids=continue_text_position_ids(vlm_outputs.position_ids, queries.shape[1]),
        )

        candidates, scores = self._choice_predictions(hidden, state.shape[1], horizon)

        absolute_error = (candidates - target[:, :, None, :]).abs() * mask[:, :, None, :]
        # Mean over the supervised entries only, per sample and candidate.
        valid = mask.sum(dim=(1, 2)).clamp(min=1.0)[:, None]
        per_choice = absolute_error.sum(dim=(1, 3)) / valid

        winner = per_choice.argmin(dim=-1, keepdim=True)
        candidate_loss = per_choice.gather(1, winner).mean()
        score_loss = ((scores - per_choice.detach()) ** 2).mean()
        return candidate_loss, score_loss

    @torch.no_grad()
    def compute_val_loss(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        """Score a full rollout against the target chunk.

        A single-step velocity error is a weak proxy for policy quality, so
        validation integrates the flow and compares actions directly.

        Args:
            batch: Preprocessed batch with ``action`` and ``action_mask``.

        Returns:
            ``(loss, metrics)`` where the loss is the masked action MSE.
        """
        target = batch["action"].to(self.dtype)
        mask = batch["action_mask"].to(torch.bool)
        predicted = self.predict_action_chunk(batch)

        squared_error = (predicted - target) ** 2
        loss = squared_error[mask].mean() if torch.any(mask) else squared_error.sum() * 0.0
        return loss, {"loss": loss.detach(), "action_mse": loss.detach()}

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Any]) -> torch.Tensor:
        """Generate an action chunk by integrating the learned velocity field.

        Args:
            batch: Preprocessed batch with ``input_ids`` and ``state``; may carry an
                ``action_prefix`` and ``prefix_length`` for asynchronous execution.

        Returns:
            Action chunk of shape ``(batch, chunk_size, max_action_dim)``.
        """
        state = batch["state"]
        batch_size = state.shape[0]
        horizon = self.config.chunk_size
        action_dim = self.config.max_action_dim

        placeholder = torch.zeros(
            (batch_size, horizon, action_dim),
            device=state.device,
            dtype=self.dtype,
        )
        prefix_length = self._requested_prefix_length(batch, horizon)
        if prefix_length:
            placeholder[:, :prefix_length] = batch["action_prefix"][:, :prefix_length].to(self.dtype)

        action_mask = torch.ones_like(placeholder)
        vlm_outputs = self.encode_prompt(batch)
        dit_kwargs, placeholder, _ = self._prepare_dit_inputs(
            batch,
            vlm_outputs,
            placeholder,
            action_mask,
            prefix_length,
        )

        noise = torch.randn_like(placeholder)
        if prefix_length:
            noise = torch.cat([placeholder[:, :prefix_length], noise[:, prefix_length:]], dim=1)

        return self.flow.generate(
            noise,
            lambda sample, step: self.dit_forward(sample, step, **dit_kwargs),
        )

    def forward(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]] | torch.Tensor:
        """Train on a batch, or predict a chunk in eval mode.

        Args:
            batch: Preprocessed batch.

        Returns:
            ``(loss, metrics)`` while training, otherwise the predicted chunk.
        """
        if self.training:
            return self.compute_loss(batch)
        return self.predict_action_chunk(batch)

    @property
    def reward_delta_indices(self) -> None:
        """Rewards are not used by XR1.

        Returns:
            ``None``.
        """
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        """Action offsets that make up one chunk.

        Returns:
            Offsets ``0 .. chunk_size - 1``.
        """
        return list(range(self.config.chunk_size))

    @property
    def observation_delta_indices(self) -> None:
        """XR1 conditions on the current observation only.

        Returns:
            ``None``.
        """
        return None


def apply_config_overrides(config: Any, overrides: dict[str, Any]) -> None:  # noqa: ANN401
    """Apply nested overrides to a ``transformers`` config in place.

    Args:
        config: Config object to modify.
        overrides: Mapping of attribute names to values; nested dicts recurse into
            sub-configs such as ``text_config``.

    Raises:
        AttributeError: If an override names an attribute the config does not have,
            which would otherwise be silently ignored.
    """
    for key, value in overrides.items():
        if not hasattr(config, key):
            msg = f"{type(config).__name__} has no config field {key!r}"
            raise AttributeError(msg)
        current = getattr(config, key)
        if isinstance(value, dict) and hasattr(current, "to_dict"):
            apply_config_overrides(current, value)
        else:
            setattr(config, key, value)
