# Copyright 2025 2toINF (https://github.com/2toINF) and The HuggingFace Inc. team.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""The XVLA model: a Florence-2 encoder feeding a flow-matching action transformer.

The forward pass has two halves. First the vision-language half: every camera is pooled by
Florence-2's vision tower, the primary view's tokens are prepended to the tokenized prompt
and run through the BART encoder, and the remaining views are kept as a flat auxiliary
stream. Then the action half: the noised action chunk, the proprioceptive state and the
flow-matching timestep are encoded into one token per action step and denoised by the
soft-prompted transformer, conditioned on both visual streams.

Unlike a velocity-field flow model, XVLA's transformer predicts the *clean* action chunk at
every step. Inference therefore re-noises its own estimate toward ``t=0`` over
``num_denoising_steps`` iterations rather than integrating a velocity.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from physicalai.data.constants import IMAGE_MASKS, IMAGES, TOKENIZED_PROMPT
from physicalai.data.observation import ACTION, STATE
from physicalai.policies.base.model import Model

from .action_hub import build_action_space
from .soft_transformer import SoftPromptedTransformer

if TYPE_CHECKING:
    from .config import XVLAConfig

logger = logging.getLogger(__name__)

DOMAIN_ID = "domain_id"
"""Batch key holding the per-sample domain index the domain-aware layers select on."""


class XVLAModel(Model):
    """Florence-2 encoder plus the soft-prompted action transformer.

    Args:
        config: The policy configuration.
        action_dim: Action width of the dataset. Only ``action_mode="auto"`` uses it, to
            size the supervised (and emitted) slice of the model's action vector; it falls
            back to ``config.max_action_dim``.
    """

    def __init__(self, config: XVLAConfig, action_dim: int | None = None) -> None:
        """Build the action space, the Florence-2 backbone and the action transformer.

        Raises:
            ImportError: If ``transformers`` is not installed.
            ValueError: If the Florence-2 config does not declare a ``projection_dim``.
        """
        super().__init__()
        try:
            from transformers import Florence2Model  # noqa: PLC0415
        except ImportError as e:
            msg = "XVLA requires transformers. Install with: pip install 'physicalai-train[xvla]'"
            raise ImportError(msg) from e

        self.config = config

        if config.action_mode.lower() == "auto":
            self.action_space = build_action_space(
                "auto",
                real_dim=action_dim if action_dim is not None else config.max_action_dim,
                max_dim=config.max_action_dim,
            )
        else:
            self.action_space = build_action_space(config.action_mode)

        self.dim_action: int = self.action_space.dim_action
        self.dim_proprio: int = config.dim_proprio

        florence_config = config.build_florence_config()
        projection_dim = getattr(florence_config.vision_config, "projection_dim", None)
        if projection_dim is None:
            msg = "Florence-2 config must provide `projection_dim` for multimodal fusion."
            raise ValueError(msg)

        self.vlm = Florence2Model(florence_config)
        # XVLA only uses the encoder-side path of Florence-2; drop the text decoder entirely.
        del self.vlm.language_model.decoder

        self.transformer = SoftPromptedTransformer(
            hidden_size=config.hidden_size,
            multi_modal_input_size=projection_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            num_domains=config.num_domains,
            dim_action=self.dim_action,
            dim_propio=self.dim_proprio,
            dim_time=config.dim_time,
            len_soft_prompts=config.len_soft_prompts,
            max_len_seq=config.max_len_seq,
            use_hetero_proj=config.use_hetero_proj,
        )

        self.apply_freezing()
        self.apply_dtype()

    # ------------------------------------------------------------------ #
    # Precision and freezing                                              #
    # ------------------------------------------------------------------ #
    @property
    def target_dtype(self) -> torch.dtype:
        """The dtype the model's weights and inputs are cast to."""
        return torch.bfloat16 if self.config.dtype == "bfloat16" else torch.float32

    def apply_dtype(self) -> None:
        """Cast the whole model to :attr:`target_dtype`.

        Call this again after loading a state dict, which may restore float32 tensors into
        a bfloat16 model.
        """
        self.to(dtype=self.target_dtype)

    def apply_freezing(self) -> None:
        """Apply the configured freezing to the VLM, the backbone and the soft prompts."""
        if self.config.freeze_vision_encoder:
            for param in self.vlm.vision_tower.parameters():
                param.requires_grad = False

        if self.config.freeze_language_encoder:
            language_model = self.vlm.language_model
            for param in language_model.encoder.parameters():
                param.requires_grad = False
            if hasattr(language_model, "shared"):
                for param in language_model.shared.parameters():
                    param.requires_grad = False

        if not self.config.train_policy_transformer:
            for name, param in self.transformer.named_parameters():
                if "soft_prompt" not in name:
                    param.requires_grad = False

        if not self.config.train_soft_prompts and hasattr(self.transformer, "soft_prompt_hub"):
            for param in self.transformer.soft_prompt_hub.parameters():
                param.requires_grad = False

    def set_action_dim(self, action_dim: int) -> None:
        """Re-fit the action space to a dataset whose action width differs.

        Only ``action_mode="auto"`` adapts: it re-slices which channels of the model's
        fixed-width action vector are supervised and emitted, which touches no weights. The
        fixed layouts cannot adapt, so a mismatch is reported instead -- their emitted width
        is part of the checkpoint's contract.

        Args:
            action_dim: Action width of the dataset.
        """
        if self.config.action_mode.lower() != "auto":
            if action_dim != self.dim_action:
                logger.warning(
                    "Dataset actions are %d-dimensional but action_mode=%r emits %d dimensions. "
                    "Use action_mode='auto', or remap the dataset into this action layout.",
                    action_dim,
                    self.config.action_mode,
                    self.dim_action,
                )
            return

        if action_dim == self.action_space.real_dim:
            return

        self.action_space = build_action_space("auto", real_dim=action_dim, max_dim=self.config.max_action_dim)
        self.dim_action = self.action_space.dim_action

    # ------------------------------------------------------------------ #
    # Delta indices                                                       #
    # ------------------------------------------------------------------ #
    @property
    def reward_delta_indices(self) -> None:
        """XVLA does not consume rewards.

        Returns:
            None.
        """
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        """Action offsets each training sample must provide.

        Returns:
            One offset per step of the predicted chunk.
        """
        return list(range(self.config.chunk_size))

    @property
    def observation_delta_indices(self) -> None:
        """XVLA conditions on the current observation only.

        Returns:
            None.
        """
        return None

    # ------------------------------------------------------------------ #
    # Encoding                                                            #
    # ------------------------------------------------------------------ #
    def encode_observation(
        self,
        input_ids: Tensor,
        images: Tensor,
        image_mask: Tensor,
    ) -> dict[str, Tensor]:
        """Encode the prompt and the cameras with Florence-2.

        Only the views the mask marks valid are run through the vision tower; masked slots
        keep zero features, so a checkpoint expecting more cameras than the dataset carries
        still sees a full-width, well-defined stream.

        Args:
            input_ids: Tokenized prompt of shape ``[B, L]``.
            images: Camera images of shape ``[B, V, C, H, W]``.
            image_mask: Per-view validity mask of shape ``[B, V]``.

        Returns:
            Dict with ``vlm_features`` (``[B, tokens + L, D]``, the primary view fused with
            the prompt) and ``aux_visual_inputs`` (``[B, (V - 1) * tokens, D]``).

        Raises:
            ValueError: If no view in the batch is valid.
        """
        batch_size, num_views = images.shape[:2]
        flat_mask = image_mask.reshape(-1).to(dtype=torch.bool)
        flat_images = images.flatten(0, 1)
        if not bool(flat_mask.any()):
            msg = "At least one image view must be valid per batch."
            raise ValueError(msg)

        valid_features = self.vlm.get_image_features(flat_images[flat_mask]).pooler_output
        tokens_per_view, hidden_dim = valid_features.shape[1:]

        image_features = valid_features.new_zeros((batch_size * num_views, tokens_per_view, hidden_dim))
        image_features[flat_mask] = valid_features
        image_features = image_features.view(batch_size, num_views, tokens_per_view, hidden_dim)

        # The primary view's tokens are prepended to the prompt and encoded jointly.
        prompt_embeds = self.vlm.get_input_embeddings()(input_ids)
        merged = torch.cat([image_features[:, 0], prompt_embeds], dim=1)
        attention_mask = torch.ones(merged.shape[:2], dtype=torch.long, device=merged.device)
        encoded = self.vlm.language_model.encoder(attention_mask=attention_mask, inputs_embeds=merged)[0]

        return {
            "vlm_features": encoded,
            "aux_visual_inputs": image_features[:, 1:].reshape(batch_size, -1, hidden_dim),
        }

    def _unpack(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, Tensor], Tensor, Tensor]:
        """Pull the model inputs out of a preprocessed batch.

        Args:
            batch: Preprocessed batch dict.

        Returns:
            Tuple of ``(input_ids, encoded observation, domain_id, proprio)``.
        """
        input_ids = batch[TOKENIZED_PROMPT]
        images = batch[IMAGES].to(dtype=self.target_dtype)
        image_mask = batch[IMAGE_MASKS]
        batch_size = input_ids.shape[0]
        device = images.device

        domain_id = batch.get(DOMAIN_ID)
        if domain_id is None:
            domain_id = torch.zeros(batch_size, dtype=torch.long, device=device)
        domain_id = domain_id.to(device=device, dtype=torch.long)

        if self.dim_proprio == 0:
            proprio = torch.zeros(batch_size, 0, device=device, dtype=self.target_dtype)
        else:
            proprio = batch[STATE].to(device=device, dtype=self.target_dtype)

        return input_ids, self.encode_observation(input_ids, images, image_mask), domain_id, proprio

    def prepare_targets(self, actions: Tensor) -> Tensor:
        """Shape ground-truth actions into the model's ``[B, chunk_size, dim_action]`` target.

        Args:
            actions: Ground-truth actions of shape ``[B, T, D]`` (or ``[B, D]`` for a single
                step), zero-padded or truncated to the model's chunk and action widths.

        Returns:
            Targets of shape ``[B, chunk_size, dim_action]``.
        """
        if actions.ndim == 2:  # noqa: PLR2004
            actions = actions.unsqueeze(1)
        actions = actions.to(dtype=self.target_dtype)

        chunk_size = self.config.chunk_size
        if actions.shape[1] > chunk_size:
            actions = actions[:, :chunk_size]
        elif actions.shape[1] < chunk_size:
            actions = F.pad(actions, (0, 0, 0, chunk_size - actions.shape[1]))

        if actions.shape[-1] > self.dim_action:
            actions = actions[..., : self.dim_action]
        elif actions.shape[-1] < self.dim_action:
            actions = F.pad(actions, (0, self.dim_action - actions.shape[-1]))

        return actions

    # ------------------------------------------------------------------ #
    # Training                                                            #
    # ------------------------------------------------------------------ #
    def forward(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, Tensor | float]] | Tensor:
        """Compute the training loss, or predict a chunk in eval mode.

        Args:
            batch: Preprocessed batch dict.

        Returns:
            ``(loss, loss_dict)`` while training, else the predicted action chunk.
        """
        if self.training:
            return self.compute_loss(batch)
        return self.predict_action_chunk(batch)

    def compute_loss(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, Tensor | float]]:
        """Compute the flow-matching reconstruction loss for one batch.

        Timesteps are drawn once per batch and spread evenly across ``[0, 1)`` (one shared
        offset plus a per-sample stride), which covers the noise schedule more evenly than
        independent uniform draws at small batch sizes.

        Args:
            batch: Preprocessed batch dict, which must carry ground-truth actions.

        Returns:
            Tuple of ``(loss, loss_dict)``; the dict holds ``"loss"`` plus one entry per
            component of the action space's loss.

        Raises:
            ValueError: If the batch carries no action targets.
        """
        if batch.get(ACTION) is None:
            msg = "Batch is missing the action targets required for training."
            raise ValueError(msg)

        input_ids, encoded, domain_id, proprio = self._unpack(batch)
        actions = self.prepare_targets(batch[ACTION])

        t = self._sample_timesteps(input_ids.shape[0], input_ids.device)
        noisy = torch.randn_like(actions) * t.view(-1, 1, 1) + actions * (1 - t).view(-1, 1, 1)
        proprio_masked, noisy_masked = self.action_space.preprocess(proprio, noisy)

        predicted = self.transformer(
            domain_id=domain_id,
            action_with_noise=noisy_masked,
            t=t,
            proprio=proprio_masked,
            **encoded,
        )

        losses = self.action_space.compute_loss(predicted, actions)
        total = torch.stack(list(losses.values())).sum()

        # Detached tensors, not `.item()` floats: see Model.compute_loss docstring.
        loss_dict: dict[str, Tensor | float] = {name: value.detach() for name, value in losses.items()}
        loss_dict["loss"] = total.detach()
        return total, loss_dict

    def _sample_timesteps(self, batch_size: int, device: torch.device) -> Tensor:
        """Draw one stratified set of flow-matching timesteps for a batch.

        A single random offset is shared across the batch and each sample is pushed a
        further ``1 / batch_size`` along, which covers ``[0, 1)`` more evenly than
        independent uniform draws do at small batch sizes.

        Args:
            batch_size: Number of samples in the batch.
            device: Device the model runs on.

        Returns:
            Timesteps of shape ``[B]`` in ``[0, 1)``.
        """
        offset = torch.rand(1, device=device, dtype=self.target_dtype)
        stride = torch.arange(batch_size, device=device, dtype=self.target_dtype) / batch_size
        return (offset + stride) % (1 - 1e-5)

    @torch.no_grad()
    def compute_val_loss(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, Tensor | float]]:
        """Compute the action-prediction MSE for one batch.

        Runs the full denoising loop and compares the emitted chunk with the ground truth,
        which is deterministic and measures prediction quality directly -- unlike the
        stochastic training loss, whose scale depends on the sampled timesteps.

        Args:
            batch: Preprocessed batch dict, which must carry ground-truth actions.

        Returns:
            Tuple of ``(mse, {"loss": mse})``.

        Raises:
            ValueError: If the batch carries no action targets.
        """
        if batch.get(ACTION) is None:
            msg = "Batch is missing the action targets required for validation."
            raise ValueError(msg)

        target = batch[ACTION]
        if target.ndim == 2:  # noqa: PLR2004
            target = target.unsqueeze(1)
        predicted = self.predict_action_chunk(batch)

        # Compare on the overlap: the action space may emit a different width or horizon
        # than the dataset provides.
        steps = min(predicted.shape[1], target.shape[1])
        width = min(predicted.shape[-1], target.shape[-1])
        loss = F.mse_loss(
            predicted[:, :steps, :width].to(dtype=torch.float32),
            target[:, :steps, :width].to(dtype=torch.float32),
        )
        return loss, {"loss": loss}

    # ------------------------------------------------------------------ #
    # Inference                                                           #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Any]) -> Tensor:
        """Denoise one action chunk from a preprocessed batch.

        Starting from pure noise, the transformer predicts the clean chunk, the estimate is
        re-noised toward the next (lower) timestep, and the loop repeats
        ``num_denoising_steps`` times.

        Args:
            batch: Preprocessed batch dict.

        Returns:
            Actions of shape ``[B, chunk_size, D]``, where ``D`` is the action space's
            emitted width (the dataset's action width under ``action_mode="auto"``).
        """
        input_ids, encoded, domain_id, proprio = self._unpack(batch)

        batch_size = input_ids.shape[0]
        device = proprio.device
        noise = torch.randn(
            batch_size,
            self.config.chunk_size,
            self.dim_action,
            device=device,
            dtype=self.target_dtype,
        )
        actions = torch.zeros_like(noise)

        steps = max(1, self.config.num_denoising_steps)
        for step in range(steps, 0, -1):
            t = torch.full((batch_size,), step / steps, device=device, dtype=self.target_dtype)
            noisy = noise * t.view(-1, 1, 1) + actions * (1 - t).view(-1, 1, 1)
            proprio_masked, noisy_masked = self.action_space.preprocess(proprio, noisy)
            actions = self.transformer(
                domain_id=domain_id,
                action_with_noise=noisy_masked,
                proprio=proprio_masked,
                t=t,
                **encoded,
            )

        return self.action_space.postprocess(actions)


__all__ = ["DOMAIN_ID", "XVLAModel"]
