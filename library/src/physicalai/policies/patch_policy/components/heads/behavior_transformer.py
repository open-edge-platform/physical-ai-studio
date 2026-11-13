# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2026 Gaoyue Zhou
# Authors: Gaoyue Zhou, Zichen Jeff Cui
# SPDX-License-Identifier: MIT

"""Patch Policy Behavior Transformer action head."""

from __future__ import annotations

import math
from typing import override

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from physicalai.policies.components.action_heads import ActionHead, Context
from physicalai.policies.patch_policy.components.gpt import PatchGPT

_DEAD_CODE_THRESHOLD = 2


class _MLP(nn.Sequential):
    """Upstream Behavior Transformer prediction MLP."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(1024, output_dim),
            nn.Dropout(0.0),
        )


class _ActionMLP(nn.Module):
    """MLP used by the upstream action VQ-VAE encoder and decoder."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.fc = nn.Linear(128, output_dim)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, values: Tensor) -> Tensor:
        """Project flattened action windows.

        Returns:
            Projected values with the configured output width.
        """
        return self.fc(self.encoder(values))


class ResidualActionQuantizer(nn.Module):
    """Residual action VQ-VAE with upstream EMA and straight-through training.

    Fit a standalone instance or a deep copy of the head's tokenizer with
    ``requires_grad_(True).train()``. Optimize the loss from ``compute_loss``;
    codebooks update through EMA, not the optimizer. The caller must set
    ``fitted.fill_(True)`` after fitting and load the state into the head.
    """

    def __init__(
        self,
        action_dim: int,
        chunk_size: int,
        latent_dim: int,
        num_embeddings: int,
        groups: int,
        *,
        action_scale: float,
    ) -> None:
        """Initialize action projections and the residual codebook.

        Raises:
            ValueError: If dimensions or action scale are not positive and finite.
        """
        super().__init__()
        if min(action_dim, chunk_size, latent_dim, num_embeddings, groups) <= 0:
            message = "Quantizer dimensions and groups must be positive."
            raise ValueError(message)
        if not math.isfinite(action_scale) or action_scale <= 0:
            message = "action_scale must be finite and positive."
            raise ValueError(message)
        flat_dim = action_dim * chunk_size
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.groups = groups
        self.action_scale = action_scale
        codebook = torch.cat([
            nn.init.kaiming_uniform_(torch.empty(1, num_embeddings, latent_dim)) for _ in range(groups)
        ])
        self.register_buffer("codebook", codebook)
        self.register_buffer("cluster_size", torch.zeros(groups, num_embeddings))
        self.register_buffer("embed_avg", codebook.clone())
        self.register_buffer("fitted", torch.zeros((), dtype=torch.bool))
        self.encoder = _ActionMLP(flat_dim, latent_dim)
        self.decoder = _ActionMLP(latent_dim, flat_dim)

    @torch.no_grad()
    def encode(self, actions: Tensor) -> Tensor:
        """Return residual code indices without mutating tokenizer state."""
        latent = self.encoder((actions / self.action_scale).flatten(-2))
        residual = latent
        codes = []
        for codebook in self.codebook:
            distances = torch.cdist(residual[:, None], codebook[None]).squeeze(1)
            code = distances.argmin(-1)
            residual = torch.sub(residual, codebook[code])
            codes.append(code)
        return torch.stack(codes, dim=-1)

    @torch.no_grad()
    def decode_codes(self, codes: Tensor) -> Tensor:
        """Decode residual code indices into action centers.

        Returns:
            Decoded actions shaped ``[B, chunk_size, action_dim]``.
        """
        groups = torch.arange(self.groups, device=codes.device)
        latent = self.codebook[groups[:, None], codes.transpose(0, 1)].sum(0)
        decoded = self.decoder(latent) * self.action_scale
        return decoded.reshape(-1, self.chunk_size, self.action_dim)

    @torch.no_grad()
    def _update_codebook(self, group: int, residual: Tensor, codes: Tensor) -> None:
        counts = torch.bincount(codes, minlength=self.codebook.shape[1]).to(self.cluster_size)
        sums = torch.zeros_like(self.embed_avg[group])
        sums.index_add_(0, codes, residual.to(sums))
        self.cluster_size[group].lerp_(counts, 0.2)
        self.embed_avg[group].lerp_(sums, 0.2)
        total = self.cluster_size[group].sum()
        smoothed = (self.cluster_size[group] + 1e-5) / (total + counts.numel() * 1e-5) * total
        self.codebook[group].copy_(self.embed_avg[group] / smoothed[:, None])

        expired = self.cluster_size[group] < _DEAD_CODE_THRESHOLD
        num_expired = int(expired.sum())
        if num_expired:
            if residual.shape[0] >= num_expired:
                indices = torch.randperm(residual.shape[0], device=residual.device)[:num_expired]
            else:
                indices = torch.randint(residual.shape[0], (num_expired,), device=residual.device)
            sampled = residual[indices].to(self.codebook)
            self.codebook[group, expired] = sampled
            self.cluster_size[group, expired] = _DEAD_CODE_THRESHOLD
            self.embed_avg[group, expired] = sampled * _DEAD_CODE_THRESHOLD

    def compute_loss(self, actions: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute scaled-action L1 reconstruction plus five times commitment.

        Args:
            actions: Action windows shaped ``[B, chunk_size, action_dim]``.

        Returns:
            Scalar loss and detached upstream metrics: ``loss``, ``encoder_loss``
            (scaled-action L1), ``vq_loss_state`` (summed stage commitment), and
            ``vqvae_recon_loss`` (scaled-action MSE). Training a trainable instance
            updates EMA buffers with decay 0.8, epsilon 1e-5, and dead threshold 2.
            Eval uses reconstruction only, without straight-through encoder gradients.
            This method never marks the tokenizer fitted.

        Raises:
            ValueError: If actions are empty or have an incompatible shape.
        """
        window_shape = (self.chunk_size, self.action_dim)
        if actions.ndim != len(window_shape) + 1 or actions.shape[0] == 0 or actions.shape[1:] != window_shape:
            message = f"Expected nonempty action windows [B, {self.chunk_size}, {self.action_dim}]."
            raise ValueError(message)
        target = (actions / self.action_scale).flatten(-2)
        latent = self.encoder(target)
        update = self.training and any(parameter.requires_grad for parameter in self.parameters())
        # Upstream computes codebook distances and EMA statistics in float32.
        with torch.autocast(device_type=latent.device.type, enabled=False):
            residual = latent.float()
            quantized = torch.zeros_like(residual)
            commitment = residual.new_zeros(())
            for group, codebook in enumerate(self.codebook):
                with torch.no_grad():
                    distances = (
                        residual.square().sum(-1, keepdim=True)
                        + codebook.square().sum(-1)[None]
                        - 2 * residual @ codebook.T
                    ).sqrt()
                    codes = distances.argmin(-1)
                # Gather before EMA so targets and reconstruction use the old codes.
                if self.training:
                    selected = F.one_hot(codes, codebook.shape[0]).to(residual) @ codebook
                    commitment = torch.add(commitment, F.mse_loss(residual, selected))
                    stage = residual + (selected - residual).detach()
                else:
                    stage = codebook[codes]
                if update:
                    self._update_codebook(group, residual.detach(), codes)
                residual = torch.sub(residual, stage.detach())
                quantized = torch.add(quantized, stage)
        decoded = self.decoder(quantized)
        reconstruction = F.l1_loss(decoded, target)
        loss = reconstruction + 5 * commitment
        return loss, {
            "loss": loss.detach(),
            "encoder_loss": reconstruction.detach(),
            "vq_loss_state": commitment.detach(),
            "vqvae_recon_loss": F.mse_loss(decoded, target).detach(),
        }


class BehaviourTransformerHead(ActionHead):
    """Upstream-compatible patch-aware Behavior Transformer head."""

    def __init__(  # noqa: PLR0913
        self,
        context_dim: int,
        action_dim: int,
        chunk_size: int,
        *,
        n_obs_steps: int,
        n_patches: int,
        hidden_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        dropout: float = 0.0,
        latent_dim: int = 512,
        num_embeddings: int = 16,
        groups: int = 2,
        action_scale: float = 500.0,
        offset_loss_weight: float = 10.0,
        secondary_code_weight: float = 0.5,
        focal_loss_gamma: float = 2.0,
        use_random_noise: bool = True,
    ) -> None:
        """Initialize the transformer, action predictors, and quantizer."""
        super().__init__(chunk_size, action_dim)
        self.n_obs_steps = n_obs_steps
        self.n_patches = n_patches
        self.groups = groups
        self.num_embeddings = num_embeddings
        self.offset_loss_weight = offset_loss_weight
        self.secondary_code_weight = secondary_code_weight
        self.focal_loss_gamma = focal_loss_gamma
        self.use_random_noise = use_random_noise
        self.trunk = PatchGPT(
            input_dim=context_dim,
            output_dim=256,
            n_patches=n_patches,
            n_obs_steps=n_obs_steps + chunk_size,
            n_layer=num_layers,
            n_head=num_heads,
            n_embd=hidden_dim,
            dropout=dropout,
        )
        self.code_head = _MLP(256, groups * num_embeddings)
        self.offset_head = _MLP(256, groups * num_embeddings * chunk_size * action_dim)
        self.quantizer = (
            ResidualActionQuantizer(
                action_dim,
                chunk_size,
                latent_dim,
                num_embeddings,
                groups,
                action_scale=action_scale,
            )
            .requires_grad_(requires_grad=False)
            .eval()
        )

    @override
    def train(self, mode: bool = True) -> BehaviourTransformerHead:
        """Set head mode while keeping the registered tokenizer frozen and in eval.

        Returns:
            This head with the requested training mode.
        """
        super().train(mode)
        self.quantizer.requires_grad_(requires_grad=False).eval()
        return self

    def _embeddings(self, context: Context) -> Tensor:
        tokens = context["tokens"]
        expected = self.n_obs_steps * self.n_patches
        if tokens.shape[1] != expected:
            message = f"Expected {expected} context tokens, got {tokens.shape[1]}."
            raise ValueError(message)
        return self.trunk(tokens.reshape(tokens.shape[0], self.n_obs_steps, self.n_patches, tokens.shape[-1]))

    def _predictions(self, context: Context) -> tuple[Tensor, Tensor]:
        embeddings = self._embeddings(context)
        logits = self.code_head(embeddings).reshape(
            embeddings.shape[0],
            self.n_obs_steps,
            self.groups,
            self.num_embeddings,
        )
        offsets = self.offset_head(embeddings).reshape(
            embeddings.shape[0],
            self.n_obs_steps,
            self.groups,
            self.num_embeddings,
            self.chunk_size,
            self.action_dim,
        )
        return logits, offsets

    def _actions(self, logits: Tensor, offsets: Tensor, codes: Tensor | None = None) -> Tensor:
        if codes is None:
            probabilities = logits.softmax(-1)
            codes = torch.multinomial(probabilities.flatten(0, -2), 1).reshape(*logits.shape[:-1])
        flat_codes = codes.flatten(0, 1)
        centers = self.quantizer.decode_codes(flat_codes).reshape(
            logits.shape[0],
            logits.shape[1],
            self.chunk_size,
            self.action_dim,
        )
        batch = torch.arange(logits.shape[0], device=logits.device)[:, None, None]
        time = torch.arange(logits.shape[1], device=logits.device)[None, :, None]
        groups = torch.arange(self.groups, device=logits.device)[None, None, :]
        selected_offsets = offsets[batch, time, groups, codes].sum(2)
        return centers.detach() + selected_offsets

    def sample(self, context: Context, *, noise: Tensor | None = None, num_steps: int | None = None) -> Tensor:
        """Sample the final chunk or select each group's highest-logit code with argmax.

        When ``use_random_noise=False``, code selection is deterministic and uses
        the predicted most likely code rather than multinomial sampling. Decoded
        codebook centers and observation-dependent offsets still form the actions.

        Returns:
            Predicted actions shaped ``[B, chunk_size, action_dim]``.
        """
        del noise, num_steps
        logits, offsets = self._predictions(context)
        codes = None if self.use_random_noise else logits.argmax(dim=-1)
        return self._actions(logits, offsets, codes)[:, -1]

    def compute_loss(self, actions: Tensor, context: Context) -> Tensor:
        """Compute the upstream focal-code and continuous-offset loss.

        All overlapping windows, including padded action steps, intentionally
        contribute for upstream parity. Masking belongs to the caller, but a
        trajectory mask must not be applied to this overlapping-window layout.

        Returns:
            Loss shaped ``[B, n_obs_steps, chunk_size, action_dim]``. Each
            observation's weighted code loss is broadcast across its window;
            ``mean()`` recovers the upstream unmasked scalar objective.

        Raises:
            ValueError: If actions do not cover every overlapping target window.
            RuntimeError: If head training starts before the tokenizer is fitted.
        """
        expected = self.n_obs_steps + self.chunk_size - 1
        if actions.shape[1] != expected:
            message = f"Expected {expected} action steps, got {actions.shape[1]}."
            raise ValueError(message)
        if self.training and not self.quantizer.fitted:
            message = "Fit the action quantizer and mark its fitted buffer before training the Behavior Transformer."
            raise RuntimeError(message)
        windows = actions.unfold(1, self.chunk_size, 1).transpose(-1, -2)
        logits, offsets = self._predictions(context)
        target_codes = self.quantizer.encode(windows.flatten(0, 1)).reshape(
            actions.shape[0],
            self.n_obs_steps,
            self.groups,
        )
        predicted = self._actions(logits, offsets)

        target_log_probability = logits.log_softmax(-1).gather(-1, target_codes[..., None]).squeeze(-1)
        focal = -((1 - target_log_probability.exp()) ** self.focal_loss_gamma) * target_log_probability
        code_loss = 5 * focal[..., 0] + self.secondary_code_weight * focal[..., 1:].sum(-1)
        offset_loss = F.l1_loss(predicted, windows, reduction="none")
        return code_loss[..., None, None] + self.offset_loss_weight * offset_loss
