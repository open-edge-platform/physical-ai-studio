# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2026 Gaoyue Zhou
# Authors: Gaoyue Zhou, Zichen Jeff Cui
# SPDX-License-Identifier: MIT

"""VQ-BeT action head for Patch Policy.

The trunk encodes patch tokens into one embedding per timestep; two MLP heads map that
embedding to residual-VQ code logits and per-code action offsets. Predicted actions are
the decoded code centers plus the sampled offsets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from .action_head import BaseActionHead
from .gpt import PatchGPT

if TYPE_CHECKING:
    from .config import PatchPolicyConfig

HEAD_HIDDEN_DIM = 1024


def _mlp(in_dim: int, out_dim: int, hidden_dim: int = HEAD_HIDDEN_DIM, n_hidden: int = 2) -> nn.Sequential:
    """Build a ReLU MLP.

    Returns:
        Sequential mapping ``in_dim`` to ``out_dim``.
    """
    layers: list[nn.Module] = []
    dim = in_dim
    for _ in range(n_hidden):
        layers += [nn.Linear(dim, hidden_dim), nn.ReLU()]
        dim = hidden_dim
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


class ResidualActionQuantizer(nn.Module):
    """Residual VQ over action chunks.

    The encoder/decoder and codebook layout are final; the quantization search and the
    fitting loop are placeholders pending the port of the reference residual VQ.
    """

    def __init__(
        self,
        act_dim: int,
        chunk_size: int,
        latent_dim: int,
        n_embed: int,
        groups: int,
        lr: float,
    ) -> None:
        """Initialize the quantizer."""
        super().__init__()
        self.act_dim = act_dim
        self.chunk_size = chunk_size
        self.latent_dim = latent_dim
        self.n_embed = n_embed
        self.groups = groups

        flat_dim = act_dim * chunk_size
        self.encoder = _mlp(flat_dim, latent_dim, hidden_dim=128)
        self.decoder = _mlp(latent_dim, flat_dim, hidden_dim=128)
        self.codebook = nn.Parameter(torch.randn(groups, n_embed, latent_dim) * 0.02)

        self._optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        self._collected: list[Tensor] = []

    def encode_to_codes(self, actions: Tensor) -> Tensor:
        """Quantize ``[..., W, A]`` action chunks into ``[..., G]`` code indices.

        Returns:
            Long tensor of code indices, one per residual group.
        """
        return torch.zeros(*actions.shape[:-2], self.groups, dtype=torch.long, device=actions.device)

    def codes_to_latent(self, codes: Tensor) -> Tensor:
        """Look up ``[..., G]`` code indices and sum the residual groups.

        Returns:
            ``[..., latent_dim]`` latent vectors.
        """
        return torch.zeros(*codes.shape[:-1], self.latent_dim, device=codes.device, dtype=self.codebook.dtype)

    def latent_to_action(self, latent: Tensor) -> Tensor:
        """Decode ``[..., latent_dim]`` latents into ``[..., W, A]`` action chunks.

        Returns:
            Decoded action chunks.
        """
        decoded = self.decoder(latent)
        return decoded.reshape(*latent.shape[:-1], self.chunk_size, self.act_dim)


class VQBeTActionHead(BaseActionHead):
    """VQ-BeT action head: patch-causal trunk plus code-classification and offset heads."""

    def __init__(
        self,
        config: PatchPolicyConfig,
        token_dim: int,
        act_dim: int,
        n_patches: int,
    ) -> None:
        """Initialize the head from the Patch Policy config."""
        super().__init__(token_dim=token_dim, act_dim=act_dim, chunk_size=config.chunk_size)
        self.groups = config.vqvae_groups
        self.n_embed = config.vqvae_n_embed
        self.offset_loss_multiplier = config.offset_loss_multiplier
        self.secondary_code_multiplier = config.secondary_code_multiplier
        self.focal_loss_gamma = config.focal_loss_gamma

        self.trunk = PatchGPT(
            input_dim=token_dim,
            output_dim=config.n_embd,
            n_patches=n_patches,
            n_obs_steps=config.n_obs_steps,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_embd=config.n_embd,
            dropout=config.dropout,
        )
        self.code_head = _mlp(config.n_embd, self.groups * self.n_embed)
        self.offset_head = _mlp(
            config.n_embd,
            self.groups * self.n_embed * self.chunk_size * act_dim,
        )
        self.quantizer = ResidualActionQuantizer(
            act_dim=act_dim,
            chunk_size=self.chunk_size,
            latent_dim=config.vqvae_latent_dim,
            n_embed=self.n_embed,
            groups=self.groups,
            lr=config.vqvae_lr,
        )

    def _forward_heads(self, embeddings: Tensor) -> tuple[Tensor, Tensor]:
        """Map ``[B, T, E]`` trunk embeddings to code logits and per-code offsets.

        Returns:
            ``[B, T, G, C]`` logits and ``[B, T, G, C, W, A]`` offsets.
        """
        logits = self.code_head(embeddings).reshape(embeddings.shape[0], embeddings.shape[1], self.groups, self.n_embed)
        offsets_shape = (
            embeddings.shape[0],
            embeddings.shape[1],
            self.groups,
            self.n_embed,
            self.chunk_size,
            self.act_dim,
        )
        offsets = self.offset_head(embeddings).reshape(offsets_shape)
        return logits, offsets

    def _sample_action(self, logits: Tensor, offsets: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Generate a random action chunk with the right shape.

        Returns:
            ``(predicted_action, decoded_action, sampled_codes)`` shaped
            ``[B, T, W, A]``, ``[B, T, W, A]`` and ``[B, T, G]``.
        """
        b, t = logits.shape[:2]
        predicted_action = torch.randn(b, t, self.chunk_size, self.act_dim, device=logits.device, dtype=logits.dtype)
        decoded_action = predicted_action.clone()
        sampled_codes = torch.zeros(b, t, self.groups, dtype=torch.long, device=logits.device)
        return predicted_action, decoded_action, sampled_codes

    def predict(self, tokens: Tensor) -> Tensor:
        """Predict a ``[B, T, W, A]`` action chunk from ``[B, T, P, E]`` tokens.

        Returns:
            Predicted action chunks.
        """
        logits, offsets = self._forward_heads(self.trunk(tokens))
        predicted_action, _, _ = self._sample_action(logits, offsets)
        return predicted_action

    def compute_loss(self, tokens: Tensor, actions: Tensor) -> tuple[Tensor, dict[str, Tensor | float]]:
        """Compute the VQ-BeT training loss.

        Returns:
            Tuple of (loss, loss dict).
        """
        logits, offsets = self._forward_heads(self.trunk(tokens))
        predicted_action, decoded_action, sampled_codes = self._sample_action(logits, offsets)
        target_codes = self.quantizer.encode_to_codes(actions)

        # Placeholder: focal classification and offset losses are pending the quantizer port.
        zero = predicted_action.sum() * 0.0
        classification_loss = zero
        offset_loss = zero
        loss = classification_loss + self.offset_loss_multiplier * offset_loss

        eq_mask = target_codes == sampled_codes
        loss_dict: dict[str, Tensor | float] = {
            "loss": loss.detach(),
            "classification_loss": classification_loss.detach(),
            "offset_loss": offset_loss.detach(),
            "equal_total_code_rate": (eq_mask.sum(-1) == self.groups).float().mean(),
            "equal_single_code_rate": eq_mask[..., 0].float().mean(),
            "action_diff": (actions - predicted_action).pow(2).mean().detach(),
            "action_diff_max": (actions - predicted_action).abs().max().detach(),
            "action_diff_decoded": (actions - decoded_action).abs().mean().detach(),
        }
        return loss, loss_dict
