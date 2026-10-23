# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Patch Policy Behavior Transformer action head."""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from physicalai.policies.components.action_heads import ActionHead, Context
from physicalai.policies.patch_policy.gpt import PatchGPT


def _mlp(input_dim: int, output_dim: int, hidden_dim: int) -> nn.Sequential:
    """Build a two-layer ReLU MLP.

    Args:
        input_dim: Input feature width.
        output_dim: Output feature width.
        hidden_dim: Hidden-layer width.

    Returns:
        The assembled MLP.
    """
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))


class ResidualActionQuantizer(nn.Module):
    """Compact residual vector quantizer for flattened action chunks.

    Args:
        action_dim: Width of each action timestep.
        chunk_size: Number of action timesteps in each chunk.
        latent_dim: Width of the encoded action latent.
        num_embeddings: Number of entries in each codebook group.
        groups: Number of residual quantization groups.
    """

    def __init__(self, action_dim: int, chunk_size: int, latent_dim: int, num_embeddings: int, groups: int) -> None:
        super().__init__()
        flat_dim = action_dim * chunk_size
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.groups = groups
        self.encoder = _mlp(flat_dim, latent_dim, 128)
        self.decoder = _mlp(latent_dim, flat_dim, 128)
        self.codebook = nn.Parameter(torch.randn(groups, num_embeddings, latent_dim) * 0.02)

    def encode(self, actions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Quantize flattened action chunks.

        Args:
            actions: Action chunks with shape ``[B, T, A]``.

        Returns:
            The straight-through latent, residual code indices, and commitment loss.
        """
        latent = self.encoder(actions.flatten(-2))
        residual = latent
        quantized = torch.zeros_like(latent)
        codes = []
        for group in range(self.groups):
            distances = (residual[:, None] - self.codebook[group][None]).square().sum(-1)
            code = distances.argmin(-1)
            selected = self.codebook[group][code]
            quantized += selected
            residual -= selected.detach()
            codes.append(code)
        commitment = (latent - quantized.detach()).square().mean(-1)
        quantized = latent + (quantized - latent).detach()
        return quantized, torch.stack(codes, dim=-1), commitment

    def decode_codes(self, codes: Tensor) -> Tensor:
        """Decode residual code indices to action chunks.

        Args:
            codes: Code index for every residual group and batch item.

        Returns:
            Reconstructed action chunks.
        """
        latent = torch.stack(
            [self.codebook[group][codes[:, group]] for group in range(self.groups)],
        ).sum(0)
        return self.decode(latent)

    def decode(self, latent: Tensor) -> Tensor:
        """Decode continuous latents to action chunks.

        Args:
            latent: Quantized action latent.

        Returns:
            Reconstructed action chunks with shape ``[B, T, A]``.
        """
        return self.decoder(latent).reshape(-1, self.chunk_size, self.action_dim)


class PatchBehaviorTransformerHead(ActionHead):
    """Predict residual-VQ action centers and continuous offsets from patch tokens.

    Args:
        context_dim: Width of each visual context token.
        action_dim: Width of each action timestep.
        chunk_size: Number of action timesteps predicted per sample.
        n_obs_steps: Number of observation timesteps in the context.
        n_patches: Number of visual patch tokens per observation timestep.
        hidden_dim: Transformer and prediction-head hidden width.
        num_layers: Number of transformer blocks.
        num_heads: Number of self-attention heads.
        dropout: Dropout probability used by the transformer.
        latent_dim: Width of the VQ action latent.
        num_embeddings: Number of entries in each codebook group.
        groups: Number of residual vector-quantization groups.
        reconstruction_weight: Weight applied to the VQ reconstruction loss.
        commitment_weight: Weight applied to the VQ commitment loss.
        offset_loss_weight: Weight applied to the continuous offset loss.
        secondary_code_weight: Relative loss weight for code groups after the first.
        focal_loss_gamma: Focusing exponent used by the code-prediction loss.
    """

    def __init__(
        self,
        context_dim: int,
        action_dim: int,
        chunk_size: int,
        *,
        n_obs_steps: int,
        n_patches: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        latent_dim: int = 128,
        num_embeddings: int = 16,
        groups: int = 2,
        reconstruction_weight: float = 1.0,
        commitment_weight: float = 0.25,
        offset_loss_weight: float = 10.0,
        secondary_code_weight: float = 0.5,
        focal_loss_gamma: float = 2.0,
    ) -> None:
        super().__init__(chunk_size, action_dim)
        self.n_obs_steps = n_obs_steps
        self.n_patches = n_patches
        self.groups = groups
        self.num_embeddings = num_embeddings
        self.reconstruction_weight = reconstruction_weight
        self.commitment_weight = commitment_weight
        self.offset_loss_weight = offset_loss_weight
        self.secondary_code_weight = secondary_code_weight
        self.focal_loss_gamma = focal_loss_gamma
        self.trunk = PatchGPT(
            input_dim=context_dim,
            output_dim=hidden_dim,
            n_patches=n_patches,
            n_obs_steps=n_obs_steps,
            n_layer=num_layers,
            n_head=num_heads,
            n_embd=hidden_dim,
            dropout=dropout,
        )
        self.code_head = _mlp(hidden_dim, groups * num_embeddings, hidden_dim)
        self.offset_head = _mlp(hidden_dim, groups * num_embeddings * chunk_size * action_dim, hidden_dim)
        self.quantizer = ResidualActionQuantizer(action_dim, chunk_size, latent_dim, num_embeddings, groups)

    def _embedding(self, context: Context) -> Tensor:
        """Encode visual context into the final observation embedding.

        Args:
            context: Context containing flattened visual tokens.

        Returns:
            The final timestep embedding.

        Raises:
            ValueError: If the context contains an unexpected number of tokens.
        """
        tokens = context["tokens"]
        expected = self.n_obs_steps * self.n_patches
        if tokens.shape[1] != expected:
            raise ValueError(f"Expected {expected} context tokens, got {tokens.shape[1]}.")
        return self.trunk(tokens.reshape(tokens.shape[0], self.n_obs_steps, self.n_patches, tokens.shape[-1]))[:, -1]

    def _predictions(self, context: Context) -> tuple[Tensor, Tensor]:
        """Predict code logits and code-conditioned continuous offsets.

        Args:
            context: Context containing flattened visual tokens.

        Returns:
            Code logits and continuous offsets.
        """
        embedding = self._embedding(context)
        logits = self.code_head(embedding).reshape(-1, self.groups, self.num_embeddings)
        offsets = self.offset_head(embedding).reshape(
            -1,
            self.groups,
            self.num_embeddings,
            self.chunk_size,
            self.action_dim,
        )
        return logits, offsets

    def _actions(self, logits: Tensor, offsets: Tensor, codes: Tensor | None = None) -> Tensor:
        """Combine quantized action centers with predicted offsets.

        Args:
            logits: Predicted code logits for each residual group.
            offsets: Predicted action offsets for every possible code.
            codes: Optional code indices to use instead of the most likely codes.

        Returns:
            Predicted action chunks.
        """
        if codes is None:
            codes = logits.argmax(-1)
        centers = self.quantizer.decode_codes(codes)
        batch = torch.arange(logits.shape[0], device=logits.device)[:, None]
        groups = torch.arange(self.groups, device=logits.device)[None]
        selected_offsets = offsets[batch, groups, codes].sum(1)
        return centers + selected_offsets

    def sample(self, context: Context, *, noise: Tensor | None = None, num_steps: int | None = None) -> Tensor:
        """Return the most likely action chunk.

        Args:
            context: Context containing flattened visual tokens.
            noise: Ignored iterative-head compatibility argument.
            num_steps: Ignored iterative-head compatibility argument.

        Returns:
            The most likely action chunk.
        """
        del noise, num_steps
        logits, offsets = self._predictions(context)
        return self._actions(logits, offsets)

    def compute_loss(self, actions: Tensor, context: Context) -> Tensor:
        """Compute focal-code, offset, reconstruction, and commitment losses.

        Args:
            actions: Ground-truth action chunks.
            context: Context containing flattened visual tokens.

        Returns:
            Unreduced per-action-element losses.
        """
        logits, offsets = self._predictions(context)
        latent, target_codes, commitment = self.quantizer.encode(actions)
        reconstructed = self.quantizer.decode(latent)
        predicted = self._actions(logits, offsets, target_codes)

        probabilities = logits.softmax(-1)
        target_probability = probabilities.gather(-1, target_codes[..., None]).squeeze(-1).clamp_min(1e-8)
        focal = -((1 - target_probability) ** self.focal_loss_gamma) * target_probability.log()
        weights = focal.new_ones(self.groups)
        if self.groups > 1:
            weights[1:] = self.secondary_code_weight
        code_loss = (focal * weights).mean(-1)
        reconstruction = F.mse_loss(reconstructed, actions, reduction="none").mean((-2, -1))
        global_loss = code_loss + self.reconstruction_weight * reconstruction + self.commitment_weight * commitment
        offset = F.l1_loss(predicted, actions, reduction="none") * self.offset_loss_weight
        return offset + global_loss[:, None, None]
