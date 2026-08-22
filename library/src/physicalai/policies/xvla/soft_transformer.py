# Copyright 2025 2toINF (https://github.com/2toINF) and The HuggingFace Inc. team.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""The soft-prompted, domain-aware transformer head of XVLA.

This is the part of XVLA that is trained from scratch. It takes the frozen-or-finetuned
Florence-2 embeddings, concatenates them with the noised action chunk, and denoises the
chunk with a plain pre-LayerNorm transformer stack.

Two mechanisms make one stack serve many embodiments at once:

- **Domain-aware projections** (:class:`DomainAwareLinear`) keep a separate weight matrix
  per domain in an ``nn.Embedding``, so the action encoder/decoder can speak a different
  action layout per robot while sharing the backbone.
- **Soft prompts** are per-domain learned tokens appended to the sequence, giving each
  domain a handful of dedicated "registers" without touching the backbone weights.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 100) -> torch.Tensor:
    """Build sinusoidal embeddings for (possibly fractional) flow-matching timesteps.

    Args:
        t: Timesteps of shape ``[B]``.
        dim: Width of the embedding.
        max_period: Controls the lowest sinusoid frequency.

    Returns:
        Embeddings of shape ``[B, dim]``.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=t.dtype, device=t.device) / half)
    args = t[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def basic_init(module: nn.Module) -> None:
    """Xavier-initialize linear weights and zero their biases.

    Args:
        module: Module to initialize; anything but ``nn.Linear`` is left alone.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


class Mlp(nn.Module):
    """The feed-forward block of a transformer layer.

    Args:
        in_features: Input width.
        hidden_features: Inner width. Defaults to ``in_features``.
        out_features: Output width. Defaults to ``in_features``.
        drop: Dropout probability applied after each linear layer.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        drop: float = 0.0,
    ) -> None:
        """Build the two-layer feed-forward network."""
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate="tanh")
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward network.

        Args:
            x: Input of shape ``[B, T, C]``.

        Returns:
            Output of shape ``[B, T, out_features]``.
        """
        x = self.drop1(self.act(self.fc1(x)))
        return self.drop2(self.fc2(x))


class Attention(nn.Module):
    """Bidirectional multi-head self-attention over the full sequence.

    Args:
        dim: Model width.
        num_heads: Number of attention heads; must divide ``dim``.
        qkv_bias: Whether the fused QKV projection carries a bias.
        attn_drop: Attention dropout probability.
        proj_drop: Output-projection dropout probability.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        *,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        """Build the fused QKV projection and the output projection.

        Raises:
            ValueError: If ``dim`` is not divisible by ``num_heads``.
        """
        super().__init__()
        if dim % num_heads != 0:
            msg = f"dim ({dim}) must be divisible by num_heads ({num_heads})"
            raise ValueError(msg)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_drop = attn_drop

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Attend over the sequence.

        Args:
            x: Input of shape ``[B, T, C]``.

        Returns:
            Output of shape ``[B, T, C]``.
        """
        batch_size, seq_len, channels = x.shape
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)

        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.attn_drop if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch_size, seq_len, channels)
        return self.proj_drop(self.proj(attended))


class DomainAwareLinear(nn.Module):
    """A linear layer whose weights are looked up per domain.

    Each domain owns a full weight matrix and bias, stored flattened in an ``nn.Embedding``,
    so a batch mixing several embodiments applies a different projection per sample.

    Args:
        input_size: Input width.
        output_size: Output width.
        num_domains: Number of domains the layer can serve.
    """

    def __init__(self, input_size: int, output_size: int, num_domains: int = 20) -> None:
        """Build the per-domain weight and bias tables."""
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc = nn.Embedding(num_domains, output_size * input_size)
        self.bias = nn.Embedding(num_domains, output_size)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.bias.weight)

    def forward(self, x: torch.Tensor, domain_id: torch.Tensor) -> torch.Tensor:
        """Apply the per-sample projection.

        Args:
            x: Input of shape ``[B, I]`` or ``[B, T, I]``.
            domain_id: Domain indices of shape ``[B]``.

        Returns:
            Output of shape ``[B, O]`` or ``[B, T, O]``, matching the input's rank.
        """
        batch_size = domain_id.shape[0]
        squeeze_seq = x.dim() == 2  # noqa: PLR2004
        if squeeze_seq:
            x = x.unsqueeze(1)
        weight = self.fc(domain_id).view(batch_size, self.input_size, self.output_size)
        bias = self.bias(domain_id).view(batch_size, 1, self.output_size)
        y = torch.matmul(x, weight) + bias
        return y.squeeze(1) if squeeze_seq else y


class TransformerBlock(nn.Module):
    """A pre-LayerNorm transformer block: ``x + attn(norm(x))`` then ``x + mlp(norm(x))``.

    Args:
        hidden_size: Model width.
        num_heads: Number of attention heads.
        mlp_ratio: Feed-forward expansion factor.
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        """Build the norms, the attention and the feed-forward network."""
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), drop=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the block.

        Args:
            x: Input of shape ``[B, T, H]``.

        Returns:
            Output of shape ``[B, T, H]``.
        """
        # Out-of-place: `+=` would mutate the caller's activations in the residual path.
        x = x + self.attn(self.norm1(x))  # noqa: PLR6104
        return x + self.mlp(self.norm2(x))


class SoftPromptedTransformer(nn.Module):
    """XVLA's action-denoising transformer.

    The sequence fed to the backbone is, in order: the noised action chunk (one token per
    action step, carrying the action, the proprioceptive state and the flow-matching
    timestep), the Florence-2 encoder output for the primary camera and the language
    prompt, the pooled features of the auxiliary cameras, and finally this domain's soft
    prompts. Only the action tokens are decoded back to actions.

    Args:
        hidden_size: Backbone width.
        multi_modal_input_size: Width of the Florence-2 features.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: Feed-forward expansion factor.
        num_domains: Number of domains the domain-aware layers can serve.
        dim_action: Width of the action vector.
        dim_propio: Width of the proprioceptive state (``0`` disables proprioception).
        dim_time: Width of the sinusoidal timestep features.
        len_soft_prompts: Soft-prompt tokens per domain (``0`` disables them).
        max_len_seq: Longest sequence the learned positional embedding covers.
        use_hetero_proj: Project the visual streams per domain instead of globally.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        multi_modal_input_size: int = 768,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_domains: int = 20,
        dim_action: int = 20,
        dim_propio: int = 20,
        dim_time: int = 32,
        len_soft_prompts: int = 32,
        max_len_seq: int = 512,
        *,
        use_hetero_proj: bool = False,
    ) -> None:
        """Build the backbone, the projections and the soft-prompt table."""
        super().__init__()
        self.hidden_size = hidden_size
        self.dim_action = dim_action
        self.dim_time = dim_time
        self.len_soft_prompts = len_soft_prompts
        self.use_hetero_proj = use_hetero_proj

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])

        if use_hetero_proj:
            self.vlm_proj: nn.Module = DomainAwareLinear(multi_modal_input_size, hidden_size, num_domains=num_domains)
            self.aux_visual_proj: nn.Module = DomainAwareLinear(
                multi_modal_input_size,
                hidden_size,
                num_domains=num_domains,
            )
        else:
            self.vlm_proj = nn.Linear(multi_modal_input_size, hidden_size)
            self.aux_visual_proj = nn.Linear(multi_modal_input_size, hidden_size)

        self.pos_emb = nn.Parameter(torch.zeros(1, max_len_seq, hidden_size))
        nn.init.normal_(self.pos_emb, std=0.02)

        self.norm = nn.LayerNorm(hidden_size)
        self.action_encoder = DomainAwareLinear(
            dim_action + dim_time + dim_propio,
            hidden_size,
            num_domains=num_domains,
        )
        self.action_decoder = DomainAwareLinear(hidden_size, dim_action, num_domains=num_domains)

        if len_soft_prompts > 0:
            self.soft_prompt_hub = nn.Embedding(num_domains, len_soft_prompts * hidden_size)
            nn.init.normal_(self.soft_prompt_hub.weight, std=0.02)

        self.apply(basic_init)

    def forward(
        self,
        domain_id: torch.Tensor,
        vlm_features: torch.Tensor,
        aux_visual_inputs: torch.Tensor,
        action_with_noise: torch.Tensor,
        proprio: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Denoise one action chunk.

        Args:
            domain_id: Domain indices of shape ``[B]``.
            vlm_features: Florence-2 encoder output, ``[B, T_vlm, D]``.
            aux_visual_inputs: Pooled auxiliary-camera features, ``[B, T_aux, D]``.
            action_with_noise: Noised action chunk, ``[B, T_action, dim_action]``.
            proprio: Proprioceptive state, ``[B, dim_propio]``.
            t: Flow-matching timesteps, ``[B]``.

        Returns:
            The denoised chunk, ``[B, T_action, dim_action]``.

        Raises:
            ValueError: If the assembled sequence is longer than ``max_len_seq``.
        """
        batch_size, num_actions = action_with_noise.shape[:2]

        time_emb = timestep_embedding(t, self.dim_time)
        time_tokens = time_emb.unsqueeze(1).expand(batch_size, num_actions, self.dim_time)
        proprio_tokens = proprio.unsqueeze(1).expand(batch_size, num_actions, proprio.shape[-1])
        action_tokens = torch.cat([action_with_noise, proprio_tokens, time_tokens], dim=-1)
        x = self.action_encoder(action_tokens, domain_id)

        if self.use_hetero_proj:
            visual = [self.vlm_proj(vlm_features, domain_id), self.aux_visual_proj(aux_visual_inputs, domain_id)]
        else:
            visual = [self.vlm_proj(vlm_features), self.aux_visual_proj(aux_visual_inputs)]
        x = torch.cat([x, *visual], dim=1)

        seq_len = x.shape[1]
        if seq_len > self.pos_emb.shape[1]:
            msg = (
                f"Sequence length {seq_len} exceeds max_len_seq={self.pos_emb.shape[1]}. "
                "Raise `max_len_seq`, use fewer cameras, or shrink `tokenizer_max_length`."
            )
            raise ValueError(msg)
        x = x + self.pos_emb[:, :seq_len, :]  # noqa: PLR6104

        if self.len_soft_prompts > 0:
            soft_prompts = self.soft_prompt_hub(domain_id).view(batch_size, self.len_soft_prompts, self.hidden_size)
            x = torch.cat([x, soft_prompts], dim=1)

        for block in self.blocks:
            x = block(x)

        return self.action_decoder(self.norm(x[:, :num_actions]), domain_id)


__all__ = [
    "Attention",
    "DomainAwareLinear",
    "Mlp",
    "SoftPromptedTransformer",
    "TransformerBlock",
    "basic_init",
    "timestep_embedding",
]
