# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Vendored from RLWRLD/RLDX-1 (Apache-2.0)
# ruff: noqa: T201

"""Physics stream components for RLDXActionModel: encoders, decoders, and init utilities."""

from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from physicalai.policies.components.nn import SinusoidalPositionalEncoding
from physicalai.policies.rldx1.components.action_model.blocks import (
    ExpandedDoubleStreamBlock,
    ExpandedSingleStreamBlock,
)

if TYPE_CHECKING:
    from physicalai.policies.rldx1.components.action_model.msat import MSAT


class PhysicalSignalEncoder(nn.Module):
    """Encode physics history tokens: (B, T_hist, input_dim) -> (B, T_hist, output_dim)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        """Initialize PhysicalSignalEncoder.

        Args:
            input_dim: Dimensionality of the raw input physics signal.
            hidden_dim: Intermediate projection dimensionality.
            output_dim: Output token dimensionality.
        """
        super().__init__()
        self.W1 = nn.Linear(input_dim, hidden_dim)
        self.W2 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.W3 = nn.Linear(hidden_dim, output_dim)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode physics history tokens.

        Args:
            x: Input tensor of shape (B, T_hist, input_dim).

        Returns:
            Encoded tensor of shape (B, T_hist, output_dim).
        """
        b, t, _ = x.shape
        h = self.W1(x)
        t_ids = torch.arange(t, device=x.device).float().unsqueeze(0).expand(b, -1)
        pos = self.pos_encoding(t_ids).to(dtype=h.dtype)
        h = F.silu(self.W2(torch.cat([h, pos], dim=-1)))
        return self.W3(h)  # (B, T_hist, output_dim)


class PhysicalSignalDecoder(nn.Module):
    """Decode physics predictions: (B, T, input_dim) -> (B, T, output_dim)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        """Initialize PhysicalSignalDecoder.

        Args:
            input_dim: Dimensionality of the input token.
            hidden_dim: Intermediate projection dimensionality.
            output_dim: Output physics prediction dimensionality.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode physics predictions.

        Args:
            x: Input tensor of shape (B, T, input_dim).

        Returns:
            Decoded tensor of shape (B, T, output_dim).
        """
        return self.net(x)  # (B, T, output_dim)


class PhysicsNoiseEncoder(nn.Module):
    """Encode noisy future physics tokens with diffusion timestep conditioning."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        """Initialize PhysicsNoiseEncoder.

        Args:
            input_dim: Dimensionality of the noisy future physics input.
            hidden_dim: Intermediate projection dimensionality.
            output_dim: Output token dimensionality.
        """
        super().__init__()
        self.W1 = nn.Linear(input_dim, hidden_dim)
        self.W2 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.W3 = nn.Linear(hidden_dim, output_dim)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_dim)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Encode noisy future physics tokens with diffusion timestep conditioning.

        Args:
            x: Noisy future physics tensor of shape (B, T_fut, input_dim).
            timesteps: Diffusion timestep (scalar per sample) of shape (B,).

        Returns:
            Encoded tensor of shape (B, T_fut, output_dim).

        Raises:
            ValueError: If ``timesteps`` does not have shape (B,).
        """
        b, t, _ = x.shape
        if timesteps.dim() == 1 and timesteps.shape[0] == b:
            timesteps = timesteps.unsqueeze(1).expand(-1, t)  # (B, T_fut)
        else:
            msg = "Expected `timesteps` to have shape (B,)"
            raise ValueError(msg)
        x_emb = self.W1(x)  # (B, T_fut, hidden_dim)
        t_emb = self.pos_encoding(timesteps).to(dtype=x_emb.dtype)  # (B, T_fut, hidden_dim)
        x = F.silu(self.W2(torch.cat([x_emb, t_emb], dim=-1)))  # (B, T_fut, hidden_dim)
        return self.W3(x)  # (B, T_fut, output_dim)


def _xavier(m: nn.Module) -> None:
    """Apply Xavier uniform initialization to a Linear layer's weight and zero its bias.

    Args:
        m: Module to initialize. Only ``nn.Linear`` instances are modified; others are skipped.
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def _small_noise(m: nn.Linear, std: float) -> None:
    """Initialize a Linear layer with near-zero Gaussian noise (exit-zero init).

    Args:
        m: Linear layer to initialize.
        std: Standard deviation of the normal distribution used for the weight.
    """
    nn.init.normal_(m.weight, mean=0.0, std=std)
    if m.bias is not None:
        nn.init.zeros_(m.bias)


def _reset_norm_identity(m: nn.Module) -> None:
    """Reset LayerNorm or RMSNorm to identity (weight=1, bias=0).

    Args:
        m: Normalization module to reset. Handles ``nn.LayerNorm`` (weight + optional bias)
           and RMSNorm-style modules that expose a ``weight`` ``nn.Parameter`` (weight only).
    """
    if isinstance(m, nn.LayerNorm):
        if m.weight is not None:
            nn.init.ones_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif hasattr(m, "weight") and isinstance(m.weight, nn.Parameter):
        # Covers RMSNorm (weight only, no bias)
        nn.init.ones_(m.weight)


def _last_linear(module: nn.Module) -> nn.Linear | None:
    """Return the last ``nn.Linear`` sub-module in a module, or ``None`` if there are none.

    Args:
        module: Module to search (recursively via ``module.modules()``).

    Returns:
        The last ``nn.Linear`` instance found, or ``None`` if the module contains no
        linear layers.
    """
    linears = [m for m in module.modules() if isinstance(m, nn.Linear)]
    return linears[-1] if linears else None


def init_physics_params_near_zero(action_model: nn.Module) -> None:  # noqa: PLR0912, PLR0915, C901
    """Apply exit-zero initialization to all physics stream parameters.

    Internal layers receive Xavier initialization for stable gradient flow; exit
    (output) layers receive near-zero initialization so the physics stream outputs
    approximately zero at the start of training and does not disturb the pretrained
    action stream.

    Args:
        action_model: The action model whose physics stream parameters are to be
            initialized. Must expose a ``.model`` attribute containing
            ``double_blocks`` and ``single_blocks``, and either a top-level
            ``physics_cond_encoder`` / ``physics_fut_encoder`` / ``physics_decoder``
            or an equivalent ``.physics`` sub-module with those attributes.
    """
    msat = cast("MSAT", action_model.model)

    # Support both old layout (action_model.physics_cond_encoder)
    # and new layout (action_model.physics.physics_cond_encoder)
    physics_owner = cast("nn.Module", getattr(action_model, "physics", None) or action_model)

    # ── (A) Encoder: W1,W2 = Xavier, W3 = near-zero (exit) ──
    # Note: physics_fut_encoder also gets near-zero init (unlike reference which uses
    # PyTorch default). This is a design choice to ensure the future physics stream
    # outputs ~0 at Day-0, combined with NewParamWarmupCallback for gradual fade-in.
    for enc_name in ("physics_cond_encoder", "physics_fut_encoder"):
        if hasattr(physics_owner, enc_name):
            enc = getattr(physics_owner, enc_name)
            _xavier(enc.W1)
            _xavier(enc.W2)
            _small_noise(enc.W3, std=1e-5)
            print(f"   [Physics init] {enc_name}: W1,W2=Xavier, W3=near-zero(1e-5)")

    # ── (A) Decoder: first Linear = Kaiming (keep), last = near-zero (exit) ──
    if hasattr(physics_owner, "physics_decoder"):
        decoder = cast("PhysicalSignalDecoder", physics_owner.physics_decoder)
        last = _last_linear(decoder.net)
        if last is not None:
            _small_noise(last, std=1e-4)
        print("   [Physics init] decoder: last_linear=near-zero(1e-4)")

    # ── (B) ExpandedDoubleStreamBlock — P stream ──
    n_double = 0
    for blk in msat.double_blocks:
        if not isinstance(blk, ExpandedDoubleStreamBlock):
            continue
        n_double += 1

        _xavier(blk.p_qkv)

        _small_noise(blk.p_proj, std=1e-4)

        # p_mlp: internal Xavier, last linear near-zero (exit)
        mlp_linears = [m for m in blk.p_mlp.modules() if isinstance(m, nn.Linear)]
        for lin in mlp_linears:
            _xavier(lin)
        if mlp_linears:
            _small_noise(mlp_linears[-1], std=1e-4)

        if hasattr(blk, "p_mod"):
            blk.p_mod.apply(_xavier)

        # Norms: identity reset (LayerNorm and RMSNorm)
        for attr in ("p_norm1", "p_norm2_attn", "p_norm2_mlp", "p_norm3_mlp"):
            m = getattr(blk, attr, None)
            if m is not None:
                _reset_norm_identity(m)

        # QK norms: identity reset (matches wip TripleStreamBlock behaviour —
        # _init_layernorm_identity is called but has no effect on RMSNorm;
        # we keep the same semantics: leave at default weight=1)
        for attr in ("q_norm_p", "k_norm_p"):
            m = getattr(blk, attr, None)
            if m is not None:
                _reset_norm_identity(m)

    if n_double > 0:
        print(
            f"   [Physics init] {n_double} ExpandedDoubleStreamBlocks: "
            f"p_qkv=Xavier, p_proj=near-zero, p_mlp exit=near-zero",
        )

    # ── (C) ExpandedSingleStreamBlock — P stream ──
    n_single = 0
    for blk in msat.single_blocks:
        if not isinstance(blk, ExpandedSingleStreamBlock):
            continue
        n_single += 1

        _xavier(blk.p_linear1)

        _small_noise(blk.p_linear2, std=1e-4)

        # p_mlp_proj: Xavier (if SwiGLU)
        if getattr(blk, "p_mlp_proj", None) is not None:
            _xavier(blk.p_mlp_proj)

        # Norms: identity reset
        if hasattr(blk, "p_pre_norm"):
            _reset_norm_identity(blk.p_pre_norm)
        if hasattr(blk, "p_post_norm"):
            _reset_norm_identity(blk.p_post_norm)

        # QK norms: Xavier (matches wip DoubleStreamUpperBlock behaviour)
        for attr in ("p_q_norm", "p_k_norm"):
            m = getattr(blk, attr, None)
            if m is not None and hasattr(m, "weight"):
                _xavier(m)

    if n_single > 0:
        print(
            f"   [Physics init] {n_single} ExpandedSingleStreamBlocks: p_linear1=Xavier, p_linear2=near-zero",
        )

    # ── (D) MSAT physics output projection ──
    if hasattr(msat, "proj_out_physics_1"):
        _small_noise(cast("nn.Linear", msat.proj_out_physics_1), std=1e-5)
        print("   [Physics init] proj_out_physics_1=near-zero(1e-5)")
    if hasattr(msat, "proj_out_physics_2"):
        _small_noise(cast("nn.Linear", msat.proj_out_physics_2), std=1e-4)
        print("   [Physics init] proj_out_physics_2=near-zero(1e-4)")
