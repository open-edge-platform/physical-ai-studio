# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Vendored from RLWRLD/RLDX-1 (Apache-2.0)

"""MSAT stream blocks: Single/Double/Expanded/Triple (extracted from msat.py)."""

from typing import NamedTuple

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from physicalai.policies.rldx1.components.action_model.ops import (
    SwiGLUFFN,
    _merge_heads,  # noqa: PLC2701
    _split_heads,  # noqa: PLC2701
    apply_rotary_emb,
)
from physicalai.policies.rldx1.components.norms import create_norm_layer, create_qk_norm_layers


class ModulationOut(NamedTuple):
    """Modulation output containing shift, scale, and gate tensors."""

    shift: torch.Tensor
    scale: torch.Tensor
    gate: torch.Tensor


class Modulation(nn.Module):
    """Flux-style modulation for AdaLN."""

    def __init__(self, dim: int, *, double: bool, remove_bias: bool = False) -> None:
        """Initialize Modulation.

        Args:
            dim: Hidden dimension used for both input and per-parameter output size.
            double: If ``True``, generates 6 parameters (two ``ModulationOut``); otherwise 3 (one).
            remove_bias: If ``True``, disables the bias term in the linear projection.
        """
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=not remove_bias)

    def forward(self, vec: torch.Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        """Compute AdaLN modulation parameters from a conditioning vector.

        Args:
            vec: Conditioning vector of shape ``(B, dim)``.

        Returns:
            Tuple ``(mod1, mod2)`` where ``mod1`` is always a ``ModulationOut`` and
            ``mod2`` is a ``ModulationOut`` when ``double=True`` or ``None`` otherwise.
        """
        out = self.lin(F.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)
        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


# Models ================================================
class SingleStreamBlock(nn.Module):
    """Flux-style SingleStreamBlock with parallel linear layers."""

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        activation_fn: str = "gelu",  # noqa: ARG002
        attention_bias: bool = True,  # noqa: FBT001, FBT002
        norm_eps: float = 1e-6,
        qk_norm: str = "none",
        use_swiglu: bool = False,  # noqa: FBT001, FBT002, ARG002
        positional_embeddings: str | None = None,
        max_seq_length: int | None = None,
        temb_type: str = "layerwise_mod",
        remove_bias: bool = False,  # noqa: FBT001, FBT002
        pre_norm: str = "layer_norm",
        post_norm: str = "none",
    ) -> None:
        """Initialize SingleStreamBlock.

        Args:
            hidden_size: Hidden dimension for the stream.
            num_attention_heads: Number of attention heads.
            attention_head_dim: Dimension per attention head.
            mlp_ratio: MLP expansion ratio. Default ``4.0``.
            dropout: Dropout probability. Default ``0.0``.
            activation_fn: Activation function name (unused; kept for API compatibility). Default ``"gelu"``.
            attention_bias: Whether to use bias in QKV/projection linears. Default ``True``.
            norm_eps: Epsilon for normalization layers. Default ``1e-6``.
            qk_norm: QK normalization strategy (``"none"``, ``"rms_norm"``, etc.). Default ``"none"``.
            use_swiglu: Unused; kept for API compatibility. Default ``False``.
            positional_embeddings: Positional embedding type. Supported: ``None``,
                ``"rope_sa_only"``, ``"rope_vl_sa"``. Default ``None``.
            max_seq_length: Maximum sequence length (for positional encodings). Default ``None``.
            temb_type: Timestep embedding injection strategy (``"layerwise_mod"``,
                ``"shared_mod"``, ``"input_token"``). Default ``"layerwise_mod"``.
            remove_bias: If ``True``, removes bias from modulation and MLP projection. Default ``False``.
            pre_norm: Pre-normalization layer type. Default ``"layer_norm"``.
            post_norm: Post-normalization layer type (``"none"`` disables it). Default ``"none"``.

        Raises:
            NotImplementedError: If ``positional_embeddings="sinusoidal"`` is requested.
        """
        super().__init__()
        if positional_embeddings == "sinusoidal":
            msg = "positional_embeddings='sinusoidal' is not supported; use 'rope_sa_only', 'rope_vl_sa', or None."
            raise NotImplementedError(msg)
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = attention_head_dim
        self.inner_dim = num_attention_heads * attention_head_dim
        self.mlp_ratio = mlp_ratio
        self.temb_type = temb_type
        self.max_seq_length = max_seq_length

        # For SwiGLU: mlp_hidden_dim is 2/3 of expanded dim, so we need
        # 2 * expanded_dim for input.
        mlp_expanded_dim = int(hidden_size * mlp_ratio)
        self.mlp_hidden_dim = int(2 / 3 * mlp_expanded_dim)
        mlp_in_dim = 2 * mlp_expanded_dim

        # Projection from SwiGLU output (expanded_dim) to mlp_hidden_dim
        # (2/3 * expanded_dim).
        self.mlp_proj = nn.Linear(mlp_expanded_dim, self.mlp_hidden_dim, bias=not remove_bias)

        # Parallel linear layers: qkv (using inner_dim = num_heads * head_dim) and mlp_in together
        self.linear1 = nn.Linear(hidden_size, self.inner_dim * 3 + mlp_in_dim, bias=attention_bias)
        # proj (inner_dim) and mlp_out together -> hidden_size
        self.linear2 = nn.Linear(
            self.inner_dim + self.mlp_hidden_dim,
            hidden_size,
            bias=attention_bias,
        )

        # QK normalization
        self.q_norm, self.k_norm = create_qk_norm_layers(qk_norm, self.head_dim, norm_eps)

        # Normalization layers
        # Pre-norm: applied before parallel computation (shared for attention and MLP)
        self.pre_norm = create_norm_layer(pre_norm, hidden_size, eps=norm_eps)
        # Post-norm: applied to the final output after linear2 (not to individual attention/MLP branches)
        self.post_norm = create_norm_layer(post_norm, hidden_size, eps=norm_eps)
        # Modulation takes inner_dim (temb dimension) and outputs hidden_size-sized parameters
        # For SingleStreamBlock, we assume hidden_size == inner_dim (from DoubleStreamBlock output)
        if temb_type not in {"shared_mod", "input_token"}:
            self.modulation = Modulation(hidden_size, double=False, remove_bias=remove_bias)

        # Positional embedding mode (RoPE or disabled)
        self.positional_embeddings = positional_embeddings
        if positional_embeddings in {"rope_sa_only", "rope_vl_sa"}:
            # RoPE will be applied in forward pass, no embedding module needed here
            self.pos_embed = None
            self.use_rope = True
        else:
            self.pos_embed = None
            self.use_rope = False

        self.dropout = nn.Dropout(dropout)

    def forward(  # noqa: PLR0914
        self,
        x: torch.Tensor,
        temb: torch.Tensor,
        pe: torch.Tensor | None = None,
        shared_modulation: ModulationOut | None = None,
        time_token: torch.Tensor | None = None,
        block_idx: int = 0,  # noqa: ARG002
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass following Flux SingleStreamBlock pattern.

        Args:
            x: Concatenated VL+SA tokens of shape ``(B, N, hidden_size)``.
            temb: Timestep embedding of shape ``(B, hidden_size)``.
            pe: RoPE complex frequencies of shape ``(B, N, head_dim // 2)`` as
                ``complex64`` when RoPE is enabled, else ``None``.
            shared_modulation: Pre-computed ``ModulationOut`` used when
                ``temb_type="shared_mod"``. Ignored otherwise.
            time_token: When not ``None``, signals that a time token is present and
                causes identity (no-op) modulation to be applied.
            block_idx: Block index (reserved for future use). Default ``0``.
            attn_mask: Optional attention mask passed to
                ``F.scaled_dot_product_attention``.

        Returns:
            Updated ``x`` of shape ``(B, N, hidden_size)``.

        Raises:
            ValueError: If ``temb_type="shared_mod"`` but ``shared_modulation`` is ``None``.
            AttributeError: If the ``modulation`` attribute is missing for
                ``temb_type="layerwise_mod"``.
        """
        # If time_token is provided (not None), use identity modulation (time_token handles conditioning)
        use_identity_mod = time_token is not None
        if use_identity_mod:
            mod = ModulationOut(
                shift=torch.zeros(x.shape[0], 1, self.hidden_size, device=x.device, dtype=x.dtype),
                scale=torch.zeros(x.shape[0], 1, self.hidden_size, device=x.device, dtype=x.dtype),
                gate=torch.ones(x.shape[0], 1, self.hidden_size, device=x.device, dtype=x.dtype),
            )
        elif self.temb_type == "shared_mod":
            if shared_modulation is None:
                msg = "temb_type='shared_mod' but shared_modulation is None"
                raise ValueError(msg)
            mod = shared_modulation
        else:
            if not hasattr(self, "modulation"):
                msg = f"modulation not found. temb_type={self.temb_type} may not create modulation modules."
                raise AttributeError(msg)
            mod, _ = self.modulation(temb)
        # Pre-norm (shared for parallel computation)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        # Optional positional embedding hook (disabled in current supported modes).
        if self.pos_embed is not None:
            x_mod = self.pos_embed(x_mod)

        # Parallel computation: qkv and SwiGLU mlp_in.
        mlp_in_dim = 2 * int(self.hidden_size * self.mlp_ratio)

        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.inner_dim, mlp_in_dim], dim=-1)

        # Split QKV and apply normalization
        q, k, v = qkv.chunk(3, dim=-1)  # (B, N, inner_dim)
        q = _split_heads(q, self.num_heads)  # (B, H, N, Dh)
        k = _split_heads(k, self.num_heads)  # (B, H, N, Dh)
        v = _split_heads(v, self.num_heads)  # (B, H, N, Dh)

        # Apply QK normalization
        batch_size, head, num, d_head = q.shape
        q = q.reshape(batch_size * head, num, d_head)
        k = k.reshape(batch_size * head, num, d_head)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = q.reshape(batch_size, head, num, d_head)
        k = k.reshape(batch_size, head, num, d_head)

        # Apply RoPE to Q and K if enabled (before attention computation)
        if self.use_rope and pe is not None:
            q, k = apply_rotary_emb(q, k, pe)

        # Compute attention
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)  # (B, H, N, Dh)
        attn = _merge_heads(attn)  # (B, N, inner_dim)

        # Apply SwiGLU: mlp is (B, N, 2 * expanded_dim), split into two parts.
        mlp_x1, mlp_x2 = mlp.chunk(2, dim=-1)
        mlp_out = F.silu(mlp_x1) * mlp_x2
        mlp_out = self.mlp_proj(mlp_out)

        # Concatenate attention and MLP outputs, then apply second linear layer
        output = self.linear2(torch.cat([attn, mlp_out], dim=-1))  # (B, N, inner_dim)
        # Apply post-norm to the final output (Z-Image style: norm after combining branches)
        output = self.post_norm(output)
        return x + mod.gate * self.dropout(output)


class DoubleStreamBlock(nn.Module):
    """Flux-style DoubleStreamBlock."""

    sa_mod_proj: nn.Linear | nn.Identity
    vl_mod_proj: nn.Linear | nn.Identity

    def __init__(  # noqa: PLR0913, PLR0917, PLR0915
        self,
        sa_dim: int,
        vl_dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 4.0,
        vl_mlp_ratio: float | None = None,  # If None, use mlp_ratio
        dropout: float = 0.0,
        attention_bias: bool = True,  # noqa: FBT001, FBT002
        norm_eps: float = 1e-6,
        qk_norm: str = "none",
        positional_embeddings: str | None = None,
        max_seq_length: int | None = None,
        temb_type: str = "layerwise_mod",
        remove_bias: bool = False,  # noqa: FBT001, FBT002
        pre_norm: str = "layer_norm",
        post_norm: str = "none",
    ) -> None:
        """Initialize DoubleStreamBlock.

        Args:
            sa_dim: Hidden dimension of the SA (action/robot-state) stream.
            vl_dim: Hidden dimension of the VL (vision-language) stream.
            num_attention_heads: Number of joint attention heads.
            attention_head_dim: Dimension per attention head.
            mlp_ratio: MLP expansion ratio for the SA stream. Default ``4.0``.
            vl_mlp_ratio: MLP expansion ratio for the VL stream. ``None`` uses
                ``mlp_ratio``. Default ``None``.
            dropout: Dropout probability. Default ``0.0``.
            attention_bias: Whether to use bias in QKV/projection linears. Default ``True``.
            norm_eps: Epsilon for normalization layers. Default ``1e-6``.
            qk_norm: QK normalization strategy. Default ``"none"``.
            positional_embeddings: Positional embedding type. Supported: ``None``,
                ``"rope_sa_only"``, ``"rope_vl_sa"``. Default ``None``.
            max_seq_length: Maximum sequence length. Default ``None``.
            temb_type: Timestep embedding injection strategy. Default ``"layerwise_mod"``.
            remove_bias: If ``True``, removes bias from modulation. Default ``False``.
            pre_norm: Pre-normalization layer type. Default ``"layer_norm"``.
            post_norm: Post-normalization layer type. Default ``"none"``.

        Raises:
            NotImplementedError: If ``positional_embeddings="sinusoidal"`` is requested.
        """
        super().__init__()
        if positional_embeddings == "sinusoidal":
            msg = ("positional_embeddings='sinusoidal' is not supported; use 'rope_sa_only', 'rope_vl_sa', or None.",)
            raise NotImplementedError(msg)
        self.sa_dim = sa_dim
        self.vl_dim = vl_dim
        self.num_heads = num_attention_heads
        self.head_dim = attention_head_dim
        self.inner_dim = num_attention_heads * attention_head_dim
        self.temb_type = temb_type
        self.max_seq_length = max_seq_length
        # Use separate MLP ratio for VL stream if specified, otherwise use same as SA
        vl_mlp_ratio_actual = vl_mlp_ratio if vl_mlp_ratio is not None else mlp_ratio

        # SA stream components
        # Modulation takes inner_dim (temb dimension) and outputs sa_dim-sized parameters
        if temb_type not in {"shared_mod", "input_token"}:
            self.sa_mod = Modulation(self.inner_dim, double=True, remove_bias=remove_bias)
        # Project modulation output to sa_dim if needed (not needed for input_token)
        if temb_type != "input_token" and self.inner_dim != sa_dim:
            self.sa_mod_proj = nn.Linear(self.inner_dim, sa_dim, bias=True)
        else:
            self.sa_mod_proj = nn.Identity()

        # SA stream normalization layers
        self.sa_norm1 = create_norm_layer(pre_norm, sa_dim, eps=norm_eps)  # Pre-attention
        self.sa_norm2_attn = create_norm_layer(post_norm, sa_dim, eps=norm_eps)  # Post-attention
        self.sa_norm2_mlp = create_norm_layer(pre_norm, sa_dim, eps=norm_eps)  # Pre-FFN
        self.sa_norm3_mlp = create_norm_layer(post_norm, sa_dim, eps=norm_eps)  # Post-FFN

        self.sa_qkv = nn.Linear(sa_dim, self.inner_dim * 3, bias=attention_bias)
        self.sa_proj = nn.Linear(self.inner_dim, sa_dim, bias=attention_bias)
        mlp_hidden_dim_sa = int(sa_dim * mlp_ratio)
        self.sa_mlp = SwiGLUFFN(sa_dim, int(2 / 3 * mlp_hidden_dim_sa))

        # VL stream components
        # Modulation takes inner_dim (temb dimension) and outputs vl_dim-sized parameters
        if temb_type not in {"shared_mod", "input_token"}:
            self.vl_mod = Modulation(self.inner_dim, double=True, remove_bias=remove_bias)
        # Project modulation output to vl_dim if needed (not needed for input_token)
        if temb_type != "input_token" and self.inner_dim != vl_dim:
            self.vl_mod_proj = nn.Linear(self.inner_dim, vl_dim, bias=True)
        else:
            self.vl_mod_proj = nn.Identity()

        # VL stream normalization layers
        self.vl_norm1 = create_norm_layer(pre_norm, vl_dim, eps=norm_eps)  # Pre-attention
        self.vl_norm2_attn = create_norm_layer(post_norm, vl_dim, eps=norm_eps)  # Post-attention
        self.vl_norm2_mlp = create_norm_layer(pre_norm, vl_dim, eps=norm_eps)  # Pre-FFN
        self.vl_norm3_mlp = create_norm_layer(post_norm, vl_dim, eps=norm_eps)  # Post-FFN

        self.vl_qkv = nn.Linear(vl_dim, self.inner_dim * 3, bias=attention_bias)
        self.vl_proj = nn.Linear(self.inner_dim, vl_dim, bias=attention_bias)
        mlp_hidden_dim_vl = int(vl_dim * vl_mlp_ratio_actual)
        self.vl_mlp = SwiGLUFFN(vl_dim, int(2 / 3 * mlp_hidden_dim_vl))

        # QK normalization (per modality)
        self.q_norm_sa, self.k_norm_sa = create_qk_norm_layers(qk_norm, self.head_dim, norm_eps)
        self.q_norm_vl, self.k_norm_vl = create_qk_norm_layers(qk_norm, self.head_dim, norm_eps)

        # Positional embedding mode (RoPE or disabled)
        self.positional_embeddings = positional_embeddings
        if positional_embeddings in {"rope_sa_only", "rope_vl_sa"}:
            # RoPE will be applied in forward pass, no embedding module needed here
            self.pos_embed_sa = None
            self.pos_embed_vl = None
            self.use_rope = True
        else:
            self.pos_embed_sa = None
            self.pos_embed_vl = None
            self.use_rope = False

        self.dropout = nn.Dropout(dropout)

    def forward(  # noqa: PLR0914, PLR0915
        self,
        sa_tokens: torch.Tensor,
        vl_tokens: torch.Tensor,
        temb: torch.Tensor,
        pe: torch.Tensor | None = None,
        shared_modulations: dict | None = None,
        has_time_token: bool = False,  # noqa: FBT001, FBT002
        block_idx: int = 0,  # noqa: ARG002
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass following Flux DoubleStreamBlock pattern.

        Args:
            sa_tokens: SA stream tokens of shape ``(B, N_sa, sa_dim)``. ``N_sa``
                includes the time token when ``has_time_token=True``.
            vl_tokens: VL stream tokens of shape ``(B, N_vl, vl_dim)``.
            temb: Timestep embedding of shape ``(B, inner_dim)``.
            pe: RoPE complex frequencies of shape ``(B, N_vl + N_sa, head_dim // 2)``
                as ``complex64`` when RoPE is enabled, else ``None``.
            shared_modulations: Optional dict with keys ``"sa_mod1_raw"``,
                ``"sa_mod2_raw"``, ``"vl_mod1_raw"``, ``"vl_mod2_raw"`` used when
                ``temb_type="shared_mod"``.
            has_time_token: If ``True``, a time token is present in ``sa_tokens`` and
                identity modulation is applied.
            block_idx: Block index (reserved for future use). Default ``0``.
            encoder_attention_mask: Optional key-value visibility mask of shape
                ``(B, N_vl)`` (1 = visible, 0 = masked) for VL tokens.

        Returns:
            Tuple ``(sa_tokens, vl_tokens)`` with updated tensors of their original shapes.

        Raises:
            ValueError: If batch sizes of ``sa_tokens`` and ``vl_tokens`` do not match.
            AttributeError: If the modulation attributes are missing for
                ``temb_type="layerwise_mod"``.
        """
        batch, n_sa, _ = sa_tokens.shape
        batch2, n_vl, _ = vl_tokens.shape
        if batch != batch2:
            msg = f"Batch size mismatch: sa_tokens {batch} vs vl_tokens {batch2}"
            raise ValueError(msg)

        use_identity_mod = has_time_token

        # Get modulation parameters for both streams
        if use_identity_mod:
            # Use identity modulation when time_token is present
            sa_mod1 = ModulationOut(
                shift=torch.zeros(
                    batch,
                    1,
                    self.sa_dim,
                    device=sa_tokens.device,
                    dtype=sa_tokens.dtype,
                ),
                scale=torch.zeros(
                    batch,
                    1,
                    self.sa_dim,
                    device=sa_tokens.device,
                    dtype=sa_tokens.dtype,
                ),
                gate=torch.ones(batch, 1, self.sa_dim, device=sa_tokens.device, dtype=sa_tokens.dtype),
            )
            sa_mod2 = ModulationOut(
                shift=torch.zeros(
                    batch,
                    1,
                    self.sa_dim,
                    device=sa_tokens.device,
                    dtype=sa_tokens.dtype,
                ),
                scale=torch.zeros(
                    batch,
                    1,
                    self.sa_dim,
                    device=sa_tokens.device,
                    dtype=sa_tokens.dtype,
                ),
                gate=torch.ones(batch, 1, self.sa_dim, device=sa_tokens.device, dtype=sa_tokens.dtype),
            )
            vl_mod1 = ModulationOut(
                shift=torch.zeros(
                    batch,
                    1,
                    self.vl_dim,
                    device=vl_tokens.device,
                    dtype=vl_tokens.dtype,
                ),
                scale=torch.zeros(
                    batch,
                    1,
                    self.vl_dim,
                    device=vl_tokens.device,
                    dtype=vl_tokens.dtype,
                ),
                gate=torch.ones(batch, 1, self.vl_dim, device=vl_tokens.device, dtype=vl_tokens.dtype),
            )
            vl_mod2 = ModulationOut(
                shift=torch.zeros(
                    batch,
                    1,
                    self.vl_dim,
                    device=vl_tokens.device,
                    dtype=vl_tokens.dtype,
                ),
                scale=torch.zeros(
                    batch,
                    1,
                    self.vl_dim,
                    device=vl_tokens.device,
                    dtype=vl_tokens.dtype,
                ),
                gate=torch.ones(batch, 1, self.vl_dim, device=vl_tokens.device, dtype=vl_tokens.dtype),
            )
        elif self.temb_type == "shared_mod" and shared_modulations is not None:
            sa_mod1_raw = shared_modulations["sa_mod1_raw"]
            sa_mod2_raw = shared_modulations["sa_mod2_raw"]
            vl_mod1_raw = shared_modulations["vl_mod1_raw"]
            vl_mod2_raw = shared_modulations["vl_mod2_raw"]

            # Project modulation parameters to match stream dimensions
            sa_mod1 = ModulationOut(
                shift=self.sa_mod_proj(sa_mod1_raw.shift),
                scale=self.sa_mod_proj(sa_mod1_raw.scale),
                gate=self.sa_mod_proj(sa_mod1_raw.gate),
            )
            sa_mod2 = ModulationOut(
                shift=self.sa_mod_proj(sa_mod2_raw.shift),
                scale=self.sa_mod_proj(sa_mod2_raw.scale),
                gate=self.sa_mod_proj(sa_mod2_raw.gate),
            )
            vl_mod1 = ModulationOut(
                shift=self.vl_mod_proj(vl_mod1_raw.shift),
                scale=self.vl_mod_proj(vl_mod1_raw.scale),
                gate=self.vl_mod_proj(vl_mod1_raw.gate),
            )
            vl_mod2 = ModulationOut(
                shift=self.vl_mod_proj(vl_mod2_raw.shift),
                scale=self.vl_mod_proj(vl_mod2_raw.scale),
                gate=self.vl_mod_proj(vl_mod2_raw.gate),
            )
        else:
            # Layerwise modulation (temb_type="layerwise_mod")
            if not hasattr(self, "sa_mod") or not hasattr(self, "vl_mod"):
                msg = f"sa_mod or vl_mod not found. temb_type={self.temb_type} may not create modulation modules."
                raise AttributeError(msg)
            sa_mod1_raw, sa_mod2_raw = self.sa_mod(temb)
            vl_mod1_raw, vl_mod2_raw = self.vl_mod(temb)

            # Project modulation parameters to match stream dimensions
            sa_mod1 = ModulationOut(
                shift=self.sa_mod_proj(sa_mod1_raw.shift),
                scale=self.sa_mod_proj(sa_mod1_raw.scale),
                gate=self.sa_mod_proj(sa_mod1_raw.gate),
            )
            sa_mod2 = ModulationOut(
                shift=self.sa_mod_proj(sa_mod2_raw.shift),
                scale=self.sa_mod_proj(sa_mod2_raw.scale),
                gate=self.sa_mod_proj(sa_mod2_raw.gate),
            )
            vl_mod1 = ModulationOut(
                shift=self.vl_mod_proj(vl_mod1_raw.shift),
                scale=self.vl_mod_proj(vl_mod1_raw.scale),
                gate=self.vl_mod_proj(vl_mod1_raw.gate),
            )
            vl_mod2 = ModulationOut(
                shift=self.vl_mod_proj(vl_mod2_raw.shift),
                scale=self.vl_mod_proj(vl_mod2_raw.scale),
                gate=self.vl_mod_proj(vl_mod2_raw.gate),
            )

        # Prepare SA for attention
        sa_modulated = self.sa_norm1(sa_tokens)
        sa_modulated = (1 + sa_mod1.scale) * sa_modulated + sa_mod1.shift
        # Optional positional embedding hook (disabled in current supported modes).
        if self.pos_embed_sa is not None:
            sa_modulated = self.pos_embed_sa(sa_modulated)
        sa_qkv = self.sa_qkv(sa_modulated)  # (B, N_sa, inner_dim * 3)
        sa_q, sa_k, sa_v = sa_qkv.chunk(3, dim=-1)  # (B, N_sa, inner_dim)
        sa_q = _split_heads(sa_q, self.num_heads)  # (B, H, N_sa, Dh)
        sa_k = _split_heads(sa_k, self.num_heads)  # (B, H, N_sa, Dh)
        sa_v = _split_heads(sa_v, self.num_heads)  # (B, H, N_sa, Dh)

        # Prepare VL for attention
        vl_modulated = self.vl_norm1(vl_tokens)
        vl_modulated = (1 + vl_mod1.scale) * vl_modulated + vl_mod1.shift
        # Optional positional embedding hook (disabled in current supported modes).
        if self.pos_embed_vl is not None:
            vl_modulated = self.pos_embed_vl(vl_modulated)
        vl_qkv = self.vl_qkv(vl_modulated)  # (B, N_vl, inner_dim * 3)
        vl_q, vl_k, vl_v = vl_qkv.chunk(3, dim=-1)  # (B, N_vl, inner_dim)
        vl_q = _split_heads(vl_q, self.num_heads)  # (B, H, N_vl, Dh)
        vl_k = _split_heads(vl_k, self.num_heads)  # (B, H, N_vl, Dh)
        vl_v = _split_heads(vl_v, self.num_heads)  # (B, H, N_vl, Dh)

        # Apply QK normalization per modality
        batch, heads, _, d_head = sa_q.shape
        sa_q = sa_q.reshape(batch * heads, n_sa, d_head)
        sa_k = sa_k.reshape(batch * heads, n_sa, d_head)
        vl_q = vl_q.reshape(batch * heads, n_vl, d_head)
        vl_k = vl_k.reshape(batch * heads, n_vl, d_head)

        sa_q = self.q_norm_sa(sa_q)
        sa_k = self.k_norm_sa(sa_k)
        vl_q = self.q_norm_vl(vl_q)
        vl_k = self.k_norm_vl(vl_k)

        sa_q = sa_q.reshape(batch, heads, n_sa, d_head)
        sa_k = sa_k.reshape(batch, heads, n_sa, d_head)
        vl_q = vl_q.reshape(batch, heads, n_vl, d_head)
        vl_k = vl_k.reshape(batch, heads, n_vl, d_head)

        # Joint attention: concat Q, K, V (VL first, then time_token+SA)
        q = torch.cat([vl_q, sa_q], dim=2)  # (B, H, N_vl + N_sa, Dh)
        k = torch.cat([vl_k, sa_k], dim=2)  # (B, H, N_vl + N_sa, Dh)
        v = torch.cat([vl_v, sa_v], dim=2)  # (B, H, N_vl + N_sa, Dh)

        # Apply RoPE to Q and K if enabled (before attention computation)
        if self.use_rope and pe is not None:
            q, k = apply_rotary_emb(q, k, pe)

        # Build joint attention mask from encoder_attention_mask if provided
        joint_attn_mask = None
        if encoder_attention_mask is not None:
            b_mask = encoder_attention_mask.shape[0]
            sa_ones = torch.ones(
                b_mask,
                n_sa,
                device=encoder_attention_mask.device,
                dtype=encoder_attention_mask.dtype,
            )
            kv_mask = torch.cat([encoder_attention_mask, sa_ones], dim=1)  # [B, N_vl + N_sa]
            joint_attn_mask = kv_mask[:, None, None, :]  # [B, 1, 1, N_kv]
            joint_attn_mask = torch.where(
                joint_attn_mask == 0,
                torch.tensor(float("-inf"), device=q.device, dtype=q.dtype),
                torch.tensor(0.0, device=q.device, dtype=q.dtype),
            )

        # Joint attention computation
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=joint_attn_mask,
        )  # (B, H, N_vl + N_sa, Dh)

        # Split attention outputs
        vl_attn, sa_attn = (
            attn_out[:, :, :n_vl],
            attn_out[:, :, n_vl:],
        )  # (B, H, N_vl, Dh), (B, H, N_sa, Dh)

        # Project and apply post-norm if enabled, then residual with gate
        sa_attn = _merge_heads(sa_attn)  # (B, N_sa, inner_dim)
        sa_attn_proj = self.sa_proj(self.dropout(sa_attn))
        sa_attn_proj = self.sa_norm2_attn(sa_attn_proj)  # Post-attention norm
        sa_tokens = sa_tokens + sa_mod1.gate * sa_attn_proj  # noqa: PLR6104

        vl_attn = _merge_heads(vl_attn)  # (B, N_vl, inner_dim)
        vl_attn_proj = self.vl_proj(self.dropout(vl_attn))
        vl_attn_proj = self.vl_norm2_attn(vl_attn_proj)  # Post-attention norm
        vl_tokens = vl_tokens + vl_mod1.gate * vl_attn_proj  # noqa: PLR6104

        # MLP blocks
        sa_mlp_input = (1 + sa_mod2.scale) * self.sa_norm2_mlp(sa_tokens) + sa_mod2.shift
        sa_mlp_out = self.sa_mlp(sa_mlp_input)
        sa_mlp_out = self.sa_norm3_mlp(sa_mlp_out)  # Post-FFN norm
        sa_tokens = sa_tokens + sa_mod2.gate * sa_mlp_out  # noqa: PLR6104

        vl_mlp_input = (1 + vl_mod2.scale) * self.vl_norm2_mlp(vl_tokens) + vl_mod2.shift
        vl_mlp_out = self.vl_mlp(vl_mlp_input)
        vl_mlp_out = self.vl_norm3_mlp(vl_mlp_out)  # Post-FFN norm
        vl_tokens = vl_tokens + vl_mod2.gate * vl_mlp_out  # noqa: PLR6104

        return sa_tokens, vl_tokens


class ExpandedDoubleStreamBlock(DoubleStreamBlock):
    """DoubleStreamBlock extended with a P (physics) stream for 3-way joint attention.

    Inherits all SA/VL parameters from DoubleStreamBlock (same attribute names,
    so pretrained weights load directly). Adds P stream parameters (p_*).

    When p_tokens is None, delegates to DoubleStreamBlock.forward (2-way attention).
    When p_tokens is given, performs 3-way joint attention [VL | SA | P].
    """

    p_mod_proj: nn.Identity | nn.Linear

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        sa_dim: int,
        vl_dim: int,
        p_dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 4.0,
        vl_mlp_ratio: float | None = None,
        p_mlp_ratio: float | None = None,
        dropout: float = 0.0,
        activation_fn: str = "gelu",  # noqa: ARG002
        attention_bias: bool = True,  # noqa: FBT001, FBT002
        norm_eps: float = 1e-6,
        qk_norm: str = "none",
        use_swiglu: bool = False,  # noqa: FBT001, FBT002, ARG002
        positional_embeddings: str | None = None,
        max_seq_length: int | None = None,
        temb_type: str = "layerwise_mod",
        remove_bias: bool = False,  # noqa: FBT001, FBT002
        pre_norm: str = "layer_norm",
        post_norm: str = "none",
    ) -> None:
        """Initialize ExpandedDoubleStreamBlock.

        Calls ``DoubleStreamBlock.__init__`` for the SA/VL streams, then registers
        the P (physics) stream parameters.

        Args:
            sa_dim: Hidden dimension of the SA stream.
            vl_dim: Hidden dimension of the VL stream.
            p_dim: Hidden dimension of the P (physics) stream.
            num_attention_heads: Number of joint attention heads.
            attention_head_dim: Dimension per attention head.
            mlp_ratio: MLP expansion ratio for the SA stream. Default ``4.0``.
            vl_mlp_ratio: MLP expansion ratio for the VL stream. ``None`` uses
                ``mlp_ratio``. Default ``None``.
            p_mlp_ratio: MLP expansion ratio for the P stream. ``None`` uses
                ``mlp_ratio``. Default ``None``.
            dropout: Dropout probability. Default ``0.0``.
            attention_bias: Whether to use bias in QKV/projection linears. Default ``True``.
            norm_eps: Epsilon for normalization layers. Default ``1e-6``.
            qk_norm: QK normalization strategy. Default ``"none"``.
            positional_embeddings: Positional embedding type. Default ``None``.
            max_seq_length: Maximum sequence length. Default ``None``.
            temb_type: Timestep embedding injection strategy. Default ``"layerwise_mod"``.
            remove_bias: If ``True``, removes bias from modulation. Default ``False``.
            pre_norm: Pre-normalization layer type. Default ``"layer_norm"``.
            post_norm: Post-normalization layer type. Default ``"none"``.
            activation_fn: Activation function for MLP. Default ``"gelu"`` (ignored, always SwiGLU).
            use_swiglu: If ``True``, uses SwiGLU for MLP. Default ``False`` (ignored, always SwiGLU).
        """
        super().__init__(
            sa_dim=sa_dim,
            vl_dim=vl_dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            mlp_ratio=mlp_ratio,
            vl_mlp_ratio=vl_mlp_ratio,
            dropout=dropout,
            attention_bias=attention_bias,
            norm_eps=norm_eps,
            qk_norm=qk_norm,
            positional_embeddings=positional_embeddings,
            max_seq_length=max_seq_length,
            temb_type=temb_type,
            remove_bias=remove_bias,
            pre_norm=pre_norm,
            post_norm=post_norm,
        )
        # ── P (physics) stream ────────────────────────────────────────────
        self.p_dim = p_dim
        p_mlp_ratio_actual = p_mlp_ratio if p_mlp_ratio is not None else mlp_ratio

        if temb_type not in {"shared_mod", "input_token"}:
            self.p_mod = Modulation(self.inner_dim, double=True, remove_bias=remove_bias)

        if temb_type != "input_token" and self.inner_dim != p_dim:
            self.p_mod_proj = nn.Linear(self.inner_dim, p_dim, bias=True)
        else:
            self.p_mod_proj = nn.Identity()

        self.p_norm1 = create_norm_layer(pre_norm, p_dim, eps=norm_eps)
        self.p_norm2_attn = create_norm_layer(post_norm, p_dim, eps=norm_eps)
        self.p_norm2_mlp = create_norm_layer(pre_norm, p_dim, eps=norm_eps)
        self.p_norm3_mlp = create_norm_layer(post_norm, p_dim, eps=norm_eps)

        self.p_qkv = nn.Linear(p_dim, self.inner_dim * 3, bias=attention_bias)
        self.p_proj = nn.Linear(self.inner_dim, p_dim, bias=attention_bias)
        mlp_hidden_dim_p = int(p_dim * p_mlp_ratio_actual)
        self.p_mlp = SwiGLUFFN(p_dim, int(2 / 3 * mlp_hidden_dim_p))

        self.q_norm_p, self.k_norm_p = create_qk_norm_layers(qk_norm, self.head_dim, norm_eps)

        self.pos_embed_p = None

    def forward(  # type: ignore[override]  # noqa: PLR0912, PLR0915, PLR0914
        self,
        sa_tokens: torch.Tensor,
        vl_tokens: torch.Tensor,
        temb: torch.Tensor,
        pe: torch.Tensor | None = None,
        shared_modulations: dict | None = None,
        has_time_token: bool = False,  # noqa: FBT001, FBT002
        block_idx: int = 0,
        encoder_attention_mask: torch.Tensor | None = None,
        p_tokens: torch.Tensor | None = None,
        physics_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with optional 3-way joint attention [VL | SA | P].

        When ``p_tokens`` is ``None``, delegates to ``DoubleStreamBlock.forward``
        (2-way SA/VL attention). When ``p_tokens`` is provided, performs 3-way joint
        attention and returns an extra physics token tensor.

        Args:
            sa_tokens: SA stream tokens of shape ``(B, N_sa, sa_dim)``.
            vl_tokens: VL stream tokens of shape ``(B, N_vl, vl_dim)``.
            temb: Timestep embedding of shape ``(B, inner_dim)``.
            pe: RoPE complex frequencies as ``complex64`` when RoPE is enabled,
                else ``None``.
            shared_modulations: Optional modulation dict for ``temb_type="shared_mod"``.
            has_time_token: If ``True``, identity modulation is applied.
            block_idx: Block index (reserved for future use). Default ``0``.
            encoder_attention_mask: Optional VL key-visibility mask of shape
                ``(B, N_vl)`` (1 = visible, 0 = masked).
            p_tokens: Optional P stream tokens of shape ``(B, N_p, p_dim)``. When
                ``None``, 2-way attention is used.
            physics_attention_mask: Optional scalar mask per sample for P-stream
                visibility of shape ``(B,)`` (1 = visible, 0 = masked).

        Returns:
            ``(sa_tokens, vl_tokens)`` when ``p_tokens`` is ``None``, or
            ``(sa_tokens, vl_tokens, p_tokens)`` when physics tokens are provided.
        """
        if p_tokens is None:
            return super().forward(
                sa_tokens,
                vl_tokens,
                temb,
                pe=pe,
                shared_modulations=shared_modulations,
                has_time_token=has_time_token,
                block_idx=block_idx,
                encoder_attention_mask=encoder_attention_mask,
            )

        # ── 3-way attention [VL | SA | P] ────────────────────────────────
        batch_size, sa, _ = sa_tokens.shape
        n_vl = vl_tokens.shape[1]
        n_p = p_tokens.shape[1]
        use_identity_mod = has_time_token

        # Modulation
        if use_identity_mod:
            sa_mod1 = sa_mod2 = ModulationOut(
                shift=torch.zeros(
                    batch_size,
                    1,
                    self.sa_dim,
                    device=sa_tokens.device,
                    dtype=sa_tokens.dtype,
                ),
                scale=torch.zeros(
                    batch_size,
                    1,
                    self.sa_dim,
                    device=sa_tokens.device,
                    dtype=sa_tokens.dtype,
                ),
                gate=torch.ones(batch_size, 1, self.sa_dim, device=sa_tokens.device, dtype=sa_tokens.dtype),
            )
            vl_mod1 = vl_mod2 = ModulationOut(
                shift=torch.zeros(
                    batch_size,
                    1,
                    self.vl_dim,
                    device=vl_tokens.device,
                    dtype=vl_tokens.dtype,
                ),
                scale=torch.zeros(
                    batch_size,
                    1,
                    self.vl_dim,
                    device=vl_tokens.device,
                    dtype=vl_tokens.dtype,
                ),
                gate=torch.ones(batch_size, 1, self.vl_dim, device=vl_tokens.device, dtype=vl_tokens.dtype),
            )
            p_mod1 = p_mod2 = ModulationOut(
                shift=torch.zeros(batch_size, 1, self.p_dim, device=p_tokens.device, dtype=p_tokens.dtype),
                scale=torch.zeros(batch_size, 1, self.p_dim, device=p_tokens.device, dtype=p_tokens.dtype),
                gate=torch.ones(batch_size, 1, self.p_dim, device=p_tokens.device, dtype=p_tokens.dtype),
            )
        elif self.temb_type == "shared_mod" and shared_modulations is not None:

            def _proj(proj: nn.Module, raw: ModulationOut) -> ModulationOut:
                """Project each component of a ``ModulationOut`` through a linear layer.

                Args:
                    proj: Projection module (e.g. ``nn.Linear`` or ``nn.Identity``).
                    raw: Source ``ModulationOut`` whose shift/scale/gate tensors are projected.

                Returns:
                    New ``ModulationOut`` with projected shift, scale, and gate tensors.
                """
                return ModulationOut(
                    shift=proj(raw.shift),
                    scale=proj(raw.scale),
                    gate=proj(raw.gate),
                )

            sa_mod1 = _proj(self.sa_mod_proj, shared_modulations["sa_mod1_raw"])
            sa_mod2 = _proj(self.sa_mod_proj, shared_modulations["sa_mod2_raw"])
            vl_mod1 = _proj(self.vl_mod_proj, shared_modulations["vl_mod1_raw"])
            vl_mod2 = _proj(self.vl_mod_proj, shared_modulations["vl_mod2_raw"])
            p_raw1 = shared_modulations.get("p_mod1_raw")
            if p_raw1 is not None:
                p_mod1 = _proj(self.p_mod_proj, p_raw1)
                p_mod2 = _proj(self.p_mod_proj, shared_modulations["p_mod2_raw"])
            else:
                p_mod1 = p_mod2 = ModulationOut(
                    shift=torch.zeros(
                        batch_size,
                        1,
                        self.p_dim,
                        device=p_tokens.device,
                        dtype=p_tokens.dtype,
                    ),
                    scale=torch.zeros(
                        batch_size,
                        1,
                        self.p_dim,
                        device=p_tokens.device,
                        dtype=p_tokens.dtype,
                    ),
                    gate=torch.ones(batch_size, 1, self.p_dim, device=p_tokens.device, dtype=p_tokens.dtype),
                )
        else:

            def _proj(proj: nn.Module, raw: ModulationOut) -> ModulationOut:
                """Project each component of a ``ModulationOut`` through a linear layer.

                Args:
                    proj: Projection module (e.g. ``nn.Linear`` or ``nn.Identity``).
                    raw: Source ``ModulationOut`` whose shift/scale/gate tensors are projected.

                Returns:
                    New ``ModulationOut`` with projected shift, scale, and gate tensors.
                """
                return ModulationOut(
                    shift=proj(raw.shift),
                    scale=proj(raw.scale),
                    gate=proj(raw.gate),
                )

            sa_mod1, sa_mod2 = self.sa_mod(temb)
            sa_mod1 = _proj(self.sa_mod_proj, sa_mod1)
            sa_mod2 = _proj(self.sa_mod_proj, sa_mod2)
            vl_mod1, vl_mod2 = self.vl_mod(temb)
            vl_mod1 = _proj(self.vl_mod_proj, vl_mod1)
            vl_mod2 = _proj(self.vl_mod_proj, vl_mod2)
            p_mod1, p_mod2 = self.p_mod(temb)
            p_mod1 = _proj(self.p_mod_proj, p_mod1)
            p_mod2 = _proj(self.p_mod_proj, p_mod2)

        # QKV preparation
        sa_mod_out = (1 + sa_mod1.scale) * self.sa_norm1(sa_tokens) + sa_mod1.shift
        if self.pos_embed_sa is not None:
            sa_mod_out = self.pos_embed_sa(sa_mod_out)
        sa_q, sa_k, sa_v = [_split_heads(t, self.num_heads) for t in self.sa_qkv(sa_mod_out).chunk(3, dim=-1)]

        vl_mod_out = (1 + vl_mod1.scale) * self.vl_norm1(vl_tokens) + vl_mod1.shift
        if self.pos_embed_vl is not None:
            vl_mod_out = self.pos_embed_vl(vl_mod_out)
        vl_q, vl_k, vl_v = [_split_heads(t, self.num_heads) for t in self.vl_qkv(vl_mod_out).chunk(3, dim=-1)]

        p_mod_out = (1 + p_mod1.scale) * self.p_norm1(p_tokens) + p_mod1.shift
        if self.pos_embed_p is not None:
            p_mod_out = self.pos_embed_p(p_mod_out)
        p_q, p_k, p_v = [_split_heads(t, self.num_heads) for t in self.p_qkv(p_mod_out).chunk(3, dim=-1)]

        # QK norm — direct reassignment so the norm modules stay on the
        # autograd graph. The previous `q_t.data = ...` loop swapped storage
        # but bypassed grad_fn, leaving q_norm_*/k_norm_* weights ungradable.
        batch_size, num_heads, _, d_head = sa_q.shape
        sa_q = self.q_norm_sa(sa_q.reshape(batch_size * num_heads, sa, d_head)).reshape(
            batch_size,
            num_heads,
            sa,
            d_head,
        )
        sa_k = self.k_norm_sa(sa_k.reshape(batch_size * num_heads, sa, d_head)).reshape(
            batch_size,
            num_heads,
            sa,
            d_head,
        )
        vl_q = self.q_norm_vl(vl_q.reshape(batch_size * num_heads, n_vl, d_head)).reshape(
            batch_size,
            num_heads,
            n_vl,
            d_head,
        )
        vl_k = self.k_norm_vl(vl_k.reshape(batch_size * num_heads, n_vl, d_head)).reshape(
            batch_size,
            num_heads,
            n_vl,
            d_head,
        )
        p_q = self.q_norm_p(p_q.reshape(batch_size * num_heads, n_p, d_head)).reshape(
            batch_size,
            num_heads,
            n_p,
            d_head,
        )
        p_k = self.k_norm_p(p_k.reshape(batch_size * num_heads, n_p, d_head)).reshape(
            batch_size,
            num_heads,
            n_p,
            d_head,
        )

        # Joint attention [VL | SA | P]
        q = torch.cat([vl_q, sa_q, p_q], dim=2)
        k = torch.cat([vl_k, sa_k, p_k], dim=2)
        v = torch.cat([vl_v, sa_v, p_v], dim=2)
        if self.use_rope and pe is not None:
            q, k = apply_rotary_emb(q, k, pe)

        # Build attention mask for [VL | SA | P]
        attn_mask = None
        if encoder_attention_mask is not None or physics_attention_mask is not None:
            # VL mask
            if encoder_attention_mask is not None:
                vl_mask = encoder_attention_mask  # [B, N_vl]
            else:
                vl_mask = torch.ones(batch_size, n_vl, device=q.device, dtype=q.dtype)
            # SA mask (always visible)
            sa_mask = torch.ones(batch_size, sa, device=q.device, dtype=vl_mask.dtype)
            # P mask
            if physics_attention_mask is not None:
                p_mask = physics_attention_mask[:, None].expand(-1, n_p).to(dtype=vl_mask.dtype)
            else:
                p_mask = torch.ones(batch_size, n_p, device=q.device, dtype=vl_mask.dtype)
            kv_mask = torch.cat([vl_mask, sa_mask, p_mask], dim=1)
            attn_mask = kv_mask[:, None, None, :]
            attn_mask = torch.where(
                attn_mask == 0,
                torch.tensor(float("-inf"), device=q.device, dtype=q.dtype),
                torch.tensor(0.0, device=q.device, dtype=q.dtype),
            )

        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        vl_attn = _merge_heads(attn_out[:, :, :n_vl])
        sa_attn = _merge_heads(attn_out[:, :, n_vl : n_vl + sa])
        p_attn = _merge_heads(attn_out[:, :, n_vl + sa :])

        # Residual + post-attn norm
        sa_tokens = sa_tokens + sa_mod1.gate * self.sa_norm2_attn(  # noqa: PLR6104
            self.sa_proj(self.dropout(sa_attn)),
        )
        vl_tokens = vl_tokens + vl_mod1.gate * self.vl_norm2_attn(  # noqa: PLR6104
            self.vl_proj(self.dropout(vl_attn)),
        )
        p_tokens = p_tokens + p_mod1.gate * self.p_norm2_attn(self.p_proj(self.dropout(p_attn)))  # noqa: PLR6104

        # MLP
        sa_tokens = sa_tokens + sa_mod2.gate * self.sa_norm3_mlp(  # noqa: PLR6104
            self.sa_mlp((1 + sa_mod2.scale) * self.sa_norm2_mlp(sa_tokens) + sa_mod2.shift),
        )
        vl_tokens = vl_tokens + vl_mod2.gate * self.vl_norm3_mlp(  # noqa: PLR6104
            self.vl_mlp((1 + vl_mod2.scale) * self.vl_norm2_mlp(vl_tokens) + vl_mod2.shift),
        )
        p_tokens = p_tokens + p_mod2.gate * self.p_norm3_mlp(  # noqa: PLR6104
            self.p_mlp((1 + p_mod2.scale) * self.p_norm2_mlp(p_tokens) + p_mod2.shift),
        )

        return sa_tokens, vl_tokens, p_tokens


class ExpandedSingleStreamBlock(SingleStreamBlock):
    """SingleStreamBlock extended with a parallel P (physics) stream.

    Inherits all parameters from SingleStreamBlock (same attribute names,
    so pretrained weights load directly). Adds parallel P stream parameters (p_*).

    When p_tokens is None, delegates to SingleStreamBlock.forward.
    When p_tokens is given, does joint attention [VL+SA | P] then separate outputs.
    """

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        hidden_size: int,
        p_dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        activation_fn: str = "gelu",
        attention_bias: bool = True,  # noqa: FBT001, FBT002
        norm_eps: float = 1e-6,
        qk_norm: str = "none",
        use_swiglu: bool = False,  # noqa: FBT001, FBT002
        positional_embeddings: str | None = None,
        max_seq_length: int | None = None,
        temb_type: str = "layerwise_mod",
        remove_bias: bool = False,  # noqa: FBT001, FBT002
        pre_norm: str = "layer_norm",
        post_norm: str = "none",
    ) -> None:
        """Initialize ExpandedSingleStreamBlock.

        Calls ``SingleStreamBlock.__init__`` for the VL+SA stream, then registers
        the P (physics) stream parameters.

        Args:
            hidden_size: Hidden dimension for the VL+SA stream.
            p_dim: Hidden dimension for the P (physics) stream.
            num_attention_heads: Number of joint attention heads.
            attention_head_dim: Dimension per attention head.
            mlp_ratio: MLP expansion ratio. Default ``4.0``.
            dropout: Dropout probability. Default ``0.0``.
            activation_fn: Activation function name (kept for API compatibility). Default ``"gelu"``.
            attention_bias: Whether to use bias in QKV/projection linears. Default ``True``.
            norm_eps: Epsilon for normalization layers. Default ``1e-6``.
            qk_norm: QK normalization strategy. Default ``"none"``.
            use_swiglu: Unused; kept for API compatibility. Default ``False``.
            positional_embeddings: Positional embedding type. Default ``None``.
            max_seq_length: Maximum sequence length. Default ``None``.
            temb_type: Timestep embedding injection strategy. Default ``"layerwise_mod"``.
            remove_bias: If ``True``, removes bias from modulation. Default ``False``.
            pre_norm: Pre-normalization layer type. Default ``"layer_norm"``.
            post_norm: Post-normalization layer type. Default ``"none"``.
        """
        super().__init__(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            activation_fn=activation_fn,
            attention_bias=attention_bias,
            norm_eps=norm_eps,
            qk_norm=qk_norm,
            use_swiglu=use_swiglu,
            positional_embeddings=positional_embeddings,
            max_seq_length=max_seq_length,
            temb_type=temb_type,
            remove_bias=remove_bias,
            pre_norm=pre_norm,
            post_norm=post_norm,
        )
        # ── P (physics) stream — mirrors SingleStreamBlock structure ──────
        self.p_dim = p_dim
        p_mlp_expanded = int(p_dim * mlp_ratio)
        self.p_mlp_hidden_dim = int(2 / 3 * p_mlp_expanded)
        p_mlp_in_dim = 2 * p_mlp_expanded
        self.p_mlp_proj = nn.Linear(p_mlp_expanded, self.p_mlp_hidden_dim, bias=not remove_bias)

        self.p_linear1 = nn.Linear(p_dim, self.inner_dim * 3 + p_mlp_in_dim, bias=attention_bias)
        self.p_linear2 = nn.Linear(
            self.inner_dim + self.p_mlp_hidden_dim,
            p_dim,
            bias=attention_bias,
        )
        self.p_q_norm, self.p_k_norm = create_qk_norm_layers(qk_norm, self.head_dim, norm_eps)
        self.p_pre_norm = create_norm_layer(pre_norm, p_dim, eps=norm_eps)
        self.p_post_norm = create_norm_layer(post_norm, p_dim, eps=norm_eps)

    def forward(  # type: ignore[override]  # noqa: PLR0914
        self,
        x: torch.Tensor,
        temb: torch.Tensor,
        pe: torch.Tensor | None = None,
        shared_modulation: ModulationOut | None = None,
        time_token: torch.Tensor | None = None,
        block_idx: int = 0,
        attn_mask: torch.Tensor | None = None,
        p_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Forward pass with optional joint attention [VL+SA | P].

        When ``p_tokens`` is ``None``, delegates to ``SingleStreamBlock.forward``.
        When ``p_tokens`` is provided, performs joint attention over both streams
        and returns updated ``x`` and ``p_tokens``.

        Args:
            x: Concatenated VL+SA tokens of shape ``(B, N, hidden_size)``.
            temb: Timestep embedding of shape ``(B, hidden_size)``.
            pe: RoPE complex frequencies as ``complex64`` when RoPE is enabled,
                else ``None``.
            shared_modulation: Pre-computed ``ModulationOut`` for
                ``temb_type="shared_mod"``.
            time_token: When not ``None``, signals identity modulation for the
                VL+SA stream.
            block_idx: Block index (reserved for future use). Default ``0``.
            attn_mask: Optional attention mask passed to
                ``F.scaled_dot_product_attention``.
            p_tokens: Optional P stream tokens of shape ``(B, N_p, p_dim)``. When
                ``None``, delegates to ``SingleStreamBlock.forward``.

        Returns:
            Updated ``x`` of shape ``(B, N, hidden_size)`` when ``p_tokens`` is
            ``None``, or tuple ``(x, p_tokens)`` when physics tokens are provided.
        """
        if p_tokens is None:
            return super().forward(
                x,
                temb,
                pe=pe,
                shared_modulation=shared_modulation,
                time_token=time_token,
                block_idx=block_idx,
                attn_mask=attn_mask,
            )

        # ── Joint attention [VL+SA | P] ──────────────────────────────────
        batch_size, n_x, _ = x.shape
        n_p = p_tokens.shape[1]

        # VL+SA modulation (same as SingleStreamBlock)
        use_identity_mod = time_token is not None
        if use_identity_mod:
            mod = ModulationOut(
                shift=torch.zeros(batch_size, 1, self.hidden_size, device=x.device, dtype=x.dtype),
                scale=torch.zeros(batch_size, 1, self.hidden_size, device=x.device, dtype=x.dtype),
                gate=torch.ones(batch_size, 1, self.hidden_size, device=x.device, dtype=x.dtype),
            )
            p_mod = ModulationOut(
                shift=torch.zeros(batch_size, 1, self.p_dim, device=p_tokens.device, dtype=p_tokens.dtype),
                scale=torch.zeros(batch_size, 1, self.p_dim, device=p_tokens.device, dtype=p_tokens.dtype),
                gate=torch.ones(batch_size, 1, self.p_dim, device=p_tokens.device, dtype=p_tokens.dtype),
            )
        elif self.temb_type == "shared_mod":
            mod = (
                shared_modulation
                if shared_modulation is not None
                else ModulationOut(
                    shift=torch.zeros(batch_size, 1, self.hidden_size, device=x.device, dtype=x.dtype),
                    scale=torch.zeros(batch_size, 1, self.hidden_size, device=x.device, dtype=x.dtype),
                    gate=torch.ones(batch_size, 1, self.hidden_size, device=x.device, dtype=x.dtype),
                )
            )
            p_mod = ModulationOut(
                shift=torch.zeros(batch_size, 1, self.p_dim, device=p_tokens.device, dtype=p_tokens.dtype),
                scale=torch.zeros(batch_size, 1, self.p_dim, device=p_tokens.device, dtype=p_tokens.dtype),
                gate=torch.ones(batch_size, 1, self.p_dim, device=p_tokens.device, dtype=p_tokens.dtype),
            )
        else:
            mod, _ = self.modulation(temb)
            p_mod = ModulationOut(
                shift=torch.zeros(batch_size, 1, self.p_dim, device=p_tokens.device, dtype=p_tokens.dtype),
                scale=torch.zeros(batch_size, 1, self.p_dim, device=p_tokens.device, dtype=p_tokens.dtype),
                gate=torch.ones(batch_size, 1, self.p_dim, device=p_tokens.device, dtype=p_tokens.dtype),
            )

        # VL+SA: pre-norm -> fused QKV+MLP
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        if self.pos_embed is not None:
            x_mod = self.pos_embed(x_mod)
        x_mlp_in_dim = 2 * int(self.hidden_size * self.mlp_ratio)
        x_qkv, x_mlp = torch.split(self.linear1(x_mod), [3 * self.inner_dim, x_mlp_in_dim], dim=-1)
        x_q, x_k, x_v = [_split_heads(t, self.num_heads) for t in x_qkv.chunk(3, dim=-1)]

        # P: pre-norm -> fused QKV+MLP
        p_mod_out = (1 + p_mod.scale) * self.p_pre_norm(p_tokens) + p_mod.shift
        p_mlp_in_dim = 2 * int(self.p_dim * self.mlp_ratio)
        p_qkv, p_mlp = torch.split(
            self.p_linear1(p_mod_out),
            [3 * self.inner_dim, p_mlp_in_dim],
            dim=-1,
        )
        p_q, p_k, p_v = [_split_heads(t, self.num_heads) for t in p_qkv.chunk(3, dim=-1)]

        # QK norm
        batch_size, head, _, d_head = x_q.shape
        x_q = self.q_norm(x_q.reshape(batch_size * head, n_x, d_head)).reshape(batch_size, head, n_x, d_head)
        x_k = self.k_norm(x_k.reshape(batch_size * head, n_x, d_head)).reshape(batch_size, head, n_x, d_head)
        p_q = self.p_q_norm(p_q.reshape(batch_size * head, n_p, d_head)).reshape(batch_size, head, n_p, d_head)
        p_k = self.p_k_norm(p_k.reshape(batch_size * head, n_p, d_head)).reshape(batch_size, head, n_p, d_head)

        # Joint attention [VL+SA | P]
        q = torch.cat([x_q, p_q], dim=2)
        k = torch.cat([x_k, p_k], dim=2)
        v = torch.cat([x_v, p_v], dim=2)
        if self.use_rope and pe is not None:
            q, k = apply_rotary_emb(q, k, pe)
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x_attn = _merge_heads(attn_out[:, :, :n_x])
        p_attn = _merge_heads(attn_out[:, :, n_x:])

        # VL+SA MLP
        mlp_x1, mlp_x2 = x_mlp.chunk(2, dim=-1)
        x_mlp_out = self.mlp_proj(F.silu(mlp_x1) * mlp_x2)
        x_out = self.linear2(torch.cat([x_attn, x_mlp_out], dim=-1))
        x = x + mod.gate * self.dropout(self.post_norm(x_out))  # noqa: PLR6104

        # P MLP
        p_mlp_x1, p_mlp_x2 = p_mlp.chunk(2, dim=-1)
        p_mlp_out = self.p_mlp_proj(F.silu(p_mlp_x1) * p_mlp_x2)
        p_out = self.p_linear2(torch.cat([p_attn, p_mlp_out], dim=-1))
        p_tokens = p_tokens + p_mod.gate * self.dropout(self.p_post_norm(p_out))  # noqa: PLR6104

        return x, p_tokens
