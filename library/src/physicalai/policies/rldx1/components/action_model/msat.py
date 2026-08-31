# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Vendored from RLWRLD/RLDX-1 (Apache-2.0)

"""MSAT: Multi-Stream Action Transformer (top-level orchestrator)."""

import logging
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from physicalai.policies.components.nn import TimestepEncoder
from physicalai.policies.rldx1.components.action_model.blocks import (
    DoubleStreamBlock,
    ExpandedDoubleStreamBlock,
    ExpandedSingleStreamBlock,
    Modulation,
    ModulationOut,
    SingleStreamBlock,
)
from physicalai.policies.rldx1.components.action_model.ops import RoPEEmbedder1D

logger = logging.getLogger(__name__)


class MSAT(nn.Module):
    """Multi-Stream Action Transformer.

    The model stacks lower multi-stream blocks followed by optional upper
    single-stream blocks, with optional physics conditioning support.
    Block-construction helpers and the internal forward paths are defined
    first, followed by the public ``__init__``/``forward`` API.
    """

    time_token_proj: nn.Linear | nn.Identity
    shared_single_mod_proj: nn.Linear | nn.Identity
    rope_embedder: RoPEEmbedder1D | nn.Identity
    vl_proj_to_sa: nn.Linear | nn.Identity

    def __init__(  # noqa: PLR0913, PLR0912, PLR0915, PLR0917
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        output_dim: int = 26,
        depth_multi_stream: int = 12,
        depth_single_stream: int = 0,
        dropout: float = 0.1,
        attention_bias: bool | None = None,  # noqa: FBT001
        norm_eps: float = 1e-6,
        compute_dtype: torch.dtype = torch.float32,
        final_dropout: bool = True,  # noqa: FBT001, FBT002, ARG002
        positional_embeddings: str | None = "rope_sa_only",
        action_model_max_seq_len: int = 512,
        sa_dim: int = 1536,
        vl_dim: int = 1536,
        qk_norm: str = "none",
        use_swiglu: bool = False,  # noqa: FBT001, FBT002, ARG002
        mlp_ratio: float = 4.0,
        vl_mlp_ratio: float | None = None,
        temb_type: str = "layerwise_mod",
        remove_bias: bool = False,  # noqa: FBT001, FBT002
        pre_norm: str = "layer_norm",
        post_norm: str = "none",
        rope_theta: float = 10000.0,
        gradient_checkpointing: bool = False,  # noqa: FBT001, FBT002
        use_physics: bool = False,  # noqa: FBT001, FBT002
        physics_dim: int = 0,
    ) -> None:
        """Initialize the Multi-Stream Action Transformer.

        Args:
            num_attention_heads: Number of attention heads per transformer block.
            attention_head_dim: Per-head channel dimension.
            output_dim: Output channel size for action (and physics) projections.
            depth_multi_stream: Number of lower multi-stream blocks.
            depth_single_stream: Number of upper single-stream blocks.
            dropout: Dropout probability used in attention/MLP branches.
            attention_bias: Whether attention-related linear layers use bias.
                If None, defaults to True unless ``remove_bias`` is enabled.
            norm_eps: Epsilon value for normalization layers.
            compute_dtype: Dtype used by the timestep encoder.
            final_dropout: Unused; accepted for config compatibility with sub-block signatures.
            positional_embeddings: Positional embedding mode. Supported values are
                ``"rope_sa_only"``, ``"rope_vl_sa"``, and ``None``.
                Default is ``"rope_sa_only"``.
            action_model_max_seq_len: Maximum sequence length for RoPE tables.
            sa_dim: State-action stream hidden size.
            vl_dim: Vision-language stream hidden size.
            qk_norm: Q/K normalization mode.
            use_swiglu: Unused; accepted for config compatibility with sub-block signatures.
            mlp_ratio: MLP expansion ratio for SA stream.
            vl_mlp_ratio: Optional MLP expansion ratio override for VL stream.
            temb_type: Timestep conditioning mode.
            remove_bias: If True, removes bias from modulation/projection layers and
                forces attention projections to be bias-free.
            pre_norm: Normalization type before attention/MLP.
            post_norm: Normalization type after attention/MLP.
            rope_theta: RoPE theta base.
            gradient_checkpointing: Whether to enable activation checkpointing
                in MSAT block loops during training.
            use_physics: Enables physics-conditioned architecture when True.
            physics_dim: Total physics signal dimension.

        Raises:
            NotImplementedError: If ``positional_embeddings`` is not one of the supported values.
        """
        super().__init__()
        if positional_embeddings not in {"rope_sa_only", "rope_vl_sa", None}:
            msg = ("Unsupported positional_embeddings. Use 'rope_sa_only', 'rope_vl_sa', or None.",)
            raise NotImplementedError(msg)
        self.use_physics = use_physics
        self.physics_dim = physics_dim
        self.inner_dim = num_attention_heads * attention_head_dim
        self.timestep_encoder = TimestepEncoder(
            embedding_dim=self.inner_dim,
            compute_dtype=compute_dtype,
        )
        self.positional_embeddings = positional_embeddings
        self.attention_head_dim = attention_head_dim
        self.temb_type = temb_type if temb_type is not None else "layerwise_mod"
        self.num_temb_tokens = 1  # Single time token when temb_type="input_token"
        self.gradient_checkpointing = gradient_checkpointing

        # Set default attention_bias if not provided (MSAT default: True)
        if attention_bias is None:
            attention_bias = True

        # If remove_bias=True, override attention_bias to False
        if remove_bias:
            attention_bias = False
            logger.info(
                "[MSAT] remove_bias=True: overriding attention_bias to False for all attention layers",
            )

        # Create time token projection if temb_type is "input_token"
        # Time token is always added right before action tokens: [VL | S | time_token | A]
        if self.temb_type == "input_token":
            # Double stream: VL, SA - time_token goes before SA (which contains action)
            if self.inner_dim != sa_dim:
                self.time_token_proj = nn.Linear(self.inner_dim, sa_dim, bias=not remove_bias)
            else:
                self.time_token_proj = nn.Identity()

            self.time_token_pos_emb = None
        else:
            self.time_token_proj = nn.Identity()
            self.time_token_pos_emb = None

        self.freq_encoder = None
        self.freq_token_proj = None
        self.num_freq_tokens = 0

        # Create shared modulation modules if temb_type is "shared_mod"
        if self.temb_type == "shared_mod":
            # Double stream: VL, SA
            self.shared_sa_mod = Modulation(self.inner_dim, double=True, remove_bias=remove_bias)
            self.shared_vl_mod = Modulation(self.inner_dim, double=True, remove_bias=remove_bias)

            # For SingleStreamBlocks: use inner_dim as input (same as temb), then project output
            self.shared_single_mod = Modulation(
                self.inner_dim,
                double=False,
                remove_bias=remove_bias,
            )
            if self.inner_dim != sa_dim:
                self.shared_single_mod_proj = nn.Linear(
                    self.inner_dim,
                    sa_dim,
                    bias=not remove_bias,
                )
            else:
                self.shared_single_mod_proj = nn.Identity()

        logger.info("Initializing MSAT...")

        # Initialize RoPE embedder if needed
        if positional_embeddings == "rope_sa_only":
            # RoPE for SA stream only (attention_head_dim assumed to be 64 below)
            # Axis 0 (dim=16): 0 (unused)
            # Axis 1 (dim=48): SA sequence position
            logger.info("[MSAT] RoPE theta: %s", rope_theta)
            self.rope_embedder = RoPEEmbedder1D(
                head_dim=attention_head_dim,
                axes_dim=[attention_head_dim // 4, attention_head_dim - attention_head_dim // 4],
                theta=rope_theta,
                max_seq_len=action_model_max_seq_len,
            )
            self.use_rope = True
        elif positional_embeddings == "rope_vl_sa":
            # RoPE for VL and SA streams (attention_head_dim assumed to be 64 below)
            # Axis 0 (dim=48): VL sequence position
            # Axis 1 (dim=16): SA sequence position
            self.rope_embedder = RoPEEmbedder1D(
                head_dim=attention_head_dim,
                axes_dim=[attention_head_dim - attention_head_dim // 4, attention_head_dim // 4],
                theta=rope_theta,
                max_seq_len=action_model_max_seq_len,
            )
            self.use_rope = True
        else:
            self.rope_embedder = nn.Identity()
            self.use_rope = False

        use_pos_emb = self.use_rope
        logger.info(
            "[MSAT] 'positional_embeddings' of MSAT: %s, action_model_max_seq_len: %s, enabled: %s",
            positional_embeddings,
            action_model_max_seq_len,
            use_pos_emb,
        )

        self.sa_dim = sa_dim
        self.vl_dim = vl_dim

        # VL→SA projection (used by both physics and non-physics paths)
        if sa_dim != vl_dim:
            self.vl_proj_to_sa = nn.Linear(vl_dim, sa_dim, bias=not remove_bias)
            logger.info("[MSAT] Projecting VL dimension from %s to %s", vl_dim, sa_dim)
        else:
            self.vl_proj_to_sa = nn.Identity()

        if self.use_physics and physics_dim > 0:
            # ── Physics-enabled architecture ──────────────────────────────────────
            # Lower: ExpandedDoubleStreamBlocks [VL | SA | P] — extends DoubleStreamBlock with P stream
            # Upper: ExpandedSingleStreamBlocks [VL+SA | P]  — extends SingleStreamBlock with P stream
            # Pretrained weights load directly (same attribute names as base blocks).
            logger.info("[MSAT] Physics mode: use_physics=True, physics_dim=%s", physics_dim)
            logger.info("[MSAT] Lower: %s ExpandedDoubleStreamBlocks [VL | SA | P]", depth_multi_stream)
            logger.info("[MSAT] Upper: %s ExpandedSingleStreamBlocks [VL+SA | P]", depth_single_stream)

            self.double_blocks = self._build_expanded_double_blocks(
                depth=depth_multi_stream,
                sa_dim=sa_dim,
                vl_dim=vl_dim,
                p_dim=sa_dim,  # Physics tokens projected to sa_dim by PhysicalSignalEncoder
                num_heads=num_attention_heads,
                head_dim=attention_head_dim,
                dropout=dropout,
                attention_bias=attention_bias,
                norm_eps=norm_eps,
                qk_norm=qk_norm,
                mlp_ratio=mlp_ratio,
                vl_mlp_ratio=vl_mlp_ratio,
                positional_embeddings=positional_embeddings,
                max_seq_length=action_model_max_seq_len,
                temb_type=self.temb_type,
                remove_bias=remove_bias,
                pre_norm=pre_norm,
                post_norm=post_norm,
            )

            single_stream_hidden_size = sa_dim
            self.single_blocks = self._build_expanded_single_blocks(
                depth=depth_single_stream,
                hidden_size=single_stream_hidden_size,
                p_dim=sa_dim,
                num_heads=num_attention_heads,
                head_dim=attention_head_dim,
                dropout=dropout,
                attention_bias=attention_bias,
                norm_eps=norm_eps,
                qk_norm=qk_norm,
                mlp_ratio=mlp_ratio,
                positional_embeddings=positional_embeddings,
                max_seq_length=action_model_max_seq_len,
                temb_type=self.temb_type,
                remove_bias=remove_bias,
                pre_norm=pre_norm,
                post_norm=post_norm,
            )

            has_single_blocks = depth_single_stream > 0
            sa_hidden_dim = single_stream_hidden_size if has_single_blocks else sa_dim
            self._sa_hidden_dim = sa_hidden_dim

            # Physics output projection (AdaLN-zero style)
            self.norm_out_physics = nn.LayerNorm(sa_dim, elementwise_affine=False, eps=1e-6)
            self.proj_out_physics_1 = nn.Linear(self.inner_dim, 2 * sa_dim, bias=not remove_bias)
            self.proj_out_physics_2 = nn.Linear(sa_dim, output_dim, bias=not remove_bias)

        else:
            # ── Standard architecture (no physics) ────────────────────────────────
            self.double_blocks = self._build_double_blocks(
                depth=depth_multi_stream,
                sa_dim=sa_dim,
                vl_dim=vl_dim,
                num_heads=num_attention_heads,
                head_dim=attention_head_dim,
                dropout=dropout,
                attention_bias=attention_bias,
                norm_eps=norm_eps,
                qk_norm=qk_norm,
                mlp_ratio=mlp_ratio,
                vl_mlp_ratio=vl_mlp_ratio,
                positional_embeddings=positional_embeddings,
                max_seq_length=action_model_max_seq_len,
                temb_type=self.temb_type,
                remove_bias=remove_bias,
                pre_norm=pre_norm,
                post_norm=post_norm,
            )

            single_stream_hidden_size = sa_dim
            self.single_blocks = self._build_single_blocks(
                depth=depth_single_stream,
                hidden_size=single_stream_hidden_size,
                num_heads=num_attention_heads,
                head_dim=attention_head_dim,
                dropout=dropout,
                attention_bias=attention_bias,
                norm_eps=norm_eps,
                qk_norm=qk_norm,
                mlp_ratio=mlp_ratio,
                positional_embeddings=positional_embeddings,
                max_seq_length=action_model_max_seq_len,
                temb_type=self.temb_type,
                remove_bias=remove_bias,
                pre_norm=pre_norm,
                post_norm=post_norm,
            )

            has_single_blocks = depth_single_stream > 0
            sa_hidden_dim = single_stream_hidden_size if has_single_blocks else sa_dim
            self._sa_hidden_dim = sa_hidden_dim

        # Output projection (AdaLN-zero style) - for action prediction
        self.norm_out = nn.LayerNorm(sa_hidden_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(self.inner_dim, 2 * sa_hidden_dim, bias=not remove_bias)
        self.proj_out_2 = nn.Linear(sa_hidden_dim, output_dim, bias=not remove_bias)
        logger.info("[MSAT] Output projection: sa_hidden_dim=%s -> output_dim=%s", sa_hidden_dim, output_dim)

        self._remove_bias = remove_bias

        logger.info(
            "[MSAT] Total number of MSAT parameters: %d",
            sum(p.numel() for p in self.parameters() if p.requires_grad),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # SA tokens
        encoder_hidden_states: torch.Tensor,  # VL tokens
        timestep: torch.LongTensor | None = None,
        return_all_hidden_states: bool = False,  # noqa: FBT001, FBT002
        encoder_attention_mask: torch.Tensor | None = None,  # [B, N_vl] VL attention mask (1=visible, 0=masked)
        physics_embs: torch.Tensor | None = None,  # [B, N_p, sa_dim] Physics tokens (when use_physics=True)
        physics_attention_mask: torch.Tensor | None = None,  # [B] per-sample physics mask (1=visible, 0=masked)
    ) -> Any:  # noqa: ANN401
        """Run a forward pass through MSAT.

        Args:
            hidden_states: State-action tokens ``[B, N_sa, sa_dim]``.
            encoder_hidden_states: Vision-language tokens ``[B, N_vl, vl_dim]``.
            timestep: Diffusion timestep tensor ``[B]``.
            return_all_hidden_states: If True, return intermediate SA hidden states.
            encoder_attention_mask: Optional VL visibility mask ``[B, N_vl]``.
            physics_embs: Optional physics tokens ``[B, N_p, sa_dim]``.
            physics_attention_mask: Optional per-sample physics visibility mask ``[B]``.

        Returns:
            Action output tensor or physics-aware output dict depending on
            configuration. Returns ``(output, hidden_states)`` when
            ``return_all_hidden_states=True``.
        """
        return self._forward_inner(
            hidden_states,
            encoder_hidden_states,
            timestep,
            return_all_hidden_states,
            encoder_attention_mask,
            physics_embs,
            physics_attention_mask,
        )

    def _apply_checkpoint(self, func: Callable[..., Any], *args: Any) -> Any:  # noqa: ANN401
        """Apply gradient checkpointing if enabled, matching Pi05Model's convention.

        Args:
            func: The function to apply checkpointing to.
            args: Arguments to pass to the function.

        Returns:
            The output of the function, with checkpointing applied if enabled.
        """
        if self.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                func,
                *args,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        return func(*args)

    @staticmethod
    def _build_double_blocks(  # noqa: PLR0913, PLR0917
        depth: int,
        sa_dim: int,
        vl_dim: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
        attention_bias: bool,  # noqa: FBT001
        norm_eps: float,
        qk_norm: str = "none",
        mlp_ratio: float = 4.0,
        vl_mlp_ratio: float | None = None,
        positional_embeddings: str | None = None,
        max_seq_length: int | None = None,
        temb_type: str = "layerwise_mod",
        remove_bias: bool = False,  # noqa: FBT001, FBT002
        pre_norm: str = "layer_norm",
        post_norm: str = "none",
    ) -> nn.ModuleList:
        """Build lower double-stream blocks.

        Args:
            depth: Number of blocks.
            sa_dim: State-action stream width.
            vl_dim: Vision-language stream width.
            num_heads: Number of attention heads.
            head_dim: Per-head attention dimension.
            dropout: Dropout probability.
            attention_bias: Whether attention projections use bias.
            norm_eps: Normalization epsilon.
            qk_norm: Q/K normalization mode.
            mlp_ratio: SA MLP expansion ratio.
            vl_mlp_ratio: Optional VL MLP ratio override.
            positional_embeddings: Positional embedding mode.
            max_seq_length: Maximum sequence length for positional encoding.
            temb_type: Timestep conditioning strategy.
            remove_bias: Whether to disable bias in selected projections.
            pre_norm: Pre-normalization type.
            post_norm: Post-normalization type.

        Returns:
            ModuleList of configured :class:`DoubleStreamBlock` instances.
        """
        return nn.ModuleList(
            [
                DoubleStreamBlock(
                    sa_dim=sa_dim,
                    vl_dim=vl_dim,
                    num_attention_heads=num_heads,
                    attention_head_dim=head_dim,
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
                for _ in range(depth)
            ],
        )

    @staticmethod
    def _build_single_blocks(  # noqa: PLR0913, PLR0917
        depth: int,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
        activation_fn: str = "gelu",
        attention_bias: bool = True,  # noqa: FBT001, FBT002
        norm_eps: float = 1e-6,
        qk_norm: str = "none",
        use_swiglu: bool = False,  # noqa: FBT001, FBT002
        mlp_ratio: float = 4.0,
        positional_embeddings: str | None = None,
        max_seq_length: int | None = None,
        temb_type: str = "layerwise_mod",
        remove_bias: bool = False,  # noqa: FBT001, FBT002
        pre_norm: str = "layer_norm",
        post_norm: str = "none",
    ) -> nn.ModuleList:
        """Build upper single-stream blocks.

        Args:
            depth: Number of blocks.
            hidden_size: Token width for the single stream.
            num_heads: Number of attention heads.
            head_dim: Per-head attention dimension.
            dropout: Dropout probability.
            activation_fn: Activation name for compatibility with upstream configs.
            attention_bias: Whether attention projections use bias.
            norm_eps: Normalization epsilon.
            qk_norm: Q/K normalization mode.
            use_swiglu: Whether SwiGLU MLP is enabled.
            mlp_ratio: MLP expansion ratio.
            positional_embeddings: Positional embedding mode.
            max_seq_length: Maximum sequence length for positional encoding.
            temb_type: Timestep conditioning strategy.
            remove_bias: Whether to disable bias in selected projections.
            pre_norm: Pre-normalization type.
            post_norm: Post-normalization type.

        Returns:
            ModuleList of configured :class:`SingleStreamBlock` instances.
        """
        return nn.ModuleList(
            [
                SingleStreamBlock(
                    hidden_size=hidden_size,
                    num_attention_heads=num_heads,
                    attention_head_dim=head_dim,
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
                for _ in range(depth)
            ],
        )

    @staticmethod
    def _build_expanded_double_blocks(  # noqa: PLR0913, PLR0917
        depth: int,
        sa_dim: int,
        vl_dim: int,
        p_dim: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
        activation_fn: str = "gelu",
        attention_bias: bool = True,  # noqa: FBT001, FBT002
        norm_eps: float = 1e-6,
        qk_norm: str = "none",
        use_swiglu: bool = False,  # noqa: FBT001, FBT002
        mlp_ratio: float = 4.0,
        vl_mlp_ratio: float | None = None,
        p_mlp_ratio: float | None = None,
        positional_embeddings: str | None = None,
        max_seq_length: int | None = None,
        temb_type: str = "layerwise_mod",
        remove_bias: bool = False,  # noqa: FBT001, FBT002
        pre_norm: str = "layer_norm",
        post_norm: str = "none",
    ) -> nn.ModuleList:
        """Build physics-aware lower expanded double-stream blocks.

        Args:
            depth: Number of blocks.
            sa_dim: State-action stream width.
            vl_dim: Vision-language stream width.
            p_dim: Physics stream width.
            num_heads: Number of attention heads.
            head_dim: Per-head attention dimension.
            dropout: Dropout probability.
            activation_fn: Activation name for compatibility with upstream configs.
            attention_bias: Whether attention projections use bias.
            norm_eps: Normalization epsilon.
            qk_norm: Q/K normalization mode.
            use_swiglu: Whether SwiGLU MLP is enabled.
            mlp_ratio: SA MLP expansion ratio.
            vl_mlp_ratio: Optional VL MLP ratio override.
            p_mlp_ratio: Optional physics MLP ratio override.
            positional_embeddings: Positional embedding mode.
            max_seq_length: Maximum sequence length for positional encoding.
            temb_type: Timestep conditioning strategy.
            remove_bias: Whether to disable bias in selected projections.
            pre_norm: Pre-normalization type.
            post_norm: Post-normalization type.

        Returns:
            ModuleList of configured :class:`ExpandedDoubleStreamBlock` instances.
        """
        return nn.ModuleList(
            [
                ExpandedDoubleStreamBlock(
                    sa_dim=sa_dim,
                    vl_dim=vl_dim,
                    p_dim=p_dim,
                    num_attention_heads=num_heads,
                    attention_head_dim=head_dim,
                    mlp_ratio=mlp_ratio,
                    vl_mlp_ratio=vl_mlp_ratio,
                    p_mlp_ratio=p_mlp_ratio,
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
                for _ in range(depth)
            ],
        )

    @staticmethod
    def _build_expanded_single_blocks(  # noqa: PLR0913, PLR0917
        depth: int,
        hidden_size: int,
        p_dim: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
        activation_fn: str = "gelu",
        attention_bias: bool = True,  # noqa: FBT001, FBT002
        norm_eps: float = 1e-6,
        qk_norm: str = "none",
        use_swiglu: bool = False,  # noqa: FBT001, FBT002
        mlp_ratio: float = 4.0,
        positional_embeddings: str | None = None,
        max_seq_length: int | None = None,
        temb_type: str = "layerwise_mod",
        remove_bias: bool = False,  # noqa: FBT001, FBT002
        pre_norm: str = "layer_norm",
        post_norm: str = "none",
    ) -> nn.ModuleList:
        """Build physics-aware upper expanded single-stream blocks.

        Args:
            depth: Number of blocks.
            hidden_size: Token width for the joint stream.
            p_dim: Physics stream width.
            num_heads: Number of attention heads.
            head_dim: Per-head attention dimension.
            dropout: Dropout probability.
            activation_fn: Activation name for compatibility with upstream configs.
            attention_bias: Whether attention projections use bias.
            norm_eps: Normalization epsilon.
            qk_norm: Q/K normalization mode.
            use_swiglu: Whether SwiGLU MLP is enabled.
            mlp_ratio: MLP expansion ratio.
            positional_embeddings: Positional embedding mode.
            max_seq_length: Maximum sequence length for positional encoding.
            temb_type: Timestep conditioning strategy.
            remove_bias: Whether to disable bias in selected projections.
            pre_norm: Pre-normalization type.
            post_norm: Post-normalization type.

        Returns:
            ModuleList of configured :class:`ExpandedSingleStreamBlock` instances.
        """
        return nn.ModuleList(
            [
                ExpandedSingleStreamBlock(
                    hidden_size=hidden_size,
                    p_dim=p_dim,
                    num_attention_heads=num_heads,
                    attention_head_dim=head_dim,
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
                for _ in range(depth)
            ],
        )

    def _forward_inner(  # noqa: C901, PLR0912, PLR0914, PLR0915
        self,
        sa_embs: torch.Tensor,
        vl_embs: torch.Tensor,
        timesteps: torch.LongTensor | None,
        return_all_hidden_states: bool = False,  # noqa: FBT001, FBT002
        encoder_attention_mask: torch.Tensor | None = None,
        physics_embs: torch.Tensor | None = None,
        physics_attention_mask: torch.Tensor | None = None,
    ) -> Any:  # noqa: ANN401
        """Run the internal MSAT forward pass.

        This method dispatches to the standard two-stream path or the
        physics-enabled path depending on ``self.use_physics`` and whether
        physics tokens are provided.

        Args:
            sa_embs: State-action tokens with shape ``[B, N_sa, D_sa]``.
            vl_embs: Vision-language tokens with shape ``[B, N_vl, D_vl]``.
            timesteps: Diffusion timesteps with shape ``[B]``.
            return_all_hidden_states: If True, also return intermediate SA hidden states.
            encoder_attention_mask: Optional VL visibility mask ``[B, N_vl]``.
            physics_embs: Optional physics tokens ``[B, N_p, D_sa]``.
            physics_attention_mask: Optional per-sample physics visibility mask ``[B]``.

        Returns:
            If physics is disabled: action tensor ``[B, N_action, output_dim]``.
            If physics is enabled: dict with ``"action"`` and ``"physics"`` outputs.
            If ``return_all_hidden_states`` is True, returns ``(output, hidden_states)``.

        Raises:
            ValueError: If ``has_time_token`` is True but ``time_token`` is None.
        """
        temb = self.timestep_encoder(timesteps)

        # Create Time Token
        time_token = None
        has_time_token = False
        if self.temb_type == "input_token":
            # 1. Projection: (B, D) -> (B, 1, D)
            t_emb = self.time_token_proj(temb).unsqueeze(1)

            # 2. Replication: (B, 1, D) -> (B, N, D)
            time_token = t_emb.repeat(1, self.num_temb_tokens, 1)
            has_time_token = True

        # Track VL length for Single Stream Block RoPE calculation
        n_vl_for_single = None

        # Generate Shared Modulations
        shared_modulations = None
        shared_single_modulation = None
        if self.temb_type == "shared_mod":
            sa_mod1_raw, sa_mod2_raw = self.shared_sa_mod(temb)
            vl_mod1_raw, vl_mod2_raw = self.shared_vl_mod(temb)
            shared_modulations = {
                "sa_mod1_raw": sa_mod1_raw,
                "sa_mod2_raw": sa_mod2_raw,
                "vl_mod1_raw": vl_mod1_raw,
                "vl_mod2_raw": vl_mod2_raw,
            }
            shared_single_mod_raw, _ = self.shared_single_mod(temb)
            if hasattr(self, "shared_single_mod_proj"):
                shared_single_modulation = ModulationOut(
                    shift=self.shared_single_mod_proj(shared_single_mod_raw.shift),
                    scale=self.shared_single_mod_proj(shared_single_mod_raw.scale),
                    gate=self.shared_single_mod_proj(shared_single_mod_raw.gate),
                )
            else:
                shared_single_modulation = shared_single_mod_raw

        sa, vl = sa_embs.contiguous(), vl_embs.contiguous()

        # ══════════════════════════════════════════════════════════════════════
        # Physics-enabled path: ExpandedDouble [VL | SA | P] -> ExpandedSingle [VL+SA | P]
        # ══════════════════════════════════════════════════════════════════════
        if self.use_physics and physics_embs is not None:
            return self._forward_physics(
                sa,
                vl,
                physics_embs,
                temb,
                time_token,
                has_time_token,
                return_all_hidden_states,
                shared_modulations,
                shared_single_modulation,
                encoder_attention_mask,
                physics_attention_mask,
            )

        # ══════════════════════════════════════════════════════════════════════
        # Standard path: DoubleStreamBlocks [VL | SA] -> SingleStreamBlocks [VL_proj | SA]
        # ══════════════════════════════════════════════════════════════════════

        # Prepend time_token to SA: [time_token | S | A]
        if has_time_token:
            if time_token is None:
                msg = "Time token is None while has_time_token is True."
                raise ValueError(msg)
            sa = torch.cat([time_token, sa], dim=1)

        all_hidden = [sa]

        pe = None
        if self.use_rope:
            batch_size, n_vl = vl.shape[0], vl.shape[1]
            n_sa_total = sa.shape[1]
            device = sa.device

            total_len = n_vl + n_sa_total
            ids = torch.zeros(batch_size, total_len, 2, dtype=torch.long, device=device)
            sa_stream_start_idx = n_vl

            if self.positional_embeddings == "rope_sa_only":
                # Position ID assignment: (time_token) | S | A
                current_idx = sa_stream_start_idx
                if has_time_token:
                    # Time token: axis 1 = 0..num_temb_tokens-1
                    ids[:, current_idx : current_idx + self.num_temb_tokens, 1] = (
                        torch.arange(self.num_temb_tokens, device=device).unsqueeze(0).expand(batch_size, -1)
                    )
                    current_idx += self.num_temb_tokens
                # SA tokens (S | A): axis 1 = starting from (num_temb_tokens)
                sa_len = n_sa_total - (self.num_temb_tokens if has_time_token else 0)
                start_pos = self.num_temb_tokens if has_time_token else 0
                ids[:, current_idx:, 1] = (
                    torch.arange(start_pos, start_pos + sa_len, device=device).unsqueeze(0).expand(batch_size, -1)
                )
            elif self.positional_embeddings == "rope_vl_sa":
                ids[:, :n_vl, 0] = torch.arange(n_vl, device=device).unsqueeze(0).expand(batch_size, -1)
                # Position ID assignment: (time_token) | S | A
                current_idx = sa_stream_start_idx
                if has_time_token:
                    # Time token: axis 1 = 0..num_temb_tokens-1
                    ids[:, current_idx : current_idx + self.num_temb_tokens, 1] = (
                        torch.arange(self.num_temb_tokens, device=device).unsqueeze(0).expand(batch_size, -1)
                    )
                    current_idx += self.num_temb_tokens
                # SA tokens (S | A): axis 1 = starting from (num_temb_tokens)
                sa_len = n_sa_total - (self.num_temb_tokens if has_time_token else 0)
                start_pos = (self.num_temb_tokens if has_time_token else 0) + 1
                ids[:, current_idx:, 1] = (
                    torch.arange(start_pos, start_pos + sa_len, device=device).unsqueeze(0).expand(batch_size, -1)
                )

            pe = self.rope_embedder(ids)

        # Track block index
        block_idx = 0
        for blk in self.double_blocks:

            def _run_double(
                sa_tokens: torch.Tensor,
                vl_tokens: torch.Tensor,
                _blk: nn.Module = blk,
                _block_idx: int = block_idx,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                return _blk(
                    sa_tokens,
                    vl_tokens,
                    temb,
                    pe=pe,
                    shared_modulations=shared_modulations,
                    has_time_token=has_time_token,
                    block_idx=_block_idx,
                    encoder_attention_mask=encoder_attention_mask,
                )

            sa, vl = self._apply_checkpoint(_run_double, sa, vl)
            all_hidden.append(sa)
            block_idx += 1

        # Separate time_token before single stream block
        if has_time_token:
            time_token = sa[:, : self.num_temb_tokens, :]
            sa = sa[:, self.num_temb_tokens :, :]

        if len(self.single_blocks) > 0:
            vl_projected = self.vl_proj_to_sa(vl)
            # Track VL length for Single Stream Block RoPE calculation
            n_vl_for_single = vl.shape[1]

            # Re-concat with updated time_token: VL | (time_token) | S | A
            if has_time_token:
                if time_token is None:
                    msg = "Time token is None while has_time_token is True."
                    raise ValueError(msg)
                x = torch.cat([vl_projected, time_token, sa], dim=1)
            else:
                x = torch.cat([vl_projected, sa], dim=1)

            # Single Stream Blocks
            pe_single = None
            if self.use_rope:
                b_single = x.shape[0]
                n_total = x.shape[1]
                device_single = x.device

                n_action_pure = sa.shape[1]
                action_start_idx_in_x = n_total - n_action_pure

                # 2D RoPE
                ids_single = torch.zeros(b_single, n_total, 2, dtype=torch.long, device=device_single)

                if self.positional_embeddings == "rope_sa_only":
                    current_idx = n_vl_for_single
                    if has_time_token:
                        # Time token: axis 1 = 0..num_temb_tokens-1
                        ids_single[:, current_idx : current_idx + self.num_temb_tokens, 1] = (
                            torch.arange(self.num_temb_tokens, device=device_single).unsqueeze(0).expand(b_single, -1)
                        )
                        current_idx += self.num_temb_tokens
                    # Action positions: axis 1 = sequence position starting from (num_temb_tokens)
                    start_pos = (self.num_temb_tokens if has_time_token else 0) + 1
                    ids_single[:, action_start_idx_in_x:, 1] = (
                        torch.arange(start_pos, start_pos + n_action_pure, device=device_single)
                        .unsqueeze(0)
                        .expand(b_single, -1)
                    )
                elif self.positional_embeddings == "rope_vl_sa":
                    # VL positions: axis 0 = sequence position
                    ids_single[:, :n_vl_for_single, 0] = (
                        torch.arange(n_vl_for_single, device=device_single).unsqueeze(0).expand(b_single, -1)
                    )
                    # Context tokens: (time_token) | S | A
                    current_idx = n_vl_for_single
                    if has_time_token:
                        # Time token: axis 1 = 0..num_temb_tokens-1
                        ids_single[:, current_idx : current_idx + self.num_temb_tokens, 1] = (
                            torch.arange(self.num_temb_tokens, device=device_single).unsqueeze(0).expand(b_single, -1)
                        )
                        current_idx += self.num_temb_tokens
                    # Action positions: axis 1 = sequence position starting from (num_temb_tokens)
                    start_pos = (self.num_temb_tokens if has_time_token else 0) + 1
                    ids_single[:, action_start_idx_in_x:, 1] = (
                        torch.arange(start_pos, start_pos + n_action_pure, device=device_single)
                        .unsqueeze(0)
                        .expand(b_single, -1)
                    )

                pe_single = self.rope_embedder(ids_single)

            # Build single-stream attention mask from encoder_attention_mask
            single_attn_mask = None
            if encoder_attention_mask is not None:
                b_mask = x.shape[0]
                n_x = x.shape[1]
                n_vl_mask = n_vl_for_single
                n_rest = n_x - n_vl_mask
                rest_mask = torch.ones(
                    b_mask,
                    n_rest,
                    device=x.device,
                    dtype=encoder_attention_mask.dtype,
                )
                kv_mask = torch.cat([encoder_attention_mask, rest_mask], dim=1)  # [B, N_x]
                single_attn_mask = kv_mask[:, None, None, :]  # [B, 1, 1, N_x]
                single_attn_mask = torch.where(
                    single_attn_mask == 0,
                    torch.tensor(float("-inf"), device=x.device, dtype=x.dtype),
                    torch.tensor(0.0, device=x.device, dtype=x.dtype),
                )

            # block_idx already tracks the number of DoubleStreamBlocks processed
            for blk in self.single_blocks:

                def _run_single(
                    x_tokens: torch.Tensor,
                    _blk: nn.Module = blk,
                    _block_idx: int = block_idx,
                ) -> torch.Tensor:
                    return _blk(
                        x_tokens,
                        temb,
                        pe=pe_single if pe_single is not None else pe,
                        shared_modulation=shared_single_modulation,
                        time_token=time_token if has_time_token else None,
                        block_idx=_block_idx,
                        attn_mask=single_attn_mask,
                    )

                x = self._apply_checkpoint(_run_single, x)
                block_idx += 1

            # Extract Action Part
            n_action_pure = sa.shape[1]
            sa = x[:, -n_action_pure:, :]

        out = self._output_projection(sa, temb)

        if return_all_hidden_states:
            return out, all_hidden
        return out

    def _forward_physics(  # noqa: C901, PLR0912, PLR0914, PLR0915
        self,
        sa: torch.Tensor,
        vl: torch.Tensor,
        p_embs: torch.Tensor,
        temb: torch.Tensor,
        time_token: torch.Tensor | None,
        has_time_token: bool,  # noqa: FBT001
        return_all_hidden_states: bool,  # noqa: FBT001
        shared_modulations: dict[str, torch.Tensor] | None,
        shared_single_modulation: ModulationOut | None,
        encoder_attention_mask: torch.Tensor | None,
        physics_attention_mask: torch.Tensor | None = None,
    ) -> Any:  # noqa: ANN401
        """Run the physics-enabled MSAT forward path.

        The lower stage applies expanded double-stream blocks over
        ``[VL | SA | P]``. The upper stage applies expanded single-stream blocks
        over ``[VL+SA | P]``.

        Args:
            sa: State-action tokens.
            vl: Vision-language tokens.
            p_embs: Physics tokens.
            temb: Timestep embedding tensor.
            time_token: Optional time token tensor.
            has_time_token: Whether ``time_token`` is active.
            return_all_hidden_states: If True, include intermediate SA states.
            shared_modulations: Optional shared modulation tensors for double-stream blocks.
            shared_single_modulation: Optional shared modulation tensor for single-stream blocks.
            encoder_attention_mask: Optional VL visibility mask.
            physics_attention_mask: Optional per-sample physics visibility mask.

        Returns:
            Dict with ``"action"`` and ``"physics"`` outputs, or
            ``(output_dict, hidden_states)`` when ``return_all_hidden_states=True``.

        Raises:
            ValueError: If ``has_time_token`` is True but ``time_token`` is None.
        """
        p = p_embs.contiguous()

        # Prepend time_token to SA only (physics stream is not diffusion-based)
        if has_time_token:
            if time_token is None:
                msg = "Time token is None while has_time_token is True."
                raise ValueError(msg)
            sa = torch.cat([time_token, sa], dim=1)

        all_hidden = [sa]

        # ── RoPE for lower blocks: [VL | SA | P] ────────────────────────
        pe = None
        if self.use_rope:
            batch_size = sa.shape[0]
            n_vl = vl.shape[1]
            n_sa = sa.shape[1]
            n_p = p.shape[1]
            device = sa.device

            total_len = n_vl + n_sa + n_p
            ids = torch.zeros(batch_size, total_len, 2, dtype=torch.long, device=device)
            sa_start = n_vl

            # Sequence layout: [VL (N_vl) | SA (N_sa) | P (N_p)]
            p_start = n_vl + n_sa

            if self.positional_embeddings == "rope_sa_only":
                # SA positions on axis1
                current_idx = sa_start
                if has_time_token:
                    ids[:, current_idx : current_idx + self.num_temb_tokens, 1] = (
                        torch.arange(self.num_temb_tokens, device=device).unsqueeze(0).expand(batch_size, -1)
                    )
                    current_idx += self.num_temb_tokens
                sa_len = n_sa - (self.num_temb_tokens if has_time_token else 0)
                start_pos = self.num_temb_tokens if has_time_token else 0
                ids[:, current_idx : current_idx + sa_len, 1] = (
                    torch.arange(start_pos, start_pos + sa_len, device=device).unsqueeze(0).expand(batch_size, -1)
                )
                # P positions: axis0=1 (distinguish from SA), axis1=sequential
                ids[:, p_start:, 0] = 1
                ids[:, p_start:, 1] = torch.arange(n_p, device=device).unsqueeze(0).expand(batch_size, -1)
            elif self.positional_embeddings == "rope_vl_sa":
                ids[:, :n_vl, 0] = torch.arange(n_vl, device=device).unsqueeze(0).expand(batch_size, -1)
                # SA positions on axis1
                current_idx = sa_start
                if has_time_token:
                    ids[:, current_idx : current_idx + self.num_temb_tokens, 1] = (
                        torch.arange(self.num_temb_tokens, device=device).unsqueeze(0).expand(batch_size, -1)
                    )
                    current_idx += self.num_temb_tokens
                sa_len = n_sa - (self.num_temb_tokens if has_time_token else 0)
                start_pos = (self.num_temb_tokens if has_time_token else 0) + 1
                ids[:, current_idx : current_idx + sa_len, 1] = (
                    torch.arange(start_pos, start_pos + sa_len, device=device).unsqueeze(0).expand(batch_size, -1)
                )
                # P positions: axis0=1 (distinguish from SA/VL), axis1=sequential
                ids[:, p_start:, 0] = 1
                ids[:, p_start:, 1] = torch.arange(n_p, device=device).unsqueeze(0).expand(batch_size, -1)

            pe = self.rope_embedder(ids)

        # ── Lower: Triple blocks [VL | SA | P] ───────────────────────────
        block_idx = 0
        for blk in self.double_blocks:

            def _run_expanded_double(
                sa_tokens: torch.Tensor,
                vl_tokens: torch.Tensor,
                p_tokens: torch.Tensor,
                _blk: nn.Module = blk,
                _block_idx: int = block_idx,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                return _blk(
                    sa_tokens,
                    vl_tokens,
                    temb,
                    pe=pe,
                    shared_modulations=shared_modulations,
                    has_time_token=has_time_token,
                    block_idx=_block_idx,
                    encoder_attention_mask=encoder_attention_mask,
                    p_tokens=p_tokens,
                    physics_attention_mask=physics_attention_mask,
                )

            sa, vl, p = self._apply_checkpoint(_run_expanded_double, sa, vl, p)
            all_hidden.append(sa)
            block_idx += 1

        # Strip time tokens from SA
        if has_time_token:
            time_token = sa[:, : self.num_temb_tokens, :]
            sa = sa[:, self.num_temb_tokens :, :]

        # ── Upper: ExpandedSingleStreamBlocks with p_tokens ───────────────
        if len(self.single_blocks) > 0:
            vl_projected = self.vl_proj_to_sa(vl)
            n_vl_for_single = vl.shape[1]
            if has_time_token:
                if time_token is None:
                    msg = "Time token is None while has_time_token is True."
                    raise ValueError(msg)
                x = torch.cat([vl_projected, time_token, sa], dim=1)
            else:
                x = torch.cat([vl_projected, sa], dim=1)

            # RoPE for upper blocks: [SA+VL (N_x) | P (N_p)]
            pe_single = None
            if self.use_rope:
                batch_size = x.shape[0]
                n_x = x.shape[1]
                n_p = p.shape[1]
                ids_s = torch.zeros(batch_size, n_x + n_p, 2, dtype=torch.long, device=x.device)

                if self.positional_embeddings in {"rope_sa_only", "rope_vl_sa"}:
                    # SA+VL stream positions (same as standard SingleStreamBlock RoPE)
                    if self.positional_embeddings == "rope_vl_sa":
                        ids_s[:, :n_vl_for_single, 0] = (
                            torch.arange(n_vl_for_single, device=x.device).unsqueeze(0).expand(batch_size, -1)
                        )
                    current_idx = n_vl_for_single
                    if has_time_token:
                        ids_s[:, current_idx : current_idx + self.num_temb_tokens, 1] = (
                            torch.arange(self.num_temb_tokens, device=x.device).unsqueeze(0).expand(batch_size, -1)
                        )
                        current_idx += self.num_temb_tokens
                    sa_pure_len = sa.shape[1]
                    start_pos = (self.num_temb_tokens if has_time_token else 0) + 1
                    ids_s[:, current_idx : current_idx + sa_pure_len, 1] = (
                        torch.arange(start_pos, start_pos + sa_pure_len, device=x.device)
                        .unsqueeze(0)
                        .expand(batch_size, -1)
                    )

                    # P stream positions: axis0=1 (distinguish from SA+VL), axis1=sequential
                    p_section_start = n_x
                    ids_s[:, p_section_start:, 0] = 1
                    ids_s[:, p_section_start:, 1] = (
                        torch.arange(n_p, device=x.device).unsqueeze(0).expand(batch_size, -1)
                    )

                pe_single = self.rope_embedder(ids_s)

            # Build single-stream attention mask covering [VL+SA | P]
            single_attn_mask = None
            if encoder_attention_mask is not None or physics_attention_mask is not None:
                b_mask = x.shape[0]
                n_x_mask = x.shape[1]
                n_p_mask = p.shape[1]
                # VL+SA part
                if encoder_attention_mask is not None:
                    n_vl_mask = n_vl_for_single
                    rest_mask_x = torch.ones(
                        b_mask,
                        n_x_mask - n_vl_mask,
                        device=x.device,
                        dtype=encoder_attention_mask.dtype,
                    )
                    x_mask = torch.cat([encoder_attention_mask, rest_mask_x], dim=1)
                else:
                    x_mask = torch.ones(b_mask, n_x_mask, device=x.device, dtype=x.dtype)
                # P part
                if physics_attention_mask is not None:
                    p_mask = physics_attention_mask[:, None].expand(-1, n_p_mask).to(dtype=x_mask.dtype)
                else:
                    p_mask = torch.ones(b_mask, n_p_mask, device=x.device, dtype=x_mask.dtype)
                kv_mask = torch.cat([x_mask, p_mask], dim=1)
                single_attn_mask = kv_mask[:, None, None, :]
                single_attn_mask = torch.where(
                    single_attn_mask == 0,
                    torch.tensor(float("-inf"), device=x.device, dtype=x.dtype),
                    torch.tensor(0.0, device=x.device, dtype=x.dtype),
                )

            for blk in self.single_blocks:

                def _run_expanded_single(
                    x_tokens: torch.Tensor,
                    p_tokens: torch.Tensor,
                    _blk: nn.Module = blk,
                    _block_idx: int = block_idx,
                ) -> tuple[torch.Tensor, torch.Tensor]:
                    return _blk(
                        x_tokens,
                        temb,
                        pe=pe_single,
                        shared_modulation=shared_single_modulation,
                        time_token=time_token if has_time_token else None,
                        block_idx=_block_idx,
                        p_tokens=p_tokens,
                        attn_mask=single_attn_mask,
                    )

                x, p = self._apply_checkpoint(_run_expanded_single, x, p)
                block_idx += 1

            sa = x[:, -sa.shape[1] :, :]

        # ── Output projections ────────────────────────────────────────────
        action_out = self._output_projection(sa, temb)
        physics_out = self._output_projection_physics(p, temb)

        out = {"action": action_out, "physics": physics_out}
        if return_all_hidden_states:
            return out, all_hidden
        return out

    def _output_projection_physics(self, p: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        """Project physics hidden states to output space.

        Args:
            p: Physics hidden states ``[B, N_p, D]``.
            temb: Timestep embedding tensor ``[B, D_t]``.

        Returns:
            Physics output tensor ``[B, N_p, output_dim]``.
        """
        shift, scale = self.proj_out_physics_1(F.silu(temb)).chunk(2, dim=1)
        p = self.norm_out_physics(p) * (1 + scale[:, None]) + shift[:, None]
        return self.proj_out_physics_2(p)

    def _output_projection(self, sa: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        """Project action hidden states to output space.

        Args:
            sa: Action hidden states ``[B, N_action, D]``.
            temb: Timestep embedding tensor ``[B, D_t]``.

        Returns:
            Action output tensor ``[B, N_action, output_dim]``.
        """
        shift, scale = self.proj_out_1(F.silu(temb)).chunk(2, dim=1)
        sa = self.norm_out(sa) * (1 + scale[:, None]) + shift[:, None]
        return self.proj_out_2(sa)
