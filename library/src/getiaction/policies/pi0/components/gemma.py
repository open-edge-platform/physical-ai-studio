# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2025 Physical Intelligence
# SPDX-License-Identifier: Apache-2.0

"""Gemma backbone components for Pi0/Pi0.5 models.

This module provides the PaliGemma backbone and Gemma action expert
used by Pi0/Pi0.5 for vision-language-action modeling.

Architecture:
- PaliGemma: SigLIP vision encoder + Gemma language model
- Action Expert: Smaller Gemma model for action prediction
- CRITICAL: Both models process through shared layer-by-layer attention
  when both prefix and suffix are provided (training mode)

Based on OpenPI and LeRobot implementations.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

if TYPE_CHECKING:
    from transformers import GemmaForCausalLM, PaliGemmaForConditionalGeneration

logger = logging.getLogger(__name__)

# Gemma variant type
GemmaVariant = Literal["gemma_300m", "gemma_2b"]

# Attention mask value for masked positions (from openpi)
OPENPI_ATTENTION_MASK_VALUE = -3.4028235e38


@dataclass
class _GemmaConfig:
    """Internal configuration for Gemma model variants.

    This is an internal implementation detail. Users should use the variant
    strings ("gemma_300m", "gemma_2b") with PaliGemmaWithExpert instead.

    Attributes:
        vocab_size: Vocabulary size.
        width: Hidden dimension.
        depth: Number of transformer layers.
        mlp_dim: MLP intermediate dimension.
        num_heads: Number of attention heads.
        num_kv_heads: Number of key-value heads (for GQA).
        head_dim: Dimension per attention head.
    """

    vocab_size: int
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int


# Gemma model configurations from OpenPI (internal lookup table)
_GEMMA_CONFIGS: dict[GemmaVariant, _GemmaConfig] = {
    "gemma_300m": _GemmaConfig(
        vocab_size=257152,
        width=1024,
        depth=18,
        mlp_dim=4096,
        num_heads=8,
        num_kv_heads=1,
        head_dim=256,
    ),
    "gemma_2b": _GemmaConfig(
        vocab_size=257152,
        width=2048,
        depth=18,
        mlp_dim=16384,
        num_heads=8,
        num_kv_heads=1,
        head_dim=256,
    ),
}


def _get_gemma_config(variant: GemmaVariant) -> _GemmaConfig:
    """Get Gemma configuration for a variant (internal).

    Args:
        variant: Gemma variant name ("gemma_300m" or "gemma_2b").

    Returns:
        _GemmaConfig for the specified variant.

    Raises:
        ValueError: If variant is unknown.
    """
    if variant not in _GEMMA_CONFIGS:
        msg = f"Unknown Gemma variant: {variant}. Available: {list(_GEMMA_CONFIGS.keys())}"
        raise ValueError(msg)
    return _GEMMA_CONFIGS[variant]


class AdaRMSNorm(nn.Module):
    """Adaptive RMSNorm for Pi0.5 timestep conditioning.

    This version matches lerobot's implementation which returns both
    the normalized output AND a gate for gated residual connections.

    Args:
        hidden_size: Dimension of the input features.
        eps: Small constant for numerical stability.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        # Dense layer for adaptive scaling (matches lerobot's structure)
        self.dense = nn.Linear(hidden_size, hidden_size)

    def forward(
        self, hidden_states: torch.Tensor, cond: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply adaptive RMSNorm.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size).
            cond: Optional conditioning tensor of shape (batch, hidden_size).

        Returns:
            Tuple of (normalized_output, gate).
            Gate is None if cond is None, otherwise it's the scale factor for gated residual.
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        output = self.weight * hidden_states.to(input_dtype)

        gate = None
        if cond is not None:
            # Project conditioning to scale factor
            scale = self.dense(cond)
            # Expand for broadcasting: (batch, hidden) -> (batch, 1, hidden)
            gate = scale.unsqueeze(1)
            output = output * (1.0 + gate)

        return output, gate


def _gated_residual(
    residual: torch.Tensor | None, hidden_states: torch.Tensor | None, gate: torch.Tensor | None
) -> torch.Tensor | None:
    """Apply gated residual connection.

    Matches lerobot's modeling_gemma._gated_residual function exactly.

    Args:
        residual: Original input (before layer). Can be None.
        hidden_states: Layer output. Can be None.
        gate: Optional gate from AdaRMSNorm.

    Returns:
        Gated residual output, or None if both inputs are None.
    """
    # Handle None cases (matches lerobot exactly)
    if residual is None and hidden_states is None:
        return None
    if residual is None or hidden_states is None:
        return residual if residual is not None else hidden_states
    if gate is None:
        return residual + hidden_states
    # Gated residual: residual + hidden_states * gate
    return residual + hidden_states * gate


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input.

    Matches transformers' rotate_half function.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors.

    Matches transformers' apply_rotary_pos_emb function signature.

    Args:
        query: Query tensor of shape (batch, heads, seq, head_dim).
        key: Key tensor of shape (batch, heads, seq, head_dim).
        cos: Cosine embeddings of shape (batch, seq, head_dim).
        sin: Sine embeddings of shape (batch, seq, head_dim).
        unsqueeze_dim: The dimension along which to unsqueeze cos/sin for broadcasting.
            Default 1 for (batch, heads, seq, head_dim) shaped tensors.

    Returns:
        Tuple of (rotated_query, rotated_key).
    """
    # Unsqueeze cos/sin for broadcasting to query/key shape
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    query_rot = (query * cos) + (_rotate_half(query) * sin)
    key_rot = (key * cos) + (_rotate_half(key) * sin)

    return query_rot, key_rot


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value states for grouped-query attention.

    This is equivalent to torch.repeat_interleave(x, dim=1, repeats=n_rep).
    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim)
    to (batch, num_attention_heads, seqlen, head_dim).

    Args:
        hidden_states: Key or value tensor.
        n_rep: Number of repetitions (num_heads // num_kv_heads).

    Returns:
        Repeated tensor.
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    scaling: float,
    num_key_value_groups: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute attention using eager (non-flash) implementation.

    Matches transformers' eager_attention_forward pattern.

    Args:
        query: Query tensor (batch, heads, seq, head_dim).
        key: Key tensor (batch, kv_heads, seq, head_dim).
        value: Value tensor (batch, kv_heads, seq, head_dim).
        attention_mask: Attention mask (batch, 1, seq, seq).
        scaling: Attention scaling factor (1/sqrt(head_dim)).
        num_key_value_groups: Number of query heads per KV head (for GQA).

    Returns:
        Tuple of (attention_output, attention_weights).
        Output shape is (batch, seq, heads, head_dim) - NOT reshaped.
    """
    # Repeat KV for grouped-query attention
    key_states = repeat_kv(key, num_key_value_groups)
    value_states = repeat_kv(value, num_key_value_groups)

    # Compute attention scores
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    # Apply attention mask
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # Softmax
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

    # Apply attention to values
    attn_output = torch.matmul(attn_weights, value_states)

    # Transpose: (batch, heads, seq, head_dim) -> (batch, seq, heads, head_dim)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def compute_layer_complete(
    layer_idx: int,
    inputs_embeds: list[torch.Tensor],
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    adarms_cond: list[torch.Tensor | None],
    paligemma: PaliGemmaForConditionalGeneration,
    gemma_expert: GemmaForCausalLM,
    adarms_layers: nn.ModuleList | None = None,
) -> list[torch.Tensor]:
    """Compute one transformer layer with shared attention across both models.

    This is the CRITICAL function that enables Pi0/Pi0.5 to work correctly.
    Both PaliGemma and the action expert share attention in the same sequence space.

    Based on lerobot's compute_layer_complete implementation.

    Args:
        layer_idx: Index of the current layer.
        inputs_embeds: List of [prefix_embeds, suffix_embeds].
        attention_mask: 4D attention mask (batch, 1, total_seq, total_seq).
        position_ids: Position IDs (batch, total_seq).
        adarms_cond: List of [prefix_cond, suffix_cond] for AdaRMSNorm.
        paligemma: PaliGemma model.
        gemma_expert: Gemma action expert model.
        adarms_layers: Optional AdaRMSNorm layers for Pi0.5.

    Returns:
        List of [prefix_output, suffix_output] hidden states.
    """
    paligemma_gemma_model = paligemma.language_model
    expert_gemma_model = gemma_expert.model
    models = [paligemma_gemma_model, expert_gemma_model]

    query_states_list: list[torch.Tensor] = []
    key_states_list: list[torch.Tensor] = []
    value_states_list: list[torch.Tensor] = []
    gates: list[torch.Tensor | None] = []

    # Process each model's embeddings through input layernorm and Q/K/V projections
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]

        # Apply input layernorm (with AdaRMSNorm for suffix if enabled)
        if adarms_layers is not None and i == 1 and adarms_cond[i] is not None:
            # Use AdaRMSNorm for action expert (suffix)
            ada_layer_idx = layer_idx * 2  # input_layernorm index
            hidden_states, gate = adarms_layers[ada_layer_idx](hidden_states, cond=adarms_cond[i])
        else:
            # Standard RMSNorm - returns just tensor, not tuple
            hidden_states = layer.input_layernorm(hidden_states)
            gate = None

        gates.append(gate)

        # Compute Q, K, V projections
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

        query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        query_states_list.append(query_state)
        key_states_list.append(key_state)
        value_states_list.append(value_state)

    # Concatenate Q, K, V across both models for shared attention
    query_states = torch.cat(query_states_list, dim=2)
    key_states = torch.cat(key_states_list, dim=2)
    value_states = torch.cat(value_states_list, dim=2)

    # Compute rotary embeddings using the GemmaModel's rotary_emb
    dummy_tensor = torch.zeros(
        query_states.shape[0],
        query_states.shape[2],  # total_seq_len
        query_states.shape[-1],  # head_dim
        device=query_states.device,
        dtype=query_states.dtype,
    )
    cos, sin = paligemma_gemma_model.rotary_emb(dummy_tensor, position_ids)

    # Apply rotary position embeddings with unsqueeze_dim=1
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=1)

    # Get attention scaling factor from the layer
    scaling = paligemma_gemma_model.layers[layer_idx].self_attn.scaling

    # Get num_key_value_groups for GQA
    num_kv_groups = paligemma_gemma_model.layers[layer_idx].self_attn.num_key_value_groups

    # Compute shared attention
    att_output, _ = eager_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling,
        num_key_value_groups=num_kv_groups,
    )

    # Get head_dim for reshape
    head_dim = paligemma_gemma_model.layers[layer_idx].self_attn.head_dim
    batch_size = query_states.shape[0]

    # Reshape attention output: (batch, seq, heads, head_dim) -> (batch, seq, heads * head_dim)
    num_heads = paligemma_gemma_model.layers[layer_idx].self_attn.config.num_attention_heads
    att_output = att_output.reshape(batch_size, -1, num_heads * head_dim)

    # Process outputs for each model
    outputs_embeds: list[torch.Tensor] = []
    start_pos = 0

    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        end_pos = start_pos + hidden_states.shape[1]

        # Slice attention output for this model's sequence
        att_slice = att_output[:, start_pos:end_pos]

        # Convert attention output dtype if needed
        if att_slice.dtype != layer.self_attn.o_proj.weight.dtype:
            att_slice = att_slice.to(layer.self_attn.o_proj.weight.dtype)

        # Output projection
        out_emb = layer.self_attn.o_proj(att_slice)

        # First residual connection (with gating for AdaRMSNorm)
        out_emb = _gated_residual(hidden_states, out_emb, gates[i])
        if out_emb is None:
            msg = "Unexpected None from _gated_residual"
            raise RuntimeError(msg)
        after_first_residual = out_emb.clone()

        # Post-attention layernorm (with AdaRMSNorm for suffix if enabled)
        if adarms_layers is not None and i == 1 and adarms_cond[i] is not None:
            ada_layer_idx = layer_idx * 2 + 1  # post_attention_layernorm index
            out_emb, gate = adarms_layers[ada_layer_idx](out_emb, cond=adarms_cond[i])
        else:
            # Standard RMSNorm
            out_emb = layer.post_attention_layernorm(out_emb)
            gate = None

        # Convert to bfloat16 if MLP uses bfloat16
        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
            out_emb = out_emb.to(dtype=torch.bfloat16)

        # MLP
        out_emb = layer.mlp(out_emb)

        # Second residual connection (with gating)
        out_emb = _gated_residual(after_first_residual, out_emb, gate)
        if out_emb is None:
            msg = "Unexpected None from _gated_residual"
            raise RuntimeError(msg)

        outputs_embeds.append(out_emb)
        start_pos = end_pos

    return outputs_embeds


class PaliGemmaWithExpert(nn.Module):
    """PaliGemma backbone with action expert for Pi0/Pi0.5.

    This module combines:
    1. PaliGemma: Vision-language model (SigLIP + Gemma)
    2. Action Expert: Smaller Gemma model for action prediction

    CRITICAL: When both prefix and suffix embeddings are provided (training),
    they are processed through shared layer-by-layer attention via
    `compute_layer_complete`. This is essential for numerical consistency
    with the original OpenPI/lerobot implementations.

    Args:
        paligemma_variant: Size of the PaliGemma backbone.
        action_expert_variant: Size of the action expert.
        use_adarms: Whether to use AdaRMSNorm (Pi0.5 mode).
        dtype: Compute dtype.
        paligemma_model_id: HuggingFace model ID for PaliGemma.

    Example:
        >>> model = PaliGemmaWithExpert(
        ...     paligemma_variant="gemma_2b",
        ...     action_expert_variant="gemma_300m",
        ...     use_adarms=True,  # Pi0.5 mode
        ... )
        >>> # Forward pass with prefix (images + language) and suffix (actions)
        >>> outputs = model(
        ...     inputs_embeds=[prefix_embeds, suffix_embeds],
        ...     attention_mask=mask_4d,
        ...     position_ids=positions,
        ...     adarms_cond=[None, timestep_emb],
        ... )
    """

    # Default PaliGemma model for each variant
    PALIGEMMA_MODEL_IDS: ClassVar[dict[GemmaVariant, str]] = {
        "gemma_300m": "google/paligemma-3b-pt-224",  # Uses smaller projection
        "gemma_2b": "google/paligemma-3b-pt-224",
    }

    def __init__(
        self,
        paligemma_variant: GemmaVariant = "gemma_2b",
        action_expert_variant: GemmaVariant = "gemma_300m",
        *,
        use_adarms: bool = False,
        dtype: str = "bfloat16",
        paligemma_model_id: str | None = None,
    ) -> None:
        """Initialize PaliGemma with action expert.

        Args:
            paligemma_variant: Size of the PaliGemma backbone.
            action_expert_variant: Size of the action expert.
            use_adarms: Whether to use AdaRMSNorm (Pi0.5 mode).
            dtype: Compute dtype ("bfloat16" or "float32").
            paligemma_model_id: Override HuggingFace model ID for PaliGemma.
        """
        super().__init__()

        self.paligemma_variant = paligemma_variant
        self.action_expert_variant = action_expert_variant
        self.use_adarms = use_adarms
        self._dtype_str = dtype

        # Get configs (internal lookup)
        self._paligemma_config = _get_gemma_config(paligemma_variant)
        self._action_expert_config = _get_gemma_config(action_expert_variant)

        # Expose key hyperparameters as explicit instance attributes
        self.paligemma_hidden_size = self._paligemma_config.width
        self.action_expert_hidden_size = self._action_expert_config.width
        self.action_expert_num_layers = self._action_expert_config.depth

        # Store model ID
        self._paligemma_model_id = paligemma_model_id or self.PALIGEMMA_MODEL_IDS.get(
            paligemma_variant,
            "google/paligemma-3b-pt-224",
        )

        # Models will be lazily loaded
        self._paligemma: PaliGemmaForConditionalGeneration | None = None
        self._action_expert: GemmaForCausalLM | None = None

        # AdaRMSNorm layers for Pi0.5 (replace standard RMSNorm in action expert)
        self._adarms_layers: nn.ModuleList | None = None

        # Track if we've been initialized
        self._initialized = False

        # Track target device for lazy loading
        self._target_device: torch.device | None = None
        self._target_dtype: torch.dtype | None = None

    def _apply(self, fn: Any, recurse: bool = True) -> PaliGemmaWithExpert:
        """Override _apply to handle lazy-loaded models.

        This is called by `.to()`, `.cuda()`, `.cpu()`, etc. We need to:
        1. Track the target device/dtype for lazy loading
        2. Apply to already-loaded models if they exist

        Args:
            fn: The function to apply (from torch.nn.Module._apply).
            recurse: Whether to recurse into child modules.

        Returns:
            Self for method chaining.
        """
        # Call parent _apply first (handles registered submodules)
        super()._apply(fn, recurse)

        # Try to extract device and dtype from fn by applying to a dummy tensor
        try:
            dummy = torch.zeros(1)
            transformed = fn(dummy)
            self._target_device = transformed.device
            self._target_dtype = transformed.dtype
        except Exception:
            pass

        # If models are already loaded, apply to them
        if self._paligemma is not None:
            self._paligemma._apply(fn, recurse)  # type: ignore[union-attr]
        if self._action_expert is not None:
            self._action_expert._apply(fn, recurse)  # type: ignore[union-attr]

        return self

    @property
    def dtype(self) -> torch.dtype:
        """Get compute dtype."""
        return torch.bfloat16 if self._dtype_str == "bfloat16" else torch.float32

    def _ensure_loaded(self) -> None:
        """Lazy load the models from HuggingFace.

        Raises:
            ImportError: If transformers>=4.40.0 is not installed.
        """
        if self._initialized:
            return

        try:
            from transformers import (  # noqa: PLC0415
                GemmaForCausalLM,
                PaliGemmaForConditionalGeneration,
            )
        except ImportError as e:
            msg = "PaliGemma requires transformers>=4.40.0. Install with: pip install transformers>=4.40.0"
            raise ImportError(msg) from e

        logger.info("Loading PaliGemma backbone: %s", self._paligemma_model_id)

        target_dtype = self._target_dtype if self._target_dtype is not None else self.dtype

        self._paligemma = PaliGemmaForConditionalGeneration.from_pretrained(  # nosec B615
            self._paligemma_model_id,
            torch_dtype=target_dtype,
            revision="main",
        )

        logger.info("Initializing action expert: %s", self.action_expert_variant)

        from transformers import GemmaConfig as HFGemmaConfig  # noqa: PLC0415

        action_config = self._action_expert_config
        hf_config = HFGemmaConfig(
            vocab_size=action_config.vocab_size,
            hidden_size=action_config.width,
            intermediate_size=action_config.mlp_dim,
            num_hidden_layers=action_config.depth,
            num_attention_heads=action_config.num_heads,
            num_key_value_heads=action_config.num_kv_heads,
            head_dim=action_config.head_dim,
        )

        self._action_expert = GemmaForCausalLM(hf_config)
        if self._action_expert is not None:
            self._action_expert = self._action_expert.to(target_dtype)  # type: ignore[assignment]

        self._action_expert.model.embed_tokens = None  # type: ignore[assignment]

        if self.use_adarms:
            self._setup_adarms()

        if self._target_device is not None:
            if self._paligemma is not None:
                self._paligemma = self._paligemma.to(self._target_device)
            if self._action_expert is not None:
                self._action_expert = self._action_expert.to(self._target_device)
            if self._adarms_layers is not None:
                self._adarms_layers = self._adarms_layers.to(self._target_device)

        self._initialized = True

    def _setup_adarms(self) -> None:
        """Setup AdaRMSNorm layers for Pi0.5 timestep conditioning.

        Creates 2 AdaRMSNorm layers per transformer layer:
        - One for input_layernorm
        - One for post_attention_layernorm
        """
        # Use explicit instance attributes
        hidden_size = self.action_expert_hidden_size
        num_layers = self.action_expert_num_layers

        # Create AdaRMSNorm layers for each transformer layer
        # 2 per layer: input_layernorm and post_attention_layernorm
        self._adarms_layers = nn.ModuleList([AdaRMSNorm(hidden_size) for _ in range(num_layers * 2)])

        logger.info("Initialized %d AdaRMSNorm layers for Pi0.5", len(self._adarms_layers))

    @property
    def paligemma(self) -> PaliGemmaForConditionalGeneration:
        """Get PaliGemma model.

        Raises:
            RuntimeError: If model failed to load.
        """
        self._ensure_loaded()
        if self._paligemma is None:
            msg = "PaliGemma model not loaded"
            raise RuntimeError(msg)
        return self._paligemma

    @property
    def action_expert(self) -> GemmaForCausalLM:
        """Get action expert model.

        Raises:
            RuntimeError: If model failed to load.
        """
        self._ensure_loaded()
        if self._action_expert is None:
            msg = "Action expert model not loaded"
            raise RuntimeError(msg)
        return self._action_expert

    def embed_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Embed images using PaliGemma's vision encoder.

        Args:
            pixel_values: Image tensor of shape (batch, channels, height, width).

        Returns:
            Image embeddings of shape (batch, num_patches, hidden_size).
        """
        self._ensure_loaded()

        # Use PaliGemma's vision tower (get_image_features handles projection)
        return self.paligemma.model.get_image_features(pixel_values)

    def embed_language_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed language tokens using PaliGemma's embedding layer.

        Args:
            input_ids: Token IDs of shape (batch, seq_len).

        Returns:
            Token embeddings of shape (batch, seq_len, hidden_size).
        """
        self._ensure_loaded()

        # Use the language model's embedding layer
        return self.paligemma.language_model.embed_tokens(input_ids)

    def forward(
        self,
        inputs_embeds: list[torch.Tensor | None],
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
        *,
        use_cache: bool = False,
        adarms_cond: list[torch.Tensor | None] | None = None,
    ) -> tuple[tuple[torch.Tensor | None, torch.Tensor | None], Any]:
        """Forward pass through PaliGemma and action expert.

        This implements the dual-stream architecture where:
        - PaliGemma processes the prefix (images + language)
        - Action expert processes the suffix (action tokens)

        CRITICAL: When BOTH prefix and suffix embeddings are provided (training mode),
        we use shared layer-by-layer attention via `compute_layer_complete`.
        This is essential for numerical consistency with OpenPI/lerobot.

        Args:
            inputs_embeds: List of [prefix_embeds, suffix_embeds]. Either can be None.
            attention_mask: 4D attention mask of shape (batch, 1, seq, seq).
            position_ids: Position IDs of shape (batch, seq_len).
            past_key_values: Cached key-values for incremental decoding.
            use_cache: Whether to return updated key-value cache.
            adarms_cond: List of [prefix_cond, suffix_cond] for AdaRMSNorm.
                Only used in Pi0.5 mode. suffix_cond is the timestep embedding.

        Returns:
            Tuple of:
                - (prefix_output, suffix_output): Hidden states from each stream
                - past_key_values: Updated cache if use_cache=True
        """
        self._ensure_loaded()

        if adarms_cond is None:
            adarms_cond = [None, None]

        prefix_embeds, suffix_embeds = inputs_embeds

        # Case 1: Prefix only (used during inference to cache prefix)
        if suffix_embeds is None and prefix_embeds is not None:
            prefix_output = self.paligemma.language_model.forward(
                inputs_embeds=prefix_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                return_dict=True,
            )
            return (prefix_output.last_hidden_state, None), prefix_output.past_key_values

        # Case 2: Suffix only (used during inference denoising steps with cached prefix)
        if prefix_embeds is None and suffix_embeds is not None:
            suffix_cond = adarms_cond[1]

            # For suffix-only with AdaRMSNorm, we need custom forward
            if self.use_adarms and suffix_cond is not None and self._adarms_layers is not None:
                suffix_output = self._forward_action_expert_with_adarms(
                    suffix_embeds,
                    attention_mask,
                    position_ids,
                    suffix_cond,
                    past_key_values,
                )
            elif past_key_values is not None:
                suffix_output = self._forward_action_expert_with_kv_cache(
                    suffix_embeds,
                    attention_mask,
                    position_ids,
                    past_key_values,
                )
            else:
                expert_outputs = self.action_expert.model.forward(
                    inputs_embeds=suffix_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    return_dict=True,
                )
                suffix_output = expert_outputs.last_hidden_state

            return (None, suffix_output), None

        # Case 3: BOTH prefix AND suffix (training mode)
        # Use shared layer-by-layer attention - this is CRITICAL for numerical consistency
        num_layers = self.paligemma.config.text_config.num_hidden_layers

        # At this point, both prefix_embeds and suffix_embeds are not None
        # (we handled the None cases above)
        assert prefix_embeds is not None
        assert suffix_embeds is not None
        current_embeds: list[torch.Tensor] = [prefix_embeds, suffix_embeds]

        # Process all layers with shared attention
        for layer_idx in range(num_layers):
            current_embeds = compute_layer_complete(
                layer_idx,
                current_embeds,
                attention_mask,
                position_ids,
                adarms_cond,
                paligemma=self.paligemma,
                gemma_expert=self.action_expert,
                adarms_layers=self._adarms_layers,
            )

        # Final layer norms - use GemmaModel instances which have .norm
        paligemma_gemma_model = self.paligemma.language_model
        expert_gemma_model = self.action_expert.model
        models = [paligemma_gemma_model, expert_gemma_model]
        outputs_embeds: list[torch.Tensor] = []

        for i, hidden_states in enumerate(current_embeds):
            # Apply final norm (with AdaRMSNorm for suffix if enabled)
            cond_i = adarms_cond[i]
            if self._adarms_layers is not None and i == 1 and cond_i is not None:
                # Use a dedicated AdaRMSNorm for final norm
                # Note: For simplicity, we create one on the fly. In production,
                # this should be a stored module.
                out_emb = models[i].norm(hidden_states)
                # Apply conditioning scaling
                scale = 1.0 + cond_i.unsqueeze(1)
                out_emb = out_emb * scale
            else:
                out_emb = models[i].norm(hidden_states)
            outputs_embeds.append(out_emb)

        prefix_output = outputs_embeds[0]
        suffix_output = outputs_embeds[1]

        return (prefix_output, suffix_output), None

    def _forward_action_expert_with_adarms(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        adarms_cond: torch.Tensor,
        past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
    ) -> torch.Tensor:
        """Forward through action expert with AdaRMSNorm conditioning (suffix-only).

        This is used during inference when we have cached prefix KV values.

        Args:
            inputs_embeds: Input embeddings.
            attention_mask: Attention mask.
            position_ids: Position IDs.
            adarms_cond: Conditioning tensor (timestep embedding).
            past_key_values: Cached key-values from prefix.

        Returns:
            Output hidden states.
        """
        # This method is only called when _adarms_layers is not None
        adarms_layers = self._adarms_layers
        if adarms_layers is None:
            msg = "AdaRMSNorm layers must be initialized for this method"
            raise RuntimeError(msg)

        hidden_states: torch.Tensor = inputs_embeds

        for layer_idx, layer in enumerate(self.action_expert.model.layers):
            # Input layernorm with AdaRMSNorm
            ada_layer_idx = layer_idx * 2
            normed_hidden, gate = adarms_layers[ada_layer_idx](hidden_states, cond=adarms_cond)

            # Self attention
            input_shape = normed_hidden.shape[:-1]
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

            query = layer.self_attn.q_proj(normed_hidden).view(hidden_shape).transpose(1, 2)
            key = layer.self_attn.k_proj(normed_hidden).view(hidden_shape).transpose(1, 2)
            value = layer.self_attn.v_proj(normed_hidden).view(hidden_shape).transpose(1, 2)

            # Get rotary embeddings
            cos, sin = self.action_expert.model.rotary_emb(hidden_states, position_ids)
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

            # Handle past key values (from cached prefix)
            if past_key_values is not None and layer_idx < len(past_key_values):
                past_key, past_value = past_key_values[layer_idx]
                key = torch.cat([past_key, key], dim=2)
                value = torch.cat([past_value, value], dim=2)

            # Compute attention
            scaling = layer.self_attn.scaling
            num_kv_groups = layer.self_attn.num_key_value_groups
            attn_output, _ = eager_attention_forward(
                query, key, value, attention_mask, scaling, num_key_value_groups=num_kv_groups
            )

            # Reshape attention output
            batch_size = query.shape[0]
            head_dim = layer.self_attn.head_dim
            num_heads = layer.self_attn.config.num_attention_heads
            attn_output = attn_output.reshape(batch_size, -1, num_heads * head_dim)

            # Output projection
            attn_output = layer.self_attn.o_proj(attn_output)

            # First residual
            result = _gated_residual(hidden_states, attn_output, gate)
            assert result is not None
            hidden_states = result
            residual = hidden_states

            # Post-attention layernorm with AdaRMSNorm
            ada_layer_idx = layer_idx * 2 + 1
            hidden_states, gate = adarms_layers[ada_layer_idx](hidden_states, cond=adarms_cond)

            # MLP
            hidden_states = layer.mlp(hidden_states)

            # Second residual
            result = _gated_residual(residual, hidden_states, gate)
            assert result is not None
            hidden_states = result

        # Final norm
        hidden_states = self.action_expert.model.norm(hidden_states)

        return hidden_states

    def _forward_action_expert_with_kv_cache(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    ) -> torch.Tensor:
        """Forward through action expert with tuple-based KV cache (suffix-only, no AdaRMS).

        This is used during inference for Pi0 (non-AdaRMS) when we have cached prefix KV values.
        We need manual forward because HuggingFace's forward expects DynamicCache, not tuples.

        Args:
            inputs_embeds: Input embeddings.
            attention_mask: Attention mask (4D).
            position_ids: Position IDs.
            past_key_values: Tuple of (key, value) pairs per layer.

        Returns:
            Output hidden states.
        """
        hidden_states: torch.Tensor = inputs_embeds

        for layer_idx, layer in enumerate(self.action_expert.model.layers):
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

            query = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            value = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            cos, sin = self.action_expert.model.rotary_emb(hidden_states, position_ids)
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

            if layer_idx < len(past_key_values):
                past_key, past_value = past_key_values[layer_idx]
                key = torch.cat([past_key, key], dim=2)
                value = torch.cat([past_value, value], dim=2)

            scaling = layer.self_attn.scaling
            num_kv_groups = layer.self_attn.num_key_value_groups
            attn_output, _ = eager_attention_forward(
                query, key, value, attention_mask, scaling, num_key_value_groups=num_kv_groups
            )

            batch_size = query.shape[0]
            head_dim = layer.self_attn.head_dim
            num_heads = layer.self_attn.config.num_attention_heads
            attn_output = attn_output.reshape(batch_size, -1, num_heads * head_dim)
            attn_output = layer.self_attn.o_proj(attn_output)

            hidden_states = residual + attn_output
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states

        hidden_states = self.action_expert.model.norm(hidden_states)

        return hidden_states

    def to_bfloat16_for_selected_params(self, dtype_str: str = "bfloat16") -> None:
        """Convert selected parameters to bfloat16 for memory efficiency.

        Args:
            dtype_str: Target dtype string.
        """
        if dtype_str != "bfloat16":
            return

        self._ensure_loaded()

        # Convert PaliGemma
        self.paligemma.to(torch.bfloat16)  # type: ignore[method-call]

        # Convert action expert
        self.action_expert.to(torch.bfloat16)  # type: ignore[method-call]

        logger.info("Converted models to bfloat16")

    def set_trainable_parameters(
        self,
        *,
        tune_paligemma: bool = False,
        tune_action_expert: bool = True,
        tune_vision_encoder: bool = False,
    ) -> None:
        """Set which parameters are trainable.

        Args:
            tune_paligemma: Whether to train PaliGemma backbone.
            tune_action_expert: Whether to train action expert.
            tune_vision_encoder: Whether to train vision encoder.
        """
        self._ensure_loaded()

        # Freeze/unfreeze PaliGemma
        for param in self.paligemma.language_model.parameters():
            param.requires_grad = tune_paligemma

        # Freeze/unfreeze vision encoder
        for param in self.paligemma.vision_tower.parameters():
            param.requires_grad = tune_vision_encoder

        # Freeze/unfreeze projector (typically trained)
        for param in self.paligemma.multi_modal_projector.parameters():
            param.requires_grad = tune_paligemma

        # Freeze/unfreeze action expert
        for param in self.action_expert.parameters():
            param.requires_grad = tune_action_expert

        # AdaRMSNorm layers are always trainable in Pi0.5
        if self._adarms_layers is not None:
            for param in self._adarms_layers.parameters():
                param.requires_grad = True

        # Log trainable params
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info("Trainable parameters: %d / %d (%.2f%%)", trainable, total, 100 * trainable / total)
