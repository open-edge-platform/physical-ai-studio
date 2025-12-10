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
- Both share the same architecture but can have different sizes

Based on OpenPI implementation with PyTorch-only support.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import torch
from torch import nn

from .attention import AdaRMSNorm

if TYPE_CHECKING:
    from transformers import GemmaForCausalLM, PaliGemmaForConditionalGeneration

logger = logging.getLogger(__name__)

# Gemma variant type
GemmaVariant = Literal["gemma_300m", "gemma_2b"]


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


class PaliGemmaWithExpert(nn.Module):
    """PaliGemma backbone with action expert for Pi0/Pi0.5.

    This module combines:
    1. PaliGemma: Vision-language model (SigLIP + Gemma)
    2. Action Expert: Smaller Gemma model for action prediction

    The two models share the same forward pass structure but have separate
    parameters. Pi0.5 uses AdaRMSNorm in the action expert for timestep conditioning.

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

        # Load PaliGemma
        self._paligemma = PaliGemmaForConditionalGeneration.from_pretrained(
            self._paligemma_model_id,
            torch_dtype=self.dtype,
        )

        # Create action expert (smaller Gemma)
        # We initialize from scratch with the right config
        logger.info("Initializing action expert: %s", self.action_expert_variant)

        # Get HF config for action expert
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
        self._action_expert = self._action_expert.to(self.dtype)  # type: ignore[assignment]

        # Setup AdaRMSNorm for Pi0.5
        if self.use_adarms:
            self._setup_adarms()

        self._initialized = True

    def _setup_adarms(self) -> None:
        """Replace RMSNorm with AdaRMSNorm in action expert for Pi0.5."""
        if self._action_expert is None:
            return

        # Use explicit instance attributes (hparams-first design)
        hidden_size = self.action_expert_hidden_size
        num_layers = self.action_expert_num_layers

        # Create AdaRMSNorm layers for each transformer layer
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

        # Use PaliGemma's vision tower
        vision_outputs = self.paligemma.vision_tower(pixel_values)
        image_features = vision_outputs.last_hidden_state

        # Project to language model dimension
        return self.paligemma.multi_modal_projector(image_features)

    def embed_language_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed language tokens using PaliGemma's embedding layer.

        Args:
            input_ids: Token IDs of shape (batch, seq_len).

        Returns:
            Token embeddings of shape (batch, seq_len, hidden_size).
        """
        self._ensure_loaded()

        # Use the language model's embedding layer
        return self.paligemma.language_model.model.embed_tokens(input_ids)

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

        prefix_embeds, suffix_embeds = inputs_embeds

        # Handle prefix (PaliGemma)
        prefix_output = None
        if prefix_embeds is not None:
            # Forward through PaliGemma language model
            pali_outputs = self.paligemma.language_model(
                inputs_embeds=prefix_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                return_dict=True,
            )
            prefix_output = pali_outputs.last_hidden_state

            if use_cache:
                past_key_values = pali_outputs.past_key_values

        # Handle suffix (Action Expert)
        suffix_output = None
        if suffix_embeds is not None:
            suffix_cond = adarms_cond[1] if adarms_cond is not None else None

            # For Pi0.5 with AdaRMSNorm, we need custom forward
            if self.use_adarms and suffix_cond is not None:
                suffix_output = self._forward_action_expert_with_adarms(
                    suffix_embeds,
                    attention_mask,
                    position_ids,
                    suffix_cond,
                    past_key_values,
                )
            else:
                # Standard forward through action expert
                expert_outputs = self.action_expert(
                    inputs_embeds=suffix_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    return_dict=True,
                )
                suffix_output = expert_outputs.last_hidden_state

        return (prefix_output, suffix_output), past_key_values

    def _forward_action_expert_with_adarms(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        adarms_cond: torch.Tensor,  # noqa: ARG002 - will be used when AdaRMSNorm injection implemented
        past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
    ) -> torch.Tensor:
        """Forward through action expert with AdaRMSNorm conditioning.

        This implements the Pi0.5 variant where timestep information is
        injected via AdaRMSNorm layers instead of concatenation.

        Args:
            inputs_embeds: Input embeddings.
            attention_mask: Attention mask.
            position_ids: Position IDs.
            adarms_cond: Conditioning tensor (timestep embedding).
            past_key_values: Cached key-values.

        Returns:
            Output hidden states.
        """
        # NOTE: AdaRMSNorm injection not yet implemented, using standard forward.
        # Implementing this requires modifying the Gemma forward pass to use AdaRMSNorm.
        logger.warning("AdaRMSNorm injection not fully implemented yet, using standard forward")

        expert_outputs = self.action_expert(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=False,
            return_dict=True,
        )

        return expert_outputs.last_hidden_state

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
