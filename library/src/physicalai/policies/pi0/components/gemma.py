# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2025 Physical Intelligence
# SPDX-License-Identifier: Apache-2.0

"""Gemma backbone components for Pi0/Pi0.5 models."""

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

GemmaVariant = Literal["gemma_300m", "gemma_2b"]


@dataclass
class _GemmaConfig:
    vocab_size: int
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int


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
    if variant not in _GEMMA_CONFIGS:
        msg = f"Unknown Gemma variant: {variant}. Available: {list(_GEMMA_CONFIGS.keys())}"
        raise ValueError(msg)
    return _GEMMA_CONFIGS[variant]


class PaliGemmaWithExpert(nn.Module):
    """PaliGemma backbone with action expert for Pi0/Pi0.5."""

    PALIGEMMA_MODEL_IDS: ClassVar[dict[GemmaVariant, str]] = {
        "gemma_300m": "google/paligemma-3b-pt-224",
        "gemma_2b": "google/paligemma-3b-pt-224",
    }

    def __init__(
        self,
        paligemma_variant: GemmaVariant = "gemma_2b",
        action_expert_variant: GemmaVariant = "gemma_300m",
        *,
        use_adarms: bool = False,
        dtype: str = "float32",
        paligemma_model_id: str | None = None,
    ) -> None:
        """Initialize PaliGemma with action expert.

        Args:
            paligemma_variant: width variant for the PaliGemma backbone.
            action_expert_variant: width variant for the action-expert Gemma.
            use_adarms: whether to enable AdaRMS scaling.
            dtype: floating point backend used for kernel weights.
            paligemma_model_id: optional override for the HuggingFace model ID.

        Raises:
            ValueError: if an unsupported paligemma_variant is requested.
        """
        super().__init__()

        self.paligemma_variant = paligemma_variant
        self.action_expert_variant = action_expert_variant
        self.use_adarms = use_adarms
        self._dtype_str = dtype

        if paligemma_variant != "gemma_2b":
            msg = (
                "PaliGemma is only available for the 3B backbone (Gemma-2B width). "
                f"Got paligemma_variant='{paligemma_variant}'."
            )
            raise ValueError(msg)

        self._paligemma_config = _get_gemma_config("gemma_2b")
        self._action_expert_config = _get_gemma_config(action_expert_variant)

        self.paligemma_hidden_size = self._paligemma_config.width
        self.action_expert_hidden_size = self._action_expert_config.width
        self.action_expert_num_layers = self._action_expert_config.depth

        self._paligemma_model_id = paligemma_model_id or self.PALIGEMMA_MODEL_IDS.get(
            paligemma_variant,
            "google/paligemma-3b-pt-224",
        )

        self._paligemma: PaliGemmaForConditionalGeneration | None = None
        self._action_expert: GemmaForCausalLM | None = None
        self._adarms_layers: nn.ModuleList | None = None
        self._initialized = False

    @property
    def dtype(self) -> torch.dtype:
        """Return compute dtype."""
        return torch.bfloat16 if self._dtype_str == "bfloat16" else torch.float32

    def _ensure_loaded(self) -> None:
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

        self._paligemma = PaliGemmaForConditionalGeneration.from_pretrained(  # nosec B615
            self._paligemma_model_id,
            dtype=self.dtype,
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
        self._action_expert = self._action_expert.to(self.dtype)  # type: ignore[assignment]

        if self.use_adarms:
            self._setup_adarms()

        self._initialized = True

    def _setup_adarms(self) -> None:
        if self._action_expert is None:
            return

        hidden_size = self.action_expert_hidden_size
        num_layers = self.action_expert_num_layers
        self._adarms_layers = nn.ModuleList([AdaRMSNorm(hidden_size) for _ in range(num_layers * 2)])

        logger.info("Initialized %d AdaRMSNorm layers for Pi0.5", len(self._adarms_layers))

    @property
    def paligemma(self) -> PaliGemmaForConditionalGeneration:
        """Return PaliGemma model, loading if needed."""  # noqa: DOC501
        self._ensure_loaded()
        if self._paligemma is None:
            msg = "PaliGemma model not loaded"
            raise RuntimeError(msg)
        return self._paligemma

    @property
    def action_expert(self) -> GemmaForCausalLM:
        """Return action expert model, loading if needed."""  # noqa: DOC501
        self._ensure_loaded()
        if self._action_expert is None:
            msg = "Action expert model not loaded"
            raise RuntimeError(msg)
        return self._action_expert

    def embed_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Embed images using PaliGemma's vision encoder."""  # noqa: DOC201
        self._ensure_loaded()
        pixel_values = pixel_values.to(dtype=self.dtype)
        vision_outputs = self.paligemma.vision_tower(pixel_values)
        image_features = vision_outputs.last_hidden_state
        return self.paligemma.multi_modal_projector(image_features)

    def embed_language_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed language tokens using PaliGemma's embedding layer."""  # noqa: DOC201
        self._ensure_loaded()
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
        """Forward pass through PaliGemma and action expert."""  # noqa: DOC201
        self._ensure_loaded()

        prefix_embeds, suffix_embeds = inputs_embeds
        if prefix_embeds is not None:
            prefix_embeds = prefix_embeds.to(self.dtype)
        if suffix_embeds is not None:
            suffix_embeds = suffix_embeds.to(self.dtype)

        prefix_len = prefix_embeds.shape[1] if prefix_embeds is not None else 0
        suffix_len = suffix_embeds.shape[1] if suffix_embeds is not None else 0

        prefix_output = None
        if prefix_embeds is not None:
            prefix_position_ids = position_ids[:, :prefix_len]
            prefix_attention_mask = attention_mask[:, :, :prefix_len, :prefix_len]
            pali_outputs = self.paligemma.language_model(
                inputs_embeds=prefix_embeds,
                attention_mask=prefix_attention_mask,
                position_ids=prefix_position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                return_dict=True,
            )
            prefix_output = pali_outputs.last_hidden_state

            if use_cache:
                past_key_values = pali_outputs.past_key_values

        suffix_output = None
        if suffix_embeds is not None:
            suffix_cond = adarms_cond[1] if adarms_cond is not None else None
            suffix_position_ids = position_ids[:, prefix_len : prefix_len + suffix_len]
            suffix_attention_mask = attention_mask[:, :, prefix_len:, :]

            if self.use_adarms and suffix_cond is not None:
                suffix_output = self._forward_action_expert_with_adarms(
                    suffix_embeds,
                    suffix_attention_mask,
                    suffix_position_ids,
                    suffix_cond,
                    past_key_values,
                )
            else:
                expert_outputs = self.action_expert(
                    inputs_embeds=suffix_embeds,
                    attention_mask=suffix_attention_mask,
                    position_ids=suffix_position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_hidden_states=True,
                    return_dict=True,
                )
                suffix_output = expert_outputs.hidden_states[-1]

        return (prefix_output, suffix_output), past_key_values

    def _forward_action_expert_with_adarms(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        adarms_cond: torch.Tensor,  # noqa: ARG002
        past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
    ) -> torch.Tensor:
        logger.warning("AdaRMSNorm injection not fully implemented yet, using standard forward")
        expert_outputs = self.action_expert(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        return expert_outputs.hidden_states[-1]

    def to_bfloat16_for_selected_params(self, dtype_str: str = "bfloat16") -> None:
        """Convert selected parameters to bfloat16 for memory efficiency."""
        if dtype_str != "bfloat16":
            return

        self._ensure_loaded()
        self.paligemma.to(torch.bfloat16)  # type: ignore[method-call]
        self.action_expert.to(torch.bfloat16)  # type: ignore[method-call]
        logger.info("Converted models to bfloat16")

    def set_trainable_parameters(
        self,
        *,
        tune_paligemma: bool = False,
        tune_action_expert: bool = True,
        tune_vision_encoder: bool = False,
    ) -> None:
        """Set which parameters are trainable."""
        self._ensure_loaded()

        for param in self.paligemma.language_model.parameters():
            param.requires_grad = tune_paligemma

        for param in self.paligemma.vision_tower.parameters():
            param.requires_grad = tune_vision_encoder

        for param in self.paligemma.multi_modal_projector.parameters():
            param.requires_grad = tune_paligemma

        for param in self.action_expert.parameters():
            param.requires_grad = tune_action_expert

        if self._adarms_layers is not None:
            for param in self._adarms_layers.parameters():
                param.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info("Trainable parameters: %d / %d (%.2f%%)", trainable, total, 100 * trainable / total)
