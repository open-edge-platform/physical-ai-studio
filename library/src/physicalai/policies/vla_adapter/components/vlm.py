# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# PrismaticVisionBackbone and PrismaticVisualProjector are vendored from the
# `modeling_prismatic.py` that ships inside the released VLA-Adapter
# checkpoints (https://huggingface.co/VLA-Adapter), itself derived from
# OpenVLA (https://github.com/openvla/openvla). Copyright (c) OpenHelix Team
# and the OpenVLA authors, licensed under the MIT License.

r"""The Prismatic VLM: fused DINOv2 + SigLIP towers feeding a Qwen2 LLM.

Mostly frozen: towers and LLM stay fixed while the projector and
``action_queries`` train; see :attr:`VLM.trainable_module_keys`.

``action_queries`` lives here, not with the head, because it is an *input* to
the LLM. The head reads the hidden states produced *at* those positions, never
the embeddings themselves.

Sequence layout consumed by the action head::

    [ 512 fused vision patches ][ prompt tokens ][ 64 action queries ]
      \_____ "task" (h_t) _____/                  \__ "action" (h_a) __/

Prompt tokens are not passed to the head; they shape the action-query states
through attention inside the LLM. Hidden states from every layer are returned
stacked, because head block ``i`` is conditioned on layer ``i + 1``. The LLM
carries no ``lm_head`` — it is a feature extractor.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import nn

from physicalai.policies.vla_adapter.config import (
    CHANNELS_PER_IMAGE,
    CHANNELS_PER_TOWER,
    NUM_VISION_TOWERS,
)

if TYPE_CHECKING:
    from timm.layers import LayerScale
    from timm.models import VisionTransformer
    from transformers import Qwen2Config, Qwen2Model
    from transformers.modeling_outputs import BaseModelOutputWithPast

    from physicalai.policies.vla_adapter.config import VLAAdapterConfig

logger = logging.getLogger(__name__)


def _ls_new_forward(self: Any, x: torch.Tensor) -> torch.Tensor:  # noqa: ANN401
    """Replacement ``LayerScale.forward`` reading ``scale_factor``.

    Args:
        self: The patched ``LayerScale`` module.
        x: Input activations.

    Returns:
        Scaled activations.
    """
    return x.mul_(self.scale_factor) if self.inplace else x * self.scale_factor


def ls_apply_patch(ls_module: LayerScale) -> None:
    """Rename a ``LayerScale`` module's ``gamma`` parameter to ``scale_factor``.

    The LIBERO-compatible weights store ``ls1.scale_factor``; without this, 48 tensors fail
    to load.

    Args:
        ls_module: A timm ``LayerScale`` instance, patched in place.
    """
    from timm.models.vision_transformer import LayerScale  # noqa: PLC0415

    ls_module.scale_factor = nn.Parameter(ls_module.gamma.clone())
    ls_module.forward = _ls_new_forward.__get__(ls_module, LayerScale)  # type: ignore[method-assign]
    del ls_module.gamma


class PrismaticVisionBackbone(nn.Module):
    """Fused DINOv2 + SigLIP towers, vendored from the checkpoint modeling code.

    ``featurizer`` is DINOv2 (width 1024), ``fused_featurizer`` is SigLIP
    (width 1152). The names are upstream's, kept verbatim so the released
    checkpoints load without remapping these tensors.
    """

    featurizer: VisionTransformer
    fused_featurizer: VisionTransformer
    embed_dim: int

    def __init__(
        self,
        image_sizes: list[int],
        timm_model_ids: list[str],
        timm_override_act_layers: list[str | None],
        *,
        pretrained: bool = False,
    ) -> None:
        """Build both towers and apply the LayerScale patch.

        Args:
            image_sizes: Input resolution per tower.
            timm_model_ids: timm identifiers for the DINOv2 and SigLIP towers.
            timm_override_act_layers: Activation-layer override per tower.
            pretrained: Fetch pretrained tower weights. The towers are frozen by
                default, so leaving this False means the head reads noise.

        """
        super().__init__()
        self.num_images_in_input = 1

        self.featurizer = self._create_featurizer(
            model_id=timm_model_ids[0],
            img_size=image_sizes[0],
            act_layer=timm_override_act_layers[0],
            pretrained=pretrained,
        )
        self.fused_featurizer = self._create_featurizer(
            model_id=timm_model_ids[1],
            img_size=image_sizes[1],
            act_layer=timm_override_act_layers[1],
            pretrained=pretrained,
        )
        self.embed_dim = self.featurizer.embed_dim + self.fused_featurizer.embed_dim

        self._patch_layer_scales()

    @staticmethod
    def _create_featurizer(
        model_id: str,
        img_size: int,
        act_layer: str | None,
        *,
        pretrained: bool,
    ) -> VisionTransformer:
        """Create a timm featurizer emitting second-to-last-block patches.

        The monkey-patched ``forward`` is load-bearing: Prismatic reads block
        ``num_blocks - 2``, and ``get_intermediate_layers`` already drops prefix
        (CLS / register) tokens.

        Args:
            model_id: timm model identifier.
            img_size: Square input resolution.
            act_layer: Activation-layer override.
            pretrained: Fetch pretrained weights.

        Returns:
            The configured featurizer.
        """
        import timm  # noqa: PLC0415

        # `create_model` is a factory keyed on a string, so it is declared
        # `-> Module`. The concrete class here is the ViT we asked for, and the
        # attributes used below (`blocks`, `embed_dim`, `patch_embed`,
        # `get_intermediate_layers`) live on `VisionTransformer`, not `Module`.
        featurizer = cast(
            "VisionTransformer",
            timm.create_model(
                model_id,
                pretrained=pretrained,
                num_classes=0,
                img_size=img_size,
                act_layer=act_layer,
            ),
        )

        second_to_last = [len(featurizer.blocks) - 2]

        def _forward(x: torch.Tensor) -> torch.Tensor:
            # A single requested index yields a one-element list.
            # (Upstream passes a set here, which is outside the declared
            #  signature; a one-element list is identical in effect.)
            return featurizer.get_intermediate_layers(x, n=second_to_last)[0]

        featurizer.forward = _forward  # type: ignore[method-assign,assignment]

        return featurizer

    def _patch_layer_scales(self) -> None:
        """Patch every ``LayerScale`` module in both towers."""
        from timm.models.vision_transformer import LayerScale  # noqa: PLC0415

        for tower in (self.featurizer, self.fused_featurizer):
            for module in tower.modules():
                if isinstance(module, LayerScale):
                    ls_apply_patch(module)

    def get_num_patches(self) -> int:
        """Count vision patches per image.

        Returns:
            Patch tokens per image, per tower.
        """
        return int(self.featurizer.patch_embed.num_patches)

    def get_num_images_in_input(self) -> int:
        """Report how many camera views the backbone expects.

        Returns:
            Images per sample.
        """
        return self.num_images_in_input

    def set_num_images_in_input(self, num_images_in_input: int) -> None:
        """Set how many camera views the backbone expects.

        Args:
            num_images_in_input: Images per sample.
        """
        self.num_images_in_input = num_images_in_input

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode channel-stacked camera views into fused patch tokens.

        Args:
            pixel_values: ``(B, 6 * num_images, H, W)`` — three channels per
                tower, per image.

        Returns:
            ``(B, num_images * num_patches, embed_dim)``.
        """
        images = torch.split(pixel_values, [CHANNELS_PER_IMAGE] * self.num_images_in_input, dim=1)

        all_patches = []
        for img in images:
            img_regular, img_fused = torch.split(img, [CHANNELS_PER_TOWER, CHANNELS_PER_TOWER], dim=1)
            patches = self.featurizer(img_regular)
            patches_fused = self.fused_featurizer(img_fused)
            all_patches.append(torch.cat([patches, patches_fused], dim=2))

        return torch.cat(all_patches, dim=1)


class PrismaticVisualProjector(nn.Module):
    """Projects fused *visual* patches into the LLM embedding space.

    Vendored from upstream's ``PrismaticProjector``; renamed to distinguish it
    from the head-side ``ProprioProjector``.

    Applied **per token** — it changes width, never token count. The token count
    grows earlier, when :class:`PrismaticVisionBackbone` concatenates views.
    Vision only: text is already in the LLM's embedding space and is simply
    concatenated alongside, never projected.

    Always the ``fused-gelu-mlp`` variant recorded in the checkpoint's
    ``arch_specifier``: 2176 -> 8704 -> 896 -> 896. Upstream's pretraining code
    calls the same module ``FusedMLPProjector``; the ``fc1``/``fc2``/``fc3``
    naming used here is the exported one the checkpoint keys carry.
    """

    def __init__(self, vision_dim: int, llm_dim: int) -> None:
        """Build the three-layer projector.

        Args:
            vision_dim: Input (fused visual) width.
            llm_dim: Output width, matching the LLM hidden size.
        """
        super().__init__()
        self.vision_dim, self.llm_dim = vision_dim, llm_dim

        initial_projection_dim = 4 * vision_dim
        self.fc1 = nn.Linear(self.vision_dim, initial_projection_dim, bias=True)
        self.fc2 = nn.Linear(initial_projection_dim, self.llm_dim, bias=True)
        self.fc3 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
        self.act_fn1 = nn.GELU()
        self.act_fn2 = nn.GELU()

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        """Project visual patches into the LLM embedding space.

        Args:
            img_patches: Fused patches ``(B, S, vision_dim)``.

        Returns:
            ``(B, S, llm_dim)``.
        """
        projected_features = self.fc1(img_patches)
        projected_features = self.act_fn1(projected_features)
        projected_features = self.fc2(projected_features)
        projected_features = self.act_fn2(projected_features)
        return self.fc3(projected_features)


class LanguageModel(nn.Module):
    """Model that wraps LLM transformer."""

    model: Qwen2Model

    def __init__(self, model_name: str, *, pretrained: bool = True) -> None:
        """Build the Qwen2 transformer.

        Args:
            model_name: HuggingFace identifier supplying both geometry and weights.
            pretrained: Boolean indicating whether to load published weights.
        """
        super().__init__()

        from transformers import Qwen2Config, Qwen2Model  # noqa: PLC0415

        if pretrained:
            self.model = Qwen2Model.from_pretrained(model_name, dtype=torch.float32)
        else:
            self.model = Qwen2Model(Qwen2Config.from_pretrained(model_name))

        # Every call is a single forward pass; a key/value cache would only add
        # memory and break the static shapes export depends on.
        self.model.config.use_cache = False

    @property
    def config(self) -> Qwen2Config:
        """Configuration of the wrapped transformer.

        Returns:
            The Qwen2 configuration.
        """
        return self.model.config

    def get_input_embeddings(self) -> nn.Module:
        """Return the token embedding table.

        Returns:
            The wrapped transformer's input embeddings.
        """
        return self.model.get_input_embeddings()

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        use_cache: bool = False,
        output_hidden_states: bool = True,
        return_dict: bool = True,
    ) -> BaseModelOutputWithPast:
        """Run the wrapped transformer.

        Args:
            inputs_embeds: Pre-embedded sequence ``(B, S, llm_dim)``.
            attention_mask: Mask over the sequence.
            use_cache: Retain key/value cache; off, as every call is one pass.
            output_hidden_states: Return all per-layer hidden states.
            return_dict: Return a dataclass rather than a tuple.

        Returns:
            The transformer output, carrying ``hidden_states``.
        """
        return self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class VLM(nn.Module):
    """The Prismatic VLM, exposing per-layer hidden states to the action head."""

    def __init__(self, config: VLAAdapterConfig) -> None:
        """Assemble the towers, projector, language model and action queries.

        Args:
            config: Supplies backbone ids, LLM geometry and trainability flags.
        """
        super().__init__()

        self.config = config
        self.arch_specifier = config.arch_specifier

        # init vision
        self.vision_backbone = PrismaticVisionBackbone(
            image_sizes=[config.image_size[0]] * NUM_VISION_TOWERS,
            timm_model_ids=list(config.vision_backbone_ids),
            timm_override_act_layers=[None, None],
            pretrained=config.load_pretrained_backbone,
        )
        self.vision_backbone.set_num_images_in_input(config.num_images_in_input)

        # init LLM
        self.language_model = LanguageModel(
            config.llm_model_name,
            pretrained=config.load_pretrained_backbone,
        )
        self.llm_dim = self.language_model.config.hidden_size
        self.num_layers = self.language_model.config.num_hidden_layers

        # init projector
        self.projector = PrismaticVisualProjector(
            vision_dim=self.vision_backbone.embed_dim,
            llm_dim=self.llm_dim,
        )

        # init action layer
        self.action_queries = nn.Embedding(config.num_action_queries, self.llm_dim)
        self.action_queries.weight.data.zero_()

        self.all_module_keys = ["vision_backbone", "language_model", "projector", "action_queries"]
        self.trainable_module_keys: list[str] = []
        self.apply_trainability()

    def apply_trainability(self) -> None:
        """Apply the trainability flags and record the outcome.

        Only the two large pretrained backbones are configurable. The visual
        projector and action queries always train.
        """
        wants_grad = {
            "vision_backbone": self.config.train_vision_backbone,
            "language_model": self.config.train_llm,
            "projector": True,
            "action_queries": True,
        }
        for key, flag in wants_grad.items():
            getattr(self, key).requires_grad_(requires_grad=flag)

        self.trainable_module_keys = [key for key, flag in wants_grad.items() if flag]

    @property
    def num_vision_tokens(self) -> int:
        """Total fused vision patch tokens across all camera views.

        Returns:
            ``num_patches * num_images_in_input``.
        """
        return self.vision_backbone.get_num_patches() * self.config.num_images_in_input

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the VLM and return stacked per-layer hidden states.

        Args:
            pixel_values: Channel-stacked pixels ``(B, 6 * num_images, H, W)``.
            input_ids: Prompt token ids ``(B, T_text)``.
            attention_mask: Optional prompt mask ``(B, T_text)``.

        Returns:
            ``(B, num_layers + 1, num_vision_tokens + num_action_queries,
            llm_dim)`` — only vision and action-query positions are kept.
        """
        batch = input_ids.shape[0]

        vision_features = self.vision_backbone(pixel_values)
        vision_embeds = self.projector(vision_features)

        text_embeds = self.language_model.get_input_embeddings()(input_ids)
        queries = self.action_queries.weight.unsqueeze(0).expand(batch, -1, -1)

        inputs_embeds = torch.cat(
            [vision_embeds, text_embeds.to(vision_embeds.dtype), queries.to(vision_embeds.dtype)],
            dim=1,
        )

        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
        ones = partial(torch.ones, dtype=attention_mask.dtype, device=attention_mask.device)
        full_mask = torch.cat(
            [
                ones((batch, vision_embeds.shape[1])),
                attention_mask,
                ones((batch, self.config.num_action_queries)),
            ],
            dim=1,
        )

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )

        num_vision = vision_embeds.shape[1]
        num_queries = self.config.num_action_queries

        # Keep only the task (vision) and action-query positions from every
        # layer, stacked on a new layer axis for the head.
        per_layer = [
            torch.cat([state[:, :num_vision], state[:, -num_queries:]], dim=1).unsqueeze(1)
            for state in outputs.hidden_states
        ]
        return torch.cat(per_layer, dim=1)
