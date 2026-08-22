# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# PrismaticVisionBackbone and PrismaticVisualProjector are vendored from the
# `modeling_prismatic.py` that ships inside the released VLA-Adapter
# checkpoints (https://huggingface.co/VLA-Adapter), itself derived from
# OpenVLA (https://github.com/openvla/openvla). Copyright (c) OpenHelix Team
# and the OpenVLA authors, licensed under the MIT License.

r"""The Prismatic VLM: fused DINOv2 + SigLIP towers feeding a Qwen2 LLM.

The vision modules are **vendored, not reimplemented**. Three checkpoint
behaviours are easy to get wrong and all fail silently — shapes agree, the model
runs, only the success rate suffers:

1. Features come from the **second-to-last** block via
   ``get_intermediate_layers(n={num_blocks - 2})``, not ``forward_features``.
2. ``LayerScale`` params are renamed ``gamma`` -> ``scale_factor`` (HF overwrites
   names containing ``gamma``). Without the patch, 48 tensors fail to load.
3. Each image arrives as **six** channels, three normalised per tower.

Upstream's ``prismatic`` package pins torch 2.2 / transformers 4.40 / timm 0.9,
so it cannot be a dependency; the checkpoint-local ``modeling_prismatic.py``
needs only timm + transformers and is the vendoring source.

:class:`VLM` takes its API from upstream's ``PrismaticVLM`` but its module names
from ``PrismaticForConditionalGeneration`` — the class the checkpoints export
through — so ``VLM.state_dict()`` matches ``model.safetensors`` exactly (982
tensors, no prefix surgery). Mostly frozen: towers and LLM stay fixed while the
projector and ``action_queries`` train; see :attr:`VLM.trainable_module_keys`.

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
from collections.abc import Callable
from functools import partial
from typing import Any

import timm
import torch
from torch import nn

from physicalai.policies.vla_adapter.config import VLAAdapterConfig

logger = logging.getLogger(__name__)

# Channels per image once both towers' normalised copies are stacked.
CHANNELS_PER_IMAGE = 6
# Channels a single tower consumes.
CHANNELS_PER_TOWER = 3
# VLA-Adapter always fuses DINOv2 with SigLIP.
NUM_VISION_TOWERS = 2


def unpack_tuple(fn: Callable[[Any], tuple[Any]]) -> Callable[[Any], Any]:
    """Unwrap a single-element sequence returned by a monkey-patched forward.

    DEVIATION FROM UPSTREAM: upstream tests ``isinstance(result, tuple)``,
    correct for timm 0.9. timm 1.0 returns a *list*, which a tuple-only check
    passes straight through, handing the caller a list where a tensor is
    expected.

    Args:
        fn: Function whose result should be unwrapped.

    Returns:
        Wrapper returning ``result[0]`` for a tuple or list.
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, (tuple, list)) else result

    return wrapper


def _ls_new_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Replacement ``LayerScale.forward`` reading ``scale_factor``.

    Args:
        self: The patched ``LayerScale`` module.
        x: Input activations.

    Returns:
        Scaled activations.
    """
    return x.mul_(self.scale_factor) if self.inplace else x * self.scale_factor


def ls_apply_patch(ls_module: nn.Module) -> None:
    """Rename a ``LayerScale`` module's ``gamma`` parameter to ``scale_factor``.

    The checkpoints store ``ls1.scale_factor``; without this, 48 tensors fail
    to load.

    Args:
        ls_module: A timm ``LayerScale`` instance, patched in place.
    """
    from timm.models.vision_transformer import LayerScale  # noqa: PLC0415

    ls_module.scale_factor = nn.Parameter(ls_module.gamma.clone())
    ls_module.forward = _ls_new_forward.__get__(ls_module, LayerScale)
    del ls_module.gamma


class PrismaticVisionBackbone(nn.Module):
    """Fused DINOv2 + SigLIP towers, vendored from the checkpoint modeling code.

    Attribute names (``featurizer``, ``fused_featurizer``) mirror the
    checkpoint's tensor keys.
    """

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

        Raises:
            ValueError: If exactly two tower ids are not supplied.
        """
        super().__init__()
        self.num_images_in_input = 1

        if len(timm_model_ids) != NUM_VISION_TOWERS:
            msg = f"VLA-Adapter always fuses exactly {NUM_VISION_TOWERS} vision towers, got {len(timm_model_ids)}."
            raise ValueError(msg)

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
    def _create_featurizer(model_id: str, img_size: int, act_layer: str | None, *, pretrained: bool) -> nn.Module:
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
        featurizer = timm.create_model(
            model_id,
            pretrained=pretrained,
            num_classes=0,
            img_size=img_size,
            act_layer=act_layer,
        )

        num_blocks = len(featurizer.blocks)
        featurizer.forward = unpack_tuple(partial(featurizer.get_intermediate_layers, n={num_blocks - 2}))

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


class VLM(nn.Module):
    """The Prismatic VLM, exposing per-layer hidden states to the action head.

    Holds exactly the four submodules ``model.safetensors`` contains, so its
    ``state_dict()`` matches that file key for key.
    """

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
        self.language_model = self._build_language_model()
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

    def _build_language_model(self) -> nn.Module:
        """Build the Qwen2 language model, taking its geometry from the reference.

        Geometry is never configured locally: it is read from
        ``llm_model_name`` so the model and its tokenizer cannot disagree, and
        so the action head's depth follows the LLM's automatically.

        Weights are pretrained unless disabled. The LLM is frozen by default, so
        a randomly initialised one would leave the head reading noise. They are
        loaded as float32 because the published weights are bfloat16, which
        would not match the float32 vision towers and head.

        Returns:
            A ``Qwen2Model`` with caching disabled.
        """
        from transformers import Qwen2Config, Qwen2Model  # noqa: PLC0415

        name = self.config.llm_model_name
        llm_config = Qwen2Config.from_pretrained(name)
        llm_config.use_cache = False

        if not self.config.load_pretrained_backbone:
            return Qwen2Model(llm_config)

        model = Qwen2Model.from_pretrained(name, dtype=torch.float32)
        model.config.use_cache = False
        return model

    def apply_trainability(self) -> None:
        """Apply the trainability flags and record the outcome.

        Only the two large pretrained backbones are configurable. The visual
        projector and action queries always train: absent a checkpoint the
        projector is randomly initialised and the queries are zero, so freezing
        either would leave the action head reading noise. Upstream agrees —
        ``freeze_backbones`` trains the projector in every stage but
        ``last-layer-finetune``, and ``finetune.py`` forces ``requires_grad``
        on the action queries.
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
