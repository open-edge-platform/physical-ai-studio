# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Vendored from RLWRLD/RLDX-1 (Apache-2.0)

"""Export-only graph-safe view over a trained :class:`Rldx1Model`.

RLDX-1's eager forward runs data-dependent ops that ``torch.export`` /
``torch.onnx.export`` cannot trace (``fast_pos_embed_interpolate`` with a
symbolic step count, ``rot_pos_emb``, ``cu_seqlens``, the VTC
``LayerWrapper`` token compression, and ``get_rope_index`` M-RoPE indices).

Upstream (``RLWRLD/RLDX-1`` ``rldx/inference``) solves this with a family of
``graph_safe_*`` wrappers that **share the trained parameters by reference**
(no state-dict copy) and precompute every data-dependent tensor once, eagerly,
in ``__init__`` from a fixed example input -- turning ``forward`` into a static
graph. This module ports that pattern for Studio export.

``GraphSafeRldx1Model`` keeps the exact ``forward(batch)`` contract of
:class:`~physicalai.policies.rldx1.model.Rldx1Model` so the export mixin can
trace it unchanged; the policy swaps it in for tracing backends via
``Rldx1._graph_safe_export_model``.

Port status (v1 = OpenVINO / ONNX, CPU, no motion / memory / RTC):
    [x] vision encoder  -> GraphSafeQwen3VLVisionModel  (pos_embeds, rotary,
        cu_seqlens, static attention splits)
    [x] LLM / VTC       -> GraphSafeQwen3VLTextModel     (static begin/end
        compression indices via _find_compress_info)
    [x] backbone glue   -> GraphSafeQwen3VLBackbone      (embed/scatter/cog
        tokens + fixed-shape dynamic prompt tensors)
    [x] action head     -> reused eagerly. The MSAT denoising loop is a fixed
        ``range`` with static shapes and no data-dependent guards; the only
        eager guard (``encoder_attention_mask.all()``) is skipped by passing a
        mask-free ``backbone_output`` (an all-ones mask is set to ``None``
        anyway, so this is numerically identical). Note: the MSAT RoPE uses
        complex64 ops (``view_as_complex`` / ``view_as_real``); those are an
        ONNX/OpenVINO op-support concern that a graph-safe port would not fix
        either -- if tracing fails there, a real-valued RoPE rewrite is the
        targeted follow-up.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from torch import nn
from transformers import BatchFeature

from physicalai.policies.rldx1 import constants as rldx1_constants
from physicalai.policies.rldx1.components.backbone.graph_safe_backbone import GraphSafeQwen3VLBackbone

if TYPE_CHECKING:
    from physicalai.policies.rldx1.config import Rldx1Config
    from physicalai.policies.rldx1.model import Rldx1Model

ACTION_PRED = rldx1_constants.ACTION_PRED
ATTENTION_MASK = rldx1_constants.ATTENTION_MASK
BACKBONE_FEATURES = rldx1_constants.BACKBONE_FEATURES
EMBODIMENT_ID = rldx1_constants.EMBODIMENT_ID
IMAGE_GRID_THW = rldx1_constants.IMAGE_GRID_THW
INPUT_IDS = rldx1_constants.INPUT_IDS
PIXEL_VALUES = rldx1_constants.PIXEL_VALUES
POSITION_IDS = rldx1_constants.POSITION_IDS
STATE = rldx1_constants.STATE


class GraphSafeRldx1Model(nn.Module):
    """Static, trace-safe view over a trained ``Rldx1Model``.

    Shares parameters with the wrapped model by reference and precomputes all
    data-dependent buffers from ``input_sample`` in the constructor. Call
    :meth:`restore` after export to undo the in-place attention swaps and leave
    the trained model untouched.
    """

    def __init__(
        self,
        model: Rldx1Model,
        *,
        input_ids: torch.Tensor,
        image_grid_thw: torch.Tensor,
        num_views: torch.Tensor,
        config: Rldx1Config,
        output_action_dim: int,
        embodiment_id: torch.Tensor,
        compress_input_ids: torch.Tensor | None = None,
    ) -> None:
        """Build the graph-safe view.

        Args:
            model: The trained ``Rldx1Model`` whose parameters are reused by
                reference (never copied).
            input_ids: Export sample token ids used to precompute static
                buffers.
            image_grid_thw: Export sample image grid tensor used for vision
                positional indexing precompute.
            num_views: Camera-view count used for static VTC compression setup.
            config: The policy config.
            output_action_dim: Real exported action width after trimming padded
                ``max_action_dim`` down to the environment action dimension.
            embodiment_id: Optional fixed embodiment id to bake as a buffer.
            compress_input_ids: Optional canonical, contract-derived ``input_ids``
                used only to resolve the static VTC compression span, so random
                trace tokens cannot shift the baked span from the runtime layout.
                Left-padded to ``config.tokenizer_max_length`` to match the
                padded export ``input_ids``.
        """
        super().__init__()
        self._config = config
        self._output_action_dim = output_action_dim

        # Wrap-by-reference: reuse the trained submodules, precompute static buffers.
        self.gs_backbone = GraphSafeQwen3VLBackbone(
            model.backbone,
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            num_views=num_views,
            config=config,
            compress_input_ids=compress_input_ids,
        )
        self._action_model = model.action_model

        self.register_buffer("embodiment_id", embodiment_id.clone())

    def restore(self) -> None:
        """Undo in-place submodule swaps so the trained model is left intact."""
        self.gs_backbone.restore()

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict an action chunk from a preprocessed batch.

        Mirrors :meth:`Rldx1Model.get_action` so the export mixin traces this
        module with the same single-dict input contract: graph-safe backbone ->
        eager action head.

        Returns:
            Predicted action tensor of shape ``(B, action_horizon, action_dim)``.
        """
        backbone_features = self.gs_backbone(
            input_ids=batch[INPUT_IDS],
            position_ids=batch[POSITION_IDS],
            attention_mask=batch[ATTENTION_MASK],
            pixel_values=batch[PIXEL_VALUES],
        )

        # Match Rldx1Model.prepare_input: cast floating inputs (e.g. state) to the
        # model dtype so they agree with the bf16 weights (get_action skips that cast).
        dtype = next(self._action_model.parameters()).dtype
        cast_batch = {
            key: value.to(dtype) if torch.is_tensor(value) and torch.is_floating_point(value) else value
            for key, value in batch.items()
        }
        # Inject the baked embodiment_id (dropped from the graph inputs).
        cast_batch[EMBODIMENT_ID] = cast("torch.Tensor", self.embodiment_id)

        # A mask-free backbone_output skips the eager get_action's
        # ``encoder_attention_mask.all()`` guard (None short-circuits); an
        # all-ones mask would be dropped to None anyway, so results match.
        backbone_output = BatchFeature(data={BACKBONE_FEATURES: backbone_features})
        action_input = self._action_model.prepare_input(cast_batch)
        result = self._action_model.get_action(backbone_output, action_input)
        return result[ACTION_PRED][..., : self._config.action_horizon, : self._output_action_dim]
