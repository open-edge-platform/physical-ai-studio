# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Vendored from RLWRLD/RLDX-1 (Apache-2.0)

"""Graph-safe unified Qwen3-VL backbone (vision + glue + LLM) for export.

Ports ``rldx/inference/backbone/model/graph_safe_qwen3vl_backbone_model.py``
from RLWRLD/RLDX-1 for the v1 export path (no motion / memory / RTC).

Composes :class:`GraphSafeQwen3VLVisionModel` and
:class:`GraphSafeQwen3VLTextModel` with the embedding / image-scatter /
cog-token glue. The tracer observes fixed-shape dynamic prompt tensors
(``input_ids`` / ``position_ids`` / ``attention_mask``) and image features.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch
from torch import nn

from physicalai.policies.rldx1.components.backbone.graph_safe_text import GraphSafeQwen3VLTextModel
from physicalai.policies.rldx1.components.backbone.graph_safe_vision import GraphSafeQwen3VLVisionModel

_PIXEL_VALUES_FLAT_NDIM = 3

if TYPE_CHECKING:
    from physicalai.policies.rldx1.config import Rldx1Config


class GraphSafeQwen3VLBackbone(nn.Module):
    """Static, trace-safe unified VLM backbone (vision + glue + LLM)."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        input_ids: torch.Tensor,
        image_grid_thw: torch.Tensor,
        num_views: torch.Tensor,
        config: Rldx1Config,
        compress_input_ids: torch.Tensor | None = None,
    ) -> None:
        """Build the graph-safe backbone.

        Args:
            backbone: The trained ``VTCQwen3VLBackbone`` (params reused by
                reference).
            input_ids: The padded export token ids.
            image_grid_thw: Export image grid tensor used to precompute vision
                buffers.
            num_views: Camera-view count used by VTC compression.
            config: Policy config (``attn_implementation`` / ``num_views``).
            compress_input_ids: Optional canonical, contract-derived ``input_ids``
                (same length as ``input_ids``) used only to resolve
                the static VTC compression span. Guards against random trace
                tokens shifting the baked span away from the runtime layout.
        """
        super().__init__()
        backbone_any = cast("Any", backbone)
        inner = backbone_any.qwen_model.model
        self.qwen_config = inner.config

        self.n_cog_tokens = (
            int(backbone_any.n_cog_tokens) if bool(getattr(backbone_any, "use_cog_tokens", False)) else 0
        )
        self.cog_mode = str(backbone_any.cog_mode)
        static_num_views = int(num_views.item()) if isinstance(num_views, torch.Tensor) else int(num_views)

        self.gs_visual = GraphSafeQwen3VLVisionModel(inner.visual, image_grid_thw)
        self.gs_text = GraphSafeQwen3VLTextModel(
            inner.language_model,
            input_ids,
            n_cog_tokens=self.n_cog_tokens,
            attn_impl=config.attn_implementation,
            num_views=static_num_views,
            compress_input_ids=compress_input_ids,
        )

        # Shared modules (references, not owned).
        self.embed_tokens = cast("nn.Module", self.gs_text._text_model.embed_tokens)  # noqa: SLF001
        self.qwen_linear = cast("nn.Module", backbone_any.qwen_linear)
        self.image_token_id = inner.config.image_token_id

        if self.n_cog_tokens > 0 and hasattr(backbone_any, "cog_emb"):
            cog_emb = cast("torch.Tensor", backbone_any.cog_emb)
            self.register_buffer("static_cog_emb", cog_emb.data.clone())
        else:
            self.static_cog_emb = None

    def restore(self) -> None:
        """Undo the vision encoder's in-place attention swap."""
        self.gs_visual.restore()

    def forward(  # noqa: PLR0914
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        """Encode the VLM inputs into backbone features.

        Args:
            input_ids: Token ids for the text input.
            position_ids: Position ids for the text input.
            attention_mask: Attention mask for the text input.
            pixel_values: Image tensor for the visual input.

        Returns:
            Backbone features ``(B, M_out, D)`` -- the cog tokens in
            ``cog_only`` mode.
        """
        image_mask_2d = input_ids == self.image_token_id
        if pixel_values.ndim == _PIXEL_VALUES_FLAT_NDIM:
            pixel_values = pixel_values.reshape(-1, pixel_values.shape[-1])
        pixel_values = pixel_values.type(self.gs_visual.dtype)

        image_emb, deepstack_features = self.gs_visual(pixel_values)

        embed_tokens = cast("Any", self.embed_tokens)
        dtype = embed_tokens.weight.dtype
        image_emb = image_emb.to(dtype=dtype)

        token_emb = cast("torch.Tensor", embed_tokens(input_ids))
        image_mask_3d = image_mask_2d.unsqueeze(-1).expand_as(token_emb)
        token_emb = token_emb.masked_scatter(image_mask_3d, image_emb)

        if self.n_cog_tokens > 0 and self.static_cog_emb is not None:
            meta = self.static_cog_emb.to(dtype).unsqueeze(0).expand(token_emb.size(0), -1, -1)
            full_emb = torch.cat([token_emb, meta], dim=1)
        else:
            full_emb = token_emb

        deepstack_add = None
        if len(deepstack_features) > 0:
            batch, length_full, dim = full_emb.shape
            vis_mask_full = torch.cat(
                [
                    image_mask_2d,
                    torch.zeros(batch, self.n_cog_tokens, dtype=torch.bool, device=full_emb.device),
                ],
                dim=1,
            )
            vis_mask_full_3d = vis_mask_full.unsqueeze(-1).expand(batch, length_full, dim)
            ds_list = []
            for ds_feat in deepstack_features:
                ds_full = torch.zeros_like(full_emb).masked_scatter(vis_mask_full_3d, ds_feat.to(dtype))
                ds_list.append(ds_full)
            deepstack_add = torch.stack(ds_list, dim=0)  # (N_ds, B, L_full, D)

        lm_out = self.gs_text(
            inputs_embeds=full_emb,
            position_ids=position_ids,
            attention_mask=attention_mask,
            deepstack_add=deepstack_add,
        )
        hidden_states = lm_out.last_hidden_state

        if self.n_cog_tokens > 0 and self.cog_mode == "cog_only":
            hidden_states = hidden_states[:, -self.n_cog_tokens :, :]

        return cast("torch.Tensor", self.qwen_linear(hidden_states))
