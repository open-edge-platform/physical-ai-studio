# Copyright (C) 2026 Xiaomi Corporation.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-VL backbone shim for the XR1 policy.

The reference implementation ships a machine-generated verbatim copy of the stock
``transformers`` Qwen3-VL model (``mibot/models/VLM/qwen3vl.py``, ~1500 lines). Its
only *functional* difference from the library model is that it surfaces the 3D
MRoPE ``position_ids`` on its output, because the DiT action expert continues the
backbone's position grid into its own query tokens.

Vendoring that copy would pin us to the ``transformers`` release it was generated
from (upstream pins ``4.57.1``, this library requires ``>=5.5,<5.6``), so this
module subclasses the installed :class:`transformers.Qwen3VLForConditionalGeneration`
instead and adds back only that behaviour. All VLM numerics are inherited from
stock ``transformers``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import torch
from transformers import Qwen3VLForConditionalGeneration

if TYPE_CHECKING:
    from collections.abc import Iterable

    from transformers.cache_utils import Cache


@dataclass
class XR1VLMOutput:
    """Backbone outputs consumed by the DiT action expert.

    Attributes:
        past_key_values: Per-layer key/value cache, one ``(key, value)`` pair per
            layer, each of shape ``(batch, kv_heads, seq, head_dim)``.
        position_ids: 3D MRoPE position grid of shape ``(3, batch, seq)``. The DiT
            continues its own positions from ``position_ids.max(dim=-1) + 1``.
        attention_mask: Padding mask of shape ``(batch, seq)`` covering the cached
            prefix, used to build the DiT's cross-attention mask.
        cache: The backbone's own cache object, which :meth:`XR1Qwen3VL.continue_sequence`
            extends. ``past_key_values`` above is a snapshot taken before any such
            extension, so the action expert always reads the prompt alone.
    """

    past_key_values: list[tuple[torch.Tensor, torch.Tensor]]
    position_ids: torch.Tensor
    attention_mask: torch.Tensor
    cache: Cache | None = None


class XR1Qwen3VL(Qwen3VLForConditionalGeneration):
    """Stock Qwen3-VL that also returns its 3D MRoPE position ids and cache.

    Stock ``transformers`` computes the 3D position ids internally and discards
    them, returning only ``rope_deltas``. :meth:`encode` recomputes them with the
    model's own :meth:`compute_3d_position_ids` and returns them alongside the
    key/value cache.
    """

    def encode(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        **kwargs: Any,  # noqa: ANN401 - forwarded verbatim to the backbone
    ) -> XR1VLMOutput:
        """Run the backbone and return everything the action expert needs.

        Args:
            input_ids: Prompt token ids of shape ``(batch, seq)``.
            attention_mask: Padding mask of shape ``(batch, seq)``. Defaults to
                all-ones.
            pixel_values: Flattened image patches, when images are present.
            image_grid_thw: Image grid sizes matching ``pixel_values``.
            mm_token_type_ids: Per-position multimodal token types from the processor.
                Required by ``transformers`` for multimodal RoPE whenever images are
                present.
            **kwargs: Extra keyword arguments forwarded to the backbone.

        Returns:
            The cache, the 3D position grid and the padding mask.

        Raises:
            ValueError: If the backbone returns no key/value cache, which means
                caching was disabled and the action expert has nothing to read.
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        position_ids = self.model.compute_3d_position_ids(
            input_ids=input_ids,
            inputs_embeds=None,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
            mm_token_type_ids=mm_token_type_ids,
        )

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            use_cache=True,
            # The action expert never reads token logits; keeping one position
            # avoids materializing a (batch, seq, vocab) tensor.
            logits_to_keep=1,
            **kwargs,
        )

        if outputs.past_key_values is None:
            msg = "Qwen3-VL returned no key/value cache; the DiT action expert requires use_cache=True"
            raise ValueError(msg)

        return XR1VLMOutput(
            # Snapshot the per-layer tensors now. ``continue_sequence`` concatenates
            # into fresh tensors and rebinds the cache's own references, so this list
            # keeps pointing at the prompt-only keys and values.
            past_key_values=as_key_value_list(outputs.past_key_values),
            position_ids=normalize_position_ids(position_ids, input_ids),
            attention_mask=attention_mask,
            cache=outputs.past_key_values,
        )

    def continue_sequence(
        self,
        inputs_embeds: torch.Tensor,
        *,
        cache: Cache,
        prompt_attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run the text stack over embeddings appended after an encoded prompt.

        The reference implementation puts the choice head's query tokens in the same
        chat turn as the prompt and then *truncates* the key/value cache the action
        expert reads, so those tokens condition on the prompt but never leak into the
        action expert. Encoding the prompt first and continuing over the cache
        afterwards is the same computation - causal masking already prevents the
        prompt from attending forward - and it makes the exclusion structural rather
        than something a later edit could quietly undo.

        Args:
            inputs_embeds: Appended token embeddings of shape ``(batch, extra, hidden)``.
            cache: The prompt cache from :meth:`encode`.
            prompt_attention_mask: Prompt padding mask of shape ``(batch, seq)``.
            position_ids: Positions for the appended tokens, shape
                ``(axes, batch, extra)``.

        Returns:
            Final hidden states for the appended tokens, shape
            ``(batch, extra, hidden)``.
        """
        batch_size, extra, _ = inputs_embeds.shape
        appended = prompt_attention_mask.new_ones((batch_size, extra))
        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=torch.cat([prompt_attention_mask, appended], dim=-1),
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
        )
        return outputs.last_hidden_state


def as_key_value_list(
    cache: Cache | list[tuple[torch.Tensor, torch.Tensor]],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Normalize a ``transformers`` cache into per-layer key/value tuples.

    ``transformers`` has moved between legacy tuple caches and :class:`Cache`
    objects across releases; the action expert only needs indexable pairs.

    Args:
        cache: Either a :class:`~transformers.cache_utils.Cache` or a sequence of
            ``(key, value)`` pairs.

    Returns:
        One ``(key, value)`` pair per layer.
    """
    layers = getattr(cache, "layers", None)
    if layers is not None:
        return [(layer.keys, layer.values) for layer in layers]
    return list(cast("Iterable[tuple[torch.Tensor, torch.Tensor]]", cache))


def normalize_position_ids(position_ids: torch.Tensor | None, input_ids: torch.Tensor) -> torch.Tensor:
    """Return a ``(3, batch, seq)`` position grid, falling back to plain positions.

    Text-only prompts can make ``compute_3d_position_ids`` return ``None``; the
    action expert still needs a grid to continue from.

    Args:
        position_ids: Grid from the backbone, or ``None``.
        input_ids: Prompt token ids of shape ``(batch, seq)``.

    Returns:
        Position grid of shape ``(3, batch, seq)``.
    """
    if position_ids is None:
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device)
        return positions.view(1, 1, -1).expand(3, batch_size, seq_len)
    if position_ids.ndim == 2:  # noqa: PLR2004 - (batch, seq) from a text-only path
        return position_ids[None].expand(3, -1, -1)
    return position_ids
