# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Vendored from RLWRLD/RLDX-1 (Apache-2.0)

"""Wrapper for transformer layers with image token compression."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Sequence


class LayerWrapper(nn.Module):
    """Wrap a transformer layer and compress image tokens at a target block.

    The wrapper keeps the surrounding sequence intact while replacing a span
    of image tokens with a compact motion token at the configured projection
    layer.
    """

    def __init__(
        self,
        layer: nn.Module,
        layer_idx: int,
        internal_projection: int = 4,
        img_pattern: Sequence[int] = (151652,),
        motion_token: int = 0,
    ) -> None:
        """Initialize the wrapper.

        Args:
            layer: The wrapped transformer layer.
            layer_idx: Index of the wrapped layer in the backbone.
            internal_projection: Layer index where token compression happens.
            img_pattern: Token pattern that marks image token spans.
            motion_token: Number of motion tokens to insert during compression.

        Raises:
            ValueError: If ``motion_token`` is not 1.
        """
        super().__init__()
        self.layer = layer
        self.layer_idx = layer_idx
        self.internal_projection = internal_projection
        self.motion_token = motion_token
        self.img_pattern = img_pattern
        if motion_token != 1:
            msg = f"motion_token must be 1, got {motion_token}"
            raise ValueError(msg)

    def get_removing_indices(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        num_views: Sequence[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Find the image token span that should be compressed.

        Args:
            hidden_states: Hidden states tensor used to infer the batch device.
            input_ids: Token IDs with image token markers.
            num_views: Optional number of views to account for when selecting
                the end index.

        Returns:
            A tuple ``(begin_idx, end_idx)`` with shape ``[B, 1]`` tensors.
        """
        pat_len = len(self.img_pattern)

        windows = input_ids.unfold(dimension=1, size=pat_len, step=1)
        pattern_tensor = torch.tensor(self.img_pattern, device=hidden_states.device).view(1, 1, -1)
        matches = (windows == pattern_tensor).all(dim=-1)

        match_lists = [torch.nonzero(matches[b], as_tuple=False).squeeze(-1) for b in range(hidden_states.shape[0])]
        begin_idx = torch.tensor(
            [m[0] for m in match_lists],
            device=hidden_states.device,
        ).unsqueeze(1)

        if num_views is not None:
            end_idx = torch.tensor(
                [m[-1 * num_view] for m, num_view in zip(match_lists, num_views, strict=False)],
                device=hidden_states.device,
            ).unsqueeze(1)
        else:
            end_idx = torch.tensor(
                [m[-1] for m in match_lists],
                device=hidden_states.device,
            ).unsqueeze(1)

        return begin_idx, end_idx

    @staticmethod
    def left_pad_emb_list(emb_list: Sequence[torch.Tensor]) -> torch.Tensor:
        """Left-pad a list of embeddings to the same length.

        Args:
            emb_list: List of tensors with sequence dimension first.

        Returns:
            A batch tensor with sequences left-padded to the same length.
        """
        rev = [e.flip(0) for e in emb_list]
        padded_rev = torch.nn.utils.rnn.pad_sequence(rev, batch_first=True, padding_value=0)
        return padded_rev.flip(1)

    def forward(  # noqa: PLR0912, PLR0914, PLR0915
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        *args: object,
        **kwargs: Any,  # noqa: ANN401
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Run the wrapped layer with optional image-token compression.

        Args:
            hidden_states: Input hidden states of shape ``[B, T, D]``.
            input_ids: Token IDs used to locate image token spans. Can be passed
                as positional argument or extracted from kwargs/module attributes
                for transformers compatibility.
            *args: Additional positional arguments forwarded to the wrapped layer.
            **kwargs: Additional keyword arguments forwarded to the wrapped layer.

        Returns:
            A tuple ``(output, kwargs)`` where ``output`` is the wrapped layer
            result and ``kwargs`` contains any updated attention or position data.

        Raises:
            ValueError: If image-token compression is triggered but ``input_ids``
                is not provided.
        """
        # The patched Qwen3VLTextModel.forward passes input_ids positionally.
        # Fall back to kwargs for any caller that routes it as a keyword.
        if input_ids is None and "input_ids" in kwargs:
            input_ids = kwargs.pop("input_ids")  # type: ignore[assignment]
        if "image_wise_encoding" in kwargs and isinstance(
            kwargs["image_wise_encoding"],
            torch.Tensor,
        ):
            if kwargs["image_wise_encoding"].shape[0] > 1:
                kwargs["image_wise_encoding"] = bool(kwargs["image_wise_encoding"][0])
            else:
                kwargs["image_wise_encoding"] = kwargs["image_wise_encoding"].bool().item()

        num_views = kwargs["num_views"] if kwargs.get("image_wise_encoding") else None

        bsz, seq_len, _dim = hidden_states.shape

        is_incremental = "cache_position" in kwargs and kwargs["cache_position"] is not None and seq_len == 1
        if self.layer_idx == self.internal_projection and not is_incremental:
            device = hidden_states.device

            token_indices = torch.arange(seq_len, device=device).view(1, -1).expand(bsz, -1)
            if input_ids is None:
                msg = "input_ids required for image-token compression"
                raise ValueError(msg)
            begin_idx, end_idx = self.get_removing_indices(
                hidden_states,
                input_ids,
                num_views=num_views,
            )

            compress_mask = (end_idx > begin_idx).reshape(-1)

            keep_mask_front = token_indices < begin_idx
            keep_mask_back = token_indices >= end_idx

            # Drop motion module tokens (inserted before image tokens) at internal_projection
            motion_drop_info = kwargs.get("motion_drop_info")
            motion_drop_mask = torch.zeros_like(keep_mask_front)  # (bsz, seq_len)
            if motion_drop_info is not None and motion_drop_info["count"] > 0:
                ms = motion_drop_info["start"]
                mc = motion_drop_info["count"]
                motion_drop_mask = (token_indices >= ms) & (token_indices < ms + mc)
                keep_mask_front = keep_mask_front & ~motion_drop_mask  # noqa: PLR6104
                kwargs["motion_drop_info"] = None  # consumed

            # Old-frame image tokens to compress (exclude motion module)
            drop_mask = ~(keep_mask_front | keep_mask_back) & ~motion_drop_mask

            motion_token = (
                (hidden_states * drop_mask.unsqueeze(-1)).sum(dim=1) / drop_mask.sum(dim=1, keepdim=True).clamp(min=1)
            ).reshape(bsz, self.motion_token, -1)

            hs_parts: list[torch.Tensor] = [
                torch.cat(
                    [
                        hidden_states[b][keep_mask_front[b]],
                        motion_token[b]
                        if compress_mask[b]
                        else torch.tensor(
                            [],
                            device=hidden_states.device,
                            dtype=hidden_states.dtype,
                        ),
                        hidden_states[b][keep_mask_back[b]],
                    ],
                    dim=0,
                )
                for b in range(bsz)
            ]

            hidden_states = self.left_pad_emb_list(hs_parts)

            if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
                att_list = [
                    torch.cat(
                        [
                            kwargs["attention_mask"][b][keep_mask_front[b]],
                            torch.ones(
                                1,
                                device=kwargs["attention_mask"].device,
                                dtype=kwargs["attention_mask"].dtype,
                            )
                            if compress_mask[b]
                            else torch.tensor(
                                [],
                                device=kwargs["attention_mask"].device,
                                dtype=kwargs["attention_mask"].dtype,
                            ),
                            kwargs["attention_mask"][b][keep_mask_back[b]],
                        ],
                    )
                    for b in range(bsz)
                ]
                kwargs["attention_mask"] = self.left_pad_emb_list(att_list)
            if "attention_mask_2" in kwargs and kwargs["attention_mask_2"] is not None:
                att_list_2 = [
                    torch.cat(
                        [
                            kwargs["attention_mask_2"][b][keep_mask_front[b]],
                            torch.ones(
                                1,
                                device=kwargs["attention_mask_2"].device,
                                dtype=kwargs["attention_mask_2"].dtype,
                            )
                            if compress_mask[b]
                            else torch.tensor(
                                [],
                                device=kwargs["attention_mask_2"].device,
                                dtype=kwargs["attention_mask_2"].dtype,
                            ),
                            kwargs["attention_mask_2"][b][keep_mask_back[b]],
                        ],
                    )
                    for b in range(bsz)
                ]
                kwargs["attention_mask_2"] = self.left_pad_emb_list(att_list_2)

            if "position_ids" in kwargs and kwargs["position_ids"] is not None:
                position_ids = kwargs["position_ids"]
                if position_ids.dim() == 2:  # noqa: PLR2004
                    pos_list = [
                        torch.cat(
                            [
                                position_ids[b][keep_mask_front[b]],
                                position_ids[b][begin_idx[b] : begin_idx[b] + 1]
                                if compress_mask[b]
                                else position_ids[b][:0],
                                position_ids[b][keep_mask_back[b]],
                            ],
                        )
                        for b in range(bsz)
                    ]
                    kwargs["position_ids"] = self.left_pad_emb_list(pos_list)
                elif position_ids.dim() == 3:  # noqa: PLR2004
                    # Keep the leading rotary dimension (e.g. 3) and compress on seq axis.
                    pos_list = [
                        torch.cat(
                            [
                                position_ids[:, b, keep_mask_front[b]],
                                position_ids[:, b, begin_idx[b] : begin_idx[b] + 1]
                                if compress_mask[b]
                                else position_ids[:, b, :0],
                                position_ids[:, b, keep_mask_back[b]],
                            ],
                            dim=-1,
                        )
                        for b in range(bsz)
                    ]
                    pos_list_for_pad = [p.transpose(0, 1) for p in pos_list]  # [seq, rope_dim]
                    padded = self.left_pad_emb_list(pos_list_for_pad)  # [bs, max_seq, rope_dim]
                    kwargs["position_ids"] = padded.permute(2, 0, 1)  # [rope_dim, bs, max_seq]
                else:
                    msg = f"Unsupported position_ids shape: {position_ids.shape}"
                    raise ValueError(msg)

            if "position_embeddings" in kwargs and kwargs["position_embeddings"] is not None:
                emb_x_list = [
                    torch.cat(
                        [
                            kwargs["position_embeddings"][0][b][keep_mask_front[b]],
                            kwargs["position_embeddings"][0][b][begin_idx[b] : begin_idx[b] + 1]
                            if compress_mask[b]
                            else torch.tensor(
                                [],
                                device=kwargs["position_embeddings"][0].device,
                                dtype=kwargs["position_embeddings"][0].dtype,
                            ),
                            kwargs["position_embeddings"][0][b][keep_mask_back[b]],
                        ],
                        dim=0,
                    )
                    for b in range(bsz)
                ]

                emb_y_list = [
                    torch.cat(
                        [
                            kwargs["position_embeddings"][1][b][keep_mask_front[b]],
                            kwargs["position_embeddings"][1][b][begin_idx[b] : begin_idx[b] + 1]
                            if compress_mask[b]
                            else torch.tensor(
                                [],
                                device=kwargs["position_embeddings"][0].device,
                                dtype=kwargs["position_embeddings"][0].dtype,
                            ),
                            kwargs["position_embeddings"][1][b][keep_mask_back[b]],
                        ],
                        dim=0,
                    )
                    for b in range(bsz)
                ]

                emb_x_padded = self.left_pad_emb_list(emb_x_list)
                emb_y_padded = self.left_pad_emb_list(emb_y_list)
                kwargs["position_embeddings"] = (emb_x_padded, emb_y_padded)

            if "cache_position" in kwargs and kwargs["cache_position"] is not None:
                kwargs["cache_position"] = kwargs["cache_position"][: hidden_states.shape[1]]

        return self.layer(hidden_states, *args, **kwargs), kwargs  # type: ignore[arg-type]
