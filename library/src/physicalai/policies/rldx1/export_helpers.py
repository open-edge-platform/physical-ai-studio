# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Vendored from RLWRLD/RLDX-1 (Apache-2.0)

"""Export-only helpers for the RLDX-1 policy graph tracing path."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Protocol

import torch

from physicalai.data import FeatureType
from physicalai.data.observation import STATE

from .constants import ATTENTION_MASK, EMBODIMENT_ID, IMAGE_GRID_THW, INPUT_IDS, PIXEL_VALUES, POSITION_IDS

if TYPE_CHECKING:
    from collections.abc import Generator

    from physicalai.policies.rldx1.config import Rldx1Config
    from physicalai.policies.rldx1.model import Rldx1Model


# Placeholder token appended for the cog-token M-RoPE positions (matches the
# eager VTCQwen3VLBackbone._forward_qwen_with_cog_tokens).
_PLACEHOLDER_TOKEN_ID = 248068
# Pad token value is irrelevant numerically: pads are masked as attention keys
# and their own outputs are discarded (only cog tokens are read), so 0 is fine.
_PAD_TOKEN_ID = 0


class _TokenizerLike(Protocol):
    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]: ...

    def convert_tokens_to_ids(self, token: str) -> int: ...


def export_image_resolution_from_stats(dataset_stats: dict[str, dict[str, Any]] | None) -> tuple[int, int]:
    """Return ``(height, width)`` from the first visual dataset-stats entry.

    Raises:
        RuntimeError: If no visual stats entry is available.
    """
    resolution = next(
        (
            tuple(feature["shape"])[-2:]
            for feature in (dataset_stats or {}).values()
            if str(FeatureType.VISUAL) in str(feature.get("type", ""))
        ),
        None,
    )
    if resolution is None:
        msg = "Failed to determine image resolution from dataset stats."
        raise RuntimeError(msg)
    return int(resolution[0]), int(resolution[1])


def build_rldx1_token_composer_params(
    *,
    tokenizer: _TokenizerLike,
    image_resolution: tuple[int, int],
    num_views: int,
    num_frames: int,
    max_token_len: int,
) -> dict[str, Any]:
    """Build manifest params for runtime RLDX-1 token composition.

    Returns:
        A manifest-serializable parameter mapping for ``rldx1_token_composer``.
    """
    image_height, image_width = int(image_resolution[0]), int(image_resolution[1])
    patch_size = 16
    merge_size = 2
    grid_h = image_height // patch_size
    grid_w = image_width // patch_size
    tokens_per_image = (grid_h * grid_w) // (merge_size**2)

    prefix_ids = tokenizer("<|im_start|>user\\n", add_special_tokens=False)["input_ids"]
    suffix_ids = tokenizer("<|im_end|>\\n", add_special_tokens=False)["input_ids"]

    special_tokens = {
        "vision_start": "<|vision_start|>",
        "vision_end": "<|vision_end|>",
        "image_pad": "<|image_pad|>",
    }
    special_ids = {name: int(tokenizer.convert_tokens_to_ids(token)) for name, token in special_tokens.items()}

    return {
        "formalize_language": True,
        "prefix_ids": [int(value) for value in prefix_ids],
        "suffix_ids": [int(value) for value in suffix_ids],
        "special_ids": special_ids,
        "tokens_per_image": int(tokens_per_image),
        "num_images": int(num_views * num_frames),
        "max_token_len": int(max_token_len),
        "padding_side": "left",
    }


def build_compress_reference_ids(token_composer_params: dict[str, Any]) -> torch.Tensor:
    """Build canonical ``input_ids`` whose image-token span matches runtime.

    Returns:
        A ``(1, max_token_len)`` tensor with runtime-aligned image token span.
    """
    special = token_composer_params["special_ids"]
    vision_block = [
        special["vision_start"],
        *([special["image_pad"]] * token_composer_params["tokens_per_image"]),
        special["vision_end"],
    ]
    tail = [*(vision_block * token_composer_params["num_images"]), *token_composer_params["suffix_ids"]]
    length = token_composer_params["max_token_len"]
    # Runtime keeps the rightmost max_token_len tokens (full[-max_token_len:]);
    # the image span stays right-aligned under both padding and truncation.
    tail = tail[-length:]
    ids = [0] * (length - len(tail)) + tail
    return torch.tensor([ids], dtype=torch.long)


def cast_sample_fp32(sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Cast floating tensors in an export sample to fp32 (ints untouched).

    Returns:
        A shallow-copied sample where floating tensors are ``float32``.
    """
    return {
        key: value.float() if isinstance(value, torch.Tensor) and value.is_floating_point() else value
        for key, value in sample.items()
    }


def trim_export_sample(input_sample: dict[str, torch.Tensor] | None) -> dict[str, torch.Tensor] | None:
    """Trim export sample to the graph-safe model's declared inputs.

    Returns:
        A filtered sample containing only declared tracing inputs, or ``None``.
    """
    if input_sample is None:
        return None
    input_names = [PIXEL_VALUES, INPUT_IDS, POSITION_IDS, ATTENTION_MASK, STATE]
    return {key: input_sample[key] for key in input_names if key in input_sample}


@contextmanager
def fp32_weights_for_export(model: torch.nn.Module | None) -> Generator[None, None, None]:
    """Cast model floats to fp32 for tracing, then restore per-tensor dtypes.

    OpenVINO/ONNX do not lower a bf16-traced graph faithfully (the vision
    patch-embed and downstream ops produce garbage), so tracing must run in
    fp32. ``bf16 -> fp32 -> bf16`` round-trips losslessly, and per-tensor
    restore preserves mixed-dtype modules (e.g. an fp32 ``cog_emb``).

    """
    if model is None:
        yield
        return
    tensors = [*model.named_parameters(), *model.named_buffers()]
    saved = {name: t.dtype for name, t in tensors if t.is_floating_point() and t.dtype != torch.float32}
    if not saved:
        yield
        return
    model.float()
    try:
        yield
    finally:
        by_name = dict([*model.named_parameters(), *model.named_buffers()])
        for name, dtype in saved.items():
            tensor = by_name.get(name)
            if tensor is not None:
                tensor.data = tensor.data.to(dtype)


def build_padded_sample(  # noqa: PLR0914
    model: Rldx1Model,
    *,
    input_ids: torch.Tensor,
    image_grid_thw: torch.Tensor,
    embodiment_id: torch.Tensor,
    config: Rldx1Config,
) -> dict[str, torch.Tensor]:
    """Left-pad ``input_ids`` to fixed length and add static prompt tensors.

    Produces fixed-shape ``input_ids`` / ``position_ids`` / ``attention_mask``
    over ``[padded ids | cog placeholders]`` for graph-safe export tracing.

    Returns:
        A sample dict containing padded prompt tensors plus
        ``image_grid_thw``/``embodiment_id``.

    Raises:
        ValueError: If the prompt is longer than ``tokenizer_max_length``.
    """
    grid_thw = image_grid_thw
    device = input_ids.device
    length = config.tokenizer_max_length
    actual = input_ids.shape[1]
    if actual > length:
        msg = f"Prompt length {actual} exceeds tokenizer_max_length {length}; increase it or shorten the task."
        raise ValueError(msg)

    pad_len = length - actual
    pads = torch.full((1, pad_len), _PAD_TOKEN_ID, dtype=input_ids.dtype, device=device)
    padded_input_ids = torch.cat([pads, input_ids], dim=1)
    attention_mask = torch.cat(
        [
            torch.zeros(1, pad_len, dtype=torch.long, device=device),
            torch.ones(1, actual, dtype=torch.long, device=device),
        ],
        dim=1,
    )

    n_cog = config.n_cog_tokens
    cog_ids = torch.full((1, n_cog), _PLACEHOLDER_TOKEN_ID, dtype=input_ids.dtype, device=device)
    extended_input_ids = torch.cat([padded_input_ids, cog_ids], dim=1)
    extended_mask = torch.cat([attention_mask, torch.ones(1, n_cog, dtype=torch.long, device=device)], dim=1)

    inner = model.backbone.qwen_model.model
    with torch.no_grad():
        position_ids, _ = inner.get_rope_index(extended_input_ids, grid_thw, attention_mask=extended_mask)

    return {
        INPUT_IDS: padded_input_ids,
        IMAGE_GRID_THW: image_grid_thw,
        EMBODIMENT_ID: embodiment_id,
        POSITION_IDS: position_ids,
        ATTENTION_MASK: extended_mask,
    }
