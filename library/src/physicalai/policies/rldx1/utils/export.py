# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Vendored from RLWRLD/RLDX-1 (Apache-2.0)

"""Export-only helpers for the RLDX-1 policy graph tracing path."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Protocol, cast

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from physicalai.data import FeatureType
from physicalai.data.observation import STATE
from physicalai.policies.rldx1.components.norms import RMSNorm
from physicalai.policies.rldx1.constants import (
    ATTENTION_MASK,
    EMBODIMENT_ID,
    IMAGE_GRID_THW,
    INPUT_IDS,
    PIXEL_VALUES,
    POSITION_IDS,
)

if TYPE_CHECKING:
    from collections.abc import Generator

    from physicalai.policies.rldx1.components.action_model.blocks import DoubleStreamBlock, SingleStreamBlock
    from physicalai.policies.rldx1.components.action_model.msat import MSAT
    from physicalai.policies.rldx1.components.action_model.ops import SwiGLUFFN
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

    Args:
        dataset_stats: Per-feature dataset statistics with ``type`` and ``shape``.

    Returns:
        The ``(height, width)`` of the first visual feature.

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


def export_image_resolution_after_preprocessor_from_stats(
    dataset_stats: dict[str, dict[str, Any]] | None,
    *,
    image_max_area: int,
    image_resize_m: int,
    image_min_area: int | None,
) -> tuple[int, int]:
    """Return post-AspectAreaResizeAndCrop ``(height, width)`` from dataset stats.

    Mirrors :class:`AspectAreaResizeAndCrop` geometry: area-budget resize then
    ``m_alignment`` center-crop. The runtime RLDX-1 preprocessor assumes this
    frozen post-transform shape.

    Args:
        dataset_stats: Per-feature dataset statistics with ``type`` and ``shape``.
        image_max_area: Pixel-area budget above which images are downscaled.
        image_resize_m: Alignment multiple for the center crop.
        image_min_area: Pixel area below which images are upscaled, or ``None``.

    Returns:
        The ``(height, width)`` after resize and crop.
    """
    orig_h, orig_w = export_image_resolution_from_stats(dataset_stats)
    current_area = orig_h * orig_w

    if image_min_area is not None and current_area < image_min_area:
        scale = (image_min_area / current_area) ** 0.5
    elif current_area > image_max_area:
        scale = (image_max_area / current_area) ** 0.5
    else:
        scale = 1.0

    resized_h = round(orig_h * scale)
    resized_w = round(orig_w * scale)

    crop_h = (resized_h // image_resize_m) * image_resize_m
    crop_w = (resized_w // image_resize_m) * image_resize_m
    return int(crop_h), int(crop_w)


def build_rldx1_token_composer_params(
    *,
    tokenizer: _TokenizerLike,
    image_resolution: tuple[int, int],
    num_views: int,
    num_frames: int,
    max_token_len: int,
) -> dict[str, Any]:
    """Build manifest params for runtime RLDX-1 token composition.

    Args:
        tokenizer: Tokenizer used to encode the chat-template prefix/suffix.
        image_resolution: Post-preprocessor ``(height, width)`` of each image.
        num_views: Number of camera views.
        num_frames: Number of frames per view.
        max_token_len: Fixed prompt length after padding/truncation.

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

    Args:
        token_composer_params: Params from :func:`build_rldx1_token_composer_params`.

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

    Args:
        sample: Export sample keyed by input name.

    Returns:
        A shallow-copied sample where floating tensors are ``float32``.
    """
    return {
        key: value.float() if isinstance(value, torch.Tensor) and value.is_floating_point() else value
        for key, value in sample.items()
    }


def trim_export_sample(input_sample: dict[str, torch.Tensor] | None) -> dict[str, torch.Tensor] | None:
    """Trim export sample to the graph-safe model's declared inputs.

    Args:
        input_sample: Full export sample, or ``None``.

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

    Args:
        model: Module to cast, or ``None`` for a no-op.

    Yields:
        None: Control while the model weights are fp32.
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


# Power of two, so dividing the weights and multiplying them back is exact.
F16_SAFE_RESIDUAL_SCALE = 16.0


def _residual_stream_layout(
    msat: MSAT,
) -> tuple[list[nn.Linear], list[nn.LayerNorm | RMSNorm], list[nn.Module]] | None:
    """Return the linear writers, norm readers and pass-through writers of the SA residual stream.

    Args:
        msat: The action head to inspect.

    Returns:
        ``(linears, norms, passthrough)`` or ``None`` when some reader of the
        stream is not scale-invariant, so the stream cannot be rescaled exactly.
    """
    if msat.use_physics:
        return None
    double = cast("list[DoubleStreamBlock]", list(msat.double_blocks))
    single = cast("list[SingleStreamBlock]", list(msat.single_blocks))
    post_norms = [m for b in double for m in (b.sa_norm2_attn, b.sa_norm3_mlp)] + [b.post_norm for b in single]
    norms = [m for b in double for m in (b.sa_norm1, b.sa_norm2_mlp)] + [b.pre_norm for b in single]
    norms.append(msat.norm_out)
    if not all(isinstance(m, nn.Identity) for m in post_norms):
        return None
    scale_invariant = [m for m in norms if isinstance(m, (nn.LayerNorm, RMSNorm))]
    if len(scale_invariant) != len(norms):
        return None
    linears = [m for b in double for m in (b.sa_proj, b.sa_mlp.w3)] + [b.linear2 for b in single]
    passthrough: list[nn.Module] = [msat.time_token_proj]
    if isinstance(msat.vl_proj_to_sa, nn.Linear):
        linears.append(msat.vl_proj_to_sa)
    else:
        passthrough.append(msat.vl_proj_to_sa)
    return linears, scale_invariant, passthrough


@contextmanager
def f16_safe_residual_stream(msat: MSAT, scale: float = F16_SAFE_RESIDUAL_SCALE) -> Generator[None, None, None]:
    """Temporarily shrink the MSAT state-action residual stream by ``scale``.

    Trained heads carry residual activations near 1e5, past the f16 maximum
    (65504), so f16 inference (e.g. OpenVINO GPU) returns NaN actions. Every
    reader of the stream is a LayerNorm/RMSNorm, so dividing every writer by
    ``scale`` and each norm's eps by ``scale**2`` leaves the outputs unchanged.

    Args:
        msat: The action head to rescale in place.
        scale: Divisor for the residual stream; a power of two keeps it exact.

    Yields:
        None: Control while the residual stream is rescaled.
    """
    layout = _residual_stream_layout(msat)
    if layout is None:
        yield
        return
    linears, norms, passthrough = layout
    inv = 1.0 / scale

    def scale_stream_input(
        _module: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if "hidden_states" in kwargs:
            return args, {**kwargs, "hidden_states": kwargs["hidden_states"] * inv}
        return (args[0] * inv, *args[1:]), kwargs

    def scale_output(_module: nn.Module, _args: tuple[Any, ...], output: torch.Tensor) -> torch.Tensor:
        return output * inv

    original_eps = [norm.eps for norm in norms]
    handles = [msat.register_forward_pre_hook(scale_stream_input, with_kwargs=True)]
    handles.extend(module.register_forward_hook(scale_output) for module in passthrough)
    with torch.no_grad():
        for linear in linears:
            linear.weight.mul_(inv)
            if linear.bias is not None:
                linear.bias.mul_(inv)
    for norm, eps in zip(norms, original_eps, strict=True):
        norm.eps = eps * inv * inv
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
        with torch.no_grad():
            for linear in linears:
                linear.weight.mul_(scale)
                if linear.bias is not None:
                    linear.bias.mul_(scale)
        for norm, eps in zip(norms, original_eps, strict=True):
            norm.eps = eps


class _FirstCallOwnWeight(nn.Module):
    """``nn.Linear`` whose first call per forward reads a separate copy of its weights."""

    def __init__(self, linear: nn.Linear) -> None:
        """Wrap ``linear`` and clone its weights for the first call.

        Args:
            linear: Shared linear layer used by every call after the first.
        """
        super().__init__()
        self.linear = linear
        self.first_weight = nn.Parameter(linear.weight.detach().clone(), requires_grad=False)
        self.first_bias = (
            nn.Parameter(linear.bias.detach().clone(), requires_grad=False) if linear.bias is not None else None
        )
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the cloned weights on the first call, the shared ones afterwards.

        Args:
            x: Input features.

        Returns:
            The linear projection of ``x``.
        """
        self.calls += 1
        if self.calls == 1:
            return F.linear(x, self.first_weight, self.first_bias)
        return self.linear(x)


@contextmanager
def untie_first_denoising_step(root: nn.Module, msat: MSAT) -> Generator[None, None, None]:
    """Give the first denoising step its own copy of the VL MLP down-projection weights.

    The unrolled denoising steps share every MSAT weight. On the OpenVINO 2026.4
    GPU plugin at f16, the later steps' ``vl_mlp.w3`` FullyConnected layers
    return garbage when they share the weight constant with the first step.
    The IR writer stores identical weight blobs once, so the ``.bin`` size
    does not change.

    Args:
        root: Module traced for export; resets the per-forward call count.
        msat: The action head whose double blocks are patched.

    Yields:
        None: Control while the first step uses its own weights.
    """
    wrapped: list[tuple[SwiGLUFFN, nn.Linear, _FirstCallOwnWeight]] = []
    for block in cast("list[DoubleStreamBlock]", list(msat.double_blocks)):
        mlp = cast("SwiGLUFFN", block.vl_mlp)
        wrapper = _FirstCallOwnWeight(mlp.w3)
        wrapped.append((mlp, mlp.w3, wrapper))
        mlp.w3 = wrapper  # type: ignore[assignment]

    def reset_calls(_module: nn.Module, _args: tuple[Any, ...]) -> None:
        for _, _, wrapper in wrapped:
            wrapper.calls = 0

    handle = root.register_forward_pre_hook(reset_calls)
    try:
        yield
    finally:
        handle.remove()
        for owner, linear, _ in wrapped:
            owner.w3 = linear


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

    Args:
        model: RLDX-1 model whose backbone computes the M-RoPE position ids.
        input_ids: Unpadded prompt token ids of shape ``(1, L)``.
        image_grid_thw: Per-image ``(t, h, w)`` patch grid.
        embodiment_id: Embodiment id tensor.
        config: Policy config providing ``tokenizer_max_length`` and ``n_cog_tokens``.

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
