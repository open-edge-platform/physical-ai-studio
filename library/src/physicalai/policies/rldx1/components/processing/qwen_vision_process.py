# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Vendored from RLWRLD/RLDX-1 (Apache-2.0)

"""Vision preprocessing utilities for Qwen3-VL image inputs."""

import math
from typing import Any

from PIL import Image

MAX_RATIO = 200
SPATIAL_MERGE_SIZE = 2
IMAGE_MIN_TOKEN_NUM = 4
IMAGE_MAX_TOKEN_NUM = 16384


def round_by_factor(number: int, factor: int) -> int:
    """Return the closest integer to ``number`` that is divisible by ``factor``."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Return the smallest integer >= ``number`` that is divisible by ``factor``."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Return the largest integer <= ``number`` that is divisible by ``factor``."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
) -> tuple[int, int]:
    """Rescale image dimensions so the following conditions are met.

    1. Both dimensions (height and width) are divisible by ``factor``.
    2. The total number of pixels is within the range ``[min_pixels, max_pixels]``.
    3. The aspect ratio of the image is maintained as closely as possible.

    Args:
        height: Original image height in pixels.
        width: Original image width in pixels.
        factor: Patch factor; both output dimensions must be divisible by this.
        min_pixels: Minimum total pixel count. Defaults to ``IMAGE_MIN_TOKEN_NUM * factor**2``.
        max_pixels: Maximum total pixel count. Defaults to ``IMAGE_MAX_TOKEN_NUM * factor**2``.

    Returns:
        A ``(resized_height, resized_width)`` tuple.

    Raises:
        ValueError: If ``max_pixels < min_pixels`` or the absolute aspect ratio
            exceeds ``MAX_RATIO``.
    """
    max_pixels = max_pixels if max_pixels is not None else (IMAGE_MAX_TOKEN_NUM * factor**2)
    min_pixels = min_pixels if min_pixels is not None else (IMAGE_MIN_TOKEN_NUM * factor**2)
    if max_pixels < min_pixels:
        msg = "The max_pixels of image must be greater than or equal to min_pixels."
        raise ValueError(msg)
    if max(height, width) / min(height, width) > MAX_RATIO:
        ratio = max(height, width) / min(height, width)
        msg = f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {ratio}"
        raise ValueError(msg)
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(int(height / beta), factor)
        w_bar = floor_by_factor(int(width / beta), factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(int(height * beta), factor)
        w_bar = ceil_by_factor(int(width * beta), factor)
    return h_bar, w_bar


def fetch_image(ele: dict[str, str | Image.Image], image_patch_size: int = 14) -> Image.Image:
    """Fetch and resize a PIL image from a conversation element dict.

    Args:
        ele: Dict with an ``"image"`` or ``"image_url"`` key holding a
            :class:`PIL.Image.Image`. Optionally carries ``"resized_height"``,
            ``"resized_width"``, ``"min_pixels"``, and ``"max_pixels"`` hints.
        image_patch_size: Patch size used to compute the resize factor.

    Returns:
        Resized :class:`PIL.Image.Image`.

    Raises:
        TypeError: If the value under "image"/"image_url" is not a PIL Image
            (file paths and URLs are not supported).
    """
    image = ele["image"] if "image" in ele else ele["image_url"]
    if not isinstance(image, Image.Image):
        msg = (
            f"Expected a PIL.Image for 'image'/'image_url', got {type(image)!r}. "
            "File paths and URLs are not supported in this pipeline."
        )
        raise TypeError(msg)
    patch_factor = int(image_patch_size * SPATIAL_MERGE_SIZE)

    # resize
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            int(ele["resized_height"]),  # type: ignore[arg-type]
            int(ele["resized_width"]),  # type: ignore[arg-type]
            factor=patch_factor,
        )
    else:
        width, height = image.size
        min_pixels = ele.get("min_pixels", IMAGE_MIN_TOKEN_NUM * patch_factor**2)
        max_pixels = ele.get("max_pixels", IMAGE_MAX_TOKEN_NUM * patch_factor**2)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=patch_factor,
            min_pixels=int(min_pixels),  # type: ignore[arg-type]
            max_pixels=int(max_pixels),  # type: ignore[arg-type]
        )
    return image.resize((resized_width, resized_height))


def extract_vision_info(
    conversations: list[dict[str, Any]] | list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Extract all image/video element dicts from a conversation structure.

    Args:
        conversations: Either a flat list of message dicts or a list of
            conversation lists (each a list of message dicts).

    Returns:
        List of element dicts that contain image or video content.
    """
    vision_infos: list[dict[str, Any]] = []
    if isinstance(conversations[0], dict):
        convs: list[list[dict[str, Any]]] = [conversations]  # type: ignore[list-item]
    else:
        convs = conversations  # type: ignore[assignment]
    for conversation in convs:
        for message in conversation:
            if isinstance(message["content"], list):
                vision_infos.extend(
                    ele
                    for ele in message["content"]
                    if "image" in ele or "image_url" in ele or ele.get("type", "text") in {"image", "image_url"}
                )
    return vision_infos


def process_vision_info(
    conversations: list[dict[str, Any]] | list[list[dict[str, Any]]],
    image_patch_size: int = 14,
) -> list[Image.Image] | None:
    """Process vision elements from conversations into PIL images.

    Args:
        conversations: Either a flat list of message dicts or a list of
            conversation lists (each a list of message dicts).
        image_patch_size: Patch size used to compute the resize factor.

    Returns:
        List of resized :class:`PIL.Image.Image` objects, or ``None`` if no
        images were found.

    Raises:
        ValueError: If a vision element contains neither ``"image"`` nor
            ``"image_url"`` (video elements are not yet supported).
    """
    vision_infos = extract_vision_info(conversations)
    image_inputs: list[Image.Image] = []
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info, image_patch_size=image_patch_size))
        else:
            msg = "image, image_url or video should in content."
            raise ValueError(msg)
    if len(image_inputs) == 0:
        return None
    return image_inputs
