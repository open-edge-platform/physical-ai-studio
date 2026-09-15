# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Vendored from RLWRLD/RLDX-1 (Apache-2.0)

"""Vision preprocessing utilities for Qwen3-VL image inputs."""

from typing import Any

from PIL import Image

SPATIAL_MERGE_SIZE = 2
IMAGE_MIN_TOKEN_NUM = 4
IMAGE_MAX_TOKEN_NUM = 16384


def fetch_image(ele: dict[str, str | Image.Image], image_patch_size: int = 14) -> Image.Image:
    """Fetch a pre-resized PIL image from a conversation element dict.

    Images are expected to already be patch-aligned and within the vision
    tiler's pixel budget: ``AspectAreaResizeAndCrop`` upstream
    guarantees this via its ``m_alignment``, which must equal ``patch_factor``
    below. This function validates that invariant instead of silently
    re-resizing to it (the resize was a no-op given the current config).

    Args:
        ele: Dict with an ``"image"`` or ``"image_url"`` key holding a
            :class:`PIL.Image.Image`. Optionally carries ``"min_pixels"`` and
            ``"max_pixels"`` hints.
        image_patch_size: Patch size used to compute the alignment factor.

    Returns:
        The unmodified :class:`PIL.Image.Image`.

    Raises:
        TypeError: If the value under "image"/"image_url" is not a PIL Image
            (file paths and URLs are not supported).
        ValueError: If the image isn't patch-aligned or its area falls outside
            the pixel budget -- both indicate ``image_resize_m``/
            ``image_max_area`` drifted out of sync with ``patch_factor``.
    """
    image = ele["image"] if "image" in ele else ele["image_url"]
    if not isinstance(image, Image.Image):
        msg = (
            f"Expected a PIL.Image for 'image'/'image_url', got {type(image)!r}. "
            "File paths and URLs are not supported in this pipeline."
        )
        raise TypeError(msg)
    patch_factor = int(image_patch_size * SPATIAL_MERGE_SIZE)

    width, height = image.size
    if height % patch_factor != 0 or width % patch_factor != 0:
        msg = (
            f"Image size {(height, width)} is not aligned to patch_factor={patch_factor} "
            "(image_patch_size * SPATIAL_MERGE_SIZE). Stage 3 AspectAreaResizeAndCrop's "
            "image_resize_m must be a multiple of patch_factor."
        )
        raise ValueError(msg)

    min_pixels_raw = ele.get("min_pixels", IMAGE_MIN_TOKEN_NUM * patch_factor**2)
    max_pixels_raw = ele.get("max_pixels", IMAGE_MAX_TOKEN_NUM * patch_factor**2)
    if isinstance(min_pixels_raw, Image.Image) or isinstance(max_pixels_raw, Image.Image):
        msg = "min_pixels/max_pixels must be numeric metadata values."
        raise TypeError(msg)
    min_pixels = int(min_pixels_raw)
    max_pixels = int(max_pixels_raw)
    area = height * width
    if not min_pixels <= area <= max_pixels:
        msg = (
            f"Image area {area} is outside the vision tiler's pixel budget "
            f"[{min_pixels}, {max_pixels}]. Adjust image_max_area/image_min_area."
        )
        raise ValueError(msg)

    return image


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
