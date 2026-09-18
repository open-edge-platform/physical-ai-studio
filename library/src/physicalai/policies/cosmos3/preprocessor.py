# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Preprocessor for Cosmos 3 policy with multi-camera view composition."""

from __future__ import annotations

import logging
from dataclasses import fields
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F  # ruff: ignore[lowercase-imported-as-non-lowercase]
from torch import nn

from physicalai.data.observation import IMAGES, Observation

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)

T_SHAPE_EMBODIMENTS = {
    "droid_lerobot",
}

HORIZONTAL_EMBODIMENTS: set[str] = set()

DEFAULT_EMBODIMENT_VIEWPOINTS: dict[str, str | None] = {
    "droid_lerobot": "concat_view",
    "pusht": "top_down_2d_view",
    "aloha": None,
}


def _resize_image_tensor(tensor: torch.Tensor, size_hw: tuple[int, int]) -> torch.Tensor:
    """Resize spatial (H, W) dimensions of an image tensor to size_hw (height, width).

    Supports:
        - 3D: (C, H, W)
        - 4D: (B, C, H, W) or (T, C, H, W)
        - 5D: (B, T, C, H, W)
        - Any arbitrary leading dimensions: (..., C, H, W)

    Args:
        tensor: Image tensor of shape (..., C, H, W).
        size_hw: Target spatial dimensions as (height, width).

    Returns:
        Resized tensor preserving leading dimensions and dtype.
    """
    if tensor.shape[-2:] == size_hw:
        return tensor

    h_target, w_target = size_hw
    *leading, c, h, w = tensor.shape
    flat_leading = 1
    for dim in leading:
        flat_leading *= dim

    reshaped = tensor.reshape(flat_leading, c, h, w)
    orig_dtype = tensor.dtype

    if not reshaped.is_floating_point():
        resized = F.interpolate(
            reshaped.float(),
            size=(h_target, w_target),
            mode="bilinear",
            align_corners=False,
        ).to(orig_dtype)
    else:
        resized = F.interpolate(
            reshaped,
            size=(h_target, w_target),
            mode="bilinear",
            align_corners=False,
        )

    return resized.reshape(*leading, c, h_target, w_target)


def compose_t_views(
    top: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    """Compose primary top view with two exterior views in a T-shape layout.

    Layout:
        Top: Primary camera (wrist or top) at full size (H x W).
        Bottom: Two exterior cameras (left and right) resized to (H // 2, W // 2)
                and concatenated horizontally.
        Stack: Concatenate top and bottom vertically to produce (H + H // 2, W).

    Args:
        top: Primary top camera tensor of shape (..., C, H, W).
        left: Left exterior camera tensor of shape (..., C, H_l, W_l).
        right: Right exterior camera tensor of shape (..., C, H_r, W_r).

    Returns:
        Composed tensor of shape (..., C, H + H // 2, W).
    """
    h_top, w_top = top.shape[-2], top.shape[-1]
    half_h = h_top // 2
    half_w = w_top // 2
    right_w = w_top - half_w

    left_small = _resize_image_tensor(left, (half_h, half_w))
    right_small = _resize_image_tensor(right, (half_h, right_w))

    bottom = torch.cat([left_small, right_small], dim=-1)
    return torch.cat([top, bottom], dim=-2)


def compose_horizontal_views(
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    """Compose two views horizontally side-by-side.

    Layout:
        Left: Left camera (e.g., agentview / third-person) at (H, W).
        Right: Right camera (e.g., wrist) resized to match left's (H, W).
        Concat: Concatenate left and right along width -> (H, 2W).

    Args:
        left: Left camera tensor of shape (..., C, H, W).
        right: Right camera tensor of shape (..., C, H_r, W_r).

    Returns:
        Composed tensor of shape (..., C, H, 2W).
    """
    h_left, w_left = left.shape[-2], left.shape[-1]
    right_matched = _resize_image_tensor(right, (h_left, w_left))
    return torch.cat([left, right_matched], dim=-1)


def _ensure_channels_first(tensor: torch.Tensor) -> torch.Tensor:
    """Ensure image tensor has channels-first layout (..., C, H, W).

    Args:
        tensor: Raw image tensor.

    Returns:
        Tensor with channels in the third-to-last dimension.
    """
    valid_channels = {1, 3, 4}
    min_spatial_ndim = 3
    if (
        tensor.ndim >= min_spatial_ndim
        and tensor.shape[-1] in valid_channels
        and tensor.shape[-3] not in valid_channels
    ):
        dims = list(range(tensor.ndim))
        dims[-3], dims[-2], dims[-1] = dims[-1], dims[-3], dims[-2]
        return tensor.permute(*dims)
    return tensor


def _extract_from_observation(batch: Observation) -> dict[str, Any]:
    """Extract raw image dict from an Observation.

    Args:
        batch: Observation instance.

    Returns:
        Dictionary of raw image fields.
    """
    if isinstance(batch.images, dict):
        return dict(batch.images)
    if isinstance(batch.images, (torch.Tensor, np.ndarray)):
        return {"primary": batch.images}
    if batch.images is None and batch.extra:
        return extract_camera_dict(batch.extra)
    return {}


def _extract_prefixed_cameras(batch: Mapping[str, Any]) -> dict[str, Any]:
    """Extract camera items matching standard prefix conventions.

    Args:
        batch: Batch mapping.

    Returns:
        Dictionary of extracted camera fields.
    """
    cams: dict[str, Any] = {}
    prefixes = ("images.", "observation.images.", "observation.image.")
    for k, v in batch.items():
        if "is_pad" in k or k.startswith("_"):
            continue
        for prefix in prefixes:
            if k.startswith(prefix):
                subkey = k.split(prefix, 1)[1]
                cams[subkey] = v
                break
    return cams


def _extract_from_mapping(batch: Mapping[str, Any]) -> dict[str, Any]:
    """Extract raw image dict from a mapping.

    Args:
        batch: Batch mapping.

    Returns:
        Dictionary of raw image fields.
    """
    if IMAGES in batch and isinstance(batch[IMAGES], dict):
        return dict(batch[IMAGES])
    if "observation.images" in batch and isinstance(batch["observation.images"], dict):
        return dict(batch["observation.images"])

    cams = _extract_prefixed_cameras(batch)
    if cams:
        return cams

    for k in (IMAGES, "observation.image", "image", "pixels"):
        if k in batch and batch[k] is not None:
            val = batch[k]
            if isinstance(val, dict):
                return dict(val)
            if isinstance(val, (torch.Tensor, np.ndarray)):
                return {"primary": val}

    for k, v in batch.items():
        if ("image" in k.lower() or "pixel" in k.lower()) and "is_pad" not in k and not k.startswith("_"):
            if isinstance(v, dict):
                cams.update(v)
            elif isinstance(v, (torch.Tensor, np.ndarray)):
                cams[k] = v

    return cams


def extract_camera_dict(batch: Observation | Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Extract named camera image tensors from an Observation or batch dict.

    Supports:
        - Observation with images dict or tensor
        - Dict with IMAGES ("images") or "observation.images" as sub-dict
        - Flattened dict keys ("images.*", "observation.images.*", "observation.image.*")
        - Single tensor keys ("image", "pixels", "observation.image")

    Args:
        batch: Observation instance or mapping.

    Returns:
        Dictionary of camera names to image tensors.
    """
    raw_cams = _extract_from_observation(batch) if isinstance(batch, Observation) else _extract_from_mapping(batch)

    tensor_cams: dict[str, torch.Tensor] = {}
    for k, v in raw_cams.items():
        if isinstance(v, np.ndarray):
            t = torch.from_numpy(v)
        elif isinstance(v, torch.Tensor):
            t = v
        elif v is not None:
            t = torch.as_tensor(v)
        else:
            continue
        tensor_cams[k] = _ensure_channels_first(t)

    return tensor_cams


def _find_top_camera(keys: list[str]) -> str | None:
    """Find top/wrist camera key from candidate list.

    Args:
        keys: Candidate camera key names.

    Returns:
        Matched top camera key or None.
    """
    for k in keys:
        kl = k.lower()
        if "wrist" in kl or kl in {"top", "head", "primary", "front"}:
            return k
    for k in keys:
        if "top" in k.lower() or "head" in k.lower():
            return k
    return None


def _find_left_camera(keys: list[str]) -> str | None:
    """Find exterior left camera key from candidate list.

    Args:
        keys: Candidate camera key names.

    Returns:
        Matched left camera key or None.
    """
    for k in keys:
        kl = k.lower()
        if "exterior_1" in kl or "exterior_image_1" in kl or "ext1" in kl:
            return k
        if "left" in kl and "wrist" not in kl and "exterior_2" not in kl and "ext2" not in kl:
            return k
    return None


def _find_right_camera(keys: list[str]) -> str | None:
    """Find exterior right camera key from candidate list.

    Args:
        keys: Candidate camera key names.

    Returns:
        Matched right camera key or None.
    """
    for k in keys:
        kl = k.lower()
        if "exterior_2" in kl or "exterior_image_2" in kl or "ext2" in kl:
            return k
        if "right" in kl and "wrist" not in kl:
            return k
    return None


def _identify_t_views(cameras: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Identify top, left, and right views from a camera dictionary.

    Follows DROID, AgibotWorld, and RoboMIND Franka camera naming conventions:
        - Top/Wrist: key containing 'wrist' (DROID) or in ('top', 'head', 'primary')
        - Left: key containing 'exterior_1', 'ext1', or ('left' and not 'wrist')
        - Right: key containing 'exterior_2', 'ext2', or ('right' and not 'wrist')

    Args:
        cameras: Dictionary of camera names to image tensors.

    Returns:
        Tuple of (top, left, right) tensors.
    """
    keys = list(cameras.keys())
    top_key = _find_top_camera(keys)

    remaining_for_left = [k for k in keys if k != top_key]
    left_key = _find_left_camera(remaining_for_left)

    remaining_for_right = [k for k in remaining_for_left if k != left_key]
    right_key = _find_right_camera(remaining_for_right)

    used = {top_key, left_key, right_key} - {None}
    unused = [k for k in keys if k not in used]

    resolved_top = top_key or (unused.pop(0) if unused else keys[0])
    resolved_left = left_key or (unused.pop(0) if unused else (keys[1] if len(keys) > 1 else resolved_top))
    min_keys_for_third = 2
    resolved_right = right_key or (
        unused.pop(0) if unused else (keys[min_keys_for_third] if len(keys) > min_keys_for_third else resolved_left)
    )

    return cameras[resolved_top], cameras[resolved_left], cameras[resolved_right]


def _identify_horizontal_views(cameras: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Identify left and right views for horizontal composition (e.g., Libero).

    Follows Libero conventions:
        - Left: agentview / third-person view
        - Right: wrist-mounted view

    Args:
        cameras: Dictionary of camera names to image tensors.

    Returns:
        Tuple of (left, right) tensors.
    """
    keys = list(cameras.keys())
    left_key: str | None = None
    right_key: str | None = None

    for k in keys:
        kl = k.lower()
        if "agent" in kl or "third" in kl or "front" in kl or kl == "image":
            left_key = k
            break
        if "wrist" not in kl and "hand" not in kl and "eye" not in kl:
            left_key = k

    for k in keys:
        kl = k.lower()
        if "wrist" in kl or "hand" in kl or "eye" in kl:
            right_key = k
            break

    if left_key is None or right_key is None or left_key == right_key:
        sorted_keys = sorted(keys)
        left_key = sorted_keys[0]
        right_key = sorted_keys[1] if len(sorted_keys) > 1 else sorted_keys[0]

    return cameras[left_key], cameras[right_key]


def _select_primary_image(cameras: dict[str, torch.Tensor]) -> torch.Tensor:
    """Select the primary image tensor from a camera dictionary.

    Args:
        cameras: Dictionary of camera names to image tensors.

    Returns:
        Primary image tensor.
    """
    for preferred in ("primary", "top", "image", "wrist", "ego", "agentview"):
        for k, val in cameras.items():
            if preferred in k.lower():
                return val
    return next(iter(cameras.values()))


class Cosmos3Preprocessor(nn.Module):
    """Preprocessor for Cosmos 3 inputs.

    Handles multi-camera view composition (T-shape mosaic, horizontal side-by-side)
    and viewpoint assignment based on the embodiment.

    Responsibilities:
        - Inspect input batch / Observation.images dictionary and embodiment.
        - If the embodiment uses a T-shape layout (e.g. droid_lerobot):
          apply T-shape composition (compose_t_views) and set view_point="concat_view".
        - If the embodiment uses a horizontal layout with multiple cameras:
          apply horizontal concatenation (compose_horizontal_views) and set view_point="concat_view".
        - Otherwise:
          pass primary image through and assign corresponding viewpoint label.
        - Return standardized observation dictionary with processed image tensor
          and view_point metadata.

    Args:
        embodiment: Embodiment identifier. Defaults to "pusht".
        view_point: Optional explicit viewpoint override. Defaults to None.
    """

    def __init__(
        self,
        embodiment: str = "pusht",
        view_point: str | None = None,
    ) -> None:
        """Initialize Cosmos3Preprocessor."""
        super().__init__()
        self.embodiment = embodiment
        self.view_point = view_point

    def forward(
        self,
        batch: Observation | dict[str, Any],
    ) -> dict[str, Any]:
        """Preprocess batch into standardized format with composed image and viewpoint.

        Args:
            batch: Input Observation or dictionary.

        Returns:
            Standardized dictionary with IMAGES and view_point metadata.

        Raises:
            KeyError: If no image tensor can be located in the batch.
        """
        # Idempotency check: already preprocessed
        if (
            isinstance(batch, dict)
            and "view_point" in batch
            and IMAGES in batch
            and isinstance(batch[IMAGES], torch.Tensor)
            and not any(k.startswith(("images.", "observation.images.")) for k in batch)
        ):
            return dict(batch)

        cameras = extract_camera_dict(batch)
        if not cameras:
            msg = "No image tensor found in batch."
            raise KeyError(msg)

        composed_img: torch.Tensor
        inferred_viewpoint: str | None

        min_t_cameras = 3
        min_horizontal_cameras = 2

        if self.embodiment in T_SHAPE_EMBODIMENTS and len(cameras) >= min_t_cameras:
            top, left, right = _identify_t_views(cameras)
            composed_img = compose_t_views(top, left, right)
            inferred_viewpoint = "concat_view"
        elif self.embodiment in HORIZONTAL_EMBODIMENTS and len(cameras) >= min_horizontal_cameras:
            left, right = _identify_horizontal_views(cameras)
            composed_img = compose_horizontal_views(left, right)
            inferred_viewpoint = "concat_view"
        else:
            composed_img = _select_primary_image(cameras)
            if self.embodiment in HORIZONTAL_EMBODIMENTS and len(cameras) == 1:
                key = next(iter(cameras.keys())).lower()
                is_wrist = "wrist" in key or "hand" in key or "eye" in key
                inferred_viewpoint = "wrist_view" if is_wrist else "third_person_view"
            else:
                inferred_viewpoint = DEFAULT_EMBODIMENT_VIEWPOINTS.get(self.embodiment)
                if inferred_viewpoint is None and len(cameras) == 1:
                    key = next(iter(cameras.keys())).lower()
                    if "wrist" in key:
                        inferred_viewpoint = "wrist_view"
                    elif "agent" in key or "third" in key:
                        inferred_viewpoint = "third_person_view"

        final_viewpoint = self.view_point if self.view_point is not None else inferred_viewpoint

        if isinstance(batch, Observation):
            result: dict[str, Any] = {
                field.name: getattr(batch, field.name)
                for field in fields(batch)
                if getattr(batch, field.name) is not None
            }
        else:
            result = dict(batch)

        result[IMAGES] = composed_img
        result["image"] = composed_img
        result["view_point"] = final_viewpoint
        result["viewpoint"] = final_viewpoint

        return result
