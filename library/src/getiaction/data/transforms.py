# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""ONNX-compatible image transforms.

This module provides drop-in replacements for torchvision transforms that are
ONNX-exportable, addressing issues where torchvision transforms use Python
operations not supported in ONNX tracing.
"""

import torch
import torchvision.transforms
from torch import nn
from torchvision.transforms.v2 import Transform

# Constants for tensor dimensions
_NDIM_IMAGE_NO_BATCH = 3
_NDIM_IMAGE_WITH_BATCH = 4
_NDIM_IMAGE_TEMPORAL = 5  # (batch, time, channels, height, width)


def _compute_center_crop_coordinates(
    image_height: int,
    image_width: int,
    crop_height: int,
    crop_width: int,
) -> tuple[int, int]:
    """Compute top-left coordinates for center cropping.

    Args:
        image_height: Height of input image
        image_width: Width of input image
        crop_height: Desired crop height
        crop_width: Desired crop width

    Returns:
        Tuple of (crop_top, crop_left) coordinates
    """
    crop_top = (image_height - crop_height) // 2
    crop_left = (image_width - crop_width) // 2
    return crop_top, crop_left


def center_crop_image(image: torch.Tensor, output_size: list[int] | tuple[int, int]) -> torch.Tensor:
    """Apply center-cropping to an input image tensor.

    This function uses integer division for computing crop coordinates,
    ensuring ONNX compatibility. Supports 3D, 4D, and 5D tensors.

    Args:
        image: Input image tensor of shape (C, H, W), (B, C, H, W), or (B, T, C, H, W)
        output_size: Desired output size [height, width] or (height, width)

    Returns:
        Center-cropped image tensor with same number of dimensions

    Raises:
        ValueError: If tensor has unexpected number of dimensions

    Example:
        >>> image = torch.randn(3, 256, 256)
        >>> output = center_crop_image(image, [224, 224])
        >>> output.shape
        torch.Size([3, 224, 224])
    """
    crop_height, crop_width = output_size if isinstance(output_size, (list, tuple)) else (output_size, output_size)
    original_ndim = image.ndim

    # Normalize to 4D or 5D for processing
    if original_ndim == _NDIM_IMAGE_NO_BATCH:
        # (C, H, W) -> (1, C, H, W)
        image = image.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    # Handle both 4D (B, C, H, W) and 5D (B, T, C, H, W) tensors
    batch_dims: tuple[slice, ...] | tuple[slice, slice] | tuple[slice, slice, slice]
    if image.ndim == _NDIM_IMAGE_WITH_BATCH:
        # 4D: (B, C, H, W)
        _, _, image_height, image_width = image.shape
        batch_dims = slice(None), slice(None)  # :, :
    elif image.ndim == _NDIM_IMAGE_TEMPORAL:
        # 5D: (B, T, C, H, W)
        _, _, _, image_height, image_width = image.shape
        batch_dims = slice(None), slice(None), slice(None)  # :, :, :
    else:
        msg = f"Expected 3D, 4D, or 5D tensor, got {image.ndim}D"
        raise ValueError(msg)

    # Compute crop coordinates
    crop_top, crop_left = _compute_center_crop_coordinates(
        image_height,
        image_width,
        crop_height,
        crop_width,
    )

    # Perform the crop using tensor slicing
    cropped = image[
        *batch_dims,
        crop_top : crop_top + crop_height,
        crop_left : crop_left + crop_width,
    ]

    if squeeze_output:
        cropped = cropped.squeeze(0)

    return cropped


class OnnxCenterCrop(Transform):
    """ONNX-compatible center crop transform.

    This is a drop-in replacement for torchvision.transforms.CenterCrop that
    uses only tensor operations compatible with ONNX export. Extends
    torchvision.transforms.v2.Transform for better compatibility with
    torchvision v2 API.

    Supports 3D (C, H, W), 4D (B, C, H, W), and 5D (B, T, C, H, W) tensors.

    Args:
        size: Desired output size. If size is an int, a square crop (size, size) is made.
            If size is a sequence of length 2, it should be (height, width).

    Example:
        >>> crop = OnnxCenterCrop(224)
        >>> # 3D tensor
        >>> input_3d = torch.randn(3, 256, 256)
        >>> output_3d = crop(input_3d)
        >>> output_3d.shape
        torch.Size([3, 224, 224])

        >>> # 4D tensor
        >>> input_4d = torch.randn(1, 3, 256, 256)
        >>> output_4d = crop(input_4d)
        >>> output_4d.shape
        torch.Size([1, 3, 224, 224])

        >>> # 5D tensor (temporal)
        >>> input_5d = torch.randn(1, 4, 3, 256, 256)
        >>> output_5d = crop(input_5d)
        >>> output_5d.shape
        torch.Size([1, 4, 3, 224, 224])
    """

    def __init__(self, size: int | tuple[int, int]) -> None:
        """Initialize the center crop transform.

        Args:
            size: Target crop size as int or (height, width) tuple
        """
        super().__init__()
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def _transform(self, inpt: torch.Tensor, params: dict) -> torch.Tensor:  # noqa: ARG002
        """Apply the center crop transform.

        Args:
            inpt: Input tensor to transform
            params: Transform parameters (unused)

        Returns:
            Center-cropped output tensor
        """
        return center_crop_image(inpt, output_size=self.size)

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        """Apply center crop to image tensor.

        This method wraps _transform to provide compatibility with both
        torchvision v2 Transform API and direct forward calls.

        Args:
            *inputs: Input image tensor(s) of shape (C, H, W), (B, C, H, W), or (B, T, C, H, W)

        Returns:
            Center-cropped image tensor with same number of dimensions
        """
        # Handle both single tensor and tuple/list of tensors
        if len(inputs) == 1 and isinstance(inputs[0], torch.Tensor):
            return self._transform(inputs[0], {})
        # If called with multiple args or other format, default to first tensor
        return self._transform(inputs[0], {})


def replace_center_crop_with_onnx_compatible(model: nn.Module) -> None:
    """Replace all CenterCrop transforms in a model with ONNX-compatible versions.

    This function recursively traverses the model and replaces any instances of
    torchvision.transforms.CenterCrop with OnnxCenterCrop, which uses only
    tensor operations that are ONNX-exportable.

    Args:
        model: The model to modify in-place

    Example:
        >>> from getiaction.policies.lerobot import Diffusion
        >>> policy = Diffusion(...)
        >>> replace_center_crop_with_onnx_compatible(policy)
        >>> # Now the policy can be exported to ONNX
    """
    for name, module in model.named_children():
        if isinstance(module, torchvision.transforms.CenterCrop):
            # Replace with ONNX-compatible version
            # Extract size from the CenterCrop module - type annotation removed due to torchvision internals
            crop_size = module.size if hasattr(module, "size") else (module.crop_height, module.crop_width)
            setattr(model, name, OnnxCenterCrop(crop_size))  # type: ignore[arg-type]
        else:
            # Recursively process child modules
            replace_center_crop_with_onnx_compatible(module)
