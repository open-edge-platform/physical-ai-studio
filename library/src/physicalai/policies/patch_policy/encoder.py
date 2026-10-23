# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Small timm-backed visual feature encoder."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from types import ModuleType

    from .config import EncoderOutput, FeatureEncoderName

_TIMM_MODELS: dict[FeatureEncoderName, str] = {
    "dinov2": "vit_small_patch14_dinov2.lvd142m",
    "webssl": "hf-hub:facebook/webssl-dino300m-full2b-224",
    "resnet18": "resnet18",
}


def _import_timm() -> ModuleType:
    """Import the optional timm dependency.

    Returns:
        The imported timm module.

    Raises:
        ImportError: If the Patch Policy optional dependencies are not installed.
    """
    try:
        import timm  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("Patch Policy requires timm. Install physicalai-train[patch_policy].") from exc
    return timm


class TimmFeatureEncoder(nn.Module):
    """Encode arbitrary leading image dimensions into patch tokens.

    Args:
        name: Supported timm encoder name.
        output: Encoder representation to return for each image.
        pretrained: Whether to load pretrained backbone weights.
        frozen: Whether to freeze the backbone in evaluation mode.
        image_size: Input image height and width.
    """

    mean: Tensor
    std: Tensor

    def __init__(
        self,
        name: FeatureEncoderName,
        output: EncoderOutput,
        *,
        pretrained: bool = True,
        frozen: bool = True,
        image_size: tuple[int, int] = (224, 224),
    ) -> None:
        super().__init__()
        self.output = output
        self.frozen = frozen
        model_kwargs = {} if name == "resnet18" else {"img_size": image_size}
        self.backbone = _import_timm().create_model(
            _TIMM_MODELS[name],
            pretrained=pretrained,
            num_classes=0,
            **model_kwargs,
        )
        self.output_dim = int(self.backbone.num_features)
        patch_embed = getattr(self.backbone, "patch_embed", None)
        self.num_tokens = int(getattr(patch_embed, "num_patches", 1)) if output == "patches" else 1

        config = getattr(self.backbone, "pretrained_cfg", {}) or {}
        self.register_buffer("mean", torch.tensor(config.get("mean", (0.485, 0.456, 0.406)))[None, :, None, None])
        self.register_buffer("std", torch.tensor(config.get("std", (0.229, 0.224, 0.225)))[None, :, None, None])
        if frozen:
            self.requires_grad_(requires_grad=False)
            self.backbone.eval()

    def train(self, mode: bool = True) -> TimmFeatureEncoder:  # noqa: FBT001, FBT002
        """Set training mode while keeping a frozen backbone in evaluation mode.

        Args:
            mode: Whether to enable training mode for trainable modules.

        Returns:
            This encoder.
        """
        super().train(mode=mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def _tokens(self, images: Tensor) -> Tensor:
        """Extract the configured token representation from flat images.

        Args:
            images: Image tensor with shape ``[N, C, H, W]``.

        Returns:
            Encoded image tokens with shape ``[N, P, E]``.

        Raises:
            TypeError: If the backbone returns an unsupported value.
            ValueError: If the returned feature tensor has an unsupported rank.
        """
        features = self.backbone.forward_features((images - self.mean) / self.std)
        if isinstance(features, dict):
            features = features.get("x_norm_patchtokens", features.get("last_hidden_state"))
        if not isinstance(features, Tensor):
            raise TypeError("timm forward_features must return a tensor or a supported tensor dictionary.")
        if features.ndim == 4:
            features = features.flatten(2).transpose(1, 2)
        elif features.ndim == 2:
            features = features[:, None]
        elif features.ndim != 3:
            raise ValueError(f"Unsupported timm feature shape {tuple(features.shape)}.")

        prefix = int(getattr(self.backbone, "num_prefix_tokens", 0))
        patches = features[:, prefix:] if prefix and features.shape[1] > prefix else features
        if self.output == "patches":
            return patches
        if self.output == "cls" and prefix:
            return features[:, :1]
        return patches.mean(dim=1, keepdim=True)

    def forward(self, images: Tensor) -> Tensor:
        """Encode images while preserving their leading dimensions.

        Args:
            images: Images with shape ``[..., C, H, W]``.

        Returns:
            Tokens with shape ``[..., patches, embedding]``.

        Raises:
            ValueError: If the input does not contain channel and spatial dimensions.
        """
        if images.ndim < 4:
            raise ValueError("Expected images with shape [..., C, H, W].")
        leading = images.shape[:-3]
        flat = images.reshape(-1, *images.shape[-3:])
        tokens = self._tokens(flat)
        return tokens.reshape(*leading, *tokens.shape[-2:])
