# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2026 Gaoyue Zhou
# Authors: Gaoyue Zhou, Zichen Jeff Cui
# SPDX-License-Identifier: MIT

"""Pretrained visual encoders used by Patch Policy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from types import ModuleType

    from physicalai.policies.patch_policy.config import EncoderOutput, FeatureEncoderName

_TIMM_MODELS: dict[FeatureEncoderName, str] = {
    "dinov2": "vit_small_patch14_dinov2.lvd142m",
}
_WEBSSL_REPO = "facebook/webssl-dino300m-full2b-224"
_WEBSSL_REVISION = "8529cdb3fb75014932af3b896455fc21c386168e"
_TIMM_DINOV2_REVISION = "4610ca143709d58a633b6397a74412c2c3842454"
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_SPATIAL_NDIM = 2
_BATCHED_IMAGE_NDIM = 4
_POOLED_FEATURE_NDIM = 2
_TOKEN_FEATURE_NDIM = 3


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
        msg = "Patch Policy requires timm. Install physicalai-train[patch_policy]."
        raise ImportError(msg) from exc
    return timm


def get_default_encoder_config(name: FeatureEncoderName) -> dict[str, Any] | None:
    """Return local architecture metadata for the selected encoder.

    Returns:
        WebSSL architecture settings, or None for architectures registered locally.

    Raises:
        ValueError: If the encoder name is unsupported.
    """
    if name not in {"dinov2", "webssl", "resnet18"}:
        msg = f"Unsupported encoder_name: {name!r}."
        raise ValueError(msg)
    if name == "webssl":
        # Architecture from the pinned WebSSL revision used for pretrained weights.
        return {
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "image_size": 224,
            "patch_size": 14,
            "mlp_ratio": 4,
            "use_swiglu_ffn": True,
            "layerscale_value": 1.0,
            "layer_norm_eps": 1e-6,
            "qkv_bias": True,
            "apply_layernorm": True,
            "reshape_hidden_states": True,
            "out_features": ["stage24"],
            "out_indices": [24],
        }
    return None


def resolve_encoder_artifacts(
    name: FeatureEncoderName,
    pretrained: bool,  # noqa: FBT001
) -> tuple[dict[str, Any] | None, dict[str, Tensor] | None]:
    """Resolve encoder architecture metadata and optional weights without building a model.

    Args:
        name: Supported visual encoder name.
        pretrained: Whether to download pretrained backbone weights.

    Returns:
        WebSSL architecture settings (otherwise None) and optional provider weights.

    Raises:
        ImportError: If a required optional encoder dependency is unavailable.
        ValueError: If the timm pretrained configuration has no Hugging Face repository.
    """
    if name == "webssl":
        try:
            from huggingface_hub import hf_hub_download  # noqa: PLC0415
            from safetensors.torch import load_file  # noqa: PLC0415
        except ImportError as exc:
            msg = "WebSSL requires transformers and safetensors. Install physicalai-train[patch_policy]."
            raise ImportError(msg) from exc

        config = get_default_encoder_config(name)
        weights = None
        if pretrained:
            weights = load_file(
                hf_hub_download(_WEBSSL_REPO, "model.safetensors", revision=_WEBSSL_REVISION),
                device="cpu",
            )
        return config, weights

    if not pretrained:
        return None, None
    if name == "dinov2":
        timm = _import_timm()
        from safetensors.torch import load_file  # noqa: PLC0415

        # timm exposes its model-free download helper only in this module.
        from timm.models._hub import download_from_hf  # noqa: PLC0415, PLC2701

        config = timm.models.get_pretrained_cfg(_TIMM_MODELS[name])
        if config is None or not config.hf_hub_id:
            msg = f"Encoder {name!r} has no Hugging Face pretrained configuration."
            raise ValueError(msg)
        weights = load_file(
            download_from_hf(f"{config.hf_hub_id}@{_TIMM_DINOV2_REVISION}", "model.safetensors"),
            device="cpu",
        )
        return None, weights

    try:
        from torchvision.models import ResNet18_Weights  # noqa: PLC0415
    except ImportError as exc:
        msg = "ResNet18 requires torchvision. Install physicalai-train[patch_policy]."
        raise ImportError(msg) from exc
    return None, ResNet18_Weights.DEFAULT.get_state_dict(check_hash=True)


class TimmFeatureEncoder(nn.Module):
    """Encode arbitrary leading image dimensions into patch tokens.

    Args:
        name: Supported timm encoder name.
        output: Encoder representation to return for each image.
        encoder_config: Resolved WebSSL architecture configuration; required for WebSSL.
        image_size: Input height and width, defaulting to (224, 224).
    """

    mean: Tensor
    std: Tensor
    backbone: nn.Module

    def __init__(
        self,
        name: FeatureEncoderName,
        output: EncoderOutput,
        *,
        encoder_config: dict[str, Any] | None = None,
        image_size: tuple[int, int] = (224, 224),
    ) -> None:
        """Initialize a frozen visual backbone and its output metadata.

        Raises:
            ImportError: If a required optional encoder dependency is unavailable.
            ValueError: If the image size or backbone metadata is invalid.
        """
        super().__init__()
        self.output = output
        self.name = name
        if len(image_size) != _SPATIAL_NDIM or any(size <= 0 for size in image_size):
            msg = "image_size must contain two positive dimensions."
            raise ValueError(msg)
        self.image_size = (image_size[0], image_size[1])
        if name == "dinov2":
            backbone = _import_timm().create_model(
                _TIMM_MODELS[name],
                pretrained=False,
                num_classes=0,
                img_size=self.image_size,
            )
            self.backbone = backbone
            self.output_dim = int(backbone.num_features)
            patch_embed = getattr(backbone, "patch_embed", None)
            if patch_embed is None:
                msg = f"Encoder {name!r} does not expose a patch embedding with image-size metadata."
                raise ValueError(msg)
            self.num_tokens = int(getattr(patch_embed, "num_patches", 1)) if output == "patches" else 1
        elif name == "webssl":
            if encoder_config is None:
                msg = "WebSSL requires encoder_config. Resolve encoder artifacts before materializing the model."
                raise ValueError(msg)
            try:
                from transformers import Dinov2Config, Dinov2Model  # noqa: PLC0415
            except ImportError as exc:
                msg = "WebSSL requires transformers. Install physicalai-train[patch_policy]."
                raise ImportError(msg) from exc

            config = Dinov2Config(**encoder_config)
            backbone = Dinov2Model(config)
            self.backbone = backbone
            self.output_dim = int(config.hidden_size)
            patch_size = config.patch_size
            if isinstance(patch_size, int):
                patch_size = (patch_size, patch_size)
            self.num_tokens = (
                (self.image_size[0] // patch_size[0]) * (self.image_size[1] // patch_size[1])
                if output == "patches"
                else 1
            )
        else:
            try:
                from torchvision.models import resnet18  # noqa: PLC0415
            except ImportError as exc:
                msg = "ResNet18 requires torchvision. Install physicalai-train[patch_policy]."
                raise ImportError(msg) from exc

            model = resnet18(weights=None)
            self.backbone = nn.Sequential(*list(model.children())[:-1])
            self.output_dim = int(model.fc.in_features)
            self.num_tokens = 1

        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN)[None, :, None, None])
        self.register_buffer("std", torch.tensor(_IMAGENET_STD)[None, :, None, None])
        self.requires_grad_(requires_grad=False)
        self.backbone.eval()

    def load_pretrained_weights(self, state_dict: dict[str, Tensor]) -> None:
        """Load provider weights into the existing backbone without modifying the input.

        Args:
            state_dict: Backbone weights returned by resolve_encoder_artifacts.

        Raises:
            TypeError: If the DINOv2 backbone is not a timm VisionTransformer.
        """
        if self.name == "dinov2":
            from timm.models.vision_transformer import VisionTransformer, checkpoint_filter_fn  # noqa: PLC0415

            if not isinstance(self.backbone, VisionTransformer):
                msg = "DINOv2 pretrained loading requires a timm VisionTransformer backbone."
                raise TypeError(msg)

            # Match timm's pretrained conversion and positional-embedding resizing.
            weights = checkpoint_filter_fn(dict(state_dict), self.backbone)
            weights = {key: value for key, value in weights.items() if not key.startswith("head.")}
        elif self.name == "resnet18":
            children = ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4", "avgpool")
            prefixes = {name: str(index) for index, name in enumerate(children)}
            weights = {}
            for key, value in state_dict.items():
                prefix, separator, suffix = key.partition(".")
                if prefix != "fc":
                    weights[prefixes.get(prefix, prefix) + separator + suffix] = value
        else:
            weights = state_dict
        self.backbone.load_state_dict(weights, strict=True)

    def train(self, mode: bool = True) -> TimmFeatureEncoder:  # noqa: FBT001, FBT002
        """Set training mode while keeping a frozen backbone in evaluation mode.

        Args:
            mode: Whether to enable training mode for trainable modules.

        Returns:
            This encoder.
        """
        super().train(mode=mode)
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
        if self.name == "webssl":
            features = self.backbone(
                pixel_values=images,
                interpolate_pos_encoding=True,
            ).last_hidden_state
            if self.output == "cls":
                return features[:, :1]
            patches = features[:, 1:]
            return patches if self.output == "patches" else patches.mean(dim=1, keepdim=True)

        if isinstance(self.backbone, nn.Sequential):
            features = self.backbone(images)
        else:
            forward_features = getattr(self.backbone, "forward_features", None)
            if not callable(forward_features):
                msg = f"Encoder {self.name!r} does not expose forward_features()."
                raise TypeError(msg)
            features = forward_features(images)
        if isinstance(features, dict):
            feature_key = "x_norm_clstoken" if self.output == "cls" else "x_norm_patchtokens"
            features = features.get(feature_key, features.get("last_hidden_state"))
        if not isinstance(features, Tensor):
            msg = "timm forward_features must return a tensor or a supported tensor dictionary."
            raise TypeError(msg)
        if features.ndim == _BATCHED_IMAGE_NDIM:
            features = features.flatten(2).transpose(1, 2)
        elif features.ndim == _POOLED_FEATURE_NDIM:
            features = features[:, None]
        elif features.ndim != _TOKEN_FEATURE_NDIM:
            msg = f"Unsupported timm feature shape {tuple(features.shape)}."
            raise ValueError(msg)

        prefix = int(getattr(self.backbone, "num_prefix_tokens", 0))
        patches = features[:, prefix:] if prefix and features.shape[1] > prefix else features
        if self.output == "patches":
            return patches
        if self.output == "cls" and prefix:
            return features[:, :1]
        return patches.mean(dim=1, keepdim=True)

    @torch.no_grad()
    def forward(self, images: Tensor) -> Tensor:
        """Encode images while preserving their leading dimensions.

        Args:
            images: Images with shape ``[..., C, H, W]``.

        Returns:
            Tokens with shape ``[..., patches, embedding]``.

        Raises:
            ValueError: If the input does not contain channel and spatial dimensions.
        """
        if images.ndim < _BATCHED_IMAGE_NDIM:
            msg = "Expected images with shape [..., C, H, W]."
            raise ValueError(msg)
        leading = images.shape[:-3]
        flat = images.reshape(-1, *images.shape[-3:])
        tokens = self._tokens(flat)
        return tokens.reshape(*leading, *tokens.shape[-2:])
