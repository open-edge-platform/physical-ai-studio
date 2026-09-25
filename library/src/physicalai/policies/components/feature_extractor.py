# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Generic image feature extractor for policy vision encoders.

Wraps a timm model, a torchvision model or any ``nn.Module`` and returns
features from one or more of its layers. Each layer can be pooled and the
layers can then be aggregated into a single tensor.

Supports both architecture families:
- Convolutional layers output feature maps of shape (B, C, H, W).
- Transformer layers output tokens of shape (B, N, D), where the first
  ``num_prefix_tokens`` tokens are CLS/register tokens.

Examples:
    DINOv2 CLS token:

    >>> encoder = FeatureExtractor("vit_small_patch14_dinov2.lvd142m", pooling="cls", dynamic_img_size=True)
    >>> encoder(torch.rand(2, 3, 224, 224)).shape
    torch.Size([2, 384])

    All four ResNet stages, average pooled and concatenated:

    >>> resnet = torchvision.models.resnet18(weights="DEFAULT")
    >>> encoder = FeatureExtractor(
    ...     resnet, layers=["layer1", "layer2", "layer3", "layer4"], pooling="avg", aggregate="concat"
    ... )
    >>> encoder(torch.rand(2, 3, 224, 224)).shape
    torch.Size([2, 960])
"""

from __future__ import annotations

import difflib
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import timm
import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Sequence

Pooling = Literal["none", "avg", "max", "cls"]
Aggregate = Literal["none", "concat", "mean"]
type ImageInput = Image.Image | np.ndarray | torch.Tensor

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
OUTPUT_KEY = "output"


class FeatureExtractor(nn.Module):
    """Extract (optionally pooled and aggregated) features from an image backbone.

    Args:
        backbone: A timm model name (created with ``timm.create_model``) or an
            already built ``nn.Module``, e.g. a torchvision model.
        layers: Names of the modules to extract features from, as listed by
            ``model.named_modules()`` (e.g. ``["layer4"]`` or ``["blocks.11"]``).
            ``None`` uses the final features of the model.
        pooling: How each layer's features are reduced:
            ``"none"`` keeps feature maps (B, C, H, W) or patch tokens (B, N, D),
            ``"avg"``/``"max"`` pool spatial positions or patch tokens to (B, C),
            ``"cls"`` takes the CLS token (B, D) of transformer layers.
        aggregate: How multiple layers are combined:
            ``"none"`` returns a dict of features keyed by layer name,
            ``"concat"`` concatenates along the channel dimension,
            ``"mean"`` averages features of equal shape.
        pretrained: Load pretrained weights when ``backbone`` is a timm model name.
        frozen: Freeze the backbone weights and keep it in eval mode.
        normalize: Normalize inputs with ``mean`` and ``std``.
        mean: Per-channel normalization mean. Defaults to the timm pretrained
            config if available, else ImageNet statistics.
        std: Per-channel normalization std. Same defaults as ``mean``.
        num_prefix_tokens: Number of CLS/register tokens before the patch tokens.
            Defaults to ``model.num_prefix_tokens`` if available, else 0.
        **model_kwargs: Extra arguments passed to ``timm.create_model``.

    Raises:
        ValueError: If ``pooling``, ``aggregate`` or a layer name is invalid.
    """

    def __init__(
        self,
        backbone: str | nn.Module,
        layers: Sequence[str] | None = None,
        pooling: Pooling = "none",
        aggregate: Aggregate = "none",
        *,
        pretrained: bool = True,
        frozen: bool = True,
        normalize: bool = True,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
        num_prefix_tokens: int | None = None,
        **model_kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize the feature extractor.

        Raises:
            ValueError: If ``pooling`` or ``aggregate`` is invalid.
        """
        super().__init__()
        if pooling not in {"none", "avg", "max", "cls"}:
            msg = f"Unknown pooling '{pooling}'. Expected one of: none, avg, max, cls."
            raise ValueError(msg)
        if aggregate not in {"none", "concat", "mean"}:
            msg = f"Unknown aggregate '{aggregate}'. Expected one of: none, concat, mean."
            raise ValueError(msg)

        if isinstance(backbone, str):
            backbone = timm.create_model(backbone, pretrained=pretrained, **model_kwargs)
        self.model = backbone
        self.layers = list(layers) if layers else []
        self.pooling = pooling
        self.aggregate = aggregate
        self.frozen = frozen
        self.normalize = normalize
        self.num_prefix_tokens = (
            num_prefix_tokens if num_prefix_tokens is not None else getattr(self.model, "num_prefix_tokens", 0)
        )

        pretrained_cfg = getattr(self.model, "pretrained_cfg", {})
        mean = mean or pretrained_cfg.get("mean", IMAGENET_MEAN)
        std = std or pretrained_cfg.get("std", IMAGENET_STD)
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1), persistent=False)

        self._features: dict[str, torch.Tensor] = {}
        self._register_hooks()
        self._feature_dims: dict[str, int] | None = None

        if frozen:
            self.model.requires_grad_(requires_grad=False)
            self.model.eval()

    def _register_hooks(self) -> None:
        """Register forward hooks that store the output of each requested layer.

        Raises:
            ValueError: If a layer name does not exist in the model.
        """
        modules = dict(self.model.named_modules())
        for name in self.layers:
            if name not in modules:
                suggestions = difflib.get_close_matches(name, modules.keys(), n=5)
                msg = f"Layer '{name}' not found in model. Did you mean one of: {suggestions}?"
                raise ValueError(msg)

            def hook(_module: nn.Module, _inputs: Any, output: Any, name: str = name) -> None:  # noqa: ANN401
                self._features[name] = output[0] if isinstance(output, (tuple, list)) else output

            modules[name].register_forward_hook(hook)

    def train(self, mode: bool = True) -> FeatureExtractor:  # noqa: FBT001, FBT002
        """Set training mode, keeping a frozen backbone in eval mode.

        Args:
            mode: Whether to set training mode.

        Returns:
            This module.
        """
        super().train(mode)
        if self.frozen:
            self.model.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        """Extract features from a batch of images.

        Args:
            images: Images of shape (B, C, H, W) with values in [0, 1].

        Returns:
            A single tensor if there is one layer or the layers are aggregated,
            otherwise a dict of tensors keyed by layer name.
        """
        features = {name: self._pool(feature) for name, feature in self._extract(images).items()}

        if len(features) == 1:
            return next(iter(features.values()))
        if self.aggregate == "concat":
            return _concat(list(features.values()))
        if self.aggregate == "mean":
            return _mean(list(features.values()))
        return features

    def _extract(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the backbone and return the raw features of each layer.

        Args:
            images: Images of shape (B, C, H, W) with values in [0, 1].

        Returns:
            Raw layer outputs keyed by layer name.
        """
        if self.normalize:
            images = (images - self.mean) / self.std

        # timm models expose forward_features, which skips the classifier head.
        run = getattr(self.model, "forward_features", self.model)
        self._features = {}
        output = run(images)
        return self._features if self.layers else {OUTPUT_KEY: output}

    def _pool(self, feature: torch.Tensor) -> torch.Tensor:
        """Reduce a single layer's features according to ``self.pooling``.

        Args:
            feature: Feature map (B, C, H, W) or tokens (B, N, D).

        Returns:
            The pooled features.

        Raises:
            ValueError: If CLS pooling is requested for a feature map.
        """
        if feature.ndim == 4:  # noqa: PLR2004
            if self.pooling == "cls":
                msg = "CLS pooling requires transformer tokens (B, N, D), got a feature map (B, C, H, W)."
                raise ValueError(msg)
            pool_dims: tuple[int, ...] = (2, 3)
        else:
            if self.pooling == "cls":
                return feature[:, 0]
            feature = feature[:, self.num_prefix_tokens :]  # keep patch tokens only
            pool_dims = (1,)

        if self.pooling == "avg":
            return feature.mean(dim=pool_dims)
        if self.pooling == "max":
            return feature.amax(dim=pool_dims)
        return feature

    @property
    def feature_dims(self) -> dict[str, int]:
        """Channel/embedding dimension of each layer's output, keyed by layer name.

        Computed once with a dummy forward pass.
        """
        if self._feature_dims is None:
            param = next(self.model.parameters())
            dummy = torch.zeros(1, 3, 224, 224, device=param.device, dtype=param.dtype)
            with torch.no_grad():
                features = self._extract(dummy)
            self._feature_dims = {name: _channels(self._pool(feature)) for name, feature in features.items()}
        return self._feature_dims

    @torch.no_grad()
    def encode_image(
        self,
        image: ImageInput | list[ImageInput],
        *,
        channels_last: bool | None = None,
        size: int | tuple[int, int] | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Extract features from one or more images in a common format.

        Args:
            image: A PIL image, numpy array or tensor, or a list of them.
                Integer images are scaled from [0, 255] to [0, 1].
            channels_last: Whether arrays/tensors are (H, W, C) instead of (C, H, W).
                Defaults to ``True`` for PIL images and numpy arrays and
                ``False`` for tensors.
            size: Optional (H, W) or square size to resize each image to.

        Returns:
            Features for the batch of images, as returned by ``forward``.
        """
        images = image if isinstance(image, list) else [image]
        param = next(self.model.parameters())
        batch = torch.stack([_to_chw_tensor(img, channels_last=channels_last, size=size) for img in images])
        return self(batch.to(device=param.device, dtype=param.dtype))


def _to_chw_tensor(
    image: ImageInput,
    *,
    channels_last: bool | None,
    size: int | tuple[int, int] | None,
) -> torch.Tensor:
    """Convert an image to a float (C, H, W) tensor with values in [0, 1].

    Args:
        image: A PIL image, numpy array or tensor.
        channels_last: Whether the image is (H, W, C). ``None`` means ``True``
            for PIL images and numpy arrays and ``False`` for tensors.
        size: Optional (H, W) or square size to resize to.

    Returns:
        The image as a (C, H, W) float tensor.
    """
    if channels_last is None:
        channels_last = not isinstance(image, torch.Tensor)
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))

    tensor = torch.as_tensor(image)
    if tensor.ndim == 2:  # noqa: PLR2004
        tensor = tensor.unsqueeze(-1 if channels_last else 0)
    if channels_last:
        tensor = tensor.permute(2, 0, 1)
    tensor = tensor.float() / 255.0 if not tensor.is_floating_point() else tensor.float()

    if size is not None:
        tensor = F.interpolate(tensor.unsqueeze(0), size=size, mode="bilinear", antialias=True).squeeze(0)
    return tensor


def _channels(feature: torch.Tensor) -> int:
    """Return the channel dimension of a feature map (dim 1) or tokens/vectors (last dim).

    Returns:
        The channel dimension size.
    """
    return feature.shape[1] if feature.ndim == 4 else feature.shape[-1]  # noqa: PLR2004


def _concat(features: list[torch.Tensor]) -> torch.Tensor:
    """Concatenate features along the channel dimension.

    Feature maps are resized to the largest spatial size first.

    Returns:
        The concatenated features.
    """
    if all(feature.ndim == 4 for feature in features):  # noqa: PLR2004
        size = max((feature.shape[-2:] for feature in features), key=lambda s: s[0] * s[1])
        features = [F.interpolate(feature, size=size, mode="bilinear", align_corners=False) for feature in features]
        return torch.cat(features, dim=1)
    return torch.cat(features, dim=-1)


def _mean(features: list[torch.Tensor]) -> torch.Tensor:
    """Average features of equal shape.

    Returns:
        The element-wise mean of the features.

    Raises:
        ValueError: If the features have different shapes.
    """
    shapes = {tuple(feature.shape) for feature in features}
    if len(shapes) > 1:
        msg = f"aggregate='mean' requires features of equal shape, got {sorted(shapes)}."
        raise ValueError(msg)
    return torch.stack(features).mean(dim=0)
