# Copyright 2025 2toINF (https://github.com/2toINF) and The HuggingFace Inc. team.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pre/post-processing for the XVLA policy.

The preprocessor turns a Studio ``Observation`` batch into the tensors
:class:`~physicalai.policies.xvla.model.XVLAModel` consumes:

- the language prompt, tokenized to a fixed length with Florence-2's BART tokenizer;
- the cameras, ImageNet-normalized and stacked into one ``[B, V, C, H, W]`` tensor with a
  per-view validity mask, so a checkpoint trained with more cameras than the dataset
  carries still lines up;
- the proprioceptive state, zero-padded to the model's fixed width;
- the domain index that selects the domain-aware projections and soft prompts.

State and action normalization is symmetric -- whatever the preprocessor applies on the way
in, the postprocessor inverts on the way out -- and defaults to identity, because XVLA's
action spaces carry their own per-channel loss scaling and the published checkpoints are
trained on raw units.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import torch
import torch.nn.functional as F  # noqa: N812

from physicalai.data import Feature, FeatureType, NormalizationParameters
from physicalai.data.constants import IMAGE_MASKS, IMAGES, TOKENIZED_PROMPT
from physicalai.data.observation import ACTION, STATE, TASK, Observation
from physicalai.policies.utils.normalization import FeatureNormalizeTransform, NormalizationType

from .model import DOMAIN_ID

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
"""Per-channel ImageNet mean; Florence-2's vision tower expects images normalized with it."""

IMAGENET_STD = (0.229, 0.224, 0.225)
"""Per-channel ImageNet standard deviation."""

_UINT8_SCALE = 255.0


def resize_with_pad(images: torch.Tensor, height: int, width: int, pad_value: float = 0.0) -> torch.Tensor:
    """Resize ``[B, C, H, W]`` images without distortion, padding on the left and top.

    Left/top padding is the XVLA (and SmolVLA) convention; changing it would shift every
    image relative to the positional embeddings a checkpoint was trained with.

    Args:
        images: Images of shape ``[B, C, H, W]``.
        height: Target height.
        width: Target width.
        pad_value: Value used for the padded border.

    Returns:
        Images of shape ``[B, C, height, width]``.

    Raises:
        ValueError: If ``images`` is not 4-dimensional.
    """
    if images.ndim != 4:  # noqa: PLR2004
        msg = f"(B, C, H, W) expected, got {tuple(images.shape)}"
        raise ValueError(msg)

    current_height, current_width = images.shape[2:]
    if current_height == height and current_width == width:
        return images

    ratio = max(current_width / width, current_height / height)
    resized_height = int(current_height / ratio)
    resized_width = int(current_width / ratio)
    resized = F.interpolate(images, size=(resized_height, resized_width), mode="bilinear", align_corners=False)

    pad_height = max(0, height - resized_height)
    pad_width = max(0, width - resized_width)
    return F.pad(resized, (pad_width, 0, pad_height, 0), value=pad_value)


def _norm_map_for_mode(mode: str) -> dict[FeatureType, NormalizationType]:
    """Return the state/action normalization mapping for the given mode string.

    Args:
        mode: ``"IDENTITY"``, ``"MEAN_STD"`` or ``"QUANTILES"``.

    Returns:
        Mapping from ``FeatureType`` to ``NormalizationType``.
    """
    norm_type = NormalizationType(mode)
    return {FeatureType.STATE: norm_type, FeatureType.ACTION: norm_type}


class XVLAPreprocessor(torch.nn.Module):
    """Turn a flattened Studio batch into XVLA's model inputs.

    Args:
        features: Feature definitions carrying the state/action normalization statistics.
        num_image_views: Number of camera slots to emit, real cameras included. Views beyond
            the ones the batch carries are appended as masked-out zero images.
        image_resolution: Resize every camera to ``(height, width)``; ``None`` keeps the
            dataset's resolution.
        max_state_dim: Width the proprioceptive state is padded or truncated to. ``0``
            disables proprioception.
        tokenizer_name: HuggingFace tokenizer for the language prompt.
        tokenizer_max_length: Fixed prompt length, padded and truncated.
        domain_id: Domain index used when the batch carries none.
        domain_feature_key: Batch key holding a per-sample domain index.
        normalization_mode: ``"IDENTITY"`` (default), ``"MEAN_STD"`` or ``"QUANTILES"``.
    """

    def __init__(
        self,
        features: dict[str, Feature] | None = None,
        *,
        num_image_views: int = 1,
        image_resolution: tuple[int, int] | None = None,
        max_state_dim: int = 32,
        tokenizer_name: str = "facebook/bart-large",
        tokenizer_max_length: int = 64,
        domain_id: int = 0,
        domain_feature_key: str | None = None,
        normalization_mode: str = "IDENTITY",
    ) -> None:
        """Build the state/action normalizer and record the tokenizer settings."""
        super().__init__()
        self.num_image_views = num_image_views
        self.image_resolution = image_resolution
        self.max_state_dim = max_state_dim
        self.tokenizer_name = tokenizer_name
        self.tokenizer_max_length = tokenizer_max_length
        self.domain_id = domain_id
        self.domain_feature_key = domain_feature_key
        self.normalization_mode = normalization_mode
        self._tokenizer: Any = None

        norm_map = _norm_map_for_mode(normalization_mode)
        if features:
            self._normalizer: torch.nn.Module = FeatureNormalizeTransform(features, norm_map)
        else:
            self._normalizer = torch.nn.Identity()

    @property
    def tokenizer(self) -> Any:  # noqa: ANN401
        """The lazily loaded prompt tokenizer.

        Raises:
            ImportError: If ``transformers`` is not installed.
        """
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer  # noqa: PLC0415
            except ImportError as e:
                msg = "XVLA requires transformers. Install with: pip install 'physicalai-train[xvla]'"
                raise ImportError(msg) from e
            self._tokenizer = AutoTokenizer.from_pretrained(  # nosec B615
                self.tokenizer_name,
                use_fast=True,
                padding_side="right",
            )
        return self._tokenizer

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Preprocess one flattened batch.

        Args:
            batch: Flattened batch dict, as produced by ``Observation.to_dict()``.

        Returns:
            A shallow copy carrying the tokenized prompt, the stacked cameras and their
            mask, the padded state, the domain index and the normalized actions.
        """
        batch = self._normalizer(dict(batch))

        images, image_mask = self._prepare_images(batch)
        device = images.device

        batch[IMAGES] = images
        batch[IMAGE_MASKS] = image_mask
        batch[TOKENIZED_PROMPT] = self._tokenize(batch.get(TASK), images.shape[0], device)
        batch[STATE] = self._prepare_state(batch, images.shape[0], device)
        batch[DOMAIN_ID] = self._resolve_domain_id(batch, images.shape[0], device)
        return batch

    def _prepare_images(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """Collect, normalize and stack the cameras.

        Args:
            batch: Flattened batch dict; the consumed camera entries are removed from it.

        Returns:
            Tuple of images ``[B, V, C, H, W]`` and a validity mask ``[B, V]``.

        Raises:
            ValueError: If the batch carries no camera.
        """
        keys = [
            key
            for key in Observation.get_flattened_keys(batch, IMAGES)
            if "is_pad" not in key and isinstance(batch.get(key), torch.Tensor)
        ]
        if not keys:
            msg = f"XVLA requires at least one camera; batch carries none. Keys: {sorted(batch)}"
            raise ValueError(msg)

        views = []
        for key in keys:
            image = batch.pop(key)
            if image.ndim == 5:  # noqa: PLR2004
                # A temporal clip: XVLA conditions on the most recent frame.
                image = image[:, -1]
            views.append(self._normalize_image(image))

        images = torch.stack(views, dim=1)
        mask = torch.ones(images.shape[:2], dtype=torch.bool, device=images.device)

        num_pad = max(0, self.num_image_views - images.shape[1])
        if num_pad:
            pad_images = images.new_zeros((images.shape[0], num_pad, *images.shape[2:]))
            pad_mask = mask.new_zeros((mask.shape[0], num_pad))
            images = torch.cat([images, pad_images], dim=1)
            mask = torch.cat([mask, pad_mask], dim=1)

        return images, mask

    def _normalize_image(self, image: torch.Tensor) -> torch.Tensor:
        """Bring one camera to ``[B, C, H, W]`` ImageNet-normalized floats.

        Args:
            image: Camera tensor in ``[B, C, H, W]`` or ``[B, H, W, C]`` layout, either
                ``uint8`` in ``[0, 255]`` or floating point already scaled to ``[0, 1]``.

        Returns:
            The normalized image of shape ``[B, C, H, W]``.
        """
        channels = 3
        if image.shape[1] != channels and image.shape[-1] == channels:
            image = image.permute(0, 3, 1, 2)

        image = image.to(torch.float32) if image.is_floating_point() else image.to(torch.float32) / _UINT8_SCALE

        if self.image_resolution is not None:
            image = resize_with_pad(image, *self.image_resolution)

        mean = torch.tensor(IMAGENET_MEAN, device=image.device, dtype=image.dtype).view(1, channels, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=image.device, dtype=image.dtype).view(1, channels, 1, 1)
        return (image - mean) / std

    def _prepare_state(self, batch: dict[str, Any], batch_size: int, device: torch.device) -> torch.Tensor:
        """Pad or truncate the proprioceptive state to the model's fixed width.

        Args:
            batch: Flattened batch dict.
            batch_size: Number of samples in the batch.
            device: Device the model runs on.

        Returns:
            State of shape ``[B, max_state_dim]``; empty when proprioception is disabled or
            the batch carries no state.
        """
        state = batch.get(STATE)
        if self.max_state_dim == 0 or state is None:
            return torch.zeros(batch_size, 0, device=device)

        if state.ndim > 2:  # noqa: PLR2004
            state = state[:, -1]
        state = state.to(device=device, dtype=torch.float32)

        if state.shape[-1] > self.max_state_dim:
            return state[..., : self.max_state_dim]
        return F.pad(state, (0, self.max_state_dim - state.shape[-1]))

    def _tokenize(self, task: Any, batch_size: int, device: torch.device) -> torch.Tensor:  # noqa: ANN401
        """Tokenize the language prompt to a fixed length.

        Args:
            task: The batch's task strings, a single string, or ``None``.
            batch_size: Number of samples in the batch.
            device: Device the model runs on.

        Returns:
            Token ids of shape ``[B, tokenizer_max_length]``.
        """
        if task is None:
            prompts = [""] * batch_size
        elif isinstance(task, str):
            prompts = [task] * batch_size
        else:
            prompts = [str(t) for t in task]

        prompts = [prompt.strip().replace("_", " ").replace("\n", " ") for prompt in prompts]
        encoded = self.tokenizer(
            prompts,
            max_length=self.tokenizer_max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return encoded["input_ids"].to(device)

    def _resolve_domain_id(self, batch: dict[str, Any], batch_size: int, device: torch.device) -> torch.Tensor:
        """Resolve the per-sample domain index.

        Args:
            batch: Flattened batch dict.
            batch_size: Number of samples in the batch.
            device: Device the model runs on.

        Returns:
            Domain indices of shape ``[B]``.
        """
        candidates = [self.domain_feature_key] if self.domain_feature_key else [DOMAIN_ID, f"extra.{DOMAIN_ID}"]
        value = next((batch[key] for key in candidates if key and batch.get(key) is not None), None)
        if value is None:
            return torch.full((batch_size,), self.domain_id, dtype=torch.long, device=device)

        domain_id = torch.as_tensor(value, device=device).to(dtype=torch.long)
        if domain_id.ndim == 0:
            domain_id = domain_id.expand(batch_size)
        elif domain_id.ndim > 1:
            domain_id = domain_id.reshape(domain_id.shape[0], -1)[:, 0]
        if domain_id.shape[0] != batch_size:
            domain_id = domain_id.expand(batch_size)
        return domain_id.contiguous()


class XVLAPostprocessor(torch.nn.Module):
    """Map predicted actions back to the dataset's units.

    Args:
        features: Feature definitions carrying the action normalization statistics.
        normalization_mode: The mode the preprocessor applied.
    """

    def __init__(
        self,
        features: dict[str, Feature] | None = None,
        normalization_mode: str = "IDENTITY",
    ) -> None:
        """Build the action denormalizer."""
        super().__init__()
        norm_map = _norm_map_for_mode(normalization_mode)
        if features:
            action_features = {k: v for k, v in features.items() if v.ftype == FeatureType.ACTION}
            self._denormalizer: torch.nn.Module = FeatureNormalizeTransform(action_features, norm_map, inverse=True)
        else:
            self._denormalizer = torch.nn.Identity()

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Denormalize the batch's actions.

        Args:
            batch: Dict holding an ``action`` entry.

        Returns:
            A shallow copy with denormalized actions.
        """
        batch = dict(batch)
        if batch.get(ACTION) is not None:
            batch[ACTION] = self._denormalizer({ACTION: batch[ACTION]})[ACTION]
        return batch


def build_features(stats: dict[str, dict[str, Any]] | None) -> dict[str, Feature]:
    """Build the state and action :class:`~physicalai.data.Feature` set from dataset statistics.

    Args:
        stats: Dataset statistics keyed by feature name (``"observation.state"``, ``"action"``).

    Returns:
        Features keyed by the Studio batch name (``"state"``, ``"action"``).
    """
    features: dict[str, Feature] = {}
    if not stats:
        return features

    for key, stat in stats.items():
        if ACTION in key:
            ftype = FeatureType.ACTION
        elif STATE in key:
            ftype = FeatureType.STATE
        else:
            continue

        # Map dataset names ("observation.state") onto Studio batch keys ("state").
        raw_name = str(stat.get("name", key))
        name = raw_name.rsplit("observation.", maxsplit=1)[-1]

        features[name] = Feature(
            name=name,
            ftype=ftype,
            shape=cast("tuple[int, ...]", tuple(stat.get("shape", ()))),
            normalization_data=NormalizationParameters(
                mean=cast("list[float] | None", stat.get("mean")),
                std=cast("list[float] | None", stat.get("std")),
                q01=cast("list[float] | None", stat.get("q01")),
                q99=cast("list[float] | None", stat.get("q99")),
            ),
        )
    return features


def count_camera_features(stats: dict[str, dict[str, Any]] | None) -> int:
    """Count the visual features in a set of dataset statistics.

    Args:
        stats: Dataset statistics keyed by feature name.

    Returns:
        The number of cameras the dataset provides.
    """
    if not stats:
        return 0
    return sum(1 for stat in stats.values() if str(FeatureType.VISUAL) in str(stat.get("type", "")))


def resolve_num_image_views(
    stats: dict[str, dict[str, Any]] | None,
    *,
    num_image_views: int | None = None,
    empty_cameras: int = 0,
) -> int:
    """Decide how many camera slots the model should expect.

    Args:
        stats: Dataset statistics, used to count the real cameras.
        num_image_views: Explicit override from the config.
        empty_cameras: Masked-out slots appended to the real cameras.

    Returns:
        The number of camera slots, never fewer than one.
    """
    from_dataset = count_camera_features(stats) + empty_cameras
    return max(1, num_image_views or 0, from_dataset)


def make_xvla_preprocessors(
    stats: dict[str, dict[str, Any]] | None = None,
    *,
    num_image_views: int = 1,
    image_resolution: tuple[int, int] | None = None,
    max_state_dim: int = 32,
    tokenizer_name: str = "facebook/bart-large",
    tokenizer_max_length: int = 64,
    domain_id: int = 0,
    domain_feature_key: str | None = None,
    normalization_mode: str = "IDENTITY",
) -> tuple[XVLAPreprocessor, XVLAPostprocessor]:
    """Create the preprocessor / postprocessor pair for XVLA.

    Args:
        stats: Dataset statistics (from a checkpoint or a training dataset).
        num_image_views: Number of camera slots to emit.
        image_resolution: Resize every camera to ``(height, width)``; ``None`` keeps the
            dataset's resolution.
        max_state_dim: Width the proprioceptive state is padded to; ``0`` disables it.
        tokenizer_name: HuggingFace tokenizer for the language prompt.
        tokenizer_max_length: Fixed prompt length.
        domain_id: Domain index used when the batch carries none.
        domain_feature_key: Batch key holding a per-sample domain index.
        normalization_mode: ``"IDENTITY"`` (default), ``"MEAN_STD"`` or ``"QUANTILES"``.

    Returns:
        Tuple of ``(preprocessor, postprocessor)``.
    """
    features = build_features(stats)
    return (
        XVLAPreprocessor(
            features,
            num_image_views=num_image_views,
            image_resolution=image_resolution,
            max_state_dim=max_state_dim,
            tokenizer_name=tokenizer_name,
            tokenizer_max_length=tokenizer_max_length,
            domain_id=domain_id,
            domain_feature_key=domain_feature_key,
            normalization_mode=normalization_mode,
        ),
        XVLAPostprocessor(features, normalization_mode=normalization_mode),
    )


__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "XVLAPostprocessor",
    "XVLAPreprocessor",
    "build_features",
    "count_camera_features",
    "make_xvla_preprocessors",
    "resize_with_pad",
    "resolve_num_image_views",
]
