# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Preprocessor and postprocessor for the VLA-Adapter model.

Two details are taken from upstream rather than chosen.

``image_resize_strategy: "resize-naive"`` in the checkpoint means a straight
**bicubic** stretch to 224x224 — no letterboxing, no centre crop.

The fused backbone feeds each tower its *own* normalization, so every view
becomes six channels: three for DINOv2 (ImageNet stats) then three for SigLIP
(symmetric +/-1). Those constants are resolved from ``timm.get_pretrained_cfg``
rather than hardcoded, so they track the tower ids. Views are concatenated on
the channel axis, giving ``(B, 6 * num_images, H, W)`` — the layout
``PrismaticVisionBackbone`` splits back apart.

State and action use ``NormalizationType.QUANTILES``, the 1st/99th-percentile
mapping to [-1, 1] that upstream calls ``BOUNDS_Q99``. Runtime's
``StatsNormalizer`` implements the identical formula under ``mode="quantiles"``.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import torch
import torch.nn.functional as F  # noqa: N812

from physicalai.data import Feature, FeatureType, NormalizationParameters
from physicalai.data.constants import TOKENIZED_PROMPT, TOKENIZED_PROMPT_MASK
from physicalai.data.observation import ACTION, IMAGES, STATE, TASK, Observation
from physicalai.policies.utils.normalization import FeatureNormalizeTransform, NormalizationType

logger = logging.getLogger(__name__)


# An image tensor carries a leading time axis when the datamodule applies
# observation delta timestamps: (B, T, C, H, W) rather than (B, C, H, W).
IMAGE_DIMS_WITH_TIME = 5

# VLA-Adapter inherits OpenVLA's BOUNDS_Q99 convention for both state and action.
NORM_MAP = {
    FeatureType.STATE: NormalizationType.QUANTILES,
    FeatureType.ACTION: NormalizationType.QUANTILES,
}


class VLAAdapterPreprocessor(torch.nn.Module):
    """Transform raw observations into VLA-Adapter model inputs."""

    # Registered buffers, declared so they type as tensors rather than modules.
    primary_mean: torch.Tensor
    primary_std: torch.Tensor
    secondary_mean: torch.Tensor
    secondary_std: torch.Tensor

    def __init__(
        self,
        max_state_dim: int = 8,
        max_action_dim: int = 7,
        image_resolution: tuple[int, int] = (224, 224),
        vision_backbone_ids: tuple[str, str] = (
            "vit_large_patch14_reg4_dinov2.lvd142m",
            "vit_so400m_patch14_siglip_224",
        ),
        image_key_reorder_map: dict[str, int] | None = None,
        num_cameras: int = 0,
        features: dict[str, Feature] | None = None,
        max_token_len: int = 48,
        tokenizer_name: str = "Qwen/Qwen2.5-0.5B",
        padding: str = "max_length",
    ) -> None:
        """Initialize the preprocessor.

        Args:
            max_state_dim: Proprioceptive state dimension.
            max_action_dim: Action dimension.
            image_resolution: Target ``(height, width)``.
            vision_backbone_ids: timm ids of the primary and secondary towers,
                used to resolve per-tower pixel statistics.
            image_key_reorder_map: Image-key to camera-slot mapping.
            num_cameras: Camera slots; <= 0 keeps only batch cameras.
            features: Feature descriptors for normalization, or None.
            max_token_len: Fixed language token length.
            tokenizer_name: HuggingFace tokenizer identifier.
            padding: Tokenizer padding; ``"max_length"`` keeps export shapes
                static.
        """
        super().__init__()

        self.max_state_dim = max_state_dim
        self.max_action_dim = max_action_dim
        self.image_resolution = image_resolution
        self.image_key_reorder_map = {
            key if key.startswith(f"{IMAGES}.") else f"{IMAGES}.{key}": order
            for key, order in (image_key_reorder_map or {}).items()
        }
        self.num_cameras = num_cameras
        self.max_token_len = max_token_len
        self.tokenizer_name = tokenizer_name
        self.padding = padding

        import timm  # noqa: PLC0415
        from transformers import AutoTokenizer  # noqa: PLC0415

        tokenizer: Any = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer

        # Per-tower pixel statistics, registered as buffers so they move with
        # the module and are captured by exported graphs.
        for name, model_id in zip(("primary", "secondary"), vision_backbone_ids, strict=False):
            cfg: Any = timm.get_pretrained_cfg(model_id)
            self.register_buffer(f"{name}_mean", torch.tensor(cfg.mean).view(1, 3, 1, 1), persistent=False)
            self.register_buffer(f"{name}_std", torch.tensor(cfg.std).view(1, 3, 1, 1), persistent=False)

        if features is not None:
            self._state_action_normalizer: torch.nn.Module = FeatureNormalizeTransform(features, NORM_MAP)
        else:
            self._state_action_normalizer = torch.nn.Identity()

    def _tokenize(self, task: str | list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize the task description(s) to a fixed length.

        Args:
            task: A string or list of task descriptions.

        Returns:
            ``(input_ids, attention_mask)``.
        """
        prompts = [task] if isinstance(task, str) else list(task)
        encoded = self.tokenizer(
            prompts,
            padding=self.padding,
            truncation=True,
            max_length=self.max_token_len,
            return_tensors="pt",
        )
        return encoded["input_ids"], encoded["attention_mask"]

    def _ordered_image_keys(self, batch_img_keys: list[str]) -> list[str]:
        """Resolve deterministic camera ordering.

        Args:
            batch_img_keys: Image keys present in the batch.

        Returns:
            Image keys in camera-slot order.

        Raises:
            ValueError: If ``image_key_reorder_map`` does not match the keys.
        """
        if not self.image_key_reorder_map:
            return sorted(batch_img_keys)

        if set(self.image_key_reorder_map) != set(batch_img_keys):
            msg = (
                "image_key_reorder_map keys must match the batch image keys exactly. "
                f"Expected {sorted(self.image_key_reorder_map)}, got {sorted(batch_img_keys)}."
            )
            raise ValueError(msg)
        return sorted(batch_img_keys, key=lambda key: self.image_key_reorder_map[key])

    def _stack_view(self, view: torch.Tensor) -> torch.Tensor:
        """Resize one view and stack its two normalised copies.

        Args:
            view: Images ``(B, 3, H, W)``, or ``(B, T, 3, H, W)`` when the
                datamodule has applied observation delta timestamps.

        Returns:
            ``(B, 6, *image_resolution)``.
        """
        # The datamodule sets observation delta timestamps from the model's
        # `observation_delta_indices`, which adds a time axis even for a single
        # step. Keep the most recent frame, as the other policies do.
        if view.dim() == IMAGE_DIMS_WITH_TIME:
            view = view[:, -1]

        resized = F.interpolate(
            view.float(),
            size=self.image_resolution,
            mode="bicubic",
            align_corners=False,
        )
        primary = (resized - self.primary_mean) / self.primary_std
        secondary = (resized - self.secondary_mean) / self.secondary_std
        return torch.cat([primary, secondary], dim=1)

    def _preprocess_images(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Resize, normalise per tower, and channel-stack all camera views.

        Args:
            batch: Batch dict. ``Observation.to_dict()`` flattens cameras to
                ``images.<name>`` keys; a nested dict or a single stacked
                tensor under ``IMAGES`` are also accepted.

        Returns:
            Batch dict with ``IMAGES`` as ``(B, 6 * num_views, H, W)``.
        """
        flat_keys = [key for key in Observation.get_flattened_keys(batch, IMAGES) if "is_pad" not in key]

        if flat_keys:
            ordered = self._ordered_image_keys(flat_keys)
            views = [batch[key] for key in ordered]
            for key in flat_keys:
                batch.pop(key, None)
        else:
            images = batch.get(IMAGES)
            if images is None:
                return batch
            if isinstance(images, torch.Tensor):
                views = [images] if images.dim() == 4 else list(images.unbind(dim=1))  # noqa: PLR2004
            else:
                lookup = {
                    key if key.startswith(f"{IMAGES}.") else f"{IMAGES}.{key}": value for key, value in images.items()
                }
                views = [lookup[key] for key in self._ordered_image_keys(list(lookup))]

        stacked = [self._stack_view(view) for view in views]

        if self.num_cameras > 0:
            while len(stacked) < self.num_cameras:
                stacked.append(torch.zeros_like(stacked[0]))
            stacked = stacked[: self.num_cameras]

        batch[IMAGES] = torch.cat(stacked, dim=1)
        return batch

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Preprocess a raw batch.

        Args:
            batch: Raw batch dict, typically ``Observation.to_dict()``.

        Returns:
            Batch with channel-stacked images, tokenized prompt and
            quantile-normalized state/action.
        """
        batch = dict(batch)
        task = batch.get(TASK)
        if not task:
            task = [""] * _batch_size(batch)
        tokens, masks = self._tokenize(task)
        device = batch[STATE].device if STATE in batch else tokens.device
        batch[TOKENIZED_PROMPT] = tokens.to(device)
        batch[TOKENIZED_PROMPT_MASK] = masks.to(device)

        batch = self._preprocess_images(batch)

        return self._state_action_normalizer(batch)


class VLAAdapterPostprocessor(torch.nn.Module):
    """Map model outputs back to the original action space."""

    def __init__(self, features: dict[str, Feature] | None = None) -> None:
        """Initialize the postprocessor.

        Args:
            features: Action features drive denormalization; None means
                identity.
        """
        super().__init__()

        if features is not None:
            action_features = {k: v for k, v in features.items() if v.ftype == FeatureType.ACTION}
            self._action_denormalizer: torch.nn.Module = FeatureNormalizeTransform(
                action_features,
                NORM_MAP,
                inverse=True,
            )
        else:
            self._action_denormalizer = torch.nn.Identity()

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Denormalize actions if present.

        Args:
            batch: Batch dict, optionally containing ``ACTION``.

        Returns:
            Batch with denormalized actions.
        """
        batch = dict(batch)
        if ACTION in batch:
            batch[ACTION] = self._action_denormalizer({ACTION: batch[ACTION]})[ACTION]
        return batch


def _batch_size(batch: dict[str, Any]) -> int:
    """Infer the batch size from whichever tensor the batch carries.

    Args:
        batch: Batch dict, before image stacking.

    Returns:
        Leading dimension of the first tensor found, or 1 if there is none.
    """
    for value in batch.values():
        if isinstance(value, torch.Tensor) and value.dim() > 0:
            return int(value.shape[0])
        if isinstance(value, dict):
            for inner in value.values():
                if isinstance(inner, torch.Tensor) and inner.dim() > 0:
                    return int(inner.shape[0])
    return 1


def _quantile_params(stat: dict[str, list[float] | str | tuple]) -> NormalizationParameters:
    """Build quantile normalization parameters from a dataset stat entry.

    LeRobot datasets do not always carry ``q01``/``q99``, so fall back to
    ``min``/``max`` — the same affine mapping over the full range rather than
    the trimmed one.

    Args:
        stat: A single feature's statistics dict.

    Returns:
        Populated normalization parameters.

    Raises:
        KeyError: If neither quantiles nor min/max are available.
    """
    if stat.get("q01") is not None and stat.get("q99") is not None:
        return NormalizationParameters(
            q01=cast("list[float]", stat["q01"]),
            q99=cast("list[float]", stat["q99"]),
        )

    if stat.get("min") is None or stat.get("max") is None:
        msg = (
            f"Feature {stat.get('name')!r} has neither q01/q99 nor min/max statistics; "
            "quantile normalization cannot be configured."
        )
        raise KeyError(msg)

    logger.warning(
        "Dataset stats for %r lack q01/q99; falling back to min/max for quantile normalization.",
        stat.get("name"),
    )
    return NormalizationParameters(
        q01=cast("list[float]", stat["min"]),
        q99=cast("list[float]", stat["max"]),
    )


def make_vla_adapter_preprocessors(
    max_state_dim: int = 8,
    max_action_dim: int = 7,
    stats: dict[str, dict[str, list[float] | str | tuple]] | None = None,
    *,
    image_resolution: tuple[int, int] = (224, 224),
    vision_backbone_ids: tuple[str, str] = (
        "vit_large_patch14_reg4_dinov2.lvd142m",
        "vit_so400m_patch14_siglip_224",
    ),
    image_key_reorder_map: dict[str, int] | None = None,
    num_cameras: int = 0,
    max_token_len: int = 48,
    token_pad_type: str = "max_length",  # noqa: S107
    tokenizer_name: str = "Qwen/Qwen2.5-0.5B",
) -> tuple[VLAAdapterPreprocessor, VLAAdapterPostprocessor]:
    """Create a matched preprocessor / postprocessor pair.

    Args:
        max_state_dim: Proprioceptive state dimension.
        max_action_dim: Action dimension.
        stats: Dataset statistics as nested dicts.
        image_resolution: Target image resolution.
        vision_backbone_ids: timm ids of the two towers.
        image_key_reorder_map: Image-key to camera-slot mapping.
        num_cameras: Total camera slots.
        max_token_len: Fixed language token length.
        token_pad_type: Tokenizer padding strategy.
        tokenizer_name: HuggingFace tokenizer identifier.

    Returns:
        ``(preprocessor, postprocessor)``.
    """
    features: dict[str, Feature] = {}
    if stats is not None:
        for key, stat in stats.items():
            if ACTION in key:
                feature_type = FeatureType.ACTION
            elif STATE in key:
                feature_type = FeatureType.STATE
            else:
                continue
            features[str(stat["name"])] = Feature(
                name=str(stat["name"]),
                ftype=feature_type,
                shape=cast("tuple[int, ...]", stat["shape"]),
                normalization_data=_quantile_params(stat),
            )

    preprocessor = VLAAdapterPreprocessor(
        max_state_dim=max_state_dim,
        max_action_dim=max_action_dim,
        image_resolution=image_resolution,
        vision_backbone_ids=vision_backbone_ids,
        image_key_reorder_map=image_key_reorder_map,
        num_cameras=num_cameras,
        features=features,
        max_token_len=max_token_len,
        tokenizer_name=tokenizer_name,
        padding=token_pad_type,
    )
    postprocessor = VLAAdapterPostprocessor(features=features)
    return preprocessor, postprocessor
