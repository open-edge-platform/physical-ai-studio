# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Preprocessor for Pi05 model.

Handles:
- State normalization and discretization into language tokens
- Image resizing and normalization
- Action normalization and padding
- Language tokenization with PaliGemma tokenizer
- Output denormalization
"""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from physicalai.data import Feature, FeatureType, NormalizationParameters
from physicalai.data.observation import ACTION, IMAGES, STATE, TASK, Observation
from physicalai.policies.utils.normalization import FeatureNormalizeTransform, NormalizationType

logger = logging.getLogger(__name__)

NORM_MAP = {
    FeatureType.STATE: NormalizationType.MEAN_STD,
    FeatureType.ACTION: NormalizationType.MEAN_STD,
}


def _pad_vector(vector: torch.Tensor, new_dim: int) -> torch.Tensor:
    """Pad the last dimension of a vector to new_dim with zeros.

    Returns:
        Padded vector tensor.
    """
    if vector.shape[-1] >= new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))


def _resize_with_pad_torch(  # noqa: PLR0914
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Resize images with padding to target dimensions without distortion.

    Args:
        images: Tensor of shape [*b, h, w, c] or [*b, c, h, w].
        height: Target height.
        width: Target width.
        mode: Interpolation mode.

    Returns:
        Resized and padded tensor.

    Raises:
        ValueError: If image dtype is unsupported.
    """
    channels_last = images.shape[-1] <= 4  # noqa: PLR2004
    if channels_last:
        if images.dim() == 3:  # noqa: PLR2004
            images = images.unsqueeze(0)
        images = images.permute(0, 3, 1, 2)
    elif images.dim() == 3:  # noqa: PLR2004
        images = images.unsqueeze(0)

    _, _, cur_height, cur_width = images.shape

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    resized_images = F.interpolate(
        images,
        size=(resized_height, resized_width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )

    if images.dtype == torch.uint8:
        resized_images = torch.round(resized_images).clamp(0, 255).to(torch.uint8)
    elif images.dtype == torch.float32:
        resized_images = resized_images.clamp(0.0, 1.0)
    else:
        msg = f"Unsupported image dtype: {images.dtype}"
        raise ValueError(msg)

    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w

    constant_value = 0 if images.dtype == torch.uint8 else 0.0
    padded_images = F.pad(
        resized_images,
        (pad_w0, pad_w1, pad_h0, pad_h1),
        mode="constant",
        value=constant_value,
    )

    if channels_last:
        padded_images = padded_images.permute(0, 2, 3, 1)

    return padded_images


class Pi05Preprocessor(torch.nn.Module):
    """Preprocessor for Pi05 model inputs.

    Transforms observations and actions into the format expected by Pi05Model:
    1. Normalizes state/action using mean-std normalization
    2. Discretizes state into 256 bins and embeds in text prompt
    3. Tokenizes text prompt with PaliGemma tokenizer
    4. Resizes images and normalizes to [-1, 1]
    5. Pads actions to max dimensions

    Args:
        max_state_dim: Maximum state dimension for padding.
        max_action_dim: Maximum action dimension for padding.
        image_resolution: Target image resolution (height, width).
        features: Dictionary mapping feature names to Feature objects for normalization.
        max_token_len: Maximum tokenized prompt length.
        tokenizer_name: HuggingFace tokenizer name for PaliGemma.
        empty_cameras: Number of empty camera slots to add as -1-filled images.
    """

    def __init__(
        self,
        max_state_dim: int = 32,
        max_action_dim: int = 32,
        image_resolution: tuple[int, int] = (224, 224),
        features: dict[str, Feature] | None = None,
        max_token_len: int = 200,
        tokenizer_name: str = "google/paligemma-3b-pt-224",
        empty_cameras: int = 0,
    ) -> None:
        """Initialize Pi05Preprocessor."""
        super().__init__()

        self.max_state_dim = max_state_dim
        self.max_action_dim = max_action_dim
        self.image_resolution = image_resolution
        self.max_token_len = max_token_len
        self.tokenizer_name = tokenizer_name
        self._tokenizer = None
        self.empty_cameras = empty_cameras

        if features is not None:
            self._state_action_normalizer = FeatureNormalizeTransform(features, NORM_MAP)
        else:
            self._state_action_normalizer = torch.nn.Identity()

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Process a batch for Pi05 model input.

        Args:
            batch: Dictionary containing STATE, TASK (text), image keys, and optionally ACTION.

        Returns:
            Dictionary with tokenized_prompt, tokenized_prompt_mask, image tensor lists,
            image_masks, and optionally padded/normalized ACTION.
        """
        # Normalize state/action
        batch = self._state_action_normalizer(batch)

        # Discretize state and build prompt with embedded state
        state = batch[STATE]
        state_dim = 2
        if state.ndim > state_dim:
            state = state[:, -1, :]

        # Discretize normalized state into 256 bins
        # NOTE: Do NOT pad state before discretization. Lerobot uses the raw
        # state dimensions (e.g. 8 for LIBERO) in the prompt, not max_state_dim.
        state_np = state.cpu().numpy()
        discretized_states = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        # Build full prompts with task + discretized state
        task = batch.get(TASK)
        if task is None:
            task = [""] * state.shape[0]
        elif isinstance(task, str):
            task = [task]

        full_prompts = []
        for i, t in enumerate(task):
            cleaned_text = t.strip().replace("_", " ").replace("\n", " ")
            state_str = " ".join(map(str, discretized_states[i]))
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            full_prompts.append(full_prompt)

        # Tokenize
        tokens, masks = self._tokenize(full_prompts)
        batch["tokenized_prompt"] = tokens.to(state.device)
        batch["tokenized_prompt_mask"] = masks.to(state.device)

        # Preprocess images
        images, img_masks = self._preprocess_images(batch)

        # Append empty cameras as -1-filled images with zero masks
        if self.empty_cameras > 0 and len(images) > 0:
            for _ in range(self.empty_cameras):
                images.append(torch.ones_like(images[-1]) * -1)
                img_masks.append(torch.zeros_like(img_masks[-1]))

        if images:
            images = torch.stack(images, dim=0)
            img_masks = torch.stack(img_masks, dim=0)
        else:
            images = torch.empty(0, device=batch[STATE].device)
            img_masks = torch.empty(0, device=batch[STATE].device)

        batch[IMAGES] = images
        batch["image_masks"] = img_masks

        # Pad actions if present
        if ACTION in batch and batch[ACTION] is not None:
            batch[ACTION] = _pad_vector(batch[ACTION], self.max_action_dim)

        return batch

    def _preprocess_images(self, batch: dict[str, Any]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Process images for Pi05 model.

        Pi05 uses PaliGemma which expects images in [B, C, H, W] format
        normalized to [-1, 1].

        Returns:
            Tuple of (list of image tensors, list of mask tensors).
        """
        images = []
        img_masks = []

        batch_img_keys = Observation.get_flattened_keys(batch, IMAGES)
        batch_img_keys = [key for key in batch_img_keys if "is_pad" not in key]

        device = batch[STATE].device if STATE in batch else torch.device("cpu")

        max_image_dim = 5
        for key in batch_img_keys:
            img = batch[key][:, -1, :, :, :] if batch[key].ndim == max_image_dim else batch[key]

            if img.dtype != torch.float32:
                img = img.to(torch.float32)

            # Check format: [B, C, H, W] vs [B, H, W, C]
            is_channels_first = img.shape[1] == 3  # noqa: PLR2004

            if is_channels_first:
                img = img.permute(0, 2, 3, 1)  # -> [B, H, W, C]

            # Resize with padding
            if img.shape[1:3] != tuple(self.image_resolution):
                img = _resize_with_pad_torch(img, *self.image_resolution)

            # Normalize [0,1] -> [-1,1]
            img = img * 2.0 - 1.0

            if is_channels_first:
                img = img.permute(0, 3, 1, 2)  # -> [B, C, H, W]

            bsize = img.shape[0]
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        return images, img_masks

    def _tokenize(self, text: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize text prompts with PaliGemma tokenizer.

        Args:
            text: List of text strings.

        Returns:
            Tuple of (token_ids, attention_mask).
        """
        encoded = self.tokenizer(
            text,
            max_length=self.max_token_len,
            truncation=True,
            padding="max_length",
            padding_side="right",
            return_tensors="pt",
        )

        return encoded["input_ids"], encoded["attention_mask"].bool()

    @property
    def tokenizer(self) -> Any:  # noqa: ANN401
        """Lazy-load PaliGemma tokenizer.

        Raises:
            ImportError: If transformers is not installed.
        """
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer  # noqa: PLC0415

                self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)  # nosec B615
            except ImportError as e:
                msg = "Tokenizer requires transformers. Install with: uv pip install transformers"
                raise ImportError(msg) from e
        return self._tokenizer


class Pi05Postprocessor(torch.nn.Module):
    """Postprocessor for Pi05 model outputs.

    Denormalizes predicted actions back to the original action space.

    Args:
        features: Dictionary mapping feature names to Feature objects.
    """

    def __init__(
        self,
        features: dict[str, Feature] | None = None,
    ) -> None:
        """Initialize Pi05Postprocessor."""
        super().__init__()

        if features is not None:
            action_features = {k: v for k, v in features.items() if v.ftype == FeatureType.ACTION}
            self._action_denormalizer = FeatureNormalizeTransform(action_features, NORM_MAP, inverse=True)
        else:
            self._action_denormalizer = torch.nn.Identity()

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Denormalize actions.

        Returns:
            Batch dict with denormalized actions.
        """
        batch = dict(batch)
        if ACTION in batch:
            batch[ACTION] = self._action_denormalizer({ACTION: batch[ACTION]})[ACTION]
        return batch


def make_pi05_preprocessors(
    max_state_dim: int = 32,
    max_action_dim: int = 32,
    stats: dict[str, dict[str, list[float] | str | tuple]] | None = None,
    *,
    image_resolution: tuple[int, int] = (224, 224),
    max_token_len: int = 200,
    empty_cameras: int = 0,
) -> tuple[Pi05Preprocessor, Pi05Postprocessor]:
    """Create preprocessor and postprocessor pair for Pi05.

    Args:
        max_state_dim: Maximum state dimension.
        max_action_dim: Maximum action dimension.
        stats: Dataset statistics as nested dicts.
        image_resolution: Target image resolution.
        max_token_len: Maximum token length.
        empty_cameras: Number of empty camera slots to add.

    Returns:
        Tuple of (preprocessor, postprocessor).
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

            # Map HF feature names (e.g. "observation.state") to Observation
            # field names (e.g. "state") so the normalizer can match batch keys.
            raw_name = str(stat["name"])
            mapped_name = raw_name.rsplit("observation.", maxsplit=1)[-1] if "observation." in raw_name else raw_name

            features[mapped_name] = Feature(
                name=mapped_name,
                ftype=feature_type,
                shape=cast("tuple[int, ...]", stat["shape"]),
                normalization_data=NormalizationParameters(
                    mean=cast("list[float]", stat["mean"]),
                    std=cast("list[float]", stat["std"]),
                ),
            )

    preprocessor = Pi05Preprocessor(
        max_state_dim=max_state_dim,
        max_action_dim=max_action_dim,
        image_resolution=image_resolution,
        features=features,
        max_token_len=max_token_len,
        empty_cameras=empty_cameras,
    )

    postprocessor = Pi05Postprocessor(
        features=features,
    )

    return preprocessor, postprocessor
