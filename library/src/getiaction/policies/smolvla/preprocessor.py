# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2025 HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0

"""Preprocessor for SmolVLA model.

This module provides preprocessing functionality for transforming observations
and actions into the format expected by SmolVLA model.

Handles:
- Image resizing and normalization
- State/action normalization
- State/action padding to max dimensions
- Language tokenization
- Output denormalization
"""

from __future__ import annotations

import logging
from typing import Any, cast

import torch
import torch.nn.functional as F  # noqa: N812

from getiaction.data import Feature, FeatureType, NormalizationParameters
from getiaction.data.observation import ACTION, STATE, TASK
from getiaction.policies.utils.normalization import FeatureNormalizeTransform, NormalizationType

logger = logging.getLogger(__name__)


NORM_MAP = {
    FeatureType.STATE: NormalizationType.MEAN_STD,
    FeatureType.ACTION: NormalizationType.MEAN_STD,
}


class SmolVLAPreprocessor(torch.nn.Module):
    """Preprocessor for SmolVLA model inputs.

    Transforms observations and actions into the format expected by SmolVLAModel:
    1. Resizes images to target resolution with padding
    2. Normalizes images to [-1, 1]
    3. Normalizes state/action using quantile or z-score normalization
    4. Pads state/action to max dimensions
    5. Tokenizes language prompts

    Args:
        max_state_dim: Maximum state dimension for padding.
        max_action_dim: Maximum action dimension for padding.
        action_horizon: Number of action steps to predict.
        image_resolution: Target image resolution (height, width).
        use_quantile_norm: Whether to use quantile normalization (Pi0.5 default).
        stats: Normalization statistics dict.
        tokenizer_name: HuggingFace tokenizer name.
        max_token_len: Maximum tokenized prompt length.

    Example:
        >>> preprocessor = Pi0Preprocessor(
        ...     max_state_dim=32,
        ...     max_action_dim=32,
        ...     stats=dataset_stats,
        ... )
        >>> batch = preprocessor(raw_batch)
    """

    def __init__(
        self,
        max_state_dim: int = 32,
        max_action_dim: int = 32,
        image_resolution: tuple[int, int] = (512, 512),
        features: dict[str, Feature] | None = None,
        max_token_len: int = 48,
        tokenizer_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        padding: str = "longest",
    ) -> None:
        """Initialize the SmolVLA preprocessor.

        Args:
            max_state_dim: Maximum dimension for state vectors. Defaults to 32.
            max_action_dim: Maximum dimension for action vectors. Defaults to 32.
            image_resolution: Target resolution for input images as (height, width).
                Defaults to (512, 512).
            features: Dictionary mapping feature names to Feature objects for
                normalization. If None, no normalization is applied. Defaults to None.
            max_token_len: Maximum length of tokenized text sequences. Defaults to 48.
            tokenizer_name: HuggingFace tokenizer identifier to use for text
                processing. Defaults to "HuggingFaceTB/SmolVLM2-500M-Video-Instruct".
            padding: Padding strategy for tokenization. Defaults to "longest".
        """
        super().__init__()

        self.max_state_dim = max_state_dim
        self.max_action_dim = max_action_dim
        self.image_resolution = image_resolution
        self.max_token_len = max_token_len
        self.tokenizer_name = tokenizer_name
        self.padding = padding
        self._tokenizer = None

        if features is not None:
            self._state_action_normalizer = FeatureNormalizeTransform(features, NORM_MAP)
        else:
            self._state_action_normalizer = torch.nn.Identity()

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Process a batch by applying newline processing, tokenization, and normalization.

        Args:
            batch: A dictionary containing input data with keys including TASK and STATE.
                TASK is used for tokenization and STATE determines the target device.

        Returns:
            A dictionary containing the processed batch with added 'tokenized_prompt'
            and 'tokenized_prompt_mask' tensors, after applying state-action normalization.
        """
        batch = self._newline_processor(batch)
        tokens, masks = self._tokenize(batch[TASK])
        batch["tokenized_prompt"] = tokens.to(batch[STATE].device)
        batch["tokenized_prompt_mask"] = masks.to(batch[STATE].device)

        return self._state_action_normalizer(batch)

    @staticmethod
    def _newline_processor(batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Ensure task descriptions end with newline character.

        Args:
            batch: Input batch dict containing 'extra' with 'task'.

        Returns:
            Updated batch with newline-terminated 'task'.
        """
        if TASK not in batch:
            return batch

        task = batch[TASK]
        if task is None:
            batch[TASK] = "\n"
            return batch

        new_batch = dict(batch)
        # Handle both string and list of strings
        if isinstance(task, str):
            # Single string: add newline if not present
            if not task.endswith("\n"):
                new_batch[TASK] = f"{task}\n"
        elif isinstance(task, list) and all(isinstance(t, str) for t in task):
            # List of strings: add newline to each if not present
            new_batch[TASK] = [t if t.endswith("\n") else f"{t}\n" for t in task]
        # If task is neither string nor list of strings, leave unchanged

        return new_batch

    def _tokenize(self, text: str | list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize text prompts.

        Args:
            text: Text string or list of strings.

        Returns:
            Tuple of (token_ids, attention_mask).
        """
        if isinstance(text, str):
            text = [text]

        encoded = self.tokenizer(
            text,
            max_length=self.max_token_len,
            truncation=True,
            padding="longest",
            padding_side="right",
            return_tensors="pt",
        )

        return encoded["input_ids"], encoded["attention_mask"].bool()

    @property
    def tokenizer(self) -> Any:  # noqa: ANN401
        """Lazy-load tokenizer.

        Raises:
            ImportError: If transformers library is not installed.
        """
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer  # noqa: PLC0415

                # Revision pinned for reproducibility and security
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.tokenizer_name,
                    revision="e977446e632670f9972fc4ff1432b414c8b813cb",
                )
            except ImportError as e:
                msg = "Tokenizer requires transformers. Install with: pip install transformers"
                raise ImportError(msg) from e
        return self._tokenizer

    @staticmethod
    def _resize_with_pad(img: torch.Tensor, width: int, height: int, pad_value: int = -1) -> torch.Tensor:
        # assume no-op when width height fits already
        img_dim = 4
        if img.ndim != img_dim:
            msg = f"(b,c,h,w) expected, but {img.shape}"
            raise ValueError(msg)

        cur_height, cur_width = img.shape[2:]

        ratio = max(cur_width / width, cur_height / height)
        resized_height = int(cur_height / ratio)
        resized_width = int(cur_width / ratio)
        resized_img = F.interpolate(
            img,
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
        )

        pad_height = max(0, int(height - resized_height))
        pad_width = max(0, int(width - resized_width))

        # pad on left and top of image
        return F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)


class SmolVLAPostprocessor(torch.nn.Module):
    """Postprocessor for SmolVLA model outputs.

    Transforms model outputs back to the original action space:
    1. Truncates to actual action dimension
    2. Denormalizes using dataset statistics

    Args:
        action_dim: Actual action dimension (before padding).
        max_action_dim: Padded action dimension.
        use_quantile_norm: Whether quantile normalization was used.
        stats: Normalization statistics dict.
    """

    def __init__(
        self,
        features: dict[str, Feature] | None = None,
    ) -> None:
        """Initialize the preprocessor.

        Args:
            features: A dictionary mapping feature names to Feature objects.
                If provided, action features will be extracted and used to create
                a denormalizer transform. If None, an identity transform is used
                for action denormalization.
        """
        super().__init__()

        if features is not None:
            action_features = {k: v for k, v in features.items() if v.ftype == FeatureType.ACTION}
            self._action_denormalizer = FeatureNormalizeTransform(action_features, NORM_MAP, inverse=True)
        else:
            self._action_denormalizer = torch.nn.Identity()

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Process a batch by denormalizing actions if present.

        Args:
            batch: A dictionary containing batch data. May optionally contain
                an ACTION key with action values to be denormalized.

        Returns:
            A dictionary with the same structure as the input batch, but with
            action values denormalized if they were present.
        """
        batch = dict(batch)
        if ACTION in batch:
            batch[ACTION] = self._action_denormalizer({ACTION: batch[ACTION]})[ACTION]
        return batch


def make_smolvla_preprocessors(
    max_state_dim: int = 32,
    max_action_dim: int = 32,
    stats: dict[str, dict[str, list[float] | str | tuple]] | None = None,
    *,
    image_resolution: tuple[int, int] = (512, 512),
    max_token_len: int = 48,
) -> tuple[SmolVLAPreprocessor, SmolVLAPostprocessor]:
    """Create preprocessor and postprocessor pair.

    Args:
        max_state_dim: Maximum state dimension.
        max_action_dim: Maximum action dimension.
        env_action_dim: Actual environment action dimension.
        stats: Dataset statistics as nested dicts.
        image_resolution: Target image resolution.
        max_token_len: Maximum token length.

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
            features[str(stat["name"])] = Feature(
                name=str(stat["name"]),
                ftype=feature_type,
                shape=cast("tuple[int, ...]", stat["shape"]),
                normalization_data=NormalizationParameters(
                    mean=cast("list[float]", stat["mean"]),
                    std=cast("list[float]", stat["std"]),
                ),
            )

    preprocessor = SmolVLAPreprocessor(
        max_state_dim=max_state_dim,
        max_action_dim=max_action_dim,
        image_resolution=image_resolution,
        features=features,
        max_token_len=max_token_len,
    )

    postprocessor = SmolVLAPostprocessor(
        features=features,
    )

    return preprocessor, postprocessor
