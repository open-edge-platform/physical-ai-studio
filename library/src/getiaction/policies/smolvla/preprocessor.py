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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F

from getiaction.data.observation import TASK

if TYPE_CHECKING:
    from collections.abc import Mapping

    from getiaction.data.observation import Observation

logger = logging.getLogger(__name__)


@dataclass
class NormStats:
    """Normalization statistics for a feature.

    Supports both z-score (mean/std) and quantile (q01/q99) normalization.

    Attributes:
        mean: Mean values for z-score normalization.
        std: Standard deviation for z-score normalization.
        q01: 1st percentile for quantile normalization.
        q99: 99th percentile for quantile normalization.
    """

    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    q01: np.ndarray | None = None
    q99: np.ndarray | None = None


class SmolVLAPreprocessor(torch.nn.Module):
    """Preprocessor for SmolVLA model inputs.

    Transforms observations and actions into the format expected by Pi0Model:
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
        stats: dict[str, NormStats] | None = None,
        max_token_len: int = 48,
        tokenizer_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        padding: str = "longest",
    ) -> None:
        super().__init__()
        self.max_state_dim = max_state_dim
        self.max_action_dim = max_action_dim
        self.image_resolution = image_resolution
        self.stats = stats or {}
        self.max_token_len = max_token_len
        self.tokenizer_name = tokenizer_name
        self.padding = padding
        self._tokenizer = None

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        batch = self._newline_processor(batch)

        tokens, masks = self._tokenize(batch[TASK])

        batch["tokenized_prompt"] = tokens
        batch["tokenized_prompt_mask"] = masks

        print(batch.keys())
        print(batch[TASK])

        exit(0)

        return batch

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
                self._tokenizer = AutoTokenizer.from_pretrained(  # nosec B615
                    self.tokenizer_name,
                    revision="main",
                )
            except ImportError as e:
                msg = "Tokenizer requires transformers. Install with: pip install transformers"
                raise ImportError(msg) from e
        return self._tokenizer

    @staticmethod
    def _resize_with_pad(img, width, height, pad_value=-1):
        # assume no-op when width height fits already
        if img.ndim != 4:
            raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

        cur_height, cur_width = img.shape[2:]

        ratio = max(cur_width / width, cur_height / height)
        resized_height = int(cur_height / ratio)
        resized_width = int(cur_width / ratio)
        resized_img = F.interpolate(
            img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
        )

        pad_height = max(0, int(height - resized_height))
        pad_width = max(0, int(width - resized_width))

        # pad on left and top of image
        padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
        return padded_img

    def _normalize(self, x: torch.Tensor, stats: NormStats) -> torch.Tensor:
        pass


@dataclass
class SmolVLAPostprocessor:
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

    action_dim: int
    max_action_dim: int = 32
    use_quantile_norm: bool = True
    stats: dict[str, NormStats] | None = None

    def __call__(self, outputs: dict[str, Any]) -> dict[str, Any]:
        """Postprocess model outputs.

        Args:
            outputs: Model output dict with "actions" key.

        Returns:
            Postprocessed outputs with denormalized actions.
        """
        result = dict(outputs)

        if "actions" in result:
            actions = result["actions"]

            # Truncate to actual dimension
            actions = actions[..., : self.action_dim]

            # Denormalize
            if self.stats is not None and "actions" in self.stats:
                actions = self._denormalize(actions, self.stats["actions"])

            result["actions"] = actions

        return result

    def _denormalize(self, x: torch.Tensor, stats: NormStats) -> torch.Tensor:
        """Denormalize tensor using stats.

        Args:
            x: Normalized tensor.
            stats: Normalization statistics.

        Returns:
            Denormalized tensor.
        """
        if self.use_quantile_norm and stats.q01 is not None and stats.q99 is not None:
            q01 = torch.tensor(stats.q01, dtype=x.dtype, device=x.device)
            q99 = torch.tensor(stats.q99, dtype=x.dtype, device=x.device)
            dim = x.shape[-1]
            q01 = q01[..., :dim]
            q99 = q99[..., :dim]
            return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
        if stats.mean is not None and stats.std is not None:
            mean = torch.tensor(stats.mean, dtype=x.dtype, device=x.device)
            std = torch.tensor(stats.std, dtype=x.dtype, device=x.device)
            dim = x.shape[-1]
            mean = mean[..., :dim]
            std = std[..., :dim]
            return x * (std + 1e-6) + mean
        return x


def make_smolvla_preprocessors(
    max_state_dim: int = 32,
    max_action_dim: int = 32,
    env_action_dim: int | None = None,
    stats: dict[str, dict[str, list[float]]] | None = None,
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
    # Convert stats format if needed
    norm_stats: dict[str, NormStats] | None = None
    if stats is not None:
        norm_stats = {}
        for key, stat_dict in stats.items():
            norm_stats[key] = NormStats(
                mean=np.array(stat_dict.get("mean")) if "mean" in stat_dict else None,
                std=np.array(stat_dict.get("std")) if "std" in stat_dict else None,
                q01=np.array(stat_dict.get("q01")) if "q01" in stat_dict else None,
                q99=np.array(stat_dict.get("q99")) if "q99" in stat_dict else None,
            )

    preprocessor = SmolVLAPreprocessor(
        max_state_dim=max_state_dim,
        max_action_dim=max_action_dim,
        image_resolution=image_resolution,
        stats=norm_stats,
        max_token_len=max_token_len,
    )

    postprocessor = SmolVLAPostprocessor(
        action_dim=env_action_dim or max_action_dim,
        max_action_dim=max_action_dim,
        stats=norm_stats,
    )

    return preprocessor, postprocessor
