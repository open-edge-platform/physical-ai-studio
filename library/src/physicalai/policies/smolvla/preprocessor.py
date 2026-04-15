# Copyright (C) 2026 Intel Corporation
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
- Camera name remapping and validation
- Output denormalization

Camera naming convention (lerobot/smolvla_base pretrained model):

    The pretrained SmolVLA base model expects cameras with standardized names.
    Each position maps to a specific physical camera view:

    ========  ==================  ==========================================
    Key       Physical view       Notes
    ========  ==================  ==========================================
    camera1   Top-down view       Required. Primary observation camera.
    camera2   Wrist-mounted view  Required. Gripper/end-effector camera.
    camera3   Side/additional     Optional. Extra viewpoint (e.g. side cam).
    ========  ==================  ==========================================

    Reference: https://huggingface.co/blog/smolvla#standardizing-camera-views
    Reference: SmolVLA paper Section 3.2 (arxiv 2506.01844)

    If your dataset uses different camera names (e.g. "top", "wrist"), use
    ``rename_map`` to remap them::

        SmolVLA(rename_map=(("top", "camera1"), ("wrist", "camera2")))

    If your dataset has fewer cameras than the pretrained model expects, use
    ``empty_cameras`` to pad with placeholder images the model will ignore::

        SmolVLA(rename_map=..., empty_cameras=1)  # 2 real cameras + 1 empty
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, cast

import torch
import torch.nn.functional as F  # noqa: N812

from physicalai.data import Feature, FeatureType, NormalizationParameters
from physicalai.data.constants import IMAGE_MASKS, TOKENIZED_PROMPT, TOKENIZED_PROMPT_MASK
from physicalai.data.observation import ACTION, EXTRA, IMAGES, STATE, TASK, Observation
from physicalai.policies.utils.normalization import FeatureNormalizeTransform, NormalizationType

logger = logging.getLogger(__name__)
if not logger.handlers and not logging.getLogger().handlers:
    logger.addHandler(logging.StreamHandler())
    logger.propagate = False


NORM_MAP = {
    FeatureType.STATE: NormalizationType.MEAN_STD,
    FeatureType.ACTION: NormalizationType.MEAN_STD,
}

SMOLVLA_EXPECTED_CAMERA_NAMES: set[str] = {"camera1", "camera2", "camera3"}


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
        padding: str = "max_length",
        rename_map: dict[str, str] | None = None,
        expected_camera_names: set[str] | None = SMOLVLA_EXPECTED_CAMERA_NAMES,
        empty_cameras: int = 0,
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
            rename_map: Optional mapping of camera base names to target names.
                Maps source camera names to the pretrained model's expected camera slots.
                Example: {"top": "camera1", "wrist": "camera2"} ensures images from
                the "top" camera are placed in the camera1 tensor slot position.
                Keys not in the map pass through in their original order after mapped keys.
                Defaults to None (no reordering).
            expected_camera_names: Camera names the pretrained model expects. Used for
                validation warnings when batch camera names don't match. Defaults to
                SMOLVLA_EXPECTED_CAMERA_NAMES (camera1, camera2, camera3).
            empty_cameras: Number of empty camera slots to add as -1-filled placeholder
                images with zero masks. Used when the pretrained model expects more cameras
                than the dataset provides. Defaults to 0.
        """
        super().__init__()

        self.max_state_dim = max_state_dim
        self.max_action_dim = max_action_dim
        self.image_resolution = image_resolution
        self.max_token_len = max_token_len
        self.tokenizer_name = tokenizer_name
        self.padding = padding
        self.rename_map = rename_map
        self._expected_camera_names = expected_camera_names
        self.empty_cameras = empty_cameras
        self._tokenizer = None
        self._camera_names_validated = False

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
        self._validate_camera_names(batch)

        batch = self._newline_processor(batch)

        tokens, masks = self._tokenize(batch[TASK])
        batch[TOKENIZED_PROMPT] = tokens.to(batch[STATE].device)
        batch[TOKENIZED_PROMPT_MASK] = masks.to(batch[STATE].device)

        images, img_masks = self._preprocess_images(batch)

        # Append empty cameras as -1-filled placeholder images with zero masks.
        # This allows training/inference with fewer cameras than the pretrained
        # model expects (e.g. 2 cameras when the model was trained with 3).
        if self.empty_cameras > 0 and len(images) > 0:
            for _ in range(self.empty_cameras):
                images.append(torch.ones_like(images[-1]) * -1)
                img_masks.append(torch.zeros_like(img_masks[-1]))

        if images:
            batch[IMAGES] = torch.stack(images, dim=0)
            batch[IMAGE_MASKS] = torch.stack(img_masks, dim=0)
        else:
            batch[IMAGES] = torch.empty(0, device=batch[STATE].device)
            batch[IMAGE_MASKS] = torch.empty(0, device=batch[STATE].device)

        return self._state_action_normalizer(batch)

    def _reorder_image_keys(self, batch_img_keys: list[str]) -> list[str]:
        """Reorder image keys to match canonical camera slot order via rename_map.

        Maps each raw batch key to its logical target name using rename_map,
        then sorts by target name to ensure deterministic camera-to-tensor-slot
        assignment matching the pretrained model's expectations.

        Keys not in rename_map retain their original base name for sorting,
        placing them after mapped keys in alphabetical order.

        Args:
            batch_img_keys: List of flattened image keys (e.g. ["images.top", "images.wrist"]).

        Returns:
            Reordered list of image keys sorted by their mapped target names.
        """
        if not self.rename_map:
            return batch_img_keys

        def _sort_key(key: str) -> str:
            base_name = key.rsplit(".", 1)[-1]
            return self.rename_map.get(base_name, base_name)

        return sorted(batch_img_keys, key=_sort_key)

    def _preprocess_images(self, batch: dict[str, torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Apply SmolVLA preprocessing to the images.

        This method processes image tensors from a batch by:
        1. Reordering image keys to match canonical camera slot order (via rename_map)
        2. Extracting the last frame if the input is a 5D tensor (video sequence)
        3. Optionally resizing images with padding to maintain aspect ratio
        4. Converting pixel values from [0.0, 1.0] range to [-1.0, 1.0] range as required by SigLIP
        5. Extracting or creating padding masks for each image

        Args:
            batch: A dictionary containing image tensors and optional padding masks.
                Image tensors should be 4D (B, C, H, W) or 5D (B, T, C, H, W).
                Optional padding masks are stored with keys prefixed by EXTRA.

        Returns:
            A tuple containing:
                - images: List of preprocessed image tensors, each with shape (B, C, H, W)
                    and pixel values in range [-1.0, 1.0]
                - img_masks: List of boolean mask tensors indicating valid (non-padded)
                    images in each batch position
        """
        images: list[torch.Tensor] = []
        img_masks: list[torch.Tensor] = []

        batch_img_keys = Observation.get_flattened_keys(batch, IMAGES)
        batch_img_keys = [key for key in batch_img_keys if "is_pad" not in key]
        batch_img_keys = self._reorder_image_keys(batch_img_keys)

        max_image_dim = 5
        for key in batch_img_keys:
            img = batch[key][:, -1, :, :, :] if batch[key].ndim == max_image_dim else batch[key]
            batch.pop(key)
            if self.image_resolution is not None:
                img = self._resize_with_pad(img, *self.image_resolution, pad_value=0)

            img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device
            if EXTRA + f".{key}_padding_mask" in batch:
                mask = batch[EXTRA + f".{key}_padding_mask"].bool()
                batch.pop(EXTRA + f".{key}_padding_mask")
            else:
                mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        return images, img_masks

    def _validate_camera_names(self, batch: dict[str, Any]) -> None:
        """Check if batch camera names match expected normalization feature names.

        When rename_map is active, validates the mapped logical names (not raw
        batch names) against expected camera names. Only runs once per
        preprocessor lifetime to avoid log spam.
        """
        if self._camera_names_validated:
            return
        self._camera_names_validated = True

        if not self._expected_camera_names:
            return

        image_keys = Observation.get_flattened_keys(batch, IMAGES)
        image_keys = [k for k in image_keys if "is_pad" not in k]
        if not image_keys:
            return

        batch_camera_bases = {k.rsplit(".", 1)[-1] for k in image_keys}

        # Map through rename_map to get logical names for validation
        if self.rename_map:
            mapped_bases = {self.rename_map.get(base, base) for base in batch_camera_bases}
        else:
            mapped_bases = batch_camera_bases

        if not mapped_bases & self._expected_camera_names:
            logger.warning(
                "Camera name mismatch detected. Dataset cameras: %s, "
                "but pretrained model expects: %s. "
                "Consider using rename_map to map your camera names to the expected names. "
                "Example: SmolVLA(rename_map=(('your_camera', 'expected_camera'),))",
                sorted(batch_camera_bases),
                sorted(self._expected_camera_names),
            )

        expected_count = len(self._expected_camera_names)
        actual_count = len(batch_camera_bases) + self.empty_cameras
        if actual_count < expected_count:
            logger.warning(
                "Camera count mismatch: dataset provides %d camera(s) + %d empty = %d total, "
                "but pretrained model expects %d. "
                "Consider setting empty_cameras=%d to pad missing slots. "
                "Example: SmolVLA(empty_cameras=%d)",
                len(batch_camera_bases),
                self.empty_cameras,
                actual_count,
                expected_count,
                expected_count - len(batch_camera_bases),
                expected_count - len(batch_camera_bases),
            )

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
            padding=self.padding,
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
                    revision="7b375e1b73b11138ff12fe22c8f2822d8fe03467",
                    use_fast=True,
                )
            except ImportError as e:
                msg = "Tokenizer requires transformers. Install with: pip install transformers"
                raise ImportError(msg) from e
        return self._tokenizer

    @property
    def exportable_tokenizer(self) -> Any:  # noqa: ANN401
        """Get tokenizer for export.

        This method is used during model export to retrieve the tokenizer for
        conversion to ONNX or OpenVINO format. It simply returns the same
        tokenizer instance used during preprocessing.

        Returns:
            The tokenizer instance used by this preprocessor.
        """
        return self.tokenizer

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


def rename_stats(
    stats: dict[str, dict[str, Any]],
    rename_map: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """Rename top-level keys in a dataset statistics dict to match renamed camera names.

    This aligns normalization statistics with renamed observation keys so that
    FeatureNormalizeTransform can match stats buffers to the correct batch keys.

    The rename_map uses the same short key format as SmolVLAPreprocessor: base camera
    names only (e.g. {"top": "camera1"}). Stats keys like "observation.images.top"
    will be matched by checking if the key ends with any source name in the map.

    Keys not matching any entry in rename_map pass through unchanged.
    A defensive deep copy is performed to avoid mutating the original stats.

    Args:
        stats: Dataset statistics dict with top-level keys like "observation.images.top",
            "observation.state", "action".
        rename_map: Mapping of source camera base names to target names.

    Returns:
        New statistics dict with renamed top-level keys.
    """
    if not stats:
        return {}
    renamed: dict[str, dict[str, Any]] = {}
    for old_key, sub_stats in stats.items():
        new_key = old_key
        for src_name, dst_name in rename_map.items():
            suffix = f".{src_name}"
            if old_key.endswith(suffix):
                new_key = old_key[: -len(src_name)] + dst_name
                break
        renamed[new_key] = deepcopy(sub_stats) if sub_stats is not None else {}
        if new_key != old_key and "name" in renamed[new_key]:
            renamed[new_key]["name"] = new_key
    return renamed


def make_smolvla_preprocessors(
    max_state_dim: int = 32,
    max_action_dim: int = 32,
    stats: dict[str, dict[str, list[float] | str | tuple]] | None = None,
    *,
    image_resolution: tuple[int, int] = (512, 512),
    max_token_len: int = 48,
    token_pad_type: str = "longest",  # noqa: S107
    rename_map: dict[str, str] | None = None,
    empty_cameras: int = 0,
    tokenizer_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
) -> tuple[SmolVLAPreprocessor, SmolVLAPostprocessor]:
    """Create preprocessor and postprocessor pair.

    Args:
        max_state_dim: Maximum state dimension.
        max_action_dim: Maximum action dimension.
        env_action_dim: Actual environment action dimension.
        stats: Dataset statistics as nested dicts.
        image_resolution: Target image resolution.
        max_token_len: Maximum token length.
        token_pad_type: Padding strategy for tokenization ("longest" or "max_length").
        rename_map: Optional mapping of camera base names to target names.
        empty_cameras: Number of empty camera slots to add.
        tokenizer_name: HuggingFace tokenizer name.

    Returns:
        Tuple of (preprocessor, postprocessor).
    """
    if rename_map and stats is not None:
        stats = rename_stats(stats, rename_map)

    expected_camera_names: set[str] | None = SMOLVLA_EXPECTED_CAMERA_NAMES
    features: dict[str, Feature] = {}
    if stats is not None:
        image_stat_names: set[str] = set()
        for key, stat in stats.items():
            if ACTION in key:
                feature_type = FeatureType.ACTION
            elif STATE in key:
                feature_type = FeatureType.STATE
            else:
                stat_name = str(stat.get("name", key))
                base = stat_name.rsplit(".", maxsplit=1)[-1] if "." in stat_name else stat_name
                if str(stat.get("type", "")).upper() == FeatureType.VISUAL.value:
                    image_stat_names.add(base)
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

        if rename_map and image_stat_names:
            expected_camera_names = image_stat_names

    preprocessor = SmolVLAPreprocessor(
        max_state_dim=max_state_dim,
        max_action_dim=max_action_dim,
        image_resolution=image_resolution,
        features=features,
        max_token_len=max_token_len,
        padding=token_pad_type,
        rename_map=rename_map,
        expected_camera_names=expected_camera_names,
        empty_cameras=empty_cameras,
        tokenizer_name=tokenizer_name,
    )

    postprocessor = SmolVLAPostprocessor(
        features=features,
    )

    return preprocessor, postprocessor
