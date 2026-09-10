# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MolmoAct2 Hugging Face checkpoint loading and conversion."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

from huggingface_hub import snapshot_download

from physicalai.data.observation import Feature, FeatureType, NormalizationParameters
from physicalai.policies.utils import JointFrameTransform
from physicalai.policies.utils.features import get_feature_by_type

from .config import MolmoAct2Config
from .constants import (
    SO101_DEGREES_PER_NORMALIZED_UNIT,
    SO101_JOINT_OFFSETS,
    SO101_JOINT_SIGNS,
)
from .pretrained_utils import (
    ACTION_EXPERT_CONFIG_MAP,
    ADAPTER_CONFIG_MAP,
    TEXT_CONFIG_MAP,
    TOP_LEVEL_CONFIG_MAP,
    VISION_CONFIG_MAP,
    copy_component,
)


def _pretrained_normalization_to_so101_runtime(
    features: list[Feature],
    feature_type: FeatureType,
) -> list[Feature]:
    feature = get_feature_by_type(features, feature_type)
    if feature is None or feature.normalization_data is None:
        return list(features)
    if not feature.shape:
        msg = f"Cannot convert pretrained {feature_type.value} normalization without a concrete feature shape."
        raise ValueError(msg)
    normalization = JointFrameTransform(
        signs=SO101_JOINT_SIGNS,
        offsets=SO101_JOINT_OFFSETS,
    ).forward_normalization_from_scaled_input(
        feature.normalization_data,
        dimension=feature.shape[-1],
        scales=SO101_DEGREES_PER_NORMALIZED_UNIT,
    )
    return [
        replace(candidate, normalization_data=normalization) if candidate is feature else candidate
        for candidate in features
    ]


class MolmoAct2FromHFMixin:
    """Load and convert MolmoAct2 Hugging Face checkpoint metadata."""

    input_features: list[Feature] | None
    output_features: list[Feature] | None
    norm_tag: str | None
    chunk_size: int
    n_action_steps: int
    n_obs_steps: int
    setup_type: str | None
    control_mode: str | None
    adapt_to_so101: bool
    convert_pretrained_so101_stats: bool
    use_random_input_noise: bool
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    lora_bias: Literal["all", "lora_only", "none"]

    @classmethod
    def _normalization_parameters(
        cls,
        stats: dict[str, Any],
        feature_key: str,
        *,
        normalize_gripper: bool,
    ) -> NormalizationParameters:
        """Build normalization metadata from saved statistics for a feature.

        Returns:
            Normalization parameters populated from the saved statistics.
        """
        feature_size = cls._feature_size(stats, feature_key)
        mask = cls._normalization_mask(
            stats,
            feature_key,
            feature_size=feature_size,
            normalize_gripper=normalize_gripper,
        )
        cls._validate_passthrough_bounds(stats, mask, feature_key)

        return NormalizationParameters(
            mean=stats.get("mean"),
            std=stats.get("std"),
            min=stats.get("min"),
            max=stats.get("max"),
            q01=stats.get("q01"),
            q99=stats.get("q99"),
            mask=mask,
        )

    @staticmethod
    def _normalization_mask(
        stats: dict[str, Any],
        feature_key: str,
        *,
        feature_size: int,
        normalize_gripper: bool,
    ) -> list[bool] | None:
        """Resolve the per-dimension normalization mask for a feature.

        Returns:
            ``None`` when every dimension should be normalized, otherwise the explicit mask.

        Raises:
            TypeError: If an explicit mask is missing or malformed.
            ValueError: If an explicit mask has the wrong size.
        """
        if normalize_gripper:
            return None

        mask = stats.get("mask")
        if not isinstance(mask, list) or not all(isinstance(value, bool) for value in mask):
            msg = f"MolmoAct2 normalization stats for {feature_key!r} require a boolean mask."
            raise TypeError(msg)
        if len(mask) != feature_size:
            msg = f"MolmoAct2 normalization mask for {feature_key!r} has {len(mask)} values; expected {feature_size}."
            raise ValueError(msg)
        return mask

    @staticmethod
    def _validate_passthrough_bounds(
        stats: dict[str, Any],
        mask: list[bool] | None,
        feature_key: str,
    ) -> None:
        """Validate that dimensions excluded from normalization use unit range.

        Raises:
            TypeError: If pass-through bounds are unavailable.
            ValueError: If pass-through bounds are outside [-1, 1].
        """
        if mask is None or all(mask):
            return

        min_values = stats.get("min")
        max_values = stats.get("max")
        if not isinstance(min_values, list) or not isinstance(max_values, list):
            msg = f"MolmoAct2 pass-through dimensions for {feature_key!r} require min/max statistics."
            raise TypeError(msg)

        passthrough_bounds = [
            (minimum, maximum)
            for minimum, maximum, should_normalize in zip(
                min_values,
                max_values,
                mask,
                strict=True,
            )
            if not should_normalize
        ]
        if any(minimum < -1.0 or maximum > 1.0 for minimum, maximum in passthrough_bounds):
            msg = (
                f"MolmoAct2 {feature_key} pass-through values are not under [-1, 1]. Please set normalize_gripper=True."
            )
            raise ValueError(msg)

    @staticmethod
    def _feature_size(stats: dict[str, Any], feature_key: str) -> int:
        """Infer a feature vector size from its saved normalization statistics.

        Returns:
            The length of the feature's vector-valued statistics.

        Raises:
            ValueError: If the statistics contain no vector-valued arrays.
        """
        for stat_name in ("mean", "std", "min", "max", "q01", "q99"):
            value = stats.get(stat_name)
            if isinstance(value, list):
                return len(value)
        msg = f"MolmoAct2 normalization stats for {feature_key!r} contain no vector values."
        raise ValueError(msg)

    def _resolve_norm_tag(self, norm_stats_config: dict[str, Any]) -> dict[str, Any]:
        """Return metadata for the selected normalization tag.

        Returns:
            The metadata associated with the configured normalization tag.

        Raises:
            ValueError: If no matching normalization tag is configured.
            TypeError: If the normalization metadata is missing or malformed.
        """
        if self.norm_tag is None:
            msg = "Normalization tag is required when loading pretrained MolmoAct2 data."
            raise ValueError(msg)
        metadata_by_tag = norm_stats_config.get("metadata_by_tag")
        if not isinstance(metadata_by_tag, dict):
            msg = "MolmoAct2 norm stats are missing metadata_by_tag."
            raise TypeError(msg)
        tag_metadata = metadata_by_tag.get(self.norm_tag)
        if tag_metadata is None:
            msg = f"Normalization tag {self.norm_tag!r} was not found in MolmoAct2 norm stats."
            raise ValueError(msg)
        if not isinstance(tag_metadata, dict):
            msg = f"Normalization metadata for tag {self.norm_tag!r} is not a JSON object."
            raise TypeError(msg)
        return tag_metadata

    def _create_features_from_norm_stats(
        self,
        tag_metadata: dict[str, Any],
        image_size: tuple[int, int],
        *,
        normalize_gripper: bool,
    ) -> tuple[list[Feature], list[Feature]]:
        """Create input and output features from normalization metadata.

        Returns:
            Input and output feature definitions derived from the metadata.

        Raises:
            TypeError: If camera, state, or action metadata is malformed.
        """
        camera_keys = tag_metadata.get("camera_keys")
        if not isinstance(camera_keys, list) or not all(isinstance(key, str) for key in camera_keys):
            msg = f"Invalid camera_keys for normalization tag {self.norm_tag!r}."
            raise TypeError(msg)

        input_features = [
            Feature(
                name=camera_key.removeprefix("observation.images."),
                ftype=FeatureType.VISUAL,
                shape=(3, *image_size),
            )
            for camera_key in camera_keys
        ]

        state_key = tag_metadata.get("state_key")
        state_stats = tag_metadata.get("state_stats")
        if not isinstance(state_key, str) or not isinstance(state_stats, dict):
            msg = f"Invalid state metadata for normalization tag {self.norm_tag!r}."
            raise TypeError(msg)
        input_features.append(
            Feature(
                name=state_key.removeprefix("observation."),
                ftype=FeatureType.STATE,
                shape=(self._feature_size(state_stats, state_key),),
                normalization_data=self._normalization_parameters(
                    state_stats,
                    state_key,
                    normalize_gripper=normalize_gripper,
                ),
            ),
        )

        action_key = tag_metadata.get("action_key")
        action_stats = tag_metadata.get("action_stats")
        if not isinstance(action_key, str) or not isinstance(action_stats, dict):
            msg = f"Invalid action metadata for normalization tag {self.norm_tag!r}."
            raise TypeError(msg)
        output_features = [
            Feature(
                name=action_key,
                ftype=FeatureType.ACTION,
                shape=(self._feature_size(action_stats, action_key),),
                normalization_data=self._normalization_parameters(
                    action_stats,
                    action_key,
                    normalize_gripper=normalize_gripper,
                ),
            ),
        ]
        return input_features, output_features

    def _convert_config(  # noqa: PLR0914
        self,
        hf_config: dict[str, Any],
        norm_stats_config: dict[str, Any],
        tokenizer_config: dict[str, Any],
        snapshot_dir: Path,
    ) -> MolmoAct2Config:
        """Convert Hugging Face metadata into a MolmoAct2 configuration.

        Returns:
            The converted policy configuration.

        Raises:
            TypeError: If normalization metadata is malformed.
        """
        flat_config: dict[str, Any] = {}
        copy_component(hf_config, flat_config, "text_config", TEXT_CONFIG_MAP)
        copy_component(hf_config, flat_config, "vit_config", VISION_CONFIG_MAP)
        copy_component(hf_config, flat_config, "adapter_config", ADAPTER_CONFIG_MAP)
        copy_component(hf_config, flat_config, "action_expert_config", ACTION_EXPERT_CONFIG_MAP)
        copy_component(hf_config, flat_config, None, TOP_LEVEL_CONFIG_MAP)

        for tuple_field in ("image_default_input_size", "adapter_vit_layers"):
            value = flat_config.get(tuple_field)
            if isinstance(value, list):
                flat_config[tuple_field] = tuple(value)

        config = MolmoAct2Config(**flat_config)
        normalization_modes = {
            "q01_q99": "QUANTILES",
            "mean_std": "MEAN_STD",
        }
        norm_mode = norm_stats_config.get("norm_mode")
        normalization_mode = normalization_modes.get(str(norm_mode), config.normalization_mode)

        input_features = self.input_features
        output_features = self.output_features
        chunk_size = self.chunk_size
        normalize_gripper = config.normalize_gripper
        setup_type = self.setup_type or config.setup_type
        control_mode = self.control_mode or config.control_mode

        if self.norm_tag is not None:
            tag_metadata = self._resolve_norm_tag(norm_stats_config)
            normalize_gripper = bool(tag_metadata.get("normalize_gripper", False))
            tag_input_features, tag_output_features = self._create_features_from_norm_stats(
                tag_metadata,
                config.image_default_input_size,
                normalize_gripper=normalize_gripper,
            )
            if self.convert_pretrained_so101_stats:
                tag_input_features = _pretrained_normalization_to_so101_runtime(
                    tag_input_features,
                    FeatureType.STATE,
                )
                tag_output_features = _pretrained_normalization_to_so101_runtime(
                    tag_output_features,
                    FeatureType.ACTION,
                )
            input_features = self.input_features if self.input_features is not None else tag_input_features
            output_features = self.output_features if self.output_features is not None else tag_output_features
            action_horizon = tag_metadata.get("action_horizon")
            if not isinstance(action_horizon, int):
                msg = f"Invalid action_horizon for normalization tag {self.norm_tag!r}."
                raise TypeError(msg)
            chunk_size = action_horizon
            if self.setup_type is None:
                setup_type = str(tag_metadata.get("setup_type") or "")
            if self.control_mode is None:
                control_mode = str(tag_metadata.get("control_mode") or "")

        return replace(
            config,
            input_features=input_features,
            output_features=output_features,
            norm_tag=self.norm_tag,
            normalize_gripper=normalize_gripper,
            chunk_size=chunk_size,
            n_action_steps=self.n_action_steps,
            n_obs_steps=self.n_obs_steps,
            setup_type=setup_type,
            control_mode=control_mode,
            adapt_to_so101=self.adapt_to_so101,
            convert_pretrained_so101_stats=self.convert_pretrained_so101_stats,
            normalization_mode=normalization_mode,
            tokenizer_config=tokenizer_config,
            tokenizer_name_or_path=str(snapshot_dir),
            use_random_input_noise=self.use_random_input_noise,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            lora_bias=self.lora_bias,
        )

    @staticmethod
    def _from_hf(
        pretrained_name_or_path: str | Path,
    ) -> tuple[dict, dict, dict, Path]:
        """Load and validate a MolmoAct2 checkpoint from a path or Hugging Face repo.

        Returns:
            The model config, normalization stats, tokenizer config, and weights path.

        Raises:
            FileNotFoundError: If a required checkpoint file is missing.
            TypeError: If a required JSON payload is malformed.
        """
        path = Path(pretrained_name_or_path)

        if not path.is_dir():
            path = Path(
                snapshot_download(  # nosec B615
                    repo_id=str(pretrained_name_or_path),
                    allow_patterns=[
                        "config.json",
                        "norm_stats.json",
                        "processor_config.json",
                        "tokenizer.json",
                        "tokenizer_config.json",
                        "*.safetensors",
                        "model.safetensors.index.json",
                    ],
                ),
            )

        config_file = path / "config.json"
        norm_stats_file = path / "norm_stats.json"
        tokenizer_config_file = path / "tokenizer_config.json"

        if not config_file.is_file():
            msg = f"MolmoAct2 checkpoint at {path} is missing config.json."
            raise FileNotFoundError(msg)
        if not norm_stats_file.is_file():
            msg = f"MolmoAct2 checkpoint at {path} is missing norm_stats.json."
            raise FileNotFoundError(msg)
        if not tokenizer_config_file.is_file():
            msg = f"MolmoAct2 checkpoint at {path} is missing tokenizer_config.json."
            raise FileNotFoundError(msg)

        weights_file = path / "model.safetensors"
        if not weights_file.is_file():
            weights_file = path / "model.safetensors.index.json"
        if not weights_file.is_file():
            msg = f"MolmoAct2 checkpoint at {path} must contain model.safetensors or model.safetensors.index.json."
            raise FileNotFoundError(msg)

        with config_file.open(encoding="utf-8") as file:
            hf_config = json.load(file)
            if not isinstance(hf_config, dict):
                msg = f"MolmoAct2 config at {config_file} is not a valid JSON object."
                raise TypeError(msg)

        with norm_stats_file.open(encoding="utf-8") as file:
            norm_stats_config = json.load(file)
            if not isinstance(norm_stats_config, dict):
                msg = f"MolmoAct2 norm stats at {norm_stats_file} is not a valid JSON object."
                raise TypeError(msg)

        with tokenizer_config_file.open(encoding="utf-8") as file:
            tokenizer_config = json.load(file)
            if not isinstance(tokenizer_config, dict):
                msg = f"MolmoAct2 tokenizer config at {tokenizer_config_file} is not a valid JSON object."
                raise TypeError(msg)

        return hf_config, norm_stats_config, tokenizer_config, weights_file
