# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MolmoAct2 policy implementation."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Literal, override

import torch
from huggingface_hub import snapshot_download
from physicalai.inference.data import InferenceFeature, InferenceFeatureDtype, InferenceFeatureType
from physicalai.inference.manifest import ComponentSpec
from torch import Tensor

from physicalai.data.dataset import Dataset
from physicalai.data.observation import ACTION, IMAGES, TASK, Feature, FeatureType, NormalizationParameters, Observation
from physicalai.export import ExportablePolicyMixin, ExportBackend
from physicalai.export.backends import ExportParameters, OpenVINOExportParameters, TorchExportParameters
from physicalai.policies.base import Policy
from physicalai.policies.utils.features import get_feature_by_type
from physicalai.train.schedulers import cosine_decay_with_warmup_scheduler

from .config import MolmoAct2Config
from .model import MolmoAct2Model
from .optimizer import MolmoAct2AdamW
from .pretrained_utils import (
    ACTION_EXPERT_CONFIG_MAP,
    ADAPTER_CONFIG_MAP,
    TEXT_CONFIG_MAP,
    TOP_LEVEL_CONFIG_MAP,
    VISION_CONFIG_MAP,
    copy_component,
)
from .processors import (
    MolmoAct2Postprocessor,
    MolmoAct2Preprocessor,
    make_molmoact2_preprocessors,
)
from .processors.joint_transform import SO101_JOINT_OFFSETS, SO101_JOINT_SIGNS, JointFrameTransform

if TYPE_CHECKING:
    from lightning.pytorch.utilities.types import OptimizerLRScheduler

    from physicalai.gyms import Gym

logger = logging.getLogger(__name__)


def _normalization_stats(
    feature: Feature | None,
) -> dict[str, float | list[float] | list[list[float]] | list[list[list[float]]] | list[bool] | None]:
    if feature is None or feature.normalization_data is None:
        return {}
    normalization = feature.normalization_data
    stats: dict[
        str,
        float | list[float] | list[list[float]] | list[list[list[float]]] | list[bool] | None,
    ] = {
        name: value
        for name in ("mean", "std", "min", "max", "q01", "q99")
        if (value := getattr(normalization, name)) is not None
    }
    if normalization.mask is not None:
        stats["mask"] = normalization.mask
    return stats


def _copy_feature_normalization(
    features: list[Feature],
    source: Feature | None,
    feature_type: FeatureType,
) -> list[Feature]:
    feature = get_feature_by_type(features, feature_type)
    if feature is None:
        msg = f"Cannot copy {feature_type.value} normalization without a replacement feature."
        raise ValueError(msg)
    if source is None or source.normalization_data is None:
        msg = f"Cannot copy {feature_type.value} normalization because the initialized policy has none."
        raise ValueError(msg)
    if feature.shape is None or feature.shape != source.shape:
        msg = f"Cannot copy {feature_type.value} normalization from shape {source.shape} to shape {feature.shape}."
        raise ValueError(msg)
    return [
        replace(candidate, normalization_data=source.normalization_data) if candidate is feature else candidate
        for candidate in features
    ]


def _normalization_to_checkpoint(features: list[Feature], feature_type: FeatureType) -> list[Feature]:
    feature = get_feature_by_type(features, feature_type)
    if feature is None or feature.normalization_data is None:
        return list(features)
    if not feature.shape:
        msg = f"Cannot adapt {feature_type.value} normalization without a concrete feature shape."
        raise ValueError(msg)
    normalization = JointFrameTransform().normalization_to_checkpoint(
        feature.normalization_data,
        dimension=feature.shape[-1],
    )
    return [
        replace(candidate, normalization_data=normalization) if candidate is feature else candidate
        for candidate in features
    ]


class MolmoAct2(ExportablePolicyMixin, Policy):  # noqa: PLR0904
    """MolmoAct2 policy wrapper for loading pretrained checkpoints and configs."""

    def __init__(  # noqa: PLR0913, PLR0915
        self,
        # Input and output features
        input_features: list[Feature] | None = None,
        output_features: list[Feature] | None = None,
        # Pretrained model and normalization tag
        pretrained_name_or_path: str | Path | None = "allenai/MolmoAct2",
        norm_tag: str | None = None,
        *,
        # Action and observation parameters
        n_action_steps: int = 30,
        chunk_size: int = 30,
        n_obs_steps: int = 1,
        setup_type: str | None = None,
        control_mode: str | None = None,
        adapt_to_so101: bool | None = None,
        # weight management
        compile_model: bool = False,
        openvino_compress_to_fp16: bool = False,
        gradient_checkpointing: bool = False,
        use_random_input_noise: bool = False,
        use_lora: bool = False,
        enable_lora_action_expert: bool = False,
        lora_rank: int = 64,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_bias: Literal["all", "lora_only", "none"] = "none",
        train_action_head_only: bool = False,
        # optimization
        optimizer_lr: float = 1e-5,
        optimizer_vit_lr: float = 5e-6,
        optimizer_connector_lr: float = 5e-6,
        optimizer_action_expert_lr: float = 5e-5,
        optimizer_betas: tuple[float, float] = (0.9, 0.95),
        optimizer_eps: float = 1e-6,
        optimizer_weight_decay: float = 0.0,
        optimizer_grad_clip_norm: float = 1.0,
        scheduler_warmup_steps: int = 200,
        scheduler_decay_steps: int = 24_000,
        scheduler_decay_lr: float = 1e-6,
    ) -> None:
        """Initialize a MolmoAct2 policy instance.

        Args:
            input_features: Input feature definitions used when initializing a local model.
            output_features: Output feature definitions used when initializing a local model.
            pretrained_name_or_path: Local path or Hugging Face repo ID for the pretrained
                checkpoint.
            norm_tag: Normalization tag identifying the dataset-specific normalization metadata.
            n_action_steps: Number of action steps predicted by the policy.
            chunk_size: Number of actions included in each action chunk.
            n_obs_steps: Number of observation steps included in the input history.
            setup_type: Optional setup identifier used by the model configuration.
            control_mode: Optional control mode used by the model configuration.
            adapt_to_so101: Whether to train in the legacy SO-101 checkpoint frame.
                When omitted, the SO-100/101 normalization tag enables it automatically.
            compile_model: Whether to compile model training and inference entrypoints.
            openvino_compress_to_fp16: Whether OpenVINO export compresses FP32 constants to FP16.
            gradient_checkpointing: Whether to enable gradient checkpointing on the model.
            use_random_input_noise: Whether action generation starts from Gaussian noise.
            use_lora: Whether to enable LoRA adapters on the model.
            enable_lora_action_expert: Whether LoRA adapters also target the action expert.
            lora_rank: LoRA rank.
            lora_alpha: LoRA scaling value.
            lora_dropout: LoRA dropout probability.
            lora_bias: LoRA bias training mode.
            train_action_head_only: Whether to freeze the VLM and train only the action head.
            optimizer_lr: Learning rate for text-model parameters.
            optimizer_vit_lr: Learning rate for vision-model parameters.
            optimizer_connector_lr: Learning rate for image connector parameters.
            optimizer_action_expert_lr: Learning rate for action-expert parameters.
            optimizer_betas: AdamW beta coefficients.
            optimizer_eps: AdamW epsilon.
            optimizer_weight_decay: AdamW weight decay.
            optimizer_grad_clip_norm: Independent gradient clipping norm for each parameter group.
            scheduler_warmup_steps: Number of linear warmup steps.
            scheduler_decay_steps: Number of cosine decay steps.
            scheduler_decay_lr: Final scheduler learning rate for the base parameter group.

        Raises:
            ValueError: If LoRA options are inconsistent or invalid.
        """
        if enable_lora_action_expert and not use_lora:
            msg = "enable_lora_action_expert requires use_lora=True."
            raise ValueError(msg)
        if use_lora and train_action_head_only:
            msg = "use_lora is incompatible with train_action_head_only."
            raise ValueError(msg)
        if lora_rank < 1:
            msg = "lora_rank must be positive."
            raise ValueError(msg)
        if not 0.0 <= lora_dropout < 1.0:
            msg = "lora_dropout must be in [0, 1)."
            raise ValueError(msg)

        # args
        self.input_features = input_features
        self.output_features = output_features
        self.pretrained_name_or_path = pretrained_name_or_path
        self.norm_tag = norm_tag
        self.n_action_steps = n_action_steps
        self.chunk_size = chunk_size
        self.n_obs_steps = n_obs_steps
        self.setup_type = setup_type
        self.control_mode = control_mode
        self.adapt_to_so101 = norm_tag == "so100_so101_molmoact2" if adapt_to_so101 is None else adapt_to_so101
        self.compile_model = compile_model
        self.openvino_compress_to_fp16 = openvino_compress_to_fp16
        self.gradient_checkpointing = gradient_checkpointing
        self.use_random_input_noise = use_random_input_noise
        self.use_lora = use_lora
        self.enable_lora_action_expert = enable_lora_action_expert
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_bias: Literal["all", "lora_only", "none"] = lora_bias
        self.train_action_head_only = train_action_head_only
        self.optimizer_lr = optimizer_lr
        self.optimizer_vit_lr = optimizer_vit_lr
        self.optimizer_connector_lr = optimizer_connector_lr
        self.optimizer_action_expert_lr = optimizer_action_expert_lr
        self.optimizer_betas = optimizer_betas
        self.optimizer_eps = optimizer_eps
        self.optimizer_weight_decay = optimizer_weight_decay
        self.optimizer_grad_clip_norm = optimizer_grad_clip_norm
        self.scheduler_warmup_steps = scheduler_warmup_steps
        self.scheduler_decay_steps = scheduler_decay_steps
        self.scheduler_decay_lr = scheduler_decay_lr

        # initialize super
        super().__init__(n_action_steps=self.n_action_steps)

        # ignore input and output features, subject to change
        self.save_hyperparameters(ignore=["input_features", "output_features", "compile_model"])

        # pre and post processors
        self._preprocessor: MolmoAct2Preprocessor | None = None  # type: ignore[assignment]
        self._postprocessor: MolmoAct2Postprocessor | None = None

        # underlying model
        self.model: MolmoAct2Model | None = None  # pyrefly: ignore[bad-override-mutable-attribute]

        # only init if features are resolved, lazy otherwise
        user_eager = input_features is not None and output_features is not None
        pretrained_eager = pretrained_name_or_path is not None and norm_tag is not None
        if user_eager or pretrained_eager:
            self.initialize_model()

    @classmethod
    def from_config(  # noqa: PLR0913
        cls,
        config: MolmoAct2Config,
        *,
        compile_model: bool = False,
        openvino_compress_to_fp16: bool = False,
        gradient_checkpointing: bool = False,
        use_lora: bool = False,
        enable_lora_action_expert: bool = False,
        train_action_head_only: bool = False,
        optimizer_lr: float = 1e-5,
        optimizer_vit_lr: float = 5e-6,
        optimizer_connector_lr: float = 5e-6,
        optimizer_action_expert_lr: float = 5e-5,
        optimizer_betas: tuple[float, float] = (0.9, 0.95),
        optimizer_eps: float = 1e-6,
        optimizer_weight_decay: float = 0.0,
        optimizer_grad_clip_norm: float = 1.0,
        scheduler_warmup_steps: int = 200,
        scheduler_decay_steps: int = 24_000,
        scheduler_decay_lr: float = 1e-6,
    ) -> MolmoAct2:
        """Create a policy directly from a resolved model configuration.

        Args:
            config: Resolved MolmoAct2 model and processor configuration.
            compile_model: Whether to compile model training and inference entrypoints.
            openvino_compress_to_fp16: Whether OpenVINO export compresses FP32 constants to FP16.
            gradient_checkpointing: Whether to enable gradient checkpointing on the model.
            use_lora: Whether to enable LoRA adapters on the model.
            enable_lora_action_expert: Whether LoRA adapters also target the action expert.
            train_action_head_only: Whether to freeze the VLM and train only the action head.
            optimizer_lr: Learning rate for text-model parameters.
            optimizer_vit_lr: Learning rate for vision-model parameters.
            optimizer_connector_lr: Learning rate for image connector parameters.
            optimizer_action_expert_lr: Learning rate for action-expert parameters.
            optimizer_betas: AdamW beta coefficients.
            optimizer_eps: AdamW epsilon.
            optimizer_weight_decay: AdamW weight decay.
            optimizer_grad_clip_norm: Independent gradient clipping norm for each parameter group.
            scheduler_warmup_steps: Number of linear warmup steps.
            scheduler_decay_steps: Number of cosine decay steps.
            scheduler_decay_lr: Final scheduler learning rate for the base parameter group.

        Returns:
            An initialized MolmoAct2 policy using ``config`` without pretrained resolution.
        """
        policy = cls(
            pretrained_name_or_path=None,
            norm_tag=config.norm_tag,
            n_action_steps=config.n_action_steps,
            chunk_size=config.chunk_size,
            n_obs_steps=config.n_obs_steps,
            setup_type=config.setup_type,
            control_mode=config.control_mode,
            adapt_to_so101=config.adapt_to_so101,
            compile_model=compile_model,
            openvino_compress_to_fp16=openvino_compress_to_fp16,
            gradient_checkpointing=gradient_checkpointing,
            use_random_input_noise=config.use_random_input_noise,
            use_lora=use_lora,
            enable_lora_action_expert=enable_lora_action_expert,
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            lora_bias=config.lora_bias,
            train_action_head_only=train_action_head_only,
            optimizer_lr=optimizer_lr,
            optimizer_vit_lr=optimizer_vit_lr,
            optimizer_connector_lr=optimizer_connector_lr,
            optimizer_action_expert_lr=optimizer_action_expert_lr,
            optimizer_betas=optimizer_betas,
            optimizer_eps=optimizer_eps,
            optimizer_weight_decay=optimizer_weight_decay,
            optimizer_grad_clip_norm=optimizer_grad_clip_norm,
            scheduler_warmup_steps=scheduler_warmup_steps,
            scheduler_decay_steps=scheduler_decay_steps,
            scheduler_decay_lr=scheduler_decay_lr,
        )
        policy._initialize_from_config(config)
        return policy

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path | IO[bytes],
        map_location: torch.device | str | int | Callable | dict | None = None,
        hparams_file: str | Path | None = None,
        strict: bool | None = None,  # noqa: FBT001
        weights_only: bool | None = None,  # noqa: FBT001
        **kwargs: Any,  # noqa: ANN401
    ) -> MolmoAct2:
        """Load a trained policy without resolving pretrained model weights.

        The checkpoint config rebuilds the model before Lightning restores its state dict.
        Tokenizer assets referenced by the config must remain available locally.

        Returns:
            The restored policy in the checkpoint's saved training mode.
        """
        kwargs["pretrained_name_or_path"] = None
        return super().load_from_checkpoint(
            checkpoint_path,
            map_location=map_location,
            hparams_file=hparams_file,
            strict=strict,
            weights_only=weights_only,
            **kwargs,
        )

    def _policy_config_for_checkpoint(self) -> dict[str, object]:
        return self._require_config().to_dict()

    def _restore_policy_config(self, config_data: Mapping[str, object]) -> None:
        config = MolmoAct2Config.from_dict(config_data)
        if self.model is not None:
            if self._require_config() != config:
                msg = "Checkpoint policy config does not match the initialized policy"
                raise ValueError(msg)
            return
        self._initialize_from_config(config)

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Save the resolved policy config alongside Lightning's state dict."""
        checkpoint["policy_config"] = self._policy_config_for_checkpoint()

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Rebuild the policy from its resolved checkpoint config.

        Raises:
            TypeError: If the checkpoint does not contain a valid policy config.
        """
        config_data = checkpoint.get("policy_config")
        if not isinstance(config_data, Mapping):
            msg = "MolmoAct2 checkpoint is missing a valid policy_config"
            raise TypeError(msg)
        self._restore_policy_config(config_data)

    def _require_model(self) -> MolmoAct2Model:
        if not isinstance(self.model, MolmoAct2Model):
            msg = "Policy model is not initialized"
            raise TypeError(msg)
        return self.model

    def _require_config(self) -> MolmoAct2Config:
        if self.config is None:
            msg = "Policy config is not initialized"
            raise RuntimeError(msg)
        return self.config

    def initialize_model(self) -> None:
        """Initialize the policy model and configuration from pretrained assets or local inputs.

        Args:
            None: This method reads the instance state and does not accept parameters.

        Raises:
            RuntimeError: If the instance is configured for local initialization without required
                input or output feature definitions.
        """
        # initialize model from pretrained if available
        if self.pretrained_name_or_path:
            # gather configs and weights from path (hf hub)
            hf_config, norm_stats_config, tokenizer_config, weights_path = self._from_hf(
                self.pretrained_name_or_path,
            )
            config = self._convert_config(
                hf_config,
                norm_stats_config,
                tokenizer_config,
                weights_path.parent,
            )
        else:
            if self.input_features is None or self.output_features is None:
                msg = "Input and output features are required to initialize MolmoAct2 without pretrained data."
                raise RuntimeError(msg)
            weights_path = None
            config = MolmoAct2Config(
                input_features=self.input_features,
                output_features=self.output_features,
                n_obs_steps=self.n_obs_steps,
                chunk_size=self.chunk_size,
                n_action_steps=self.n_action_steps,
                setup_type=self.setup_type or "",
                control_mode=self.control_mode or "",
                adapt_to_so101=self.adapt_to_so101,
                use_random_input_noise=self.use_random_input_noise,
                lora_rank=self.lora_rank,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                lora_bias=self.lora_bias,
            )

        # init model
        self._initialize_from_config(config, weights_path=weights_path)

    def _initialize_from_config(
        self,
        config: MolmoAct2Config,
        *,
        weights_path: Path | None = None,
    ) -> None:
        if self.model is not None:
            msg = "Policy model is already initialized"
            raise RuntimeError(msg)

        self.config = config
        self._weights_path = weights_path

        # update instance attributes from config
        self.input_features = config.input_features
        self.output_features = config.output_features
        self.n_action_steps = config.n_action_steps
        self.chunk_size = config.chunk_size
        self.n_obs_steps = config.n_obs_steps
        self.setup_type = config.setup_type
        self.control_mode = config.control_mode
        self.adapt_to_so101 = config.adapt_to_so101

        self.model = MolmoAct2Model.from_config(config)
        self._preprocessor, self._postprocessor = make_molmoact2_preprocessors(config)

        if weights_path is not None:
            self.model.load_weights(weights_path)

        self._apply_model_modifications()

    def set_features(
        self,
        input_features: list[Feature],
        output_features: list[Feature],
        *,
        copy_state_normalization: bool = False,
        copy_action_normalization: bool = False,
    ) -> None:
        """Replace policy features without reloading the initialized model.

        Args:
            input_features: Replacement input feature definitions.
            output_features: Replacement output feature definitions.
            copy_state_normalization: Whether to fill missing replacement state normalization
                with normalization resolved during policy initialization.
            copy_action_normalization: Whether to fill missing replacement action normalization
                with normalization resolved during policy initialization.
        """
        model = self._require_model()
        config = self._require_config()
        training = self.training

        resolved_input_features = list(input_features)
        resolved_output_features = list(output_features)
        if config.adapt_to_so101:
            resolved_input_features = _normalization_to_checkpoint(resolved_input_features, FeatureType.STATE)
            resolved_output_features = _normalization_to_checkpoint(resolved_output_features, FeatureType.ACTION)
        if copy_state_normalization:
            resolved_input_features = _copy_feature_normalization(
                resolved_input_features,
                get_feature_by_type(list(config.input_features or []), FeatureType.STATE),
                FeatureType.STATE,
            )
        if copy_action_normalization:
            resolved_output_features = _copy_feature_normalization(
                resolved_output_features,
                get_feature_by_type(list(config.output_features or []), FeatureType.ACTION),
                FeatureType.ACTION,
            )

        replacement_config = replace(
            config,
            input_features=resolved_input_features,
            output_features=resolved_output_features,
        )
        preprocessor, postprocessor = make_molmoact2_preprocessors(replacement_config)
        parameter = next(model.parameters())
        preprocessor.to(device=parameter.device, dtype=parameter.dtype)
        postprocessor.to(device=parameter.device, dtype=parameter.dtype)

        self.input_features = resolved_input_features
        self.output_features = resolved_output_features
        self.config = replacement_config
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self.train(training)
        self.reset()

    def _apply_model_modifications(self) -> None:
        model = self._require_model()

        if self.gradient_checkpointing:
            model.enable_gradient_checkpointing()

        if self.use_lora:
            model.enable_lora(enable_action_expert=self.enable_lora_action_expert)

        if self.train_action_head_only:
            model.freeze_vlm()

        if self.compile_model:
            model.enable_compile()

    @classmethod
    def _normalization_parameters(
        cls,
        stats: dict[str, Any],
        feature_key: str,
        *,
        normalize_gripper: bool,
    ) -> NormalizationParameters:
        """Build normalization metadata from saved statistics for a feature.

        Args:
            stats: Dictionary of saved normalization statistics for a single feature.
            feature_key: Name of the feature being normalized.
            normalize_gripper: Whether gripper dimensions should be normalized.

        Returns:
            NormalizationParameters: The normalized feature metadata populated from the input
                statistics.

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
        """Resolve the generic per-dimension normalization mask for a feature.

        Returns:
            ``None`` when every dimension should be normalized, otherwise the explicit
            pretrained mask.

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
        """Validate that dimensions excluded from normalization already use unit range.

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
        """Infer the vector size for a feature from its saved normalization statistics.

        Args:
            stats: Dictionary of normalization statistics associated with a feature.
            feature_key: Name of the feature being inspected.

        Returns:
            int: The length of the feature's vector-valued normalization statistics.

        Raises:
            ValueError: If the statistics do not contain a vector-valued array for the feature.
        """
        for stat_name in ("mean", "std", "min", "max", "q01", "q99"):
            value = stats.get(stat_name)
            if isinstance(value, list):
                return len(value)
        msg = f"MolmoAct2 normalization stats for {feature_key!r} contain no vector values."
        raise ValueError(msg)

    def _resolve_norm_tag(self, norm_stats_config: dict[str, Any]) -> dict[str, Any]:
        """Return the metadata for the selected normalization tag.

        Args:
            norm_stats_config: Dictionary containing the pretrained normalization statistics.

        Returns:
            dict[str, Any]: The metadata payload associated with the configured normalization tag.

        Raises:
            ValueError: If no normalization tag has been configured for the policy.
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
        """Create input and output feature definitions from normalization metadata.

        Args:
            tag_metadata: Metadata describing the selected normalization tag.
            image_size: Spatial dimensions used to construct visual feature shapes.
            normalize_gripper: Whether gripper dimensions should be normalized.

        Returns:
            tuple[list[Feature], list[Feature]]: Input and output feature definitions derived from
                the normalization metadata.

        Raises:
            TypeError: If camera, state, or action metadata is missing or malformed.
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
        """Convert Hugging Face metadata into the library's MolmoAct2 config object.

        Args:
            hf_config: Raw Hugging Face configuration dictionary.
            norm_stats_config: Normalization statistics metadata loaded from the checkpoint.
            tokenizer_config: Tokenizer configuration loaded from the checkpoint.
            snapshot_dir: Directory containing the checkpoint snapshot.

        Returns:
            MolmoAct2Config: The converted configuration object used by the policy.

        Raises:
            TypeError: If the normalization metadata is malformed for the selected tag or action
                horizon.
        """
        flat_config: dict[str, Any] = {}
        copy_component(hf_config, flat_config, "text_config", TEXT_CONFIG_MAP)
        copy_component(hf_config, flat_config, "vit_config", VISION_CONFIG_MAP)
        copy_component(hf_config, flat_config, "adapter_config", ADAPTER_CONFIG_MAP)
        copy_component(hf_config, flat_config, "action_expert_config", ACTION_EXPERT_CONFIG_MAP)
        copy_component(hf_config, flat_config, None, TOP_LEVEL_CONFIG_MAP)

        # convert lists to tuples
        for tuple_field in ("image_default_input_size", "adapter_vit_layers"):
            value = flat_config.get(tuple_field)
            if isinstance(value, list):
                flat_config[tuple_field] = tuple(value)

        # create config from flattened configuration
        config = MolmoAct2Config(**flat_config)

        # determine normalization mode based on norm_stats_config
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
        """Load and validate a MolmoAct2 checkpoint from a local path or Hugging Face repo.

        Args:
            pretrained_name_or_path: Local path or Hugging Face repository ID of the checkpoint.

        Returns:
            tuple[dict, dict, dict, Path]: The Hugging Face config, normalization stats, tokenizer
                config, and checkpoint weight file path.

        Raises:
            FileNotFoundError: If required checkpoint files are missing.
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

        # Parse config_file.
        with config_file.open(encoding="utf-8") as f:
            hf_config = json.load(f)
            if not isinstance(hf_config, dict):
                msg = f"MolmoAct2 config at {config_file} is not a valid JSON object."
                raise TypeError(msg)

        # Parse norm_stats_file.
        with norm_stats_file.open(encoding="utf-8") as f:
            norm_stats_config = json.load(f)
            if not isinstance(norm_stats_config, dict):
                msg = f"MolmoAct2 norm stats at {norm_stats_file} is not a valid JSON object."
                raise TypeError(msg)

        with tokenizer_config_file.open(encoding="utf-8") as f:
            tokenizer_config = json.load(f)
            if not isinstance(tokenizer_config, dict):
                msg = f"MolmoAct2 tokenizer config at {tokenizer_config_file} is not a valid JSON object."
                raise TypeError(msg)

        return hf_config, norm_stats_config, tokenizer_config, weights_file

    def setup(self, stage: str) -> None:
        """Setup the policy for a given stage.

        Raises:
            TypeError: If the training dataset is not a PhysicalAI Dataset.
        """
        # we should only set up the policy for the "fit" stage.
        if stage != "fit":
            return

        # retrieve train dataset
        train_dataset = self.trainer.datamodule.train_dataset  # type: ignore[attr-defined]
        if not isinstance(train_dataset, Dataset):
            msg = "Train dataset is not a PhysicalAI Dataset."
            raise TypeError(msg)

        # gather input and output features
        dataset_input_features, dataset_output_features = self._dataset_features(train_dataset)

        # Replace eager features with the training dataset contract without reloading weights.
        if self.model is not None:
            config = self._require_config()
            if config.input_features != dataset_input_features or config.output_features != dataset_output_features:
                logger.warning(
                    "Eager MolmoAct2 features differ from the training dataset; "
                    "replacing them with the dataset features and normalization statistics.",
                )
                self.set_features(dataset_input_features, dataset_output_features)
            return

        if self.adapt_to_so101:
            dataset_input_features = _normalization_to_checkpoint(dataset_input_features, FeatureType.STATE)
            dataset_output_features = _normalization_to_checkpoint(dataset_output_features, FeatureType.ACTION)
        self.input_features = dataset_input_features
        self.output_features = dataset_output_features
        self.initialize_model()

    @staticmethod
    def _dataset_features(dataset: Dataset) -> tuple[list[Feature], list[Feature]]:
        return (
            list(dataset.observation_features.values()),
            list(dataset.action_features.values()),
        )

    @override
    def forward(self, batch: Observation) -> Tensor | tuple[Tensor, dict[str, Tensor | float]]:
        """Compute training loss or predict an action chunk.

        Returns:
            A loss tuple in training mode or denormalized actions in evaluation mode.

        Raises:
            RuntimeError: If the model or preprocessor is not initialized.
        """
        if not self.training:
            return self.predict_action_chunk(batch)
        model = self._require_model()
        if self._preprocessor is None:
            msg = "Policy preprocessor is not initialized"
            raise RuntimeError(msg)
        return model(self._preprocessor(batch.to_dict()))

    @torch.no_grad()
    @override
    def predict_action_chunk(self, batch: Observation) -> Tensor:
        """Predict and denormalize an action chunk.

        Returns:
            Action tensor shaped ``(batch, n_action_steps, action_dim)``.

        Raises:
            RuntimeError: If the model or processors are not initialized.
        """
        model = self._require_model()
        if self._preprocessor is None or self._postprocessor is None:
            msg = "Policy processors are not initialized"
            raise RuntimeError(msg)
        processed = self._preprocessor(batch.to(self.device).to_dict())
        return self._postprocessor({ACTION: model.predict_action_chunk(processed)})[ACTION]

    def training_step(self, batch: Observation, batch_idx: int) -> Tensor:
        """Compute and log the training loss.

        Returns:
            The differentiable training loss.
        """
        del batch_idx
        loss, metrics = self(batch)
        self.log("train/loss", metrics["loss"], prog_bar=True)
        self.log("train/action_flow_loss", metrics["action_flow_loss"])
        return loss

    @override
    def compute_val_loss(self, batch: Observation) -> tuple[Tensor, dict[str, Tensor | float]]:
        """Compute denoised action MSE and flow-matching validation loss.

        Returns:
            The primary action MSE and detached validation metrics.

        Raises:
            RuntimeError: If the model or preprocessor is not initialized.
        """
        model = self._require_model()
        if self._preprocessor is None:
            msg = "Policy preprocessor is not initialized"
            raise RuntimeError(msg)
        return model.compute_val_loss(self._preprocessor(batch.to_dict()))

    @override
    def validation_step(self, batch: Gym | Observation, batch_idx: int) -> dict[str, float] | Tensor:
        """Evaluate an observation loss batch or a Gym rollout.

        Returns:
            The validation loss for observations or rollout metrics for Gym batches.
        """
        if not isinstance(batch, Observation):
            return self.evaluate_gym(batch, batch_idx, stage="val")
        loss, metrics = self.compute_val_loss(batch)
        for name in ("loss", "action_mse", "action_flow_loss"):
            self.log(
                f"val/{name}",
                metrics[name],
                prog_bar=name == "loss",
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        return loss

    def get_optim_params(self) -> list[dict[str, Any]]:
        """Group trainable parameters by model component.

        Returns:
            Non-empty optimizer groups with component-specific learning rates.
        """
        grouped: dict[str, list[torch.nn.Parameter]] = {
            "vlm": [],
            "vit": [],
            "connector": [],
            "action_expert": [],
        }
        for name, parameter in self._require_model().named_parameters():
            if not parameter.requires_grad:
                continue
            if "action_expert" in name:
                grouped["action_expert"].append(parameter)
            elif any(part in name for part in ("image_pooling_2d", "image_projector", "wte.new_embedding")):
                grouped["connector"].append(parameter)
            elif "vision_backbone" in name:
                grouped["vit"].append(parameter)
            else:
                grouped["vlm"].append(parameter)

        learning_rates = {
            "vlm": self.optimizer_lr,
            "vit": self.optimizer_vit_lr,
            "connector": self.optimizer_connector_lr,
            "action_expert": self.optimizer_action_expert_lr,
        }
        return [
            {"params": parameters, "lr": learning_rates[name], "name": name}
            for name, parameters in grouped.items()
            if parameters
        ]

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Build the MolmoAct2 optimizer and step-wise cosine scheduler.

        Returns:
            Lightning optimizer and scheduler configuration.
        """
        optimizer = MolmoAct2AdamW(
            self.get_optim_params(),
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            group_grad_clip_norm=self.optimizer_grad_clip_norm,
        )
        training_steps = int(self.trainer.estimated_stepping_batches)
        scheduler = cosine_decay_with_warmup_scheduler(
            optimizer,
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
            num_training_steps=training_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    @override
    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        """Leave clipping to :class:`MolmoAct2AdamW` for independent groups."""
        del optimizer, gradient_clip_val, gradient_clip_algorithm

    @property
    @override
    def sample_input(self) -> dict[str, torch.Tensor | str] | None:
        """A deterministic export sample valid for pass-through state dimensions.

        The synthetic state is zeroed only when its mask contains pass-through dimensions.
        """
        sample = super().sample_input
        state_feature = get_feature_by_type(self.input_features or [], FeatureType.STATE)
        normalization = state_feature.normalization_data if state_feature is not None else None
        if (
            sample is not None
            and state_feature is not None
            and normalization is not None
            and normalization.mask
            and not all(normalization.mask)
        ):
            state = sample.get(str(state_feature.name))
            if torch.is_tensor(state):
                sample[str(state_feature.name)] = torch.zeros_like(state)
        return sample

    @property
    def inputs_schema(self) -> list[InferenceFeature] | None:
        """Describe raw observation inputs exposed by exported MolmoAct2 policies.

        Raises:
            ValueError: If an input feature has no concrete shape.
        """
        if self.model is None or self.input_features is None:
            return None

        schema: list[InferenceFeature] = []
        for feature in self.input_features:
            if feature.shape is None:
                msg = f"Input feature '{feature.name}' requires a concrete shape for export."
                raise ValueError(msg)
            if feature.ftype == FeatureType.VISUAL:
                schema.append(
                    InferenceFeature(
                        ftype=InferenceFeatureType.VISUAL,
                        shape=tuple(feature.shape),
                        name=f"{IMAGES}.{feature.name}",
                        dtype=InferenceFeatureDtype.FLOAT32,
                    ),
                )
            elif feature.ftype == FeatureType.STATE:
                schema.append(
                    InferenceFeature(
                        ftype=InferenceFeatureType.STATE,
                        shape=tuple(feature.shape),
                        name=str(feature.name),
                        dtype=InferenceFeatureDtype.FLOAT32,
                    ),
                )
        schema.append(
            InferenceFeature(
                ftype=InferenceFeatureType.LANGUAGE,
                shape=(),
                name=TASK,
                dtype=InferenceFeatureDtype.STRING,
            ),
        )
        return schema

    @property
    def outputs_schema(self) -> list[InferenceFeature] | None:
        """Describe the denormalized action chunk emitted by exported policies.

        Raises:
            ValueError: If the action feature has no concrete shape.
        """
        if self.model is None or self.output_features is None:
            return None
        action_feature = get_feature_by_type(self.output_features, FeatureType.ACTION)
        if action_feature is None or action_feature.shape is None:
            msg = "MolmoAct2 export requires an action feature with a concrete shape."
            raise ValueError(msg)
        return [
            InferenceFeature(
                ftype=InferenceFeatureType.ACTION,
                shape=(self.n_action_steps, *action_feature.shape),
                name=ACTION,
                dtype=InferenceFeatureDtype.FLOAT32,
            ),
        ]

    def _openvino_token_ids(self) -> tuple[int, int, list[int]]:
        config = self._require_config()
        required = {
            "image_start_token_id": config.image_start_token_id,
            "image_end_token_id": config.image_end_token_id,
            "image_patch_id": config.image_patch_id,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            msg = f"MolmoAct2 OpenVINO export requires token IDs: {', '.join(missing)}"
            raise ValueError(msg)
        if self._preprocessor is None:
            msg = "MolmoAct2 preprocessor must be initialized before export."
            raise ValueError(msg)

        tokenizer = self._preprocessor.tokenizer
        bos_token_id = tokenizer.bos_token_id
        if not isinstance(bos_token_id, int):
            bos_token_id = tokenizer.eos_token_id
        pad_token_id = tokenizer.pad_token_id
        if not isinstance(bos_token_id, int) or not isinstance(pad_token_id, int):
            msg = "MolmoAct2 tokenizer must define integer BOS/EOS and padding token IDs."
            raise TypeError(msg)

        image_token_ids = [
            token_id
            for token_id in (
                config.image_patch_id,
                config.image_col_id,
                config.image_start_token_id,
                config.low_res_image_start_token_id,
                config.frame_start_token_id,
                config.image_end_token_id,
                config.frame_end_token_id,
                config.image_low_res_id,
            )
            if token_id is not None
        ]
        return bos_token_id, pad_token_id, image_token_ids

    @property
    def extra_export_args(self) -> dict[str, ExportParameters]:
        """Build Torch and OpenVINO export parameters.

        Raises:
            ValueError: If export features, token IDs, or processors are unavailable.
        """
        config = self._require_config()
        if self.input_features is None or self.output_features is None:
            msg = "MolmoAct2 export requires initialized input and output features."
            raise ValueError(msg)

        state_feature = get_feature_by_type(self.input_features, FeatureType.STATE)
        action_feature = get_feature_by_type(self.output_features, FeatureType.ACTION)
        if action_feature is None or action_feature.shape is None:
            msg = "MolmoAct2 export requires an action feature with a concrete shape."
            raise ValueError(msg)

        bos_token_id, pad_token_id, image_token_ids = self._openvino_token_ids()
        image_size = (
            int(config.image_processor_size["height"]),
            int(config.image_processor_size["width"]),
        )
        joint_params = {
            "joint_signs": list(SO101_JOINT_SIGNS),
            "joint_offsets": list(SO101_JOINT_OFFSETS),
        }
        preprocessors = [
            ComponentSpec(
                type="molmoact2",
                image_keys=[
                    str(feature.name)
                    for feature in self.input_features
                    if feature.ftype == FeatureType.VISUAL and feature.name
                ],
                state_stats=_normalization_stats(state_feature),
                normalization_mode=config.normalization_mode,
                image_size=image_size,
                num_state_tokens=config.num_state_tokens,
                setup_type=config.setup_type,
                control_mode=config.control_mode,
                add_setup_tokens=config.add_setup_tokens,
                add_control_tokens=config.add_control_tokens,
                adapt_to_so101=config.adapt_to_so101,
                **joint_params,
            ),
            ComponentSpec(
                type="ov_tokenizer",
                artifact="tokenizer.xml",
            ),
            ComponentSpec(
                type="molmoact2_inputs",
                max_action_dim=config.max_action_dim,
                action_dim=int(action_feature.shape[-1]),
                bos_token_id=bos_token_id,
                pad_token_id=pad_token_id,
                image_placeholder_token_id=config.image_placeholder_token_id,
                image_start_token_id=config.image_start_token_id,
                image_end_token_id=config.image_end_token_id,
                image_patch_id=config.image_patch_id,
                image_col_id=config.image_col_id,
                low_res_image_start_token_id=config.low_res_image_start_token_id,
                frame_start_token_id=config.frame_start_token_id,
                frame_end_token_id=config.frame_end_token_id,
                image_low_res_id=config.image_low_res_id,
                image_size=image_size,
                patch_size=config.image_processor_patch_size,
                pooling_size=tuple(config.image_processor_pooling_size),
                image_mean=config.image_processor_mean,
                image_std=config.image_processor_std,
                image_crop_mode=config.image_processor_crop_mode,
                image_use_col_tokens=config.image_use_col_tokens,
                use_single_crop_col_tokens=config.use_single_crop_col_tokens,
                use_single_crop_start_token=config.use_single_crop_start_token,
                image_token_ids=image_token_ids,
            ),
        ]
        return {
            ExportBackend.TORCH: TorchExportParameters(
                preprocessors_specs=[ComponentSpec(type="to_float_tensor")],
            ),
            ExportBackend.OPENVINO: OpenVINOExportParameters(
                outputs=[feature.name for feature in (self.outputs_schema or [])],
                export_tokenizer=True,
                compress_to_fp16=self.openvino_compress_to_fp16,
                via_onnx=False,
                preprocessors_specs=preprocessors,
                postprocessors_specs=[
                    ComponentSpec(
                        type="molmoact2_postprocess",
                        action_stats=_normalization_stats(action_feature),
                        normalization_mode=config.normalization_mode,
                        adapt_to_so101=config.adapt_to_so101,
                        **joint_params,
                    ),
                ],
            ),
        }

    @staticmethod
    def get_supported_export_backends() -> list[str | ExportBackend]:
        """Return export backends implemented by MolmoAct2."""
        return [ExportBackend.TORCH, ExportBackend.OPENVINO]
