# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MolmoAct2 policy implementation."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import IO, TYPE_CHECKING, Any, Literal, override

import torch
from torch import Tensor

from physicalai.data.dataset import Dataset
from physicalai.data.observation import (
    ACTION,
    Feature,
    FeatureType,
    Observation,
)
from physicalai.policies.base import Policy
from physicalai.policies.utils import JointFrameTransform
from physicalai.policies.utils.features import get_feature_by_type

from .config import MolmoAct2Config
from .constants import SO101_JOINT_OFFSETS, SO101_JOINT_SIGNS
from .export import MolmoAct2ExportMixin
from .from_hf import MolmoAct2FromHFMixin
from .model import MolmoAct2Model
from .optimizer import MolmoAct2AdamW, molmoact2_cosine_with_warmup_scheduler
from .processors import (
    MolmoAct2Postprocessor,
    MolmoAct2Preprocessor,
    make_molmoact2_preprocessors,
)

if TYPE_CHECKING:
    from pathlib import Path

    from lightning.pytorch.utilities.types import OptimizerLRScheduler

    from physicalai.gyms import Gym

logger = logging.getLogger(__name__)


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
    normalization = JointFrameTransform(
        signs=SO101_JOINT_SIGNS,
        offsets=SO101_JOINT_OFFSETS,
    ).forward_normalization(
        feature.normalization_data,
        dimension=feature.shape[-1],
    )
    return [
        replace(candidate, normalization_data=normalization) if candidate is feature else candidate
        for candidate in features
    ]


class MolmoAct2(MolmoAct2ExportMixin, MolmoAct2FromHFMixin, Policy):
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
        convert_pretrained_so101_stats: bool = False,
        preserve_pretrained_normalization_in_training: bool = False,
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
        optimizer_lr: float = 5e-5,
        optimizer_vit_lr: float = 5e-5,
        optimizer_connector_lr: float = 5e-5,
        optimizer_action_expert_lr: float = 5e-5,
        optimizer_betas: tuple[float, float] = (0.9, 0.95),
        optimizer_eps: float = 1e-6,
        optimizer_weight_decay: float = 0.0,
        optimizer_grad_clip_norm: float = 1.0,
        scheduler_warmup_steps: int = 200,
        scheduler_decay_steps: int = 30_000,
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
            convert_pretrained_so101_stats: Whether to convert the released SO-101
                checkpoint's degree-based statistics for the PhysicalAI SO101 driver's
                normalized joint units. This compatibility option requires
                ``adapt_to_so101=True`` and the ``so100_so101_molmoact2`` normalization tag.
            preserve_pretrained_normalization_in_training: Whether ``setup("fit")`` keeps state and action
                normalization from an initialized pretrained policy when adopting the training
                dataset's feature contract. This does not affect explicit ``set_features`` calls.
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
            scheduler_warmup_steps: Number of linear warmup optimizer steps.
            scheduler_decay_steps: Cosine decay horizon in optimizer steps.
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

        resolved_adapt_to_so101 = norm_tag == "so100_so101_molmoact2" if adapt_to_so101 is None else adapt_to_so101
        if convert_pretrained_so101_stats and not resolved_adapt_to_so101:
            msg = "convert_pretrained_so101_stats requires adapt_to_so101=True."
            raise ValueError(msg)
        if convert_pretrained_so101_stats and norm_tag != "so100_so101_molmoact2":
            msg = "convert_pretrained_so101_stats is only supported with norm_tag='so100_so101_molmoact2'."
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
        self.adapt_to_so101 = resolved_adapt_to_so101
        self.convert_pretrained_so101_stats = convert_pretrained_so101_stats
        self.preserve_pretrained_normalization_in_training = preserve_pretrained_normalization_in_training
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
        preserve_pretrained_normalization_in_training: bool = False,
        compile_model: bool = False,
        openvino_compress_to_fp16: bool = False,
        gradient_checkpointing: bool = False,
        use_lora: bool = False,
        enable_lora_action_expert: bool = False,
        train_action_head_only: bool = False,
        optimizer_lr: float = 5e-5,
        optimizer_vit_lr: float = 5e-5,
        optimizer_connector_lr: float = 5e-5,
        optimizer_action_expert_lr: float = 5e-5,
        optimizer_betas: tuple[float, float] = (0.9, 0.95),
        optimizer_eps: float = 1e-6,
        optimizer_weight_decay: float = 0.0,
        optimizer_grad_clip_norm: float = 1.0,
        scheduler_warmup_steps: int = 200,
        scheduler_decay_steps: int = 30_000,
        scheduler_decay_lr: float = 1e-6,
    ) -> MolmoAct2:
        """Create a policy directly from a resolved model configuration.

        Args:
            config: Resolved MolmoAct2 model and processor configuration.
            preserve_pretrained_normalization_in_training: Whether ``setup("fit")`` keeps state and action
                normalization from the supplied configuration when adopting the training
                dataset's feature contract. This does not affect explicit ``set_features`` calls.
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
            scheduler_warmup_steps: Number of linear warmup optimizer steps.
            scheduler_decay_steps: Cosine decay horizon in optimizer steps.
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
            convert_pretrained_so101_stats=config.convert_pretrained_so101_stats,
            preserve_pretrained_normalization_in_training=preserve_pretrained_normalization_in_training,
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
                convert_pretrained_so101_stats=self.convert_pretrained_so101_stats,
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
        self.convert_pretrained_so101_stats = config.convert_pretrained_so101_stats

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

        Example:
            Initializing, setting features, and exporting a policy model:

            >>> import torch
            >>> policy = MolmoAct2(
            ...     pretrained_name_or_path="allenai/MolmoAct2-SO100_101",
            ...     norm_tag="so100_so101_molmoact2",
            ...     adapt_to_so101=True,
            ...     convert_pretrained_so101_stats=True,
            ... )
            >>> policy.set_features(
            ...     input_features=input_features,
            ...     output_features=output_features,
            ...     copy_state_normalization=True,
            ...     copy_action_normalization=True,
            ... )
            >>> policy.export("exports/molmoact2-so101-torch", backend="torch")
        """
        self._require_model()
        config = self._require_config()

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

        self._set_resolved_features(resolved_input_features, resolved_output_features)

    def rename_features(self, mapping: Mapping[str, str]) -> None:
        """Rename resolved input features without changing their metadata or order.

        Args:
            mapping: Current input feature names mapped to replacement names.

        Raises:
            ValueError: If a source name is unknown, a replacement name is invalid,
                or the result contains duplicate feature names.

        Example:
            Renaming a checkpoint camera feature to match the environment:

            >>> policy = MolmoAct2(
            ...     pretrained_name_or_path="allenai/MolmoAct2-LIBERO",
            ...     norm_tag="libero",
            ...     n_action_steps=10,
            ...     use_random_input_noise=True,
            ...     compile_model=True,
            ... )
            >>> policy.rename_features({"wrist_image": "image2"})
        """
        self._require_model()
        config = self._require_config()
        input_features = list(config.input_features or [])
        output_features = list(config.output_features or [])
        if not mapping:
            return

        if any(not isinstance(name, str) or not name for name in mapping):
            msg = "Feature names to rename must be non-empty strings."
            raise ValueError(msg)
        if any(not isinstance(name, str) or not name for name in mapping.values()):
            msg = "Replacement feature names must be non-empty strings."
            raise ValueError(msg)

        current_names = {feature.name for feature in input_features}
        unknown_names = sorted(set(mapping) - current_names)
        if unknown_names:
            msg = f"Cannot rename unknown input features: {unknown_names}."
            raise ValueError(msg)

        renamed_features = [
            replace(feature, name=mapping[feature.name])
            if feature.name is not None and feature.name in mapping
            else feature
            for feature in input_features
        ]
        renamed_names = [feature.name for feature in renamed_features]
        if len(renamed_names) != len(set(renamed_names)):
            msg = f"Feature renaming creates duplicate input names: {renamed_names}."
            raise ValueError(msg)

        self._set_resolved_features(renamed_features, output_features)

    def _set_resolved_features(
        self,
        input_features: list[Feature],
        output_features: list[Feature],
    ) -> None:
        """Install resolved features and rebuild their processors atomically."""
        model = self._require_model()
        config = self._require_config()
        training = self.training

        replacement_config = replace(
            config,
            input_features=input_features,
            output_features=output_features,
        )
        preprocessor, postprocessor = make_molmoact2_preprocessors(replacement_config)
        parameter = next(model.parameters())
        preprocessor.to(device=parameter.device, dtype=parameter.dtype)
        postprocessor.to(device=parameter.device, dtype=parameter.dtype)

        self.input_features = input_features
        self.output_features = output_features
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
        self._warn_if_dataset_quantiles_missing(dataset_input_features, dataset_output_features)

        # Replace eager features with the training dataset contract without reloading weights.
        if self.model is not None:
            config = self._require_config()
            if config.input_features != dataset_input_features or config.output_features != dataset_output_features:
                if self.preserve_pretrained_normalization_in_training:
                    logger.warning(
                        "Eager MolmoAct2 features differ from the training dataset; replacing the feature contract "
                        "while preserving the initialized state and action normalization statistics.",
                    )
                    self.set_features(
                        dataset_input_features,
                        dataset_output_features,
                        copy_state_normalization=True,
                        copy_action_normalization=True,
                    )
                else:
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

    def _warn_if_dataset_quantiles_missing(
        self,
        input_features: list[Feature],
        output_features: list[Feature],
    ) -> None:
        config = getattr(self, "config", None)
        if isinstance(config, MolmoAct2Config) and config.normalization_mode != "QUANTILES":
            return

        missing = []
        for feature in input_features + output_features:
            if feature.ftype not in {FeatureType.STATE, FeatureType.ACTION}:
                continue
            stats = feature.normalization_data
            if stats is None or stats.q01 is None or stats.q99 is None:
                missing.append(feature.name or str(feature.ftype))

        if missing:
            logger.warning(
                "MolmoAct2 uses quantile normalization, but the training dataset is missing q01/q99 statistics "
                "for: %s. Add them before training with `python -m "
                "lerobot.scripts.augment_dataset_quantile_stats --repo-id=your_dataset`.",
                ", ".join(missing),
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
            Action tensor shaped ``(batch, chunk_size, action_dim)``.

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

        Raises:
            RuntimeError: If Lightning cannot determine the optimizer-step budget.
        """
        optimizer = MolmoAct2AdamW(
            self.get_optim_params(),
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            group_grad_clip_norm=self.optimizer_grad_clip_norm,
        )
        num_training_steps = int(self.trainer.estimated_stepping_batches)
        if num_training_steps < 1:
            msg = f"Expected at least one optimizer step, got {num_training_steps}."
            raise RuntimeError(msg)
        scheduler = molmoact2_cosine_with_warmup_scheduler(
            optimizer,
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
            num_training_steps=num_training_steps,
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
