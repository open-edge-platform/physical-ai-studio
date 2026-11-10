# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2026 Gaoyue Zhou
# Authors: Gaoyue Zhou, Zichen Jeff Cui
# SPDX-License-Identifier: MIT

"""Lightning policy for Patch Policy."""

from __future__ import annotations

import warnings
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import IO, TYPE_CHECKING, Any, cast

import torch
from torch import Tensor

from physicalai.data import Dataset, Feature, FeatureType, NormalizationParameters, Observation
from physicalai.policies.base import Model, Policy

from .components.encoder import get_default_encoder_config, resolve_encoder_artifacts
from .components.heads import BehaviourTransformerHead
from .config import (
    ActionHeadName,
    FeatureEncoderName,
    PatchPolicyConfig,
    resolve_action_dim,
)
from .export import PatchPolicyExportMixin
from .model import PatchPolicyModel
from .preparation import fit_tokenizer
from .processors import (
    PatchPolicyPostprocessor,
    PatchPolicyPreprocessor,
    make_policy_processors,
    resolve_camera_name,
)

if TYPE_CHECKING:
    from pathlib import Path


def _feature_key(feature: Feature) -> tuple[FeatureType | None, str | None]:
    name = resolve_camera_name(feature) if feature.ftype is FeatureType.VISUAL else feature.name
    return feature.ftype, name


def _without_normalization(config: PatchPolicyConfig) -> PatchPolicyConfig:
    return replace(
        config,
        input_features=[
            replace(feature, name=_feature_key(feature)[1], normalization_data=None)
            for feature in config.input_features
        ],
        output_features=[replace(feature, normalization_data=None) for feature in config.output_features],
    )


def _copy_normalization_data(features: list[Feature], sources: list[Feature]) -> list[Feature]:
    """Replace configured statistics with matching training dataset statistics.

    Returns:
        Features with statistics copied from matching dataset features.

    Raises:
        ValueError: If matching features have different shapes.
    """
    source_features = {_feature_key(feature): feature for feature in sources}
    resolved = []
    for feature in features:
        source = source_features.get(_feature_key(feature))
        if source is None:
            resolved.append(feature)
            continue
        if feature.shape != source.shape:
            msg = f"Feature {feature.name!r} shape {feature.shape} does not match dataset feature shape {source.shape}."
            raise ValueError(msg)
        if feature.normalization_data is not None and feature.normalization_data != source.normalization_data:
            warnings.warn(
                f"Replacing normalization statistics for feature {feature.name!r} with training dataset statistics.",
                stacklevel=2,
            )
        resolved.append(replace(feature, normalization_data=source.normalization_data))
    return resolved


def _attach_encoder_normalization(
    features: list[Feature],
    mean: Tensor,
    std: Tensor,
) -> list[Feature]:
    """Attach the selected timm encoder's input statistics to visual features.

    Returns:
        Features with encoder statistics attached to visual inputs.
    """
    normalization = NormalizationParameters(
        mean=mean.flatten().detach().cpu().tolist(),
        std=std.flatten().detach().cpu().tolist(),
    )
    resolved = []
    for feature in features:
        if feature.ftype is not FeatureType.VISUAL:
            resolved.append(feature)
            continue
        if feature.normalization_data is not None and feature.normalization_data != normalization:
            warnings.warn(
                f"Replacing normalization statistics for visual feature {feature.name!r} with encoder statistics.",
                stacklevel=2,
            )
        resolved.append(replace(feature, normalization_data=normalization))
    return resolved


class PatchPolicy(PatchPolicyExportMixin, Policy):
    """Train a timm visual encoder followed by a diffusion or Behavior Transformer head.

    Args:
        input_features: Optional observation schema. Resolved from training data when omitted.
        output_features: Optional action schema. Resolved from training data when omitted.
        encoder_name: Name of the timm visual encoder architecture.
        encoder_pretrained: Whether to initialize the encoder with pretrained weights.
        action_head: Action-head architecture used to predict action chunks.
        n_obs_steps: Number of observation timesteps used as context.
        chunk_size: Number of action timesteps predicted per chunk.
        n_action_steps: Number of predicted actions executed per invocation.
        hidden_dim: Hidden width of the action head.
        num_layers: Number of transformer layers in the action head.
        num_heads: Number of attention heads in each transformer layer.
        num_inference_steps: Number of denoising steps used during inference.
        vqvae_iters: Full passes over unique action windows when fitting a new tokenizer.
        vqvae_batch_size: Tokenizer fitting minibatch size.
        optimizer_lr: AdamW learning rate.
        optimizer_weight_decay: AdamW weight-decay coefficient.
        compile_model: Compile the frozen feature encoder with torch.compile.
    """

    model: Model | None  # pyrefly: ignore[bad-override-mutable-attribute]
    _config: PatchPolicyConfig | None
    encoder_name: FeatureEncoderName
    action_head: ActionHeadName
    _preprocessor: PatchPolicyPreprocessor | None  # pyrefly: ignore[bad-override-mutable-attribute]
    _postprocessor: PatchPolicyPostprocessor | None
    _n_action_steps: int
    _action_queue: deque[Tensor]

    def __init__(  # noqa: PLR0913
        self,
        input_features: list[Feature] | None = None,
        output_features: list[Feature] | None = None,
        *,
        encoder_name: FeatureEncoderName = "dinov2",
        encoder_pretrained: bool = True,
        action_head: ActionHeadName = "behavior_transformer",
        n_obs_steps: int = 5,
        chunk_size: int = 5,
        n_action_steps: int = 1,
        hidden_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        num_inference_steps: int = 100,
        vqvae_iters: int = 1000,
        vqvae_batch_size: int = 1024,
        optimizer_lr: float = 5.5e-5,
        optimizer_weight_decay: float = 2e-4,
        compile_model: bool = False,
    ) -> None:
        """Initialize policy settings and configure the model when features are available.

        Raises:
            ValueError: If tokenizer fitting controls are not positive.
        """
        super().__init__(n_action_steps=n_action_steps)
        if vqvae_iters <= 0 or vqvae_batch_size <= 0:
            msg = "vqvae_iters and vqvae_batch_size must be positive."
            raise ValueError(msg)

        self.model = None
        self._config = None

        self.input_features = input_features
        self.output_features = output_features

        self._preprocessor = None
        self._postprocessor = None
        self._processor_config: PatchPolicyConfig | None = None

        # Action-related parameters
        self.n_obs_steps = n_obs_steps
        self.chunk_size = chunk_size
        self.n_action_steps = n_action_steps

        # Model params
        self.encoder_name = encoder_name
        self.encoder_pretrained = encoder_pretrained
        self._encoder_config = get_default_encoder_config(encoder_name)
        self._encoder_weights: dict[str, Tensor] | None = None
        self.action_head = action_head
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_inference_steps = num_inference_steps
        self.vqvae_iters = vqvae_iters
        self.vqvae_batch_size = vqvae_batch_size

        # Extra params for training
        self.optimizer_lr = optimizer_lr
        self.optimizer_weight_decay = optimizer_weight_decay
        self.compile_model = compile_model
        self._encoder_compiled = False

        # Don't save possibly unresolved input / output features
        self.save_hyperparameters(ignore=["input_features", "output_features", "compile_model"])

        if encoder_pretrained:
            self._encoder_config, self._encoder_weights = resolve_encoder_artifacts(encoder_name, pretrained=True)

        if self.input_features is not None and self.output_features is not None:
            self.configure_model()

    @property
    def config(self) -> PatchPolicyConfig:
        """The resolved feature and architecture contract.

        Raises:
            RuntimeError: If features have not been resolved.
        """
        if self._config is None:
            msg = "Patch Policy config is not initialized."
            raise RuntimeError(msg)
        return self._config

    def setup(self, stage: str) -> None:
        """Resolve training features and normalization statistics during fitting.

        Raises:
            TypeError: If the training dataset is not a physicalai Dataset.
            ValueError: If configured features differ from the training dataset.
        """
        # only setup on fit stage
        if stage != "fit":
            return

        # Get the training features from the datamodule
        train_dataset = self.trainer.datamodule.train_dataset  # type: ignore[attr-defined]
        if not isinstance(train_dataset, Dataset):
            msg = f"Expected physicalai Dataset, got {type(train_dataset)}."
            raise TypeError(msg)

        dataset_input_features = list(train_dataset.observation_features.values())
        dataset_output_features = list(train_dataset.action_features.values())

        # Validate that the configured input and output features match the dataset features
        if self.input_features is not None and [
            (_feature_key(feature), feature.shape) for feature in self.input_features
        ] != [(_feature_key(feature), feature.shape) for feature in dataset_input_features]:
            msg = "Configured input_features do not match the training dataset features."
            raise ValueError(msg)
        if self.output_features is not None and [
            (_feature_key(feature), feature.shape) for feature in self.output_features
        ] != [(_feature_key(feature), feature.shape) for feature in dataset_output_features]:
            msg = "Configured output_features do not match the training dataset features."
            raise ValueError(msg)

        # Copy normalization data from the dataset to the configured features for new training
        input_features = _copy_normalization_data(
            dataset_input_features if self.input_features is None else self.input_features,
            dataset_input_features,
        )
        output_features = _copy_normalization_data(
            dataset_output_features if self.output_features is None else self.output_features,
            dataset_output_features,
        )
        self.input_features = input_features
        self.output_features = output_features

    def on_fit_start(self) -> None:
        """Fit the action tokenizer after checkpoint restoration and device placement."""
        super().on_fit_start()
        fit_tokenizer(self)

    @classmethod
    def from_config(
        cls,
        config: PatchPolicyConfig,
        *,
        optimizer_lr: float = 5.5e-5,
        optimizer_weight_decay: float = 2e-4,
        compile_model: bool = False,
        vqvae_iters: int = 1000,
        vqvae_batch_size: int = 1024,
    ) -> PatchPolicy:
        """Build a policy from a resolved configuration.

        Returns:
            Configured policy with its model and processors initialized.
        """
        policy = cls(
            input_features=None,
            output_features=None,
            encoder_name=config.encoder_name,
            encoder_pretrained=False,
            action_head=config.action_head,
            n_obs_steps=config.n_obs_steps,
            chunk_size=config.chunk_size,
            n_action_steps=config.n_action_steps,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            num_inference_steps=config.num_inference_steps,
            vqvae_iters=vqvae_iters,
            vqvae_batch_size=vqvae_batch_size,
            optimizer_lr=optimizer_lr,
            optimizer_weight_decay=optimizer_weight_decay,
            compile_model=compile_model,
        )
        policy._config = config
        policy.input_features = config.input_features
        policy.output_features = config.output_features
        policy.configure_model()
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
    ) -> PatchPolicy:
        """Restore a policy from a Lightning checkpoint.

        Returns:
            Policy restored with its configuration and learned weights.
        """
        # Saved config supplies encoder metadata; Lightning supplies all weights.
        kwargs.update(encoder_pretrained=False, input_features=None, output_features=None)
        return super().load_from_checkpoint(
            checkpoint_path,
            map_location=map_location,
            hparams_file=hparams_file,
            strict=strict,
            weights_only=weights_only,
            **kwargs,
        )

    def configure_model(self) -> None:
        """Resolve config, validate compatibility, build once, and synchronize processors."""
        if self.input_features is None or self.output_features is None:
            return

        config = self._resolve_config(self.input_features, self.output_features)

        # Validate before creating modules or changing policy state.
        model = cast("PatchPolicyModel | None", self.model)
        if model is not None:
            self._validate_model_config(config)

        # All construction routes build the model here, exactly once.
        if model is None:
            model = PatchPolicyModel.from_config(config)
            if self._encoder_weights is not None:
                model.encoder.load_pretrained_weights(self._encoder_weights)

        config = replace(
            config,
            input_features=_attach_encoder_normalization(config.input_features, model.encoder.mean, model.encoder.std),
        )
        if config == self._config:
            config = self.config

        # Statistics can change without replacing the model or its parameters.
        preprocessor = self._preprocessor
        postprocessor = self._postprocessor
        refresh_processors = config != self._processor_config
        if preprocessor is None:
            refresh_processors = True
        if postprocessor is None:
            refresh_processors = True
        if refresh_processors:
            preprocessor, postprocessor = make_policy_processors(config, model.encoder.image_size)

        if self.compile_model and not self._encoder_compiled:
            model.encoder.compile()
            self._encoder_compiled = True

        # Commit the synchronized state only after construction succeeds.
        self.model = model
        self._config = config
        self._encoder_weights = None
        self._processor_config = config
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self.input_features = config.input_features
        self.output_features = config.output_features
        if self._n_action_steps != config.n_action_steps:
            self._action_queue = deque(maxlen=config.n_action_steps)
            self._n_action_steps = config.n_action_steps
        self._sync_config_settings(config)

    def _resolve_config(self, input_features: list[Feature], output_features: list[Feature]) -> PatchPolicyConfig:
        if self._config is not None:
            return replace(self._config, input_features=list(input_features), output_features=list(output_features))

        return PatchPolicyConfig(
            input_features=list(input_features),
            output_features=list(output_features),
            action_dim=resolve_action_dim(output_features),
            encoder_name=self.encoder_name,
            encoder_config=self._encoder_config,
            action_head=self.action_head,
            n_obs_steps=self.n_obs_steps,
            chunk_size=self.chunk_size,
            n_action_steps=self.n_action_steps,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            num_inference_steps=self.num_inference_steps,
        )

    def _validate_model_config(self, config: PatchPolicyConfig) -> None:
        """Check the saved architecture and ordered feature contract before mutation.

        Raises:
            TypeError: If the initialized model is not a PatchPolicyModel.
            ValueError: If the initialized model has no matching config.
        """
        if not isinstance(self.model, PatchPolicyModel):
            msg = "Patch Policy requires a PatchPolicyModel instance."
            raise TypeError(msg)
        if self._config is None:
            msg = "Initialized Patch Policy model is missing its config."
            raise ValueError(msg)
        if _without_normalization(config) != _without_normalization(self._config):
            msg = "Resolved policy config does not match the initialized model."
            raise ValueError(msg)

    def _sync_config_settings(self, config: PatchPolicyConfig) -> None:
        self.encoder_name = config.encoder_name
        self._encoder_config = config.encoder_config
        self.action_head = config.action_head
        self.n_obs_steps = config.n_obs_steps
        self.chunk_size = config.chunk_size
        self.n_action_steps = config.n_action_steps
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.num_inference_steps = config.num_inference_steps
        self.hparams.update(
            encoder_name=config.encoder_name,
            action_head=config.action_head,
            n_obs_steps=config.n_obs_steps,
            chunk_size=config.chunk_size,
            n_action_steps=config.n_action_steps,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            num_inference_steps=config.num_inference_steps,
        )

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Store the resolved policy configuration in a checkpoint.

        Raises:
            RuntimeError: If the policy has not been configured.
        """
        if self._config is None:
            msg = "Cannot save a Patch Policy checkpoint before the policy is configured."
            raise RuntimeError(msg)
        checkpoint["policy_config"] = self.config.to_dict()
        super().on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Initialize the policy from its checkpoint configuration.

        Raises:
            TypeError: If the checkpoint lacks a valid policy configuration.
        """
        config_data = checkpoint.get("policy_config")
        if not isinstance(config_data, Mapping):
            msg = "Patch Policy checkpoint is missing a valid policy_config."
            raise TypeError(msg)
        config = PatchPolicyConfig.from_dict(config_data)
        if self.model is not None:
            self._validate_model_config(config)
        self._encoder_weights = None
        self.encoder_pretrained = False
        self.hparams.update(encoder_pretrained=False)
        self._config = config
        self.input_features = config.input_features
        self.output_features = config.output_features
        self.configure_model()
        super().on_load_checkpoint(checkpoint)

    def _prepare_batch(
        self,
        batch: Observation,
        *,
        require_actions: bool,
    ) -> dict[str, Any]:
        if self._config is None or self._preprocessor is None:
            msg = "Patch Policy is not initialized."
            raise RuntimeError(msg)
        values = batch.to(self.device).to_dict()
        if require_actions and not isinstance(values.get("action"), Tensor):
            msg = "Training requires tensor actions."
            raise TypeError(msg)
        return self._preprocessor(values)

    def forward(self, batch: Observation) -> Tensor | tuple[Tensor, dict[str, Tensor | float]]:
        """Compute training loss or predict actions according to the current mode.

        Returns:
            Training loss and metrics, or predicted actions during inference.

        Raises:
            RuntimeError: If the policy is not initialized.
        """
        if self.model is None:
            msg = "Patch Policy is not initialized."
            raise RuntimeError(msg)
        if self.training:
            return self.model(self._prepare_batch(batch, require_actions=True))
        return self.predict_action_chunk(batch)

    @torch.no_grad()
    def predict_action_chunk(self, batch: Observation) -> Tensor:
        """Predict and postprocess an action chunk.

        Returns:
            Actions for the configured execution horizon.

        Raises:
            RuntimeError: If the model or postprocessor is not initialized.
        """
        if not isinstance(self.model, PatchPolicyModel) or self._postprocessor is None:
            msg = "Patch Policy is not initialized."
            raise RuntimeError(msg)
        prepared = self._prepare_batch(batch, require_actions=False)
        return self._postprocessor(self.model.predict_action_chunk(prepared))

    def compute_val_loss(self, batch: Observation) -> tuple[Tensor, dict[str, Tensor | float]]:
        """Compute validation loss for an observation batch.

        Returns:
            Validation loss and per-component metrics.

        Raises:
            RuntimeError: If the policy is not initialized.
        """
        if self.model is None:
            msg = "Patch Policy is not initialized."
            raise RuntimeError(msg)
        return self.model.compute_val_loss(self._prepare_batch(batch, require_actions=True))

    def training_step(self, batch: Observation, batch_idx: int) -> Tensor:
        """Compute and log training loss and metrics.

        Returns:
            Loss used for backpropagation.

        Raises:
            TypeError: If training forward does not return loss and metrics.
        """
        del batch_idx
        result = self(batch)
        if not isinstance(result, tuple):
            msg = "Training forward must return loss and metrics."
            raise TypeError(msg)
        loss, metrics = result
        self.log_dict({f"train/{name}": value for name, value in metrics.items()}, prog_bar=True)
        return loss

    def on_before_zero_grad(self, optimizer: torch.optim.Optimizer) -> None:
        """Update diffusion EMA weights after each optimizer step."""
        del optimizer
        if isinstance(self.model, PatchPolicyModel):
            self.model.update_ema()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Create the optimizer for trainable policy parameters.

        Returns:
            AdamW optimizer excluding frozen parameters.

        Raises:
            RuntimeError: If the model has not been configured.
        """
        if self.model is None:
            msg = "Cannot configure optimizers before Patch Policy features and model are resolved."
            raise RuntimeError(msg)
        head = self.model.action_head
        if isinstance(head, BehaviourTransformerHead):
            decay: dict[str, torch.nn.Parameter] = {}
            no_decay: dict[str, torch.nn.Parameter] = {}
            for module_name, module in head.trunk.named_modules():
                for name, parameter in module.named_parameters(recurse=False):
                    full_name = f"{module_name}.{name}" if module_name else name
                    group = decay if isinstance(module, torch.nn.Linear) and name == "weight" else no_decay
                    group[full_name] = parameter
            return torch.optim.AdamW(
                [
                    {"params": [decay[name] for name in sorted(decay)], "weight_decay": self.optimizer_weight_decay},
                    {"params": [no_decay[name] for name in sorted(no_decay)], "weight_decay": 0.0},
                    {"params": head.code_head.parameters(), "weight_decay": 0.01},
                    {"params": head.offset_head.parameters(), "weight_decay": self.optimizer_weight_decay},
                ],
                lr=self.optimizer_lr,
                betas=(0.9, 0.999),
            )
        return torch.optim.AdamW(
            (parameter for parameter in self.parameters() if parameter.requires_grad),
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )
