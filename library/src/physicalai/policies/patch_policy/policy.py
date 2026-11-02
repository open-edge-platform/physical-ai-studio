# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2026 Gaoyue Zhou
# Authors: Gaoyue Zhou, Zichen Jeff Cui
# SPDX-License-Identifier: MIT

"""Lightning policy for Patch Policy."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import IO, TYPE_CHECKING, Any

import torch
from torch import Tensor

from physicalai.data import Dataset, Feature, FeatureType, NormalizationParameters, Observation
from physicalai.policies.base import Model, Policy

from .components.heads import BehaviourTransformerHead
from .config import (
    ActionHeadName,
    EncoderOutput,
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
        encoder_output: Visual representation extracted from the encoder.
        encoder_pretrained: Whether to initialize the encoder with pretrained weights.
        image_size: Encoder input height and width, defaulting to (224, 224).
        action_head: Action-head architecture used to predict action chunks.
        n_obs_steps: Number of observation timesteps used as context.
        chunk_size: Number of action timesteps predicted per chunk.
        n_action_steps: Number of predicted actions executed per invocation.
        hidden_dim: Hidden width of the action head.
        num_layers: Number of transformer layers in the action head.
        num_heads: Number of attention heads in each transformer layer.
        dropout: Dropout probability used by transformer layers.
        num_train_timesteps: Number of diffusion timesteps used during training.
        num_inference_steps: Number of denoising steps used during inference.
        eta: DDIM stochasticity coefficient used during diffusion inference.
        use_random_input_noise: Whether diffusion sampling starts from random noise.
        vqvae_iters: Full passes over unique action windows when fitting a new tokenizer.
        vqvae_batch_size: Tokenizer fitting minibatch size.
        optimizer_lr: AdamW learning rate.
        optimizer_weight_decay: AdamW weight-decay coefficient.
        compile_model: Compile the frozen feature encoder with torch.compile.
    """

    model: Model | None  # pyrefly: ignore[bad-override-mutable-attribute]
    config: PatchPolicyConfig | None
    encoder_name: FeatureEncoderName
    encoder_output: EncoderOutput
    action_head: ActionHeadName
    _preprocessor: PatchPolicyPreprocessor | None  # pyrefly: ignore[bad-override-mutable-attribute]
    _postprocessor: PatchPolicyPostprocessor | None

    def __init__(  # noqa: PLR0913
        self,
        input_features: list[Feature] | None = None,
        output_features: list[Feature] | None = None,
        *,
        encoder_name: FeatureEncoderName = "dinov2",
        encoder_output: EncoderOutput = "patches",
        encoder_pretrained: bool = True,
        image_size: tuple[int, int] = (224, 224),
        action_head: ActionHeadName = "behavior_transformer",
        n_obs_steps: int = 5,
        chunk_size: int = 5,
        n_action_steps: int = 1,
        hidden_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        dropout: float = 0.0,
        embedding_dropout: float = 0.0,
        num_train_timesteps: int = 100,
        num_inference_steps: int = 100,
        eta: float = 1.0,
        use_random_input_noise: bool = True,
        vq_latent_dim: int = 512,
        vq_num_embeddings: int = 16,
        vq_num_groups: int = 2,
        vq_action_scale: float = 500.0,
        vqvae_iters: int = 1000,
        vqvae_batch_size: int = 1024,
        offset_loss_weight: float = 10.0,
        secondary_code_weight: float = 0.5,
        focal_loss_gamma: float = 2.0,
        optimizer_lr: float = 5.5e-5,
        optimizer_weight_decay: float = 2e-4,
        compile_model: bool = False,
    ) -> None:
        """Initialize policy settings and configure the model when features are available."""
        super().__init__(n_action_steps=n_action_steps)

        self.model = None
        self.config = None

        self.input_features = input_features
        self.output_features = output_features

        self._preprocessor = None
        self._postprocessor = None

        # Action-related parameters
        self.n_obs_steps = n_obs_steps
        self.chunk_size = chunk_size
        self.n_action_steps = n_action_steps

        # Model params
        self.encoder_name = encoder_name
        self.encoder_output = encoder_output
        self.encoder_pretrained = encoder_pretrained
        self.image_size = image_size
        self.action_head = action_head
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.embedding_dropout = embedding_dropout
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.eta = eta
        self.use_random_input_noise = use_random_input_noise
        self.vq_latent_dim = vq_latent_dim
        self.vq_num_embeddings = vq_num_embeddings
        self.vq_num_groups = vq_num_groups
        self.vq_action_scale = vq_action_scale
        self.vqvae_iters = vqvae_iters
        self.vqvae_batch_size = vqvae_batch_size
        self.offset_loss_weight = offset_loss_weight
        self.secondary_code_weight = secondary_code_weight
        self.focal_loss_gamma = focal_loss_gamma

        # Extra params for training
        self.optimizer_lr = optimizer_lr
        self.optimizer_weight_decay = optimizer_weight_decay
        self.compile_model = compile_model
        self._encoder_compiled = False

        # Don't save possibly unresolved input / output features
        self.save_hyperparameters(ignore=["input_features", "output_features", "compile_model"])

        if self.input_features is not None and self.output_features is not None:
            self.configure_model()

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
        input_features = dataset_input_features if self.input_features is None else self.input_features
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
    ) -> PatchPolicy:
        """Build a policy from a resolved configuration.

        Returns:
            Configured policy with its model and processors initialized.
        """
        policy = cls(
            input_features=None,
            output_features=None,
            encoder_name=config.encoder_name,
            encoder_output=config.encoder_output,
            encoder_pretrained=config.encoder_pretrained,
            image_size=config.image_size,
            action_head=config.action_head,
            n_obs_steps=config.n_obs_steps,
            chunk_size=config.chunk_size,
            n_action_steps=config.n_action_steps,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout=config.dropout,
            embedding_dropout=config.embedding_dropout,
            num_train_timesteps=config.num_train_timesteps,
            num_inference_steps=config.num_inference_steps,
            eta=config.eta,
            use_random_input_noise=config.use_random_input_noise,
            vq_latent_dim=config.vq_latent_dim,
            vq_num_embeddings=config.vq_num_embeddings,
            vq_num_groups=config.vq_num_groups,
            vq_action_scale=config.vq_action_scale,
            vqvae_iters=config.vqvae_iters,
            vqvae_batch_size=config.vqvae_batch_size,
            offset_loss_weight=config.offset_loss_weight,
            secondary_code_weight=config.secondary_code_weight,
            focal_loss_gamma=config.focal_loss_gamma,
            optimizer_lr=optimizer_lr,
            optimizer_weight_decay=optimizer_weight_decay,
            compile_model=compile_model,
        )
        policy._initialize_from_checkpoint_config(config)
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
        return super().load_from_checkpoint(
            checkpoint_path,
            map_location=map_location,
            hparams_file=hparams_file,
            strict=strict,
            weights_only=weights_only,
            **kwargs,
        )

    def configure_model(self) -> None:
        """Create the model once and synchronize feature normalization and processors.

        Raises:
            TypeError: If an existing model is not a PatchPolicyModel.
        """
        # config requires input and output features
        if self.input_features is None or self.output_features is None:
            return

        # if no config, construct (for the first time)
        if self.config is None:
            # construct config
            self.config = PatchPolicyConfig(
                input_features=list(self.input_features),
                output_features=list(self.output_features),
                action_dim=resolve_action_dim(self.output_features),
                encoder_name=self.encoder_name,
                encoder_output=self.encoder_output,
                encoder_pretrained=self.encoder_pretrained,
                image_size=self.image_size,
                action_head=self.action_head,
                n_obs_steps=self.n_obs_steps,
                chunk_size=self.chunk_size,
                n_action_steps=self.n_action_steps,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                dropout=self.dropout,
                embedding_dropout=self.embedding_dropout,
                num_train_timesteps=self.num_train_timesteps,
                num_inference_steps=self.num_inference_steps,
                eta=self.eta,
                use_random_input_noise=self.use_random_input_noise,
                vq_latent_dim=self.vq_latent_dim,
                vq_num_embeddings=self.vq_num_embeddings,
                vq_num_groups=self.vq_num_groups,
                vq_action_scale=self.vq_action_scale,
                vqvae_iters=self.vqvae_iters,
                vqvae_batch_size=self.vqvae_batch_size,
                offset_loss_weight=self.offset_loss_weight,
                secondary_code_weight=self.secondary_code_weight,
                focal_loss_gamma=self.focal_loss_gamma,
            )

        # Init model from local config
        config = self.config

        self.encoder_name = config.encoder_name
        self.encoder_output = config.encoder_output
        self.encoder_pretrained = config.encoder_pretrained
        self.image_size = config.image_size
        self.action_head = config.action_head
        self.n_obs_steps = config.n_obs_steps
        self.chunk_size = config.chunk_size
        self.n_action_steps = config.n_action_steps
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.dropout = config.dropout
        self.embedding_dropout = config.embedding_dropout
        self.num_train_timesteps = config.num_train_timesteps
        self.num_inference_steps = config.num_inference_steps
        self.eta = config.eta
        self.use_random_input_noise = config.use_random_input_noise
        self.vq_latent_dim = config.vq_latent_dim
        self.vq_num_embeddings = config.vq_num_embeddings
        self.vq_num_groups = config.vq_num_groups
        self.vq_action_scale = config.vq_action_scale
        self.vqvae_iters = config.vqvae_iters
        self.vqvae_batch_size = config.vqvae_batch_size
        self.offset_loss_weight = config.offset_loss_weight
        self.secondary_code_weight = config.secondary_code_weight
        self.focal_loss_gamma = config.focal_loss_gamma
        self.hparams.update(
            encoder_name=config.encoder_name,
            encoder_output=config.encoder_output,
            encoder_pretrained=config.encoder_pretrained,
            image_size=config.image_size,
            action_head=config.action_head,
            n_obs_steps=config.n_obs_steps,
            chunk_size=config.chunk_size,
            n_action_steps=config.n_action_steps,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout=config.dropout,
            embedding_dropout=config.embedding_dropout,
            num_train_timesteps=config.num_train_timesteps,
            num_inference_steps=config.num_inference_steps,
            eta=config.eta,
            use_random_input_noise=config.use_random_input_noise,
            vq_latent_dim=config.vq_latent_dim,
            vq_num_embeddings=config.vq_num_embeddings,
            vq_num_groups=config.vq_num_groups,
            vq_action_scale=config.vq_action_scale,
            vqvae_iters=config.vqvae_iters,
            vqvae_batch_size=config.vqvae_batch_size,
            offset_loss_weight=config.offset_loss_weight,
            secondary_code_weight=config.secondary_code_weight,
            focal_loss_gamma=config.focal_loss_gamma,
        )

        model = self.model
        if model is None:
            model = PatchPolicyModel.from_config(config)
        elif not isinstance(model, PatchPolicyModel):
            msg = "Patch Policy requires a PatchPolicyModel instance."
            raise TypeError(msg)

        # Replace input features with normalized versions based on encoder statistics
        # Normalization is only used by pre / post processor
        config = replace(
            config,
            input_features=_attach_encoder_normalization(
                self.input_features,
                model.encoder.mean,
                model.encoder.std,
            ),
            output_features=self.output_features,
        )

        self.input_features = config.input_features
        self.output_features = config.output_features
        if (
            self.model is model
            and self.config == config
            and self._preprocessor is not None
            and self._postprocessor is not None
        ):
            return

        # Store the final configuration and model instance in Policy
        self.config = config
        self.input_features = config.input_features
        self.model = model
        self._preprocessor, self._postprocessor = make_policy_processors(config, model.encoder.image_size)

        if self.compile_model and not self._encoder_compiled:
            model.encoder.compile()
            self._encoder_compiled = True

    def _initialize_from_checkpoint_config(self, config: PatchPolicyConfig) -> None:
        if self.model is not None:
            if (
                self.config is None
                or replace(
                    config,
                    input_features=self.config.input_features,
                    output_features=self.config.output_features,
                )
                != self.config
                or [(_feature_key(feature), feature.shape) for feature in config.input_features]
                != [(_feature_key(feature), feature.shape) for feature in self.config.input_features]
                or [(_feature_key(feature), feature.shape) for feature in config.output_features]
                != [(_feature_key(feature), feature.shape) for feature in self.config.output_features]
            ):
                msg = "Checkpoint policy config does not match the initialized policy."
                raise ValueError(msg)
        else:
            self.config = config
        self.input_features = config.input_features
        self.output_features = config.output_features
        self.configure_model()

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Store the resolved policy configuration in a checkpoint.

        Raises:
            RuntimeError: If the policy has not been configured.
        """
        if self.config is None:
            msg = "Cannot save a Patch Policy checkpoint before the policy is configured."
            raise RuntimeError(msg)
        checkpoint["policy_config"] = self.config.to_dict()

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Initialize the policy from its checkpoint configuration.

        Raises:
            TypeError: If the checkpoint lacks a valid policy configuration.
        """
        config_data = checkpoint.get("policy_config")
        if not isinstance(config_data, Mapping):
            msg = "Patch Policy checkpoint is missing a valid policy_config."
            raise TypeError(msg)
        self._initialize_from_checkpoint_config(PatchPolicyConfig.from_dict(config_data))

    def _prepare_batch(
        self,
        batch: Observation,
        *,
        require_actions: bool,
    ) -> dict[str, Any]:
        if self.config is None or self._preprocessor is None:
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
            decay = {}
            no_decay = {}
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
