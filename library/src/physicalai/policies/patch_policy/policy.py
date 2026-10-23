# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lightning policy for Patch Policy."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import IO, TYPE_CHECKING, Any, cast

import torch
from torch import Tensor

from physicalai.data import Dataset, Feature, Observation
from physicalai.export import ExportablePolicyMixin
from physicalai.policies.base import Model, Policy

from .config import (
    ActionHeadName,
    EncoderOutput,
    FeatureEncoderName,
    PatchPolicyConfig,
    resolve_action_dim,
)
from .export import PatchPolicyExportMixin
from .model import PatchPolicyModel
from .processors import PatchPolicyPostprocessor, PatchPolicyPreprocessor, make_policy_processors

if TYPE_CHECKING:
    from pathlib import Path


class PatchPolicy(PatchPolicyExportMixin, ExportablePolicyMixin, Policy):
    """Train a timm visual encoder followed by a diffusion or Behavior Transformer head.

    Export supports Torch and OpenVINO with the configured observation window.

    Args:
        input_features: Optional observation schema. Resolved from training data when omitted.
        output_features: Optional action schema. Resolved from training data when omitted.
        encoder_name: Name of the timm visual encoder architecture.
        encoder_output: Visual representation extracted from the encoder.
        encoder_pretrained: Whether to initialize the encoder with pretrained weights.
        freeze_encoder: Whether to prevent updates to encoder parameters.
        image_size: Input image height and width.
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
        optimizer_lr: AdamW learning rate.
        optimizer_weight_decay: AdamW weight-decay coefficient.
    """

    model: Model | None  # pyrefly: ignore[bad-override-mutable-attribute]

    def __init__(  # noqa: PLR0913
        self,
        input_features: list[Feature] | None = None,
        output_features: list[Feature] | None = None,
        *,
        encoder_name: FeatureEncoderName = "dinov2",
        encoder_output: EncoderOutput = "patches",
        encoder_pretrained: bool = True,
        freeze_encoder: bool = True,
        image_size: tuple[int, int] = (224, 224),
        action_head: ActionHeadName = "diffusion",
        n_obs_steps: int = 1,
        chunk_size: int = 16,
        n_action_steps: int = 16,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_train_timesteps: int = 100,
        num_inference_steps: int = 100,
        eta: float = 1.0,
        use_random_input_noise: bool = True,
        optimizer_lr: float = 1e-4,
        optimizer_weight_decay: float = 0.0,
    ) -> None:
        super().__init__(n_action_steps=n_action_steps)
        self.save_hyperparameters(ignore=["input_features", "output_features"])
        self.optimizer_lr = optimizer_lr
        self.optimizer_weight_decay = optimizer_weight_decay
        self._preprocessor: PatchPolicyPreprocessor | None = None  # pyrefly: ignore[bad-override-mutable-attribute]
        self._postprocessor: PatchPolicyPostprocessor | None = None
        self._input_features = input_features
        self._output_features = output_features
        self._config: PatchPolicyConfig | None = None
        self._encoder_name = encoder_name
        self._encoder_output = encoder_output
        self._encoder_pretrained = encoder_pretrained
        self._freeze_encoder = freeze_encoder
        self._image_size = image_size
        self._action_head = action_head
        self._n_obs_steps = n_obs_steps
        self._chunk_size = chunk_size
        self._n_action_steps = n_action_steps
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._dropout = dropout
        self._num_train_timesteps = num_train_timesteps
        self._num_inference_steps = num_inference_steps
        self._eta = eta
        self._use_random_input_noise = use_random_input_noise
        if input_features is not None and output_features is not None:
            self.configure_model()

    @classmethod
    def from_config(
        cls,
        config: PatchPolicyConfig,
        *,
        optimizer_lr: float = 1e-4,
        optimizer_weight_decay: float = 0.0,
    ) -> PatchPolicy:
        """Construct a policy from a complete serialized config.

        Args:
            config: Resolved Patch Policy configuration.
            optimizer_lr: AdamW learning rate.
            optimizer_weight_decay: AdamW weight-decay coefficient.

        Returns:
            An initialized Patch Policy.
        """
        policy = cls(
            input_features=None,
            output_features=None,
            encoder_name=config.encoder_name,
            encoder_output=config.encoder_output,
            encoder_pretrained=config.encoder_pretrained,
            freeze_encoder=config.freeze_encoder,
            image_size=config.image_size,
            action_head=config.action_head,
            n_obs_steps=config.n_obs_steps,
            chunk_size=config.chunk_size,
            n_action_steps=config.n_action_steps,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout=config.dropout,
            num_train_timesteps=config.num_train_timesteps,
            num_inference_steps=config.num_inference_steps,
            eta=config.eta,
            use_random_input_noise=config.use_random_input_noise,
            optimizer_lr=optimizer_lr,
            optimizer_weight_decay=optimizer_weight_decay,
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
    ) -> PatchPolicy:
        """Load a checkpoint whose saved config reconstructs the model before tensor restoration.

        Args:
            checkpoint_path: Checkpoint path or binary file object.
            map_location: Device mapping passed to Lightning checkpoint loading.
            hparams_file: Optional path to a separate hyperparameter file.
            strict: Whether checkpoint keys must exactly match the model state.
            weights_only: Whether to restrict deserialization to tensor weights and safe types.
            **kwargs: Additional constructor arguments forwarded by Lightning.

        Returns:
            The restored Patch Policy.
        """
        return super().load_from_checkpoint(
            checkpoint_path,
            map_location=map_location,
            hparams_file=hparams_file,
            strict=strict,
            weights_only=weights_only,
            **kwargs,
        )

    @property
    def config(self) -> PatchPolicyConfig:
        """Resolved policy configuration.

        Returns:
            The resolved Patch Policy configuration.

        Raises:
            RuntimeError: If feature metadata has not been resolved yet.
        """
        if self._config is None:
            raise RuntimeError("Patch Policy config is not resolved yet.")
        return self._config

    def _initialize_from_config(self, config: PatchPolicyConfig) -> None:
        """Resolve policy state and build modules from a configuration.

        Args:
            config: Complete Patch Policy configuration.

        Raises:
            ValueError: If an initialized policy has a different configuration.
        """
        if self.model is not None:
            if self.config != config:
                raise ValueError("Checkpoint policy config does not match the initialized policy.")
            return
        self._config = config
        self._input_features = config.input_features
        self._output_features = config.output_features
        self.configure_model()

    def configure_model(self) -> None:
        """Create the model and processors exactly once."""
        if self.model is not None:
            return
        if self._config is None:
            if self._input_features is None or self._output_features is None:
                return
            self._config = PatchPolicyConfig(
                input_features=list(self._input_features),
                output_features=list(self._output_features),
                action_dim=resolve_action_dim(self._output_features),
                encoder_name=self._encoder_name,
                encoder_output=self._encoder_output,
                encoder_pretrained=self._encoder_pretrained,
                freeze_encoder=self._freeze_encoder,
                image_size=self._image_size,
                action_head=self._action_head,
                n_obs_steps=self._n_obs_steps,
                chunk_size=self._chunk_size,
                n_action_steps=self._n_action_steps,
                hidden_dim=self._hidden_dim,
                num_layers=self._num_layers,
                num_heads=self._num_heads,
                dropout=self._dropout,
                num_train_timesteps=self._num_train_timesteps,
                num_inference_steps=self._num_inference_steps,
                eta=self._eta,
                use_random_input_noise=self._use_random_input_noise,
            )
        config = self._config
        self.model = PatchPolicyModel.from_config(config)
        self._preprocessor, self._postprocessor = make_policy_processors(config)

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Save the complete resolved config needed to reconstruct the model.

        Args:
            checkpoint: Mutable Lightning checkpoint payload.
        """
        checkpoint["policy_config"] = self.config.to_dict()

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Reconstruct the model before Lightning restores checkpoint tensors.

        Args:
            checkpoint: Lightning checkpoint payload.

        Raises:
            TypeError: If the checkpoint does not contain a valid policy configuration.
        """
        config_data = checkpoint.get("policy_config")
        if not isinstance(config_data, Mapping):
            raise TypeError("Patch Policy checkpoint is missing a valid policy_config.")
        self._initialize_from_config(PatchPolicyConfig.from_dict(config_data))

    def _prepare_batch(self, batch: Observation, *, require_actions: bool) -> dict[str, Tensor]:
        """Flatten action features and preprocess an observation.

        Args:
            batch: Structured policy observation.
            require_actions: Whether actions must be included for loss computation.

        Returns:
            A preprocessed model batch.

        Raises:
            RuntimeError: If the policy processors have not been initialized.
        """
        if self._preprocessor is None:
            raise RuntimeError("Patch Policy is not initialized.")
        values = batch.to_dict(flatten=False)
        if isinstance(values.get("action"), Mapping):
            action_map = values["action"]
            values["action"] = torch.cat(
                [action_map[feature.name] for feature in self.config.output_features if feature.name is not None],
                dim=-1,
            )
        return self._preprocessor(values, require_actions=require_actions)

    def forward(self, batch: Observation) -> Tensor | tuple[Tensor, dict[str, Tensor | float]]:
        """Compute training loss or predict actions according to module mode.

        Args:
            batch: Structured policy observation.

        Returns:
            Training loss and metrics in training mode, otherwise an action chunk.

        Raises:
            RuntimeError: If the policy model has not been initialized.
        """
        if self.model is None:
            raise RuntimeError("Patch Policy is not initialized.")
        if self.training:
            return self.model(self._prepare_batch(batch, require_actions=True))
        return self.predict_action_chunk(batch)

    def predict_action_chunk(self, batch: Observation) -> Tensor:
        """Predict and postprocess one action chunk.

        Args:
            batch: Structured policy observation.

        Returns:
            A denormalized action chunk limited to the execution horizon.

        Raises:
            RuntimeError: If the policy model or processors have not been initialized.
        """
        if self.model is None or self._postprocessor is None:
            raise RuntimeError("Patch Policy is not initialized.")
        model = cast("PatchPolicyModel", self.model)
        return self._postprocessor(model.predict_action_chunk(self._prepare_batch(batch, require_actions=False)))

    def compute_val_loss(self, batch: Observation) -> tuple[Tensor, dict[str, Tensor | float]]:
        """Compute validation loss for an observation batch.

        Args:
            batch: Structured observation containing target actions.

        Returns:
            The validation loss and logging metrics.

        Raises:
            RuntimeError: If the policy model has not been initialized.
        """
        if self.model is None:
            raise RuntimeError("Patch Policy is not initialized.")
        return self.model.compute_val_loss(self._prepare_batch(batch, require_actions=True))

    def training_step(self, batch: Observation, batch_idx: int) -> Tensor:
        """Run and log one Lightning training step.

        Args:
            batch: Structured training observation.
            batch_idx: Batch index supplied by Lightning.

        Returns:
            The scalar training loss.

        Raises:
            TypeError: If training forward does not return a loss and metrics.
        """
        del batch_idx
        result = self(batch)
        if not isinstance(result, tuple):
            raise TypeError("Training forward must return loss and metrics.")
        loss, metrics = result
        self.log_dict({f"train/{name}": value for name, value in metrics.items()}, prog_bar=True)
        return loss

    def setup(self, stage: str) -> None:
        """Resolve feature metadata from the training dataset when needed.

        Args:
            stage: Lightning lifecycle stage.

        Raises:
            TypeError: If the training dataset is not a Physical AI dataset.
        """
        if stage != "fit" or self.model is not None:
            return
        datamodule = self.trainer.datamodule  # pyrefly: ignore[missing-attribute]
        dataset = datamodule.train_dataset
        if not isinstance(dataset, Dataset):
            raise TypeError(f"Expected physicalai Dataset, got {type(dataset)}.")
        self._input_features = list(dataset.observation_features.values())
        self._output_features = list(dataset.action_features.values())
        self.configure_model()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Create the optimizer for trainable policy parameters.

        Returns:
            An AdamW optimizer configured for the policy.
        """
        return torch.optim.AdamW(
            (parameter for parameter in self.parameters() if parameter.requires_grad),
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )
