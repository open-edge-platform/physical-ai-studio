# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lightning policy definition for the native policy design."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from physicalai.data import Feature, FeatureType, Observation
from physicalai.data.dataset import Dataset
from physicalai.data.observation import STATE
from physicalai.policies.mixins import PeftPolicyMixin, RTCPolicyMixin

from .base import TemplatePolicy
from .config import NewPolicyModelConfig, resolve_action_dim
from .export import NewPolicyExportMixin
from .model import NewPolicyModel
from .processor import NewPolicyPostprocessor, NewPolicyPreprocessor, make_policy_processors


class NewPolicy(PeftPolicyMixin, RTCPolicyMixin, NewPolicyExportMixin, TemplatePolicy):  # type: ignore[misc]
    """Orchestrate config resolution, model lifecycle, processing, and training.

    A concrete policy must expose its construction inputs, implement ``setup()`` as
    the data-policy boundary, materialize model and processors only through guarded
    ``configure_model()``, define the runtime/training flow, and configure its
    optimizer. Optional capabilities and export behavior remain in their mixins.
    """
    def __init__(
        self,
        # input and output features are eager init
        input_features: list[Feature] | None = None,
        output_features: list[Feature] | None = None,
        # pretrained checkpoint path or name, if any
        pretrained_name_or_path: str | Path | None = None,
        *,
        # model args
        n_action_steps: int = 32,
        chunk_size: int = 32,
        # weights args
        gradient_checkpointing: bool = False,
        lora_enabled: bool = False,
        rtc_enabled: bool = False,
        # training args
        optimizer_lr: float = 1e-4,
        optimizer_weight_decay: float = 0.01,
    ) -> None:

        # model params
        self._input_features = input_features
        self._output_features = output_features
        self._pretrained_name_or_path = pretrained_name_or_path
        self._n_action_steps = n_action_steps
        self._chunk_size = chunk_size

        # initialize Policy with n_action_steps for action queue
        super().__init__(n_action_steps=n_action_steps)
        self.model: NewPolicyModel | None = None

        # Checkpoints restore the resolved config, never mutable features or artifact locations.
        self.save_hyperparameters(ignore=["input_features", "output_features", "pretrained_name_or_path"])

        # training params
        self.gradient_checkpointing = gradient_checkpointing
        self.lora_enabled = lora_enabled
        self.rtc_enabled = rtc_enabled
        self.optimizer_lr = optimizer_lr
        self.optimizer_weight_decay = optimizer_weight_decay

        self._config: NewPolicyModelConfig | None = None

        # processors
        self._preprocessor: NewPolicyPreprocessor | None = None
        self._postprocessor: NewPolicyPostprocessor | None = None

        if pretrained_name_or_path is not None or (
            input_features is not None and output_features is not None
        ):
            self.configure_model()

    @classmethod
    def from_config(
        cls,
        config: NewPolicyModelConfig,
        *,
        gradient_checkpointing: bool = False,
        optimizer_lr: float = 1e-4,
        optimizer_weight_decay: float = 0.01,
    ) -> "NewPolicy":
        policy = cls(
            pretrained_name_or_path=None,
            n_action_steps=config.n_action_steps,
            gradient_checkpointing=gradient_checkpointing,
            optimizer_lr=optimizer_lr,
            optimizer_weight_decay=optimizer_weight_decay,
        )

        policy._config = config
        policy.configure_model()
        return policy

    def _apply_model_modifications(self) -> None:
        assert isinstance(self.model, NewPolicyModel)

        if self.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        assert self._config is not None
        if self._config.use_lora:
            self._inject_lora()

        self._sync_rtc_to_model()

    @staticmethod
    def _resolve_config_from_hf(
        pretrained_name_or_path: str | Path,
    ) -> tuple[NewPolicyModelConfig, Path]:
        """Fake resolver standing in for downloading and parsing Hugging Face checkpoint artifacts."""
        fake_input_features = [
            Feature(name=STATE, shape=(4,), ftype=FeatureType.STATE),
            Feature(name="front", shape=(3, 16, 16), ftype=FeatureType.VISUAL),
        ]
        fake_output_features = [Feature(name="action", shape=(2,), ftype=FeatureType.ACTION)]
        config = NewPolicyModelConfig(
            input_features=fake_input_features,
            output_features=fake_output_features,
            action_dim=resolve_action_dim(fake_output_features),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            vocab_size=32,
            chunk_size=3,
            image_size=(16, 16),
            tokenizer_max_length=6,
        )
        weights_path = Path(str(pretrained_name_or_path)) / "model.safetensors"
        return config, weights_path

    def configure_model(self) -> None:
        """Create the model once in Lightning's strategy and precision aware context."""
        if self.model is not None:
            return

        # when loading a pretrained checkpoint, keep the checkpoint config but replace only the
        # feature contract and action horizon that are known at policy construction time.
        if self._config is not None:
            config = self._config
            weights_path = None
        elif self._pretrained_name_or_path is not None:
            pretrained_config, weights_path = self._resolve_config_from_hf(self._pretrained_name_or_path)
            resolved_output_features = (
                self._output_features
                if self._output_features is not None
                else pretrained_config.output_features
            )
            config = replace(
                pretrained_config,
                input_features=(
                    self._input_features
                    if self._input_features is not None
                    else pretrained_config.input_features
                ),
                output_features=resolved_output_features,
                action_dim=resolve_action_dim(resolved_output_features),
                n_action_steps=self._n_action_steps,
                lora_enabled=self.lora_enabled,
            )
        else:
            if self._input_features is None or self._output_features is None:
                return

            weights_path = None
            # build the model config from the values already available on the policy; leave the rest
            # of the model defaults to the config/dataclass defaults.
            config = NewPolicyModelConfig(
                input_features=self._input_features,
                output_features=self._output_features,
                action_dim=resolve_action_dim(self._output_features),
                chunk_size=self._chunk_size,
                n_action_steps=self._n_action_steps,
                lora_enabled=self.lora_enabled,
            )

        self._config = config
        self._input_features = config.input_features
        self._output_features = config.output_features
        self._n_action_steps = config.n_action_steps
        self._chunk_size = config.chunk_size
        self.model = NewPolicyModel.from_config(config)
        self._preprocessor, self._postprocessor = make_policy_processors(config)  # type: ignore[assignment]

        if weights_path is not None:
            self.model.load_weights(weights_path)

        self._apply_model_modifications()

    def set_features(
        self,
        input_features: list[Feature],
        output_features: list[Feature],
    ) -> None:
        """Replace the feature contract and rebuild processors without rebuilding the model."""
        if self.model is None or self._config is None:
            raise RuntimeError("Policy model is not initialized")

        action_dim = resolve_action_dim(output_features)
        if action_dim != self._config.action_dim:
            raise ValueError(
                f"Output width {action_dim} does not match model action width {self._config.action_dim}"
            )

        config = replace(
            self._config,
            input_features=list(input_features),
            output_features=list(output_features),
            action_dim=action_dim,
        )
        self._config = config
        self._input_features = config.input_features
        self._output_features = config.output_features
        self._preprocessor, self._postprocessor = make_policy_processors(config)  # type: ignore[assignment]
        self.reset()

    def rename_features(self, mapping: Mapping[str, str]) -> None:
        """Rename resolved input features without changing their metadata or order."""
        if self._config is None:
            raise RuntimeError("Policy config is not initialized")
        if not mapping:
            return
        if any(not isinstance(name, str) or not name for name in mapping):
            raise ValueError("Source feature names must be non-empty strings")
        if any(not isinstance(name, str) or not name for name in mapping.values()):
            raise ValueError("Replacement feature names must be non-empty strings")

        current_names = {feature.name for feature in self._config.input_features}
        unknown_names = sorted(set(mapping) - current_names)
        if unknown_names:
            raise ValueError(f"Cannot rename unknown input features: {unknown_names}")

        input_features = [
            replace(feature, name=mapping[feature.name])
            if feature.name is not None and feature.name in mapping
            else feature
            for feature in self._config.input_features
        ]
        names = [feature.name for feature in input_features]
        if len(names) != len(set(names)):
            raise ValueError(f"Feature renaming creates duplicate input names: {names}")

        self.set_features(input_features, self._config.output_features)

    def _prepare_batch(self, batch: Observation, *, require_actions: bool) -> dict[str, Tensor]:
        if self._preprocessor is None:
            raise RuntimeError("Policy is not initialized")
        processed = self._preprocessor(batch.to_dict())
        if require_actions:
            if not isinstance(batch.action, Tensor):
                raise TypeError("Expected Observation.action to contain action targets")
            processed["action"] = self._preprocessor.normalize_actions(batch.action)
        return processed

    def forward(self, batch: Observation) -> Tensor | tuple[Tensor, dict[str, Tensor | float]]:
        if not isinstance(self.model, NewPolicyModel):
            raise RuntimeError("Policy model is not initialized")
        if self.training:
            return self.model(self._prepare_batch(batch, require_actions=True))
        return self.predict_action_chunk(batch)

    def compute_val_loss(self, batch: Observation) -> tuple[Tensor, dict[str, Tensor | float]]:
        if not isinstance(self.model, NewPolicyModel):
            raise RuntimeError("Policy model is not initialized")
        return self.model.compute_val_loss(self._prepare_batch(batch, require_actions=True))

    def predict_action_chunk(self, batch: Observation) -> Tensor:
        if not isinstance(self.model, NewPolicyModel) or self._postprocessor is None:
            raise RuntimeError("Policy is not initialized")
        actions = cast("Any", self.model).predict_action_chunk(
            self._prepare_batch(batch, require_actions=False)
        )
        return self._postprocessor(actions)

    def training_step(self, batch: Observation, batch_idx: int) -> Tensor:
        del batch_idx
        result = self(batch)
        if not isinstance(result, tuple):
            raise RuntimeError("Training forward must return loss and metrics")
        loss, metrics = result
        self.log("train/loss", metrics["loss"], prog_bar=True)
        return loss

    def setup(self, stage: str) -> None:
        """Set up the model from the training dataset."""
        if stage != "fit":
            return

        datamodule = self.trainer.datamodule  # type: ignore[attr-defined]
        train_dataset = datamodule.train_dataset
        if not isinstance(train_dataset, Dataset):
            raise TypeError(f"Expected physicalai Dataset, got {type(train_dataset)}")

        dataset_input_features = list(train_dataset.observation_features.values())
        dataset_output_features = list(train_dataset.action_features.values())

        if self._config is not None:
            if (
                self._config.input_features != dataset_input_features
                or self._config.output_features != dataset_output_features
            ):
                self.set_features(dataset_input_features, dataset_output_features)
            return

        self._input_features = dataset_input_features
        self._output_features = dataset_output_features

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )
