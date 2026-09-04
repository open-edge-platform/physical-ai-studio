# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Executable reference implementation for the native policy design."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from math import prod
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn

from physicalai.config import Config, FromConfig
from physicalai.data import DataModule, Feature, FeatureType, NormalizationParameters, Observation
from physicalai.data.dataset import Dataset
from physicalai.data.observation import ACTION, IMAGES, STATE, TASK
from physicalai.export import ExportablePolicyMixin, ExportBackend
from physicalai.export.backends import (
    ExecuTorchExportParameters,
    ExportParameters,
    ONNXExportParameters,
    OpenVINOExportParameters,
    TorchExportParameters,
)
from physicalai.inference.data import InferenceFeature, InferenceFeatureDtype, InferenceFeatureType
from physicalai.inference.manifest import ComponentSpec
from physicalai.policies.base import Model, Policy
from physicalai.train import Trainer


class NewPolicy(ExportablePolicyMixin, Policy):  # type: ignore[misc]
    """Policy design, fake transormer-based model, and training loop for demonstration purposes."""
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
        use_lora: bool = False,
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

        # ignore input/output features in hyperparameters, as they are subject to change
        self.save_hyperparameters(ignore=["input_features", "output_features", "pretrained_name_or_path"])

        # training params
        self.gradient_checkpointing = gradient_checkpointing
        self.use_lora = use_lora
        self.optimizer_lr = optimizer_lr
        self.optimizer_weight_decay = optimizer_weight_decay

        # model
        self._model: NewPolicyModel | None = None

        # processors
        self._preprocessor: NewPolicyPreprocessor | None = None
        self._postprocessor: NewPolicyPostprocessor | None = None

        # only eager init if features supplied
        if input_features is not None and output_features is not None:
            self.initialize_model()

    @classmethod
    def from_config(
        cls,
        config: NewPolicyModelConfig,
        *,
        gradient_checkpointing: bool = False,
        use_lora: bool = False,
        optimizer_lr: float = 1e-4,
        optimizer_weight_decay: float = 0.01,
    ) -> "NewPolicy":
        policy = cls(
            # only pass policy and training args, which prevents eager initialization
            n_action_steps=config.n_action_steps,
            gradient_checkpointing=gradient_checkpointing,
            use_lora=use_lora,
            optimizer_lr=optimizer_lr,
            optimizer_weight_decay=optimizer_weight_decay,
        )
    
        policy._initialize_from_config(config)
        return policy

    def _require_model(self) -> NewPolicyModel:
        if not isinstance(self._model, NewPolicyModel):
            raise RuntimeError("Policy model is not initialized")
        return self._model

    def _initialize_from_config(
        self,
        config: NewPolicyModelConfig,
        *,
        weights_path: Path | None = None,
    ) -> None:
        if self._model is not None:
            raise RuntimeError("Policy model is already initialized")

        self._input_features = config.input_features
        self._output_features = config.output_features
        self._n_action_steps = config.n_action_steps
        self._chunk_size = config.chunk_size
        self._model = NewPolicyModel.from_config(config)
        self._preprocessor, self._postprocessor = make_policy_processors(config)  # type: ignore[assignment]

        if weights_path is not None:
            self._model.load_weights(weights_path)

        self._apply_model_modifications()

    def _apply_model_modifications(self) -> None:
        model = self._require_model()

        if self.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        if self.use_lora:
            model.enable_lora()

    def _from_hf(
        self,
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

    def initialize_model(self) -> None:
        # when loading a pretrained checkpoint, keep the checkpoint config but replace only the
        # feature contract and action horizon that are known at policy construction time.
        if self._pretrained_name_or_path is not None:
            pretrained_config, weights_path = self._from_hf(self._pretrained_name_or_path)
            config = replace(
                pretrained_config,
                input_features=(
                    self._input_features
                    if self._input_features is not None
                    else pretrained_config.input_features
                ),
                output_features=(
                    self._output_features
                    if self._output_features is not None
                    else pretrained_config.output_features
                ),
                n_action_steps=self._n_action_steps,
            )
        else:
            if self._input_features is None or self._output_features is None:
                raise RuntimeError("Input and output features are required to initialize the model")

            weights_path = None
            # build the model config from the values already available on the policy; leave the rest
            # of the model defaults to the config/dataclass defaults.
            config = NewPolicyModelConfig(
                input_features=self._input_features,
                output_features=self._output_features,
                chunk_size=self._chunk_size,
                n_action_steps=self._n_action_steps,
            )

        self._initialize_from_config(config, weights_path=weights_path)

    def _model_config_for_checkpoint(self) -> dict[str, Any]:
        return self._require_model().config.to_dict()

    def _restore_model_config(self, config_data: Mapping[str, Any]) -> None:
        resolved_config = NewPolicyModelConfig.from_dict(config_data)
        if self._model is not None:
            if self._require_model().config != resolved_config:
                raise ValueError("Checkpoint feature contract does not match the initialized policy")
            return

        self._initialize_from_config(resolved_config)

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["model_config"] = self._model_config_for_checkpoint()

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        config_data = checkpoint.get("model_config")
        if isinstance(config_data, Mapping):
            self._restore_model_config(config_data)

    @staticmethod
    def _dataset_features(dataset: Dataset) -> tuple[list[Feature], list[Feature]]:
        return (
            list(dataset.observation_features.values()),
            list(dataset.action_features.values()),
        )

    def setup(self, stage: str) -> None:
        """Set up the model from the training dataset."""
        if stage != "fit":
            return

        datamodule = self.trainer.datamodule  # type: ignore[attr-defined]
        train_dataset = datamodule.train_dataset
        if not isinstance(train_dataset, Dataset):
            raise TypeError(f"Expected physicalai Dataset, got {type(train_dataset)}")

        dataset_input_features, dataset_output_features = self._dataset_features(train_dataset)

        if self._model is not None:
            config = self._require_model().config
            if (
                config.input_features != dataset_input_features
                or config.output_features != dataset_output_features
            ):
                raise ValueError(
                    "Eager policy features do not match the training dataset; "
                    "construct the policy lazily to replace pretrained features during setup"
                )
            return

        self._input_features = dataset_input_features
        self._output_features = dataset_output_features
        self.initialize_model()

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
        model = self._require_model()
        if self.training:
            return model(self._prepare_batch(batch, require_actions=True))
        return self.predict_action_chunk(batch)

    def compute_val_loss(self, batch: Observation) -> tuple[Tensor, dict[str, Tensor | float]]:
        model = self._require_model()
        return model.compute_val_loss(self._prepare_batch(batch, require_actions=True))

    def predict_action_chunk(self, batch: Observation) -> Tensor:
        if self._postprocessor is None:
            raise RuntimeError("Policy is not initialized")
        model = self._require_model()
        actions = model.predict_action_chunk(self._prepare_batch(batch, require_actions=False))
        return self._postprocessor(actions)

    def training_step(self, batch: Observation, batch_idx: int) -> Tensor:
        del batch_idx
        result = self(batch)
        if not isinstance(result, tuple):
            raise RuntimeError("Training forward must return loss and metrics")
        loss, metrics = result
        self.log("train/loss", metrics["loss"], prog_bar=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    @property
    def inputs_schema(self) -> list[InferenceFeature] | None:
        """Describe ACT-style state/image inputs with appended language metadata."""
        if self._model is None:
            return None

        config = self._require_model().config
        state_features = [feature for feature in config.input_features if feature.ftype == FeatureType.STATE]
        if len(state_features) != 1 or state_features[0].shape is None:
            raise ValueError("Export requires exactly one state feature with a concrete shape")

        schema = [
            InferenceFeature(
                ftype=InferenceFeatureType.STATE,
                shape=tuple(state_features[0].shape),
                name=STATE,
                dtype=InferenceFeatureDtype.FLOAT32,
            )
        ]

        image_features = [feature for feature in config.input_features if feature.ftype == FeatureType.VISUAL]
        for feature in image_features:
            if feature.name is None or feature.shape is None:
                raise ValueError("Export image features must define a name and shape")
            image_name = IMAGES if len(image_features) == 1 else f"{IMAGES}.{feature.name}"
            schema.append(
                InferenceFeature(
                    ftype=InferenceFeatureType.VISUAL,
                    shape=tuple(feature.shape),
                    name=image_name,
                    dtype=InferenceFeatureDtype.FLOAT32,
                )
            )

        schema.append(
            InferenceFeature(
                ftype=InferenceFeatureType.LANGUAGE,
                shape=(config.tokenizer_max_length,),
                name=TASK,
                dtype=InferenceFeatureDtype.STRING,
            )
        )
        return schema

    @property
    def outputs_schema(self) -> list[InferenceFeature] | None:
        """Describe the ACT-style action chunk produced by the model."""
        if self._model is None:
            return None

        config = self._require_model().config
        if len(config.output_features) != 1 or config.output_features[0].shape is None:
            raise ValueError("Export requires exactly one action feature with a concrete shape")
        action_feature = config.output_features[0]
        action_shape = action_feature.shape
        if action_shape is None:
            raise ValueError("Export action feature must define a concrete shape")
        return [
            InferenceFeature(
                ftype=InferenceFeatureType.ACTION,
                shape=(config.chunk_size, *action_shape),
                name=ACTION,
                dtype=InferenceFeatureDtype.FLOAT32,
            )
        ]

    @property
    def extra_export_args(self) -> dict[str, ExportParameters]:
        """Build ACT-style backend parameters from the resolved model config."""
        config = self._require_model().config
        output_names = [feature.name for feature in (self.outputs_schema or [])]
        postprocessors: list[ComponentSpec] = []
        if config.chunk_size != config.n_action_steps:
            postprocessors.append(
                ComponentSpec.model_validate(
                    {
                        "type": "action_chunk_trimmer",
                        "n_action_steps": config.n_action_steps,
                    }
                )
            )

        preprocessors = [
            ComponentSpec.model_validate(
                {
                    "type": "resize",
                    "image_resolution": config.image_size,
                    "mode": "letterbox",
                }
            )
        ]
        return {
            "onnx": ONNXExportParameters(
                exporter_kwargs={"output_names": output_names},
                preprocessors_specs=preprocessors,
                postprocessors_specs=postprocessors,
            ),
            "openvino": OpenVINOExportParameters(
                outputs=output_names,
                export_tokenizer=False,
                compress_to_fp16=True,
                exporter_kwargs={},
                preprocessors_specs=preprocessors,
                postprocessors_specs=postprocessors,
            ),
            "executorch": ExecuTorchExportParameters(
                preprocessors_specs=preprocessors,
                postprocessors_specs=postprocessors,
            ),
            "torch": TorchExportParameters(
                preprocessors_specs=[ComponentSpec(type="to_float_tensor")],
                postprocessors_specs=postprocessors,
            ),
        }

    @staticmethod
    def get_supported_export_backends() -> list[str | ExportBackend]:
        return [
            ExportBackend.TORCH,
            ExportBackend.OPENVINO,
            ExportBackend.ONNX,
            ExportBackend.EXECUTORCH,
        ]


@dataclass
class NewPolicyModelConfig(Config):
    """Configuration for the NewPolicyModel.

    All config options only relate to the model. 
    """
    input_features: list[Feature]
    output_features: list[Feature]
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_hidden_layers: int = 18
    num_attention_heads: int = 8
    attention_dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    vocab_size: int = 256_000
    chunk_size: int = 32
    n_action_steps: int = 32
    image_size: tuple[int, int] = (224, 224)
    tokenizer_max_length: int = 48
    lora_config: dict[str, int | float] = field(
        default_factory=lambda: {"rank": 64, "alpha": 16, "dropout": 0.05}
    )


class NewPolicyPreprocessor(nn.Module):
    """Preprocessor - handles normlaization from feature and defaults if None"""
    def __init__(self, input_features: list[Feature], output_features: list[Feature]) -> None:
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features

    @staticmethod
    def _normalize(value: Tensor, feature: Feature) -> Tensor:
        normalization = feature.normalization_data
        mean = 0.0 if normalization is None or normalization.mean is None else normalization.mean
        std = 1.0 if normalization is None or normalization.std is None else normalization.std
        mean_tensor = torch.as_tensor(mean, dtype=value.dtype, device=value.device)
        std_tensor = torch.as_tensor(std, dtype=value.dtype, device=value.device)
        return (value - mean_tensor) / (std_tensor + 1e-8)

    def normalize_actions(self, actions: Tensor) -> Tensor:
        normalized_actions = []
        offset = 0
        for feature in self.output_features:
            if feature.shape is None:
                raise ValueError(f"Output feature {feature.name!r} must define a shape")

            feature_size = prod(feature.shape)
            feature_actions = actions[..., offset : offset + feature_size]
            normalized_actions.append(self._normalize(feature_actions, feature))
            offset += feature_size

        if actions.shape[-1] != offset:
            raise ValueError(
                f"Action width {actions.shape[-1]} does not match configured output width {offset}"
            )
        return torch.cat(normalized_actions, dim=-1)

    def forward(self, batch: Mapping[str, Any]) -> dict[str, Tensor]:
        task = batch.get("task")
        if not isinstance(task, Tensor):
            raise TypeError("Expected Observation.task to contain token IDs")

        processed = {"input_ids": task.long()}
        for feature in self.input_features:
            if feature.name is None:
                continue
            value = batch.get(feature.name)
            if isinstance(value, Tensor) and value.is_floating_point():
                processed[feature.name] = self._normalize(value, feature)
        return processed


class NewPolicyPostprocessor(nn.Module):
    """Postprocessor - handles denormalization from feature and defaults if None"""
    def __init__(self, output_features: list[Feature]) -> None:
        super().__init__()
        self.output_features = output_features

    def forward(self, actions: Tensor) -> Tensor:
        processed_actions = []
        offset = 0
        for feature in self.output_features:
            if feature.shape is None:
                raise ValueError(f"Output feature {feature.name!r} must define a shape")

            feature_size = prod(feature.shape)
            feature_actions = actions[..., offset : offset + feature_size]
            normalization = feature.normalization_data
            mean = 0.0 if normalization is None or normalization.mean is None else normalization.mean
            std = 1.0 if normalization is None or normalization.std is None else normalization.std
            mean_tensor = torch.as_tensor(mean, dtype=actions.dtype, device=actions.device)
            std_tensor = torch.as_tensor(std, dtype=actions.dtype, device=actions.device)
            processed_actions.append(feature_actions * std_tensor + mean_tensor)
            offset += feature_size

        if actions.shape[-1] != offset:
            raise ValueError(
                f"Action width {actions.shape[-1]} does not match configured output width {offset}"
            )
        return torch.cat(processed_actions, dim=-1)


def make_policy_processors(
    config: NewPolicyModelConfig,
) -> tuple[NewPolicyPreprocessor, NewPolicyPostprocessor]:
    return (
        NewPolicyPreprocessor(config.input_features, config.output_features),
        NewPolicyPostprocessor(config.output_features),
    )


class TextModule(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.embedding(input_ids)


class TransformerModule(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        attention_dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_attention_heads,
                dim_feedforward=intermediate_size,
                dropout=attention_dropout,
                layer_norm_eps=layer_norm_eps,
                batch_first=True,
            )
            for _ in range(num_hidden_layers)
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class NewPolicyModel(Model, FromConfig):
    def __init__(
        self,
        input_features: list[Feature],
        output_features: list[Feature],
        *,
        hidden_size: int = 1024,
        intermediate_size: int = 4096,
        num_hidden_layers: int = 18,
        num_attention_heads: int = 8,
        attention_dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        vocab_size: int = 256_000,
        chunk_size: int = 32,
        n_action_steps: int = 32,
        image_size: tuple[int, int] = (224, 224),
        tokenizer_max_length: int = 48,
        lora_config: dict[str, int | float] | None = None,
    ) -> None:
        super().__init__()
        # resolve un-serializable args 
        resolved_lora_config = lora_config or {"rank": 64, "alpha": 16, "dropout": 0.05}

        # set the model config and build the model components
        self._config = NewPolicyModelConfig(
            input_features=input_features,
            output_features=output_features,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            layer_norm_eps=layer_norm_eps,
            vocab_size=vocab_size,
            chunk_size=chunk_size,
            n_action_steps=n_action_steps,
            image_size=image_size,
            tokenizer_max_length=tokenizer_max_length,
            lora_config=resolved_lora_config,
        )
        self.text_model = TextModule(vocab_size=vocab_size, hidden_size=hidden_size)
        self.transformer = TransformerModule(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            layer_norm_eps=layer_norm_eps,
        )
        self.action_head = nn.Linear(hidden_size, self._action_dim(output_features) * chunk_size)
        self.gradient_checkpointing_enabled = False
        self.lora_config: tuple[int, int, float] | None = None
        self.weights_load_count = 0

    @property
    def config(self) -> NewPolicyModelConfig:
        return self._config

    def load_weights(self, weights_path: str | Path) -> None:
        del weights_path
        self.weights_load_count += 1
        fake_state_dict = {name: value.detach().clone() for name, value in self.state_dict().items()}
        self.load_state_dict(fake_state_dict, strict=True)

    @staticmethod
    def _action_dim(output_features: list[Feature]) -> int:
        if not output_features:
            raise ValueError("At least one output feature is required")

        action_dim = 0
        for feature in output_features:
            if feature.shape is None or not feature.shape:
                raise ValueError(f"Output feature {feature.name!r} must define a non-empty shape")
            action_dim += prod(feature.shape)
        return action_dim

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing_enabled = True

    def enable_lora(self) -> None:
        lora_config = self.config.lora_config
        self.lora_config = (
            int(lora_config["rank"]),
            int(lora_config["alpha"]),
            float(lora_config["dropout"]),
        )

    def _predict_actions(self, batch: Mapping[str, Tensor]) -> Tensor:
        hidden_states = self.text_model(batch["input_ids"])
        hidden_states = self.transformer(hidden_states)
        actions = self.action_head(hidden_states[:, -1])
        return actions.reshape(actions.shape[0], self.config.chunk_size, -1)

    def forward(self, batch: dict[str, Tensor]) -> Tensor | tuple[Tensor, dict[str, Tensor | float]]:
        if self.training:
            return self.compute_loss(batch)
        return self.predict_action_chunk(batch)

    def compute_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor | float]]:
        actions = self._predict_actions(batch)
        loss = torch.nn.functional.mse_loss(actions, batch["action"])
        return loss, {"loss": loss.detach()}

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        return self._predict_actions(batch)

    @property
    def reward_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.config.chunk_size))

    @property
    def observation_delta_indices(self) -> None:
        return None


# Below is testing code

class _FakePolicyDataset(Dataset):
    def __init__(
        self,
        input_features: list[Feature],
        output_features: list[Feature],
        *,
        size: int,
        sequence_length: int,
        chunk_size: int,
        action_dim: int,
        vocab_size: int,
        state_dim: int,
        image_size: tuple[int, int],
    ) -> None:
        self._input_features = {cast("str", feature.name): feature for feature in input_features}
        self._output_features = {cast("str", feature.name): feature for feature in output_features}
        self._size = size
        self._sequence_length = sequence_length
        self._chunk_size = chunk_size
        self._action_dim = action_dim
        self._vocab_size = vocab_size
        self._state_dim = state_dim
        self._image_size = image_size
        self._delta_indices: dict[str, list[int]] = {}

    def __getitem__(self, index: int) -> Observation:
        generator = torch.Generator().manual_seed(index)
        return Observation(
            task=torch.randint(self._vocab_size, (self._sequence_length,), generator=generator),
            state=torch.randn(self._state_dim, generator=generator),
            images={
                "front": torch.randn(3, *self._image_size, generator=generator),
            },
            action=torch.randn(self._chunk_size, self._action_dim, generator=generator),
        )

    def __len__(self) -> int:
        return self._size

    @property
    def raw_features(self) -> dict[str, Feature]:
        return {**self._input_features, **self._output_features}

    @property
    def observation_features(self) -> dict[str, Feature]:
        return self._input_features

    @property
    def action_features(self) -> dict[str, Feature]:
        return self._output_features

    @property
    def fps(self) -> int:
        return 30

    @property
    def tolerance_s(self) -> float:
        return 1e-4

    @property
    def delta_indices(self) -> dict[str, list[int]]:
        return self._delta_indices

    @delta_indices.setter
    def delta_indices(self, indices: dict[str, list[int]]) -> None:
        self._delta_indices = indices


if __name__ == "__main__":
    torch.manual_seed(7)
    sequence_length = 6
    chunk_size = 3
    action_dim = 2
    vocab_size = 32
    state_dim = 4
    image_size = (16, 16)

    input_features = [
        Feature(name=STATE, shape=(state_dim,), ftype=FeatureType.STATE),
        Feature(name="front", shape=(3, *image_size), ftype=FeatureType.VISUAL),
    ]
    output_features = [
        Feature(
            name="action",
            shape=(action_dim,),
            ftype=FeatureType.ACTION,
            normalization_data=NormalizationParameters(
                mean=[0.25, -0.25],
                std=[0.5, 2.0],
            ),
        ),
    ]

    tiny_config = NewPolicyModelConfig(
        input_features=input_features,
        output_features=output_features,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        attention_dropout=0.2,
        layer_norm_eps=1e-4,
        vocab_size=vocab_size,
        chunk_size=chunk_size,
        n_action_steps=2,
        image_size=image_size,
        tokenizer_max_length=sequence_length,
    )

    dataset = _FakePolicyDataset(
        input_features,
        output_features,
        size=8,
        sequence_length=sequence_length,
        chunk_size=chunk_size,
        action_dim=action_dim,
        vocab_size=vocab_size,
        state_dim=state_dim,
        image_size=image_size,
    )
    datamodule = DataModule(
        train_dataset=dataset,
        train_batch_size=2,
        num_workers=0,
    )
    policy = NewPolicy.from_config(tiny_config, optimizer_lr=1e-3)

    batch = next(iter(datamodule.train_dataloader()))
    policy.eval()
    prediction = cast("Tensor", policy(batch))
    assert prediction.shape == (2, chunk_size, action_dim)
    print(f"Forward pass: shape={tuple(prediction.shape)}")

    inputs_schema = policy.inputs_schema
    outputs_schema = policy.outputs_schema
    assert inputs_schema is not None and [feature.name for feature in inputs_schema] == [STATE, IMAGES, TASK]
    assert inputs_schema[-1].ftype == InferenceFeatureType.LANGUAGE
    assert inputs_schema[-1].dtype == InferenceFeatureDtype.STRING
    assert outputs_schema is not None and outputs_schema[0].shape == (chunk_size, action_dim)
    assert outputs_schema[0].name == ACTION
    onnx_export_args = cast("ONNXExportParameters", policy.extra_export_args["onnx"])
    torch_export_args = cast("TorchExportParameters", policy.extra_export_args["torch"])
    openvino_export_args = cast("OpenVINOExportParameters", policy.extra_export_args["openvino"])
    executorch_export_args = cast("ExecuTorchExportParameters", policy.extra_export_args["executorch"])
    assert onnx_export_args.exporter_kwargs["output_names"] == [ACTION]
    assert openvino_export_args.outputs == [ACTION]
    assert executorch_export_args.preprocessors_specs[0].type == "resize"
    assert torch_export_args.postprocessors_specs[0].type == "action_chunk_trimmer"
    print("Export contract: torch, openvino, onnx, executorch")

    trainer = Trainer(
        accelerator="cpu",
        max_steps=1,
        limit_train_batches=1,
        limit_val_batches=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    policy.train()
    trainer.fit(policy, datamodule=datamodule)
    assert trainer.global_step == 1
    print(f"Training complete: global_step={trainer.global_step}")
