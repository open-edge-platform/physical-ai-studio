# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lightning lifecycle for the first-party VLA-JEPA implementation."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import asdict, replace
from pathlib import Path  # noqa: TC003 - jsonargparse resolves constructor annotations.
from typing import IO, TYPE_CHECKING, Any, Literal, Self

import torch
from physicalai.inference.data import InferenceFeature, InferenceFeatureDtype, InferenceFeatureType
from physicalai.inference.manifest import ComponentSpec
from torch import Tensor

from physicalai.data import Feature, FeatureType, Observation
from physicalai.data.dataset import Dataset
from physicalai.export import ExportablePolicyMixin, ExportBackend
from physicalai.export.backends import ExportParameters, TorchExportParameters
from physicalai.policies.base import Model, Policy
from physicalai.policies.mixins.peft import PeftConfigMixin, PeftPolicyMixin, is_lora_injected
from physicalai.train.schedulers import cosine_decay_with_warmup_scheduler
from physicalai.train.utils import reformat_dataset_to_match_policy

from .config import VLAJEPAConfig, feature_dimension
from .model import VLAJEPAModel
from .pretrained import DEFAULT_CHECKPOINT, resolve_pretrained
from .processors import VLAJEPAPostprocessor, VLAJEPAPreprocessor, _batch_key

if TYPE_CHECKING:
    from lightning.pytorch.utilities.types import OptimizerLRScheduler


def _feature_contract(features: list[Feature]) -> list[Feature]:
    return [replace(feature, normalization_data=None) for feature in features]


class VLAJEPA(PeftPolicyMixin, ExportablePolicyMixin, Policy):
    """Native pretrained VLA-JEPA with joint JEPA/action or action-only fine-tuning.

    Construction without features is lazy; training resolves the dataset's ordered
    features and normalization. Use ``from_pretrained`` for eager inference or
    ``from_config`` for a fully specified architecture. Full JEPA-assisted training
    is the default; disable ``enable_world_model`` for action-only training.
    LoRA/DoRA and Torch export are supported. Human-video-only pretraining is not.
    """

    def __init__(  # noqa: PLR0913
        self,
        input_features: list[Feature] | None = None,
        output_features: list[Feature] | None = None,
        pretrained_name_or_path: str | Path | None = DEFAULT_CHECKPOINT,
        *,
        pretrained_revision: str | None = None,
        chunk_size: int | None = None,
        n_action_steps: int | None = None,
        dtype: str | None = None,
        num_inference_steps: int | None = None,
        gradient_checkpointing: bool = True,
        train_action_head_only: bool = False,
        enable_world_model: bool | None = None,
        world_model_loss_weight: float = 0.1,
        reinitialize_projections: bool = True,
        compile_model: bool = False,
        pre_snap_gripper: bool | None = None,
        binarize_gripper: bool | None = None,
        lora_enabled: bool = False,
        lora_rank: int = 32,
        lora_alpha: int | None = None,
        lora_dropout: float = 0.05,
        lora_target_modules: str | tuple[str, ...] | None = None,
        lora_adapter_dtype: Literal["float32", "auto"] = "float32",
        lora_use_dora: bool = False,
        lora_lr_scale: float = 10.0,
        optimizer_lr: float = 1e-5,
        optimizer_action_head_lr: float = 1e-4,
        optimizer_weight_decay: float = 1e-8,
        scheduler_warmup_steps: int = 1000,
        scheduler_decay_lr: float = 1e-6,
    ) -> None:
        """Store construction and training settings; defer feature-free initialization.

        Raises:
            ValueError: If features or training settings are inconsistent.
        """
        super().__init__(n_action_steps=n_action_steps or 7)
        if (input_features is None) != (output_features is None):
            msg = "Provide input_features and output_features together."
            raise ValueError(msg)
        if any(value is not None and value <= 0 for value in (chunk_size, n_action_steps, num_inference_steps)):
            msg = "Action horizons and inference steps must be positive."
            raise ValueError(msg)
        if min(optimizer_lr, optimizer_action_head_lr) <= 0 or optimizer_weight_decay < 0 or scheduler_warmup_steps < 0:
            msg = "Invalid optimizer settings."
            raise ValueError(msg)
        if not 0 <= scheduler_decay_lr <= optimizer_lr:
            msg = "scheduler_decay_lr must be between zero and optimizer_lr."
            raise ValueError(msg)
        if lora_enabled and train_action_head_only:
            msg = "lora_enabled is incompatible with train_action_head_only; use lora_target_modules instead."
            raise ValueError(msg)
        self._peft_config = PeftConfigMixin(
            lora_enabled=lora_enabled,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
            lora_adapter_dtype=lora_adapter_dtype,
            lora_use_dora=lora_use_dora,
            lora_lr_scale=lora_lr_scale,
        )
        self.input_features = input_features
        self.output_features = output_features
        self.pretrained_name_or_path = pretrained_name_or_path
        self.pretrained_revision = pretrained_revision
        self.chunk_size = chunk_size
        self.n_action_steps = n_action_steps
        self.dtype_name = dtype
        self.num_inference_steps = num_inference_steps
        self.gradient_checkpointing = gradient_checkpointing
        self.train_action_head_only = train_action_head_only
        self._requested_world_model = enable_world_model
        self.enable_world_model = enable_world_model is not False and not train_action_head_only
        self.world_model_loss_weight = world_model_loss_weight
        self.reinitialize_projections = reinitialize_projections
        self.compile_model = compile_model
        self.pre_snap_gripper = pre_snap_gripper
        self.binarize_gripper = binarize_gripper
        self.optimizer_lr = optimizer_lr
        self.optimizer_action_head_lr = optimizer_action_head_lr
        self.optimizer_weight_decay = optimizer_weight_decay
        self.scheduler_warmup_steps = scheduler_warmup_steps
        self.scheduler_decay_lr = scheduler_decay_lr
        self.model: Model | None  # pyrefly: ignore[bad-override-mutable-attribute]
        self.config: VLAJEPAConfig | None = None
        self._weights_path: Path | None = None
        self._preprocessor: VLAJEPAPreprocessor | None = None  # pyrefly: ignore[bad-override-mutable-attribute]
        self._postprocessor: VLAJEPAPostprocessor | None = None
        self.save_hyperparameters(
            ignore=[
                "input_features",
                "output_features",
                "pretrained_name_or_path",
                "pretrained_revision",
                "compile_model",
            ],
        )
        if input_features is not None:
            self._resolve_configuration()
            self.configure_model()

    @classmethod
    def from_config(
        cls,
        config: VLAJEPAConfig,
        *,
        gradient_checkpointing: bool = True,
        train_action_head_only: bool = False,
        compile_model: bool = False,
        optimizer_lr: float = 1e-5,
        optimizer_action_head_lr: float = 1e-4,
        optimizer_weight_decay: float = 1e-8,
        scheduler_warmup_steps: int = 1000,
        scheduler_decay_lr: float = 1e-6,
    ) -> Self:
        """Return a policy built from resolved config without network access.

        Raises:
            ValueError: If head-only training is combined with LoRA.
        """
        if train_action_head_only and config.use_lora:
            msg = "lora_enabled is incompatible with train_action_head_only; use lora_target_modules instead."
            raise ValueError(msg)
        if train_action_head_only:
            config = replace(config, enable_world_model=False)
        policy = cls(
            pretrained_name_or_path=None,
            enable_world_model=config.enable_world_model,
            world_model_loss_weight=config.world_model_loss_weight,
            lora_enabled=config.lora_enabled,
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            lora_target_modules=config.lora_target_modules,
            lora_adapter_dtype=config.lora_adapter_dtype,
            lora_use_dora=config.lora_use_dora,
            lora_lr_scale=config.lora_lr_scale,
            gradient_checkpointing=gradient_checkpointing,
            train_action_head_only=train_action_head_only,
            compile_model=compile_model,
            optimizer_lr=optimizer_lr,
            optimizer_action_head_lr=optimizer_action_head_lr,
            optimizer_weight_decay=optimizer_weight_decay,
            scheduler_warmup_steps=scheduler_warmup_steps,
            scheduler_decay_lr=scheduler_decay_lr,
        )
        policy.config = config
        policy.configure_model()
        return policy

    @classmethod
    def from_pretrained(cls, name_or_path: str | Path = DEFAULT_CHECKPOINT, **kwargs: Any) -> Self:  # noqa: ANN401
        """Return a released policy for inference, rejecting random projection replacement."""
        kwargs.setdefault("reinitialize_projections", False)
        kwargs.setdefault("enable_world_model", False)
        policy = cls(pretrained_name_or_path=name_or_path, **kwargs)
        if policy.model is None:
            policy._resolve_configuration()
            policy.configure_model()
        return policy

    def _resolve_configuration(self) -> None:
        if self.config is not None:
            return
        if self.pretrained_name_or_path is None:
            msg = "Use VLAJEPA.from_config with a complete architecture when not loading pretrained weights."
            raise ValueError(msg)
        config, self._weights_path = resolve_pretrained(
            self.pretrained_name_or_path,
            self.pretrained_revision,
            enable_world_model=self.enable_world_model,
        )
        if self.input_features is not None and self.output_features is not None:
            config = replace(
                config,
                input_features=list(self.input_features),
                output_features=list(self.output_features),
                action_dim=feature_dimension(self.output_features, FeatureType.ACTION),
                state_dim=feature_dimension(self.input_features, FeatureType.STATE),
                # Dataset grippers keep their own units; released LIBERO conventions are not universal.
                pre_snap_gripper=False,
                binarize_gripper=False,
            )
        self.config = replace(
            config,
            enable_world_model=self.enable_world_model,
            world_model_loss_weight=self.world_model_loss_weight,
            chunk_size=self.chunk_size if self.chunk_size is not None else config.chunk_size,
            n_action_steps=self.n_action_steps
            if self.n_action_steps is not None
            else min(config.n_action_steps, self.chunk_size or config.chunk_size),
            dtype=self.dtype_name if self.dtype_name is not None else config.dtype,
            num_inference_steps=self.num_inference_steps
            if self.num_inference_steps is not None
            else config.num_inference_steps,
            pre_snap_gripper=self.pre_snap_gripper if self.pre_snap_gripper is not None else config.pre_snap_gripper,
            binarize_gripper=self.binarize_gripper if self.binarize_gripper is not None else config.binarize_gripper,
            **asdict(self._peft_config),
        )

    def configure_model(self) -> None:
        """Materialize the resolved architecture once, without downloading artifacts.

        Raises:
            ValueError: If architecture or processor assets are unresolved.
            RuntimeError: If the LoRA targets match no supported modules.
        """
        if self.model is not None or self.config is None:
            return
        config = self.config
        if not config.qwen_config or not config.tokenizer_json or not config.chat_template:
            msg = "VLA-JEPA requires resolved Qwen architecture and processor assets."
            raise ValueError(msg)
        if config.enable_world_model and (not config.world_model_config or not config.video_processor_config):
            msg = "JEPA-assisted training requires resolved encoder and video-processor configuration."
            raise ValueError(msg)
        model = VLAJEPAModel.from_config(config)
        preprocessor = VLAJEPAPreprocessor(config)
        postprocessor = VLAJEPAPostprocessor(config)
        if self._weights_path is not None:
            model.load_weights(self._weights_path, reinitialize_projections=self.reinitialize_projections)
        if self.gradient_checkpointing:
            model.enable_gradient_checkpointing()
        if self.train_action_head_only:
            model.freeze_backbone()
        if self.compile_model:
            model.predict_action_chunk = torch.compile(model.predict_action_chunk)
        self.model = model
        try:
            if config.use_lora:
                self._inject_lora()
        except RuntimeError:
            self.model = None
            raise
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._weights_path = None
        self.train(self.training)
        self.input_features = config.input_features
        self.output_features = config.output_features
        self.chunk_size = config.chunk_size
        self.n_action_steps = config.n_action_steps
        self.enable_world_model = config.enable_world_model
        self.world_model_loss_weight = config.world_model_loss_weight
        self._n_action_steps = config.n_action_steps
        self._action_queue: deque[Tensor] = deque(maxlen=config.n_action_steps)

    def setup(self, stage: str) -> None:
        """Resolve training features before Lightning materializes the architecture.

        Raises:
            TypeError: If the data module does not supply a PhysicalAI Dataset.
            ValueError: If training features differ from the initialized contract.
        """
        if stage != "fit":
            return
        dataset = getattr(getattr(self.trainer, "datamodule", None), "train_dataset", None)
        if not isinstance(dataset, Dataset):
            msg = "VLA-JEPA requires a PhysicalAI training Dataset."
            raise TypeError(msg)
        inputs = list(dataset.observation_features.values())
        outputs = list(dataset.action_features.values())
        if self.config is None:
            self.input_features, self.output_features = inputs, outputs
            if getattr(self.trainer, "ckpt_path", None) is None:
                self._resolve_configuration()
            return

        if _feature_contract(inputs) != _feature_contract(self.config.input_features) or _feature_contract(
            outputs,
        ) != _feature_contract(self.config.output_features):
            msg = (
                "Training features differ from the initialized VLA-JEPA contract; "
                "construct a lazy policy for this dataset."
            )
            raise ValueError(msg)
        config = replace(self.config, input_features=inputs, output_features=outputs)
        if self.model is not None:
            self._preprocessor = VLAJEPAPreprocessor(config).to(self.device)
            self._postprocessor = VLAJEPAPostprocessor(config).to(self.device)
        self.config = config
        self.input_features, self.output_features = inputs, outputs

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Store architecture, features, normalization, and tokenizer assets together.

        Raises:
            RuntimeError: If the policy has not been initialized.
        """
        if self.config is None:
            msg = "Cannot checkpoint an uninitialized VLA-JEPA policy."
            raise RuntimeError(msg)
        checkpoint["policy_config"] = {
            **self.config.to_dict(),
            "lora_enabled": self.model is not None and is_lora_injected(self.model),
        }

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Reconstruct saved architecture before Lightning restores tensors.

        Raises:
            TypeError: If the checkpoint has no configuration.
            ValueError: If it conflicts with an initialized policy.
        """
        data = checkpoint.get("policy_config")
        if not isinstance(data, Mapping):
            msg = "VLA-JEPA checkpoint is missing policy_config."
            raise TypeError(msg)
        config = VLAJEPAConfig.from_dict(dict(data))
        if config.use_lora and self.train_action_head_only:
            msg = "Cannot restore a LoRA checkpoint with train_action_head_only."
            raise ValueError(msg)
        requested_mode = False if self.train_action_head_only else self._requested_world_model
        if requested_mode is True and not config.enable_world_model:
            msg = "This checkpoint is action-only; initialize from pretrained weights to enable JEPA training."
            raise ValueError(msg)
        if requested_mode is False and config.enable_world_model:
            config = replace(config, enable_world_model=False)
            checkpoint["state_dict"] = {
                name: value
                for name, value in checkpoint["state_dict"].items()
                if not name.startswith(("model.video_encoder.", "model.video_predictor."))
            }
        if self.config is not None and self.config != config:
            msg = "Checkpoint config differs from the initialized VLA-JEPA config."
            raise ValueError(msg)
        if (
            self.input_features is not None
            and self.output_features is not None
            and (
                _feature_contract(self.input_features) != _feature_contract(config.input_features)
                or _feature_contract(self.output_features) != _feature_contract(config.output_features)
            )
        ):
            msg = "Checkpoint feature contract does not match the training dataset."
            raise ValueError(msg)
        self.pretrained_name_or_path = None
        self._weights_path = None
        self.config = config
        self.configure_model()

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path | IO[bytes],
        map_location: torch.device | str | int | Callable | dict | None = None,
        hparams_file: str | Path | None = None,
        strict: bool | None = None,  # noqa: FBT001
        weights_only: bool | None = None,  # noqa: FBT001
        **kwargs: Any,  # noqa: ANN401
    ) -> Self:
        """Return an offline-restored policy without reapplying pretrained weights."""
        kwargs["pretrained_name_or_path"] = None
        return super().load_from_checkpoint(
            checkpoint_path,
            map_location=map_location,
            hparams_file=hparams_file,
            strict=strict,
            weights_only=weights_only,
            **kwargs,
        )

    def _prepare_batch(self, batch: Observation, *, include_video: bool = False) -> dict[str, Tensor]:
        if self._preprocessor is None:
            msg = "VLA-JEPA is not initialized."
            raise RuntimeError(msg)
        return self._preprocessor(batch.to(self.device).to_dict(), include_video=include_video)

    def forward(self, batch: Observation) -> Tensor | tuple[Tensor, dict[str, Tensor | float]]:
        """Return training loss or denormalized actions.

        Raises:
            RuntimeError: If the policy has not been initialized.
        """
        if not self.training:
            return self.predict_action_chunk(batch)
        if self.model is None:
            msg = "VLA-JEPA is not initialized."
            raise RuntimeError(msg)
        return self.model(self._prepare_batch(batch, include_video=self.enable_world_model))

    @torch.no_grad()
    def predict_action_chunk(self, batch: Observation) -> Tensor:
        """Return a full action chunk in the dataset's action units.

        Raises:
            RuntimeError: If the policy has not been initialized.
        """
        if not isinstance(self.model, VLAJEPAModel) or self._postprocessor is None:
            msg = "VLA-JEPA is not initialized."
            raise RuntimeError(msg)
        return self._postprocessor(self.model.predict_action_chunk(self._prepare_batch(batch)))

    def training_step(self, batch: Observation, batch_idx: int) -> Tensor:
        """Return and log the differentiable training loss."""
        del batch_idx
        loss, metrics = self(batch)
        self.log("train/loss", metrics["loss"], prog_bar=True, batch_size=batch.batch_size)
        for name in ("action_flow_loss", "world_model_loss"):
            self.log(f"train/{name}", metrics[name], batch_size=batch.batch_size)
        return loss

    def compute_val_loss(self, batch: Observation) -> tuple[Tensor, dict[str, Tensor | float]]:
        """Return deterministic, fully denoised action MSE and metrics.

        Raises:
            RuntimeError: If the policy has not been initialized.
        """
        if self.model is None:
            msg = "VLA-JEPA is not initialized."
            raise RuntimeError(msg)
        return self.model.compute_val_loss(self._prepare_batch(batch))

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Return separate backbone/head optimizer groups and a shared cosine schedule.

        Raises:
            TypeError: If the policy has not been initialized.
        """
        if not isinstance(self.model, VLAJEPAModel) or self.config is None:
            msg = "VLA-JEPA is not initialized."
            raise TypeError(msg)
        lr_scale = self.config.lora_lr_scale if self.config.use_lora else 1.0
        groups: list[dict[str, Any]] = [
            {
                "params": [parameter for parameter in self.model.qwen.parameters() if parameter.requires_grad],
                "lr": self.optimizer_lr * lr_scale,
            },
            {
                "params": [parameter for parameter in self.model.action_model.parameters() if parameter.requires_grad],
                "lr": self.optimizer_action_head_lr * lr_scale,
            },
        ]
        if self.model.video_predictor is not None:
            groups.append({
                "params": [
                    parameter for parameter in self.model.video_predictor.parameters() if parameter.requires_grad
                ],
                "lr": self.optimizer_lr * lr_scale,
            })
        optimizer = torch.optim.AdamW(
            [group for group in groups if group["params"]],
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=self.optimizer_weight_decay,
        )
        # Estimating the step budget can start workers; set their temporal windows first.
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is not None:
            reformat_dataset_to_match_policy(self, datamodule)
        scheduler = cosine_decay_with_warmup_scheduler(
            optimizer,
            peak_lr=self.optimizer_lr * lr_scale,
            decay_lr=self.scheduler_decay_lr * lr_scale,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=max(1, int(self.trainer.estimated_stepping_batches)),
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def _should_merge_lora_for_export(self) -> bool:
        """Return whether this initialized policy has adapters to merge."""
        return self.config is not None and super()._should_merge_lora_for_export()

    @property
    def inputs_schema(self) -> list[InferenceFeature] | None:
        """Describe raw camera, state, and task inputs consumed by Torch Runtime."""
        if self.model is None or self.config is None:
            return None
        schema = [
            InferenceFeature(
                name=_batch_key(feature),
                ftype=InferenceFeatureType(feature.ftype),
                shape=feature.shape or (),
                dtype=InferenceFeatureDtype.FLOAT32,
            )
            for feature in self.config.input_features
            if feature.ftype in {FeatureType.VISUAL, FeatureType.STATE}
        ]
        schema.append(
            InferenceFeature(
                name="task",
                ftype=InferenceFeatureType.LANGUAGE,
                shape=(),
                dtype=InferenceFeatureDtype.STRING,
            ),
        )
        return schema

    @property
    def outputs_schema(self) -> list[InferenceFeature] | None:
        """Describe the full denormalized action chunk before Runtime trimming."""
        if self.model is None or self.config is None:
            return None
        return [
            InferenceFeature(
                name="action",
                ftype=InferenceFeatureType.ACTION,
                shape=(self.config.chunk_size, self.config.action_dim),
                dtype=InferenceFeatureDtype.FLOAT32,
            ),
        ]

    @property
    def extra_export_args(self) -> dict[str, ExportParameters]:
        """Keep preprocessing in the Torch policy and trim only the executed prefix.

        Raises:
            RuntimeError: If the policy has not been initialized.
        """
        if self.config is None or self.model is None:
            msg = "Cannot export an uninitialized VLA-JEPA policy."
            raise RuntimeError(msg)
        postprocessors = []
        if self.config.n_action_steps != self.config.chunk_size:
            postprocessors.append(ComponentSpec(type="action_chunk_trimmer", n_action_steps=self.config.n_action_steps))
        return {
            ExportBackend.TORCH: TorchExportParameters(
                postprocessors_specs=postprocessors,
            ),
        }
