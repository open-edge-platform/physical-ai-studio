# Copyright (C) 2026 Xiaomi Corporation.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""XR0 Policy - Lightning wrapper for training and inference."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal, cast

import torch
from physicalai.inference.data import InferenceFeature, InferenceFeatureDtype, InferenceFeatureType
from physicalai.inference.manifest import ComponentSpec

from physicalai.data.dataset import Dataset
from physicalai.data.observation import ACTION, IMAGES, STATE, TASK, Feature, FeatureType, NormalizationParameters
from physicalai.export import ExportablePolicyMixin, ExportBackend
from physicalai.export.backends import ExportParameters, OpenVINOExportParameters, TorchExportParameters
from physicalai.policies.base import Policy
from physicalai.train.schedulers import cosine_decay_with_warmup_scheduler
from physicalai.train.utils import reformat_dataset_to_match_policy

from .config import XR0Config
from .model import XR0Model
from .preprocessor import make_xr0_preprocessors
from .pretrained_utils import extract_xr0_dataset_stats, load_xr0_pretrained_weights, resolve_pretrained_path

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from physicalai.data import Observation

    from .preprocessor import XR0Postprocessor, XR0Preprocessor

logger = logging.getLogger(__name__)

_DTYPES: dict[str, torch.dtype] = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


class XR0(ExportablePolicyMixin, Policy):
    """XR0 Policy - Xiaomi's flow-matching VLA model.

    Lightning wrapper for training and inference with :class:`XR0Model`.

    Args:
        pretrained_name_or_path: Optional local path or HuggingFace repo id of a
            pretrained XR0 checkpoint (e.g.
            ``"XiaomiRobotics/Xiaomi-Robotics-0-LIBERO"``). When given, the
            weights are loaded into the model once it is built.
        input_features: Optional explicit observation feature schema
            (``list[Feature]``). When omitted, it is traced back from the
            training dataset in :meth:`setup`. Must be given together with
            ``output_features``.
        output_features: Optional explicit action feature schema
            (``list[Feature]``). Must be given together with ``input_features``.
        vlm_model_id: HuggingFace id of the Qwen3-VL backbone.
        vlm_attn_implementation: Attention backend for the VLM.
        dtype: Model precision (``"bfloat16"``, ``"float16"`` or ``"float32"``).
        n_obs_steps: Number of observation steps. Unused: XR0 always conditions
            on the single current observation (``observation_delta_indices`` is
            fixed to ``None``); kept only for config parity with other policies.
        chunk_size: Number of action steps to predict.
        n_action_steps: Number of action steps to execute.
        max_state_dim: Padded state dimension.
        max_action_dim: Padded action dimension.
        state_len: Number of state tokens.
        dit_num_layers: DiT decoder layers.
        dit_hidden_size: DiT hidden width.
        dit_head_dim: DiT attention head dim.
        dit_kv_heads: DiT key/value heads.
        num_inference_steps: Euler integration steps for inference.
        flow_sampling: Training timestep distribution.
        local_window: Local-attention window for the action tokens.
        training_repeat: Per-sample training repeat factor.
        enable_freq: Add the frequency-domain loss term.
        prefix_mask_prob: Probability of masking a prefix token in training.
        async_train: Randomly condition on an action prefix in training.
        camera_views: Ordered camera view names for the prompt.
        image_resolution: Target image resolution (unused placeholder kept for
            config parity; the Qwen3-VL processor performs area-based resizing).
        tokenizer_max_length: Maximum tokenizer length.
        gradient_checkpointing: Enable gradient checkpointing.
        compile_model: Whether to use torch.compile.
        compile_mode: Torch compile mode.
        freeze_vision_encoder: Freeze the vision encoder.
        freeze_input_embeddings: Freeze the VLM token-embedding table (matches
            the original XR0 recipe; saves the embedding grads/optimizer state).
        normalize_state: Normalize the proprioceptive state with the dataset's
            per-dimension mean/std. Defaults to False (raw state), keeping
            existing raw-state checkpoints/exports unchanged; enable it for
            embodiments whose raw state is off the pretrained checkpoint's scale.
        action_mode: ``"absolute"`` (default) predicts the raw action;
            ``"delta"`` predicts ``action[t] - state`` and re-adds the state at
            inference, matching the pretrained flow head's delta prior.
        action_delta_mean: Per-timestep delta-action mean
            (``(chunk_size, max_action_dim)``) used when ``action_mode="delta"``;
            compute it with :func:`compute_delta_action_stats`.
        action_delta_std: Per-timestep delta-action std, same shape as
            ``action_delta_mean``.
        normalization_mode: Normalization method for state/action features.
        optimizer_lr: Learning rate.
        optimizer_betas: Adam beta coefficients.
        optimizer_eps: Optimizer epsilon.
        optimizer_weight_decay: Weight decay coefficient.
        optimizer_grad_clip_norm: Maximum gradient norm for clipping.
        scheduler_warmup_steps: Number of warmup steps.
        scheduler_decay_steps: Cosine decay horizon in steps (``None`` auto).
        scheduler_decay_lr: Final learning rate after decay.
        dataset_stats: Dataset stats for eager initialization.

    Example:
        Training:

        >>> policy = XR0(optimizer_lr=2.5e-5)
        >>> trainer = physicalai.train.Trainer(max_epochs=100)
        >>> trainer.fit(policy, datamodule)

        Fine-tuning from the pretrained LIBERO checkpoint:

        >>> policy = XR0(pretrained_name_or_path="XiaomiRobotics/Xiaomi-Robotics-0-LIBERO")
        >>> trainer.fit(policy, datamodule)
    """

    def __init__(  # noqa: PLR0913
        self,
        pretrained_name_or_path: str | Path | None = None,
        vlm_model_id: str = "Qwen/Qwen3-VL-4B-Instruct",
        vlm_attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = "flash_attention_2",
        dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16",
        n_obs_steps: int = 1,
        chunk_size: int = 30,
        n_action_steps: int = 30,
        max_state_dim: int = 32,
        max_action_dim: int = 32,
        state_len: int = 1,
        *,
        input_features: list[Feature] | None = None,
        output_features: list[Feature] | None = None,
        dit_num_layers: int = 16,
        dit_hidden_size: int = 1024,
        dit_head_dim: int = 128,
        dit_kv_heads: int = 8,
        num_inference_steps: int = 5,
        flow_sampling: Literal["beta", "logit_normal", "uniform"] = "beta",
        local_window: int = 4,
        training_repeat: int = 4,
        enable_freq: bool = True,
        prefix_mask_prob: float = 0.5,
        async_train: bool = False,
        camera_views: tuple[str, ...] = ("base", "wrist_left"),
        image_resolution: tuple[int, int] = (256, 256),
        tokenizer_max_length: int = 256,
        gradient_checkpointing: bool = True,
        compile_model: bool = False,
        compile_mode: str = "max-autotune",
        freeze_vision_encoder: bool = False,
        freeze_input_embeddings: bool = True,
        normalize_state: bool = False,
        action_mode: Literal["absolute", "delta"] = "absolute",
        action_delta_mean: Sequence[float] | torch.Tensor | None = None,
        action_delta_std: Sequence[float] | torch.Tensor | None = None,
        normalization_mode: Literal["MEAN_STD", "QUANTILES"] = "QUANTILES",
        optimizer_lr: float = 1.0e-4,
        optimizer_betas: tuple[float, float] = (0.9, 0.95),
        optimizer_eps: float = 1e-8,
        optimizer_weight_decay: float = 0.1,
        optimizer_grad_clip_norm: float = 1.0,
        scheduler_warmup_steps: int = 2_000,
        scheduler_decay_steps: int | None = 30_000,
        scheduler_decay_lr: float = 5.0e-7,
        dataset_stats: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Initialize the XR0 policy.

        Raises:
            ValueError: If only one of ``input_features`` / ``output_features``
                is provided.
        """
        super().__init__(n_action_steps=n_action_steps)

        # Input/output features must be provided together (or both omitted and
        # traced back from the dataset in ``setup``), mirroring MolmoAct2.
        if bool(input_features) != bool(output_features):
            msg = f"Need both input and output features: input: {input_features} - output: {output_features}"
            raise ValueError(msg)

        self.config = XR0Config(
            vlm_model_id=vlm_model_id,
            vlm_attn_implementation=vlm_attn_implementation,
            dtype=dtype,
            n_obs_steps=n_obs_steps,
            chunk_size=chunk_size,
            n_action_steps=n_action_steps,
            max_state_dim=max_state_dim,
            max_action_dim=max_action_dim,
            state_len=state_len,
            input_features=input_features,
            output_features=output_features,
            dit_num_layers=dit_num_layers,
            dit_hidden_size=dit_hidden_size,
            dit_head_dim=dit_head_dim,
            dit_kv_heads=dit_kv_heads,
            num_inference_steps=num_inference_steps,
            flow_sampling=flow_sampling,
            local_window=local_window,
            training_repeat=training_repeat,
            enable_freq=enable_freq,
            prefix_mask_prob=prefix_mask_prob,
            async_train=async_train,
            camera_views=camera_views,
            image_resolution=image_resolution,
            tokenizer_max_length=tokenizer_max_length,
            gradient_checkpointing=gradient_checkpointing,
            compile_model=compile_model,
            compile_mode=compile_mode,
            freeze_vision_encoder=freeze_vision_encoder,
            freeze_input_embeddings=freeze_input_embeddings,
            normalize_state=normalize_state,
            action_mode=action_mode,
            normalization_mode=normalization_mode,
            optimizer_lr=optimizer_lr,
            optimizer_betas=optimizer_betas,
            optimizer_eps=optimizer_eps,
            optimizer_weight_decay=optimizer_weight_decay,
            optimizer_grad_clip_norm=optimizer_grad_clip_norm,
            scheduler_warmup_steps=scheduler_warmup_steps,
            scheduler_decay_steps=scheduler_decay_steps,
            scheduler_decay_lr=scheduler_decay_lr,
        )

        self.save_hyperparameters(ignore=["config", "compile_model", "pretrained_name_or_path"])
        self._set_hparam_keys()

        # Per-timestep delta-action stats (only used when action_mode="delta").
        # Stored as tensors for the preprocessors and mirrored into hparams as
        # plain lists so they round-trip through Lightning checkpoints.
        self._action_delta_mean: torch.Tensor | None = (
            None if action_delta_mean is None else torch.as_tensor(action_delta_mean, dtype=torch.float32)
        )
        self._action_delta_std: torch.Tensor | None = (
            None if action_delta_std is None else torch.as_tensor(action_delta_std, dtype=torch.float32)
        )
        if self._action_delta_mean is not None and self._action_delta_std is not None:
            self.hparams["action_delta_mean"] = self._action_delta_mean.tolist()
            self.hparams["action_delta_std"] = self._action_delta_std.tolist()

        self.model: XR0Model | None = None
        self._preprocessor: XR0Preprocessor | None = None
        self._postprocessor: XR0Postprocessor | None = None
        self._dataset_stats = dataset_stats
        self._input_features = input_features
        self._output_features = output_features

        # Resolve (download) the pretrained checkpoint now; load it into the
        # model once it is built (eager path here, or lazily in ``setup``).
        self._pretrained_path: Path | None = (
            resolve_pretrained_path(pretrained_name_or_path) if pretrained_name_or_path is not None else None
        )

        # When explicit input/output features are given without dataset stats,
        # derive the normalization stats from them so the model can be built
        # eagerly (no training dataset required).
        if dataset_stats is None and input_features is not None and output_features is not None:
            dataset_stats = self._features_to_stats(input_features, output_features)
            self._dataset_stats = dataset_stats

        # When a pretrained checkpoint is given without explicit dataset stats,
        # recover the action-normalization stats from the checkpoint so the
        # policy is usable for standalone inference (no training dataset).
        if dataset_stats is None and pretrained_name_or_path is not None:
            dataset_stats = extract_xr0_dataset_stats(pretrained_name_or_path)
            self._dataset_stats = dataset_stats

        if dataset_stats is not None:
            self._initialize_model(dataset_stats)

    def _set_hparam_keys(self) -> None:
        """Sync top-level checkpoint hparams from the resolved policy config."""
        for key, value in self.config.__dict__.items():
            if key == "compile_model" or key not in self.hparams:
                continue
            self.hparams[key] = value
        self.hparams["config"] = self.config.to_dict()

    def _initialize_model(self, dataset_stats: dict[str, dict[str, Any]]) -> None:
        """Build the model and preprocessors from dataset statistics."""
        cfg = self.config
        self.model = XR0Model(
            vlm_model_id=cfg.vlm_model_id,
            vlm_attn_implementation=cfg.vlm_attn_implementation,
            state_shape=(cfg.state_len, cfg.max_state_dim),
            action_shape=(cfg.chunk_size, cfg.max_action_dim),
            dit_num_layers=cfg.dit_num_layers,
            dit_hidden_size=cfg.dit_hidden_size,
            dit_head_dim=cfg.dit_head_dim,
            dit_kv_heads=cfg.dit_kv_heads,
            num_steps=cfg.num_inference_steps,
            flow_sampling=cfg.flow_sampling,
            local_window=cfg.local_window,
            training_repeat=cfg.training_repeat,
            enable_freq=cfg.enable_freq,
            prefix_mask_prob=cfg.prefix_mask_prob,
            async_train=cfg.async_train,
            gradient_checkpointing=cfg.gradient_checkpointing,
            freeze_vision_encoder=cfg.freeze_vision_encoder,
            freeze_input_embeddings=cfg.freeze_input_embeddings,
            dtype=_DTYPES[cfg.dtype],
        )

        if self._pretrained_path is not None:
            self._load_pretrained_weights(self._pretrained_path)

        self._preprocessor, self._postprocessor = make_xr0_preprocessors(
            camera_views=cfg.camera_views,
            max_state_dim=cfg.max_state_dim,
            max_action_dim=cfg.max_action_dim,
            stats=dataset_stats,
            processor_name=cfg.vlm_model_id,
            normalize_state=cfg.normalize_state,
            action_mode=cfg.action_mode,
            action_delta_mean=self._action_delta_mean,
            action_delta_std=self._action_delta_std,
        )
        self._dataset_stats = dataset_stats

        # When features were not provided (or traced from a dataset) yet,
        # reconstruct the typed schema from the stats dict so the export
        # ``inputs_schema`` / ``outputs_schema`` are feature-driven.
        if self._input_features is None or self._output_features is None:
            self._input_features, self._output_features = self._stats_to_features(dataset_stats)

    def _rebuild_preprocessors(self, dataset_stats: dict[str, dict[str, Any]]) -> None:
        """Rebuild the pre/post-processors from dataset stats, keeping model weights.

        Used when the model was already built eagerly (e.g. from a pretrained
        checkpoint) but fine-tuning needs the dataset's own normalization.
        """
        cfg = self.config
        self._preprocessor, self._postprocessor = make_xr0_preprocessors(
            camera_views=cfg.camera_views,
            max_state_dim=cfg.max_state_dim,
            max_action_dim=cfg.max_action_dim,
            stats=dataset_stats,
            processor_name=cfg.vlm_model_id,
            normalize_state=cfg.normalize_state,
            action_mode=cfg.action_mode,
            action_delta_mean=self._action_delta_mean,
            action_delta_std=self._action_delta_std,
        )
        self._dataset_stats = dataset_stats

    def _load_pretrained_weights(self, pretrained_path: Path) -> None:
        """Load remapped pretrained weights into ``self.model`` (non-strict).

        Raises:
            ValueError: If the model has not been built yet.
        """
        if self.model is None:
            msg = "Cannot load pretrained weights before the model is initialized"
            raise ValueError(msg)

        state_dict = load_xr0_pretrained_weights(pretrained_path)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False, assign=True)
        self.model.to(_DTYPES[self.config.dtype])

        if missing:
            msg = f"Missing keys when loading pretrained XR0 weights: {len(missing)} keys"
            logger.warning(msg)
            for key in missing[:10]:
                logger.warning("  - %s", key)
        if unexpected:
            msg = f"Unexpected keys when loading pretrained XR0 weights: {len(unexpected)} keys"
            logger.warning(msg)
            for key in unexpected[:10]:
                logger.warning("  - %s", key)

    def setup(self, stage: str) -> None:
        """Build the model from the datamodule statistics (lazy path).

        Raises:
            TypeError: If the train dataset is not a physicalai Dataset.
        """
        datamodule = self.trainer.datamodule  # type: ignore[attr-defined]
        train_dataset = datamodule.train_dataset
        if not isinstance(train_dataset, Dataset):
            msg = f"Expected physicalai Dataset, got {type(train_dataset)}"
            raise TypeError(msg)

        # Trace the input/output feature schema back from the dataset when it
        # was not provided explicitly at construction time.
        if self._input_features is None or self._output_features is None:
            input_features, output_features = self._dataset_features(train_dataset)
            self._input_features = input_features
            self._output_features = output_features
            self.hparams["input_features"] = input_features
            self.hparams["output_features"] = output_features

        stats_dict = train_dataset.stats
        self.hparams["dataset_stats"] = stats_dict
        if self.model is None:
            self._initialize_model(stats_dict)
        else:
            # The model was built eagerly in ``__init__`` (e.g. from a pretrained
            # checkpoint whose normalization stats belong to a *different*
            # embodiment -- delta actions with tiny std). For fine-tuning, the
            # normalization must come from the fine-tuning dataset, otherwise the
            # action/state get divided by the wrong (tiny) std and the flow
            # target explodes. Rebuild the pre/post-processors from the datamodule
            # stats while keeping the already-loaded model weights.
            self._rebuild_preprocessors(stats_dict)

        reformat_dataset_to_match_policy(self, datamodule)

        # The Qwen3-VL backbone is built via ``from_pretrained``, which returns an
        # eval-mode module, and Lightning does not implicitly flip module
        # train/eval state -- it only warns ("N module(s) in eval mode at the
        # start of training"). Put the whole model into train mode for fitting so
        # the backbone trains correctly and the warning does not fire. ``setup``
        # runs before Lightning's eval-mode check, so this also clears the warning.
        if stage == "fit" and self.model is not None:
            self.model.train()

    def forward(self, batch: Observation) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        """Forward pass: training loss (train) or action chunk (eval).

        Returns:
            Loss tuple in training mode, or action tensor in eval mode.

        Raises:
            ValueError: If the model is not initialized.
        """
        if self.training:
            if self.model is None or self._preprocessor is None:
                msg = "Model is not initialized"
                raise ValueError(msg)
            processed = self._preprocessor(batch.to_dict())
            return self.model(processed)
        return self.predict_action_chunk(batch)

    def compute_val_loss(self, batch: Observation) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        """Compute the validation loss.

        Returns:
            Tuple of (loss tensor, loss dict).

        Raises:
            ValueError: If the model is not initialized.
        """
        if self.model is None or self._preprocessor is None:
            msg = "Model is not initialized"
            raise ValueError(msg)
        processed = self._preprocessor(batch.to_dict())
        return self.model.compute_val_loss(processed)

    @torch.no_grad()
    def predict_action_chunk(self, batch: Observation) -> torch.Tensor:
        """Predict a chunk of actions from an observation.

        Returns:
            Denormalized action chunk tensor.

        Raises:
            ValueError: If the model is not initialized.
        """
        from physicalai.data.observation import ACTION, STATE  # noqa: PLC0415

        if self.model is None or self._preprocessor is None or self._postprocessor is None:
            msg = "Model is not initialized"
            raise ValueError(msg)
        observation = batch.to(self.device).to_dict()
        processed = self._preprocessor(observation)
        actions = self.model.predict_action_chunk(processed)
        return self._postprocessor({ACTION: actions, STATE: observation[STATE]})[ACTION]

    def prepare_ingraph_export(self, processed: dict[str, torch.Tensor]) -> None:
        """Bake the fixed image geometry into the VLM for a self-contained export.

        Args:
            processed: A preprocessor output dict containing ``input_ids``,
                ``attention_mask`` and ``image_grid_thw``.

        Raises:
            ValueError: If the model is not initialized.
        """
        if self.model is None:
            msg = "Model is not initialized"
            raise ValueError(msg)
        self.model.prepare_ingraph_export(
            cast("torch.LongTensor", processed["input_ids"]),
            processed["attention_mask"],
            cast("torch.LongTensor", processed["image_grid_thw"]),
        )

    def _build_padded_export_sample(self) -> dict[str, torch.Tensor]:
        """Preprocess the policy's sample input and right-pad it to the graph length.

        Returns:
            The padded ``processed`` dict (``input_ids``, ``attention_mask``,
            ``pixel_values``, ``image_grid_thw``, ``state``).

        Raises:
            ValueError: If the model/preprocessor are not initialized, or the
                sample prompt is longer than ``tokenizer_max_length``.
        """
        if self.model is None or self._preprocessor is None or self.sample_input is None:
            msg = "Model is not initialized"
            raise ValueError(msg)

        seq_len = self.config.tokenizer_max_length
        processed = self._preprocessor(self.sample_input)
        pad_id = self._preprocessor.processor.tokenizer.pad_token_id or 0
        cur_len = processed["input_ids"].shape[1]
        if cur_len > seq_len:
            msg = f"Sample prompt ({cur_len} tokens) exceeds tokenizer_max_length={seq_len}."
            raise ValueError(msg)
        pad = seq_len - cur_len
        if pad:
            processed["input_ids"] = torch.nn.functional.pad(processed["input_ids"], (0, pad), value=pad_id)
            processed["attention_mask"] = torch.nn.functional.pad(processed["attention_mask"], (0, pad), value=0)
        return processed

    def _bake_ingraph_export(self) -> None:
        """Pre-export hook: bake the vision geometry and OpenVINO-friendly RMSNorm.

        Registered as a ``pre_export_hooks`` entry for the OpenVINO backend
        so the base :meth:`ExportablePolicyMixin.to_openvino` runs it in place
        before tracing.
        """
        if self.model is not None:
            self.model.export_state_passthrough = self.config.action_mode == "delta"
        self.prepare_ingraph_export(self._build_padded_export_sample())

    def _get_default_export_input_sample(self) -> dict[str, torch.Tensor]:
        """Return the traced input sample for the self-contained OpenVINO graph.

        Overrides the base helper: the exported graph consumes the *padded*
        preprocessor tensors and excludes ``image_grid_thw``

        Returns:
            The padded traced-input dict, without ``image_grid_thw``.

        Raises:
            ValueError: If the preprocessor is not initialized.
        """
        if self._preprocessor is None or self.sample_input is None:
            msg = "Preprocessor is not initialized"
            raise ValueError(msg)
        processed = self._build_padded_export_sample()
        processed["pixel_values"] = torch.from_numpy(self._preprocessor.image_grid(self.sample_input))
        return {
            name: tensor
            for name, tensor in processed.items()
            if name != "image_grid_thw" and isinstance(tensor, torch.Tensor)
        }

    def training_step(self, batch: Observation, batch_idx: int) -> torch.Tensor:
        """Lightning training step.

        Returns:
            Training loss tensor.
        """
        del batch_idx
        loss, loss_dict = self(batch)
        self.log("train/loss", loss_dict["loss"], prog_bar=True)
        self.log("train/loss_mse", loss_dict["loss_mse"])
        self.log("train/loss_freq", loss_dict["loss_freq"])
        return loss

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure the AdamW optimizer and cosine-decay-with-warmup scheduler.

        Returns:
            Dict with optimizer and lr_scheduler config.
        """
        no_decay = ("bias", "norm", "ln", "rotary_emb", "adaln")
        decay_params: list[torch.nn.Parameter] = []
        no_decay_params: list[torch.nn.Parameter] = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if any(token in name.lower() for token in no_decay):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": self.config.optimizer_weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.config.optimizer_lr,
            betas=self.config.optimizer_betas,
            eps=self.config.optimizer_eps,
        )

        num_training_steps = int(self.trainer.estimated_stepping_batches)
        num_decay_steps = self.config.scheduler_decay_steps
        if num_decay_steps is None:
            num_decay_steps = num_training_steps

        scheduler = cosine_decay_with_warmup_scheduler(
            optimizer,
            peak_lr=self.config.optimizer_lr,
            decay_lr=self.config.scheduler_decay_lr,
            num_warmup_steps=self.config.scheduler_warmup_steps,
            num_decay_steps=num_decay_steps,
            num_training_steps=num_training_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        """Configure gradient clipping from the policy config."""
        clip_val = gradient_clip_val if gradient_clip_val is not None else self.config.optimizer_grad_clip_norm
        if clip_val and clip_val > 0:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=clip_val,
                gradient_clip_algorithm=gradient_clip_algorithm or "norm",
            )

    @staticmethod
    def get_supported_export_backends() -> list[str | ExportBackend]:
        """Get the list of export backends supported by the policy.

        Returns:
            list[str | ExportBackend]: The supported export backends.
        """
        return [ExportBackend.TORCH, ExportBackend.OPENVINO]

    @staticmethod
    def _coerce_dataset_feature(feature: Feature) -> Feature:
        """Return a defensive copy of a dataset feature for the schema.

        Returns:
            A new :class:`Feature` with copied normalization data and a concrete
            integer-tuple shape.
        """
        norm = feature.normalization_data
        copied_norm: NormalizationParameters | None = None
        if norm is not None:
            copied_norm = NormalizationParameters(
                mean=norm.mean,
                std=norm.std,
                min=norm.min,
                max=norm.max,
                q01=norm.q01,
                q99=norm.q99,
            )
        shape = tuple(int(dim) for dim in feature.shape) if feature.shape is not None else ()
        return Feature(
            name=str(feature.name),
            ftype=FeatureType(feature.ftype) if feature.ftype is not None else None,
            shape=shape,
            normalization_data=copied_norm,
        )

    @staticmethod
    def _dataset_features(train_dataset: Dataset) -> tuple[list[Feature], list[Feature]]:
        """Trace the input/output feature schema back from the dataset.

        Returns:
            A ``(input_features, output_features)`` tuple built from the
            dataset's observation and action features.
        """
        input_features = [XR0._coerce_dataset_feature(f) for f in train_dataset.observation_features.values()]
        output_features = [XR0._coerce_dataset_feature(f) for f in train_dataset.action_features.values()]
        return input_features, output_features

    @staticmethod
    def _feature_stat_entry(feature: Feature) -> dict[str, Any]:
        """Serialize one feature into a LeRobot-style stats dict entry.

        Returns:
            The per-feature stats entry (normalization values + metadata).
        """
        entry: dict[str, Any] = {}
        norm = feature.normalization_data
        if norm is not None:
            for stat in ("mean", "std", "min", "max", "q01", "q99"):
                value = getattr(norm, stat, None)
                if value is not None:
                    entry[stat] = value
        entry["type"] = feature.ftype.value if feature.ftype is not None else ""
        entry["name"] = feature.name if feature.name is not None else ""
        entry["shape"] = feature.shape if feature.shape is not None else ()
        return entry

    @staticmethod
    def _features_to_stats(
        input_features: list[Feature],
        output_features: list[Feature],
    ) -> dict[str, dict[str, Any]]:
        """Build the stats dict consumed by the preprocessor from typed features.

        Returns:
            A stats dict keyed like :attr:`Dataset.stats`
            (``observation.<name>`` for inputs, ``action`` for outputs).
        """
        stats: dict[str, dict[str, Any]] = {}
        for feature in input_features:
            stats[f"observation.{feature.name}"] = XR0._feature_stat_entry(feature)
        for feature in output_features:
            stats[str(feature.name)] = XR0._feature_stat_entry(feature)
        return stats

    @staticmethod
    def _stats_to_features(
        stats: dict[str, dict[str, Any]],
    ) -> tuple[list[Feature], list[Feature]]:
        """Reconstruct typed input/output features from a stats dict.

        Used when only a stats dict is available (e.g. a pretrained checkpoint),
        so :attr:`inputs_schema` / :attr:`outputs_schema` stay feature-driven.

        Returns:
            A ``(input_features, output_features)`` tuple.
        """
        input_features: list[Feature] = []
        output_features: list[Feature] = []
        for key, stat in stats.items():
            ftype_str = str(stat.get("type", ""))
            if str(FeatureType.ACTION) in ftype_str or ACTION in key:
                ftype = FeatureType.ACTION
            elif str(FeatureType.VISUAL) in ftype_str:
                ftype = FeatureType.VISUAL
            elif str(FeatureType.STATE) in ftype_str or STATE in key:
                ftype = FeatureType.STATE
            else:
                continue
            name = str(stat.get("name", key)).removeprefix("observation.")
            feature = Feature(
                name=name,
                ftype=ftype,
                shape=tuple(stat["shape"]) if stat.get("shape") else (),
                normalization_data=NormalizationParameters(
                    mean=stat.get("mean"),
                    std=stat.get("std"),
                    min=stat.get("min"),
                    max=stat.get("max"),
                    q01=stat.get("q01"),
                    q99=stat.get("q99"),
                ),
            )
            if ftype == FeatureType.ACTION:
                output_features.append(feature)
            else:
                input_features.append(feature)
        return input_features, output_features

    @property
    def input_features(self) -> list[Feature]:
        """Explicit observation feature schema.

        Returns:
            The list of input :class:`Feature` descriptors.

        Raises:
            ValueError: If the features have not been initialized yet.
        """
        if self._input_features is None:
            msg = "Model has not been initialized, no input features exist yet."
            raise ValueError(msg)
        return self._input_features

    @property
    def output_features(self) -> list[Feature]:
        """Explicit action feature schema.

        Returns:
            The list of output :class:`Feature` descriptors.

        Raises:
            ValueError: If the features have not been initialized yet.
        """
        if self._output_features is None:
            msg = "Model has not been initialized, no output features exist yet."
            raise ValueError(msg)
        return self._output_features

    @property
    def inputs_schema(self) -> list[InferenceFeature] | None:
        """Describe the policy's expected model inputs for export.

        Derived from :attr:`input_features` (traced back from the dataset when
        not provided explicitly at construction time).

        Returns:
            A list of feature descriptors covering the robot state, one image
            feature per camera view, and the language task. Returns ``None`` if
            the model or the input features have not been initialized yet.

        Raises:
            ValueError: If an input feature is missing a concrete shape.
        """
        if self.model is None or self._input_features is None:
            return None

        num_image_features = sum(1 for feature in self._input_features if feature.ftype == FeatureType.VISUAL)

        schema: list[InferenceFeature] = []
        for feature in self._input_features:
            if feature.ftype not in {FeatureType.STATE, FeatureType.VISUAL}:
                continue
            if feature.shape is None:
                msg = "input feature missing concrete shape for export"
                raise ValueError(msg)
            if feature.ftype == FeatureType.STATE:
                schema.append(
                    InferenceFeature(
                        ftype=InferenceFeatureType.STATE,
                        shape=tuple(feature.shape),
                        name=STATE,
                        dtype=InferenceFeatureDtype.FLOAT32,
                    ),
                )
            else:
                feature_name = str(feature.name or "").removeprefix("observation.").removeprefix(f"{IMAGES}.")
                name = IMAGES if num_image_features == 1 else f"{IMAGES}.{feature_name}"
                schema.append(
                    InferenceFeature(
                        ftype=InferenceFeatureType.VISUAL,
                        shape=tuple(feature.shape),
                        name=name,
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
        """Describe the policy's model output for export.

        Derived from :attr:`output_features`. Returns ``None`` if the model or
        the output features have not been initialized yet.

        Returns:
            A list with a single ``action`` feature of shape
            ``(chunk_size, *action_dim)``, where ``action_dim`` comes from the
            action feature.

        Raises:
            ValueError: If the action feature is missing a concrete shape.
        """
        if self.model is None or self._output_features is None:
            return None

        action_feature = next(
            (feature for feature in self._output_features if feature.ftype == FeatureType.ACTION),
            None,
        )
        if action_feature is None:
            return None
        if action_feature.shape is None:
            msg = "output feature missing concrete shape for export"
            raise ValueError(msg)

        schema = [
            InferenceFeature(
                ftype=InferenceFeatureType.ACTION,
                shape=(self.config.chunk_size, *tuple(action_feature.shape)),
                name=ACTION,
                dtype=InferenceFeatureDtype.FLOAT32,
            ),
        ]
        # In delta mode the graph echoes the current-frame state as a second
        # output so the Runtime ``xr0_denormalize`` can rebuild the absolute
        # action (``delta + state``). Declare it so the manifest maps the port.
        if self.config.action_mode == "delta":
            schema.append(
                InferenceFeature(
                    ftype=InferenceFeatureType.STATE,
                    shape=tuple(self.model.state_shape),
                    name=STATE,
                    dtype=InferenceFeatureDtype.FLOAT32,
                ),
            )
        return schema

    @property
    def extra_export_args(self) -> dict[str, ExportParameters]:
        """Additional export arguments for model conversion.

        Returns:
            dict[str, ExportParameters]: A mapping from backend name to its export
            parameters.
        """
        chunk_trimmer: ComponentSpec | None = None
        if self.config.chunk_size != self.config.n_action_steps:
            chunk_trimmer = ComponentSpec(
                type="action_chunk_trimmer",
                n_action_steps=self.config.n_action_steps,
            )

        torch_postproc_specs: list[ComponentSpec] = []
        if chunk_trimmer is not None:
            torch_postproc_specs.append(chunk_trimmer)

        extra_args: dict[str, ExportParameters] = {
            "torch": TorchExportParameters(
                preprocessors_specs=[ComponentSpec(type="to_float_tensor")],
                postprocessors_specs=torch_postproc_specs,
            ),
        }

        if self.model is not None and self._preprocessor is not None and self._postprocessor is not None:
            cfg = self.config
            # The OpenVINO tokenizer pads to the graph's baked prompt length.
            self._preprocessor.max_token_len = cfg.tokenizer_max_length
            # Bake the Qwen3-VL image geometry + normalization so the lightweight
            # NumPy inference preprocessor needs no HuggingFace processor at runtime.
            image_processor = self._preprocessor.processor.image_processor
            ov_preproc = ComponentSpec(
                type="xr0",
                camera_views=list(cfg.camera_views),
                max_state_dim=cfg.max_state_dim,
                image_factor=self._preprocessor.image_factor,
                image_max_pixels=self._preprocessor.image_max_pixels,
                image_mean=list(image_processor.image_mean),
                image_std=list(image_processor.image_std),
                rescale_factor=float(image_processor.rescale_factor),
                patch_size=int(image_processor.patch_size),
                merge_size=int(image_processor.merge_size),
                # Bake the state normalization so the exported graph applies
                # the exact transform used at training time (identity when
                # ``normalize_state`` is disabled -> raw-state parity).
                normalize_state=self._preprocessor.normalize_state,
                state_mean=self._preprocessor.state_mean.tolist(),
                state_std=self._preprocessor.state_std.tolist(),
            )
            ov_postproc = ComponentSpec(
                type="xr0_denormalize",
                # In ``action_mode="delta"`` the baked ``action_mean``/``action_std``
                # are per-timestep ``(chunk_size, max_action_dim)`` delta stats and
                action_mode=self._postprocessor.action_mode,
                action_mean=self._postprocessor.action_mean.tolist(),
                action_std=self._postprocessor.action_std.tolist(),
                action_dim=self._postprocessor.action_dim,
            )
            ov_postproc_specs: list[ComponentSpec] = [ov_postproc]
            if chunk_trimmer is not None:
                ov_postproc_specs.append(chunk_trimmer)

            extra_args["openvino"] = OpenVINOExportParameters(
                via_onnx=True,
                export_tokenizer=True,
                # Delta mode adds a second graph output: the current-frame state.
                outputs=["action", "state_passthrough"] if cfg.action_mode == "delta" else ["action"],
                # The NumPy preprocessor emits the prompt as a ``task`` string; a
                # sibling OpenVINO tokenizer (``tokenizer.xml``) turns it into
                # ``tokenized_prompt`` / ``tokenized_prompt_mask``.
                preprocessors_specs=[
                    ov_preproc,
                    ComponentSpec(type="ov_tokenizer", artifact="tokenizer.xml"),
                ],
                postprocessors_specs=ov_postproc_specs,
                # Rename the traced graph inputs to the tokenizer's output keys so
                # the exported ``ov_tokenizer`` step feeds them directly.
                input_name_map={
                    "input_ids": "tokenized_prompt",
                    "attention_mask": "tokenized_prompt_mask",
                },
                # Bake the vision geometry + install OpenVINO-friendly RMSNorm
                # before tracing, then rewrite boolean ``GatherND`` ops to ``i32``
                # in the written IR so it also loads on the Intel GPU plugin.
                pre_export_hooks=[self._bake_ingraph_export],
                post_export_hooks=[],
            )

        return extra_args
