# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""XR1 policy - Lightning wrapper for training, inference and export."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal, cast

import torch
from physicalai.inference.data import InferenceFeature, InferenceFeatureDtype, InferenceFeatureType
from physicalai.inference.manifest import ComponentSpec

from physicalai.data import Feature, NormalizationParameters
from physicalai.data.dataset import Dataset
from physicalai.data.observation import ACTION, IMAGES, STATE, TASK, FeatureType
from physicalai.export import ExportablePolicyMixin, ExportBackend
from physicalai.export.backends import ExportParameters, TorchExportParameters
from physicalai.policies.base import Policy
from physicalai.train.schedulers import cosine_decay_with_warmup_scheduler
from physicalai.train.utils import reformat_dataset_to_match_policy

from .config import XR1Config
from .preprocessor import XR1Postprocessor, XR1Preprocessor, make_xr1_preprocessors
from .vla import XR1Model

if TYPE_CHECKING:
    from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig

    from physicalai.data import Observation

logger = logging.getLogger(__name__)


class XR1(ExportablePolicyMixin, Policy):
    """XR1 (Xiaomi-Robotics-1) vision-language-action policy.

    A Qwen3-VL backbone paired with a DiT action expert in a
    Mixture-of-Transformers layout, trained with flow matching. See
    `arXiv:2607.15330 <https://arxiv.org/abs/2607.15330>`_.

    The model is built lazily: :meth:`setup` reads normalization statistics from the
    training datamodule, so the same policy object adapts to whichever dataset it is
    trained on. Passing ``dataset_stats`` to the constructor builds it eagerly, which
    is the path checkpoint loading takes.

    ``async_train`` and ``enable_choice_head`` are on by default, as in the reference
    implementation. Both are training-only - the Choice Policy head's query tokens are
    kept out of the action expert's key/value cache - so inference is unaffected either
    way, and both can be turned off to trade fidelity for throughput.

    Memory: the released configuration is 5.04B parameters, 9.4 GiB of bf16 weights.
    That fits a 24 GB card for inference, but a full AdamW fine-tune needs roughly
    66 GiB. Set ``freeze_vlm=True`` to train only the action expert and projectors
    (~0.7B parameters), which does fit.

    Example:
        Training:

        >>> policy = XR1(freeze_vlm=True)
        >>> trainer = physicalai.train.Trainer(max_epochs=100)
        >>> trainer.fit(policy, datamodule)

        Inference:

        >>> policy = XR1.load_from_checkpoint("checkpoint.ckpt")
        >>> action = policy.select_action(obs)
    """

    def __init__(  # noqa: PLR0913
        self,
        # Backbone
        vlm_model_id: str = "Qwen/Qwen3-VL-4B-Instruct",
        vlm_attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = "sdpa",
        dtype: Literal["bfloat16", "float32"] = "bfloat16",
        # Input / output structure
        n_obs_steps: int = 1,
        chunk_size: int = 30,
        n_action_steps: int = 30,
        max_state_dim: int = 32,
        max_action_dim: int = 32,
        state_len: int = 1,
        state_slot_map: tuple[int, ...] | None = None,
        action_slot_map: tuple[int, ...] | None = None,
        # Action expert
        dit_num_layers: int = 36,
        dit_hidden_size: int = 1024,
        dit_head_dim: int = 128,
        dit_kv_heads: int = 8,
        *,
        vlm_pretrained: bool = True,
        vlm_config_overrides: dict[str, Any] | None = None,
        # Flow matching
        num_inference_steps: int = 5,
        flow_sampling: Literal["beta", "logit_normal", "uniform"] = "beta",
        beta_alpha: float = 1.5,
        beta_beta: float = 1.0,
        training_repeat: int = 4,
        prefix_mask_prob: float = 0.5,
        async_train: bool = True,
        # Loss terms
        enable_freq: bool = True,
        freq_coefficient: float = 1.0,
        freq_excluded_dims: tuple[int, ...] = (17, 18, 19),
        enable_choice_head: bool = True,
        n_choices: int = 5,
        # Observation preprocessing
        image_resolution: tuple[int, int] = (256, 256),
        camera_views: tuple[str, ...] = ("base", "wrist_left"),
        tokenizer_max_length: int = 256,
        # Optimization
        gradient_checkpointing: bool = True,
        freeze_vlm: bool = False,
        freeze_vision_encoder: bool = False,
        normalization_mode: Literal["MEAN_STD", "QUANTILES"] = "MEAN_STD",
        optimizer_lr: float = 1.0e-4,
        optimizer_betas: tuple[float, float] = (0.9, 0.95),
        optimizer_eps: float = 1e-8,
        optimizer_weight_decay: float = 0.1,
        optimizer_grad_clip_norm: float = 1.0,
        scheduler_warmup_steps: int = 2_000,
        scheduler_decay_steps: int | None = 30_000,
        scheduler_decay_lr: float = 5.0e-7,
        # Eager initialization
        dataset_stats: dict[str, dict[str, list[float] | str | tuple]] | None = None,
    ) -> None:
        """Initialize the XR1 policy.

        Every argument mirrors a field of :class:`XR1Config`; see that class for the
        meaning and provenance of each default.
        """
        super().__init__(n_action_steps=n_action_steps)

        self.config = XR1Config(
            vlm_model_id=vlm_model_id,
            vlm_pretrained=vlm_pretrained,
            vlm_config_overrides=vlm_config_overrides,
            vlm_attn_implementation=vlm_attn_implementation,
            dtype=dtype,
            n_obs_steps=n_obs_steps,
            chunk_size=chunk_size,
            n_action_steps=n_action_steps,
            max_state_dim=max_state_dim,
            max_action_dim=max_action_dim,
            state_len=state_len,
            state_slot_map=None if state_slot_map is None else tuple(state_slot_map),
            action_slot_map=None if action_slot_map is None else tuple(action_slot_map),
            dit_num_layers=dit_num_layers,
            dit_hidden_size=dit_hidden_size,
            dit_head_dim=dit_head_dim,
            dit_kv_heads=dit_kv_heads,
            num_inference_steps=num_inference_steps,
            flow_sampling=flow_sampling,
            beta_alpha=beta_alpha,
            beta_beta=beta_beta,
            training_repeat=training_repeat,
            prefix_mask_prob=prefix_mask_prob,
            async_train=async_train,
            enable_freq=enable_freq,
            freq_coefficient=freq_coefficient,
            freq_excluded_dims=tuple(freq_excluded_dims),
            enable_choice_head=enable_choice_head,
            n_choices=n_choices,
            image_resolution=tuple(image_resolution),  # type: ignore[arg-type]
            camera_views=tuple(camera_views),
            tokenizer_max_length=tokenizer_max_length,
            gradient_checkpointing=gradient_checkpointing,
            freeze_vlm=freeze_vlm,
            freeze_vision_encoder=freeze_vision_encoder,
            normalization_mode=normalization_mode,
            optimizer_lr=optimizer_lr,
            optimizer_betas=tuple(optimizer_betas),  # type: ignore[arg-type]
            optimizer_eps=optimizer_eps,
            optimizer_weight_decay=optimizer_weight_decay,
            optimizer_grad_clip_norm=optimizer_grad_clip_norm,
            scheduler_warmup_steps=scheduler_warmup_steps,
            scheduler_decay_steps=scheduler_decay_steps,
            scheduler_decay_lr=scheduler_decay_lr,
        )

        self.save_hyperparameters(ignore=["config"])
        self._set_hparam_keys()

        self.model: XR1Model | None = None
        self._preprocessor: XR1Preprocessor | None = None
        self._postprocessor: XR1Postprocessor | None = None
        self._dataset_stats = dataset_stats

        if dataset_stats is not None:
            self._initialize_model(dataset_stats)

    def _set_hparam_keys(self) -> None:
        """Sync checkpoint hparams from the resolved policy config."""
        for key, value in self.config.__dict__.items():
            if key not in self.hparams:
                continue
            self.hparams[key] = value
        self.hparams["config"] = self.config.to_dict()

    @staticmethod
    def features_from_stats(
        dataset_stats: dict[str, dict[str, list[float] | str | tuple]],
    ) -> dict[str, Feature]:
        """Rebuild the feature schema from dataset statistics.

        Args:
            dataset_stats: Per-feature statistics as exposed by
                :attr:`physicalai.data.dataset.Dataset.stats`.

        Returns:
            Mapping from feature name to :class:`~physicalai.data.Feature`.
        """
        features: dict[str, Feature] = {}
        for stat in dataset_stats.values():
            name = str(stat["name"])
            features[name] = Feature(
                name=name,
                ftype=cast("FeatureType", stat["type"]),
                shape=cast("tuple[int, ...]", stat["shape"]),
                normalization_data=NormalizationParameters(
                    mean=cast("list[float]", stat.get("mean")),
                    std=cast("list[float]", stat.get("std")),
                    min=cast("list[float]", stat.get("min")),
                    max=cast("list[float]", stat.get("max")),
                    q01=cast("list[float]", stat.get("q01")),
                    q99=cast("list[float]", stat.get("q99")),
                ),
            )
        return features

    def _initialize_model(
        self,
        dataset_stats: dict[str, dict[str, list[float] | str | tuple]],
    ) -> None:
        """Build the model and its pre/postprocessors.

        Args:
            dataset_stats: Normalization statistics from the training dataset.
        """
        features = self.features_from_stats(dataset_stats)
        self.model = XR1Model(self.config)
        self._preprocessor, self._postprocessor = make_xr1_preprocessors(self.config, features)
        self._dataset_stats = dataset_stats

    def setup(self, stage: str) -> None:
        """Build the model from the datamodule before fit/validate/test/predict.

        Args:
            stage: Lightning stage name, unused.

        Raises:
            TypeError: If the training dataset is not a physicalai ``Dataset``.
        """
        del stage

        datamodule = self.trainer.datamodule  # type: ignore[attr-defined]
        train_dataset = datamodule.train_dataset

        if not isinstance(train_dataset, Dataset):
            msg = f"Expected physicalai Dataset, got {type(train_dataset)}"
            raise TypeError(msg)

        stats_dict = train_dataset.stats

        if self.model is not None:
            # Fine-tuning path: keep the weights, refresh normalization so it
            # matches the new data distribution.
            features = self.features_from_stats(stats_dict)
            self._preprocessor, self._postprocessor = make_xr1_preprocessors(self.config, features)
            self._dataset_stats = stats_dict
            self.hparams["dataset_stats"] = stats_dict
            reformat_dataset_to_match_policy(self, datamodule)
            return

        self.hparams["dataset_stats"] = stats_dict
        self._initialize_model(stats_dict)
        reformat_dataset_to_match_policy(self, datamodule)

    def _components(self) -> tuple[XR1Model, XR1Preprocessor, XR1Postprocessor]:
        """Return the model and processors, or explain why they are missing.

        Returning them rather than merely asserting keeps the types narrowed for
        callers, which is what the model attribute being ``XR1Model | None`` costs.

        Returns:
            The model, preprocessor and postprocessor.

        Raises:
            ValueError: If the policy has not been initialized yet.
        """
        if self.model is None or self._preprocessor is None or self._postprocessor is None:
            msg = (
                "Model is not initialized. Pass dataset_stats to the constructor, or let "
                "Trainer.fit() build it from the datamodule."
            )
            raise ValueError(msg)
        return self.model, self._preprocessor, self._postprocessor

    def forward(self, batch: Observation) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        """Compute the training loss, or predict actions in eval mode.

        Args:
            batch: Observation batch.

        Returns:
            ``(loss, metrics)`` while training, otherwise the predicted action chunk.
        """
        if self.training:
            model, preprocessor, _ = self._components()
            return model(preprocessor(batch.to_dict()))
        return self.predict_action_chunk(batch)

    def compute_val_loss(self, batch: Observation) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        """Score a full action rollout against the ground truth.

        Args:
            batch: Observation batch including actions.

        Returns:
            ``(loss, metrics)`` with the masked action MSE.
        """
        model, preprocessor, _ = self._components()
        return model.compute_val_loss(preprocessor(batch.to_dict()))

    @torch.no_grad()
    def predict_action_chunk(self, batch: Observation) -> torch.Tensor:
        """Predict an action chunk for an observation batch.

        Args:
            batch: Observation batch.

        Returns:
            Denormalized actions of shape ``(batch, chunk_size, action_dim)``.
        """
        model, preprocessor, postprocessor = self._components()
        actions = model.predict_action_chunk(preprocessor(batch.to(self.device).to_dict()))
        return postprocessor({ACTION: actions})[ACTION]

    def training_step(self, batch: Observation, batch_idx: int) -> torch.Tensor:
        """Run one training step.

        Args:
            batch: Observation batch.
            batch_idx: Batch index, unused.

        Returns:
            The training loss.
        """
        del batch_idx
        loss, loss_dict = self(batch)
        self.log("train/loss", loss_dict["loss"], prog_bar=True)
        for key in ("loss_mse", "loss_freq", "loss_choice", "loss_score"):
            if key in loss_dict:
                self.log(f"train/{key}", loss_dict[key])
        return loss

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        """Configure AdamW with a warmup plus cosine-decay schedule.

        Returns:
            Lightning optimizer configuration.
        """
        params = [p for p in self.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW(
            params,
            lr=self.config.optimizer_lr,
            weight_decay=self.config.optimizer_weight_decay,
            betas=self.config.optimizer_betas,
            eps=self.config.optimizer_eps,
        )

        num_training_steps = int(self.trainer.estimated_stepping_batches)
        num_decay_steps = self.config.scheduler_decay_steps
        if num_decay_steps is None:
            num_decay_steps = num_training_steps
            logger.info("scheduler_decay_steps=None, using total training steps: %s", num_decay_steps)

        scheduler = cosine_decay_with_warmup_scheduler(
            optimizer,
            peak_lr=self.config.optimizer_lr,
            decay_lr=self.config.scheduler_decay_lr,
            num_warmup_steps=self.config.scheduler_warmup_steps,
            num_decay_steps=num_decay_steps,
            num_training_steps=num_training_steps,
        )

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        """Clip gradients using the configured norm.

        Args:
            optimizer: The optimizer being stepped.
            gradient_clip_val: Trainer-provided clip value, if any.
            gradient_clip_algorithm: Clipping algorithm, defaults to ``"norm"``.
        """
        clip_val = gradient_clip_val if gradient_clip_val is not None else self.config.optimizer_grad_clip_norm

        if clip_val and clip_val > 0:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=clip_val,
                gradient_clip_algorithm=gradient_clip_algorithm or "norm",
            )

    @staticmethod
    def get_supported_export_backends() -> list[str | ExportBackend]:
        """List the export backends this policy supports.

        Returns:
            Supported backends. Graph backends are not listed yet: the backbone's
            key/value cache and the iterative sampler need a component-split export,
            which is tracked separately.
        """
        return [ExportBackend.TORCH]

    @property
    def inputs_schema(self) -> list[InferenceFeature] | None:
        """Describe the policy inputs for export and runtime loading.

        Returns:
            Feature descriptors for state, each camera, and the language task, or
            ``None`` when the policy has not been initialized.
        """
        if self.model is None or self._dataset_stats is None:
            return None

        schema: list[InferenceFeature] = []
        num_image_features = sum(
            1 for feature in self._dataset_stats.values() if str(FeatureType.VISUAL) in str(feature.get("type", ""))
        )

        for feature_id, feature in self._dataset_stats.items():
            feature_type = str(feature.get("type", ""))
            if STATE in feature_id:
                schema.append(
                    InferenceFeature(
                        ftype=InferenceFeatureType.STATE,
                        shape=cast("tuple", feature["shape"]),
                        name=STATE,
                        dtype=InferenceFeatureDtype.FLOAT32,
                    ),
                )
            elif str(FeatureType.VISUAL) in feature_type:
                feature_name = (
                    str(feature.get("name", feature_id)).removeprefix("observation.").removeprefix(f"{IMAGES}.")
                )
                name = IMAGES if num_image_features == 1 else f"{IMAGES}.{feature_name}"
                schema.append(
                    InferenceFeature(
                        ftype=InferenceFeatureType.VISUAL,
                        shape=cast("tuple", feature["shape"]),
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
        """Describe the policy output for export.

        Returns:
            A single ``action`` feature of shape ``(chunk_size, action_dim)``, or
            ``None`` when the policy has not been initialized.
        """
        if self.model is None or self._dataset_stats is None:
            return None

        action_shape = cast("tuple", self._dataset_stats[ACTION]["shape"])
        return [
            InferenceFeature(
                ftype=InferenceFeatureType.ACTION,
                shape=(self.config.chunk_size, *action_shape),
                name=ACTION,
                dtype=InferenceFeatureDtype.FLOAT32,
            ),
        ]

    @property
    def extra_export_args(self) -> dict[str, ExportParameters]:
        """Per-backend export parameters and manifest component specs.

        Returns:
            Mapping from backend name to its export parameters.

        Raises:
            ValueError: If dataset statistics are not available.
        """
        if self._dataset_stats is None:
            msg = (
                "Dataset stats are required for export. Initialize the policy with dataset_stats"
                " or train for at least one epoch to populate them."
            )
            raise ValueError(msg)

        postproc_specs = []
        if self.config.chunk_size != self.config.n_action_steps:
            postproc_specs.append(
                ComponentSpec(type="action_chunk_trimmer", n_action_steps=self.config.n_action_steps),
            )

        return {
            "torch": TorchExportParameters(
                preprocessors_specs=[ComponentSpec(type="to_float_tensor")],
                postprocessors_specs=postproc_specs,
            ),
        }
