# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Architecture derived from https://github.com/OpenHelix-Team/VLA-Adapter (MIT).

"""VLA-Adapter policy — Lightning wrapper for training and inference.

A frozen 0.5B Prismatic VLM (fused DINOv2 + SigLIP towers feeding Qwen2.5-0.5B)
paired with a lightweight Bridge-Attention head. Only the head, proprio
projector, visual projector and action queries train by default, which is what
makes it trainable on a single consumer GPU.

Actions come from **continuous L1 regression in one forward pass** — no
diffusion or flow-matching loop — so training and inference share a code path
and the exported graph stays static.

Export beyond the Torch backend is added later; ``get_supported_export_backends``
currently advertises Torch only.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch

from physicalai.data.observation import ACTION
from physicalai.export import ExportablePolicyMixin, ExportBackend
from physicalai.policies.base import Policy
from physicalai.policies.vla_adapter.config import VLAAdapterConfig
from physicalai.policies.vla_adapter.model import VLAAdapterModel
from physicalai.policies.vla_adapter.preprocessor import VLAAdapterPostprocessor, VLAAdapterPreprocessor
from physicalai.train.schedulers import cosine_decay_with_warmup_scheduler
from physicalai.train.utils import reformat_dataset_to_match_policy

if TYPE_CHECKING:
    from physicalai.data import Observation

logger = logging.getLogger(__name__)


class VLAAdapter(ExportablePolicyMixin, Policy):
    """Lightning policy for the VLA-Adapter vision-language-action model.

    Example:
        >>> from physicalai.policies import get_policy
        >>> policy = get_policy("vla_adapter")  # doctest: +SKIP
    """

    def __init__(  # noqa: PLR0913
        self,
        # Input / output structure
        n_obs_steps: int = 1,
        chunk_size: int = 8,
        n_action_steps: int = 8,
        max_state_dim: int = 8,
        max_action_dim: int = 7,
        # Image preprocessing
        image_size: tuple[int, int] = (224, 224),
        image_key_reorder_map: dict[str, int] | None = None,
        num_cameras: int = 0,
        num_images_in_input: int = 2,
        *,
        # Architecture
        tokenizer_max_length: int = 48,
        num_task_tokens: int = 512,
        num_action_queries: int = 64,
        head_num_heads: int = 8,
        llm_model_name: str = "Qwen/Qwen2.5-0.5B",
        load_pretrained_backbone: bool = True,
        vision_backbone_ids: tuple[str, str] = (
            "vit_large_patch14_reg4_dinov2.lvd142m",
            "vit_so400m_patch14_siglip_224",
        ),
        arch_specifier: str = "no-align+fused-gelu-mlp",
        use_proprio: bool = True,
        # Trainability of the two pretrained backbones; everything else always trains
        train_vision_backbone: bool = False,
        train_llm: bool = False,
        # Training presets
        optimizer_lr: float = 5e-4,
        optimizer_betas: tuple[float, float] = (0.9, 0.95),
        optimizer_eps: float = 1e-8,
        optimizer_weight_decay: float = 1e-10,
        optimizer_grad_clip_norm: float = 1.0,
        scheduler_warmup_steps: int = 1_000,
        scheduler_decay_steps: int = 30_000,
        scheduler_decay_lr: float = 2.5e-6,
        # Eager initialization (for checkpoint loading)
        dataset_stats: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Initialize the policy.

        Builds a :class:`VLAAdapterConfig` from the explicit arguments and saves
        it as Lightning hyperparameters. The model is created lazily in
        :meth:`setup`, or immediately when ``dataset_stats`` is given.
        """
        super().__init__(n_action_steps=n_action_steps)

        self.config = VLAAdapterConfig(
            n_obs_steps=n_obs_steps,
            chunk_size=chunk_size,
            n_action_steps=n_action_steps,
            max_state_dim=max_state_dim,
            max_action_dim=max_action_dim,
            image_size=image_size,
            image_key_reorder_map=image_key_reorder_map or {},
            num_cameras=num_cameras,
            num_images_in_input=num_images_in_input,
            tokenizer_max_length=tokenizer_max_length,
            num_task_tokens=num_task_tokens,
            num_action_queries=num_action_queries,
            head_num_heads=head_num_heads,
            llm_model_name=llm_model_name,
            load_pretrained_backbone=load_pretrained_backbone,
            vision_backbone_ids=vision_backbone_ids,
            arch_specifier=arch_specifier,
            use_proprio=use_proprio,
            train_vision_backbone=train_vision_backbone,
            train_llm=train_llm,
            optimizer_lr=optimizer_lr,
            optimizer_betas=optimizer_betas,
            optimizer_eps=optimizer_eps,
            optimizer_weight_decay=optimizer_weight_decay,
            optimizer_grad_clip_norm=optimizer_grad_clip_norm,
            scheduler_warmup_steps=scheduler_warmup_steps,
            scheduler_decay_steps=scheduler_decay_steps,
            scheduler_decay_lr=scheduler_decay_lr,
        )

        self.save_hyperparameters(ignore=["config"])
        self._set_hparam_keys()

        self.model: VLAAdapterModel | None = None
        self._preprocessor: VLAAdapterPreprocessor | None = None
        self._postprocessor: VLAAdapterPostprocessor | None = None

        if dataset_stats is not None:
            self._initialize_model(dataset_stats)

        self._dataset_stats = dataset_stats

    def _set_hparam_keys(self) -> None:
        """Sync top-level checkpoint hparams from the resolved policy config."""
        for key, value in self.config.__dict__.items():
            if key not in self.hparams:
                continue
            self.hparams[key] = value
        self.hparams["config"] = self.config.to_dict()

    def _initialize_model(self, dataset_stats: dict[str, dict[str, Any]]) -> None:
        """Build the model and its pre/postprocessors.

        Args:
            dataset_stats: Dataset normalization statistics.
        """
        self.model = VLAAdapterModel(self.config, dataset_stats=dataset_stats)
        self._update_preprocessor_stats(dataset_stats)

    def _update_preprocessor_stats(self, dataset_stats: dict[str, dict[str, Any]]) -> None:
        """Rebuild pre- and postprocessors from dataset statistics.

        Args:
            dataset_stats: Dataset normalization statistics.
        """
        from physicalai.policies.vla_adapter.preprocessor import make_vla_adapter_preprocessors  # noqa: PLC0415

        self._preprocessor, self._postprocessor = make_vla_adapter_preprocessors(
            max_state_dim=self.config.max_state_dim,
            max_action_dim=self.config.max_action_dim,
            stats=dataset_stats,
            image_resolution=self.config.image_size,
            vision_backbone_ids=self.config.vision_backbone_ids,
            image_key_reorder_map=self.config.image_key_reorder_map,
            num_cameras=self.config.num_cameras,
            max_token_len=self.config.tokenizer_max_length,
            tokenizer_name=self.config.llm_model_name,
        )
        self._dataset_stats = dataset_stats
        self.hparams["dataset_stats"] = dataset_stats
        if self.model is not None:
            self.model.set_dataset_stats(dataset_stats)

    def setup(self, stage: str) -> None:
        """Build the model from the datamodule (lazy initialization path).

        Args:
            stage: Lightning stage (unused).

        Raises:
            TypeError: If the train dataset is not a physicalai ``Dataset``.
        """
        del stage

        from physicalai.data.dataset import Dataset  # noqa: PLC0415

        datamodule = self.trainer.datamodule  # type: ignore[attr-defined]
        train_dataset = datamodule.train_dataset

        if not isinstance(train_dataset, Dataset):
            msg = f"Expected physicalai Dataset, got {type(train_dataset)}"
            raise TypeError(msg)

        stats_dict = train_dataset.stats

        if self.model is not None:
            self._update_preprocessor_stats(stats_dict)
            reformat_dataset_to_match_policy(self, datamodule)
            return

        self.hparams["dataset_stats"] = stats_dict
        self._initialize_model(stats_dict)
        reformat_dataset_to_match_policy(self, datamodule)

    def forward(self, batch: Observation) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        """Run a forward pass.

        Args:
            batch: Input observation batch.

        Returns:
            ``(loss, loss_dict)`` in training mode, else the action chunk.

        Raises:
            ValueError: If the model has not been initialized.
        """
        if self.training:
            if self.model is None or self._preprocessor is None:
                msg = "Model is not initialized"
                raise ValueError(msg)
            processed_batch = self._preprocessor(batch.to_dict())
            return self.model(processed_batch)
        return self.predict_action_chunk(batch)

    @torch.no_grad()
    def predict_action_chunk(self, batch: Observation) -> torch.Tensor:
        """Predict a chunk of actions from an observation batch.

        Args:
            batch: Input observation batch.

        Returns:
            Denormalized ``(B, chunk_size, action_dim)``.

        Raises:
            ValueError: If the model has not been initialized.
        """
        if self.model is None or self._preprocessor is None or self._postprocessor is None:
            msg = "Model is not initialized"
            raise ValueError(msg)

        processed_batch = self._preprocessor(batch.to(self.device).to_dict())
        chunk = self.model.predict_action_chunk(processed_batch)
        return self._postprocessor({ACTION: chunk})[ACTION]

    def training_step(self, batch: Observation, batch_idx: int) -> torch.Tensor:
        """Lightning training step.

        Args:
            batch: Input batch.
            batch_idx: Batch index (unused).

        Returns:
            Loss tensor for backpropagation.
        """
        del batch_idx
        loss, loss_dict = self(batch)
        self.log("train/loss", loss_dict["loss"], prog_bar=True)
        return loss

    def compute_val_loss(self, batch: Observation) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        """Compute validation loss on a batch.

        Args:
            batch: Observation batch with ground-truth actions.

        Returns:
            ``(loss, loss_dict)``.

        Raises:
            ValueError: If the model has not been initialized.
        """
        if self.model is None or self._preprocessor is None:
            msg = "Model is not initialized"
            raise ValueError(msg)
        processed_batch = self._preprocessor(batch.to_dict())
        return self.model.compute_val_loss(processed_batch)

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure the optimizer and learning-rate scheduler.

        Returns:
            Lightning optimizer configuration dict.
        """
        params = [p for p in self.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW(
            params,
            lr=self.config.optimizer_lr,
            weight_decay=self.config.optimizer_weight_decay,
            betas=self.config.optimizer_betas,
            eps=self.config.optimizer_eps,
        )

        scheduler = cosine_decay_with_warmup_scheduler(
            optimizer,
            peak_lr=self.config.optimizer_lr,
            decay_lr=self.config.scheduler_decay_lr,
            num_warmup_steps=self.config.scheduler_warmup_steps,
            num_decay_steps=self.config.scheduler_decay_steps,
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
        """Apply the configured gradient clipping.

        Args:
            optimizer: The optimizer being stepped.
            gradient_clip_val: Clip value from the Trainer, if any.
            gradient_clip_algorithm: Clipping algorithm from the Trainer.
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
        """Get the export backends supported by this policy.

        Returns:
            Supported export backends.
        """
        return [ExportBackend.TORCH]
