# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Vendored from RLWRLD/RLDX-1 (Apache-2.0)

"""RLDX-1 Policy - first-party Lightning wrapper for RLWRLD's RLDX-1 VLA.

This module provides the PyTorch Lightning policy for training and inference
with RLDX-1, a flow-matching Vision-Language-Action model built on a Qwen3-VL-8B
backbone and a Multi-Stream Action Transformer (MSAT) action head.

Scope (v1): pre-train (PT) -> fine-tune (FT) path only, starting from
``RLWRLD/RLDX-1-PT``. See ``library/docs/rldx-1-integration.md``.

.. note::
    The architecture port is landing incrementally. The Lightning contract
    (config, dual-path init, action queue) is in place; the underlying
    :class:`~physicalai.policies.rldx1.model.Rldx1Model` forward passes are
    filled in by subsequent component ports.

## Quick Start

```python
from physicalai.data.lerobot import LeRobotDataModule
from physicalai.policies.rldx1 import Rldx1
from physicalai.train import Trainer

# Default: LoRA on both backbone and action model
policy = Rldx1(base_model_path="RLWRLD/RLDX-1-PT")

# Paper Table 6, Row 1: Full FT backbone + LoRA action (62.67% success)
policy = Rldx1(
    base_model_path="RLWRLD/RLDX-1-PT",
    backbone_use_lora=False,  # Full fine-tune top-4 LLM layers
    action_model_use_lora=True,     # LoRA on MSAT (r=64)
)

datamodule = LeRobotDataModule(repo_id="<user dataset>", train_batch_size=4)
trainer = Trainer(max_steps=60000, precision="bf16-mixed")
trainer.fit(policy, datamodule)
```
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import RemoteEntryNotFoundError
from transformers.optimization import Adafactor

from physicalai.data import Dataset, Observation
from physicalai.policies.base import Policy
from physicalai.policies.rldx1.config import Rldx1Config
from physicalai.policies.rldx1.model import Rldx1Model
from physicalai.policies.rldx1.pretrained_utils import extract_dataset_stats, retrieve_safetensors_shards
from physicalai.train.schedulers import cosine_decay_with_warmup_scheduler
from physicalai.train.utils import reformat_dataset_to_match_policy

try:
    from lightning.pytorch.utilities.types import OptimizerLRScheduler
except ImportError:
    OptimizerLRScheduler = Any  # type: ignore[assignment, misc]

from .preprocessor import make_rldx1_transforms
from .vtc_buffer import VtcWindowBuffer

if TYPE_CHECKING:
    from .preprocessor import Rldx1Postprocessor, Rldx1Preprocessor

logger = logging.getLogger(__name__)


class Rldx1(Policy):
    """RLDX-1 Policy - first-party Lightning wrapper.

    All hyperparameters are explicit in the signature for discoverability.
    Supports the dual-path initialization shared across Studio policies:

    - **Lazy path**: ``Rldx1()`` + ``trainer.fit()`` - model built in ``setup()``
      once dataset features are known.
    - **Eager path**: ``Rldx1.load_from_checkpoint()`` or ``Rldx1(env_action_dim=...)``
      - model built immediately.

    Args:
        n_action_steps: Number of action steps to execute per chunk.
        max_state_dim: Maximum state dimension (shorter states zero-padded).
        max_action_dim: Maximum action dimension (shorter actions zero-padded).
        base_model_path: HuggingFace model ID or path to the base checkpoint.
        revision: Pinned git commit SHA for the checkpoint download (lib.security rule 9).
        model_name: HuggingFace ID of the Qwen3-VL backbone.
        select_layer: VLM hidden layer used as cognition features.
        attn_implementation: Attention backend ('sdpa', 'flash_attention_2', 'eager').
        n_cog_tokens: Number of cognition tokens routed to MSAT.
        tune_top_llm_layers: Number of top LLM layers to fine-tune.
        tune_llm: Whether to fine-tune the entire LLM backbone (all decoder
            layers + input embeddings + lm_head). Overrides tune_top_llm_layers.
        backbone_trainable_params_fp32: Whether to cast trainable backbone
            parameters to float32 after bf16 loading for optimizer stability.
            Default ``False`` here, diverging from upstream's ``True``.
            TODO(Eugene): upstream defaults this to ``True``, but the fp32
            copies of trainable backbone params OOM on an A100. DeepSpeed
            ZeRO-Offload (CPU) avoids the OOM but is very slow in practice.
            Explore a better way (e.g. selective fp32 casting, ZeRO-3 without
            offload, or partial offload) to re-enable ``True`` by default.
        tune_visual: Whether to fine-tune the vision tower.
        tune_projector: Whether to fine-tune the projectors.
        tune_diffusion_model: Whether to fine-tune the MSAT action model.
        tune_vlln: Whether to fine-tune the VLM-output layer norm in the action head.
        num_inference_timesteps: Number of flow-matching denoising steps at inference.
        backbone_use_lora: Whether to use LoRA on the backbone top layers.
            Default False (full fine-tuning). Set to True for LoRA.
        action_use_lora: Whether to use LoRA on the MSAT action model.
            Default False (full fine-tuning). Set to True for LoRA (Paper Table 6, row 1).
        learning_rate: Learning rate for the optimizer.
        weight_decay: Weight decay for the optimizer.
        warmup_ratio: Warmup ratio (0.0-1.0) of total training steps.
        scheduler_decay_lr: Final learning rate after cosine decay (default 1e-5).
        use_bf16: Whether to use bfloat16 precision.
        gradient_checkpointing: Whether to enable activation checkpointing in
            MSAT during training.
        video_length: Number of VTC temporal frames per observation step (default 4).
        video_stride: Action-step stride between VTC video frames (default 2).
            With ``video_length=4, video_stride=2`` the offsets are ``[-6,-4,-2,0]``
            (600 ms at 10 fps). Set ``video_stride=1`` for contiguous frames.
        clip_outliers: Clip normalized state/action to ``[-1, 1]`` at train and
            inference (upstream default ``True``). Set ``False`` (Pi05-style, no
            clip) for wide-range action spaces where ``QUANTILES`` bounds would
            truncate task-critical extremes (e.g. PushT).
        env_action_dim: Environment action dimension. If provided, enables eager init.
        dataset_stats: Dataset normalization statistics for eager init.
    """

    def __init__(  # noqa: PLR0913
        self,
        # Model architecture
        n_action_steps: int = 16,
        max_state_dim: int = 64,
        max_action_dim: int = 64,
        # Model source
        pretrained_name_or_path: str | None = "RLWRLD/RLDX-1-PT",
        revision: str | None = None,
        # Backbone
        attn_implementation: Literal["sdpa", "flash_attention_2"] = "sdpa",
        # Fine-tuning control
        *,
        tune_top_llm_layers: int = 6,
        tune_llm: bool = False,
        backbone_trainable_params_fp32: bool = False,
        tune_visual: bool = True,
        tune_projector: bool = True,
        use_vlln: bool = False,
        tune_diffusion_model: bool = True,
        tune_vlln: bool = True,
        num_inference_timesteps: int = 4,
        backbone_use_lora: bool = False,
        action_model_use_lora: bool = False,
        # Optimizer
        optim: Literal["adamw_torch", "adamw_torch_fused", "adafactor"] = "adamw_torch",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.05,
        scheduler_decay_lr: float = 1e-5,
        # Precision / compilation
        use_bf16: bool = True,
        gradient_checkpointing: bool = True,
        # VTC video window
        video_length: int = 4,
        video_stride: int = 2,
        # Image geometry
        image_min_area: int | None = None,
        # Normalization
        clip_outliers: bool = True,
        dataset_stats: dict[str, dict[str, list[float] | str | tuple]] | None = None,
        embodiment_tag: str = "general_embodiment",
    ) -> None:
        """Initialize the RLDX-1 policy and save hyperparameters."""
        super().__init__(n_action_steps=n_action_steps)

        shard_files = None
        if pretrained_name_or_path is not None:
            # dataset_stats already provided (e.g. restored from a checkpoint's
            # hparams during load_from_checkpoint) takes precedence -- _from_hf's
            # own extract_dataset_stats() is only a narrow, defaulted fallback.
            self.config, hf_dataset_stats, shard_files = self._from_hf(
                pretrained_name_or_path,
                revision=revision,
                max_state_dim=max_state_dim,
                max_action_dim=max_action_dim,
                attn_implementation=attn_implementation,
                # Fine-tuning
                tune_diffusion_model=tune_diffusion_model,
                use_vlln=use_vlln,
                tune_vlln=tune_vlln,
                tune_top_llm_layers=tune_top_llm_layers,
                tune_llm=tune_llm,
                backbone_trainable_params_fp32=backbone_trainable_params_fp32,
                tune_visual=tune_visual,
                tune_projector=tune_projector,
                num_inference_timesteps=num_inference_timesteps,
                backbone_use_lora=backbone_use_lora,
                action_model_use_lora=action_model_use_lora,
                # optimizer
                optim=optim,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                warmup_ratio=warmup_ratio,
                scheduler_decay_lr=scheduler_decay_lr,
                # Precision / compilation
                use_bf16=use_bf16,
                gradient_checkpointing=gradient_checkpointing,
                # VTC video window
                video_length=video_length,
                video_stride=video_stride,
                # Image geometry
                image_min_area=image_min_area,
                # Normalization
                clip_outliers=clip_outliers,
                # Action prediciton
                action_horizon=n_action_steps,
                embodiment_tag=embodiment_tag,
            )
            if dataset_stats is None:
                dataset_stats = hf_dataset_stats
        else:
            self.config = Rldx1Config(
                base_model_path=pretrained_name_or_path,
                revision=revision,
                max_state_dim=max_state_dim,
                max_action_dim=max_action_dim,
                attn_implementation=attn_implementation,
                # Fine-tuning
                tune_diffusion_model=tune_diffusion_model,
                use_vlln=use_vlln,
                tune_vlln=tune_vlln,
                tune_top_llm_layers=tune_top_llm_layers,
                tune_llm=tune_llm,
                backbone_trainable_params_fp32=backbone_trainable_params_fp32,
                tune_visual=tune_visual,
                tune_projector=tune_projector,
                num_inference_timesteps=num_inference_timesteps,
                backbone_use_lora=backbone_use_lora,
                action_model_use_lora=action_model_use_lora,
                # optimizer
                optim=optim,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                warmup_ratio=warmup_ratio,
                scheduler_decay_lr=scheduler_decay_lr,
                # Precision / compilation
                use_bf16=use_bf16,
                gradient_checkpointing=gradient_checkpointing,
                # VTC video window
                video_length=video_length,
                video_stride=video_stride,
                # Image geometry
                image_min_area=image_min_area,
                # Normalization
                clip_outliers=clip_outliers,
                action_horizon=n_action_steps,
                embodiment_tag=embodiment_tag,
            )

        # Save `pretrained_name_or_path` so load_from_checkpoint() reconstructs
        # from the same base repo the checkpoint was actually fine-tuned from,
        # instead of silently falling back to this constructor's default.
        self.save_hyperparameters(ignore=["config"])

        self.model: Rldx1Model | None = None  # type: ignore[assignment]
        self._preprocessor: Rldx1Preprocessor | None = None
        self._postprocessor: Rldx1Postprocessor | None = None

        # Per-view VTC frame buffer for rollout. Populated every env-step via
        # ``select_action``; ``prepare`` assembles the temporal window for
        # ``predict_action_chunk``. Cleared on ``reset``.
        self._vtc_buffer = VtcWindowBuffer(
            video_length=self.config.video_length,
            video_stride=self.config.video_stride,
        )

        if dataset_stats is not None:
            self._initialize_model(dataset_stats, shard_files)

    def _from_hf(  # noqa: PLR6301, PLR0913, PLR0917
        self,
        pretrained_name_or_path: str,
        revision: str | None,
        max_state_dim: int,
        max_action_dim: int,
        attn_implementation: Literal["sdpa", "flash_attention_2"],
        # Fine-tuning
        tune_diffusion_model: bool,  # noqa: FBT001
        use_vlln: bool,  # noqa: FBT001
        tune_vlln: bool,  # noqa: FBT001
        tune_top_llm_layers: int,
        tune_llm: bool,  # noqa: FBT001
        backbone_trainable_params_fp32: bool,  # noqa: FBT001
        tune_visual: bool,  # noqa: FBT001
        tune_projector: bool,  # noqa: FBT001
        num_inference_timesteps: int,
        backbone_use_lora: bool,  # noqa: FBT001
        action_model_use_lora: bool,  # noqa: FBT001
        # optimizer
        optim: Literal["adamw_torch", "adamw_torch_fused", "adafactor"],
        learning_rate: float,
        weight_decay: float,
        warmup_ratio: float,
        scheduler_decay_lr: float,
        # Precision / compilation
        use_bf16: bool,  # noqa: FBT001
        gradient_checkpointing: bool,  # noqa: FBT001
        # VTC video window
        video_length: int,
        video_stride: int,
        # Image geometry
        image_min_area: int | None,
        # Normalization
        clip_outliers: bool,  # noqa: FBT001
        embodiment_tag: str,
        # Action prediciton
        action_horizon: int,
    ) -> tuple[Rldx1Config, dict[str, dict[str, list[float] | str | tuple]], list[Path]]:
        config_file = Path(hf_hub_download(pretrained_name_or_path, "config.json", revision=revision))  # nosec B615
        shard_files = retrieve_safetensors_shards(pretrained_name_or_path, revision=revision)
        try:
            stats_file = Path(hf_hub_download(pretrained_name_or_path, "processor/statistics.json", revision=revision))  # nosec B615
        except RemoteEntryNotFoundError:
            try:
                stats_file = Path(hf_hub_download(pretrained_name_or_path, "statistics.json", revision=revision))  # nosec B615
            except RemoteEntryNotFoundError as e2:
                msg = "statistics.json not found in the root of the repo. Falling back to processor/statistics.json"
                raise RuntimeError(msg) from e2

        # --- parse config.json ---
        with Path(config_file).open(encoding="utf-8") as f:
            hf_config = json.load(f)

        hf_config["base_model_path"] = pretrained_name_or_path
        hf_config["backbone_trainable_params_fp32"] = backbone_trainable_params_fp32
        hf_config["max_state_dim"] = max_state_dim
        hf_config["max_action_dim"] = max_action_dim
        hf_config["attn_implementation"] = attn_implementation
        hf_config["tune_diffusion_model"] = tune_diffusion_model
        hf_config["use_vlln"] = use_vlln
        hf_config["tune_vlln"] = tune_vlln
        hf_config["tune_top_llm_layers"] = tune_top_llm_layers
        hf_config["tune_llm"] = tune_llm
        hf_config["tune_visual"] = tune_visual
        hf_config["tune_projector"] = tune_projector
        hf_config["num_inference_timesteps"] = num_inference_timesteps
        hf_config["backbone_use_lora"] = backbone_use_lora
        hf_config["action_model_use_lora"] = action_model_use_lora
        hf_config["optim"] = optim
        hf_config["learning_rate"] = learning_rate
        hf_config["weight_decay"] = weight_decay
        hf_config["warmup_ratio"] = warmup_ratio
        hf_config["scheduler_decay_lr"] = scheduler_decay_lr
        hf_config["use_bf16"] = use_bf16
        hf_config["gradient_checkpointing"] = gradient_checkpointing
        hf_config["video_length"] = video_length
        hf_config["video_stride"] = video_stride
        hf_config["clip_outliers"] = clip_outliers
        hf_config["image_min_area"] = image_min_area
        hf_config["action_horizon"] = action_horizon
        hf_config["embodiment_tag"] = embodiment_tag

        # strict=False: ignore upstream config.json keys with no Rldx1Config field
        # (e.g. architectures, model_type, rtc_inference_*) instead of denylisting
        # them one by one.
        config = Rldx1Config.from_dict(hf_config, strict=False)

        # --- build dataset_stats from HF artefacts ---
        dataset_stats = extract_dataset_stats(
            stats_file,
            embodiment_tag=embodiment_tag,
            max_state_dim=max_state_dim,
            max_action_dim=max_action_dim,
        )
        return config, dataset_stats, shard_files

    def _initialize_model(
        self,
        dataset_stats: dict[str, dict[str, list[float] | str | tuple]],
        shard_files: list[Path] | None = None,
    ) -> None:
        """Build the model (and preprocessors) for both init paths.

        Args:
            dataset_stats: Dataset normalization statistics.
            shard_files: List of shard files containing model weights.
        """
        config: Rldx1Config = self.config
        self.model = Rldx1Model(
            base_model_path=config.base_model_path,
            max_state_dim=config.max_state_dim,
            max_action_dim=config.max_action_dim,
            action_horizon=config.action_horizon,
            attn_implementation=config.attn_implementation,
            num_inference_timesteps=config.num_inference_timesteps,
            use_vlln=config.use_vlln,
            tune_projector=config.tune_projector,
            tune_visual=config.tune_visual,
            tune_vlln=config.tune_vlln,
            tune_diffusion_model=config.tune_diffusion_model,
            tune_top_llm_layers=config.tune_top_llm_layers,
            tune_llm=config.tune_llm,
            backbone_trainable_params_fp32=config.backbone_trainable_params_fp32,
            backbone_use_lora=config.backbone_use_lora,
            action_model_use_lora=config.action_model_use_lora,
            gradient_checkpointing=config.gradient_checkpointing,
            diffusion_model_cfg=config.diffusion_model_cfg,
            backbone_lora_target_modules=config.backbone_lora_target_modules,
            action_model_lora_target_modules=config.action_model_lora_target_modules,
            video_length=config.video_length,
            video_stride=config.video_stride,
        )

        if shard_files is not None:
            self.model.load_sharded_weights(shard_files)

        if config.use_bf16:
            self.model.to(torch.bfloat16)

            if config.backbone_trainable_params_fp32:
                for _, p in self.model.backbone.named_parameters():
                    if p.requires_grad:
                        p.data = p.data.to(torch.float32)

        self._preprocessor, self._postprocessor = make_rldx1_transforms(
            stats=dataset_stats,  # type: ignore[arg-type]
            max_state_dim=config.max_state_dim,
            max_action_dim=config.max_action_dim,
            action_horizon=config.action_horizon,
            revision=config.revision,
            use_percentiles=config.use_percentiles,
            clip_outliers=config.clip_outliers,
            image_max_area=config.image_max_area,
            image_min_area=config.image_min_area or 0,  # type: ignore[arg-type]
            image_resize_m=config.image_resize_m,
            embodiment_id=int(config.embodiment_id),  # type: ignore[arg-type]
        )

    def setup(self, stage: str) -> None:
        """Set up model from datamodule (lazy or fine-tuning path).

        Called by Lightning before fit/validate/test/predict.

        - **Lazy path**: model is None → build model + preprocessors from dataset stats.
        - **Fine-tuning path**: model already loaded from pretrained → rebuild
          preprocessors with the training dataset's stats so normalization
          matches the new data distribution.

        Raises:
            TypeError: If the train dataset is not a physicalai Dataset.
        """
        del stage

        datamodule = self.trainer.datamodule  # type: ignore[attr-defined]
        train_dataset = datamodule.train_dataset

        if not isinstance(train_dataset, Dataset):
            msg = f"Expected physicalai Dataset, got {type(train_dataset)}"
            raise TypeError(msg)

        stats_dict = train_dataset.stats

        if self.model is not None:
            # Fine-tuning path: model exists from pretrained, but the
            # preprocessor stats must match the training data distribution.
            self._update_preprocessor_stats(stats_dict)
            reformat_dataset_to_match_policy(self, datamodule)
            return

        self.hparams["dataset_stats"] = stats_dict

        self._initialize_model(stats_dict)

        reformat_dataset_to_match_policy(self, datamodule)

    def _update_preprocessor_stats(
        self,
        dataset_stats: dict[str, dict[str, list[float] | str | tuple]],
    ) -> None:
        """Rebuild preprocessor/postprocessor with new dataset stats.

        Used when fine-tuning a pretrained model on a new dataset: the model
        weights come from the checkpoint, but normalization statistics must
        reflect the training data.
        """
        logger.info("Updating preprocessor stats for fine-tuning dataset")
        config = self.config
        self._preprocessor, self._postprocessor = make_rldx1_transforms(
            stats=dataset_stats,  # type: ignore[arg-type]
            max_state_dim=config.max_state_dim,
            max_action_dim=config.max_action_dim,
            action_horizon=config.action_horizon,
            revision=config.revision,
            use_percentiles=config.use_percentiles,
            clip_outliers=config.clip_outliers,
            image_max_area=config.image_max_area,
            image_min_area=config.image_min_area or 0,  # type: ignore[arg-type]
            image_resize_m=config.image_resize_m,
            embodiment_id=int(config.embodiment_id),  # type: ignore[arg-type]
        )
        self._dataset_stats = dataset_stats
        self.hparams["dataset_stats"] = dataset_stats

    def forward(self, batch: Observation) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        """Forward pass: training loss in train mode, action chunk in eval.

        Args:
            batch: Input observation batch.

        Returns:
            Training: ``(loss, loss_dict)``. Eval: action chunk tensor.

        Raises:
            RuntimeError: If the model has not been initialized.
        """
        if self.model is None or self._preprocessor is None:
            msg = "Model not initialized. Call trainer.fit() or pass env_action_dim."
            raise RuntimeError(msg)
        if not self.training:
            return self.predict_action_chunk(batch)
        preprocessed = self._preprocessor(batch)
        return self.model.compute_loss(preprocessed)

    def training_step(self, batch: Observation, batch_idx: int) -> torch.Tensor:
        """Training step that computes and logs the optimization loss.

        Args:
            batch: Input observation batch.
            batch_idx: Batch index (unused).

        Returns:
            Scalar training loss tensor.
        """
        del batch_idx

        loss, loss_dict = self(batch)
        self.log("train/loss", loss_dict["loss"], prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def compute_val_loss(self, batch: Observation) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        """Compute validation loss for the current batch.

        Args:
            batch: Input observation batch.

        Returns:
            Tuple of ``(loss, loss_dict)``.

        Raises:
            RuntimeError: If model or preprocessor has not been initialized.
        """
        if self.model is None or self._preprocessor is None:
            msg = "Model not initialized. Call trainer.fit() or pass env_action_dim."
            raise RuntimeError(msg)
        preprocessed = self._preprocessor(batch)
        return self.model.compute_val_loss(preprocessed)

    def configure_optimizers(self) -> OptimizerLRScheduler:  # type: ignore[override]
        """Create the configured optimizer and a cosine-decay-with-warmup scheduler.

        The optimizer is selected by ``config.optim``:
        ``"adamw_torch"``/``"adamw_torch_fused"`` use AdamW; ``"adafactor"`` uses
        transformers' Adafactor with a fixed learning rate to cut optimizer memory.

        The LR schedule matches pi05: linear warmup for ``warmup_ratio`` of total
        steps, then cosine decay from ``learning_rate`` down to
        ``config.scheduler_decay_lr`` over the remaining training steps.

        Returns:
            Lightning optimizer configuration dictionary.

        Raises:
            RuntimeError: If model has not been initialized.
        """
        if self.model is None:
            msg = "Model not initialized. Call trainer.fit() or pass env_action_dim."
            raise RuntimeError(msg)

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = self._build_optimizer(trainable_params)

        total_steps = int(self.trainer.estimated_stepping_batches)
        warmup_steps = max(1, int(total_steps * self.config.warmup_ratio))

        scheduler = cosine_decay_with_warmup_scheduler(
            optimizer,
            peak_lr=self.config.learning_rate,
            decay_lr=self.config.scheduler_decay_lr,
            num_warmup_steps=warmup_steps,
            num_decay_steps=total_steps,
            num_training_steps=total_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def _build_optimizer(
        self,
        params: list[torch.nn.Parameter],
    ) -> torch.optim.Optimizer:
        """Build the optimizer selected by ``config.optim``.

        Args:
            params: Trainable parameters to optimize.

        Returns:
            The instantiated optimizer.

        Raises:
            ValueError: If ``config.optim`` is not a supported value.
        """
        optim = self.config.optim
        if optim in {"adamw_torch", "adamw_torch_fused"}:
            return torch.optim.AdamW(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.95),
                fused=optim == "adamw_torch_fused",
            )
        if optim == "adafactor":
            return Adafactor(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                scale_parameter=False,
                relative_step=False,
                warmup_init=False,
            )
        msg = f"Unsupported optim {optim!r}; expected one of 'adamw_torch', 'adamw_torch_fused', 'adafactor'."
        raise ValueError(msg)

    def predict_action_chunk(self, batch: Observation) -> torch.Tensor:
        """Predict a chunk of actions of shape ``(B, T, D)``.

        Assembles the VTC temporal window (``video_length`` frames at
        ``video_stride`` env-steps, offsets ``[-6, -4, -2, 0]`` for the
        defaults) so the backbone receives the same multi-frame stack it was
        trained on. A batch that already carries a temporal axis (the training /
        validation ``delta_timestamps`` path) is passed through unchanged; a
        single-frame rollout observation is stacked from the per-view history
        buffer maintained by :meth:`select_action` (see :class:`~physicalai.policies.rldx1.vtc_buffer.VtcWindowBuffer`).

        Args:
            batch: Input observation batch.

        Returns:
            Action chunk tensor of shape ``(B, n_action_steps, action_dim)``.

        Raises:
            RuntimeError: If the model has not been initialized.
        """
        if self.model is None or self._preprocessor is None or self._postprocessor is None:
            msg = "Model not initialized. Call trainer.fit() or pass env_action_dim."
            raise RuntimeError(msg)
        self.model.eval()
        model_input = self._vtc_buffer.prepare(batch)
        preprocessed = self._preprocessor(model_input)
        preprocessed = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in preprocessed.items()}
        actions = self.model.get_action(preprocessed)
        return self._postprocessor(actions)

    # -- VTC rollout frame stacking --------------------------------------- #

    def reset(self) -> None:
        """Reset the policy state at an episode boundary.

        Clears the action queue (base) and the VTC frame-history buffer so the
        next episode's video window is rebuilt from scratch.
        """
        super().reset()
        self._vtc_buffer.reset()

    def select_action(self, batch: Observation) -> torch.Tensor:
        """Select a single action, recording the frame for the VTC window.

        Records the current observation into the per-view history every
        env-step (regardless of the action queue) so
        :meth:`predict_action_chunk` can sample the ``[-6, -4, -2, 0]`` temporal
        window from the correct env-step strides, then delegates to the base
        action-chunking logic.

        Args:
            batch: Input observation batch.

        Returns:
            Single action tensor of shape ``(B, D)`` or ``(D,)``.
        """
        self._vtc_buffer.record(batch)

        # A malformed or externally managed history may be incomplete. Normal
        # rollout histories are seeded with their first frame and skip this path.
        if self._vtc_buffer.is_warming_up:
            env_action_dim = self.hparams.get("env_action_dim", self.config.max_action_dim)
            return self._get_warmup_hold_action(batch, env_action_dim)

        return super().select_action(batch)

    def _get_warmup_hold_action(self, batch: Observation, env_action_dim: int) -> torch.Tensor:
        """Return a safe warmup action while frame history is filling.

        For PushT-like position-control tasks, action represents the target
        position. Returning zeros would command motion toward ``(0, 0)``.
        During warmup we instead try to hold the current state position.

        Args:
            batch: Current observation batch.
            env_action_dim: Real environment action dimension.

        Returns:
            Warmup action tensor of shape ``(B, env_action_dim)``.
        """
        batch_size = batch.batch_size

        state = batch.state
        if isinstance(state, torch.Tensor):
            state_t = state
        elif isinstance(state, dict):
            state_t = None
            for key in ("agent_pos", "state"):
                value = state.get(key)
                if isinstance(value, torch.Tensor):
                    state_t = value
                    break
        else:
            state_t = None

        if state_t is not None:
            if state_t.ndim == 1:
                state_t = state_t.unsqueeze(0)
            if state_t.shape[-1] >= env_action_dim:
                return state_t[..., :env_action_dim].to(device=self.device, dtype=torch.bfloat16)

        # Fallback for tasks without compatible state-to-action mapping.
        return torch.zeros(batch_size, env_action_dim, dtype=torch.bfloat16, device=self.device)
