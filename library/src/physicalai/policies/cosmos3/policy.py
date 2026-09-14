# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lightning module for Cosmos3 policy."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch

from physicalai.data.dataset import Dataset
from physicalai.policies.base import Policy
from physicalai.train.utils import reformat_dataset_to_match_policy

from .config import Cosmos3Config
from .model import Cosmos3Model
from .pipeline import PolicyPipelineWithState, require_xpu_driver
from .surgery import HEAD_KEYS, load_finetuned, split_trainable_params

if TYPE_CHECKING:
    from physicalai.data import DataModule
    from physicalai.data.observation import Observation

logger = logging.getLogger(__name__)


class Cosmos3(Policy):
    """NVIDIA Cosmos 3 Policy - Lightning wrapper for training and inference.

    Supports post-training (PEFT LoRA/DoRA and full generation-tower fine-tuning)
    and closed-loop action-chunk inference on XPU, CUDA, and CPU.

    Note on export: Export via ONNX / OpenVINO / ExecuTorch is currently out of scope
    for the 4B multimodal diffusion backbone; deployment is performed via native
    PyTorch inference using `Cosmos3` and `InferenceModel`.

    Args:
        pretrained_model_name_or_path: Hugging Face model repo ID or local checkpoint path.
        mode: Training mode ("peft" or "full"). Default: "peft".
        paradigm: Denoising objective ("policy", "fd", "id", or "joint"). Default: "policy".
        rank: LoRA/DoRA rank for PEFT mode. Default: 32.
        alpha_scale: LoRA alpha scaling factor. Default: 1.0.
        dora: Whether to use DoRA instead of plain LoRA. Default: False.
        head_lr_mult: Multiplier on optimizer_lr for the domain action head.
            Default: 2.0 for peft, 10.0 for full.
        action_weight: Weight of action loss vs video loss. Default: 10.0.
        chunk_size: Size of action prediction chunk. Default: 32.
        n_action_steps: Number of action steps to execute per invocation. Default: 32.
        resolution_tier: Video conditioning short-side px (256, 480, 720). Default: 256.
        fps: Video and action frame rate. Default: 10.
        grad_checkpoint: Enable gradient checkpointing. Default: True.
        domain: Embodiment domain identifier. Default: "pusht".
        prompt: Task instruction string. Default: "".
        guidance_scale: Classifier-free guidance scale. Default: 3.0.
        flow_shift: Flow shift for UniPC scheduler. Default: 8.0.
        num_inference_steps: Denoising steps during inference. Default: 4.
        dtype: Model precision ("bfloat16", "float32", "float16"). Default: "bfloat16".
        optimizer_lr: Base learning rate. Default: 1e-4.
        optimizer_betas: AdamW beta coefficients. Default: (0.9, 0.999).
        optimizer_eps: AdamW epsilon. Default: 1e-8.
        optimizer_weight_decay: AdamW weight decay. Default: 0.01.
        optimizer_grad_clip_norm: Max gradient norm for clipping. Default: 1.0.
        dataset_stats: Dataset normalization statistics for eager initialization.
        pipeline: Pre-initialized PolicyPipelineWithState instance (optional).
    """

    def __init__(  # ruff: ignore[too-many-arguments]
        self,
        pretrained_model_name_or_path: str = "nvidia/Cosmos3-Edge",
        *,
        mode: Literal["peft", "full"] = "peft",
        paradigm: Literal["policy", "fd", "id", "joint"] = "policy",
        rank: int = 32,
        alpha_scale: float = 1.0,
        dora: bool = False,
        head_lr_mult: float | None = None,
        action_weight: float = 10.0,
        chunk_size: int = 32,
        n_action_steps: int | None = None,
        resolution_tier: int = 256,
        fps: int = 10,
        grad_checkpoint: bool = True,
        domain: str = "pusht",
        prompt: str = "",
        guidance_scale: float = 3.0,
        flow_shift: float = 8.0,
        num_inference_steps: int = 4,
        dtype: Literal["bfloat16", "float32", "float16"] = "bfloat16",
        optimizer_lr: float = 1e-4,
        optimizer_betas: tuple[float, float] = (0.9, 0.999),
        optimizer_eps: float = 1e-8,
        optimizer_weight_decay: float = 0.01,
        optimizer_grad_clip_norm: float = 1.0,
        dataset_stats: dict[str, Any] | None = None,
        pipeline: PolicyPipelineWithState | None = None,
    ) -> None:
        """Initialize Cosmos3 Policy."""
        resolved_head_lr_mult = head_lr_mult if head_lr_mult is not None else (10.0 if mode == "full" else 2.0)

        self.config = Cosmos3Config(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            mode=mode,
            paradigm=paradigm,
            rank=rank,
            alpha_scale=alpha_scale,
            dora=dora,
            head_lr_mult=resolved_head_lr_mult,
            action_weight=action_weight,
            chunk_size=chunk_size,
            n_action_steps=n_action_steps,
            resolution_tier=resolution_tier,
            fps=fps,
            grad_checkpoint=grad_checkpoint,
            domain=domain,
            prompt=prompt,
            guidance_scale=guidance_scale,
            flow_shift=flow_shift,
            num_inference_steps=num_inference_steps,
            dtype=dtype,
            optimizer_lr=optimizer_lr,
            optimizer_betas=optimizer_betas,
            optimizer_eps=optimizer_eps,
            optimizer_weight_decay=optimizer_weight_decay,
            optimizer_grad_clip_norm=optimizer_grad_clip_norm,
        )

        super().__init__(n_action_steps=self.config.n_action_steps)

        self.save_hyperparameters(ignore=["config", "pipeline"])
        self.hparams["config"] = self.config.to_dict()

        self.model: Cosmos3Model | None = None
        self._dataset_stats = dataset_stats

        # Eager initialization if dataset_stats or pipeline is explicitly provided
        if dataset_stats is not None or pipeline is not None:
            self._initialize_model(dataset_stats=dataset_stats, pipeline=pipeline)

    def _initialize_model(
        self,
        dataset_stats: dict[str, Any] | None = None,
        pipeline: PolicyPipelineWithState | None = None,
    ) -> None:
        """Construct the underlying Cosmos3Model."""
        self.model = Cosmos3Model(
            config=self.config,
            pipeline=pipeline,
            dataset_stats=dataset_stats,
            device=self.device,
        )

    def setup(self, stage: str) -> None:
        """Set up model from datamodule before training or validation.

        Args:
            stage: Stage of training ("fit", "validate", "test", or "predict").

        Raises:
            TypeError: If datamodule train_dataset is not a physicalai Dataset.
        """
        del stage

        if hasattr(self.device, "type") and self.device.type == "xpu" and require_xpu_driver is not None:
            require_xpu_driver()

        datamodule: DataModule = self.trainer.datamodule  # type: ignore[assignment]
        train_dataset = datamodule.train_dataset

        if not isinstance(train_dataset, Dataset):
            msg = f"Expected train_dataset to be physicalai.data.Dataset, got {type(train_dataset)}."
            raise TypeError(msg)

        stats_dict = train_dataset.stats

        if self.model is None:
            self.hparams["dataset_stats"] = stats_dict
            self._initialize_model(dataset_stats=stats_dict)
        else:
            self.model.set_dataset_stats(stats_dict)

        reformat_dataset_to_match_policy(self, datamodule)

    def forward(
        self,
        batch: Observation,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]] | torch.Tensor:
        """Forward pass for training or action chunk prediction.

        Args:
            batch: Input Observation batch.

        Returns:
            Tuple of (loss, loss_dict) during training, or action chunk predictions during eval.

        Raises:
            RuntimeError: If the Cosmos3 model is not initialized.
        """
        if self.model is None:
            msg = "Cosmos3 model is not initialized."
            raise RuntimeError(msg)

        if self.training:
            return self.model.compute_loss(batch.to(self.device).to_dict())
        return self.predict_action_chunk(batch)

    def compute_val_loss(
        self,
        batch: Observation,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        """Compute validation loss without gradients.

        Args:
            batch: Input Observation batch.

        Returns:
            Tuple of (val_loss, val_loss_dict).

        Raises:
            RuntimeError: If the Cosmos3 model is not initialized.
        """
        if self.model is None:
            msg = "Cosmos3 model is not initialized."
            raise RuntimeError(msg)
        return self.model.compute_val_loss(batch.to(self.device).to_dict())

    def training_step(self, batch: Observation, batch_idx: int) -> torch.Tensor:
        """Lightning training step.

        Args:
            batch: Input Observation batch.
            batch_idx: Index of current batch.

        Returns:
            Training loss tensor.
        """
        del batch_idx
        loss, loss_dict = self(batch)  # type: ignore[misc]
        self.log("train/loss", loss_dict["loss"], prog_bar=True)
        if "loss_vision" in loss_dict:
            self.log("train/loss_vision", loss_dict["loss_vision"], prog_bar=False)
        if "loss_action" in loss_dict:
            self.log("train/loss_action", loss_dict["loss_action"], prog_bar=False)
        return loss

    def predict_action_chunk(self, batch: Observation) -> torch.Tensor:
        """Predict a chunk of actions from observation.

        Args:
            batch: Input observation batch.

        Returns:
            Action chunk tensor of shape (B, chunk_size, raw_dim).

        Raises:
            RuntimeError: If the Cosmos3 model is not initialized.
        """
        if self.model is None:
            msg = "Cosmos3 model is not initialized."
            raise RuntimeError(msg)

        batch_dict = batch.to(self.device).to_dict()
        return self.model.predict_action_chunk(batch_dict)

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure AdamW optimizer with distinct base and head parameter learning rates.

        Returns:
            Dictionary containing optimizer and learning rate scheduler.

        Raises:
            RuntimeError: If the Cosmos3 model is not initialized.
        """
        if self.model is None:
            msg = "Cosmos3 model is not initialized."
            raise RuntimeError(msg)

        base_params, head_params = split_trainable_params(self.model.transformer)
        optimizer = torch.optim.AdamW(
            [
                {"params": base_params, "lr": self.config.optimizer_lr},
                {"params": head_params, "lr": self.config.optimizer_lr * self.config.head_lr_mult},
            ],
            betas=self.config.optimizer_betas,
            eps=self.config.optimizer_eps,
            weight_decay=self.config.optimizer_weight_decay,
        )

        num_updates = (
            self.trainer.estimated_stepping_batches
            if self.trainer is not None and getattr(self.trainer, "estimated_stepping_batches", None) is not None
            else 2000
        )
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=max(1, int(num_updates)),
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Filter checkpoint state dict to save only trainable parameters and normalizer buffers.

        Args:
            checkpoint: Checkpoint dictionary containing state_dict.
        """
        if "state_dict" not in checkpoint:
            return

        filtered_sd = {}
        trainable_names = {name for name, param in self.named_parameters() if param.requires_grad}

        for k, v in checkpoint["state_dict"].items():
            # Check if parameter is trainable or a critical buffer
            clean_name = k.removeprefix("model.")
            is_trainable = clean_name in trainable_names
            is_head = any(hk in k for hk in HEAD_KEYS)
            is_buffer = "a_min" in k or "a_max" in k or "domain_id" in k
            if is_trainable or is_head or is_buffer:
                filtered_sd[k] = v

        checkpoint["state_dict"] = filtered_sd

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Load checkpoint state dict with strict=False to preserve frozen backbone.

        Args:
            checkpoint: Checkpoint dictionary containing state_dict.
        """
        del checkpoint
        if self.model is None:
            self._initialize_model()

    def save_pretrained_adapter(self, output_dir: str | Path) -> None:
        """Save fine-tuned weights and domain action head to target directory.

        Args:
            output_dir: Target directory path for saved adapter weights.

        Raises:
            RuntimeError: If the model is not initialized.
        """
        if self.model is None:
            msg = "Model is not initialized."
            raise RuntimeError(msg)

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        tf = self.model.transformer

        if self.config.mode == "full":
            torch.save(
                {n: p.detach().cpu() for n, p in tf.named_parameters() if p.requires_grad},
                out_path / "transformer_full.pt",
            )
        else:
            tf.save_lora_adapter(str(out_path))

        head = {n: p.detach().cpu() for n, p in tf.named_parameters() if any(k in n for k in HEAD_KEYS)}
        torch.save(
            {
                "head": head,
                "action_min": self.model.a_min.cpu(),
                "action_max": self.model.a_max.cpu(),
                "prompt": self.config.prompt,
                "paradigm": self.config.paradigm,
                "domain": self.config.domain,
            },
            out_path / f"{self.config.domain}_head.pt",
        )
        logger.info("Saved Cosmos3 %s adapter and head to %s", self.config.mode, out_path)

    def load_pretrained_adapter(
        self,
        adapter_dir: str | Path,
        domain: str | None = None,
        head_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Restore fine-tuned weights onto the pipeline.

        Args:
            adapter_dir: Path to directory holding saved adapter weights.
            domain: Domain name override.
            head_path: Explicit path to head checkpoint file.

        Returns:
            Loaded checkpoint dictionary.

        Raises:
            RuntimeError: If the model cannot be initialized.
        """
        if self.model is None:
            self._initialize_model()
            if self.model is None:
                msg = "Failed to initialize Cosmos3 model."
                raise RuntimeError(msg)

        dom = domain or self.config.domain
        ckpt = load_finetuned(
            self.model.pipe,
            adapter=str(adapter_dir),
            domain=dom,
            head=str(head_path) if head_path else None,
        )
        if "action_min" in ckpt and "action_max" in ckpt:
            self.model.a_min = ckpt["action_min"].to(self.model.a_min.device)
            self.model.a_max = ckpt["action_max"].to(self.model.a_max.device)
        return ckpt

    @classmethod
    def from_config(cls, config: Cosmos3Config, **kwargs: object) -> Cosmos3:
        """Create Cosmos3 policy from a Cosmos3Config instance.

        Args:
            config: Cosmos3Config instance.
            **kwargs: Extra arguments passed to constructor.

        Returns:
            Initialized Cosmos3 policy instance.
        """
        return cls(**config.to_dict(), **kwargs)
