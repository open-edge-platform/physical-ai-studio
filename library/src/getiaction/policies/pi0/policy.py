# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2025 Physical Intelligence
# SPDX-License-Identifier: Apache-2.0

"""Pi0/Pi0.5 Policy - Lightning wrapper for training and inference.

This module provides PyTorch Lightning policies for Pi0 and Pi0.5 models,
enabling easy training, checkpoint management, and inference.

Example:
    >>> from getiaction.policies.pi0 import Pi0
    >>> from getiaction.data.lerobot import LeRobotDataModule
    >>> import lightning as L

    >>> # Create Pi0 policy
    >>> policy = Pi0(
    ...     variant="pi0",
    ...     chunk_size=50,
    ...     learning_rate=2.5e-5,
    ... )

    >>> # Create datamodule
    >>> datamodule = LeRobotDataModule(
    ...     repo_id="lerobot/aloha_sim_transfer_cube_human",
    ...     train_batch_size=4,
    ... )

    >>> # Train
    >>> trainer = L.Trainer(max_epochs=100, precision="bf16-mixed")
    >>> trainer.fit(policy, datamodule)

    >>> # Load from checkpoint
    >>> policy = Pi0.load_from_checkpoint("checkpoint.ckpt")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

import torch

from getiaction.policies.base import Policy

from .config import Pi0Config
from .model import Pi0Model

if TYPE_CHECKING:
    from getiaction.data import Observation
    from getiaction.gyms import Gym

logger = logging.getLogger(__name__)


class Pi0(Policy):
    """Pi0/Pi0.5 Policy - Physical Intelligence's flow matching VLA model.

    Lightning wrapper for training and inference with Pi0/Pi0.5 models.
    Supports both variants via the `variant` parameter.

    Uses dual-path initialization:
    - **Lazy path**: `Pi0()` + `trainer.fit()` - model built in setup()
    - **Eager path**: `Pi0.load_from_checkpoint()` - model built immediately

    Args:
        variant: Model variant ("pi0" or "pi05").
        chunk_size: Number of action predictions per forward pass.
        n_action_steps: Number of action steps to execute per chunk.
        max_state_dim: Maximum state dimension for padding.
        max_action_dim: Maximum action dimension for padding.
        paligemma_variant: PaliGemma backbone size.
        action_expert_variant: Action expert size.
        num_inference_steps: Number of denoising steps during inference.
        tune_paligemma: Whether to fine-tune PaliGemma backbone.
        tune_action_expert: Whether to fine-tune action expert.
        tune_vision_encoder: Whether to fine-tune vision encoder.
        lora_rank: LoRA rank. 0 disables LoRA.
        lora_alpha: LoRA alpha scaling factor.
        lora_dropout: LoRA dropout rate.
        learning_rate: Learning rate for optimizer.
        weight_decay: Weight decay for optimizer.
        warmup_steps: Number of warmup steps for scheduler.
        use_bf16: Whether to use bfloat16 precision.
        gradient_checkpointing: Enable gradient checkpointing for memory.
        env_action_dim: Environment action dimension. If provided, enables eager initialization.
        dataset_stats: Dataset normalization statistics.

    Example:
        Training:

        >>> policy = Pi0(variant="pi0", learning_rate=2.5e-5)
        >>> trainer = L.Trainer(max_epochs=100)
        >>> trainer.fit(policy, datamodule)

        Inference:

        >>> policy = Pi0.load_from_checkpoint("checkpoint.ckpt")
        >>> action = policy.select_action(obs)
    """

    def __init__(  # noqa: PLR0913
        self,
        # Model variant
        variant: Literal["pi0", "pi05"] = "pi0",
        # Model architecture
        chunk_size: int = 50,
        n_action_steps: int = 50,
        max_state_dim: int = 32,
        max_action_dim: int = 32,
        # Backbone configuration
        paligemma_variant: str = "gemma_2b",
        action_expert_variant: str = "gemma_300m",
        # Inference
        num_inference_steps: int = 10,
        # Fine-tuning control
        *,
        tune_paligemma: bool = False,
        tune_action_expert: bool = True,
        tune_vision_encoder: bool = False,
        # LoRA
        lora_rank: int = 0,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        # Optimizer
        learning_rate: float = 1.0e-4,
        weight_decay: float = 1.0e-5,
        warmup_ratio: float = 0.05,  # Warmup ratio (0.0-1.0) of total training steps
        grad_clip_norm: float = 1.0,
        # Precision
        use_bf16: bool = True,
        # Memory optimization
        gradient_checkpointing: bool = False,
        # Eager initialization (for checkpoint loading)
        env_action_dim: int | None = None,
        dataset_stats: dict[str, dict[str, list[float]]] | None = None,
    ) -> None:
        """Initialize Pi0 policy.

        Creates Pi0Config from explicit args and saves it as hyperparameters.
        """
        super().__init__(n_action_steps=n_action_steps)

        # Create config from explicit args (policy-level config)
        self.config = Pi0Config(
            variant=variant,
            paligemma_variant=paligemma_variant,
            action_expert_variant=action_expert_variant,
            action_dim=max_action_dim,  # Use max_action_dim as action_dim
            action_horizon=chunk_size,
            max_state_dim=max_state_dim,
            max_action_dim=max_action_dim,
            num_inference_steps=num_inference_steps,
            tune_paligemma=tune_paligemma,
            tune_action_expert=tune_action_expert,
            tune_vision_encoder=tune_vision_encoder,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            gradient_checkpointing=gradient_checkpointing,
            dtype="bfloat16" if use_bf16 else "float32",
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            grad_clip_norm=grad_clip_norm,
        )

        # Save config as hyperparameters for checkpoint restoration
        self.save_hyperparameters(ignore=["config"])  # Save individual args, not config object
        # Also save config dict for compatibility
        self.hparams["config"] = self.config.to_dict()

        # Model pre/post-processors will be built in setup() or _initialize_model()
        self.model: Pi0Model | None = None
        self._preprocessor: Any = None
        self._postprocessor: Any = None

        # Eager initialization if env_action_dim is provided
        if env_action_dim is not None:
            self._initialize_model(env_action_dim, dataset_stats)

        # Track initialization state
        self._is_setup_complete: bool = False

    def _initialize_model(
        self,
        env_action_dim: int,
        dataset_stats: dict[str, dict[str, list[float]]] | None = None,
    ) -> None:
        """Initialize model and preprocessors.

        Called by both lazy (setup) and eager (checkpoint) paths.

        Args:
            env_action_dim: Environment action dimension.
            dataset_stats: Dataset normalization statistics.

        Raises:
            ValueError: If max_token_len is not set in config.
        """
        from .preprocessor import make_pi0_preprocessors  # noqa: PLC0415

        # Use config (policy-level config created in __init__)
        config = self.config

        # Derived values
        is_pi05 = config.is_pi05
        use_lora = config.use_lora

        # Create model with explicit args (no config dependency)
        # Type cast: config stores as str, but Pi0Model expects GemmaVariant Literal
        self.model = Pi0Model(
            variant=config.variant,
            paligemma_variant=config.paligemma_variant,  # type: ignore[arg-type]
            action_expert_variant=config.action_expert_variant,  # type: ignore[arg-type]
            max_action_dim=config.max_action_dim,
            max_state_dim=config.max_state_dim,
            action_horizon=config.action_horizon,
            num_inference_steps=config.num_inference_steps,
            dtype=config.dtype,
        )

        # Apply LoRA if enabled
        if use_lora:
            self._apply_lora(config)

        # Set trainable parameters (including projection heads)
        self.model.set_trainable_parameters(
            tune_paligemma=config.tune_paligemma,
            tune_action_expert=config.tune_action_expert,
            tune_vision_encoder=config.tune_vision_encoder,
            tune_projection_heads=True,  # Always train projection heads
        )

        # Enable gradient checkpointing if requested
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Create preprocessor/postprocessor
        # Pi0.5 uses max_token_len=200, Pi0 uses 48
        max_token_len = config.max_token_len
        if max_token_len is None:
            msg = "max_token_len must be set in config"
            raise ValueError(msg)
        self._preprocessor, self._postprocessor = make_pi0_preprocessors(
            max_state_dim=config.max_state_dim,
            max_action_dim=config.max_action_dim,
            action_horizon=config.action_horizon,
            env_action_dim=env_action_dim,
            stats=dataset_stats,
            use_quantile_norm=is_pi05,  # Pi0.5 uses quantile normalization
            image_resolution=config.image_resolution,
            max_token_len=max_token_len,
        )

        self._is_setup_complete = True
        logger.info("Pi0 model initialized (variant=%s, action_dim=%d)", config.variant, env_action_dim)

    def _apply_lora(self, config: Pi0Config) -> None:
        """Apply LoRA to the model.

        Args:
            config: Model configuration with LoRA settings.
        """
        from .components.lora import apply_lora  # noqa: PLC0415

        # Apply LoRA to PaliGemma if tuning
        if config.tune_paligemma and self.model is not None:
            self.model.paligemma_with_expert._paligemma = apply_lora(  # noqa: SLF001
                self.model.paligemma_with_expert.paligemma,
                rank=config.lora_rank,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout,
                target_modules=config.lora_target_modules,
            )

        # Apply LoRA to action expert if tuning
        if config.tune_action_expert and self.model is not None:
            self.model.paligemma_with_expert._action_expert = apply_lora(  # noqa: SLF001
                self.model.paligemma_with_expert.action_expert,
                rank=config.lora_rank,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout,
                target_modules=config.lora_target_modules,
            )

        logger.info("LoRA applied (rank=%d, alpha=%d)", config.lora_rank, config.lora_alpha)

    def setup(self, stage: str) -> None:  # noqa: ARG002
        """Set up model from datamodule (lazy initialization path).

        Called by Lightning before fit/validate/test/predict.

        Args:
            stage: Lightning stage (unused, required by Lightning API).

        Raises:
            TypeError: If train dataset is not a getiaction Dataset.
            ValueError: If dataset lacks action features.
        """
        if self._is_setup_complete or self.model is not None:
            return  # Already initialized

        from getiaction.data.dataset import Dataset  # noqa: PLC0415

        datamodule = self.trainer.datamodule  # type: ignore[attr-defined]
        train_dataset = datamodule.train_dataset

        # Use getiaction dataset interface
        if not isinstance(train_dataset, Dataset):
            msg = f"Expected getiaction Dataset, got {type(train_dataset)}"
            raise TypeError(msg)

        # Extract action dimension from features
        action_features = train_dataset.action_features
        if not action_features:
            msg = "Dataset must have action features"
            raise ValueError(msg)

        # Get action feature (typically "action")
        action_feature = next(iter(action_features.values()))
        if action_feature.shape is None:
            msg = "Action feature must have shape defined"
            raise ValueError(msg)
        env_action_dim = action_feature.shape[-1]

        # Extract stats from dataset
        stats_dict = train_dataset.stats

        # Save to hparams for checkpoint
        self.hparams["env_action_dim"] = env_action_dim
        self.hparams["dataset_stats"] = stats_dict

        # Initialize model
        self._initialize_model(env_action_dim, stats_dict)

    def forward(self, batch: Observation) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        """Forward pass.

        Behavior depends on training mode:
        - Training: Returns loss tensor or (loss, loss_dict)
        - Evaluation: Returns action via select_action

        Args:
            batch: Input observation batch.

        Returns:
            Loss tensor in training mode, action tensor in eval mode.
        """
        if self.training:
            return self._training_forward(batch)
        return self.select_action(batch)

    def _training_forward(self, batch: Observation) -> tuple[torch.Tensor, dict[str, float]]:
        """Training forward pass.

        Args:
            batch: Input batch with observations and actions.

        Returns:
            Tuple of (loss, loss_dict).

        Raises:
            RuntimeError: If model is not initialized.
        """
        if self.model is None:
            msg = "Model not initialized. Call setup() first."
            raise RuntimeError(msg)

        # Preprocess batch
        processed = self._preprocessor(batch)

        # Get device
        device = next(self.model.parameters()).device

        # Move to device
        observation = {
            "images": {k: v.to(device) for k, v in processed["images"].items()},
            "image_masks": {k: v.to(device) for k, v in processed["image_masks"].items()},
            "state": processed["state"].to(device),
            "tokenized_prompt": processed["tokenized_prompt"].to(device),
            "tokenized_prompt_mask": processed["tokenized_prompt_mask"].to(device),
        }
        actions = processed["actions"].to(device)

        # Forward pass
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self.hparams["use_bf16"]):
            loss_per_sample = self.model.forward(observation, actions)

        # Average loss
        loss = loss_per_sample.mean()
        loss_dict = {"loss": loss.item()}

        return loss, loss_dict

    def training_step(self, batch: Observation, batch_idx: int) -> torch.Tensor:  # noqa: ARG002
        """Lightning training step.

        Args:
            batch: Input batch.
            batch_idx: Batch index (unused, required by Lightning API).

        Returns:
            Loss tensor for backpropagation.
        """
        loss, loss_dict = self._training_forward(batch)

        # Log metrics
        self.log("train/loss", loss_dict["loss"], prog_bar=True)

        return loss

    def validation_step(self, batch: Gym, batch_idx: int) -> dict[str, float]:  # type: ignore[override]
        """Lightning validation step.

        Runs gym-based validation via rollout evaluation. The DataModule's val_dataloader
        returns Gym environment instances directly.

        Args:
            batch: Gym environment to evaluate.
            batch_idx: Index of the batch (used as seed for reproducibility).

        Returns:
            Dictionary of metrics from the gym rollout evaluation.
        """
        return self.evaluate_gym(batch, batch_idx, stage="val")

    @torch.no_grad()
    def predict_action_chunk(self, batch: Observation) -> torch.Tensor:
        """Predict a chunk of actions from observation.

        Implements the abstract method from Policy base class.
        Returns action chunk that will be queued by select_action().

        Args:
            batch: Input observation.

        Returns:
            Action chunk tensor of shape (B, T, D) where T is chunk size.

        Raises:
            RuntimeError: If model is not initialized.
        """
        if self.model is None:
            msg = "Model not initialized"
            raise RuntimeError(msg)

        # Preprocess batch
        processed = self._preprocessor(batch)

        # Get device
        device = next(self.model.parameters()).device

        # Move to device
        observation = {
            "images": {k: v.to(device) for k, v in processed["images"].items()},
            "image_masks": {k: v.to(device) for k, v in processed["image_masks"].items()},
            "state": processed["state"].to(device),
            "tokenized_prompt": processed["tokenized_prompt"].to(device),
            "tokenized_prompt_mask": processed["tokenized_prompt_mask"].to(device),
        }

        # Sample actions
        self.model.eval()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self.hparams["use_bf16"]):
            action_chunk = self.model.sample_actions(device, observation)

        # Postprocess
        outputs = self._postprocessor({"actions": action_chunk})
        return outputs["actions"]

    # select_action() is inherited from Policy base class - uses queue with predict_action_chunk()

    def reset(self) -> None:
        """Reset policy state for new episode."""
        super().reset()  # Clears action queue

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and scheduler.

        Returns:
            Optimizer configuration dict.
        """
        # Get trainable parameters
        params = [p for p in self.parameters() if p.requires_grad]

        # Create optimizer (use config values)
        optimizer = torch.optim.AdamW(
            params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.95),
        )

        # Create scheduler with warmup
        # Calculate warmup_steps from warmup_ratio and total training steps
        warmup_ratio = self.config.warmup_ratio
        # Get total training steps from trainer (if available) or use a default
        if hasattr(self, "trainer") and self.trainer is not None:
            total_steps = getattr(self.trainer, "estimated_stepping_batches", 10000)
        else:
            # Default for unit tests or when trainer not attached
            total_steps = 10000
        warmup_steps = max(1, int(total_steps * warmup_ratio))

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def get_optim_params(self) -> dict[str, Any]:
        """Get optimizer parameters for external configuration.

        Returns:
            Dict with trainable parameters grouped.
        """
        return {
            "params": [p for p in self.parameters() if p.requires_grad],
            "lr": self.config.learning_rate,
            "weight_decay": self.config.weight_decay,
        }

    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        """Configure gradient clipping from policy config.

        This overrides Lightning's default gradient clipping to use
        the policy's grad_clip_norm config value.

        Args:
            optimizer: The optimizer being used.
            gradient_clip_val: Ignored (uses config value instead).
            gradient_clip_algorithm: Ignored (always uses 'norm').
        """
        # Use Trainer's value if set, otherwise fall back to policy config
        clip_val = gradient_clip_val if gradient_clip_val is not None else self.config.grad_clip_norm

        if clip_val and clip_val > 0:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=clip_val,
                gradient_clip_algorithm=gradient_clip_algorithm or "norm",
            )
