# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0

"""SmolVLA Policy - Lightning wrapper for training and inference."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING, Any

from getiaction.policies.base import Policy
from getiaction.data.observation import ACTION, EXTRA, STATE, IMAGES
from getiaction.data import Observation

from .model import SmolVLAModel
from .config import SmolVLAConfig


if TYPE_CHECKING:
    from getiaction.gyms import Gym


class SmolVLA(Policy):
    """SmolVLA Policy - Hugging Face's flow matching VLA model.

    Lightning wrapper for training and inference with SmolVLA model.

    Uses dual-path initialization:
    - **Lazy path**: `SmolVLA()` + `trainer.fit()` - model built in setup()
    - **Eager path**: `SmolVLA.load_from_checkpoint()` - model built immediately

    Args:


    Example:
        Training:

        >>> policy = SmolVLA(learning_rate=2.5e-5)
        >>> trainer = L.Trainer(max_epochs=100)
        >>> trainer.fit(policy, datamodule)

        Inference:

        >>> policy = SmolVLA.load_from_checkpoint("checkpoint.ckpt")
        >>> action = policy.select_action(obs)
    """

    def __init__(  # noqa: PLR0913
        self,
        # Input / output structure.
        n_obs_steps: int = 1,
        chunk_size: int = 50,
        n_action_steps: int = 50,
        # Shorter state and action vectors will be padded
        max_state_dim: int = 32,
        max_action_dim: int = 32,
        # Image preprocessing
        resize_imgs_with_padding: tuple[int, int] = (512, 512),
        # Architecture
        tokenizer_max_length: int = 48,

        vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",  # Select the VLM backbone.
        load_vlm_weights: bool = False,  # Set to True in case of training the expert from scratch. True when init from pretrained SmolVLA weights

        add_image_special_tokens: bool = False,  # Whether to use special image tokens around image features.

        attention_mode: str = "cross_attn",

        prefix_length: int = -1,
        pad_language_to: str = "longest",  # "max_length"

        num_expert_layers: int = -1,  # Less or equal to 0 is the default where the action expert has the same number of layers of VLM. Otherwise the expert have less layers.
        num_vlm_layers: int = 16,  # Number of layers used in the VLM (first num_vlm_layers layers)
        self_attn_every_n_layers: int = 2,  # Interleave SA layers each self_attn_every_n_layers
        expert_width_multiplier: float = 0.75,  # The action expert hidden size (wrt to the VLM)

        min_period: float = 4e-3,  # sensitivity range for the timestep used in sine-cosine positional encoding
        max_period: float = 4.0,

        # Decoding
        num_steps: int = 10,
        # Attention utils
        use_cache: bool = True,

        # Finetuning settings
        freeze_vision_encoder: bool = True,
        train_expert_only: bool = True,
        train_state_proj: bool = True,

        # Training presets
        optimizer_lr: float = 1e-4,
        optimizer_betas: tuple[float, float] = (0.9, 0.95),
        optimizer_eps: float = 1e-8,
        optimizer_weight_decay: float = 1e-10,
        optimizer_grad_clip_norm: float = 10,

        scheduler_warmup_steps: int = 1_000,
        scheduler_decay_steps: int = 30_000,
        scheduler_decay_lr: float = 2.5e-6,

        # Eager initialization (for checkpoint loading)
        dataset_stats: dict[str, dict[str, list[float] | str | tuple]] | None = None,
    ) -> None:
        """Initialize Pi0 policy.

        Creates Pi0Config from explicit args and saves it as hyperparameters.
        """
        super().__init__(n_action_steps=n_action_steps)

        # Create config from explicit args (policy-level config)
        self.config = SmolVLAConfig(
            n_obs_steps=n_obs_steps,
            chunk_size=chunk_size,
            n_action_steps=n_action_steps,
            max_state_dim=max_state_dim,
            max_action_dim=max_action_dim,
            resize_imgs_with_padding=resize_imgs_with_padding,
            tokenizer_max_length=tokenizer_max_length,
            vlm_model_name=vlm_model_name,
            load_vlm_weights=load_vlm_weights,
            add_image_special_tokens=add_image_special_tokens,
            attention_mode=attention_mode,
            prefix_length=prefix_length,
            pad_language_to=pad_language_to,
            num_expert_layers=num_expert_layers,
            num_vlm_layers=num_vlm_layers,
            self_attn_every_n_layers=self_attn_every_n_layers,
            expert_width_multiplier=expert_width_multiplier,
            min_period=min_period,
            max_period=max_period,
            num_steps=num_steps,
            use_cache=use_cache,
            freeze_vision_encoder=freeze_vision_encoder,
            train_expert_only=train_expert_only,
            train_state_proj=train_state_proj,
            optimizer_lr=optimizer_lr,
            optimizer_betas=optimizer_betas,
            optimizer_eps=optimizer_eps,
            optimizer_weight_decay=optimizer_weight_decay,
            optimizer_grad_clip_norm=optimizer_grad_clip_norm,
            scheduler_warmup_steps=scheduler_warmup_steps,
            scheduler_decay_steps=scheduler_decay_steps,
            scheduler_decay_lr=scheduler_decay_lr,
        )

        # Save config as hyperparameters for checkpoint restoration
        self.save_hyperparameters(ignore=["config"])  # Save individual args, not config object
        # Also save config dict for compatibility
        self.hparams["config"] = self.config.to_dict()

        # Model will be built in setup() or immediately if env_action_dim provided
        self.model: SmolVLAModel | None = None

        # Preprocessor/postprocessor set in setup() or _initialize_model()
        self._preprocessor: Any = None
        self._postprocessor: Any = None

        # Track initialization state
        self._is_setup_complete: bool = False

        # Eager initialization if dataset_stats is provided
        if dataset_stats is not None:
            self._initialize_model(dataset_stats)

        self._dataset_stats = dataset_stats

    def _initialize_model(
        self,
        dataset_stats: dict[str, dict[str, list[float] | str | tuple]] | None = None,
    ) -> None:
        """Initialize model and preprocessors.

        Called by both lazy (setup) and eager (checkpoint) paths.

        Args:
            env_action_dim: Environment action dimension.
            dataset_stats: Dataset normalization statistics.

        Raises:
            ValueError: If max_token_len is not set in config.
        """
        from .preprocessor import make_smolvla_preprocessors  # noqa: PLC0415

        # Use config (policy-level config created in __init__)
        config = self.config

        # Create model with explicit args (no config dependency)
        self.model = SmolVLAModel(config)

        # Create preprocessor/postprocessor
        self._preprocessor, self._postprocessor = make_smolvla_preprocessors(
            max_state_dim=config.max_state_dim,
            max_action_dim=config.max_action_dim,
            stats=dataset_stats,
            image_resolution=config.resize_imgs_with_padding,
            max_token_len=config.tokenizer_max_length,
        )

        self._is_setup_complete = True

    def setup(self, stage: str) -> None:
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
        self._initialize_model(stats_dict)

    def forward(self, batch: Observation) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        """Do a full training forward pass to compute the loss"""
        if self.training:
            if self.model is None:
                raise ValueError("Model not initialized")

            processed_batch = self._preprocessor(batch.to_dict())
            return self.model(processed_batch)
        else:
            return self.predict_action_chunk(batch)

    @torch.no_grad()
    def predict_action_chunk(self, batch: Observation) -> torch.Tensor:
        if self.model is None:
            raise ValueError("Model not initialized")

        processed_batch = self._preprocessor(batch.to(self.device).to_dict())
        return self.model.predict_action_chunk(processed_batch)

    def training_step(self, batch: Observation, batch_idx: int) -> torch.Tensor:
        """Lightning training step.

        Args:
            batch: Input batch.
            batch_idx: Batch index (unused, required by Lightning API).

        Returns:
            Loss tensor for backpropagation.
        """
        loss, loss_dict = self(batch)

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
            lr=self.config.optimizer_lr,
            weight_decay=self.config.optimizer_weight_decay,
            betas=self.config.optimizer_betas,
        )

        warmup_steps = self.config.scheduler_warmup_steps
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            return 1.0

        # also add decay after warmup

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
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
        clip_val = gradient_clip_val if gradient_clip_val is not None else self.config.optimizer_grad_clip_norm

        if clip_val and clip_val > 0:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=clip_val,
                gradient_clip_algorithm=gradient_clip_algorithm or "norm",
            )
