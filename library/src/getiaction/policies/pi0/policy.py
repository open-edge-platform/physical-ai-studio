# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2025 Physical Intelligence
# SPDX-License-Identifier: Apache-2.0

"""Pi0 Policy - Lightning wrapper for training and inference."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

import torch

from getiaction.config.mixin import FromConfig
from getiaction.export.mixin_export import Export
from getiaction.policies.base import Policy
from getiaction.train.utils import reformat_dataset_to_match_policy

from .config import Pi0Config
from .model import GemmaVariant, Pi0Model

if TYPE_CHECKING:
    from getiaction.data import Observation
    from getiaction.gyms import Gym

    from .preprocessor import Pi0Postprocessor, Pi0Preprocessor


class Pi0(Export, Policy, FromConfig):
    """Pi0 Policy - Physical Intelligence's flow matching VLA model.

    Lightning wrapper for training and inference with Pi0 model.

    Uses dual-path initialization:
    - **Lazy path**: `Pi0()` + `trainer.fit()` - model built in setup()
    - **Eager path**: `Pi0.load_from_checkpoint()` - model built immediately

    Args:
        variant: Model variant ("pi0" or "pi05"). Default: "pi0".
        paligemma_variant: PaliGemma backbone variant. Default: "gemma_2b".
        action_expert_variant: Action expert variant. Default: "gemma_300m".
        dtype: Data type for model. Default: "float32".
        n_obs_steps: Number of observation steps to use. Default: 1.
        chunk_size: Size of action chunks for prediction. Default: 50.
        n_action_steps: Number of action steps to execute. Default: 50.
        max_state_dim: Maximum state dimension (shorter vectors are padded). Default: 32.
        max_action_dim: Maximum action dimension (shorter vectors are padded). Default: 32.
        max_token_len: Maximum length for tokenizer. Default: None (auto-computed).
        image_resolution: Target image resolution (height, width). Default: (224, 224).
        num_inference_steps: Number of flow matching inference steps. Default: 10.
        time_beta_alpha: Beta distribution alpha parameter. Default: 1.5.
        time_beta_beta: Beta distribution beta parameter. Default: 1.0.
        time_scale: Time scaling factor. Default: 0.999.
        time_offset: Time offset. Default: 0.001.
        time_min_period: Minimum period for sinusoidal encoding. Default: 4e-3.
        time_max_period: Maximum period for sinusoidal encoding. Default: 4.0.
        tune_paligemma: Whether to tune PaliGemma weights. Default: False.
        tune_action_expert: Whether to tune action expert weights. Default: False.
        tune_vision_encoder: Whether to tune vision encoder weights. Default: False.
        lora_rank: LoRA rank (0 disables LoRA). Default: 0.
        lora_alpha: LoRA alpha parameter. Default: 16.
        lora_dropout: LoRA dropout rate. Default: 0.1.
        lora_target_modules: Target modules for LoRA. Default: ("q_proj", "v_proj", "k_proj", "o_proj").
        gradient_checkpointing: Whether to enable gradient checkpointing. Default: False.
        learning_rate: Learning rate for optimizer. Default: 2.5e-5.
        weight_decay: Weight decay for optimizer. Default: 0.01.
        warmup_ratio: Warmup ratio for scheduler. Default: 0.05.
        grad_clip_norm: Gradient clipping norm value. Default: 1.0.
        dataset_stats: Dataset normalization statistics for eager initialization. Default: None.

    Example:
        Training:

        >>> policy = Pi0(paligemma_variant="gemma_300m")
        >>> trainer = getiaction.Trainer(max_epochs=100)
        >>> trainer.fit(policy, datamodule)

        Inference:

        >>> policy = Pi0.load_from_checkpoint("checkpoint.ckpt")
        >>> action = policy.select_action(obs)
    """

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        # Model variant
        variant: Literal["pi0", "pi05"] = "pi0",
        paligemma_variant: str = "gemma_2b",
        action_expert_variant: str = "gemma_300m",
        dtype: str = "float32",
        # Input / output structure
        n_obs_steps: int = 1,
        chunk_size: int = 50,
        n_action_steps: int = 50,
        # Shorter state and action vectors will be padded
        max_state_dim: int = 32,
        max_action_dim: int = 32,
        max_token_len: int | None = None,
        # Image preprocessing
        image_resolution: tuple[int, int] = (224, 224),
        # Flow matching parameters
        num_inference_steps: int = 10,
        time_beta_alpha: float = 1.5,
        time_beta_beta: float = 1.0,
        time_scale: float = 0.999,
        time_offset: float = 0.001,
        time_min_period: float = 4e-3,
        time_max_period: float = 4.0,
        # Finetuning settings
        tune_paligemma: bool = False,  # noqa: FBT001, FBT002
        tune_action_expert: bool = False,  # noqa: FBT001, FBT002
        tune_vision_encoder: bool = False,  # noqa: FBT001, FBT002
        # LoRA settings
        lora_rank: int = 0,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lora_target_modules: tuple[str, ...] = ("q_proj", "v_proj", "k_proj", "o_proj"),
        # Gradient checkpointing
        gradient_checkpointing: bool = False,  # noqa: FBT001, FBT002
        # Training presets
        learning_rate: float = 2.5e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.05,
        grad_clip_norm: float = 1.0,
        *,
        # Eager initialization (for checkpoint loading)
        dataset_stats: dict[str, dict[str, list[float] | str | tuple]] | None = None,
    ) -> None:
        """Initialize Pi0 policy.

        Creates Pi0Config from explicit args and saves it as hyperparameters.
        """
        super().__init__(n_action_steps=n_action_steps)

        # Create config from explicit args (policy-level config)
        self.config: Pi0Config = Pi0Config(
            variant=variant,
            paligemma_variant=paligemma_variant,
            action_expert_variant=action_expert_variant,
            dtype=dtype,
            n_obs_steps=n_obs_steps,
            chunk_size=chunk_size,
            n_action_steps=n_action_steps,
            max_state_dim=max_state_dim,
            max_action_dim=max_action_dim,
            max_token_len=max_token_len,
            image_resolution=image_resolution,
            num_inference_steps=num_inference_steps,
            time_beta_alpha=time_beta_alpha,
            time_beta_beta=time_beta_beta,
            time_scale=time_scale,
            time_offset=time_offset,
            time_min_period=time_min_period,
            time_max_period=time_max_period,
            tune_paligemma=tune_paligemma,
            tune_action_expert=tune_action_expert,
            tune_vision_encoder=tune_vision_encoder,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
            gradient_checkpointing=gradient_checkpointing,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            grad_clip_norm=grad_clip_norm,
        )

        # Save config as hyperparameters for checkpoint restoration
        self.save_hyperparameters(ignore=["config"])  # Save individual args, not config object
        # Also save config dict for compatibility
        self.hparams["config"] = self.config.to_dict()

        # Model will be built in setup() or immediately if dataset_stats provided
        self.model: torch.nn.Module | None = None

        # Preprocessor/postprocessor set in setup() or _initialize_model()
        self._preprocessor: Pi0Preprocessor | None = None
        self._postprocessor: Pi0Postprocessor | None = None

        # Eager initialization if dataset_stats is provided
        if dataset_stats is not None:
            self._initialize_model(dataset_stats)

        self._dataset_stats = dataset_stats

    def _initialize_model(
        self,
        dataset_stats: dict[str, dict[str, list[float] | str | tuple[int, ...]]],
    ) -> None:
        """Initialize model and preprocessors.

        Called by both lazy (setup) and eager (checkpoint) paths.

        Args:
            dataset_stats: Dataset normalization statistics.
        """
        from .preprocessor import make_pi0_preprocessors  # noqa: PLC0415

        # Determine action dimension from stats
        action_key = next((k for k in dataset_stats if "action" in k.lower()), None)
        env_action_dim: int | None = None
        if action_key and "shape" in dataset_stats[action_key]:
            shape = dataset_stats[action_key]["shape"]
            if isinstance(shape, (list, tuple)):
                env_action_dim = int(shape[-1])

        self.model = Pi0Model(
            variant=self.config.variant,
            paligemma_variant=cast("GemmaVariant", self.config.paligemma_variant),
            action_expert_variant=cast("GemmaVariant", self.config.action_expert_variant),
            max_action_dim=self.config.max_action_dim,
            max_state_dim=self.config.max_state_dim,
            chunk_size=self.config.chunk_size,
            num_inference_steps=self.config.num_inference_steps,
            dtype=self.config.dtype,
            time_beta_alpha=self.config.time_beta_alpha,
            time_beta_beta=self.config.time_beta_beta,
            time_scale=self.config.time_scale,
            time_offset=self.config.time_offset,
            time_min_period=self.config.time_min_period,
            time_max_period=self.config.time_max_period,
        )

        # Set trainable parameters
        self.model.set_trainable_parameters(
            tune_paligemma=self.config.tune_paligemma,
            tune_action_expert=self.config.tune_action_expert,
            tune_vision_encoder=self.config.tune_vision_encoder,
            tune_projection_heads=True,
        )

        # Enable gradient checkpointing if requested
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self._preprocessor, self._postprocessor = make_pi0_preprocessors(
            max_state_dim=self.config.max_state_dim,
            max_action_dim=self.config.max_action_dim,
            chunk_size=self.config.chunk_size,
            env_action_dim=env_action_dim,
            stats=dataset_stats,
            use_quantile_norm=self.config.is_pi05,
            image_resolution=self.config.image_resolution,
            max_token_len=self.config.max_token_len or 48,
        )

        if self.model is not None:
            cast("Pi0Model", self.model).set_preprocessors(self._preprocessor, self._postprocessor)

    def setup(self, stage: str) -> None:
        """Set up model from datamodule (lazy initialization path).

        Called by Lightning before fit/validate/test/predict.

        Args:
            stage: Lightning stage (unused, required by Lightning API).

        Raises:
            TypeError: If train dataset is not a getiaction Dataset.
        """
        del stage  # Unused argument

        if self.model is not None:
            return

        from getiaction.data.dataset import Dataset  # noqa: PLC0415

        datamodule = self.trainer.datamodule  # type: ignore[attr-defined]
        train_dataset = datamodule.train_dataset

        if not isinstance(train_dataset, Dataset):
            msg = f"Expected getiaction Dataset, got {type(train_dataset)}"
            raise TypeError(msg)

        stats_dict = train_dataset.stats

        # Save to hparams for checkpoint
        self.hparams["dataset_stats"] = stats_dict

        self._initialize_model(stats_dict)

        reformat_dataset_to_match_policy(self, datamodule)

    def forward(self, batch: Observation) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        """Forward pass through the model.

        Processes the input batch and either trains the model or predicts actions
        depending on the current mode.

        Args:
            batch: An Observation object containing the input data for the model.

        Returns:
            If training: Returns the model output as a tuple
                containing a loss tensor and a dictionary of loss metrics.
            If not training: Returns the predicted action chunk as a tensor.

        Raises:
            ValueError: If the model is not initialized during training mode.
        """
        if self.training:
            if self.model is None or self._preprocessor is None:
                msg = "Model is not initialized"
                raise ValueError(msg)

            return self.model(batch)
        return self.predict_action_chunk(batch)

    @torch.no_grad()
    def predict_action_chunk(self, batch: Observation) -> torch.Tensor:
        """Predict a chunk of actions from the given observation batch.

        Args:
            batch: An Observation object containing the input data for action prediction.

        Returns:
            torch.Tensor: The predicted action chunk after post-processing.

        Raises:
            ValueError: If the model has not been initialized.
        """
        if self.model is None or self._preprocessor is None or self._postprocessor is None:
            msg = "Model is not initialized"
            raise ValueError(msg)

        model = cast("Pi0Model", self.model)
        return model.predict_action_chunk(batch)

    def training_step(self, batch: Observation, batch_idx: int) -> torch.Tensor:
        """Lightning training step.

        Args:
            batch: Input batch.
            batch_idx: Batch index (unused, required by Lightning API).

        Returns:
            Loss tensor for backpropagation.
        """
        del batch_idx
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

    def configure_optimizers(self) -> Any:  # noqa: ANN401
        """Configure optimizer and scheduler.

        Returns:
            Optimizer configuration dict.
        """
        # Get trainable parameters
        params = [p for p in self.parameters() if p.requires_grad]

        # Create optimizer
        optimizer = torch.optim.AdamW(
            params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Calculate warmup steps based on trainer's estimated stepping batches
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * self.config.warmup_ratio)

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

    @property
    def metadata_extra(self) -> dict[str, Any]:
        """Return extra metadata for policy export."""
        return {
            "chunk_size": self.config.chunk_size,
            "use_action_queue": True,
        }


class Pi05(Pi0):
    """Pi0.5 Policy - Physical Intelligence's improved flow matching VLA model.

    This is a convenience alias for Pi0 with variant="pi05".
    Pi0.5 uses AdaRMS conditioning and quantile normalization.

    Example:
        Training:

        >>> policy = Pi05(paligemma_variant="gemma_300m")
        >>> trainer = getiaction.Trainer(max_epochs=100)
        >>> trainer.fit(policy, datamodule)
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        paligemma_variant: str = "gemma_2b",
        action_expert_variant: str = "gemma_300m",
        dtype: str = "float32",
        n_obs_steps: int = 1,
        chunk_size: int = 50,
        n_action_steps: int = 50,
        max_state_dim: int = 32,
        max_action_dim: int = 32,
        max_token_len: int | None = 200,
        image_resolution: tuple[int, int] = (224, 224),
        num_inference_steps: int = 10,
        time_beta_alpha: float = 1.5,
        time_beta_beta: float = 1.0,
        time_scale: float = 0.999,
        time_offset: float = 0.001,
        time_min_period: float = 4e-3,
        time_max_period: float = 4.0,
        tune_paligemma: bool = False,
        tune_action_expert: bool = False,
        tune_vision_encoder: bool = False,
        lora_rank: int = 0,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lora_target_modules: tuple[str, ...] = ("q_proj", "v_proj", "k_proj", "o_proj"),
        gradient_checkpointing: bool = False,
        learning_rate: float = 2.5e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.05,
        grad_clip_norm: float = 1.0,
        dataset_stats: dict[str, dict[str, list[float] | str | tuple[int, ...]]] | None = None,
    ) -> None:
        """Initialize Pi0.5 policy with explicit arguments."""
        super().__init__(
            variant="pi05",
            paligemma_variant=paligemma_variant,
            action_expert_variant=action_expert_variant,
            dtype=dtype,
            n_obs_steps=n_obs_steps,
            chunk_size=chunk_size,
            n_action_steps=n_action_steps,
            max_state_dim=max_state_dim,
            max_action_dim=max_action_dim,
            max_token_len=max_token_len,
            image_resolution=image_resolution,
            num_inference_steps=num_inference_steps,
            time_beta_alpha=time_beta_alpha,
            time_beta_beta=time_beta_beta,
            time_scale=time_scale,
            time_offset=time_offset,
            time_min_period=time_min_period,
            time_max_period=time_max_period,
            tune_paligemma=tune_paligemma,
            tune_action_expert=tune_action_expert,
            tune_vision_encoder=tune_vision_encoder,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
            gradient_checkpointing=gradient_checkpointing,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            grad_clip_norm=grad_clip_norm,
            dataset_stats=dataset_stats,
        )
