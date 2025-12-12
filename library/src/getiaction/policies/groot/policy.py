# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Groot Policy - First-party Lightning wrapper for NVIDIA's GR00T-N1.5 foundation model.

This module provides a PyTorch Lightning policy for training and inference with
NVIDIA's GR00T-N1.5-3B model, using PyTorch native SDPA attention for wider
device support (CUDA, XPU) without requiring the Flash Attention CUDA package.

## Quick Start

```python
from getiaction.policies.groot import Groot, GrootConfig
from getiaction.data.lerobot import LeRobotDataModule
import lightning as L

# Create policy with explicit args
policy = Groot(
    chunk_size=50,
    attn_implementation='sdpa',  # PyTorch native attention
    tune_projector=True,
    tune_diffusion_model=True,
)

# Or create from config
config = GrootConfig(chunk_size=50, learning_rate=1e-4)
policy = Groot.from_config(config)

# Create datamodule
datamodule = LeRobotDataModule(
    repo_id="lerobot/aloha_sim_transfer_cube_human",
    train_batch_size=4,
)

# Train
trainer = L.Trainer(max_epochs=100, precision="bf16-mixed")
trainer.fit(policy, datamodule)

# Load checkpoint (native Lightning - just works!)
policy = Groot.load_from_checkpoint("checkpoint.ckpt")
```

## Attention Implementations

- `sdpa` (default): PyTorch native SDPA - works on CUDA and XPU
- `flash_attention_2`: Requires flash-attn CUDA package
- `eager`: Fallback Python implementation
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch

from getiaction.policies.base import Policy

from .config import GrootConfig
from .model import GrootModel

if TYPE_CHECKING:
    from getiaction.data import Observation
    from getiaction.gyms import Gym

logger = logging.getLogger(__name__)


class Groot(Policy):
    """Groot (GR00T-N1.5) Policy - NVIDIA's foundation model for humanoid robots.

    First-party Lightning wrapper with explicit hyperparameters in __init__.
    Uses PyTorch native SDPA attention by default for wider device support.

    Supports dual-path initialization per the design docs:
    - **Lazy path**: `Groot()` + `trainer.fit()` - model built in setup()
    - **Eager path**: `Groot.load_from_checkpoint()` or `Groot(env_action_dim=2)` - model built immediately

    All hyperparameters are explicit in the signature for discoverability.
    Native Lightning checkpoint loading works automatically via save_hyperparameters().

    Args:
        chunk_size: Number of action predictions per forward pass.
        n_action_steps: Number of action steps to execute per chunk.
        max_state_dim: Maximum state dimension (shorter states zero-padded).
        max_action_dim: Maximum action dimension (shorter actions zero-padded).
        base_model_path: HuggingFace model ID or path to base Groot model.
        tokenizer_assets_repo: HF repo ID for Eagle tokenizer assets.
        embodiment_tag: Embodiment tag for training.
        attn_implementation: Attention backend ('sdpa', 'flash_attention_2', 'eager').
        tune_llm: Whether to fine-tune the LLM backbone.
        tune_visual: Whether to fine-tune the vision tower.
        tune_projector: Whether to fine-tune the projector.
        tune_diffusion_model: Whether to fine-tune the diffusion model.
        learning_rate: Learning rate for optimizer.
        weight_decay: Weight decay for optimizer.
        warmup_ratio: Warmup ratio (0.0-1.0) of total training steps.
        grad_clip_norm: Gradient clipping norm (0.0 = disabled).
        use_bf16: Whether to use bfloat16 precision.
        env_action_dim: Environment action dimension. If provided, enables eager initialization.
            This is saved during training and restored during checkpoint loading.
        dataset_stats: Dataset normalization statistics. If provided with env_action_dim,
            enables full eager initialization including preprocessor.

    Examples:
        Training (lazy initialization):

        >>> policy = Groot(chunk_size=50, learning_rate=1e-4)
        >>> trainer = L.Trainer(max_epochs=100)
        >>> trainer.fit(policy, datamodule)

        Load from checkpoint (eager initialization - just works!):

        >>> policy = Groot.load_from_checkpoint("checkpoint.ckpt")
        >>> action = policy.select_action(obs)  # Works immediately!

        Standalone inference (eager initialization):

        >>> policy = Groot(env_action_dim=2, dataset_stats=stats)
        >>> action = policy.select_action(obs)

        Using config:

        >>> config = GrootConfig(chunk_size=50, learning_rate=1e-4)
        >>> policy = Groot.from_config(config)
    """

    def __init__(  # noqa: PLR0913
        self,
        # Model architecture
        chunk_size: int = 50,
        n_action_steps: int = 50,
        max_state_dim: int = 64,
        max_action_dim: int = 32,
        # Model source
        base_model_path: str = "nvidia/GR00T-N1.5-3B",
        tokenizer_assets_repo: str = "lerobot/eagle2hg-processor-groot-n1p5",
        embodiment_tag: str = "new_embodiment",
        # Attention implementation
        attn_implementation: str = "sdpa",
        # Fine-tuning control
        *,
        tune_llm: bool = False,
        tune_visual: bool = False,
        tune_projector: bool = True,
        tune_diffusion_model: bool = True,
        # Optimizer
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        warmup_ratio: float = 0.05,
        grad_clip_norm: float = 1.0,
        # Precision
        use_bf16: bool = True,
        # Eager initialization (optional - for checkpoint loading and standalone use)
        env_action_dim: int | None = None,
        dataset_stats: dict[str, dict[str, list[float]]] | None = None,
    ) -> None:
        """Initialize Groot policy.

        Creates GrootConfig from explicit args and saves it as hyperparameters.
        Supports dual-path initialization:
        - Lazy: model=None, built in setup() when dataset features are known
        - Eager: model built immediately when env_action_dim is provided

        The eager path is used by load_from_checkpoint() since env_action_dim
        is saved in hyperparameters during training.
        """
        super().__init__()

        # Create config from explicit args (policy-level config)
        self.config = GrootConfig(
            chunk_size=chunk_size,
            n_action_steps=n_action_steps,
            max_state_dim=max_state_dim,
            max_action_dim=max_action_dim,
            base_model_path=base_model_path,
            tokenizer_assets_repo=tokenizer_assets_repo,
            embodiment_tag=embodiment_tag,
            attn_implementation=attn_implementation,
            tune_llm=tune_llm,
            tune_visual=tune_visual,
            tune_projector=tune_projector,
            tune_diffusion_model=tune_diffusion_model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            grad_clip_norm=grad_clip_norm,
            use_bf16=use_bf16,
        )

        # Save config as hyperparameters for checkpoint restoration
        self.save_hyperparameters(ignore=["config"])  # Save individual args, not config object
        # Also save config dict for compatibility
        self.hparams["config"] = self.config.to_dict()

        # Model will be built in setup() or immediately if env_action_dim provided
        self.model: GrootModel | None = None

        # Preprocessor/postprocessor set in setup() or _initialize_model()
        self._preprocessor: Any = None
        self._postprocessor: Any = None

        # Track initialization state
        self._is_setup_complete: bool = False

        # Eager initialization if env_action_dim is provided (e.g., from checkpoint)
        if env_action_dim is not None:
            self._initialize_model(env_action_dim, dataset_stats)

    def _initialize_model(
        self,
        env_action_dim: int,
        dataset_stats: dict[str, dict[str, list[float]]] | None = None,
    ) -> None:
        """Initialize model and preprocessors.

        This is the core initialization method called by both paths:
        - Lazy: Called from setup() with features extracted from DataModule
        - Eager: Called from __init__ when env_action_dim is provided

        Args:
            env_action_dim: Environment action dimension.
            dataset_stats: Dataset normalization statistics (with list values, not tensors).
        """
        from .preprocessor import make_groot_preprocessors  # noqa: PLC0415

        # Use config (policy-level config created in __init__)
        config = self.config

        # Load pretrained model with explicit args from config
        self.model = GrootModel.from_pretrained(
            pretrained_model_name_or_path=config.base_model_path,
            n_action_steps=config.n_action_steps,
            use_bf16=config.use_bf16,
            tokenizer_assets_repo=config.tokenizer_assets_repo,
            attn_implementation=config.attn_implementation,
            tune_llm=config.tune_llm,
            tune_visual=config.tune_visual,
            tune_projector=config.tune_projector,
            tune_diffusion_model=config.tune_diffusion_model,
            chunk_size=config.chunk_size,
            max_action_dim=config.max_action_dim,
            revision=config.revision,
        )

        # Create first-party preprocessor/postprocessor
        self._preprocessor, self._postprocessor = make_groot_preprocessors(
            max_state_dim=config.max_state_dim,
            max_action_dim=config.max_action_dim,
            action_horizon=min(config.chunk_size, 16),  # GR00T max is 16
            embodiment_tag=config.embodiment_tag,
            env_action_dim=env_action_dim,
            stats=dataset_stats,
            eagle_processor_repo=config.tokenizer_assets_repo,
        )

        self._is_setup_complete = True

    def setup(self, stage: str) -> None:  # noqa: ARG002
        """Set up model from datamodule (lazy initialization path).

        Called by Lightning before fit/validate/test/predict.
        Skips if already initialized (eager path via checkpoint or env_action_dim).

        This implements the lazy path of dual-path initialization:
        - Extracts features from dataset
        - Calls _initialize_model() to build model
        - Saves env_action_dim and stats to hparams for checkpoint

        Args:
            stage: Lightning stage ('fit', 'validate', 'test', 'predict').

        Raises:
            TypeError: If dataset is not a LeRobot dataset.
        """
        if self._is_setup_complete or self.model is not None:
            return  # Already initialized (eager path)

        # Get dataset for features and stats
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415

        from getiaction.data.lerobot.dataset import _LeRobotDatasetAdapter  # noqa: PLC0415

        datamodule = self.trainer.datamodule  # type: ignore[attr-defined]
        train_dataset = datamodule.train_dataset

        # Get underlying LeRobot dataset
        if isinstance(train_dataset, _LeRobotDatasetAdapter):
            lerobot_dataset = train_dataset._lerobot_dataset  # noqa: SLF001
        elif isinstance(train_dataset, LeRobotDataset):
            lerobot_dataset = train_dataset
        else:
            msg = f"Expected LeRobot dataset, got {type(train_dataset)}"
            raise TypeError(msg)

        # Get dataset stats for normalization
        dataset_stats = lerobot_dataset.meta.stats

        # Determine environment action dimension from features
        env_action_dim = 0
        if hasattr(lerobot_dataset.meta, "features") and "action" in lerobot_dataset.meta.features:
            action_feature = lerobot_dataset.meta.features["action"]
            if hasattr(action_feature, "shape"):
                env_action_dim = action_feature.shape[0]

        # Convert stats tensors to lists for JSON serialization in checkpoint
        serializable_stats = self._serialize_stats(dataset_stats)

        # Save to hparams so checkpoint loading can use eager path
        self.hparams["env_action_dim"] = env_action_dim
        self.hparams["dataset_stats"] = serializable_stats

        # Initialize model using shared method
        self._initialize_model(env_action_dim, serializable_stats)

    @staticmethod
    def _serialize_stats(
        stats: dict[str, dict[str, Any]] | None,
    ) -> dict[str, dict[str, list[float]]] | None:
        """Convert stats with tensors to JSON-serializable format.

        Args:
            stats: Dataset stats with tensor values.

        Returns:
            Stats with list values (JSON-serializable for checkpoint hparams).
        """
        if stats is None:
            return None

        import torch  # noqa: PLC0415

        serializable: dict[str, dict[str, list[float]]] = {}
        for key, stat_dict in stats.items():
            serializable[key] = {}
            for stat_name, value in stat_dict.items():
                if isinstance(value, torch.Tensor) or hasattr(value, "tolist"):
                    serializable[key][stat_name] = value.tolist()
                else:
                    serializable[key][stat_name] = list(value) if hasattr(value, "__iter__") else [value]
        return serializable

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:  # type: ignore[override]
        """Forward pass - delegates to model.

        Args:
            batch: Preprocessed input batch.

        Returns:
            Model output (loss dict during training, actions during inference).

        Raises:
            RuntimeError: If model is not initialized.
        """
        if self.model is None:
            msg = "Model not initialized. Call setup() first."
            raise RuntimeError(msg)
        return self.model(batch)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step - compute loss.

        Args:
            batch: Input batch in LeRobot dict format.
            batch_idx: Batch index.

        Returns:
            Training loss.
        """
        del batch_idx  # Unused

        # Apply preprocessing
        batch = self._preprocessor(batch)

        # Forward pass
        outputs = self(batch)
        loss = outputs.get("loss")

        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Gym, batch_idx: int) -> dict[str, float]:
        """Validation step - gym rollout.

        Args:
            batch: Gym environment to evaluate.
            batch_idx: Batch index.

        Returns:
            Metrics from rollout.
        """
        return self.evaluate_gym(batch, batch_idx, stage="val")

    def test_step(self, batch: Gym, batch_idx: int) -> dict[str, float]:
        """Test step - gym rollout.

        Args:
            batch: Gym environment to evaluate.
            batch_idx: Batch index.

        Returns:
            Metrics from rollout.
        """
        return self.evaluate_gym(batch, batch_idx, stage="test")

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and scheduler.

        Returns:
            Optimizer configuration dict with scheduler.

        Raises:
            RuntimeError: If model is not initialized.
        """
        if self.model is None:
            msg = "Model not initialized."
            raise RuntimeError(msg)

        # Get trainable parameters
        params = self.model.get_optim_params()

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

    def select_action(self, batch: Observation | dict[str, torch.Tensor]) -> torch.Tensor:
        """Select action for inference.

        Args:
            batch: Input observation.

        Returns:
            Selected action tensor.

        Raises:
            RuntimeError: If model is not initialized.
        """
        from getiaction.data import Observation  # noqa: PLC0415
        from getiaction.data.lerobot import FormatConverter  # noqa: PLC0415

        if self.model is None:
            msg = "Model not initialized."
            raise RuntimeError(msg)

        # Convert to dict format if needed
        batch_dict = FormatConverter.to_lerobot_dict(batch) if isinstance(batch, Observation) else batch

        # Preprocess
        batch_dict = self._preprocessor(batch_dict)

        # Move to model device for inference
        batch_dict = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch_dict.items()}

        # Get action
        action = self.model.select_action(batch_dict)

        # Postprocess
        return self._postprocessor(action)

    def reset(self) -> None:
        """Reset policy state for new episode."""
        if self.model is not None:
            self.model.reset()

    def get_optim_params(self) -> dict[str, Any]:
        """Get optimizer parameters for external configuration.

        Returns:
            Dict with trainable parameters grouped.

        Raises:
            RuntimeError: If model is not initialized.
        """
        if self.model is None:
            msg = "Model not initialized."
            raise RuntimeError(msg)

        return {
            "params": self.model.get_optim_params(),
            "lr": self.config.learning_rate,
            "weight_decay": self.config.weight_decay,
        }

    @classmethod
    def from_config(
        cls,
        config: GrootConfig,
        *,
        env_action_dim: int | None = None,
        dataset_stats: dict[str, dict[str, list[float]]] | None = None,
    ) -> Groot:
        """Create Groot policy from a config object.

        This provides an alternative to passing explicit args to __init__.

        Args:
            config: GrootConfig instance with policy settings.
            env_action_dim: Environment action dimension for eager initialization.
            dataset_stats: Dataset normalization statistics for eager initialization.

        Returns:
            Groot policy instance.

        Example:
            >>> config = GrootConfig(chunk_size=50, learning_rate=1e-4)
            >>> policy = Groot.from_config(config)
        """
        return cls(
            chunk_size=config.chunk_size,
            n_action_steps=config.n_action_steps,
            max_state_dim=config.max_state_dim,
            max_action_dim=config.max_action_dim,
            base_model_path=config.base_model_path,
            tokenizer_assets_repo=config.tokenizer_assets_repo,
            embodiment_tag=config.embodiment_tag,
            attn_implementation=config.attn_implementation,
            tune_llm=config.tune_llm,
            tune_visual=config.tune_visual,
            tune_projector=config.tune_projector,
            tune_diffusion_model=config.tune_diffusion_model,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            warmup_ratio=config.warmup_ratio,
            grad_clip_norm=config.grad_clip_norm,
            use_bf16=config.use_bf16,
            env_action_dim=env_action_dim,
            dataset_stats=dataset_stats,
        )
