# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Groot Policy - First-party Lightning wrapper for NVIDIA's GR00T-N1.5 foundation model.

This module provides a PyTorch Lightning policy for training and inference with
NVIDIA's GR00T-N1.5-3B model, using PyTorch native SDPA attention for wider
device support (CUDA, XPU) without requiring the Flash Attention CUDA package.

## Quick Start

```python
from getiaction.policies.groot import Groot
from getiaction.data.lerobot import LeRobotDataModule
import lightning as L

# Create policy with explicit args
policy = Groot(
    chunk_size=50,
    attn_implementation='sdpa',  # PyTorch native attention
    tune_projector=True,
    tune_diffusion_model=True,
)

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

from typing import TYPE_CHECKING, Any

import torch

from getiaction.policies.base import Policy

from .model import GrootModel

if TYPE_CHECKING:
    from getiaction.data import Observation
    from getiaction.gyms import Gym


class Groot(Policy):
    """Groot (GR00T-N1.5) Policy - NVIDIA's foundation model for humanoid robots.

    First-party Lightning wrapper with explicit hyperparameters in __init__.
    Uses PyTorch native SDPA attention by default for wider device support.

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
        use_bf16: Whether to use bfloat16 precision.

    Examples:
        Training:

        >>> policy = Groot(chunk_size=50, learning_rate=1e-4)
        >>> trainer = L.Trainer(max_epochs=100)
        >>> trainer.fit(policy, datamodule)

        Load from checkpoint (native Lightning):

        >>> policy = Groot.load_from_checkpoint("checkpoint.ckpt")

        Access model for export:

        >>> model = policy.model
        >>> scripted = torch.jit.script(model)
    """

    def __init__(
        self,
        # Model architecture
        chunk_size: int = 50,  # noqa: ARG002 - used via save_hyperparameters()
        n_action_steps: int = 50,  # noqa: ARG002 - used via save_hyperparameters()
        max_state_dim: int = 64,  # noqa: ARG002 - used via save_hyperparameters()
        max_action_dim: int = 32,  # noqa: ARG002 - used via save_hyperparameters()
        # Model source
        base_model_path: str = "nvidia/GR00T-N1.5-3B",  # noqa: ARG002 - used via save_hyperparameters()
        tokenizer_assets_repo: str = "lerobot/eagle2hg-processor-groot-n1p5",  # noqa: ARG002
        embodiment_tag: str = "new_embodiment",  # noqa: ARG002 - used via save_hyperparameters()
        # Attention implementation
        attn_implementation: str = "sdpa",  # noqa: ARG002 - used via save_hyperparameters()
        # Fine-tuning control
        *,
        tune_llm: bool = False,  # noqa: ARG002 - used via save_hyperparameters()
        tune_visual: bool = False,  # noqa: ARG002 - used via save_hyperparameters()
        tune_projector: bool = True,  # noqa: ARG002 - used via save_hyperparameters()
        tune_diffusion_model: bool = True,  # noqa: ARG002 - used via save_hyperparameters()
        # Optimizer
        learning_rate: float = 1e-4,  # noqa: ARG002 - used via save_hyperparameters()
        weight_decay: float = 1e-5,  # noqa: ARG002 - used via save_hyperparameters()
        # Precision
        use_bf16: bool = True,  # noqa: ARG002 - used via save_hyperparameters()
    ) -> None:
        """Initialize Groot policy.

        Model is built lazily in setup() when dataset features are known.
        """
        super().__init__()

        # Save ALL hyperparameters - enables native load_from_checkpoint()
        self.save_hyperparameters()

        # Model will be built in setup()
        self.model: GrootModel | None = None

        # Preprocessor/postprocessor set in setup()
        self._preprocessor: Any = None
        self._postprocessor: Any = None

    def setup(self, stage: str) -> None:  # noqa: ARG002
        """Set up model from datamodule.

        Called by Lightning before fit/validate/test/predict.
        Extracts features from dataset and initializes the model.

        Args:
            stage: Lightning stage ('fit', 'validate', 'test', 'predict').

        Raises:
            TypeError: If dataset is not a LeRobot dataset.
        """
        if self.model is not None:
            return  # Already initialized

        # Get dataset for features and stats
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415

        from getiaction.data.lerobot.dataset import _LeRobotDatasetAdapter  # noqa: PLC0415

        from .preprocessor import make_groot_preprocessors  # noqa: PLC0415

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

        # Load pretrained model with explicit args
        self.model = GrootModel.from_pretrained(
            pretrained_model_name_or_path=self.hparams.base_model_path,
            n_action_steps=self.hparams.n_action_steps,
            use_bf16=self.hparams.use_bf16,
            tokenizer_assets_repo=self.hparams.tokenizer_assets_repo,
            attn_implementation=self.hparams.attn_implementation,
            tune_llm=self.hparams.tune_llm,
            tune_visual=self.hparams.tune_visual,
            tune_projector=self.hparams.tune_projector,
            tune_diffusion_model=self.hparams.tune_diffusion_model,
            chunk_size=self.hparams.chunk_size,
            max_action_dim=self.hparams.max_action_dim,
        )

        # Create first-party preprocessor/postprocessor
        self._preprocessor, self._postprocessor = make_groot_preprocessors(
            max_state_dim=self.hparams.max_state_dim,
            max_action_dim=self.hparams.max_action_dim,
            action_horizon=min(self.hparams.chunk_size, 16),  # GR00T max is 16
            embodiment_tag=self.hparams.embodiment_tag,
            env_action_dim=env_action_dim,
            stats=dataset_stats,
            eagle_processor_repo=self.hparams.tokenizer_assets_repo,
            device=str(self.device),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
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

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer from hyperparameters.

        Returns:
            Configured AdamW optimizer.

        Raises:
            RuntimeError: If model is not initialized.
        """
        if self.model is None:
            msg = "Model not initialized."
            raise RuntimeError(msg)

        return torch.optim.AdamW(
            self.model.get_optim_params(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

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

        # Get action
        action = self.model.select_action(batch_dict)

        # Postprocess
        return self._postprocessor(action)

    def reset(self) -> None:
        """Reset policy state for new episode."""
        if self.model is not None:
            self.model.reset()
