# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Groot (GR00T-N1) policy wrapper for NVIDIA's foundation model.

This module provides a Lightning-compatible wrapper around LeRobot's Groot implementation,
NVIDIA's GR00T-N1.5-3B foundation model for generalist humanoid robots.

## Quick Start

Train Groot using the provided YAML config:

```bash
# Install dependencies
pip install getiaction[groot]

# Train with default config
getiaction fit --config configs/lerobot/groot.yaml

# Override parameters
getiaction fit --config configs/lerobot/groot.yaml \
    --model.learning_rate 1e-5 \
    --data.train_batch_size 2
```

## Requirements

- **GPU Memory**: 24GB+ VRAM recommended (RTX 3090/4090, A100, etc.)
- **Dependencies**: `pip install getiaction[groot]` installs flash-attn, transformers, peft
- **Device**: CUDA only (uses `flash-attn` package which has no XPU kernel)

Note: While PyTorch 2.9+ has FlexAttention for XPU, Groot requires the external
`flash-attn` package (Dao-AILab) which only supports CUDA GPUs.

## Fine-tuning Strategies

Groot supports selective fine-tuning to reduce memory usage:

1. **Projector + Diffusion** (default, ~20GB):
   `tune_projector=True, tune_diffusion_model=True`

2. **LoRA fine-tuning** (~16GB):
   `lora_rank=16, tune_llm=True`

3. **Full fine-tuning** (requires 80GB+ A100):
   `tune_llm=True, tune_visual=True, tune_projector=True, tune_diffusion_model=True`

## Example

```python
from getiaction.policies.lerobot import Groot
from getiaction.data.lerobot import LeRobotDataModule
from getiaction.train import Trainer

# Create policy
policy = Groot(
    chunk_size=50,
    tune_projector=True,
    tune_diffusion_model=True,
)

# Create datamodule
datamodule = LeRobotDataModule(
    repo_id="lerobot/aloha_sim_transfer_cube_human",
    train_batch_size=4,
    data_format="lerobot",
)

# Train
trainer = Trainer(max_epochs=100, precision="bf16-mixed")
trainer.fit(policy, datamodule)
```

## See Also

- `getiaction.data.lerobot.LeRobotDataModule`: Data loading
- `configs/lerobot/groot.yaml`: Default training configuration
- [LeRobot Groot](https://github.com/huggingface/lerobot): Upstream implementation
- [NVIDIA GR00T](https://developer.nvidia.com/isaac/groot): Model documentation
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from lightning_utilities.core.imports import module_available

from getiaction.data import Observation
from getiaction.data.lerobot import FormatConverter
from getiaction.data.lerobot.dataset import _LeRobotDatasetAdapter
from getiaction.policies.base import Policy
from getiaction.policies.lerobot.mixin import LeRobotFromConfig

if TYPE_CHECKING:
    from getiaction.gyms import Gym

if TYPE_CHECKING or module_available("lerobot"):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import dataset_to_policy_features
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.groot.configuration_groot import GrootConfig as _LeRobotGrootConfig
    from lerobot.policies.groot.modeling_groot import GrootPolicy as _LeRobotGrootPolicy

    LEROBOT_AVAILABLE = True
else:
    LeRobotDataset = None
    dataset_to_policy_features = None
    _LeRobotGrootConfig = None
    _LeRobotGrootPolicy = None
    make_pre_post_processors = None
    LEROBOT_AVAILABLE = False


class Groot(LeRobotFromConfig, Policy):
    """Groot (GR00T-N1) policy from LeRobot/NVIDIA.

    PyTorch Lightning wrapper around LeRobot's Groot implementation, NVIDIA's
    GR00T-N1.5-3B foundation model (2.4B parameters) for generalist humanoid robots.

    Groot is a vision-language-action (VLA) model that uses a diffusion-based
    action head on top of a multimodal foundation model (Eagle2). It supports
    fine-tuning with LoRA and selective component freezing.

    ## Quick Start

    Train using the CLI with the provided config:

    ```bash
    getiaction fit --config configs/lerobot/groot.yaml
    ```

    Or with custom parameters:

    ```bash
    getiaction fit --config configs/lerobot/groot.yaml \
        --model.learning_rate 1e-5 \
        --data.train_batch_size 2
    ```

    **Why XPU is not supported:**

    While PyTorch 2.9+ includes FlexAttention support for Intel XPU, Groot depends
    on the external `flash-attn` package (Dao-AILab), which only has CUDA kernels.
    The LeRobot Groot implementation requires this package and cannot use PyTorch's
    native attention mechanisms as a fallback. XPU support would require upstream
    changes to either `flash-attn` or LeRobot's Groot implementation.

    ## Export Limitations

    Groot cannot be exported to ONNX/OpenVINO/TorchExportIR because the `flash-attn`
    package uses custom CUDA kernels that cannot be traced by torch.export or ONNX.
    For deployment, use PyTorch Lightning directly for inference.

    ## Memory Requirements

    | Configuration | VRAM Required |
    |---------------|---------------|
    | Projector + Diffusion (default) | ~20GB |
    | LoRA fine-tuning | ~16GB |
    | Full fine-tuning | ~80GB (A100) |

    ## Examples

    **CLI Training (Recommended)**

    ```bash
    # Basic training
    getiaction fit --config configs/lerobot/groot.yaml

    # Custom dataset
    getiaction fit --config configs/lerobot/groot.yaml \
        --data.repo_id your-hf-username/your-dataset

    # LoRA fine-tuning (lower memory)
    getiaction fit --config configs/lerobot/groot.yaml \
        --model.lora_rank 16 --model.tune_llm true
    ```

    **Python API**

    ```python
    from getiaction.policies.lerobot import Groot
    from getiaction.data.lerobot import LeRobotDataModule
    from getiaction.train import Trainer

    policy = Groot(
        chunk_size=50,
        tune_projector=True,
        tune_diffusion_model=True,
    )

    datamodule = LeRobotDataModule(
        repo_id="lerobot/aloha_sim_transfer_cube_human",
        train_batch_size=4,
        data_format="lerobot",
    )

    trainer = Trainer(max_epochs=100, precision="bf16-mixed")
    trainer.fit(policy, datamodule)
    ```

    **Inference**

    ```python
    # Load trained checkpoint
    policy = Groot.load_from_checkpoint("path/to/checkpoint.ckpt")
    policy.eval()

    # Run inference
    observation = env.reset()
    action = policy.select_action(observation)
    env.step(action)
    ```

    Note:
        The policy is initialized lazily during setup(). Input/output features
        are extracted from the dataset and used to configure the model.

    ## See Also

    - `getiaction.data.lerobot.LeRobotDataModule`: Data loading
    - `getiaction.policies.lerobot.mixin.LeRobotFromConfig`: Config mixin
    - `configs/lerobot/groot.yaml`: Default training configuration
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        # Basic policy settings
        n_obs_steps: int = 1,
        chunk_size: int = 50,
        n_action_steps: int = 50,
        # Dimension settings
        max_state_dim: int = 64,
        max_action_dim: int = 32,
        # Image preprocessing
        image_size: tuple[int, int] = (224, 224),
        # Model path
        base_model_path: str = "nvidia/GR00T-N1.5-3B",
        tokenizer_assets_repo: str = "lerobot/eagle2hg-processor-groot-n1p5",
        embodiment_tag: str = "new_embodiment",
        # Fine-tuning control
        tune_llm: bool = False,
        tune_visual: bool = False,
        tune_projector: bool = True,
        tune_diffusion_model: bool = True,
        # LoRA parameters
        lora_rank: int = 0,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lora_full_model: bool = False,
        # Training parameters
        learning_rate: float = 1e-4,
        optimizer_betas: tuple[float, float] = (0.95, 0.999),
        optimizer_eps: float = 1e-8,
        optimizer_weight_decay: float = 1e-5,
        warmup_ratio: float = 0.05,
        use_bf16: bool = True,
        # Additional parameters via kwargs
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize Groot policy wrapper.

        The LeRobot policy is created lazily in setup() after the dataset is loaded.
        This is called automatically by Lightning before training begins.

        Args:
            n_obs_steps: Number of observation steps (typically 1 for Groot).
            chunk_size: Number of action predictions per forward pass.
            n_action_steps: Number of action steps to execute.
            max_state_dim: Maximum state dimension (shorter states zero-padded).
            max_action_dim: Maximum action dimension (shorter actions zero-padded).
            image_size: (H, W) image size for preprocessing.
            base_model_path: HuggingFace model ID or path to base Groot model.
            tokenizer_assets_repo: HF repo ID for Eagle tokenizer assets.
            embodiment_tag: Embodiment tag for training (e.g., 'new_embodiment', 'gr1').
            tune_llm: Whether to fine-tune the LLM backbone.
            tune_visual: Whether to fine-tune the vision tower.
            tune_projector: Whether to fine-tune the projector.
            tune_diffusion_model: Whether to fine-tune the diffusion model.
            lora_rank: LoRA rank (0 disables LoRA).
            lora_alpha: LoRA alpha value.
            lora_dropout: LoRA dropout rate.
            lora_full_model: Whether to apply LoRA to full model.
            learning_rate: Learning rate for optimizer.
            optimizer_betas: Beta parameters for AdamW optimizer.
            optimizer_eps: Epsilon for AdamW optimizer.
            optimizer_weight_decay: Weight decay for optimizer.
            warmup_ratio: Warmup ratio for learning rate scheduler.
            use_bf16: Whether to use bfloat16 precision.
            **kwargs: Additional GrootConfig parameters.

        Raises:
            ImportError: If LeRobot or Groot dependencies are not installed.
        """
        if not LEROBOT_AVAILABLE:
            msg = (
                "Groot requires LeRobot framework with Groot support.\n\n"
                "Install with:\n"
                "    pip install lerobot[groot]\n\n"
                "Or install getiaction with Groot support:\n"
                "    pip install getiaction[groot]"
            )
            raise ImportError(msg)

        super().__init__()

        # Build config dict from explicit args
        self._config_object = None
        self._config_kwargs = {
            "n_obs_steps": n_obs_steps,
            "chunk_size": chunk_size,
            "n_action_steps": n_action_steps,
            "max_state_dim": max_state_dim,
            "max_action_dim": max_action_dim,
            "image_size": image_size,
            "base_model_path": base_model_path,
            "tokenizer_assets_repo": tokenizer_assets_repo,
            "embodiment_tag": embodiment_tag,
            "tune_llm": tune_llm,
            "tune_visual": tune_visual,
            "tune_projector": tune_projector,
            "tune_diffusion_model": tune_diffusion_model,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "lora_full_model": lora_full_model,
            "optimizer_lr": learning_rate,  # Map to LeRobot's parameter name
            "optimizer_betas": optimizer_betas,
            "optimizer_eps": optimizer_eps,
            "optimizer_weight_decay": optimizer_weight_decay,
            "warmup_ratio": warmup_ratio,
            "use_bf16": use_bf16,
            **kwargs,
        }

        self.learning_rate = learning_rate
        self._framework = "lerobot"

        # Policy will be initialized in setup()
        self._lerobot_policy: _LeRobotGrootPolicy

        self.save_hyperparameters()

    @property
    def lerobot_policy(self) -> _LeRobotGrootPolicy:
        """Get the initialized LeRobot policy.

        Returns:
            The initialized LeRobot Groot policy.

        Raises:
            RuntimeError: If the policy hasn't been initialized yet.
        """
        if not hasattr(self, "_lerobot_policy") or self._lerobot_policy is None:
            msg = "Policy not initialized. Call setup() first."
            raise RuntimeError(msg)
        return self._lerobot_policy

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer using LeRobot's parameters.

        Returns:
            The configured optimizer instance.
        """
        return torch.optim.AdamW(
            self.lerobot_policy.get_optim_params(),
            lr=self.learning_rate,
            betas=self._config_kwargs.get("optimizer_betas", (0.95, 0.999)),
            eps=self._config_kwargs.get("optimizer_eps", 1e-8),
            weight_decay=self._config_kwargs.get("optimizer_weight_decay", 1e-5),
        )

    def forward(self, batch: Observation | dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass for inference.

        In evaluation mode, selects actions from observations.
        This method is required by the base Policy class.

        Note:
            Groot does not support export to ONNX/OpenVINO due to Flash Attention.
            Use select_action() directly for inference.

        Args:
            batch: Input observations.

        Returns:
            Action tensor from select_action.
        """
        return self.select_action(batch)

    def setup(self, stage: str) -> None:
        """Set up the policy from datamodule if not already initialized.

        This method is called by Lightning before fit/validate/test/predict.
        It extracts features from the datamodule's training dataset and
        initializes the policy if it wasn't already created in __init__.

        Args:
            stage: The stage of training ('fit', 'validate', 'test', or 'predict')

        Raises:
            TypeError: If the train_dataset is not a LeRobot dataset.
        """
        del stage  # Unused argument

        if hasattr(self, "_lerobot_policy") and self._lerobot_policy is not None:
            return  # Already initialized

        datamodule = self.trainer.datamodule  # type: ignore[attr-defined]
        train_dataset = datamodule.train_dataset

        # Get the underlying LeRobot dataset - handle both data formats
        if isinstance(train_dataset, _LeRobotDatasetAdapter):
            # Wrapped in adapter for getiaction format conversion
            lerobot_dataset = train_dataset._lerobot_dataset  # noqa: SLF001
        elif LeRobotDataset is not None and isinstance(train_dataset, LeRobotDataset):
            # Dataset is raw LeRobotDataset (data_format="lerobot")
            lerobot_dataset = train_dataset
        else:
            msg = (
                f"Expected train_dataset to be _LeRobotDatasetAdapter or LeRobotDataset, "
                f"got {type(train_dataset)}. Use LeRobotDataModule with appropriate data_format."
            )
            raise TypeError(msg)

        features = dataset_to_policy_features(lerobot_dataset.meta.features)
        dataset_stats = lerobot_dataset.meta.stats

        # Create or update LeRobot Groot configuration based on what user provided
        if self._config_object is not None:
            # User provided a full config object - update input/output features
            lerobot_config = self._config_object
            lerobot_config.input_features = features
            lerobot_config.output_features = features
        else:
            # User provided dict or explicit args - create config
            lerobot_config = _LeRobotGrootConfig(  # type: ignore[misc]
                input_features=features,
                output_features=features,
                **self._config_kwargs,  # type: ignore[arg-type]
            )

        # Initialize the policy
        policy = _LeRobotGrootPolicy(lerobot_config)
        self.add_module("_lerobot_policy", policy)

        # Create preprocessor/postprocessor for normalization
        self._preprocessor, self._postprocessor = make_pre_post_processors(
            lerobot_config,
            dataset_stats=dataset_stats,
        )

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step uses LeRobot's loss computation.

        Args:
            batch: Input batch in LeRobot dict format with keys like
                "observation.state", "observation.images.*", "action".
            batch_idx: Index of the batch.

        Returns:
            The total loss for the batch.
        """
        del batch_idx  # Unused argument

        # Apply preprocessing (normalization, image transforms, etc.)
        batch = self._preprocessor(batch)

        # Run forward through LeRobot policy
        total_loss, loss_dict = self.lerobot_policy(batch)
        # Log individual loss components (skip 'loss' since we log total_loss as train/loss)
        for key, value in loss_dict.items():
            if key != "loss":
                self.log(f"train/{key}", value, prog_bar=False)
        self.log("train/loss", total_loss, prog_bar=True)
        return total_loss

    def validation_step(self, batch: Gym, batch_idx: int) -> dict[str, float]:
        """Validation step.

        Runs gym-based validation by executing rollouts in the environment.
        The DataModule's val_dataloader returns Gym environment instances directly.

        Args:
            batch: Gym environment to evaluate.
            batch_idx: Index of the batch.

        Returns:
            Metrics dict from gym rollout evaluation.
        """
        return self.evaluate_gym(batch, batch_idx, stage="val")

    def test_step(self, batch: Gym, batch_idx: int) -> dict[str, float]:
        """Test step.

        Runs gym-based testing by executing rollouts in the environment.
        The DataModule's test_dataloader returns Gym environment instances directly.

        Args:
            batch: Gym environment to evaluate.
            batch_idx: Index of the batch.

        Returns:
            Metrics dict from gym rollout evaluation.
        """
        return self.evaluate_gym(batch, batch_idx, stage="test")

    def select_action(self, batch: Observation | dict[str, torch.Tensor]) -> torch.Tensor:
        """Select action (inference mode) through LeRobot.

        Converts the Observation to LeRobot dict format, applies preprocessing,
        gets action prediction, and applies postprocessing (denormalization).

        Args:
            batch: Input batch of observations (raw, from gym).

        Returns:
            The selected action tensor (denormalized).
        """
        # Convert to LeRobot format if needed
        batch_dict = FormatConverter.to_lerobot_dict(batch) if isinstance(batch, Observation) else batch

        # Apply preprocessing
        batch_dict = self._preprocessor(batch_dict)

        # Get action from policy
        action = self.lerobot_policy.select_action(batch_dict)

        # Apply postprocessing
        return self._postprocessor(action)

    def reset(self) -> None:
        """Reset the policy state."""
        self.lerobot_policy.reset()
