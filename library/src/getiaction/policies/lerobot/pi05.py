# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pi0.5 (Physical Intelligence) policy wrapper.

Pi0.5 is an improved version of Pi0 with adaptive RMS normalization conditioning
(adaRMS), which provides better action generation through scale conditioning.

## Quick Start

Train Pi0.5 using the provided YAML config:

```bash
# Install patched transformers (required until fix is merged to PyPI)
pip install "transformers @ git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi"
pip install getiaction lerobot

# Train with default config (requires 40GB+ GPU - A100/H100)
getiaction fit --config configs/lerobot/pi05.yaml

# Override parameters
getiaction fit --config configs/lerobot/pi05.yaml \
    --model.learning_rate 1e-5 \
    --data.train_batch_size 2
```

## Differences from Pi0

| Feature | Pi0 | Pi0.5 |
|---------|-----|-------|
| Conditioning | Standard | adaRMS (scale conditioning) |
| Action Quality | Good | Better generation quality |
| Architecture | PaliGemma + Gemma expert | Same, with adaRMS layers |

## Requirements

### Memory Requirements

| Mode          | VRAM    | Hardware              |
|---------------|---------|---------------------- |
| Inference     | ~13GB   | RTX 3090/4090         |
| Training      | ~40GB+  | A100, H100            |

### Hardware Support

| Device | Supported | Notes                          |
|--------|-----------|--------------------------------|
| CUDA   | ✓         | Full support                   |
| XPU    | ✓         | Intel Arc/Data Center GPUs     |
| CPU    | ✓         | Slow, for testing only         |
| Export | ✗         | Iterative denoising not traceable |

### Dependencies

Requires patched transformers (until fix is merged to PyPI):

```bash
pip install "transformers @ git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi"
```

## XPU Support

Pi0.5 fully supports Intel XPU (Data Center GPUs, Arc GPUs) because it uses
PyTorch's standard "eager" attention implementation instead of Flash Attention.
No special PyTorch version is required - any PyTorch with XPU support will work.

## Export Limitations

Pi0.5 **cannot be exported to ONNX/OpenVINO/TorchExportIR** because:
- Iterative denoising requires while loops (not traceable)
- KV-caching creates dynamic tensor shapes
- Complex control flow in sample_actions method

For deployment, use PyTorch Lightning directly for inference or consider
distillation to a simpler model architecture.

## Example

```python
from getiaction.policies.lerobot import Pi05
from getiaction.data.lerobot import LeRobotDataModule
from getiaction.train import Trainer

# Create policy
policy = Pi05(
    chunk_size=50,
    num_inference_steps=10,
    gradient_checkpointing=True,  # Recommended for memory efficiency
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

- `getiaction.policies.lerobot.Pi0`: Original Pi0 implementation
- `getiaction.data.lerobot.LeRobotDataModule`: Data loading
- `configs/lerobot/pi05.yaml`: Default training configuration
- [LeRobot Pi0.5](https://github.com/huggingface/lerobot): Upstream implementation
- [Physical Intelligence](https://physicalintelligence.company/): Model documentation
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
    from lerobot.policies.pi0.configuration_pi0 import PI0Config as _LeRobotPI0Config
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy as _LeRobotPI0Policy

    LEROBOT_AVAILABLE = True
else:
    LeRobotDataset = None
    dataset_to_policy_features = None
    _LeRobotPI0Config = None
    _LeRobotPI0Policy = None
    make_pre_post_processors = None
    LEROBOT_AVAILABLE = False


class Pi05(LeRobotFromConfig, Policy):
    """Pi0.5 (Physical Intelligence) policy with adaRMS conditioning.

    PyTorch Lightning wrapper around LeRobot's Pi0.5 implementation, an improved
    version of Physical Intelligence's foundation model for robot manipulation.

    Pi0.5 extends Pi0 with adaptive RMS normalization (adaRMS) for better
    conditioning during action generation. It uses scale conditioning to improve
    the quality of generated actions compared to standard Pi0.

    ## Device Support

    Pi0.5 supports CUDA, Intel XPU, and CPU because it uses PyTorch's standard
    "eager" attention implementation. Unlike Groot which requires Flash Attention,
    Pi0.5 can run on any PyTorch-supported accelerator.

    ## Quick Start

    Train using the CLI with the provided config:

    ```bash
    getiaction fit --config configs/lerobot/pi05.yaml
    ```

    Or with custom parameters:

    ```bash
    getiaction fit --config configs/lerobot/pi05.yaml \
        --model.learning_rate 1e-5 \
        --model.gradient_checkpointing true
    ```

    ## Export Limitations

    Pi0.5 cannot be exported to ONNX/OpenVINO/TorchExportIR because:
    - Iterative denoising uses while loops (not traceable)
    - KV-caching creates dynamic tensor shapes
    - Complex control flow in the sample_actions method

    For deployment, use PyTorch Lightning directly or consider model distillation.

    ## Memory Requirements

    | Mode | VRAM Required | Notes |
    |------|---------------|-------|
    | Inference | ~13GB | RTX 3090/4090 works |
    | Training (bf16) | ~40GB+ | A100/H100 required |
    | Training (fp32) | ~80GB+ | Not recommended |

    ## Examples

    **CLI Training (Recommended)**

    ```bash
    # Basic training
    getiaction fit --config configs/lerobot/pi05.yaml

    # Custom dataset
    getiaction fit --config configs/lerobot/pi05.yaml \
        --data.repo_id your-hf-username/your-dataset

    # Memory-efficient training
    getiaction fit --config configs/lerobot/pi05.yaml \
        --model.gradient_checkpointing true
    ```

    **Python API**

    ```python
    from getiaction.policies.lerobot import Pi05
    from getiaction.data.lerobot import LeRobotDataModule
    from getiaction.train import Trainer

    policy = Pi05(
        chunk_size=50,
        num_inference_steps=10,
        gradient_checkpointing=True,
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
    policy = Pi05.load_from_checkpoint("path/to/checkpoint.ckpt")
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

    - `getiaction.policies.lerobot.Pi0`: Original Pi0 implementation
    - `getiaction.data.lerobot.LeRobotDataModule`: Data loading
    - `getiaction.policies.lerobot.mixin.LeRobotFromConfig`: Config mixin
    - `configs/lerobot/pi05.yaml`: Default training configuration
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        # Model architecture - Pi0.5 specific
        paligemma_variant: str = "gemma_2b",
        action_expert_variant: str = "gemma_300m",
        dtype: str = "float32",  # "bfloat16" or "float32"
        # Basic policy settings
        n_obs_steps: int = 1,
        chunk_size: int = 50,
        n_action_steps: int = 50,
        # Dimension settings
        max_state_dim: int = 32,
        max_action_dim: int = 32,
        # Flow matching parameters
        num_inference_steps: int = 10,
        time_sampling_beta_alpha: float = 1.5,
        time_sampling_beta_beta: float = 1.0,
        time_sampling_scale: float = 0.999,
        time_sampling_offset: float = 0.001,
        min_period: float = 4e-3,
        max_period: float = 4.0,
        # Image settings
        image_resolution: tuple[int, int] = (224, 224),
        empty_cameras: int = 0,
        # Training parameters
        gradient_checkpointing: bool = False,
        learning_rate: float = 2.5e-5,
        optimizer_betas: tuple[float, float] = (0.9, 0.95),
        optimizer_eps: float = 1e-8,
        optimizer_weight_decay: float = 0.01,
        optimizer_grad_clip_norm: float = 1.0,
        # Scheduler parameters
        scheduler_warmup_steps: int = 1000,
        scheduler_decay_steps: int = 30000,
        scheduler_decay_lr: float = 2.5e-6,
        # Additional parameters via kwargs
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize Pi0.5 policy wrapper.

        The LeRobot policy is created lazily in setup() after the dataset is loaded.
        This is called automatically by Lightning before training begins.

        Args:
            paligemma_variant: PaliGemma model variant ("gemma_2b" or "gemma_300m").
            action_expert_variant: Action expert variant ("gemma_2b" or "gemma_300m").
            dtype: Model dtype ("bfloat16" or "float32").
            n_obs_steps: Number of observation steps (typically 1 for Pi0.5).
            chunk_size: Number of action predictions per forward pass.
            n_action_steps: Number of action steps to execute.
            max_state_dim: Maximum state dimension (shorter states zero-padded).
            max_action_dim: Maximum action dimension (shorter actions zero-padded).
            num_inference_steps: Number of denoising steps during inference.
            time_sampling_beta_alpha: Beta distribution alpha for time sampling.
            time_sampling_beta_beta: Beta distribution beta for time sampling.
            time_sampling_scale: Scale for time sampling.
            time_sampling_offset: Offset for time sampling.
            min_period: Minimum period for sinusoidal positional encoding.
            max_period: Maximum period for sinusoidal positional encoding.
            image_resolution: (H, W) image size for preprocessing.
            empty_cameras: Number of empty camera slots to add.
            gradient_checkpointing: Enable gradient checkpointing for memory efficiency.
            learning_rate: Learning rate for optimizer.
            optimizer_betas: Beta parameters for AdamW optimizer.
            optimizer_eps: Epsilon for AdamW optimizer.
            optimizer_weight_decay: Weight decay for optimizer.
            optimizer_grad_clip_norm: Gradient clipping norm.
            scheduler_warmup_steps: Number of warmup steps for scheduler.
            scheduler_decay_steps: Number of decay steps for scheduler.
            scheduler_decay_lr: Final learning rate after decay.
            **kwargs: Additional PI0Config parameters.

        Raises:
            ImportError: If LeRobot is not installed.
        """
        if not LEROBOT_AVAILABLE:
            msg = (
                "Pi05 requires LeRobot framework.\n\n"
                "Install with:\n"
                "    pip install lerobot\n\n"
                "Note: Pi0.5 uses standard PyTorch attention (no Flash Attention required),\n"
                "so it works on CUDA, XPU, and CPU."
            )
            raise ImportError(msg)

        super().__init__()

        # Build config dict from explicit args
        self._config_object = None
        self._config_kwargs = {
            "paligemma_variant": paligemma_variant,
            "action_expert_variant": action_expert_variant,
            "dtype": dtype,
            "n_obs_steps": n_obs_steps,
            "chunk_size": chunk_size,
            "n_action_steps": n_action_steps,
            "max_state_dim": max_state_dim,
            "max_action_dim": max_action_dim,
            "num_inference_steps": num_inference_steps,
            "time_sampling_beta_alpha": time_sampling_beta_alpha,
            "time_sampling_beta_beta": time_sampling_beta_beta,
            "time_sampling_scale": time_sampling_scale,
            "time_sampling_offset": time_sampling_offset,
            "min_period": min_period,
            "max_period": max_period,
            "image_resolution": image_resolution,
            "empty_cameras": empty_cameras,
            "gradient_checkpointing": gradient_checkpointing,
            "optimizer_lr": learning_rate,  # Map to LeRobot's parameter name
            "optimizer_betas": optimizer_betas,
            "optimizer_eps": optimizer_eps,
            "optimizer_weight_decay": optimizer_weight_decay,
            "optimizer_grad_clip_norm": optimizer_grad_clip_norm,
            "scheduler_warmup_steps": scheduler_warmup_steps,
            "scheduler_decay_steps": scheduler_decay_steps,
            "scheduler_decay_lr": scheduler_decay_lr,
            # Pi0.5 specific: enable adaRMS conditioning
            "use_ada_rms_conditioning": True,
            **kwargs,
        }

        self.learning_rate = learning_rate
        self._gradient_checkpointing = gradient_checkpointing
        self._framework = "lerobot"

        # Policy will be initialized in setup()
        self._lerobot_policy: _LeRobotPI0Policy

        self.save_hyperparameters()

    @property
    def lerobot_policy(self) -> _LeRobotPI0Policy:
        """Get the initialized LeRobot policy.

        Returns:
            The initialized LeRobot Pi0 policy (with Pi0.5 configuration).

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
            betas=self._config_kwargs.get("optimizer_betas", (0.9, 0.95)),
            eps=self._config_kwargs.get("optimizer_eps", 1e-8),
            weight_decay=self._config_kwargs.get("optimizer_weight_decay", 0.01),
        )

    def forward(self, batch: Observation | dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass for inference.

        In evaluation mode, selects actions from observations.
        This method is required by the base Policy class.

        Note:
            Pi0.5 does NOT support export to ONNX/OpenVINO due to:
            - Iterative denoising with while loops
            - Dynamic KV-caching
            - Complex control flow

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

        # Create or update LeRobot Pi0 configuration based on what user provided
        if self._config_object is not None:
            # User provided a full config object - update input/output features
            lerobot_config = self._config_object
            lerobot_config.input_features = features
            lerobot_config.output_features = features
        else:
            # User provided dict or explicit args - create config
            lerobot_config = _LeRobotPI0Config(  # type: ignore[misc]
                input_features=features,
                output_features=features,
                **self._config_kwargs,  # type: ignore[arg-type]
            )

        # Initialize the policy
        policy = _LeRobotPI0Policy(lerobot_config)
        self.add_module("_lerobot_policy", policy)

        # Enable gradient checkpointing if requested
        if self._gradient_checkpointing:
            self.lerobot_policy.model.gradient_checkpointing_enable()

        # Create preprocessor/postprocessor for normalization
        self._preprocessor, self._postprocessor = make_pre_post_processors(
            lerobot_config,
            dataset_stats=dataset_stats,  # type: ignore[arg-type]
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

        # Log individual loss components (skip non-scalar values like loss_per_dim)
        for key, value in loss_dict.items():
            if key == "loss":
                continue  # Skip main loss, logged separately
            if isinstance(value, (int, float)):
                self.log(f"train/{key}", value, prog_bar=False)
            # Skip lists (e.g., loss_per_dim) - not loggable directly
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
