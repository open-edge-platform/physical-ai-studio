# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pi0.5 (Physical Intelligence) policy wrapper.

Pi0.5 is an improved version of Pi0 with adaptive RMS normalization conditioning
(adaRMS), which provides better action generation through scale conditioning.

Quick Start:
    Train Pi0.5 using the provided YAML config:

    ```bash
    # Install patched transformers (required until fix is merged to PyPI)
    pip install "transformers @ git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi"
    pip install getiaction lerobot

    # Train with default config (requires 40GB+ GPU - A100/H100)
    getiaction fit --config configs/lerobot/pi05.yaml
    ```

Differences from Pi0:
    | Feature | Pi0 | Pi0.5 |
    |---------|-----|-------|
    | Conditioning | Standard | adaRMS (scale conditioning) |
    | Action Quality | Good | Better generation quality |
    | Architecture | PaliGemma + Gemma expert | Same, with adaRMS layers |

Requirements:
    - GPU Memory: ~13GB inference, ~40GB+ training (A100/H100)
    - Dependencies: `pip install getiaction[pi]` installs transformers
    - Device: CUDA, XPU, CPU (uses eager attention, no Flash Attention required)
    - Export: Not supported (iterative denoising not traceable)
"""

from __future__ import annotations

from typing import Any

from lightning_utilities.core.imports import module_available

from getiaction.policies.lerobot.universal import LeRobotPolicy

LEROBOT_AVAILABLE = bool(module_available("lerobot"))


class Pi05(LeRobotPolicy):
    """Pi0.5 (Physical Intelligence) policy with adaRMS conditioning.

    PyTorch Lightning wrapper around LeRobot's Pi0.5 implementation, an improved
    version of Physical Intelligence's foundation model for robot manipulation.

    Pi0.5 extends Pi0 with adaptive RMS normalization (adaRMS) for better
    conditioning during action generation. It uses scale conditioning to improve
    the quality of generated actions compared to standard Pi0.

    Device Support:
        Pi0.5 supports CUDA, Intel XPU, and CPU because it uses PyTorch's standard
        "eager" attention implementation. Unlike Groot which requires Flash Attention,
        Pi0.5 can run on any PyTorch-supported accelerator.

    Examples:
        Train using the CLI with the provided config:

            ```bash
            getiaction fit --config configs/lerobot/pi05.yaml
            ```

        Create from dataset (eager initialization):

            >>> policy = Pi05.from_dataset(
            ...     "lerobot/aloha_sim_transfer_cube_human",
            ...     chunk_size=50,
            ...     gradient_checkpointing=True,
            ... )

        Train from scratch with Python API:

            >>> from getiaction.policies.lerobot import Pi05
            >>> from getiaction.data.lerobot import LeRobotDataModule
            >>> from getiaction.train import Trainer

            >>> policy = Pi05(
            ...     chunk_size=50,
            ...     num_inference_steps=10,
            ...     gradient_checkpointing=True,
            ... )

            >>> datamodule = LeRobotDataModule(
            ...     repo_id="lerobot/aloha_sim_transfer_cube_human",
            ...     train_batch_size=4,
            ...     data_format="lerobot",
            ... )

            >>> trainer = Trainer(max_epochs=100, precision="bf16-mixed")
            >>> trainer.fit(policy, datamodule)

        YAML configuration with LightningCLI:

            ```yaml
            model:
              class_path: getiaction.policies.lerobot.Pi05
              init_args:
                chunk_size: 50
                num_inference_steps: 10
                gradient_checkpointing: true
            ```

    Note:
        Pi0.5 cannot be exported to ONNX/OpenVINO due to iterative denoising loops.
        Use PyTorch Lightning directly for inference.

    Note:
        This class provides explicit typed parameters for IDE autocomplete.
        For dynamic policy selection, use LeRobotPolicy directly.

    See Also:
        - Pi0: Original Pi0 implementation
        - LeRobotPolicy: Universal wrapper for any LeRobot policy
        - LeRobotDataModule: For loading LeRobot datasets
        - configs/lerobot/pi05.yaml: Default training configuration
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        # Model architecture
        paligemma_variant: str = "gemma_2b",
        action_expert_variant: str = "gemma_300m",
        dtype: str = "float32",
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
        optimizer_lr: float = 2.5e-5,
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
            optimizer_lr: Learning rate for optimizer.
            optimizer_betas: Beta parameters for AdamW optimizer.
            optimizer_eps: Epsilon for AdamW optimizer.
            optimizer_weight_decay: Weight decay for optimizer.
            optimizer_grad_clip_norm: Gradient clipping norm.
            scheduler_warmup_steps: Number of warmup steps for scheduler.
            scheduler_decay_steps: Number of decay steps for scheduler.
            scheduler_decay_lr: Final learning rate after decay.
            **kwargs: Additional PI05Config parameters.

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

        super().__init__(
            policy_name="pi05",
            paligemma_variant=paligemma_variant,
            action_expert_variant=action_expert_variant,
            dtype=dtype,
            n_obs_steps=n_obs_steps,
            chunk_size=chunk_size,
            n_action_steps=n_action_steps,
            max_state_dim=max_state_dim,
            max_action_dim=max_action_dim,
            num_inference_steps=num_inference_steps,
            time_sampling_beta_alpha=time_sampling_beta_alpha,
            time_sampling_beta_beta=time_sampling_beta_beta,
            time_sampling_scale=time_sampling_scale,
            time_sampling_offset=time_sampling_offset,
            min_period=min_period,
            max_period=max_period,
            image_resolution=image_resolution,
            empty_cameras=empty_cameras,
            gradient_checkpointing=gradient_checkpointing,
            optimizer_lr=optimizer_lr,
            optimizer_betas=optimizer_betas,
            optimizer_eps=optimizer_eps,
            optimizer_weight_decay=optimizer_weight_decay,
            optimizer_grad_clip_norm=optimizer_grad_clip_norm,
            scheduler_warmup_steps=scheduler_warmup_steps,
            scheduler_decay_steps=scheduler_decay_steps,
            scheduler_decay_lr=scheduler_decay_lr,
            **kwargs,
        )
