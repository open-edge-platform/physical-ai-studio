# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Groot (GR00T-N1) policy wrapper for NVIDIA's foundation model.

This module provides a Lightning-compatible wrapper around LeRobot's Groot implementation,
NVIDIA's GR00T-N1.5-3B foundation model for generalist humanoid robots.

Quick Start:
    Train Groot using the provided YAML config:

    ```bash
    # Install dependencies
    pip install getiaction[groot]

    # Train with default config
    getiaction fit --config configs/lerobot/groot.yaml
    ```

Requirements:
    - GPU Memory: 24GB+ VRAM recommended (RTX 3090/4090, A100, etc.)
    - Dependencies: `pip install getiaction[groot]` installs flash-attn, transformers, peft
    - Device: CUDA only (uses `flash-attn` package which has no XPU kernel)

Fine-tuning Strategies:
    1. Projector + Diffusion (default, ~20GB):
       `tune_projector=True, tune_diffusion_model=True`

    2. LoRA fine-tuning (~16GB):
       `lora_rank=16, tune_llm=True`

    3. Full fine-tuning (requires 80GB+ A100):
       `tune_llm=True, tune_visual=True, tune_projector=True, tune_diffusion_model=True`
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lightning_utilities.core.imports import module_available

from getiaction.policies.lerobot.universal import LeRobotPolicy

if TYPE_CHECKING:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

if module_available("lerobot"):
    LEROBOT_AVAILABLE = True
else:
    LEROBOT_AVAILABLE = False


class Groot(LeRobotPolicy):
    """Groot (GR00T-N1) policy from LeRobot/NVIDIA.

    PyTorch Lightning wrapper around LeRobot's Groot implementation, NVIDIA's
    GR00T-N1.5-3B foundation model (2.4B parameters) for generalist humanoid robots.

    Groot is a vision-language-action (VLA) model that uses a diffusion-based
    action head on top of a multimodal foundation model (Eagle2). It supports
    fine-tuning with LoRA and selective component freezing.

    Examples:
        Train using the CLI with the provided config:

            ```bash
            getiaction fit --config configs/lerobot/groot.yaml
            ```

        Create from dataset (eager initialization):

            >>> policy = Groot.from_dataset(
            ...     "lerobot/aloha_sim_transfer_cube_human",
            ...     chunk_size=50,
            ...     tune_projector=True,
            ... )

        Train from scratch with Python API:

            >>> from getiaction.policies.lerobot import Groot
            >>> from getiaction.data.lerobot import LeRobotDataModule
            >>> from getiaction.train import Trainer

            >>> policy = Groot(
            ...     chunk_size=50,
            ...     tune_projector=True,
            ...     tune_diffusion_model=True,
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
              class_path: getiaction.policies.lerobot.Groot
              init_args:
                chunk_size: 50
                tune_projector: true
                tune_diffusion_model: true
            ```

    Note:
        Groot cannot be exported to ONNX/OpenVINO due to Flash Attention.
        Use PyTorch Lightning directly for inference.

    Note:
        This class provides explicit typed parameters for IDE autocomplete.
        For dynamic policy selection, use LeRobotPolicy directly.

    See Also:
        - LeRobotPolicy: Universal wrapper for any LeRobot policy
        - LeRobotDataModule: For loading LeRobot datasets
        - configs/lerobot/groot.yaml: Default training configuration
    """

    def __init__(
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
        optimizer_lr: float = 1e-4,
        optimizer_betas: tuple[float, float] = (0.95, 0.999),
        optimizer_eps: float = 1e-8,
        optimizer_weight_decay: float = 1e-5,
        warmup_ratio: float = 0.05,
        use_bf16: bool = True,
        # Additional parameters via kwargs
        **kwargs: Any,
    ) -> None:
        """Initialize Groot policy wrapper.

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
            optimizer_lr: Learning rate for optimizer.
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

        super().__init__(
            policy_name="groot",
            n_obs_steps=n_obs_steps,
            chunk_size=chunk_size,
            n_action_steps=n_action_steps,
            max_state_dim=max_state_dim,
            max_action_dim=max_action_dim,
            image_size=image_size,
            base_model_path=base_model_path,
            tokenizer_assets_repo=tokenizer_assets_repo,
            embodiment_tag=embodiment_tag,
            tune_llm=tune_llm,
            tune_visual=tune_visual,
            tune_projector=tune_projector,
            tune_diffusion_model=tune_diffusion_model,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_full_model=lora_full_model,
            optimizer_lr=optimizer_lr,
            optimizer_betas=optimizer_betas,
            optimizer_eps=optimizer_eps,
            optimizer_weight_decay=optimizer_weight_decay,
            warmup_ratio=warmup_ratio,
            use_bf16=use_bf16,
            **kwargs,
        )

    @classmethod
    def from_dataset(
        cls,
        dataset: LeRobotDataset | str,
        **kwargs: Any,
    ) -> "Groot":
        """Create Groot policy with eager initialization from a dataset.

        Args:
            dataset: Either a LeRobotDataset instance or a HuggingFace Hub repo ID.
            **kwargs: Groot configuration parameters.

        Returns:
            Fully initialized Groot policy ready for inference.

        Examples:
            >>> policy = Groot.from_dataset(
            ...     "lerobot/aloha_sim_transfer_cube_human",
            ...     chunk_size=50,
            ... )
        """
        return LeRobotPolicy.from_dataset("groot", dataset, **kwargs)  # type: ignore[return-value]
