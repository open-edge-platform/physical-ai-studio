# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Configuration for Groot (GR00T-N1.5) policy.

This module provides the dataclass configuration for the Groot policy,
which wraps NVIDIA's GR00T-N1.5 foundation model for humanoid robotics.

The configuration inherits from the base `Config` class, enabling:
- YAML serialization/deserialization
- Checkpoint compatibility with weights_only=True
- jsonargparse integration for CLI usage

For CLI usage, use the YAML config in `configs/groot/groot.yaml`:

    physicalai fit --config configs/physicalai/groot.yaml

Example (API):
    >>> from physicalai.policies.groot import GrootConfig
    >>> config = GrootConfig(
    ...     chunk_size=50,
    ...     tune_projector=True,
    ...     tune_diffusion_model=True,
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass, field

from physicalai.config import Config


@dataclass
class GrootConfig(Config):
    """Configuration for Groot (GR00T-N1.5) policy.

    Groot is NVIDIA's foundation model for humanoid robots, using Eagle2 VLM
    backbone and flow matching for action generation.

    Attributes:
        chunk_size: Number of action predictions per forward pass (action_horizon).
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
        revision: Git revision (branch, tag, or commit hash) to download from.

    Examples:
        Basic config:

        >>> config = GrootConfig(chunk_size=50, learning_rate=1e-4)
        >>> print(config.tune_projector)
        True

        Full fine-tuning:

        >>> config = GrootConfig(
        ...     tune_llm=True,
        ...     tune_visual=True,
        ...     tune_projector=True,
        ...     tune_diffusion_model=True,
        ... )
    """

    # Model architecture
    chunk_size: int = 50  # action_horizon
    n_action_steps: int = 50
    max_state_dim: int = 64
    max_action_dim: int = 32

    # Model source
    base_model_path: str = "nvidia/GR00T-N1.5-3B"
    tokenizer_assets_repo: str = "lerobot/eagle2hg-processor-groot-n1p5"
    embodiment_tag: str = "new_embodiment"

    # Attention implementation
    attn_implementation: str = "sdpa"

    # Fine-tuning control
    tune_llm: bool = False
    tune_visual: bool = False
    tune_projector: bool = True
    tune_diffusion_model: bool = True

    # Optimizer/training hyperparameters
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.05  # Warmup ratio (0.0-1.0) of total training steps
    grad_clip_norm: float = 1.0  # Gradient clipping norm (0.0 = disabled)

    # Precision
    use_bf16: bool = True

    # HuggingFace args
    revision: str | None = field(default=None)

    @property
    def action_horizon(self) -> int:
        """Alias for chunk_size (action horizon)."""
        return self.chunk_size
