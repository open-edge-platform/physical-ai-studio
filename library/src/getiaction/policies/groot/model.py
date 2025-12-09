# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Groot Model - Pure PyTorch nn.Module wrapping NVIDIA's GR00T-N1.5 foundation model.

This module provides a clean PyTorch interface to the GR00T-N1.5-3B model,
with support for PyTorch native SDPA attention (no Flash Attention CUDA dependency).

The implementation uses first-party components from the `components/` subpackage,
avoiding any LeRobot dependencies. Only HuggingFace transformers and diffusers
are used as external dependencies.

Supported Devices:
    - CUDA: NVIDIA GPUs with CUDA support
    - XPU: Intel GPUs (Arc, Data Center Max) with PyTorch XPU support
    - CPU: Fallback for development/debugging

"""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from .components import EagleBackbone, FlowMatchingActionHead

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)


@dataclass
class GrootConfig:
    """Configuration for GrootModel.

    This dataclass can be used with `GrootModel.from_config()` for convenient
    configuration-based instantiation, or values can be passed directly as
    explicit arguments to `GrootModel.from_pretrained()`.

    Attributes:
        pretrained_model_name_or_path: HuggingFace model ID or local path.
        n_action_steps: Number of action steps to execute per chunk.
        use_bf16: Whether to use bfloat16 precision for compute.
        tokenizer_assets_repo: HF repo ID for Eagle tokenizer assets.
        attn_implementation: Attention implementation ('sdpa', 'flash_attention_2', 'eager').
        tune_llm: Whether to fine-tune the LLM backbone.
        tune_visual: Whether to fine-tune the vision tower.
        tune_projector: Whether to fine-tune the projector.
        tune_diffusion_model: Whether to fine-tune the diffusion model.
        chunk_size: Fallback action horizon if not in checkpoint.
        max_action_dim: Fallback action dimension if not in checkpoint.

    Examples:
        Using with from_config:

        >>> config = GrootConfig(attn_implementation="sdpa", tune_projector=True)
        >>> model = GrootModel.from_config(config)

        Loading from YAML:

        >>> model = GrootModel.from_yaml("groot_config.yaml")

        Using explicit args (equivalent):

        >>> model = GrootModel.from_pretrained(
        ...     attn_implementation="sdpa",
        ...     tune_projector=True,
        ... )
    """

    # Model path
    pretrained_model_name_or_path: str = field(default="nvidia/GR00T-N1.5-3B")

    # Model args
    n_action_steps: int = field(default=50)
    use_bf16: bool = field(default=True)

    # Backbone args
    tokenizer_assets_repo: str = field(default="nvidia/Eagle2-2B")
    attn_implementation: str = field(default="sdpa")
    tune_llm: bool = field(default=False)
    tune_visual: bool = field(default=False)

    # Action head args
    tune_projector: bool = field(default=True)
    tune_diffusion_model: bool = field(default=True)
    chunk_size: int = field(default=50)
    max_action_dim: int = field(default=32)


def _import_snapshot_download() -> tuple[Any, ...]:
    """Import snapshot_download and error classes from huggingface_hub.

    Returns:
        Tuple of (snapshot_download, HfHubHTTPError, RepositoryNotFoundError).

    Raises:
        ImportError: If huggingface_hub is not installed.
    """
    try:
        from huggingface_hub import snapshot_download  # noqa: PLC0415
        from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError  # noqa: PLC0415
    except ImportError as e:
        msg = "Groot requires huggingface_hub. Install with: pip install huggingface_hub"
        raise ImportError(msg) from e

    return snapshot_download, HfHubHTTPError, RepositoryNotFoundError


DEFAULT_BASE_MODEL_PATH = "nvidia/GR00T-N1.5-3B"
DEFAULT_TOKENIZER_ASSETS_REPO = "nvidia/Eagle2-2B"


class GrootModel(nn.Module):
    """GR00T-N1.5 Vision-Language-Action Model.

    Pure PyTorch nn.Module wrapping NVIDIA's GR00T-N1.5-3B foundation model.
    Uses PyTorch native SDPA attention by default for wider device support
    (CUDA, XPU) without requiring the Flash Attention CUDA package.

    This model can be used standalone for inference/export, or wrapped
    in a Lightning policy for training.

    All constructor parameters are explicit for clarity and testability.

    Args:
        n_action_steps: Number of action steps to execute per chunk.
        use_bf16: Whether to use bfloat16 precision for compute.
        tokenizer_assets_repo: HF repo ID for Eagle tokenizer assets.
        attn_implementation: Attention implementation ('sdpa', 'flash_attention_2', 'eager').
        tune_llm: Whether to fine-tune the LLM backbone.
        tune_visual: Whether to fine-tune the vision tower.
        tune_projector: Whether to fine-tune the projector.
        tune_diffusion_model: Whether to fine-tune the diffusion model.
        chunk_size: Action horizon for action head.
        max_action_dim: Maximum action dimension.

    Examples:
        Direct initialization (no pretrained weights):

        >>> model = GrootModel(attn_implementation="sdpa")
        >>> model.train()

        With pretrained weights:

        >>> model = GrootModel.from_pretrained("nvidia/GR00T-N1.5-3B")
        >>> model.eval()
        >>> actions = model.get_action(batch)

        From config dataclass:

        >>> config = GrootConfig(attn_implementation="sdpa", tune_projector=True)
        >>> model = GrootModel.from_config(config)


    Note:
        Use `from_pretrained()` to load NVIDIA's pretrained weights.
        Direct `__init__` creates a model with random weights.
    """

    def __init__(
        self,
        n_action_steps: int = 50,
        *,
        use_bf16: bool = True,
        # Backbone args
        tokenizer_assets_repo: str = DEFAULT_TOKENIZER_ASSETS_REPO,
        attn_implementation: str = "sdpa",
        tune_llm: bool = False,
        tune_visual: bool = False,
        # Action head args
        tune_projector: bool = True,
        tune_diffusion_model: bool = True,
        chunk_size: int = 50,
        max_action_dim: int = 32,
        # Config from pretrained (optional)
        backbone_embedding_dim: int = 1536,
        diffusion_model_cfg: dict[str, Any] | None = None,
        vl_self_attention_cfg: dict[str, Any] | None = None,
    ) -> None:
        """Initialize Groot model with all components.

        Args:
            n_action_steps: Number of action steps to execute per chunk.
            use_bf16: Whether to use bfloat16 precision for compute.
            tokenizer_assets_repo: HF repo ID for Eagle tokenizer assets.
            attn_implementation: Attention implementation ('sdpa', 'flash_attention_2', 'eager').
            tune_llm: Whether to fine-tune the LLM backbone.
            tune_visual: Whether to fine-tune the vision tower.
            tune_projector: Whether to fine-tune the projector.
            tune_diffusion_model: Whether to fine-tune the diffusion model.
            chunk_size: Action horizon for action head.
            max_action_dim: Maximum action dimension.
            backbone_embedding_dim: Backbone output dimension (from config).
            diffusion_model_cfg: DiT model config dict (from pretrained).
            vl_self_attention_cfg: VL self-attention config dict (from pretrained).
        """
        super().__init__()

        # Store args
        self.n_action_steps = n_action_steps
        self.use_bf16 = use_bf16

        # Initialize backbone
        # Note: Pretrained GR00T uses backbone_embedding_dim=2048 with no projection
        # (project_to_dim=None), outputting the raw 2048-dim backbone embeddings
        self.backbone = EagleBackbone(
            tokenizer_assets_repo=tokenizer_assets_repo,
            attn_implementation=attn_implementation,
            tune_llm=tune_llm,
            tune_visual=tune_visual,
            project_to_dim=None,  # No projection, use raw backbone dim
        )

        # Initialize action head
        self.action_head = FlowMatchingActionHead(
            action_horizon=chunk_size,
            action_dim=max_action_dim,
            backbone_embedding_dim=backbone_embedding_dim,
            diffusion_model_cfg=diffusion_model_cfg,
            vl_self_attention_cfg=vl_self_attention_cfg,
        )
        self.action_head.set_trainable_parameters(
            tune_projector=tune_projector,
            tune_diffusion_model=tune_diffusion_model,
        )

        self._action_queue: deque[torch.Tensor] = deque(maxlen=n_action_steps)

    @classmethod
    def from_pretrained(  # noqa: PLR0914
        cls,
        pretrained_model_name_or_path: str = DEFAULT_BASE_MODEL_PATH,
        *,
        # Model args
        n_action_steps: int = 50,
        use_bf16: bool = True,
        # Backbone args
        tokenizer_assets_repo: str = DEFAULT_TOKENIZER_ASSETS_REPO,
        attn_implementation: str = "sdpa",
        tune_llm: bool = False,
        tune_visual: bool = False,
        # Action head args
        tune_projector: bool = True,
        tune_diffusion_model: bool = True,
        chunk_size: int = 50,
        max_action_dim: int = 32,
    ) -> GrootModel:
        """Load Groot model with pretrained weights.

        Downloads weights from HuggingFace and initializes the model.
        This is a convenience method that calls `__init__` and then loads weights.

        Args:
            pretrained_model_name_or_path: HuggingFace model ID or local path.
            n_action_steps: Number of action steps to execute per chunk.
            use_bf16: Whether to use bfloat16 precision for compute.
            tokenizer_assets_repo: HF repo ID for Eagle tokenizer assets.
            attn_implementation: Attention implementation ('sdpa', 'flash_attention_2', 'eager').
            tune_llm: Whether to fine-tune the LLM backbone.
            tune_visual: Whether to fine-tune the vision tower.
            tune_projector: Whether to fine-tune the projector.
            tune_diffusion_model: Whether to fine-tune the diffusion model.
            chunk_size: Fallback action horizon if not in checkpoint.
            max_action_dim: Fallback action dimension if not in checkpoint.

        Returns:
            Initialized GrootModel with pretrained weights.

        Examples:
            >>> model = GrootModel.from_pretrained()  # Uses defaults
            >>> model = GrootModel.from_pretrained(
            ...     "nvidia/GR00T-N1.5-3B",
            ...     attn_implementation="sdpa",
            ...     tune_projector=True,
            ... )
        """
        snapshot_download, hf_hub_http_error, repository_not_found_error = _import_snapshot_download()

        # Download model
        logger.info("[GROOT] Loading pretrained model from %s", pretrained_model_name_or_path)
        logger.info("[GROOT] Using attention implementation: %s", attn_implementation)

        try:
            local_model_path = snapshot_download(pretrained_model_name_or_path, repo_type="model")
        except (hf_hub_http_error, repository_not_found_error):
            logger.info("[GROOT] Model not found in HF hub, using local path: %s", pretrained_model_name_or_path)
            local_model_path = pretrained_model_name_or_path

        # Read config from pretrained model
        weights_path = Path(local_model_path)
        config_path = weights_path / "config.json"
        action_head_cfg_dict: dict[str, Any] = {}
        diffusion_cfg: dict[str, Any] = {}
        vl_cfg: dict[str, Any] = {}
        backbone_embedding_dim = 1536  # Default

        if config_path.exists():
            with config_path.open(encoding="utf-8") as f:
                full_config = json.load(f)
            action_head_cfg_dict = full_config.get("action_head_cfg", {})
            diffusion_cfg = action_head_cfg_dict.get("diffusion_model_cfg", {})
            vl_cfg = action_head_cfg_dict.get("vl_self_attention_cfg", {})
            backbone_embedding_dim = action_head_cfg_dict.get("backbone_embedding_dim", 1536)
            chunk_size = action_head_cfg_dict.get("action_horizon", chunk_size)
            max_action_dim = action_head_cfg_dict.get("action_dim", max_action_dim)
            logger.info("[GROOT] Loaded config: backbone_dim=%d, chunk_size=%d", backbone_embedding_dim, chunk_size)

        # Check for legacy action head config
        action_head_config_path = weights_path / "action_head_config.json"
        if action_head_config_path.exists():
            with action_head_config_path.open(encoding="utf-8") as f:
                legacy_cfg = json.load(f)
            chunk_size = legacy_cfg.get("action_horizon", chunk_size)
            max_action_dim = legacy_cfg.get("action_dim", max_action_dim)

        # Create model instance with all args including pretrained config
        model = cls(
            n_action_steps=n_action_steps,
            use_bf16=use_bf16,
            tokenizer_assets_repo=tokenizer_assets_repo,
            attn_implementation=attn_implementation,
            tune_llm=tune_llm,
            tune_visual=tune_visual,
            tune_projector=tune_projector,
            tune_diffusion_model=tune_diffusion_model,
            chunk_size=chunk_size,
            max_action_dim=max_action_dim,
            # Config from pretrained model
            backbone_embedding_dim=backbone_embedding_dim,
            diffusion_model_cfg=diffusion_cfg or None,
            vl_self_attention_cfg=vl_cfg or None,
        )

        # Load pretrained weights
        backbone_weights = weights_path / "backbone.pt"
        action_head_weights = weights_path / "action_head.pt"

        if backbone_weights.exists():
            model.backbone.load_state_dict(
                torch.load(backbone_weights, map_location="cpu"),
                strict=False,
            )
        if action_head_weights.exists():
            model.action_head.load_state_dict(
                torch.load(action_head_weights, map_location="cpu"),
                strict=False,
            )

        return model

    @classmethod
    def from_config(cls, config: GrootConfig) -> GrootModel:
        """Create GrootModel from a GrootConfig dataclass.

        This is a convenience method for configuration-based instantiation.
        It delegates to `from_pretrained()` with values from the config.

        Args:
            config: GrootConfig dataclass with model configuration.

        Returns:
            Initialized GrootModel with pretrained weights.

        Examples:
            >>> config = GrootConfig(attn_implementation="sdpa", tune_projector=True)
            >>> model = GrootModel.from_config(config)

            >>> # Equivalent to:
            >>> model = GrootModel.from_pretrained(
            ...     attn_implementation="sdpa",
            ...     tune_projector=True,
            ... )
        """
        return cls.from_pretrained(
            pretrained_model_name_or_path=config.pretrained_model_name_or_path,
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
        )

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Training forward pass - computes loss.

        Args:
            batch: Input batch with:
                - eagle_* tensors from preprocessor
                - state, state_mask
                - action, action_mask
                - embodiment_id

        Returns:
            Dict with 'loss' key containing training loss.
        """
        # Filter inputs for Groot
        allowed_base = {"state", "state_mask", "action", "action_mask", "embodiment_id"}
        groot_inputs = {
            k: v
            for k, v in batch.items()
            if (k in allowed_base or k.startswith("eagle_")) and not (k.startswith("next.") or k == "info")
        }

        # Use bf16 autocast if enabled for entire forward pass
        device = next(self.parameters()).device
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self.use_bf16):
            backbone_outputs = self.backbone(groot_inputs)
            return self.action_head(backbone_outputs, groot_inputs)

    def get_action(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Inference - predict actions.

        Args:
            batch: Input batch with eagle_* and state tensors.

        Returns:
            Predicted action tensor of shape (B, n_action_steps, action_dim).
        """
        self.eval()

        # Filter inputs (no action during inference)
        allowed_base = {"state", "state_mask", "embodiment_id"}
        groot_inputs = {
            k: v
            for k, v in batch.items()
            if (k in allowed_base or k.startswith("eagle_")) and not (k.startswith("next.") or k == "info")
        }

        device = next(self.parameters()).device
        with (
            torch.inference_mode(),
            torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self.use_bf16),
        ):
            backbone_outputs = self.backbone(groot_inputs)
            action_head_outputs = self.action_head.get_action(backbone_outputs, groot_inputs)

        return action_head_outputs.get("action_pred")

    def select_action(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Select single action using action chunking.

        Maintains internal queue of predicted actions, refilling
        when empty by calling get_action().

        Args:
            batch: Input batch for inference.

        Returns:
            Single action tensor.
        """
        if len(self._action_queue) == 0:
            actions = self.get_action(batch)
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    def reset(self) -> None:
        """Reset action queue for new episode."""
        self._action_queue.clear()

    def get_optim_params(self) -> list[nn.Parameter]:
        """Get parameters for optimizer.

        Returns only trainable parameters based on tune_* settings.

        Returns:
            List of trainable parameters.
        """
        return [p for p in self.parameters() if p.requires_grad]
