# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Mixin classes for handling PyTorch model checkpoints."""

from typing import Any, Self

import torch

from getiaction.config import Config
from getiaction.export.mixin_export import CONFIG_KEY


class FromCheckpoint:
    """Mixin class for loading torch models from checkpoints."""

    model_type: type[torch.nn.Module]
    model_config_type: type[Config]

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        map_location: torch.device | str | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> Self:
        """Load a policy from a Lightning checkpoint.

        This method loads the checkpoint, reconstructs the underlying model from the saved
        config, and restores the model weights.

        Args:
            checkpoint_path: Path to the checkpoint file (.ckpt).
            map_location: Device to map tensors to. If None, uses default device.
            **kwargs: Additional arguments passed to the policy constructor.

        Returns:
            Loaded policy with weights restored, ready for inference.

        Raises:
            KeyError: If checkpoint doesn't contain required model config.

        Examples:
            Load checkpoint for inference:

                >>> from getiaction.policies import ACT
                >>> policy = ACT.load_from_checkpoint("checkpoints/epoch=10.ckpt")
                >>> action = policy.select_action(observation)

            Load checkpoint to specific device:

                >>> policy = ACT.load_from_checkpoint(
                ...     "checkpoints/best.ckpt",
                ...     map_location="cuda:0",
                ... )

            Load checkpoint to CPU:

                >>> policy = ACT.load_from_checkpoint(
                ...     "checkpoints/best.ckpt",
                ...     map_location="cpu",
                ... )
        """
        # Load checkpoint - config is stored as plain dict (not dataclass) so
        # default weights_only=True works without needing pickle
        # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=True)  # nosec B614

        # Extract model config dict and reconstruct ACTConfig dataclass
        if CONFIG_KEY not in checkpoint:
            msg = (
                f"Checkpoint missing '{CONFIG_KEY}'. "
                "This checkpoint may have been saved with an older version. "
                "Please re-train and save a new checkpoint."
            )
            raise KeyError(msg)

        config_dict = checkpoint[CONFIG_KEY]
        config = cls.model_config_type.from_dict(config_dict)

        # Reconstruct model from config
        model = cls.model_type.from_config(config)

        # Create policy instance
        policy = cls(model=model, **kwargs)  # type: ignore[call-arg]

        # Load state dict (model weights + normalizer stats)
        policy.load_state_dict(checkpoint["state_dict"])  # type: ignore[attr-defined]

        return policy
