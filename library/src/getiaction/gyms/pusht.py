# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PushT Gym Environment."""

from typing import Any

import gym_pusht  # noqa: F401
import torch

from getiaction.data.observation import Observation
from getiaction.gyms import Gym


class PushTGym(Gym):
    """A PushT Gymnasium environment wrapper for the PushT task."""

    def __init__(
        self,
        gym_id: str = "gym_pusht/PushT-v0",
        obs_type: str = "pixels_agent_pos",
    ) -> None:
        """Initialize the PushT Gym environment.

        Args:
            gym_id (str): The identifier for the environment.
            obs_type (str): The type of observation to use (e.g., pixels, state).
        """
        super().__init__(
            gym_id=gym_id,
            obs_type=obs_type,
        )

    @staticmethod
    def convert_raw_observation(
        raw_obs: dict[str, Any],
        camera_keys: list[str] | None = None,
    ) -> Observation:
        """Convert PushT gym observation to Observation dataclass.

        Static method for converting raw PushT observations without needing a gym instance.

        PushT environments provide observations with "pixels" (RGB image) and
        "agent_pos" (2D position) keys.

        Args:
            raw_obs: Raw observation dictionary from PushT gym environment.
                    Expected keys: "pixels" (numpy.ndarray), "agent_pos" (numpy.ndarray)
            camera_keys: Camera names for multi-camera setups (defaults to ["top"])

        Returns:
            Standardized Observation dataclass

        Examples:
            >>> # Use as static method (no gym instance needed)
            >>> raw_obs = {
            ...     "pixels": np.zeros((320, 512, 3), dtype=np.uint8),
            ...     "agent_pos": np.array([0.1, 0.2], dtype=np.float32)
            ... }
            >>> obs = PushTGym.convert_raw_observation(raw_obs)
            >>> # obs.images.shape = torch.Size([1, 3, 320, 512])  # Batched, CHW
            >>> # obs.state.shape = torch.Size([1, 2])  # Batched agent position
        """
        if camera_keys is None:
            camera_keys = ["top"]

        images: dict[str, torch.Tensor] | torch.Tensor | None = None
        state: torch.Tensor | None = None

        # Handle image observations
        if "pixels" in raw_obs:
            pixels = raw_obs["pixels"]

            # Convert to torch tensor
            if not isinstance(pixels, torch.Tensor):
                pixels = torch.from_numpy(pixels)

            # Ensure float32 (gym often returns float64 or uint8)
            if pixels.dtype not in {torch.float32, torch.float16}:
                pixels = pixels.float()

            # Convert HWC → CHW if needed
            if pixels.ndim == 3 and pixels.shape[-1] in {1, 3, 4}:  # noqa: PLR2004
                pixels = pixels.permute(2, 0, 1)  # (H, W, C) → (C, H, W)

            # Add batch dimension
            if pixels.ndim == 3:  # noqa: PLR2004
                pixels = pixels.unsqueeze(0)  # (C, H, W) → (1, C, H, W)

            # For single camera: direct tensor for compatibility with existing models
            # For multiple cameras: dict with camera names
            images = pixels if len(camera_keys) == 1 else {camera_keys[0]: pixels}

        # Handle state observations (PushT uses "agent_pos")
        state_keys = ["agent_pos", "state"]  # Common gym state keys
        for key in state_keys:
            if key in raw_obs:
                state_data = raw_obs[key]

                # Convert to torch tensor
                if not isinstance(state_data, torch.Tensor):
                    state_data = torch.from_numpy(state_data)

                # Ensure float32
                if state_data.dtype != torch.float32:
                    state_data = state_data.float()

                # Add batch dimension
                if state_data.ndim == 1:
                    state_data = state_data.unsqueeze(0)  # (D,) → (1, D)

                state = state_data
                break

        return Observation(images=images, state=state)  # type: ignore[arg-type]

    def to_observation(
        self,
        raw_obs: dict[str, Any],
        camera_keys: list[str] | None = None,
    ) -> Observation:
        """Convert PushT gym observation to Observation dataclass.

        Instance method that delegates to the static method for compatibility with base class.

        Args:
            raw_obs: Raw observation dictionary from PushT gym environment.
            camera_keys: Camera names for multi-camera setups (defaults to ["top"])

        Returns:
            Standardized Observation dataclass
        """
        return self.convert_raw_observation(raw_obs, camera_keys)
