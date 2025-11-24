# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PushT Gym Environment."""

from typing import Any

import gym_pusht  # noqa: F401
import torch

from getiaction.data.observation import Observation

from .gymnasium_wrapper import GymnasiumWrapper


class PushTGym(GymnasiumWrapper):
    """Convenience wrapper for the PushT Gym environment."""

    def __init__(
        self,
        gym_id: str = "gym_pusht/PushT-v0",
        obs_type: str = "pixels_agent_pos",
        device: str | torch.device = "cpu",
        **gym_kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize the PushT Gym environment.

        Args:
            gym_id: Environment ID for ``gym.make``.
            obs_type: Requested observation type.
            device: Torch device.
            **gym_kwargs: Additional gym keyword arguments.
        """
        super().__init__(
            gym_id=gym_id,
            obs_type=obs_type,
            device=device,
            **gym_kwargs,
        )

    @staticmethod
    def convert_raw_to_observation(
        raw_obs: dict[str, Any],
        camera_keys: list[str] | None = None,
    ) -> Observation:
        """Convert PushT observations into the standardized Observation format.

        Args:
            raw_obs: Normalized (batched) observation dict from GymnasiumWrapper.
                     Expected keys include:
                       - "pixels": np.ndarray[B,H,W,C]
                       - "agent_pos": np.ndarray[B,2]
            camera_keys: Optional camera identifiers.

        Returns:
            Observation: With images in BCHW format and state in [B,dim].
        """
        if camera_keys is None:
            camera_keys = ["top"]

        images: dict[str, torch.Tensor] | torch.Tensor | None = None
        state: torch.Tensor | None = None

        # Images: pixels[B,H,W,C] → [B,C,H,W], float32
        if "pixels" in raw_obs:
            pixels = raw_obs["pixels"]

            if not isinstance(pixels, torch.Tensor):
                pixels = torch.from_numpy(pixels)

            if pixels.dtype not in {torch.float32, torch.float16}:
                original_dtype = pixels.dtype
                pixels = pixels.float()
                if original_dtype == torch.uint8:
                    pixels /= 255.0

            if pixels.ndim == 4 and pixels.shape[-1] in {1, 3, 4}:  # noqa: PLR2004
                pixels = pixels.permute(0, 3, 1, 2)

            images = pixels if len(camera_keys) == 1 else {camera_keys[0]: pixels}

        # State agent_pos[B,2]
        state_keys = ["agent_pos", "state"]
        for key in state_keys:
            if key in raw_obs:
                state_data = raw_obs[key]

                if not isinstance(state_data, torch.Tensor):
                    state_data = torch.from_numpy(state_data)

                if state_data.dtype != torch.float32:
                    state_data = state_data.float()

                state = state_data
                break

        return Observation(images=images, state=state)  # type: ignore[arg-type]
