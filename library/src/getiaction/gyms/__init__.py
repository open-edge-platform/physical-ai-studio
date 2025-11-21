# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Action trainer gym simulation environments."""

from .base import Gym
from .gymnasium_wrapper import GymnasiumWrapper


class PushTGym(GymnasiumWrapper):
    """Convenience wrapper for popularly used gym."""
    def __init__(
        self,
        gym_id="gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",
        device="cpu",
        **kwargs
    ):
        super().__init__(
            gym_id=gym_id,
            obs_type=obs_type,
            device=device,
            **kwargs,
        )


__all__ = ["Gym", "GymnasiumWrapper", "PushTGym"]
