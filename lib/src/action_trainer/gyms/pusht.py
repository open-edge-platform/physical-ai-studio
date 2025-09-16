# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
PushT Gym Environment
"""

import gym_pusht  # noqa: F401

from action_trainer.gyms import BaseGym


class PushTGym(BaseGym):
    """
    A  Gymnasium environment wrapper for the PushT task.
    """

    def __init__(
        self,
        gym_id: str = "gym_pusht/PushT-v0",
        obs_type: str = "pixels_agent_pos",
    ) -> None:
        """
        Initialize the PushT Gym environment.

        Args:
            gym_id (str): The identifier for the environment.
            obs_type (str): The type of observation to use (e.g., pixels, state).
        """
        super().__init__(
            gym_id=gym_id,
            obs_type=obs_type,
        )
