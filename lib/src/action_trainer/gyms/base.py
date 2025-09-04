# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Base class for gym environments
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import gymnasium as gym

if TYPE_CHECKING:
    from gymnasium.core import ActType, ObsType


class BaseGym:
    """Base class for Gym environments with configurable observation type,
    number of rollouts, and maximum episode steps.

    This class wraps a Gym environment and provides standard methods
    like `reset`, `step`, `render`, and `close`. It also exposes
    properties for `num_rollouts` and `max_episode_steps`.
    """

    def __init__(
        self,
        gym_id: str,
        obs_type: str,
        max_episode_steps: int,
    ) -> None:
        """Initializes the base Gym environment wrapper.

        Args:
            gym_id (str): The identifier for the Gymnasium environment.
            obs_type (str): The type of observation to use (e.g., 'pixels', 'state').
            max_episode_steps (int): The maximum number of steps allowed per episode.
        """

        self._max_episode_steps = max_episode_steps
        self._gym_id = gym_id
        self._obs_type = obs_type

        # create wrapped environment
        self.env = gym.make(
            self._gym_id,
            obs_type=self._obs_type,
            max_episode_steps=self._max_episode_steps,
        )

        # Assign the observation and action spaces from the wrapped environment
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """Resets the environment to its initial state.

        Args:
            seed (int, optional): The seed for the environment's random number generator.
                Defaults to None.
            options (dict, optional): Additional options for the reset.
                Defaults to None.

        Returns:
            tuple[ObsType, dict[str, Any]]: A tuple containing the initial observation
                and an info dictionary.
        """
        return self.env.reset(seed=seed, options=options)

    def step(
        self,
        action: ActType,
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        """Takes a step in the environment with the given action.

        Args:
            action (ActType): The action to perform.

        Returns:
            tuple[ObsType, float, bool, bool, dict[str, Any]]: A tuple containing the
                observation, reward, termination, truncation, and info dictionary.
        """
        return self.env.step(action)

    def render(self, *args: Any, **kwargs: Any) -> Any:
        """Renders the environment for visualization."""
        return self.env.render(*args, **kwargs)

    def close(self) -> None:
        """Closes the environment and releases resources."""
        return self.env.close()

    @property
    def max_episode_steps(self) -> int:
        """The maximum number of steps allowed per episode."""
        return self._max_episode_steps
