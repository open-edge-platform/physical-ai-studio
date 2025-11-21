# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Helpful Wrappers """

from typing import Any, Tuple
from getiaction.data import Observation
import torch

from .base import Gym


class GymWrapper(Gym):
    """Base class for wrapping a ``Gym`` environment.

    This wrapper forwards all method calls to the underlying environment unless
    explicitly overridden.

    Args:
        env (Gym): The environment to be wrapped.
    """

    def __init__(self, env: Gym) -> None:
        """Initialize the wrapper.

        Args:
            env (Gym): The environment instance to wrap.
        """
        self.env = env

    def reset(self, *args: Any, **kwargs: Any) -> tuple[Observation, dict[str, Any]]:
        """Reset the environment.

        Args:
            *args: Positional arguments forwarded to the wrapped environment.
            **kwargs: Keyword arguments forwarded to the wrapped environment.

        Returns:
            tuple[Observation, dict[str, Any]]:
                - Observation: Environment observation after reset.
                - dict[str, Any]: Additional info dictionary.
        """
        return self.env.reset(*args, **kwargs)

    def step(self, *args: Any, **kwargs: Any) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """Perform a step in the environment.

        Args:
            *args: Positional arguments forwarded to the wrapped environment.
            **kwargs: Keyword arguments forwarded to the wrapped environment.

        Returns:
            Tuple[Observation, float, bool, bool, dict]:
                A tuple containing observation, reward, terminated, truncated,
                and info.
        """
        return self.env.step(*args, **kwargs)

    def render(self, *args: Any, **kwargs: Any) -> Any:
        """Render the environment.

        Args:
            *args: Positional arguments forwarded to the environment.
            **kwargs: Keyword arguments forwarded.

        Returns:
            Any: The rendered frame or result defined by the wrapped environment.
        """
        return self.env.render(*args, **kwargs)

    def close(self) -> None:
        """Close the environment and free resources."""
        return self.env.close()

    def sample_action(self) -> torch.Tensor:
        """Sample a random action from the environment's action space.

        Returns:
            torch.Tensor: A randomly sampled action.
        """
        return self.env.sample_action()

    def to_observation(self, *args: Any, **kwargs: Any) -> Observation:
        """Convert environment output to an observation.

        Args:
            *args: Positional arguments forwarded.
            **kwargs: Keyword arguments forwarded.

        Returns:
            Observation: The resulting observation.
        """
        return self.env.to_observation(*args, **kwargs)


class StepLimit(GymWrapper):
    """Limit the number of steps taken in an environment.

    This wrapper enforces a maximum number of steps per episode. Once this
    threshold is reached, the episode is truncated.

    Args:
        env (Gym): The environment to wrap.
        max_steps (int): Maximum number of allowed steps before truncation.
    """

    def __init__(self, env: Gym, max_steps: int) -> None:
        """Initialize the ``StepLimit`` wrapper.

        Args:
            env (Gym): The environment to wrap.
            max_steps (int): The maximum number of steps before truncation.
        """
        super().__init__(env)
        self.max_steps = max_steps
        self.step_count = 0

    def reset(self, *args: Any, **kwargs: Any) -> tuple[Observation, dict[str, Any]]: 
        """Reset the environment and step counter.

        Args:
            *args: Forwarded positional arguments.
            **kwargs: Forwarded keyword arguments.

        Returns:
            tuple[Observation, dict[str, Any]]:
                - Observation: Environment observation after reset.
                - dict[str, Any]: Additional info dictionary.
        """
        self.step_count = 0
        return super().reset(*args, **kwargs)

    def step(
        self,
        action: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[Observation, float, bool, bool, dict]:
        """Perform a step and enforce the step limit.

        Args:
            action (Any): The action to take.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple[Observation, float, bool, bool, dict]:
                A tuple of observation, reward, terminated, truncated, and info.
        """
        self.step_count += 1

        obs, reward, terminated, truncated, info = super().step(
            action, *args, **kwargs
        )

        if self.step_count >= self.max_steps:
            truncated = True
            info = dict(info)
            info["TimeLimit.truncated"] = True

        return obs, reward, terminated, truncated, info
