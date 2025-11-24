# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Helpful wrappers for Gym environments."""

from typing import Any

import torch

from getiaction.data.observation import Observation
from getiaction.gyms.base import Gym


class GymWrapper(Gym):
    """A wrapper that forwards all interface calls to an inner Gym.

    Subclasses may override selected methods while all other
    method and attribute access is transparently forwarded to the wrapped env.
    """

    def __init__(self, env: Gym) -> None:
        """Initialize the wrapper.

        Args:
            env: The concrete Gym environment to wrap.
        """
        self.env = env

    def reset(
        self,
        *args: Any,  # noqa: ANN401
        seed: int | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> tuple["Observation", dict[str, Any]]:
        """Reset the environment.

        Args:
            *args: Positional arguments forwarded to the wrapped environment.
            seed: Optional RNG seed passed to the underlying environment.
            **kwargs: Additional reset parameters forwarded to the wrapped env.

        Returns:
            A tuple ``(observation, info)`` from the wrapped environment.
        """
        return self.env.reset(*args, seed=seed, **kwargs)

    def step(
        self,
        action: torch.Tensor,
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> tuple["Observation", float, bool, bool, dict[str, Any]]:
        """Advance the environment by one step.

        Args:
            action: Action forwarded to the underlying environment.
            *args: Additional positional arguments forwarded to the wrapped env.
            **kwargs: Additional keyword arguments forwarded to the wrapped env.

        Returns:
            A tuple ``(observation, reward, terminated, truncated, info)``.
        """
        return self.env.step(action, *args, **kwargs)

    def render(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """Render the environment.

        Args:
            *args: Positional render arguments forwarded to the environment.
            **kwargs: Keyword render arguments forwarded to the environment.

        Returns:
            The render output from the wrapped environment.
        """
        return self.env.render(*args, **kwargs)

    def close(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        """Release environment resources.

        Args:
            *args: Positional arguments forwarded to the close method.
            **kwargs: Keyword arguments forwarded to the close method.
        """
        return self.env.close(*args, **kwargs)

    def sample_action(self, *args: Any, **kwargs: Any) -> torch.Tensor:  # noqa: ANN401
        """Sample a valid action from the environment.

        Args:
            *args: Arguments forwarded to the underlying sampler.
            **kwargs: Additional keyword arguments.

        Returns:
            A sampled action tensor.
        """
        return self.env.sample_action(*args, **kwargs)

    def to_observation(
        self,
        raw_obs: Any,  # noqa: ANN401
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> "Observation":
        """Convert raw env output into a standardized Observation.

        Args:
            raw_obs: Raw observation from the backend.
            *args: Additional positional arguments forwarded to the env.
            **kwargs: Additional keyword arguments forwarded to the env.

        Returns:
            A normalized Observation instance.
        """
        return self.env.to_observation(raw_obs, *args, **kwargs)

    @staticmethod
    def convert_raw_to_observation(raw_obs: Any) -> Observation:  # noqa: ANN401
        """Static conversion fallback.

        Wrapper-level static conversion cannot know which env instance to use.
        This implementation exists only to satisfy the Gym ABC.
        """
        msg = "Use instance.convert_raw_to_observation(...) instead, which forwards to the wrapped env."
        raise NotImplementedError(
            msg,
        )

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        """Forward attribute access to the wrapped environment.

        Args:
            name: Attribute name to resolve.

        Returns:
            The attribute resolved on the wrapped env.

        Raises:
            AttributeError: If neither wrapper nor env define the attribute.
        """
        try:
            return getattr(self.env, name)
        except AttributeError:
            msg = f"'{type(self).__name__}' and '{type(self.env).__name__}' do not define attribute '{name}'."
            raise AttributeError(
                msg,
            ) from None


class StepLimit(GymWrapper):
    """Wrapper enforcing a maximum number of steps per episode."""

    def __init__(self, env: Gym, max_steps: int) -> None:
        """Initialize the step-limit wrapper.

        Args:
            env: Environment instance to wrap.
            max_steps: Maximum allowed steps before truncation.
        """
        super().__init__(env)
        self.max_steps = max_steps
        self.step_count = 0

    def reset(
        self,
        *args: Any,  # noqa: ANN401
        seed: int | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> tuple["Observation", dict[str, Any]]:
        """Reset the environment and reset the step counter.

        Args:
            *args: Additional positional arguments.
            seed: Optional RNG seed.
            **kwargs: Additional reset kwargs.

        Returns:
            A tuple ``(observation, info)``.
        """
        self.step_count = 0
        return self.env.reset(*args, seed=seed, **kwargs)

    def step(
        self,
        action: Any,  # noqa: ANN401
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> tuple["Observation", float, bool, bool, dict[str, Any]]:
        """Step the environment and apply the step limit.

        Args:
            action: Action forwarded to the env.
            *args: Additional step arguments.
            **kwargs: Additional step keyword arguments.

        Returns:
            A tuple ``(observation, reward, terminated, truncated, info)``.
        """
        self.step_count += 1
        obs, reward, terminated, truncated, info = self.env.step(action, *args, **kwargs)

        if self.step_count >= self.max_steps:
            truncated = True
            info = dict(info)
            info["TimeLimit.truncated"] = True

        return obs, reward, terminated, truncated, info
