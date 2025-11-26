# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""A lightweight wrapper enforcing a maximum number of steps per episode."""

from typing import Any

import torch

from getiaction.data.observation import Observation

from .base import Gym
from .types import SingleOrBatch


class StepLimit(Gym):
    """A lightweight wrapper that enforces a maximum number of steps per episode.

    This wrapper monitors how many steps have been executed in the underlying
    Gym environment and automatically sets the `truncated` flag to ``True`` once
    the configured limit is reached. This allows environments that may not
    terminate naturally to be capped at a controlled horizon, which is useful
    for training, evaluation, or preventing infinite rollouts.
    """

    def __init__(self, gym: Gym, max_steps: int) -> None:
        """Wrapper for Gym that truncates environment based on steps.

        Args:
            gym (Gym): the underlying gym to be wrapped.
            max_steps (int): the maximum number of steps before truncation.
        """
        self.gym = gym
        self.max_steps = max_steps
        self.step_count = 0

    def reset(
        self,
        *args: Any,  # noqa: ANN401
        seed: int | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> tuple[Observation, dict[str, Any] | list[dict[str, Any]]]:
        """Reset the environment and reset the step counter.

        Args:
            *args: Additional positional arguments.
            seed: Optional RNG seed.
            **kwargs: Additional reset kwargs.

        Returns:
            A tuple ``(observation, info)``.
        """
        self.step_count = 0
        return self.gym.reset(*args, seed=seed, **kwargs)

    def step(
        self,
        action: Any,  # noqa: ANN401
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> tuple[
        Observation,
        SingleOrBatch[float],
        SingleOrBatch[bool],
        SingleOrBatch[bool],
        dict[str, Any] | list[dict[str, Any]],
    ]:
        """Step the environment and apply the step limit.

        Args:
            action: Action forwarded to the env.
            *args: Additional step arguments.
            **kwargs: Additional step keyword arguments.

        Returns:
            A tuple ``(observation, reward, terminated, truncated, info)``.
        """
        self.step_count += 1
        obs, reward, terminated, truncated, info = self.gym.step(action, *args, **kwargs)

        if self.step_count >= self.max_steps:
            batch_size = obs.batch_size
            truncated = True if batch_size == 1 else [True] * batch_size
            # if info is a dict, attach the Timilimit truncation
            if isinstance(info, dict):
                info = dict(info)
                info["TimeLimit.truncated"] = True
            else:
                info = [dict(d) for d in info]
                for d in info:
                    d["TimeLimit.truncated"] = True

        return obs, reward, terminated, truncated, info

    def render(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """Render the environment.

        Args:
            *args: Positional render arguments forwarded to the environment.
            **kwargs: Keyword render arguments forwarded to the environment.

        Returns:
            The render output from the wrapped environment.
        """
        return self.gym.render(*args, **kwargs)

    def close(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        """Release environment resources.

        Args:
            *args: Positional arguments forwarded to the close method.
            **kwargs: Keyword arguments forwarded to the close method.
        """
        return self.gym.close(*args, **kwargs)

    def sample_action(self, *args: Any, **kwargs: Any) -> torch.Tensor:  # noqa: ANN401
        """Sample a valid action from the environment.

        Args:
            *args: Arguments forwarded to the underlying sampler.
            **kwargs: Additional keyword arguments.

        Returns:
            A sampled action tensor.
        """
        return self.gym.sample_action(*args, **kwargs)

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
        return self.gym.to_observation(raw_obs, *args, **kwargs)

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
            return getattr(self.gym, name)
        except AttributeError:
            msg = f"'{type(self).__name__}' and '{type(self.gym).__name__}' do not define attribute '{name}'."
            raise AttributeError(
                msg,
            ) from None


def with_step_limit(gym: Gym, max_steps: int) -> Gym:
    """Add step limit to gym environment.

    Args:
        gym (Gym): Gym environment to wrap step limit.
        max_steps (int): the maximum number of steps before truncation.

    Returns:
        Gym: Gym with functionality to truncate after max_steps reached.
    """
    return StepLimit(gym=gym, max_steps=max_steps)
