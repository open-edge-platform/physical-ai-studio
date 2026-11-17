# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Selected-key observation history for any gym implementing the Gym interface."""

from __future__ import annotations

from collections import deque
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

from physicalai.data.observation import Observation

from .base import Gym

if TYPE_CHECKING:
    from .types import SingleOrBatch


class HistoryGym(Gym):
    """Stack selected batched observation values, oldest to newest, on axis 1.

    Keys are observation paths such as ``images.top``, ``images.image1``, or
    ``state``. Other values, including task text, remain current. Reset pads
    each window with the initial value. Selected values always gain a time
    axis, including when ``n_obs_steps=1``.

    ``stride`` spaces window entries by environment steps. History advances on
    every step, and the newest entry is always the current observation.

    The wrapped gym must reset explicitly between episodes; independent
    vector-environment auto-resets are not supported.
    """

    def __init__(self, gym: Gym, n_obs_steps: int, keys: set[str], stride: int = 1) -> None:
        """Initialize an observation-history wrapper.

        Args:
            gym: Environment implementing the Gym interface.
            n_obs_steps: Positive history window length.
            keys: Nonempty set of tensor or NumPy observation paths to stack.
            stride: Positive spacing between window entries in environment steps.

        Raises:
            ValueError: If the window length, stride, or keys are invalid.
        """
        if isinstance(n_obs_steps, bool) or not isinstance(n_obs_steps, int) or n_obs_steps < 1:
            msg = "n_obs_steps must be a positive integer."
            raise ValueError(msg)
        if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
            msg = "stride must be a positive integer."
            raise ValueError(msg)
        field_names = Observation.keys()
        if not keys or any(
            not isinstance(key, str)
            or len(key.split(".")) > 2  # noqa: PLR2004
            or key.split(".")[0] not in field_names
            or any(not part for part in key.split("."))
            for key in keys
        ):
            msg = "keys must contain observation paths such as 'images.top' or 'state'."
            raise ValueError(msg)
        if any(key.split(".")[0] in keys for key in keys if "." in key):
            msg = "Select either a field or its nested keys, not both."
            raise ValueError(msg)
        self.gym = gym
        self.n_obs_steps = n_obs_steps
        self.stride = stride
        self.keys = frozenset(keys)
        self._history: dict[str, deque[torch.Tensor | np.ndarray]] = {
            key: deque(maxlen=(n_obs_steps - 1) * stride + 1) for key in sorted(keys)
        }

    def _stack_observation(self, observation: Observation) -> Observation:
        """Copy selected values into history without mutating backend observations.

        Returns:
            A new observation with history at the selected paths.

        Raises:
            ValueError: If a selected value is missing or is not a batched array.
        """
        updates: dict[str, Any] = {}
        for key, history in self._history.items():
            field, _, nested_key = key.partition(".")
            value = getattr(observation, field)
            if nested_key:
                if not isinstance(value, dict) or nested_key not in value:
                    msg = f"Observation is missing history key {key!r}."
                    raise ValueError(msg)
                value = value[nested_key]
            if not isinstance(value, (torch.Tensor, np.ndarray)) or value.ndim < 1:
                msg = f"History key {key!r} must contain a batched tensor or NumPy array."
                raise ValueError(msg)
            snapshot = value.clone() if isinstance(value, torch.Tensor) else value.copy()
            history.append(snapshot)
            while len(history) < (self.n_obs_steps - 1) * self.stride + 1:
                history.appendleft(snapshot)
            window = tuple(history)[:: self.stride]
            stacked = (
                torch.stack(cast("tuple[torch.Tensor, ...]", window), dim=1)
                if isinstance(value, torch.Tensor)
                else np.stack(window, axis=1)
            )
            if nested_key:
                if field not in updates:
                    updates[field] = dict(getattr(observation, field))
                updates[field][nested_key] = stacked
            else:
                updates[field] = stacked
        return replace(observation, **updates)

    def reset(
        self,
        *,
        seed: int | None = None,
        episode_index: int = 0,
        **reset_kwargs: Any,  # noqa: ANN401
    ) -> tuple[Observation, dict[str, Any] | list[dict[str, Any]]]:
        """Reset the gym and fill history with its initial observation.

        Returns:
            History observation and unchanged reset info.
        """
        for history in self._history.values():
            history.clear()
        observation, info = self.gym.reset(seed=seed, episode_index=episode_index, **reset_kwargs)
        return self._stack_observation(observation), info

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[
        Observation,
        SingleOrBatch[float],
        SingleOrBatch[bool],
        SingleOrBatch[bool],
        SingleOrBatch[dict[str, Any]],
    ]:
        """Step the gym and append the resulting observation to history.

        Returns:
            History observation and unchanged reward, termination, truncation, and info.
        """
        observation, reward, terminated, truncated, info = self.gym.step(action)
        return self._stack_observation(observation), reward, terminated, truncated, info

    def render(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """Render the wrapped gym without advancing history.

        Returns:
            The wrapped gym's render output.
        """
        return self.gym.render(*args, **kwargs)

    def get_max_episode_steps(self) -> int | None:
        """Return the wrapped gym's episode limit."""
        return self.gym.get_max_episode_steps()

    def close(self) -> None:
        """Release wrapped gym resources."""
        self.gym.close()

    def sample_action(self) -> torch.Tensor:
        """Sample an action from the wrapped gym.

        Returns:
            An action in the wrapped gym's format.
        """
        return self.gym.sample_action()

    def to_observation(self, raw_obs: Any) -> Observation:  # noqa: ANN401
        """Convert backend data without advancing history.

        Returns:
            The wrapped gym's unstacked observation.
        """
        return self.gym.to_observation(raw_obs)

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        """Forward backend metadata, including task descriptions, to the gym.

        Returns:
            The attribute on the wrapped gym.
        """
        gym = object.__getattribute__(self, "gym")
        return getattr(gym, name)
