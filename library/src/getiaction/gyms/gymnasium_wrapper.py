# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""GymnasiumWrapper: adapts any Gymnasium environment to the abstract Gym interface.

Note:
    If you want a GPU-optimized gym, please implement your own. This wrapper
    assumes NumPy-style Gymnasium environments.
"""

import logging
from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

from getiaction.data.observation import Observation

from .base import Gym

logger = logging.getLogger(__name__)


class ActionValidationError(ValueError):
    """Error raised when an invalid action is provided."""


class GymnasiumWrapper(Gym):
    """Adapter that makes a Gymnasium environment conform to the unified Gym API."""

    def __init__(
        self,
        gym_id: str | None = None,
        vector_env: gym.Env | None = None,
        device: str | torch.device = "cpu",
        render_mode: str | None = "rgb_array",
        **gym_kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize a GymnasiumWrapper.

        Args:
            gym_id (str | None): The environment ID passed to ``gym.make``.
                Required if ``vector_env`` is not provided.
            vector_env (gym.Env | None): A preconstructed vectorized environment,
                typically ``SyncVectorEnv`` or ``AsyncVectorEnv``.
            device (str | torch.device): Torch device used for returned tensors.
            render_mode (str | None): Rendering mode passed to ``gym.make``.
            **gym_kwargs (Any): Additional keyword arguments forwarded to ``gym.make``.
        """
        if vector_env is not None:
            self._env = vector_env
        else:
            if render_mode is not None:
                gym_kwargs["render_mode"] = render_mode
            self._env = gym.make(gym_id, **gym_kwargs)

        self._device = torch.device(device)

        self.num_envs = getattr(self._env, "num_envs", 1)
        self._is_vectorized = self.num_envs > 1

    @property
    def device(self) -> torch.device:
        """Return the device Gym expects to return on.

        Returns:
            torch.device: The device
        """
        return self._device

    @property
    def is_vectorized(self) -> bool:
        """Returns whether the Gym is vectorized.

        Returns:
            bool: Whether the env is vectorized.
        """
        return self._is_vectorized

    @property
    def render_mode(self) -> str | None:
        """Return the underlying environment's render mode.

        Returns:
            str | None: The render mode if available.
        """
        return getattr(self._env, "render_mode", None)

    @property
    def observation_space(self) -> gym.Space | None:
        """Return the observation space of the environment.

        Returns:
            gym.Space | None: The observation space.
        """
        return getattr(self._env, "observation_space", None)

    @property
    def action_space(self) -> gym.Space | None:
        """Return the action space of the environment.

        Returns:
            gym.Space | None: The action space.
        """
        return getattr(self._env, "action_space", None)

    def reset(
        self,
        *,
        seed: int | None = None,
        **reset_kwargs: Any,  # noqa: ANN401
    ) -> tuple[Observation, dict[str, Any]]:
        """Reset the environment.

        Args:
            seed (int | None): Optional random seed for resetting the environment.
            **reset_kwargs (Any): Additional arguments forwarded to ``env.reset``.

        Returns:
            tuple[Observation, dict[str, Any]]: A tuple containing the initial
            observation and info dictionary.
        """
        raw_obs, info = self._env.reset(seed=seed, **reset_kwargs)
        obs = self.to_observation(raw_obs)
        return obs, info

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """Perform a single step in the environment.

        Args:
            action (torch.Tensor): The action to apply. May be unbatched or batched
                (shape ``[B, ...]``). If unvectorized, a leading batch dimension of
                size 1 is removed.

        Returns:
            tuple[Observation, float, bool, bool, dict[str, Any]]: A tuple containing:
                - the next observation
                - reward
                - terminated flag
                - truncated flag
                - info dictionary
        """
        if not self.is_vectorized and action.ndim == 2 and action.shape[0] == 1:  # noqa: PLR2004
            action = action[0]

        raw_action = action.detach().cpu().numpy()
        raw_obs, reward, terminated, truncated, info = self._env.step(raw_action)
        obs = self.to_observation(raw_obs)
        return obs, reward, terminated, truncated, info

    def render(self, *render_args: Any, **render_kwargs: Any) -> Any:  # noqa: ANN401
        """Render the environment.

        Args:
            *render_args (Any): Positional arguments forwarded to the environment's
                render function.
            **render_kwargs (Any): Keyword arguments forwarded to the environment's
                render function.

        Returns:
            Any: The rendered output, if the environment supports rendering.
        """
        if hasattr(self._env, "render"):
            return self._env.render(*render_args, **render_kwargs)
        return None

    def close(self) -> None:
        """Close the environment."""
        self._env.close()

    def sample_action(self) -> torch.Tensor:
        """Sample a valid action from the action space.

        Returns:
            torch.Tensor: A sampled action converted to a torch tensor on the
                configured device.
        """
        a = self._env.action_space.sample()
        return torch.as_tensor(a, device=self.device)

    def get_max_episode_steps(self) -> int | None:
        """Return the maximum allowed episode length, if available.

        Returns:
            int | None: The maximum episode step count, or ``None`` if the
            environment does not specify a limit.
        """
        if hasattr(self._env, "get_wrapper_attr"):
            try:
                return self._env.get_wrapper_attr("max_episode_steps")
            except AttributeError:
                logger.debug(
                    "get_wrapper_attr('max_episode_steps') not found on %r",
                    self._env,
                )
        return None

    def to_observation(
        self,
        raw_obs: np.ndarray | dict[str, Any],
    ) -> Observation:
        """Convert raw environment observations into an ``Observation`` instance.

        Args:
            raw_obs (np.ndarray | dict[str, Any]): Raw observation returned by
                Gymnasium.

        Returns:
            Observation: A processed ``Observation`` object on the correct device.
        """
        return self.convert_raw_to_observation(raw_obs=raw_obs).to_torch(device=self.device)

    @staticmethod
    def convert_raw_to_observation(
        raw_obs: np.ndarray | dict[str, Any],
    ) -> Observation:
        """Convert a Gymnasium observation to an ``Observation`` dataclass instance.

        Conversion rules:
            * If the observation is not a dict, it is treated as a ``state`` entry.
            * If the dict already appears to match Observation fields, it is passed
              directly to ``Observation.from_dict``.
            * Otherwise, keys are routed to ``images``, ``state``, or ``extra`` fields
              based on naming conventions.

        Args:
            raw_obs (np.ndarray | dict[str, Any]): Raw Gymnasium observation.

        Returns:
            Observation: A populated ``Observation`` instance.
        """
        if not isinstance(raw_obs, dict):
            return Observation(state=raw_obs).to_torch()

        obs_fields = {
            "action",
            "task",
            "state",
            "images",
        }

        if any(k in raw_obs for k in obs_fields):
            return Observation.from_dict(raw_obs)

        images = {}
        state = {}
        extra = {}

        for key, value in raw_obs.items():
            key_lower = key.lower()

            if any(tok in key_lower for tok in ("pixel", "pixels", "image", "rgb", "camera")):
                arr = value
                if isinstance(arr, np.ndarray):
                    if arr.ndim == 3 and arr.shape[-1] in {1, 3, 4}:  # noqa: PLR2004
                        arr = np.transpose(arr, (2, 0, 1))
                    elif arr.ndim == 4 and arr.shape[-1] in {1, 3, 4}:  # noqa: PLR2004
                        arr = np.transpose(arr, (0, 3, 1, 2))
                images[key] = arr
                continue

            if any(tok in key_lower for tok in ("pos", "agent_pos", "state", "obs", "feature")):
                state[key] = value
                continue

            extra[key] = value

        if not images and not state:
            state = dict(raw_obs.items())

        return Observation(
            images=images or None,
            state=state or None,
            extra=extra or None,
        ).to_torch()

    @staticmethod
    def vectorize(
        gym_id: str,
        num_envs: int,
        *,
        async_mode: bool = False,
        render_mode: str | None = "rgb_array",
        **gym_kwargs: Any,  # noqa: ANN401
    ) -> "GymnasiumWrapper":
        """Create a vectorized GymnasiumWrapper.

        Args:
            gym_id (str): Environment ID used for ``gym.make``.
            num_envs (int): Number of parallel environments.
            async_mode (bool): Whether to create an asynchronous vector environment.
            render_mode (str | None): Rendering mode.
            **gym_kwargs (Any): Extra arguments passed to ``gym.make``.

        Returns:
            GymnasiumWrapper: A wrapper around the constructed vector environment.
        """
        if async_mode:
            vec = make_async_vector_env(
                gym_id,
                num_envs,
                render_mode=render_mode,
                **gym_kwargs,
            )
        else:
            vec = make_sync_vector_env(
                gym_id,
                num_envs,
                render_mode=render_mode,
                **gym_kwargs,
            )
        return GymnasiumWrapper(vector_env=vec)


def make_sync_vector_env(
    gym_id: str,
    num_envs: int,
    *,
    render_mode: str | None = None,
    **gym_kwargs: Any,  # noqa: ANN401
) -> SyncVectorEnv:
    """Create a synchronous vectorized environment.

    Args:
        gym_id (str): Environment ID.
        num_envs (int): Number of parallel synchronized environments.
        render_mode (str | None): Rendering mode.
        **gym_kwargs (Any): Additional arguments passed to ``gym.make``.

    Returns:
        SyncVectorEnv: A synchronized vector environment.
    """

    def make_thunk() -> Callable[[], gym.Env]:
        def _thunk() -> gym.Env:
            return gym.make(gym_id, render_mode=render_mode, **gym_kwargs)

        return _thunk

    return SyncVectorEnv([make_thunk() for _ in range(num_envs)])


def make_async_vector_env(
    gym_id: str,
    num_envs: int,
    *,
    render_mode: str | None = None,
    **gym_kwargs: Any,  # noqa: ANN401
) -> AsyncVectorEnv:
    """Create an asynchronous vectorized environment.

    Args:
        gym_id (str): Environment ID.
        num_envs (int): Number of parallel async environments.
        render_mode (str | None): Rendering mode.
        **gym_kwargs (Any): Additional arguments passed to ``gym.make``.

    Returns:
        AsyncVectorEnv: An asynchronous vector environment.
    """

    def make_thunk() -> Callable[[], gym.Env]:
        def _thunk() -> gym.Env:
            return gym.make(gym_id, render_mode=render_mode, **gym_kwargs)

        return _thunk

    return AsyncVectorEnv([make_thunk() for _ in range(num_envs)])
