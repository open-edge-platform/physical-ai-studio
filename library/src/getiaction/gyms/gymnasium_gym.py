# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""GymnasiumGym: adapts any Gymnasium environment to the abstract Gym interface.

Note:
    This wrapper assumes NumPy Gymnasium environments.
    This wrapper is intended to work nicely with gymnasium supported by https://gymnasium.farama.org/.
    If you want a GPU-optimized gymnasium or have a custom gymnasium style environment,
    please implement your own.
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
from .types import SingleOrBatch

logger = logging.getLogger(__name__)


class ActionValidationError(ValueError):
    """Error raised when an invalid action is provided."""


class GymnasiumGym(Gym):
    """Adapter that makes a Gymnasium environment conform to the unified Gym API."""

    def __init__(
        self,
        gym_id: str | None = None,
        vector_env: gym.Env | None = None,
        device: str | torch.device = "cpu",
        render_mode: str | None = "rgb_array",
        **gym_kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize a Gymnasium environment.

        Args:
            gym_id: The environment ID given to ``gym.make``.
            vector_env: A preconstructed vectorized environment.
            device: Torch device used for returned tensors.
            render_mode: Rendering mode passed to ``gym.make``.
            **gym_kwargs: Additional arguments forwarded to ``gym.make``.
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
        """Return the configured torch device."""
        return self._device

    @property
    def is_vectorized(self) -> bool:
        """Return whether the environment is vectorized."""
        return self._is_vectorized

    @property
    def render_mode(self) -> str | None:
        """Return the underlying render mode."""
        return getattr(self._env, "render_mode", None)

    @property
    def observation_space(self) -> gym.Space | None:
        """Return the observation space."""
        return getattr(self._env, "observation_space", None)

    @property
    def action_space(self) -> gym.Space | None:
        """Return the action space."""
        return getattr(self._env, "action_space", None)

    def _normalize_raw_obs(
        self,
        raw_obs: np.ndarray | dict[str, Any],
    ) -> np.ndarray | dict[str, Any]:
        """Normalize raw observations into consistent batch format.

        Args:
            raw_obs: Observation from Gym.

        Returns:
            Batched observation where array data follows:
            * unvectorized → shape [1, ...]
            * vectorized scalar → shape [B, 1]
        """
        if not self.is_vectorized:
            if isinstance(raw_obs, dict):
                return {k: np.expand_dims(v, 0) for k, v in raw_obs.items()}
            return np.expand_dims(raw_obs, 0)

        if isinstance(raw_obs, dict):
            out: dict[str, Any] = {}
            for k, v in raw_obs.items():
                if isinstance(v, np.ndarray) and v.ndim == 1:
                    out[k] = np.expand_dims(v, 1)
                else:
                    out[k] = v
            return out

        if isinstance(raw_obs, np.ndarray) and raw_obs.ndim == 1:
            return np.expand_dims(raw_obs, 1)

        return raw_obs

    def _normalize_action_for_env(
        self,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize user-provided actions for passing to the environment.

        Args:
            action: Batched user action tensor.

        Returns:
            Action tensor in the format expected by Gym:
            * unvectorized: dim
            * vectorized: [B] or [B, dim]
        """
        if not self.is_vectorized:
            # when we are not vectorized we assume [1, dim]
            # environment prefers dim, squeeze to match.
            if action.ndim == 2 and action.shape[0] == 1:  # noqa: PLR2004
                return action[0]
            return action

        if action.ndim == 2 and action.shape[1] == 1:  # noqa: PLR2004
            # when we are vectorized, we may have [B, 1] where we only have 1 action.
            # in this case we just want B,. Example is cartpole with action dimension of 1.
            # This is MultiDiscrete Action space.
            return action.squeeze(1)
        return action

    def _normalize_action_for_user(
        self,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize actions returned to the user.

        Args:
            action: Raw sampled action.

        Returns:
            Batched actions with shape:
            * unvectorized: [1, dim]
            * vectorized: [B, dim] or [B, 1]
        """
        if not self.is_vectorized:
            # if we are not vectorized, we havae to unsqueeze to meet [B, dim]
            if action.ndim == 0:
                # if scalar we need to unsqueze twice, [1, 1]
                return action.unsqueeze(0).unsqueeze(0)
            if action.ndim == 1:
                # if [dim], need [1, 1]
                return action.unsqueeze(0)
            return action

        # if vectorized and only one action dim i.e [B]
        # convert to [B, 1] for clarity.
        if action.ndim == 1:
            return action.unsqueeze(1)
        return action

    def reset(
        self,
        *,
        seed: int | None = None,
        **reset_kwargs: Any,  # noqa: ANN401
    ) -> tuple[Observation, dict[str, Any] | list[dict[str, Any]]]:
        """Reset the environment.

        Args:
            seed: Optional seed.
            **reset_kwargs: Additional reset arguments.

        Returns:
            A tuple ``(Observation, info)``.
        """
        raw_obs, info = self._env.reset(seed=seed, **reset_kwargs)
        raw_obs = self._normalize_raw_obs(raw_obs)
        obs = self.to_observation(raw_obs)
        return obs, info

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[
        Observation,
        SingleOrBatch[float],
        SingleOrBatch[bool],
        SingleOrBatch[bool],
        dict[str, Any] | list[dict[str, Any]],
    ]:
        """Step the environment.

        Args:
            action: Batched action tensor with shape ``[1, dim]`` or ``[B, dim]``.

        Returns:
            A tuple ``(Observation, reward, terminated, truncated, info)``.
        """
        action_for_env = self._normalize_action_for_env(action)
        raw_action = action_for_env.detach().cpu().numpy()

        raw_obs, reward, terminated, truncated, info = self._env.step(raw_action)
        raw_obs = self._normalize_raw_obs(raw_obs)
        obs = self.to_observation(raw_obs)

        return obs, reward, terminated, truncated, info

    def render(self, *render_args: Any, **render_kwargs: Any) -> Any:  # noqa: ANN401
        """Renders the environment.

        Args:
            *render_args (Any): Positional arguments forwarded to the underlying
                render implementation.
            **render_kwargs (Any): Keyword arguments forwarded to the render
                implementation.

        Returns:
            Any: The render output, if provided by the environment.
        """
        if hasattr(self._env, "render"):
            return self._env.render(*render_args, **render_kwargs)
        return None

    def close(self) -> None:
        """Close the environment."""
        self._env.close()

    def sample_action(self) -> torch.Tensor:
        """Sample a valid action in normalized batched format.

        Returns:
            A tensor with shape ``[1, dim]`` or ``[B, dim]``.
        """
        a = self._env.action_space.sample()
        t = torch.as_tensor(a, device=self.device)
        return self._normalize_action_for_user(t)

    def get_max_episode_steps(self) -> int | None:
        """Return the max episode steps if available."""
        if hasattr(self._env, "get_wrapper_attr"):
            try:
                return self._env.get_wrapper_attr("max_episode_steps")
            except AttributeError:
                logger.debug(
                    "get_wrapper_attr('max_episode_steps') not found on %r", self._env
                )
        return None

    def to_observation(
        self,
        raw_obs: np.ndarray | dict[str, Any],
    ) -> Observation:
        """Converts a raw environment observation into an `Observation` instance.

        Args:
            raw_obs (np.ndarray | dict[str, Any]): The unprocessed observation from
                the environment.

        Returns:
            Observation: The parsed and structured observation.
        """
        return self.convert_raw_to_observation(raw_obs=raw_obs).to_torch(
            device=self.device
        )

    @staticmethod
    def convert_raw_to_observation(
        raw_obs: np.ndarray | dict[str, Any],
    ) -> Observation:
        """Converts a Gymnasium raw observation into an `Observation` instance.

        Args:
            raw_obs (np.ndarray | dict[str, Any]): Raw observation, either a
                NumPy array or a dict.

        Returns:
            Observation: The normalized observation.

        Notes:
            * Non-dict inputs are treated as the `state` field.
            * Dicts matching `Observation` fields are passed directly to
              `Observation.from_dict`.
            * Other dicts are split into `images`, `state`, and `extra` based on
              key naming conventions.
        """
        if not isinstance(raw_obs, dict):
            return Observation(state=raw_obs).to_torch()

        obs_fields = {"action", "task", "state", "images"}
        if any(k in raw_obs for k in obs_fields):
            return Observation.from_dict(raw_obs)

        images: dict[str, Any] = {}
        state: dict[str, Any] = {}
        extra: dict[str, Any] = {}

        for key, value in raw_obs.items():
            key_lower = key.lower()

            arr = value
            if isinstance(arr, list):
                arr = np.asarray(arr)
            is_img_name = any(
                tok in key_lower
                for tok in ("pixel", "pixels", "image", "rgb", "camera")
            )
            is_image_like = (
                isinstance(arr, np.ndarray)
                and arr.ndim >= 3
                and arr.shape[-1] in {1, 3, 4}
            )  # noqa: PLR2004

            if is_img_name:
                if is_image_like:
                    if arr.ndim == 3:  # noqa: PLR2004
                        arr = np.transpose(arr, (2, 0, 1))
                    elif arr.ndim == 4:  # noqa: PLR2004
                        arr = np.transpose(arr, (0, 3, 1, 2))
                images[key] = arr
                continue

            if any(
                tok in key_lower
                for tok in ("pos", "agent_pos", "state", "obs", "feature")
            ):
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
    ) -> "GymnasiumGym":
        """Creates a vectorized `GymnasiumWrapper` for parallel environments.

        Args:
            gym_id (str): Gymnasium environment ID.
            num_envs (int): Number of environments to create.
            async_mode (bool, optional): Whether to run environments asynchronously.
                Defaults to False.
            render_mode (str | None, optional): Render mode for the environment.
                Defaults to `"rgb_array"`.
            **gym_kwargs (Any): Additional arguments passed to `gym.make`.

        Returns:
            GymnasiumWrapper: A vectorized environment wrapper.
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
        return GymnasiumGym(vector_env=vec)


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
