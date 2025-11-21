# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""GymnasiumWrapper: adapts any Gymnasium environment to the abstract Gym interface.

Note: if you want a GPU optimized gym, please implement your own.
We assume that this is numpy style gymnasium.
"""

from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

from getiaction.data.observation import Observation

from .base import Gym


class ActionValidationError(ValueError):
    """Invalid actions will raise a custom ValueErorr."""


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
        """Initializes the Gymnasium environment.

        Args:
            gym_id: Environment ID.
            vector_env: Preconstructed SyncVectorEnv or AsyncVectorEnv.
            device: Torch device for tensors.
            render_mode: Render mode passed to gym.make.
            **gym_kwargs: Additional arguments for gym.make.
        """
        if vector_env is not None:
            self._env = vector_env
        else:
            if render_mode is not None:
                gym_kwargs["render_mode"] = render_mode
            self._env = gym.make(gym_id, **gym_kwargs)

        self.device = torch.device(device)

        # vectorized environments
        self.num_envs = getattr(self._env, "num_envs", 1)
        self.is_vectorized = self.num_envs > 1

    @property
    def render_mode(self) -> str | None:
        """Returns the render mode.

        Returns:
            str or None
        """
        return getattr(self._env, "render_mode", None)

    @property
    def observation_space(self) -> gym.Space | None:
        """Returns the underlying observation space.

        Returns:
            gym.Space or None.
        """
        return getattr(self._env, "observation_space", None)

    @property
    def action_space(self) -> gym.Space | None:
        """Returns the underlying action space.

        Returns:
            gym.Space or None.
        """
        return getattr(self._env, "action_space", None)

    @staticmethod
    def vectorized(
        gym_id: str,
        num_envs: int,
        *,
        async_mode: bool = False,
        render_mode: str | None = "rgb_array",
        **kwargs: Any,
    ) -> "GymnasiumWrapper":
        if async_mode:
            vec = make_async_vector_env(gym_id, num_envs, render_mode=render_mode, **kwargs)
        else:
            vec = make_sync_vector_env(gym_id, num_envs, render_mode=render_mode, **kwargs)
        return GymnasiumWrapper(vector_env=vec)

    def reset(
        self,
        *,
        seed: int | None = None,
        **reset_kwargs: Any,  # noqa: ANN401
    ) -> tuple[Observation, dict[str, Any]]:
        """Resets the environment.

        Args:
            seed: Optional random seed.
            **reset_kwargs: Extra reset parameters.

        Returns:
            Tuple of (Observation, info dict).
        """
        raw_obs, info = self._env.reset(seed=seed, **reset_kwargs)
        obs = self.to_observation(raw_obs)
        return obs, info

    def validate_action(self, action: torch.Tensor) -> None:
        """Validates an action tensor against the action space.

        Args:
            action: Torch tensor action.

        Raises:
            ActionValidationError: If the action is invalid.
        """
        if not isinstance(action, torch.Tensor):
            msg = f"Action must be torch.Tensor, got {type(action)}"
            raise ActionValidationError(msg)

        space = self.action_space
        if space is None:
            msg = "Environment has no action_space defined."
            raise ActionValidationError(msg)

        if not self.is_vectorized:
            # allow either shape [dim] or [1, dim]
            if action.ndim == 2 and action.shape[0] == 1:
                # will squeeze later
                return
            if action.ndim != 1:
                msg = f"Single-env expects [dim] or [1,dim], got {action.shape}"
                raise ActionValidationError(
                    msg,
                )
        # vectorized requires [num_envs, dim]
        elif action.ndim != 2 or action.shape[0] != self.num_envs:
            msg = f"Vectorized env expects [num_envs, dim] = [{self.num_envs}, ...], got {action.shape}"
            raise ActionValidationError(
                msg,
            )

        if isinstance(space, gym.spaces.Discrete):
            if action.ndim != 0:
                msg = f"Discrete action must be scalar, got {action.shape}"
                raise ActionValidationError(msg)
            if action.dtype not in {torch.int64, torch.int32, torch.int16, torch.int8}:
                msg = f"Discrete action dtype must be integer, got {action.dtype}"
                raise ActionValidationError(msg)
            v = int(action.item())
            if not (0 <= v < space.n):
                msg = f"Discrete action {v} out of range [0, {space.n - 1}]"
                raise ActionValidationError(msg)
            return

        if isinstance(space, gym.spaces.MultiDiscrete):
            if action.shape != space.nvec.shape:
                msg = f"MultiDiscrete shape mismatch: expected {space.nvec.shape}, got {action.shape}"
                raise ActionValidationError(
                    msg,
                )
            if action.dtype not in {torch.int64, torch.int32, torch.int16, torch.int8}:
                msg = f"MultiDiscrete requires integer dtype, got {action.dtype}"
                raise ActionValidationError(msg)
            if (action < 0).any() or (action >= torch.as_tensor(space.nvec)).any():
                msg = f"MultiDiscrete action values exceed ranges {space.nvec}"
                raise ActionValidationError(msg)
            return

        if isinstance(space, gym.spaces.MultiBinary):
            if action.shape != (space.n,):
                msg = f"MultiBinary shape mismatch: expected {(space.n,)}, got {action.shape}"
                raise ActionValidationError(msg)
            if action.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64, torch.bool}:
                msg = f"MultiBinary requires int/bool dtype, got {action.dtype}"
                raise ActionValidationError(msg)
            if not (((action == 0) | (action == 1)).all()):
                msg = "MultiBinary must contain only 0/1 values"
                raise ActionValidationError(msg)
            return

        if isinstance(space, gym.spaces.Box):
            if action.shape != space.shape:
                msg = f"Box shape mismatch: expected {space.shape}, got {action.shape}"
                raise ActionValidationError(msg)
            if action.dtype not in {torch.float32, torch.float64, torch.float16}:
                msg = f"Box requires floating dtype, got {action.dtype}"
                raise ActionValidationError(msg)
            return

        if isinstance(space, gym.spaces.Tuple):
            msg = "Tuple action spaces require structured actions."
            raise ActionValidationError(msg)

        if isinstance(space, gym.spaces.Dict):
            msg = "Dict action spaces require dict-of-tensors."
            raise ActionValidationError(msg)

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """Performs one environment step.

        Args:
            action: Torch tensor action.

        Returns:
            Tuple of (Observation, reward, terminated, truncated, info).
        """
        self.validate_action(action)

        # Single env allow [dim] or [1,dim], but squeeze before .step()
        if not self.is_vectorized and action.ndim == 2 and action.shape[0] == 1:  # noqa: PLR2004
            action = action[0]

        raw_action = action.detach().cpu().numpy()
        raw_obs, reward, terminated, truncated, info = self._env.step(raw_action)
        obs = self.to_observation(raw_obs)
        return obs, reward, terminated, truncated, info

    def render(self, *args: Any, **kwargs: Any) -> Any:
        """Renders the environment.

        Returns:
            Rendered output or None.
        """
        if hasattr(self._env, "render"):
            return self._env.render(*args, **kwargs)
        return None

    def close(self) -> None:
        """Closes the environment."""
        self._env.close()

    def sample_action(self) -> torch.Tensor:
        """Samples a valid action.

        Returns:
            Torch tensor containing a sampled action.
        """
        a = self._env.action_space.sample()
        return torch.as_tensor(a, device=self.device)

    def get_max_episode_steps(self) -> int | None:
        """Returns the environment time limit if available.

        Returns:
            Maximum number of episode steps or None.
        """
        if hasattr(self._env, "get_wrapper_attr"):
            try:
                return self._env.get_wrapper_attr("max_episode_steps")
            except Exception:
                pass
        if hasattr(self._env, "max_episode_steps"):
            v = self._env.max_episode_steps
            return v() if callable(v) else v
        return None

    def to_observation(
        self,
        raw_obs: Any,
        camera_keys: list[str] | None = None,
    ) -> Observation:
        """Converts raw observations to unified Observation using instance settings.

        Args:
            raw_obs: Raw environment observation.
            camera_keys: Optional camera names.

        Returns:
            Observation.
        """
        return GymnasiumWrapper.convert_raw_to_observation(
            raw_obs=raw_obs,
            observation_space=self.observation_space,
            device=self.device,
            camera_keys=camera_keys,
            is_vectorized=self.is_vectorized,
        )

    @staticmethod
    def convert_raw_to_observation(
        raw_obs: Any,
        observation_space: gym.Space | None,
        device: torch.device,
        camera_keys: list[str] | None = None,
        is_vectorized: bool = False,
    ):
        """Converts raw observations into structured Observation objects.
        Handles both single and vectorized environments.
        """
        if camera_keys is None:
            camera_keys = ["top"]

        def to_tensor(x):
            if isinstance(x, torch.Tensor):
                return x.to(device)
            return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)

        # ---------------------------------------------------------
        # Image detection based on vectorization
        # ---------------------------------------------------------
        def is_image_tensor(t: torch.Tensor):
            shape = t.shape

            # non-vectorized: (H,W,C) or (C,H,W)
            if not is_vectorized:
                if len(shape) == 3:
                    return shape[0] in {1, 3, 4} or shape[-1] in {1, 3, 4}
                return False

            # vectorized: (B,H,W,C) or (B,C,H,W)
            if len(shape) == 4:
                return shape[1] in {1, 3, 4} or shape[-1] in {1, 3, 4}
            return False

        # ---------------------------------------------------------
        # 1. DICT ENV CASE
        # ---------------------------------------------------------
        if isinstance(raw_obs, dict):
            tdict = {k: to_tensor(v) for k, v in raw_obs.items()}
            image_dict, state_dict = {}, {}

            for k, v in tdict.items():
                if is_image_tensor(v):
                    # Normalize uint8
                    if v.dtype == torch.uint8:
                        v = v.float() / 255.0

                    # Non-vectorized: HWC → CHW
                    if not is_vectorized:
                        if v.ndim == 3 and v.shape[-1] in {1, 3, 4}:
                            v = v.permute(2, 0, 1)

                        # Add batch dim
                        if v.ndim == 3:
                            v = v.unsqueeze(0)

                    # Vectorized: BHWC → BCHW
                    elif v.ndim == 4 and v.shape[-1] in {1, 3, 4}:
                        v = v.permute(0, 3, 1, 2)

                    image_dict[k] = v

                else:
                    # State handling
                    if v.ndim == 1 and not is_vectorized:
                        v = v.unsqueeze(0)
                    state_dict[k] = v

            # build output
            images = image_dict or None

            if len(state_dict) == 1:
                state = list(state_dict.values())[0]
            else:
                state = state_dict or None

            return Observation(images=images, state=state)

        # ---------------------------------------------------------
        # 2. BOX SPACE (single obs)
        # ---------------------------------------------------------
        if isinstance(observation_space, gym.spaces.Box):
            v = to_tensor(raw_obs)

            if is_image_tensor(v):
                # Normalize
                if v.dtype == torch.uint8:
                    v = v.float() / 255.0

                # NON-VEC: HWC → CHW + batch
                if not is_vectorized:
                    if v.ndim == 3 and v.shape[-1] in {1, 3, 4}:
                        v = v.permute(2, 0, 1)
                    if v.ndim == 3:
                        v = v.unsqueeze(0)

                # VEC: BHWC → BCHW (no unsqueeze)
                elif v.ndim == 4 and v.shape[-1] in {1, 3, 4}:
                    v = v.permute(0, 3, 1, 2)

                return Observation(images={"default": v})

            # Otherwise a state
            if v.ndim == 1 and not is_vectorized:
                v = v.unsqueeze(0)

            return Observation(state=v)

        # ---------------------------------------------------------
        # 3. FALLBACK (scalar, etc.)
        # ---------------------------------------------------------
        v = to_tensor(raw_obs)
        if v.ndim == 0:
            v = v.unsqueeze(0).unsqueeze(0)

        return Observation(state=v)


def make_sync_vector_env(
    gym_id: str,
    num_envs: int,
    *,
    render_mode: str | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> SyncVectorEnv:
    """Creates a vectorized synchronous environment."""

    def make_thunk():
        def _thunk():
            return gym.make(gym_id, render_mode=render_mode, **kwargs)

        return _thunk

    return SyncVectorEnv([make_thunk() for _ in range(num_envs)])


def make_async_vector_env(
    gym_id: str,
    num_envs: int,
    *,
    render_mode: str | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> AsyncVectorEnv:
    """Creates a vectorized asynchronous environment."""

    def make_thunk():
        def _thunk():
            return gym.make(gym_id, render_mode=render_mode, **kwargs)

        return _thunk

    return AsyncVectorEnv([make_thunk() for _ in range(num_envs)])
