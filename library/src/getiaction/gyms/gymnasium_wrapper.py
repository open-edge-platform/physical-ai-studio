# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
GymnasiumWrapper: adapts any Gymnasium environment to the abstract Gym interface.

Note: if you want a GPU optimized gym, please implement your own.
We assume that this is numpy style gymnasium.
"""

import gymnasium as gym
import torch
import numpy as np
from typing import Any
from getiaction.data.observation import Observation
from .base import Gym


class ActionValidationError(ValueError):
    """Invalid actions will raise a custom ValueErorr"""
    pass


class GymnasiumWrapper(Gym):
    """Adapter that makes a Gymnasium environment conform to the unified Gym API."""

    def __init__(
        self,
        gym_id: str,
        *,
        device: str | torch.device = "cpu",
        render_mode: str | None = "rgb_array",
        **kwargs: Any,
    ) -> None:
        """Initializes the Gymnasium environment.

        Args:
            gym_id: Environment ID.
            device: Torch device for tensors.
            render_mode: Render mode passed to gym.make.
            **kwargs: Additional arguments for gym.make.
        """
        if render_mode is not None:
            kwargs["render_mode"] = render_mode
        self._env: gym.Env = gym.make(gym_id, **kwargs)
        self.device = torch.device(device)
    
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

    def reset(
        self,
        *,
        seed: int | None = None,
        **reset_kwargs: Any,
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
            raise ActionValidationError(f"Action must be torch.Tensor, got {type(action)}")

        space = self.action_space
        if space is None:
            raise ActionValidationError("Environment has no action_space defined.")

        if isinstance(space, gym.spaces.Discrete):
            if action.ndim != 0:
                raise ActionValidationError(f"Discrete action must be scalar, got {action.shape}")
            if action.dtype not in (torch.int64, torch.int32, torch.int16, torch.int8):
                raise ActionValidationError(f"Discrete action dtype must be integer, got {action.dtype}")
            v = int(action.item())
            if not (0 <= v < space.n):
                raise ActionValidationError(f"Discrete action {v} out of range [0, {space.n-1}]")
            return

        if isinstance(space, gym.spaces.MultiDiscrete):
            if action.shape != space.nvec.shape:
                raise ActionValidationError(f"MultiDiscrete shape mismatch: expected {space.nvec.shape}, got {action.shape}")
            if action.dtype not in (torch.int64, torch.int32, torch.int16, torch.int8):
                raise ActionValidationError(f"MultiDiscrete requires integer dtype, got {action.dtype}")
            if (action < 0).any() or (action >= torch.as_tensor(space.nvec)).any():
                raise ActionValidationError(f"MultiDiscrete action values exceed ranges {space.nvec}")
            return

        if isinstance(space, gym.spaces.MultiBinary):
            if action.shape != (space.n,):
                raise ActionValidationError(f"MultiBinary shape mismatch: expected {(space.n,)}, got {action.shape}")
            if action.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.bool):
                raise ActionValidationError(f"MultiBinary requires int/bool dtype, got {action.dtype}")
            if not (((action == 0) | (action == 1)).all()):
                raise ActionValidationError("MultiBinary must contain only 0/1 values")
            return

        if isinstance(space, gym.spaces.Box):
            if action.shape != space.shape:
                raise ActionValidationError(f"Box shape mismatch: expected {space.shape}, got {action.shape}")
            if action.dtype not in (torch.float32, torch.float64, torch.float16):
                raise ActionValidationError(f"Box requires floating dtype, got {action.dtype}")
            return

        if isinstance(space, gym.spaces.Tuple):
            raise ActionValidationError("Tuple action spaces require structured actions.")

        if isinstance(space, gym.spaces.Dict):
            raise ActionValidationError("Dict action spaces require dict-of-tensors.")

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
        raw_action = action.detach().cpu().numpy()
        raw_obs, reward, terminated, truncated, info = self._env.step(raw_action)
        obs = self.to_observation(raw_obs)
        return obs, float(reward), bool(terminated), bool(truncated), info

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
            v = getattr(self._env, "max_episode_steps")
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
        )

    @staticmethod
    def convert_raw_to_observation(
        raw_obs: Any,
        observation_space: gym.Space | None,
        device: torch.device,
        camera_keys: list[str] | None = None,
    ) -> Observation:
        """Converts a raw observation using explicit device and observation space.

        Args:
            raw_obs: Raw observation.
            observation_space: Gym space describing the observation.
            device: Torch device.
            camera_keys: Optional camera names.

        Returns:
            Observation.
        """
        if camera_keys is None:
            camera_keys = ["top"]

        def to_tensor(x):
            if isinstance(x, torch.Tensor):
                return x.to(device)
            return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)

        def is_image_shape(shape):
            return len(shape) == 3 and (shape[0] in {1, 3, 4} or shape[-1] in {1, 3, 4})

        images = None
        state = None

        if isinstance(raw_obs, dict):
            tdict = {k: to_tensor(raw_obs[k]) for k in raw_obs.keys()}
            image_dict, state_dict = {}, {}

            for k, v in tdict.items():
                if isinstance(v, torch.Tensor) and is_image_shape(v.shape):
                    if v.dtype == torch.uint8:
                        v = v.float() / 255.0
                    if v.shape[-1] in {1, 3, 4}:
                        v = v.permute(2, 0, 1)
                    v = v.unsqueeze(0)
                    image_dict[k] = v
                else:
                    if v.ndim == 1:
                        v = v.unsqueeze(0)
                    state_dict[k] = v

            images = image_dict or None
            if len(state_dict) == 1:
                state = list(state_dict.values())[0]
            elif len(state_dict) > 1:
                state = state_dict
            return Observation(images=images, state=state)

        if isinstance(observation_space, gym.spaces.Box):
            v = to_tensor(raw_obs)
            if is_image_shape(observation_space.shape):
                if v.dtype == torch.uint8:
                    v = v.float() / 255.0
                if v.shape[-1] in {1, 3, 4}:
                    v = v.permute(2, 0, 1)
                v = v.unsqueeze(0)
                return Observation(images={"default": v})
            if v.ndim == 1:
                v = v.unsqueeze(0)
            return Observation(state=v)

        v = to_tensor(raw_obs)
        if v.ndim == 0:
            v = v.unsqueeze(0).unsqueeze(0)
        return Observation(state=v)
