# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base class for gym environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

    from getiaction.data import Observation


class Gym(ABC):
    """Abstract interface for Gymnasium-style environments.

    This class defines a unified environment API used across different
    simulators, without assuming a specific backend (e.g., Gymnasium, DMC,
    IsaacGym, custom robotics simulators).

    We make no assumption of underlying env capabilities.
    """

    @abstractmethod
    def reset(
        self,
        *,
        seed: int | None = None,
        **reset_kwargs: Any,  # noqa: ANN401
    ) -> tuple[Observation, dict[str, Any]]:
        """Resets the environment.

        Args:
            seed: Optional random seed for resetting the environment.
            **reset_kwargs: Additional backend-specific reset arguments.

        Returns:
            tuple[Observation, dict[str, Any]]:
                - Observation: Environment observation after reset.
                - dict[str, Any]: Additional info dictionary.
        """

    @abstractmethod
    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """Steps the environment by one action.

        Args:
            action: Action as a torch tensor, already preprocessed for the backend.

        Returns:
            tuple[Observation, float, bool, bool, dict[str, Any]]:
                - Observation: Next environment observation.
                - float: Reward for this transition.
                - bool: Whether the episode terminated.
                - bool: Whether the episode was truncated (e.g., time limit).
                - dict[str, Any]: Additional environment info.
        """

    def render(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401, ARG002, PLR6301
        """Renders the environment if supported.

        Args:
            *args: Backend-specific render args.
            **kwargs: Backend-specific render kwargs.

        Returns:
            Any: Rendered output, or None if unsupported.
        """
        return None

    @abstractmethod
    def close(self) -> None:
        """Closes the environment and releases resources."""

    @abstractmethod
    def sample_action(self) -> torch.Tensor:
        """Samples a valid action.

        Returns:
            torch.Tensor: An action in the correct format for `step()`.
        """

    @abstractmethod
    def to_observation(
        self,
        raw_obs: Any,  # noqa: ANN401
        camera_keys: list[str] | None = None,
    ) -> Observation:
        """Converts a raw backend observation to a unified Observation.

        Args:
            raw_obs: Raw observation object from the simulator backend.
            camera_keys: Optional list of camera names to extract image data.

        Returns:
            Observation: Standardized Observation dataclass representation.
        """
