# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Configuration mixin specialized for LeRobot policies.

This module extends the base FromConfig mixin to handle LeRobot-specific
configuration patterns, particularly LeRobot's PreTrainedConfig dataclasses.
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import TYPE_CHECKING, Any, Self

from getiaction.config.mixin import FromConfig

if TYPE_CHECKING:
    from lerobot.configs.policies import PreTrainedConfig


class LeRobotFromConfig(FromConfig):
    """Extended FromConfig mixin for LeRobot policies.

    This mixin extends the base FromConfig functionality to support LeRobot's
    PreTrainedConfig dataclasses, which are used by all LeRobot policies.

    The key feature is the ability to pass a LeRobot config object directly
    to `from_config()`, which will be forwarded to the appropriate constructor
    parameter (either `lerobot_config` for explicit wrappers or `config` for
    the universal wrapper).

    Supported configuration formats:
        1. Dict: Standard dictionary of parameters
        2. YAML: YAML file with parameters
        3. Pydantic: Pydantic model
        4. Dataclass: Generic dataclass
        5. LeRobot PreTrainedConfig: LeRobot's config dataclasses (ACTConfig, DiffusionConfig, etc.)

    Examples:
        Using with explicit wrapper (ACT):
            >>> from getiaction.policies.lerobot import ACT
            >>> from lerobot.policies.act.configuration_act import ACTConfig

            >>> # Create LeRobot config
            >>> lerobot_config = ACTConfig(
            ...     dim_model=512,
            ...     chunk_size=100,
            ...     use_vae=True,
            ... )

            >>> # Use from_config to instantiate
            >>> policy = ACT.from_config(lerobot_config)
            >>> # Equivalent to: ACT(lerobot_config=lerobot_config)

        Using with universal wrapper (LeRobotPolicy):
            >>> from getiaction.policies.lerobot import LeRobotPolicy
            >>> from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig

            >>> # Create LeRobot config
            >>> lerobot_config = DiffusionConfig(
            ...     num_steps=100,
            ...     noise_scheduler="ddpm",
            ... )

            >>> # Use from_config to instantiate
            >>> policy = LeRobotPolicy.from_config(
            ...     policy_name="diffusion",
            ...     config=lerobot_config,
            ... )

        Mixed usage (dict + LeRobot config):
            >>> # Can also pass additional parameters
            >>> policy = ACT.from_dict({
            ...     "dim_model": 512,
            ...     "chunk_size": 100,
            ... })
    """

    @classmethod
    def from_lerobot_config(
        cls,
        config: PreTrainedConfig,
        **kwargs: Any,  # noqa: ANN401
    ) -> Self:
        """Create instance from a LeRobot PreTrainedConfig dataclass.

        This method handles LeRobot's configuration dataclasses (ACTConfig,
        DiffusionConfig, VQBeTConfig, etc.) and forwards them to the appropriate
        constructor parameter or unpacks them as kwargs.

        Args:
            config: LeRobot PreTrainedConfig instance (e.g., ACTConfig, DiffusionConfig).
            **kwargs: Additional parameters to pass to the constructor.
                For explicit wrappers (ACT, Diffusion), this might include learning_rate.
                For universal wrapper (LeRobotPolicy), this must include policy_name.

        Returns:
            An instance of the policy class.

        Raises:
            TypeError: If the class doesn't support LeRobot config.

        Examples:
            With explicit wrapper (ACT):
                >>> from lerobot.policies.act.configuration_act import ACTConfig
                >>> config = ACTConfig(dim_model=512, chunk_size=100)
                >>> policy = ACT.from_lerobot_config(config, learning_rate=1e-5)

            With universal wrapper:
                >>> from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
                >>> config = DiffusionConfig(num_steps=100)
                >>> policy = LeRobotPolicy.from_lerobot_config(
                ...     config,
                ...     policy_name="diffusion",
                ... )
        """
        # Check if the class has a 'config' parameter (universal wrapper pattern)
        sig = inspect.signature(cls.__init__)
        has_config_param = "config" in sig.parameters

        if has_config_param:
            # Try universal wrapper pattern (config= parameter)
            try:
                return cls(config=config, **kwargs)  # type: ignore[call-arg]
            except TypeError as e:
                # Config parameter exists but doesn't work with this config type
                msg = f"{cls.__name__} config parameter doesn't accept this config type: {e}"
                raise TypeError(msg) from e

        # Fall back to unpacking config as kwargs (explicit wrappers like ACT)
        if not dataclasses.is_dataclass(config):
            msg = f"Expected dataclass for explicit wrapper, got {type(config)}"
            raise TypeError(msg)

        try:
            # Convert config to dict
            config_dict = dataclasses.asdict(config)  # type: ignore[arg-type]

            # Filter to only parameters accepted by the constructor
            valid_params = set(sig.parameters.keys()) - {"self"}
            filtered_config = {k: v for k, v in config_dict.items() if k in valid_params}

            # Merge with kwargs (kwargs take precedence)
            filtered_config.update(kwargs)

            return cls(**filtered_config)  # type: ignore[arg-type]
        except TypeError as e:
            msg = f"{cls.__name__} does not support LeRobot config instantiation. Original error: {e}"
            raise TypeError(msg) from e

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any] | PreTrainedConfig | Any,  # noqa: ANN401
        *,
        key: str | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> Self:
        """Generic method to instantiate from any configuration format.

        This method extends the base FromConfig.from_config() to additionally
        support LeRobot's PreTrainedConfig dataclasses.

        Args:
            config: Configuration in any supported format:
                - dict: Parameter dictionary
                - str/Path: YAML file path
                - BaseModel: Pydantic model
                - dataclass: Generic dataclass
                - PreTrainedConfig: LeRobot config dataclass (NEW!)
            key: Optional key to extract a sub-configuration.
            **kwargs: Additional parameters passed to the constructor.

        Returns:
            An instance of the class.

        Examples:
            Auto-detect LeRobot config:
                >>> from lerobot.policies.act.configuration_act import ACTConfig
                >>> config = ACTConfig(dim_model=512)
                >>> policy = ACT.from_config(config)

            Auto-detect dict:
                >>> config = {"dim_model": 512, "chunk_size": 100}
                >>> policy = ACT.from_config(config)

            Auto-detect YAML:
                >>> policy = ACT.from_config("config.yaml")
        """
        # Check if it's a LeRobot PreTrainedConfig (dataclass with specific attributes)
        if (
            dataclasses.is_dataclass(config)
            and not isinstance(config, type)
            and hasattr(config, "input_features")
            and hasattr(config, "output_features")
        ):
            # This is likely a LeRobot PreTrainedConfig
            return cls.from_lerobot_config(config, **kwargs)  # type: ignore[arg-type]

        # Fall back to base FromConfig logic for other types
        return super().from_config(config, key=key, **kwargs)  # type: ignore[misc]

    @classmethod
    def from_dataclass(
        cls,
        config: object,
        *,
        key: str | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> Self:
        """Load configuration from a dataclass.

        This method extends the base from_dataclass() to handle LeRobot
        PreTrainedConfig dataclasses specially.

        Args:
            config: Dataclass instance (generic or LeRobot PreTrainedConfig).
            key: Optional key to extract a sub-configuration from the dataclass.
            **kwargs: Additional parameters passed to the constructor.

        Returns:
            An instance of the class.

        Raises:
            TypeError: If config is not a dataclass instance.

        Examples:
            Generic dataclass:
                >>> @dataclass
                >>> class Config:
                ...     dim_model: int = 512
                >>> policy = ACT.from_dataclass(Config())

            LeRobot config:
                >>> from lerobot.policies.act.configuration_act import ACTConfig
                >>> policy = ACT.from_dataclass(ACTConfig(dim_model=512))
        """
        if not dataclasses.is_dataclass(config):
            msg = f"Expected dataclass instance, got {type(config)}"
            raise TypeError(msg)

        # Check if it's a LeRobot PreTrainedConfig
        if hasattr(config, "input_features") and hasattr(config, "output_features"):
            return cls.from_lerobot_config(config, **kwargs)  # type: ignore[arg-type]

        # Fall back to base FromConfig logic
        return super().from_dataclass(config, key=key)  # type: ignore[misc]
