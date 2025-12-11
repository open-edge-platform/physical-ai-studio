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
    from pathlib import Path

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
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> Self:
        """Load a pretrained policy from HuggingFace Hub or local path.

        This method delegates to LeRobot's from_pretrained implementation and wraps
        the loaded policy in the appropriate getiaction wrapper. The policy is loaded
        with its trained weights and configuration.

        Args:
            pretrained_name_or_path: Model ID on HuggingFace Hub (e.g.,
                "lerobot/act_aloha_sim_transfer_cube_human") or path to local directory.
            force_download: Force download even if file exists in cache.
            resume_download: Resume incomplete downloads.
            proxies: Proxy configuration for downloads.
            token: HuggingFace authentication token.
            cache_dir: Directory to cache downloaded models.
            local_files_only: Only use local files, no downloads.
            revision: Model revision (branch, tag, or commit hash).
            **kwargs: Additional wrapper-specific arguments (e.g., learning_rate).

        Returns:
            Initialized policy wrapper with pretrained weights loaded.

        Raises:
            ImportError: If LeRobot is not installed.

        Examples:
            Load ACT model:
                >>> from getiaction.policies.lerobot import ACT
                >>> policy = ACT.from_pretrained(
                ...     "lerobot/act_aloha_sim_transfer_cube_human"
                ... )

            Load from local path:
                >>> policy = ACT.from_pretrained("/path/to/saved/model")

            Load with custom learning rate:
                >>> policy = ACT.from_pretrained(
                ...     "lerobot/act_aloha_sim_transfer_cube_human",
                ...     learning_rate=1e-4,
                ... )

        Note:
            The loaded policy is in eval mode by default. Use `policy.train()`
            to switch to training mode for fine-tuning.
        """
        try:
            from lerobot.configs.policies import PreTrainedConfig  # noqa: PLC0415
            from lerobot.policies.factory import get_policy_class  # noqa: PLC0415
        except ImportError as e:
            msg = (
                "LeRobot is required for from_pretrained functionality.\n\n"
                "Install with:\n"
                "    pip install lerobot\n\n"
                "Or install getiaction with LeRobot support:\n"
                "    pip install getiaction[lerobot]\n\n"
                "For more information, see: https://github.com/huggingface/lerobot"
            )
            raise ImportError(msg) from e

        # Load config to identify policy type
        config = PreTrainedConfig.from_pretrained(
            pretrained_name_or_path,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
        )

        # Get policy class and load pretrained weights
        policy_cls = get_policy_class(config.type)
        lerobot_policy = policy_cls.from_pretrained(
            pretrained_name_or_path,
            config=config,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
        )

        # Create wrapper instance without calling __init__
        wrapper: Any = cls.__new__(cls)

        # Initialize as PyTorch Module - we need to call the parent __init__
        # since we bypassed it with __new__. PyTorch Module.__init__ sets up
        # critical internal state (_modules, _parameters, etc.)
        super(cls, wrapper).__init__()

        # Set required attributes
        wrapper._is_pretrained = True  # noqa: SLF001
        wrapper._framework = "lerobot"  # noqa: SLF001
        # Use learning_rate from kwargs, or fall back to config's optimizer_lr
        # Config from pretrained models should have this; if not, user must provide it for training
        wrapper.learning_rate = kwargs.get("learning_rate", getattr(config, "optimizer_lr", None))

        # For LeRobotPolicy (universal wrapper), set policy_name
        if cls.__name__ == "LeRobotPolicy":
            wrapper.policy_name = config.type

        # Register the loaded policy
        wrapper.add_module("_lerobot_policy", lerobot_policy)

        # Expose model attribute if available
        if hasattr(lerobot_policy, "model"):
            wrapper.model = lerobot_policy.model
        elif hasattr(lerobot_policy, "diffusion"):
            wrapper.model = lerobot_policy.diffusion
        else:
            wrapper.model = None

        # Set to eval mode (LeRobot's from_pretrained sets policy to eval)
        wrapper.eval()

        return wrapper

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
