# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Configuration mixin specialized for LeRobot policies.

This module extends the base FromConfig mixin to handle LeRobot-specific
configuration patterns, particularly LeRobot's PreTrainedConfig dataclasses.
"""

from __future__ import annotations

import dataclasses
import inspect
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

from physicalai.config.mixin import FromConfig

if TYPE_CHECKING:
    from lerobot.configs.policies import PreTrainedConfig

logger = logging.getLogger(__name__)

_PROCESSOR_CONFIG_FILENAME = "policy_preprocessor.json"


def _has_processor_file(pretrained_name_or_path: str) -> bool:
    """Check whether a model directory or Hub repo contains a processor config.

    Returns:
        True if ``policy_preprocessor.json`` exists at the given path or Hub repo.
    """
    local_path = Path(pretrained_name_or_path)
    if local_path.is_dir():
        return (local_path / _PROCESSOR_CONFIG_FILENAME).exists()

    # Hub repo — lightweight HEAD request via huggingface_hub
    try:
        from huggingface_hub import file_exists  # noqa: PLC0415

        return file_exists(pretrained_name_or_path, _PROCESSOR_CONFIG_FILENAME)
    except Exception:  # noqa: BLE001
        logger.debug("Could not probe Hub for %s; assuming no processor config", pretrained_name_or_path)
        return False


class LeRobotFromConfig(FromConfig):
    """Extended FromConfig mixin for LeRobot policies.

    This mixin extends the base FromConfig functionality to support LeRobot's
    PreTrainedConfig dataclasses, which are used by all LeRobot policies.

    The key feature is the ability to pass a LeRobot ``PreTrainedConfig``
    instance (``ACTConfig``, ``DiffusionConfig``, …) directly to
    :meth:`from_config`. Every LeRobot wrapper -- the dynamic
    :class:`~physicalai.policies.lerobot.policy.LeRobotPolicy` and each
    :class:`~physicalai.policies.lerobot.policy.NamedLeRobotPolicy` alias --
    exposes a ``config=`` parameter that consumes the dataclass as-is, so a
    single forwarding rule works uniformly across the family.

    Supported configuration formats:
        1. Dict: Standard dictionary of parameters
        2. YAML: YAML file with parameters
        3. Pydantic: Pydantic model
        4. Dataclass: Generic dataclass
        5. LeRobot PreTrainedConfig: LeRobot's config dataclasses (ACTConfig, DiffusionConfig, etc.)

    Examples:
        Using a named alias (binds ``policy_name`` automatically):
            >>> from physicalai.policies.lerobot import ACT
            >>> from lerobot.policies.act.configuration_act import ACTConfig

            >>> lerobot_config = ACTConfig(
            ...     dim_model=512,
            ...     chunk_size=100,
            ...     use_vae=True,
            ... )

            >>> policy = ACT.from_config(lerobot_config)
            >>> # Equivalent to: ACT(config=lerobot_config)

        Using the dynamic ``LeRobotPolicy`` (must supply ``policy_name``):
            >>> from physicalai.policies.lerobot import LeRobotPolicy
            >>> from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig

            >>> lerobot_config = DiffusionConfig(
            ...     num_steps=100,
            ...     noise_scheduler="ddpm",
            ... )

            >>> policy = LeRobotPolicy.from_config(
            ...     lerobot_config,
            ...     policy_name="diffusion",
            ... )

        Mixed usage (dict of overrides):
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

        Forwards a LeRobot configuration dataclass (``ACTConfig``,
        ``DiffusionConfig``, ``VQBeTConfig``, …) to the wrapper's
        constructor via its ``config=`` parameter. Every LeRobot wrapper
        accepts the dataclass as-is, so the same forwarding rule applies
        uniformly to the dynamic :class:`LeRobotPolicy` and to every
        :class:`NamedLeRobotPolicy` alias.

        For non-LeRobot subclasses that may inherit this mixin without
        exposing a ``config=`` parameter, the implementation falls back to
        unpacking the dataclass fields as kwargs and filtering against the
        constructor signature. This branch is defensive: every wrapper
        currently shipped accepts ``config=``.

        Args:
            config: LeRobot PreTrainedConfig instance (e.g., ACTConfig, DiffusionConfig).
            **kwargs: Additional parameters to pass to the constructor.
                For named LeRobot wrappers (ACT, Diffusion, …), this might
                include ``optimizer_lr`` to override the config's learning rate.
                For the bare ``LeRobotPolicy`` wrapper, this must include
                ``policy_name``.

        Returns:
            An instance of the policy class.

        Raises:
            TypeError: If the class doesn't support LeRobot config.

        Examples:
            With a named alias (ACT):
                >>> from lerobot.policies.act.configuration_act import ACTConfig
                >>> config = ACTConfig(dim_model=512, chunk_size=100)
                >>> policy = ACT.from_lerobot_config(config, optimizer_lr=1e-5)

            With the dynamic LeRobotPolicy:
                >>> from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
                >>> config = DiffusionConfig(num_steps=100)
                >>> policy = LeRobotPolicy.from_lerobot_config(
                ...     config,
                ...     policy_name="diffusion",
                ... )
        """
        # All shipped LeRobot wrappers expose ``config=`` (defined on
        # ``LeRobotPolicy.__init__`` and inherited by every
        # ``NamedLeRobotPolicy`` alias). Forward directly when available.
        sig = inspect.signature(cls.__init__)
        has_config_param = "config" in sig.parameters

        if has_config_param:
            try:
                return cls(config=config, **kwargs)  # type: ignore[call-arg]
            except TypeError as e:
                msg = f"{cls.__name__} config parameter doesn't accept this config type: {e}"
                raise TypeError(msg) from e

        # Defensive fallback: a subclass that inherited this mixin but
        # does not expose ``config=`` -- unpack the dataclass into kwargs
        # filtered against the constructor signature.
        if not dataclasses.is_dataclass(config):
            msg = f"Expected dataclass instance, got {type(config)}"
            raise TypeError(msg)

        try:
            config_dict = dataclasses.asdict(config)  # type: ignore[arg-type]
            valid_params = set(sig.parameters.keys()) - {"self"}
            filtered_config = {k: v for k, v in config_dict.items() if k in valid_params}
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
        the loaded policy in the appropriate physicalai wrapper. The policy is loaded
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
            **kwargs: Additional wrapper-specific arguments. Pass
                ``optimizer_lr=<float>`` to override the loaded config's
                learning rate (used by ``configure_optimizers`` for
                fine-tuning).

        Returns:
            Initialized policy wrapper with pretrained weights loaded.

        Raises:
            ImportError: If LeRobot is not installed.
            AttributeError: If ``optimizer_lr`` is supplied but the loaded
                config dataclass does not expose an ``optimizer_lr`` field.

        Examples:
            Load ACT model:
                >>> from physicalai.policies.lerobot import ACT
                >>> policy = ACT.from_pretrained(
                ...     "lerobot/act_aloha_sim_transfer_cube_human"
                ... )

            Load from local path:
                >>> policy = ACT.from_pretrained("/path/to/saved/model")

            Load with custom learning rate for fine-tuning:
                >>> policy = ACT.from_pretrained(
                ...     "lerobot/act_aloha_sim_transfer_cube_human",
                ...     optimizer_lr=1e-4,
                ... )

        Note:
            The loaded policy is in eval mode by default. Use `policy.train()`
            to switch to training mode for fine-tuning.
        """
        try:
            from lerobot.configs.policies import PreTrainedConfig  # noqa: PLC0415
            from lerobot.policies.factory import get_policy_class, make_pre_post_processors  # noqa: PLC0415
        except ImportError as e:
            msg = (
                "LeRobot is required for from_pretrained functionality.\n\n"
                "Install with:\n"
                "    pip install lerobot\n\n"
                "Or install physicalai with LeRobot support:\n"
                "    pip install physicalai-train[lerobot]\n\n"
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

        # Apply optimizer learning-rate override BEFORE constructing the
        # wrapper so configure_optimizers picks it up. LeRobot's config
        # field is ``optimizer_lr``.
        optimizer_lr_override = kwargs.pop("optimizer_lr", None)
        if optimizer_lr_override is not None:
            if not hasattr(config, "optimizer_lr"):
                msg = (
                    f"Cannot override optimizer_lr on {type(config).__name__}: "
                    "this LeRobot config does not expose an optimizer_lr field."
                )
                raise AttributeError(msg)
            config.optimizer_lr = optimizer_lr_override

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

        # Initialize Lightning internals, action queue, and rollout metrics
        # without requiring the full LeRobotPolicy.__init__ arguments.
        wrapper._init_pretrained_shell()  # noqa: SLF001

        # Set required attributes
        wrapper._is_pretrained = True  # noqa: SLF001
        wrapper._framework = "lerobot"  # noqa: SLF001
        wrapper._config = config  # noqa: SLF001

        # Set policy_name from the loaded config (needed by all wrapper types)
        wrapper.policy_name = config.type

        # Register the loaded policy
        wrapper.add_module("_lerobot_policy", lerobot_policy)

        from physicalai.devices import get_available_device  # noqa: PLC0415

        device = get_available_device()
        device_overrides = {
            "preprocessor_overrides": {"device_processor": {"device": device}},
            "postprocessor_overrides": {"device_processor": {"device": device}},
        }

        # Probe whether the model ships a saved processor config.
        # Models saved with lerobot ≥ 0.5.1 include policy_preprocessor.json;
        # older Hub models don't — fall back to building from config only.
        pretrained_path = str(pretrained_name_or_path)
        has_processor_config = _has_processor_file(pretrained_path)

        wrapper._preprocessor, wrapper._postprocessor = make_pre_post_processors(  # noqa: SLF001
            config,
            **({"pretrained_path": pretrained_path} if has_processor_config else {}),
            **device_overrides,
        )

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
