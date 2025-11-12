# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Configuration mixin specialized for LeRobot policies.

This module extends the base FromConfig mixin to handle LeRobot-specific
configuration patterns, particularly LeRobot's PreTrainedConfig dataclasses.
"""

from __future__ import annotations

import dataclasses
import inspect
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import openvino
import torch

from getiaction.config.mixin import FromConfig
from getiaction.export import Export, ExportBackend
from getiaction.export.mixin_export import _postprocess_openvino_model

if TYPE_CHECKING:
    from os import PathLike

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


class LeRobotExport(Export):
    """Generic export mixin for LeRobot policies.

    This mixin provides common export functionality for all LeRobot policies,
    including ONNX, OpenVINO, and TorchExportIR backends. It handles:
    - Proper forward() method wrapping for export
    - Metadata generation with policy-specific parameters
    - ONNX→OpenVINO conversion to avoid TorchScript issues
    - Sample input generation based on policy configuration

    Policy-specific classes (ACT, Diffusion, etc.) can:
    1. Use this mixin as-is for standard export behavior
    2. Override specific methods if custom logic is needed
    3. Override metadata_extra property to add policy-specific metadata

    Examples:
        Basic usage in a policy:
            >>> from getiaction.export import Export
            >>> from getiaction.policies.base import Policy
            >>> from getiaction.policies.lerobot.mixin import LeRobotFromConfig, LeRobotExport

            >>> class MyLeRobotPolicy(Export, Policy, LeRobotFromConfig, LeRobotExport):
            ...     # Policy implementation...
            ...     pass

        With custom metadata:
            >>> class ACT(Export, Policy, LeRobotFromConfig, LeRobotExport):
            ...     @property
            ...     def metadata_extra(self) -> dict[str, Any]:
            ...         '''Add ACT-specific metadata.'''
            ...         return {
            ...             "chunk_size": self.lerobot_policy.config.chunk_size,
            ...             "use_action_queue": True,
            ...         }

        With custom sample input generation:
            >>> class CustomPolicy(Export, Policy, LeRobotFromConfig, LeRobotExport):
            ...     @property
            ...     def sample_input(self) -> dict[str, torch.Tensor]:
            ...         '''Generate custom sample input.'''
            ...         # Custom logic here...
            ...         return sample_dict
    """

    if TYPE_CHECKING:
        # Tell type checkers that self will have these attributes/methods from the host class
        lerobot_policy: Any
        eval: Any  # Callable[[], Self]
        _prepare_export_path: Any  # Callable[[PathLike | str, str], Path]
        _get_export_extra_args: Any  # Callable[[ExportBackend], dict[str, Any]]
        _get_forward_arg_name: Any  # Callable[[], str]
        _create_metadata: Any  # Callable[[Path, ExportBackend, ...], None]

    @property
    def metadata_extra(self) -> dict[str, Any]:
        """Get policy-specific metadata for export.

        Override this method in subclasses to add policy-specific metadata
        (e.g., chunk_size for ACT, num_steps for Diffusion).

        Returns:
            Dictionary of policy-specific metadata to include in export.
        """
        return {}

    @property
    def sample_input(self) -> dict[str, torch.Tensor]:
        """Generate sample input for model export.

        Creates sample tensors matching the format expected by LeRobot policies.
        For policies with n_obs_steps > 1, includes temporal dimension.

        Override this method if your policy requires custom input format.

        Returns:
            Dictionary containing sample tensors for model tracing/export.
        """
        config = self.lerobot_policy.config
        batch_size = 1

        # Create sample inputs based on policy configuration
        sample = {}

        # Add observation.state if robot state is used
        # Match LeRobot's expected format: no temporal dim for n_obs_steps==1
        if config.robot_state_feature and "observation.state" in config.input_features:
            state_dim = config.input_features["observation.state"].shape[0]
            sample["observation.state"] = torch.randn(batch_size, state_dim)

        # Add observation.images if image features are used
        # Match LeRobot's expected format: no temporal dim for n_obs_steps==1
        if config.image_features:
            for img_key in config.image_features:
                if img_key in config.input_features:
                    img_shape = config.input_features[img_key].shape  # (C, H, W)
                    sample[img_key] = torch.randn(batch_size, *img_shape)

        return sample

    def to_onnx(
        self,
        output_path: PathLike | str,
        input_sample: dict[str, torch.Tensor] | None = None,
        **export_kwargs: dict,
    ) -> None:
        """Export to ONNX format.

        For LeRobot policies, exports the wrapper (self) instead of the underlying model
        to ensure proper forward() behavior during export.

        Args:
            output_path: Path to save the ONNX model.
            input_sample: Optional sample input. If None, uses self.sample_input.
            **export_kwargs: Additional arguments passed to torch.onnx.export.
        """
        if input_sample is None:
            input_sample = self.sample_input

        model_path = self._prepare_export_path(output_path, ".onnx")
        export_dir = model_path.parent

        extra_model_args = self._get_export_extra_args(ExportBackend.ONNX)
        extra_model_args.update(export_kwargs)

        arg_name = self._get_forward_arg_name()

        # Export the wrapper (self) instead of self.model to use wrapper's forward()
        self.eval()
        torch.onnx.export(
            self,  # type: ignore[arg-type]  # Export wrapper, not self.model
            args=(),
            kwargs={arg_name: input_sample},
            f=str(model_path),
            input_names=list(input_sample.keys()),
            **extra_model_args,
        )

        # Create metadata files with policy-specific info
        self._create_metadata(export_dir, ExportBackend.ONNX, **self.metadata_extra)

    def to_openvino(
        self,
        output_path: PathLike | str,
        input_sample: dict[str, torch.Tensor] | None = None,
        **export_kwargs: dict,
    ) -> None:
        """Export to OpenVINO format.

        For LeRobot policies, this converts to ONNX first, then to OpenVINO.
        This approach avoids TorchScript tracing issues with stateful models (action queue).

        Args:
            output_path: Path to save the OpenVINO model.
            input_sample: Optional sample input. If None, uses self.sample_input.
            **export_kwargs: Additional arguments passed to openvino.convert_model.
        """
        if input_sample is None:
            input_sample = self.sample_input

        model_path = self._prepare_export_path(output_path, ".xml")
        export_dir = model_path.parent

        # First export to ONNX in a temporary location
        with tempfile.TemporaryDirectory() as tmp_dir:
            onnx_path = Path(tmp_dir) / "model.onnx"
            self.to_onnx(onnx_path, input_sample, **export_kwargs)

            # Convert ONNX to OpenVINO
            ov_model = openvino.convert_model(str(onnx_path))

            # Post-process if needed (output_names should be list[str] or None)
            output_names = export_kwargs.get("output")
            if output_names is not None and isinstance(output_names, list):
                _postprocess_openvino_model(ov_model, output_names)

        # Save OpenVINO model
        openvino.save_model(ov_model, str(model_path))

        # Create metadata files with policy-specific info
        self._create_metadata(export_dir, ExportBackend.OPENVINO, **self.metadata_extra)

    def to_torch_export_ir(
        self,
        output_path: PathLike | str,
        input_sample: dict[str, torch.Tensor] | None = None,
        **export_kwargs: dict,
    ) -> None:
        """Export to TorchExportIR format.

        For LeRobot policies, exports the wrapper (self) to ensure proper forward() behavior.

        Args:
            output_path: Path to save the TorchExportIR model.
            input_sample: Optional sample input. If None, uses self.sample_input.
            **export_kwargs: Additional arguments passed to torch.export.export.
        """
        if input_sample is None:
            input_sample = self.sample_input

        model_path = self._prepare_export_path(output_path, ".pt2")
        export_dir = model_path.parent

        extra_model_args = self._get_export_extra_args(ExportBackend.TORCH_EXPORT_IR)
        extra_model_args.update(export_kwargs)

        arg_name = self._get_forward_arg_name()

        # Export the wrapper (self) instead of self.model
        self.eval()
        exported_program = torch.export.export(
            self,  # type: ignore[arg-type]
            args=(),  # No positional args
            kwargs={arg_name: input_sample},
            **extra_model_args,
        )
        torch.export.save(exported_program, str(model_path))  # nosec B614

        # Create metadata files with policy-specific info
        self._create_metadata(export_dir, ExportBackend.TORCH_EXPORT_IR, **self.metadata_extra)
