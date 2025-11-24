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
import yaml

from getiaction import __version__
from getiaction.config.mixin import FromConfig
from getiaction.data.transforms import replace_center_crop_with_onnx_compatible
from getiaction.export import Export, ExportBackend
from getiaction.export.mixin_export import _postprocess_openvino_model, _serialize_model_config

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
        _preprocessor: Any  # PolicyProcessorPipeline for normalization
        _postprocessor: Any  # PolicyProcessorPipeline for denormalization
        training: bool  # PyTorch training flag
        eval: Any  # Callable[[], Self]
        _prepare_export_path: Any  # Callable[[PathLike | str, str], Path]
        _get_export_extra_args: Any  # Callable[[ExportBackend], dict[str, Any]]
        _get_forward_arg_name: Any  # Callable[[], str]

    def forward(  # type: ignore[override]
        self,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Generic forward for export/inference (eval mode only).

        This method provides self-contained export behavior:
        1. Accepts un-normalized inputs (raw observations)
        2. Normalizes using preprocessor pipeline
        3. Calls lerobot_policy.predict_action_chunk() for predictions
        4. Denormalizes using postprocessor pipeline
        5. Returns ready-to-use actions

        Override this method in child classes if the policy needs custom export logic
        (e.g., Diffusion bypasses queue mechanism).

        Args:
            batch: Input batch in LeRobot dict format with un-normalized values.
                Keys like "observation.state", "observation.image".

        Returns:
            Denormalized action tensor ready for direct use.
                Shape: (batch, chunk_size, action_dim) for chunked policies
                       (batch, action_dim) for non-chunked policies

        Raises:
            RuntimeError: If called in training mode. Use training_step() instead.

        Note:
            This is the entry point for ONNX/OpenVINO/TorchScript export.
            The full pipeline (preprocess → predict → postprocess) is self-contained.
        """
        if self.training:
            msg = "forward() should not be called in training mode. Use training_step() instead."
            raise RuntimeError(msg)

        # Step 1: Normalize inputs
        normalized_batch = self._preprocessor(batch)

        # Step 2: Get predictions from LeRobot policy
        normalized_actions = self.lerobot_policy.predict_action_chunk(normalized_batch)

        # Step 3: Denormalize and return
        return self._postprocessor(normalized_actions)

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

        Creates un-normalized sample tensors matching the format expected by LeRobot policies.
        The forward() method will handle normalization internally, making the exported model
        self-contained and able to accept raw (un-normalized) inputs from InferenceModel.

        This ensures:
        1. Exported models are standalone (no external normalization needed)
        2. InferenceModel can provide raw observations
        3. Forward() handles all preprocessing during export

        Override this method if your policy requires custom input format.

        Returns:
            Dictionary containing raw (un-normalized) sample tensors for model tracing/export.
            Keys follow LeRobot convention (e.g., "observation.state", "observation.images.camera1").
        """
        config = self.lerobot_policy.config
        batch_size = 1
        device = torch.device("cpu")

        # Determine if temporal dimension is needed
        n_obs_steps = getattr(config, "n_obs_steps", 1)
        use_temporal = n_obs_steps > 1

        sample = {}

        # Add observation.state if robot state is used
        if config.robot_state_feature and "observation.state" in config.input_features:
            state_dim = config.input_features["observation.state"].shape[0]
            shape = (batch_size, n_obs_steps, state_dim) if use_temporal else (batch_size, state_dim)
            sample["observation.state"] = torch.randn(*shape, device=device)

        # Add observation.images if image features are used
        if config.image_features:
            for img_key in config.image_features:
                if img_key in config.input_features:
                    img_shape = config.input_features[img_key].shape  # (C, H, W)
                    shape = (batch_size, n_obs_steps, *img_shape) if use_temporal else (batch_size, *img_shape)
                    sample[img_key] = torch.randn(*shape, device=device)

        return sample

    @property
    def model(self) -> Any:  # noqa: ANN401
        """Alias for self._lerobot_policy for compatibility with base Export mixin.

        The base Export mixin expects self.model, but LeRobot policies use self._lerobot_policy.
        This property provides the mapping for methods like _get_export_extra_args.

        Returns:
            The underlying LeRobot policy instance.
        """
        return self._lerobot_policy  # type: ignore[attr-defined,return-value]

    def _create_metadata(
        self,
        export_dir: Path,
        backend: ExportBackend,
        **metadata_kwargs: dict,
    ) -> None:
        """Create metadata files for exported LeRobot model.

        Overrides base implementation to use self._lerobot_policy instead of self.model.

        Args:
            export_dir: Directory containing exported model
            backend: Export backend used
            **metadata_kwargs: Additional metadata to include
        """
        # Build metadata
        metadata = {
            "getiaction_version": __version__,
            "policy_class": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "backend": str(backend),
            **metadata_kwargs,
        }

        # Add model config if available - use _lerobot_policy instead of self.model
        if hasattr(self, "_lerobot_policy") and hasattr(self._lerobot_policy, "config"):  # type: ignore[attr-defined]
            config_dict = _serialize_model_config(self._lerobot_policy.config)  # type: ignore[attr-defined]
            metadata["config"] = config_dict

        # Save as YAML (preferred)
        yaml_path = export_dir / "metadata.yaml"
        with yaml_path.open("w") as f:
            yaml.dump(metadata, f, default_flow_style=False)

    def _move_processor_steps_to_device(
        self,
        device: torch.device | str,
    ) -> tuple[list[torch.device | str | None], list[torch.device | str | None]]:
        """Move preprocessor and postprocessor steps to specified device.

        Why this complexity is necessary:
        1. LeRobot's NormalizerProcessorStep auto-adapts to input tensor device during forward().
           Simply calling .to() before export isn't enough - we need to force device assignment.
        2. Not all processor steps have .to() method - some only expose .device attribute.
        3. Some steps have additional .tensor_device attribute that must be set separately.
        4. Different steps may be on different devices (though rare in practice).
        5. Per-step tracking is required to correctly restore original device placement.

        The fallback logic (.to() → .device + .tensor_device) ensures all steps are moved,
        preventing device mismatches during ONNX tracing when model is on CPU but processor
        stats auto-adapt to CUDA inputs.

        Args:
            device: Target device ("cpu" or "cuda").

        Returns:
            Tuple of (original_preprocessor_devices, original_postprocessor_devices).
            Each list contains original device for each step (or None if step has no device).
        """
        original_preprocessor_devices = []
        for step in self._preprocessor.steps:  # type: ignore[attr-defined]
            if hasattr(step, "device"):
                original_preprocessor_devices.append(step.device)
                if hasattr(step, "to"):
                    step.to(device)
                else:
                    # Fallback: directly set device attribute
                    step.device = device
                    if hasattr(step, "tensor_device"):
                        step.tensor_device = torch.device(device) if isinstance(device, str) else device
            else:
                original_preprocessor_devices.append(None)

        original_postprocessor_devices = []
        for step in self._postprocessor.steps:  # type: ignore[attr-defined]
            if hasattr(step, "device"):
                original_postprocessor_devices.append(step.device)
                if hasattr(step, "to"):
                    step.to(device)
                else:
                    # Fallback: directly set device attribute
                    step.device = device
                    if hasattr(step, "tensor_device"):
                        step.tensor_device = torch.device(device) if isinstance(device, str) else device
            else:
                original_postprocessor_devices.append(None)

        return original_preprocessor_devices, original_postprocessor_devices

    def _restore_processor_devices(
        self,
        original_preprocessor_devices: list[torch.device | str | None],
        original_postprocessor_devices: list[torch.device | str | None],
    ) -> None:
        """Restore preprocessor and postprocessor steps to original devices.

        Mirrors the logic in _move_processor_steps_to_device() to ensure each step is
        restored to its exact original device. Uses the same fallback mechanism:
        - Try .to() method first (preferred, handles internal state)
        - Fall back to direct .device and .tensor_device attribute assignment

        This granular per-step restoration is necessary because:
        1. Steps may have been on different devices before export
        2. Some steps only support direct attribute assignment
        3. Ensures processor state is identical to pre-export state

        Args:
            original_preprocessor_devices: Original device for each preprocessor step.
            original_postprocessor_devices: Original device for each postprocessor step.
        """
        for i, step in enumerate(self._preprocessor.steps):  # type: ignore[attr-defined]
            if i < len(original_preprocessor_devices) and original_preprocessor_devices[i] is not None:
                orig_device = original_preprocessor_devices[i]
                if hasattr(step, "to"):
                    step.to(orig_device)
                elif hasattr(step, "device"):
                    # Fallback: directly set device attribute
                    step.device = orig_device
                    if hasattr(step, "tensor_device"):
                        step.tensor_device = torch.device(orig_device) if isinstance(orig_device, str) else orig_device

        for i, step in enumerate(self._postprocessor.steps):  # type: ignore[attr-defined]
            if i < len(original_postprocessor_devices) and original_postprocessor_devices[i] is not None:
                orig_device = original_postprocessor_devices[i]
                if hasattr(step, "to"):
                    step.to(orig_device)
                elif hasattr(step, "device"):
                    # Fallback: directly set device attribute
                    step.device = orig_device
                    if hasattr(step, "tensor_device"):
                        step.tensor_device = torch.device(orig_device) if isinstance(orig_device, str) else orig_device

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

        # Convert dots to underscores in input names for ONNX compatibility
        input_names_normalized = [name.replace(".", "_") for name in input_sample]

        # Move everything to CPU for ONNX export compatibility
        original_device = next(self.lerobot_policy.parameters()).device  # type: ignore[union-attr]
        self.cpu()  # type: ignore[attr-defined]
        input_sample_cpu = {k: v.cpu() for k, v in input_sample.items()}

        # Move preprocessor/postprocessor steps to CPU
        orig_prep_devices, orig_post_devices = self._move_processor_steps_to_device("cpu")

        # Replace non-ONNX-compatible transforms (must be after moving to CPU)
        replace_center_crop_with_onnx_compatible(self._lerobot_policy)  # type: ignore[attr-defined]

        # Export
        self.eval()
        torch.onnx.export(
            self,  # type: ignore[arg-type]
            args=(),
            kwargs={arg_name: input_sample_cpu},
            f=str(model_path),
            input_names=input_names_normalized,
            **extra_model_args,
        )

        # Restore original devices
        self.to(original_device)  # type: ignore[attr-defined]
        self._restore_processor_devices(orig_prep_devices, orig_post_devices)

        # Create metadata files with policy-specific info, including input names
        metadata_extra = {**self.metadata_extra, "input_names": input_names_normalized}
        self._create_metadata(export_dir, ExportBackend.ONNX, **metadata_extra)

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

        # Normalize input names (dots to underscores for ONNX compatibility)
        input_names_normalized = [name.replace(".", "_") for name in input_sample]

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

        # Store ONNX input names in metadata for correct inference mapping
        # OpenVINO may expose intermediate Cast nodes as inputs, but we need the semantic names
        metadata_extra = {**self.metadata_extra, "input_names": input_names_normalized}

        # Create metadata files with policy-specific info
        self._create_metadata(export_dir, ExportBackend.OPENVINO, **metadata_extra)

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
