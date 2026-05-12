# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Dynamic LeRobot policy wrapper and named-alias base class.

Provides :class:`LeRobotPolicy`, a generic Lightning wrapper that dispatches
to any registered LeRobot policy by name at construction time, and
:class:`NamedLeRobotPolicy`, a thin specialization that binds ``policy_name``
from a class-level ClassVar so subclasses become trivial aliases — one per
policy in LeRobot's :class:`PreTrainedConfig` registry, all defined in
:mod:`physicalai.policies.lerobot.aliases`.

Use :class:`LeRobotPolicy` directly only when the policy name must be
selected at runtime (e.g. CLI- or YAML-driven dispatch). Otherwise prefer
the named subclasses, which bind ``policy_name`` and so behave uniformly
in configs.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import IO, TYPE_CHECKING, Any, ClassVar

import torch
from lightning_utilities import module_available

from physicalai.config.serializable import dataclass_to_dict, dict_to_dataclass
from physicalai.data import Observation
from physicalai.data.lerobot import FormatConverter
from physicalai.data.lerobot.dataset import _LeRobotDatasetAdapter  # noqa: PLC2701
from physicalai.export.mixin_policy import CONFIG_KEY, DATASET_STATS_KEY, POLICY_NAME_KEY
from physicalai.policies.base import Policy
from physicalai.policies.lerobot.mixin import LeRobotFromConfig

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable

    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.pretrained import PreTrainedPolicy

    from physicalai.gyms import Gym

if TYPE_CHECKING or module_available("lerobot"):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.types import FeatureType
    from lerobot.datasets.feature_utils import dataset_to_policy_features
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.factory import get_policy_class, make_policy_config, make_pre_post_processors

    LEROBOT_AVAILABLE = True
else:
    PreTrainedConfig = None
    FeatureType = None
    LeRobotDataset = None
    LeRobotDatasetMetadata = None
    dataset_to_policy_features = None
    get_policy_class = None
    make_policy_config = None
    make_pre_post_processors = None
    LEROBOT_AVAILABLE = False


_WARNED_UNSUPPORTED_NAMES: set[str] = set()


def _warn_if_unsupported_policy(policy_name: str) -> None:
    """Emit a one-time UserWarning when ``LeRobotPolicy`` is used as escape hatch.

    Names in :data:`physicalai.policies.lerobot.SUPPORTED_POLICIES` carry a
    wrapper-vs-native equivalence guarantee. Any other LeRobot-registered
    name is reachable but unvalidated; signal that to users without
    blocking the call.
    """
    from physicalai.policies.lerobot import SUPPORTED_POLICIES  # noqa: PLC0415

    name = policy_name.lower()
    if name in SUPPORTED_POLICIES or name in _WARNED_UNSUPPORTED_NAMES:
        return
    _WARNED_UNSUPPORTED_NAMES.add(name)
    warnings.warn(
        f"LeRobot policy {policy_name!r} is not in physicalai's supported set "
        f"({', '.join(SUPPORTED_POLICIES)}); wrapper behavior is best-effort "
        "and not covered by the equivalence test suite.",
        UserWarning,
        stacklevel=3,
    )


class LeRobotPolicy(Policy, LeRobotFromConfig):
    """Dynamic Lightning wrapper around any registered LeRobot policy.

    Dispatches to the LeRobot policy identified by ``policy_name`` and
    serves as the shared base for the named aliases in
    :mod:`physicalai.policies.lerobot` — one per policy in LeRobot's
    :class:`PreTrainedConfig` registry. Configuration flows through
    LeRobot's own :class:`PreTrainedConfig` dataclasses; this wrapper does
    not redeclare policy-specific fields.

    Use this class directly only when the policy must be chosen at runtime
    (e.g. ``policy_name`` driven by a CLI flag or YAML key). Otherwise
    prefer the named subclasses, which bind ``policy_name`` from a
    class-level ``POLICY_NAME`` and behave uniformly in configs.

    Examples:
        >>> # Dynamic dispatch
        >>> policy = LeRobotPolicy.from_pretrained("lerobot/act_pusht")
        >>> policy = LeRobotPolicy.from_dataset("act", "lerobot/pusht", optimizer_lr=1e-4)
        >>> policy = LeRobotPolicy(policy_name="diffusion", num_inference_steps=100)

        >>> # Named alias (preferred for static use)
        >>> from physicalai.policies.lerobot import ACT
        >>> policy = ACT.from_dataset("lerobot/pusht", optimizer_lr=1e-4)

    Args:
        policy_name: LeRobot registry name (e.g. ``"act"``, ``"diffusion"``).
            Must be registered with LeRobot's :func:`get_policy_class`.
        input_features: Optional input feature definitions. When ``None``
            features are extracted lazily in :meth:`setup` from the
            attached DataModule.
        output_features: Optional output feature definitions. Same lazy
            semantics as ``input_features``.
        config: Pre-built :class:`PreTrainedConfig`. When provided, it is
            used as-is and ``policy_config`` / ``**kwargs`` apply field
            overrides on it.
        dataset_stats: Dataset statistics for the normalization processors.
        policy_config: Mapping of policy-specific parameters (typically
            from YAML configs). Merged with ``**kwargs``; ``**kwargs`` win
            on key collisions for CLI override support.
        **kwargs: Policy-specific parameters supplied positionally in
            Python. Forwarded to LeRobot's :func:`make_policy_config`.

    Raises:
        ImportError: If LeRobot is not installed.
        ValueError: If ``policy_name`` is not registered.
    """

    def __init__(
        self,
        policy_name: str,
        input_features: dict[str, PolicyFeature] | None = None,
        output_features: dict[str, PolicyFeature] | None = None,
        config: PreTrainedConfig | None = None,
        dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
        policy_config: dict[str, Any] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize the LeRobot policy wrapper.

        Supports both eager initialization (when input_features provided) and lazy
        initialization (features extracted in setup() hook from DataModule).

        Args:
            policy_name: Name of the policy ('diffusion', 'act', 'vqbet', etc.)
            input_features: Optional input feature definitions (lazy if None)
            output_features: Optional output feature definitions (lazy if None)
            config: Pre-built LeRobot config object (optional)
            dataset_stats: Dataset statistics for normalization (optional)
            policy_config: Policy-specific parameters as a dict (for YAML configs).
            **kwargs: Policy-specific parameters (for Python usage). Merged with
                policy_config; kwargs take precedence if both specify the same key.

        Raises:
            ImportError: If LeRobot is not installed.
            AttributeError: If ``optimizer_lr`` is supplied alongside a
                pre-built ``config`` whose dataclass lacks an
                ``optimizer_lr`` field.
        """
        if not LEROBOT_AVAILABLE:
            msg = (
                "LeRobotPolicy requires LeRobot framework.\n\n"
                "Install with:\n"
                "    pip install lerobot\n\n"
                "Or install physicalai with LeRobot support:\n"
                "    pip install physicalai-train[lerobot]\n\n"
                "For more information, see: https://github.com/huggingface/lerobot"
            )
            raise ImportError(msg)

        super().__init__(n_action_steps=1)  # LeRobot handles its own action chunking

        _warn_if_unsupported_policy(policy_name)

        self.policy_name = policy_name

        # Merge policy_config (from YAML) with kwargs (from Python)
        # kwargs take precedence for CLI override support
        merged_policy_config = {**(policy_config or {}), **kwargs}

        # When a pre-built config is supplied, ``optimizer_lr`` must be
        # applied to the config object itself (not stuffed into
        # ``_policy_config``, which is only consumed during lazy build).
        if config is not None and "optimizer_lr" in merged_policy_config:
            override_lr = merged_policy_config.pop("optimizer_lr")
            if not hasattr(config, "optimizer_lr"):
                msg = (
                    f"Cannot override optimizer_lr on {type(config).__name__}: "
                    "this LeRobot config does not expose an optimizer_lr field."
                )
                raise AttributeError(msg)
            config.optimizer_lr = override_lr

        # Store for lazy initialization
        self._input_features = input_features
        self._output_features = output_features
        self._provided_config = config
        self._dataset_stats = dataset_stats
        self._policy_config = merged_policy_config

        # Will be initialized in setup() if not provided
        self._lerobot_policy: PreTrainedPolicy
        self._config: PreTrainedConfig | None = None

        # If features are provided, initialize immediately (backward compatibility)
        if input_features is not None and output_features is not None:
            self._initialize_policy(input_features, output_features, config, dataset_stats)
        elif config is not None:
            # Config provided directly - can initialize now
            self._initialize_policy(None, None, config, dataset_stats)

        self.save_hyperparameters()

    def _init_pretrained_shell(self) -> None:
        """Minimal initialization for instances created by ``from_pretrained``.

        Calls ``Policy.__init__`` (which chains to ``LightningModule.__init__``)
        to set up the action queue, rollout metrics, and Lightning internals,
        **without** requiring the ``policy_name`` or feature arguments that the
        full ``LeRobotPolicy.__init__`` demands.

        This avoids hard-coding ``Policy.__init__(self, …)`` inside the mixin,
        keeping the initializer hierarchy-safe if intermediate classes are
        inserted later.
        """
        Policy.__init__(self, n_action_steps=1)

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Save model config and policy name to checkpoint for reconstruction.

        Args:
            checkpoint: Lightning checkpoint dictionary to modify in-place.
        """
        if self._config is not None:
            config_dict = dataclass_to_dict(self._config)
            config_dict["type"] = self.policy_name
            checkpoint[CONFIG_KEY] = config_dict
            checkpoint[POLICY_NAME_KEY] = self.policy_name
            if self._dataset_stats is not None:
                checkpoint[DATASET_STATS_KEY] = dataclass_to_dict(self._dataset_stats)

    def save_pretrained(
        self,
        save_directory: str | Path,
        *,
        repo_id: str | None = None,
        private: bool | None = None,
        token: str | bool | None = None,
    ) -> str | None:
        """Save policy in LeRobot-compatible format (config.json + model.safetensors).

        The output directory is directly loadable by both LeRobot's
        ``PreTrainedPolicy.from_pretrained()`` and physicalai's
        ``LeRobotPolicy.from_pretrained()``, enabling full round-trip
        compatibility between the two frameworks.

        If ``repo_id`` is provided the model is automatically pushed to the
        HuggingFace Hub after saving locally.

        Args:
            save_directory: Path to directory where the policy will be saved.
            repo_id: Repository ID on HuggingFace Hub (e.g., ``"username/my-policy"``).
                When provided, the model is pushed to the Hub after saving.
            private: Whether the Hub repository should be private.
            token: HuggingFace authentication token.

        Returns:
            URL of the Hub commit if pushed, else ``None``.

        Raises:
            RuntimeError: If the wrapper has no initialized preprocessor or
                postprocessor (the saved directory would not be reloadable).

        Examples:
            Save to local directory:

                >>> policy = LeRobotPolicy.from_dataset("act", "lerobot/pusht")
                >>> policy.save_pretrained("./my_act_model")

            Save and push to Hub:

                >>> policy.save_pretrained(
                ...     "./my_act_model",
                ...     repo_id="username/act-pusht",
                ... )
        """
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.lerobot_policy._save_pretrained(save_dir)  # noqa: SLF001

        # Processors are mandatory: ``LeRobotPolicy.from_pretrained`` (and
        # LeRobot's own loader) reads ``policy_preprocessor.json`` /
        # ``policy_postprocessor.json`` from the saved directory. A wrapper
        # without them was either never initialized (lazy mode pre-setup) or
        # constructed via an incomplete code path; silently skipping them
        # produces an unloadable directory. Fail loudly instead.
        preprocessor = getattr(self, "_preprocessor", None)
        postprocessor = getattr(self, "_postprocessor", None)
        if preprocessor is None or postprocessor is None:
            msg = (
                f"Cannot save {type(self).__name__} to {save_dir}: processors are not "
                "initialized. Ensure the policy was constructed with input/output "
                "features (or via from_dataset / from_pretrained / Trainer.fit) before "
                "calling save_pretrained."
            )
            raise RuntimeError(msg)
        preprocessor._save_pretrained(save_dir)  # noqa: SLF001
        postprocessor._save_pretrained(save_dir)  # noqa: SLF001

        if repo_id is not None:
            return self.push_to_hub(
                repo_id=repo_id,
                private=private,
                token=token,
            )
        return None

    def push_to_hub(
        self,
        repo_id: str,
        *,
        commit_message: str | None = None,
        private: bool | None = None,
        token: str | bool | None = None,
        revision: str | None = None,
    ) -> str:
        """Push policy to the HuggingFace Hub in LeRobot-compatible format.

        Saves the policy to a temporary directory using :meth:`save_pretrained`
        and uploads the result via ``HfApi.upload_folder``.  The uploaded
        artefacts are loadable by both LeRobot and physicalai.

        Args:
            repo_id: Repository ID on HuggingFace Hub (e.g., ``"username/my-policy"``).
            commit_message: Custom commit message. Defaults to ``"Upload policy"``.
            private: Whether the repository should be private.
            token: HuggingFace authentication token.
            revision: Branch to push to. Defaults to ``"main"``.

        Returns:
            URL of the commit on the HuggingFace Hub.

        Examples:
            Push a trained policy to the Hub:

                >>> policy = LeRobotPolicy.from_dataset("act", "lerobot/pusht")
                >>> url = policy.push_to_hub("username/act-pusht")
                >>> print(url)
        """
        from huggingface_hub import HfApi  # noqa: PLC0415

        api = HfApi(token=token)
        repo_id = api.create_repo(repo_id=repo_id, private=private, exist_ok=True).repo_id

        if commit_message is None:
            commit_message = "Upload policy"

        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            saved_path = Path(tmp) / repo_id
            self.save_pretrained(saved_path)
            result = api.upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=saved_path,
                commit_message=commit_message,
                revision=revision,
                allow_patterns=["*.safetensors", "*.json", "*.yaml", "*.md"],
                ignore_patterns=["*.tmp", "*.log"],
            )

        logger.info("Model pushed to %s", result)
        return result

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path | IO[bytes],
        map_location: torch.device | str | int | Callable | dict | None = None,
        hparams_file: str | Path | None = None,
        strict: bool | None = None,  # noqa: FBT001
        weights_only: bool | None = None,  # noqa: FBT001
        **kwargs: Any,  # noqa: ANN401
    ) -> LeRobotPolicy:
        """Load a LeRobot policy from a Lightning checkpoint.

        This method loads the checkpoint, reconstructs the underlying LeRobot policy
        from the saved config, and restores the model weights.

        Args:
            checkpoint_path: Path to the checkpoint file (.ckpt or .pt).
            map_location: Device to map tensors to. If None, uses default device.
            hparams_file: Unused. Kept for Lightning compatibility.
            strict: Unused. Kept for Lightning compatibility.
            weights_only: Whether to load only weights (default True for security).
            **kwargs: Additional arguments passed to the policy constructor.

        Returns:
            Loaded policy with weights restored, ready for inference.

        Raises:
            KeyError: If checkpoint doesn't contain required model config.
            ImportError: If LeRobot is not installed.

        Examples:
            Load checkpoint for inference:

                >>> from physicalai.policies.lerobot import LeRobotPolicy
                >>> policy = LeRobotPolicy.load_from_checkpoint("checkpoints/epoch=10.ckpt")
                >>> action = policy.select_action(observation)

            Load checkpoint to specific device:

                >>> policy = LeRobotPolicy.load_from_checkpoint(
                ...     "checkpoints/best.ckpt",
                ...     map_location="xpu",
                ... )
        """
        del hparams_file, strict  # Unused, kept for Lightning compatibility

        if not LEROBOT_AVAILABLE:
            msg = (
                "LeRobotPolicy.load_from_checkpoint requires LeRobot to be installed.\n\n"
                "Install with:\n"
                "    uv pip install lerobot\n\n"
                "Or install physicalai with LeRobot support:\n"
                "    uv pip install physicalai-train[lerobot]"
            )
            raise ImportError(msg)

        # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
        checkpoint = torch.load(  # nosec B614
            checkpoint_path,
            map_location=map_location,
            weights_only=weights_only if weights_only is not None else True,
        )

        # Extract model config dict
        if CONFIG_KEY not in checkpoint:
            msg = f"Checkpoint missing '{CONFIG_KEY}'. Cannot reconstruct policy without config."
            raise KeyError(msg)

        config_dict = checkpoint[CONFIG_KEY]

        # Get policy_name from checkpoint or config dict
        policy_name = checkpoint.get(POLICY_NAME_KEY)
        if policy_name is None:
            # Fallback: try to get from config dict's 'type' field
            policy_name = config_dict.get("type")
            if policy_name is None:
                msg = f"Checkpoint missing '{POLICY_NAME_KEY}'. Cannot determine which LeRobot policy to reconstruct."
                raise KeyError(msg)

        # Reconstruct LeRobot config from dict
        policy_cls = get_policy_class(policy_name)
        config_cls = policy_cls.config_class  # type: ignore[attr-defined]
        config = dict_to_dataclass(config_cls, config_dict)
        dataset_stats = checkpoint.get(DATASET_STATS_KEY)

        # All wrappers (LeRobotPolicy and every NamedLeRobotPolicy subclass)
        # accept ``config=`` and forward it to the same initializer, so a
        # single construction path suffices. Named subclasses bind their own
        # ``policy_name`` from POLICY_NAME; pass ``policy_name`` only for the
        # bare LeRobotPolicy where it is required positionally.
        ctor_kwargs: dict[str, Any] = {
            "config": config,
            "dataset_stats": dataset_stats,
            **kwargs,
        }
        if cls is LeRobotPolicy:
            ctor_kwargs["policy_name"] = policy_name
        policy = cls(**ctor_kwargs)

        # Load state dict (model weights + normalizer stats)
        if "state_dict" in checkpoint:
            policy.load_state_dict(checkpoint["state_dict"])

        return policy

    @classmethod
    def from_dataset(
        cls,
        policy_name: str,
        dataset: LeRobotDataset | _LeRobotDatasetAdapter | str,
        **kwargs: Any,  # noqa: ANN401
    ) -> LeRobotPolicy:
        """Create policy with eager initialization from a dataset or repo ID.

        This factory method extracts features from the dataset and builds the policy
        immediately, making it ready for inference without a Lightning Trainer.

        Note:
            LeRobot policies require LeRobot-compatible data sources for feature
            extraction and normalization statistics. Generic physicalai datasets
            are not supported.

        Args:
            policy_name: Name of the policy ('act', 'diffusion', 'vqbet', etc.)
            dataset: Either a LeRobotDataset instance or a HuggingFace Hub repo ID string.
                If a string is provided, only metadata is fetched (lightweight).
            **kwargs: Additional policy configuration parameters.

        Returns:
            Fully initialized LeRobotPolicy ready for inference.

        Raises:
            ImportError: If LeRobot is not installed.

        Examples:
            Create from repo ID (lightweight, only fetches metadata):

                >>> policy = LeRobotPolicy.from_dataset(
                ...     "act",
                ...     "lerobot/pusht",
                ...     dim_model=512,
                ... )

            Create from an already loaded dataset:

                >>> dataset = LeRobotDataset("lerobot/pusht")
                >>> policy = LeRobotPolicy.from_dataset(
                ...     "diffusion",
                ...     dataset,
                ...     num_inference_steps=100,
                ... )

            Create from a datamodule's train_dataset:

                >>> datamodule = LeRobotDataModule(repo_id="lerobot/pusht")
                >>> datamodule.setup("fit")
                >>> policy = LeRobotPolicy.from_dataset("act", datamodule.train_dataset)
        """
        if not LEROBOT_AVAILABLE:
            msg = "LeRobotPolicy.from_dataset requires LeRobot to be installed."
            raise ImportError(msg)

        # Handle _LeRobotDatasetAdapter (wrapper for physicalai format)
        if isinstance(dataset, _LeRobotDatasetAdapter):
            dataset = dataset._lerobot_dataset  # noqa: SLF001

        # Get metadata from dataset or repo ID
        meta = LeRobotDatasetMetadata(dataset) if isinstance(dataset, str) else dataset.meta

        # Convert dataset features to policy features
        features = dataset_to_policy_features(meta.features)

        # Split into input/output features
        input_features = {k: f for k, f in features.items() if f.type != FeatureType.ACTION}
        output_features = {k: f for k, f in features.items() if f.type == FeatureType.ACTION}

        return cls(
            policy_name=policy_name,
            input_features=input_features,
            output_features=output_features,
            dataset_stats=meta.stats,
            **kwargs,
        )

    @property
    def lerobot_policy(self) -> PreTrainedPolicy:
        """Get the initialized LeRobot policy.

        Returns:
            The initialized LeRobot policy.

        Raises:
            RuntimeError: If the policy hasn't been initialized yet.
        """
        if not hasattr(self, "_lerobot_policy") or self._lerobot_policy is None:
            msg = "Policy not initialized. Call setup() or provide input_features during __init__."
            raise RuntimeError(msg)
        return self._lerobot_policy

    @property  # type: ignore[override]
    def model(self) -> PreTrainedPolicy:  # type: ignore[override]
        """Alias for :attr:`lerobot_policy`.

        Defined as a ``@property`` so that the underlying ``_lerobot_policy``
        sub-module is **not** registered twice in the state dict.  When
        ``model`` was a plain attribute set to ``self._lerobot_policy``,
        PyTorch recorded every parameter under both ``model.*`` **and**
        ``_lerobot_policy.*``, doubling the checkpoint size and forcing
        ``strict=False`` on reload.

        The base :class:`Policy.__init__` already skips
        ``self.model = None`` when it detects a property descriptor on the
        subclass, so this is fully compatible with the base class.
        """
        return self.lerobot_policy

    @model.setter
    def model(self, value: Any) -> None:  # noqa: ANN401
        """No-op setter to absorb stale assignments during deserialization."""

    def _initialize_policy(
        self,
        input_features: dict[str, PolicyFeature] | None,
        output_features: dict[str, PolicyFeature] | None,
        config: PreTrainedConfig | None,
        dataset_stats: dict[str, dict[str, torch.Tensor]] | None,
    ) -> None:
        """Initialize the LeRobot policy instance.

        Args:
            input_features: Input feature definitions.
            output_features: Output feature definitions.
            config: Pre-built config object.
            dataset_stats: Dataset statistics for normalization.

        Raises:
            ValueError: If neither config nor features are provided.
        """
        # Build or use provided config
        if config is None:
            if input_features is None or output_features is None:
                msg = (
                    "Either 'config' must be provided, or both 'input_features' and 'output_features' must be provided."
                )
                raise ValueError(msg)

            # Remove dataset_stats from policy_config if present
            # (it should be passed to policy constructor, not config)
            clean_policy_config = {k: v for k, v in self._policy_config.items() if k != "dataset_stats"}

            # Create config dynamically using LeRobot's factory
            config = make_policy_config(
                self.policy_name,
                input_features=input_features,
                output_features=output_features,
                **clean_policy_config,
            )

        # Get the policy class dynamically
        policy_cls = get_policy_class(self.policy_name)

        # Instantiate the LeRobot policy
        policy = policy_cls(config)
        self.add_module("_lerobot_policy", policy)

        # Create preprocessor/postprocessor for normalization
        self._preprocessor, self._postprocessor = make_pre_post_processors(config, dataset_stats=dataset_stats)

        # Expose framework info
        self._framework = "lerobot"
        self._config = config

    def setup(self, stage: str) -> None:
        """Lightning hook called before training/validation/test.

        Extracts input/output features from the DataModule's dataset if not provided
        during initialization. This enables YAML-based configuration without requiring
        features to be specified in advance.

        Args:
            stage: Current stage ('fit', 'validate', 'test', or 'predict').

        Raises:
            RuntimeError: If DataModule or train_dataset is not available.
        """
        del stage  # Unused argument

        if hasattr(self, "_lerobot_policy") and self._lerobot_policy is not None:
            # Already initialized
            return

        # Lazy initialization: extract features from DataModule
        if not hasattr(self.trainer, "datamodule"):
            msg = (
                "Lazy initialization requires a DataModule with train_dataset. "
                "Either provide input_features/output_features during __init__, "
                "or ensure a DataModule is attached to the trainer."
            )
            raise RuntimeError(msg)

        # Get the training dataset - handle both data formats
        train_dataset = self.trainer.datamodule.train_dataset

        # Extract LeRobot dataset based on type
        if isinstance(train_dataset, _LeRobotDatasetAdapter):
            # Wrapped in adapter for physicalai format conversion
            lerobot_dataset = train_dataset._lerobot_dataset  # noqa: SLF001
        elif hasattr(train_dataset, "meta") and hasattr(train_dataset.meta, "features"):
            # Assume it's a raw LeRobotDataset (data_format="lerobot")
            lerobot_dataset = train_dataset
        else:
            msg = (
                f"Expected train_dataset to be _LeRobotDatasetAdapter or LeRobotDataset, "
                f"got {type(train_dataset)}. Use LeRobotDataModule with appropriate data_format."
            )
            raise RuntimeError(msg)

        # Convert LeRobot dataset features to policy features
        features = dataset_to_policy_features(lerobot_dataset.meta.features)

        # Split into input/output features (same logic as from_dataset)
        input_features = {k: f for k, f in features.items() if f.type != FeatureType.ACTION}
        output_features = {k: f for k, f in features.items() if f.type == FeatureType.ACTION}

        # Get dataset statistics if not provided
        stats = self._dataset_stats
        if stats is None:
            stats = lerobot_dataset.meta.stats

        # Initialize policy now
        self._initialize_policy(input_features, output_features, self._provided_config, stats)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer for Lightning.

        Uses LeRobot's get_optimizer_preset from config which includes proper
        parameter grouping (e.g., different lr for backbone) and optimizer settings.

        Returns:
            Configured optimizer from LeRobot's preset.

        Raises:
            RuntimeError: If policy has not been initialized yet.
        """
        # Use LeRobot's optimizer preset from config - this handles:
        # - Proper optimizer type (Adam, AdamW, etc.)
        # - Learning rate and weight decay from config
        # - Parameter grouping via get_optim_params (e.g., backbone lr)
        if self._config is None:
            msg = "Policy must be initialized before configure_optimizers"
            raise RuntimeError(msg)
        optimizer_config = self._config.get_optimizer_preset()
        params = self.lerobot_policy.get_optim_params()
        return optimizer_config.build(params)

    def training_step(self, batch: Observation | dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step uses LeRobot's loss computation.

        Args:
            batch: Input batch (Observation or LeRobot dict format).
            batch_idx: Index of the batch.

        Returns:
            The total loss for the batch.
        """
        del batch_idx  # Unused argument

        total_loss, loss_dict = self(batch)

        # Log individual loss components if available (skip non-scalar values and 'loss' key)
        if loss_dict is not None:
            for key, value in loss_dict.items():
                # Skip 'loss' key (we log it separately as train/loss)
                if key == "loss":
                    continue
                # Only log scalar values
                if isinstance(value, (int, float, torch.Tensor)) and (
                    not isinstance(value, torch.Tensor) or value.numel() == 1
                ):
                    self.log(f"train/{key}", value, prog_bar=False)

        self.log("train/loss", total_loss, prog_bar=True)
        return total_loss

    def validation_step(self, batch: Gym, batch_idx: int) -> dict[str, float]:
        """Validation step for Lightning.

        Runs gym-based validation by executing rollouts in the environment.
        The DataModule's val_dataloader returns Gym environment instances directly.

        Args:
            batch: Gym environment to evaluate.
            batch_idx: Batch index.

        Returns:
            Metrics dict from gym rollout evaluation.
        """
        return self.evaluate_gym(batch, batch_idx, stage="val")

    def test_step(self, batch: Gym, batch_idx: int) -> dict[str, float]:
        """Test step for Lightning.

        Runs gym-based testing by executing rollouts in the environment.
        The DataModule's test_dataloader returns Gym environment instances directly.

        Args:
            batch: Gym environment to evaluate.
            batch_idx: Batch index.

        Returns:
            Metrics dict from gym rollout evaluation.
        """
        return self.evaluate_gym(batch, batch_idx, stage="test")

    def forward(  # type: ignore[override]
        self,
        batch: Observation | dict[str, torch.Tensor],
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        """Forward pass for the LeRobot policy.

        Handles both training and inference modes:
        - Training: Returns (loss, loss_dict) for backpropagation
        - Inference: Returns denormalized actions for prediction/export

        Args:
            batch: Input batch (Observation or LeRobot dict).

        Returns:
            - Training mode: Tuple of (loss, loss_dict or None)
            - Inference mode: Action tensor
        """
        # Convert to LeRobot format if needed
        batch_dict = FormatConverter.to_lerobot_dict(batch) if isinstance(batch, Observation) else batch

        # Apply preprocessor for normalization, padding to max_action_dim, etc.
        # This is required for policies like Groot that expect padded actions
        batch_dict = self._preprocessor(batch_dict)

        if self.training:
            # Training mode: compute loss
            output = self.lerobot_policy(batch_dict)

            # Handle different return formats (some policies return tuple, some just loss)
            if isinstance(output, tuple):
                return output
            return output, None

        # Inference mode: predict actions and denormalize
        action = self.lerobot_policy.select_action(batch_dict)
        return self._postprocessor(action)

    def predict_action_chunk(self, batch: Observation | dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict full action chunk using the wrapped LeRobot policy.

        Returns the complete action chunk predicted by the model without
        queue management. Use this when you need all predicted future actions.

        Args:
            batch: Input batch of observations.

        Returns:
            Action chunk tensor of shape (B, chunk_size, action_dim) or
            (chunk_size, action_dim) for unbatched input.
        """
        batch_dict = FormatConverter.to_lerobot_dict(batch) if isinstance(batch, Observation) else batch
        batch_dict = self._preprocessor(batch_dict)
        actions = self.lerobot_policy.predict_action_chunk(batch_dict)
        return self._postprocessor(actions)

    def select_action(self, batch: Observation | dict[str, torch.Tensor]) -> torch.Tensor:
        """Select single action using LeRobot's internal action queue.

        Delegates to the LeRobot policy's select_action, which manages
        its own action queue internally. When the queue is empty, it calls
        predict_action_chunk to get a new chunk, queues the actions, and
        returns the first one.

        Args:
            batch: Input batch of observations (raw, from gym).

        Returns:
            Single action tensor of shape (action_dim,).
        """
        was_training = self.training
        self.eval()
        try:
            batch_dict = FormatConverter.to_lerobot_dict(batch) if isinstance(batch, Observation) else batch
            batch_dict = self._preprocessor(batch_dict)
            action = self.lerobot_policy.select_action(batch_dict)
            return self._postprocessor(action)
        finally:
            if was_training:
                self.train()

    def reset(self) -> None:
        """Reset the policy state for a new episode.

        Forwards the reset call to the underlying LeRobot policy,
        which clears action queues, observation histories, and any
        other stateful components.
        """
        super().reset()
        self.lerobot_policy.reset()

    @property
    def config(self) -> PreTrainedConfig:
        """Access the underlying LeRobot config.

        Returns:
            The policy's configuration object.
        """
        return self._config

    def __repr__(self) -> str:
        """String representation.

        Returns:
            String summarizing the policy instance.
        """
        # Get policy_name from either attribute or config
        policy_name = getattr(self, "policy_name", None)
        if policy_name is None and hasattr(self, "_lerobot_policy"):
            policy_name = self._lerobot_policy.config.type

        # Get learning rate from config if available
        lr_info = "N/A"
        if hasattr(self, "_config") and self._config is not None:
            optimizer_preset = self._config.get_optimizer_preset()
            lr_info = f"{optimizer_preset.lr}"

        return f"{self.__class__.__name__}(\n  policy_name={policy_name!r},\n  lr={lr_info},\n)"


class NamedLeRobotPolicy(LeRobotPolicy):
    """LeRobotPolicy specialization that binds ``policy_name`` from a ClassVar.

    Subclasses set :attr:`POLICY_NAME` and inherit the entire
    :class:`LeRobotPolicy` lifecycle. The base enforces two invariants the
    bare :class:`LeRobotPolicy` cannot:

    1. ``Subclass.from_dataset(...)`` returns an instance of ``Subclass``,
       not a bare ``LeRobotPolicy`` (preserves ``isinstance`` discrimination).
    2. ``Subclass(policy_name=other)`` is rejected, so a class named ``ACT``
       can never silently build a different policy.
    """

    POLICY_NAME: ClassVar[str]

    def __init__(
        self,
        config: PreTrainedConfig | None = None,
        *,
        input_features: dict[str, PolicyFeature] | None = None,
        output_features: dict[str, PolicyFeature] | None = None,
        dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
        policy_config: dict[str, Any] | None = None,
        **overrides: Any,  # noqa: ANN401
    ) -> None:
        """Forward to :class:`LeRobotPolicy` with ``policy_name`` bound from POLICY_NAME.

        Raises:
            NotImplementedError: If the subclass did not set :attr:`POLICY_NAME`.
            ValueError: If a conflicting ``policy_name`` kwarg is supplied.
        """
        policy_name = getattr(type(self), "POLICY_NAME", None)
        if not policy_name:
            msg = f"{type(self).__name__} must define a class-level POLICY_NAME."
            raise NotImplementedError(msg)

        explicit_name = overrides.pop("policy_name", None)
        if explicit_name is not None and explicit_name != policy_name:
            msg = (
                f"{type(self).__name__} is bound to policy_name={policy_name!r}; "
                f"refusing to override with policy_name={explicit_name!r}."
            )
            raise ValueError(msg)

        super().__init__(
            policy_name=policy_name,
            input_features=input_features,
            output_features=output_features,
            config=config,
            dataset_stats=dataset_stats,
            policy_config=policy_config,
            **overrides,
        )

    @classmethod
    def from_dataset(  # type: ignore[override]
        cls,
        dataset: LeRobotDataset | _LeRobotDatasetAdapter | str,
        **kwargs: Any,  # noqa: ANN401
    ) -> NamedLeRobotPolicy:
        """Build the subclass from a dataset; returns ``cls`` (not bare LeRobotPolicy).

        Returns:
            Instance of the calling subclass with features extracted from the dataset.
        """
        policy_name = getattr(cls, "POLICY_NAME", None)
        if not policy_name:
            msg = f"{cls.__name__} must define a class-level POLICY_NAME."
            raise NotImplementedError(msg)

        # Reuse the parent factory's feature-extraction logic but rebind
        # ``cls`` so the returned instance is the subclass.
        return LeRobotPolicy.from_dataset.__func__(  # type: ignore[attr-defined,no-any-return]
            cls,
            policy_name,
            dataset,
            **kwargs,
        )
