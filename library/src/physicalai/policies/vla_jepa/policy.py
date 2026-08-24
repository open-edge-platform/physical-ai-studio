# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Copyright 2026 The HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0

"""VLA-JEPA Policy - Lightning wrapper for training and inference."""

from __future__ import annotations

import functools
import inspect
import json
import logging
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from huggingface_hub import hf_hub_download
from physicalai.inference.data import InferenceFeature, InferenceFeatureDtype, InferenceFeatureType
from physicalai.inference.manifest import ComponentSpec
from safetensors.torch import load_file

from physicalai.data.observation import ACTION, IMAGES, STATE, TASK, FeatureType
from physicalai.export import ExportablePolicyMixin, ExportBackend
from physicalai.export.backends import ExportParameters, TorchExportParameters
from physicalai.policies.base import Policy
from physicalai.train.schedulers import cosine_decay_with_warmup_scheduler
from physicalai.train.utils import reformat_dataset_to_match_policy

from .config import VLAJEPAConfig
from .model import VLAJEPAModel
from .pretrained_utils import (
    drop_unused_module_keys,
    extract_dataset_stats,
    filter_reinit_modules,
    fix_state_dict_keys,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from physicalai.data import Observation

    from .preprocessor import VLAJEPAPostprocessor, VLAJEPAPreprocessor

logger = logging.getLogger(__name__)

# Fields that describe the *architecture*: they are baked into the published checkpoint shapes and
# are always taken from its config.json rather than from the caller's defaults.
_PRETRAINED_ARCHITECTURE_FIELDS = frozenset({
    "action_attention_head_dim",
    "action_dim",
    "action_hidden_size",
    "action_max_seq_len",
    "action_model_type",
    "action_num_heads",
    "action_num_layers",
    "chunk_size",
    "embodied_action_token",
    "jepa_encoder_name",
    "jepa_tubelet_size",
    "num_action_tokens_per_timestep",
    "num_embodied_action_tokens_per_instruction",
    "num_video_frames",
    "predictor_depth",
    "predictor_mlp_ratio",
    "predictor_num_heads",
    "prompt_template",
    "qwen_model_name",
    "special_action_token",
    "state_dim",
    "world_model_num_views",
})


def _track_explicit_args(init: Callable[..., None]) -> Callable[..., None]:
    """Record which ``__init__`` arguments the caller actually passed.

    By the time ``__init__`` runs, every unspecified argument already holds its default, which
    erases the difference between "the caller asked for this value" and "the caller said nothing".
    :meth:`VLAJEPA._from_hf` needs that difference: a default must never silently overwrite what a
    published checkpoint recorded for the same field.

    Args:
        init: The ``__init__`` to wrap.

    Returns:
        The wrapped ``__init__``, which sets ``self._explicit_args`` before delegating.
    """
    signature = inspect.signature(init)

    @functools.wraps(init)
    def wrapper(self: VLAJEPA, *args: object, **kwargs: object) -> None:
        bound = signature.bind_partial(self, *args, **kwargs)
        self._explicit_args = frozenset(bound.arguments) - {"self"}
        init(self, *args, **kwargs)

    # `functools.wraps` sets `__wrapped__`, which `inspect.signature` follows but
    # `inspect.getfullargspec` does not: it would report the wrapper's own `**kwargs`. Lightning
    # reads that spec in `_load_state` and skips filtering the checkpoint hparams whenever the
    # constructor appears to accept `**kwargs`, so `load_from_checkpoint` would hand `__init__`
    # saved keys it never declared. Setting `__signature__` makes both introspection paths agree.
    wrapper.__signature__ = signature  # type: ignore[attr-defined]

    return wrapper


class VLAJEPA(ExportablePolicyMixin, Policy):
    """VLA-JEPA Policy - Qwen3-VL backbone, flow-matching action head and a V-JEPA2 world model.

    Lightning wrapper for training and inference with the VLA-JEPA model.

    Uses dual-path initialization:
    - **Lazy path**: `VLAJEPA()` + `trainer.fit()` - model built in setup()
    - **Eager path**: `VLAJEPA.load_from_checkpoint()` - model built immediately

    Export is limited to the Torch backend. ``to_torch`` serializes the Lightning checkpoint and
    a manifest, so it needs no tracing. Graph-capturing backends (ONNX, OpenVINO, ExecuTorch) are
    intentionally unsupported: the Qwen chat-template tokenization and the variable-length
    vision-token sequence it produces are not traceable.

    Args:
        pretrained_name_or_path: HuggingFace repo id or local path holding a published VLA-JEPA
            ``config.json`` / ``model.safetensors``. Every field it records is used as-is;
            architecture fields cannot be overridden, and the remaining runtime and training
            fields are overridden only by arguments passed explicitly to this constructor.
        dataset_stats: Dataset normalization statistics for eager initialization.

    See :class:`~physicalai.policies.vla_jepa.VLAJEPAConfig` for every other argument.

    Example:
        Training:

        >>> policy = VLAJEPA(chunk_size=8, n_action_steps=8, enable_world_model=False)
        >>> trainer = physicalai.Trainer(max_epochs=100)
        >>> trainer.fit(policy, datamodule)

        Inference:

        >>> policy = VLAJEPA.load_from_checkpoint("checkpoint.ckpt")
        >>> action = policy.select_action(obs)
    """

    model: Any
    _preprocessor: Any
    _explicit_args: frozenset[str]

    @_track_explicit_args
    def __init__(  # noqa: PLR0913
        self,
        pretrained_name_or_path: str | Path | None = None,
        # Input / output structure.
        n_obs_steps: int = 1,
        chunk_size: int = 7,
        n_action_steps: int = 7,
        action_dim: int = 7,
        state_dim: int = 8,
        *,
        # Backbones.
        qwen_model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
        jepa_encoder_name: str = "facebook/vjepa2-vitl-fpc64-256",
        freeze_qwen: bool = False,
        enable_world_model: bool = True,
        reinit_modules: list[str] | None = None,
        torch_dtype: str = "bfloat16",
        # Prompting.
        tokenizer_padding_side: str = "left",
        prompt_template: str | None = None,
        special_action_token: str = "<|action_{}|>",  # noqa: S107
        embodied_action_token: str = "<|embodied_action|>",  # noqa: S107
        # Normalization and action-space handling.
        state_normalization: str = "MEAN_STD",
        action_normalization: str = "MIN_MAX",
        use_relative_actions: bool = False,
        relative_exclude_joints: list[str] | None = None,
        action_feature_names: list[str] | None = None,
        # Action head.
        num_action_tokens_per_timestep: int = 8,
        num_embodied_action_tokens_per_instruction: int = 32,
        num_inference_timesteps: int = 4,
        action_hidden_size: int = 1024,
        action_model_type: str = "DiT-B",
        action_num_layers: int = 16,
        action_num_heads: int | None = None,
        action_attention_head_dim: int | None = None,
        action_dropout: float = 0.2,
        action_num_timestep_buckets: int = 1000,
        action_noise_beta_alpha: float = 1.5,
        action_noise_beta_beta: float = 1.0,
        action_noise_s: float = 0.999,
        action_max_seq_len: int = 1024,
        repeated_diffusion_steps: int = 8,
        # World model.
        num_video_frames: int = 8,
        predictor_depth: int = 12,
        predictor_num_heads: int = 8,
        predictor_mlp_ratio: float = 4.0,
        predictor_dropout: float = 0.0,
        world_model_loss_weight: float = 0.1,
        jepa_tubelet_size: int = 2,
        world_model_num_views: int | None = None,
        causal_world_model_context: bool = False,
        # Image and gripper handling.
        resize_images_to: tuple[int, int] | None = None,
        binarize_gripper_action: bool = False,
        pre_snap_gripper_action: bool = False,
        clip_normalized_actions: bool = True,
        gripper_dim: int = 6,
        gripper_threshold: float = 0.5,
        gripper_joint_names: list[str] | None = None,
        # Training presets.
        optimizer_lr: float = 1e-4,
        optimizer_betas: tuple[float, float] = (0.9, 0.95),
        optimizer_eps: float = 1e-8,
        optimizer_weight_decay: float = 1e-10,
        optimizer_grad_clip_norm: float = 10.0,
        scheduler_warmup_steps: int = 1_000,
        scheduler_decay_steps: int = 30_000,
        scheduler_decay_lr: float = 2.5e-6,
        # Eager initialization (for checkpoint loading).
        dataset_stats: dict[str, dict[str, list[float] | str | tuple]] | None = None,
    ) -> None:
        """Initialize the VLA-JEPA policy.

        Creates a :class:`VLAJEPAConfig` from the explicit arguments and saves it as hyperparameters.
        """
        super().__init__(n_action_steps=n_action_steps)

        config_kwargs: dict[str, Any] = {
            "n_obs_steps": n_obs_steps,
            "chunk_size": chunk_size,
            "n_action_steps": n_action_steps,
            "action_dim": action_dim,
            "state_dim": state_dim,
            "qwen_model_name": qwen_model_name,
            "jepa_encoder_name": jepa_encoder_name,
            "freeze_qwen": freeze_qwen,
            "enable_world_model": enable_world_model,
            "reinit_modules": reinit_modules,
            "torch_dtype": torch_dtype,
            "tokenizer_padding_side": tokenizer_padding_side,
            "special_action_token": special_action_token,
            "embodied_action_token": embodied_action_token,
            "state_normalization": state_normalization,
            "action_normalization": action_normalization,
            "use_relative_actions": use_relative_actions,
            "relative_exclude_joints": (["gripper"] if relative_exclude_joints is None else relative_exclude_joints),
            "action_feature_names": action_feature_names,
            "num_action_tokens_per_timestep": num_action_tokens_per_timestep,
            "num_embodied_action_tokens_per_instruction": num_embodied_action_tokens_per_instruction,
            "num_inference_timesteps": num_inference_timesteps,
            "action_hidden_size": action_hidden_size,
            "action_model_type": action_model_type,
            "action_num_layers": action_num_layers,
            "action_num_heads": action_num_heads,
            "action_attention_head_dim": action_attention_head_dim,
            "action_dropout": action_dropout,
            "action_num_timestep_buckets": action_num_timestep_buckets,
            "action_noise_beta_alpha": action_noise_beta_alpha,
            "action_noise_beta_beta": action_noise_beta_beta,
            "action_noise_s": action_noise_s,
            "action_max_seq_len": action_max_seq_len,
            "repeated_diffusion_steps": repeated_diffusion_steps,
            "num_video_frames": num_video_frames,
            "predictor_depth": predictor_depth,
            "predictor_num_heads": predictor_num_heads,
            "predictor_mlp_ratio": predictor_mlp_ratio,
            "predictor_dropout": predictor_dropout,
            "world_model_loss_weight": world_model_loss_weight,
            "jepa_tubelet_size": jepa_tubelet_size,
            "world_model_num_views": world_model_num_views,
            "causal_world_model_context": causal_world_model_context,
            "resize_images_to": resize_images_to,
            "binarize_gripper_action": binarize_gripper_action,
            "pre_snap_gripper_action": pre_snap_gripper_action,
            "clip_normalized_actions": clip_normalized_actions,
            "gripper_dim": gripper_dim,
            "gripper_threshold": gripper_threshold,
            "gripper_joint_names": ["gripper"] if gripper_joint_names is None else gripper_joint_names,
            "optimizer_lr": optimizer_lr,
            "optimizer_betas": optimizer_betas,
            "optimizer_eps": optimizer_eps,
            "optimizer_weight_decay": optimizer_weight_decay,
            "optimizer_grad_clip_norm": optimizer_grad_clip_norm,
            "scheduler_warmup_steps": scheduler_warmup_steps,
            "scheduler_decay_steps": scheduler_decay_steps,
            "scheduler_decay_lr": scheduler_decay_lr,
        }
        if prompt_template is not None:
            config_kwargs["prompt_template"] = prompt_template

        weights_file = None
        if pretrained_name_or_path is not None:
            self.config, dataset_stats, weights_file = self._from_hf(
                pretrained_name_or_path,
                config_kwargs,
                self._explicit_args,
            )
            # `super().__init__` above ran on the caller's `n_action_steps`; the checkpoint may
            # carry a different one, and the action queue is sized from it.
            if self.config.n_action_steps != self._n_action_steps:
                self._n_action_steps = self.config.n_action_steps
                self._action_queue = deque(maxlen=self._n_action_steps)
        else:
            self.config = VLAJEPAConfig(**config_kwargs)

        # Save config as hyperparameters for checkpoint restoration.
        self.save_hyperparameters(ignore=["config", "pretrained_name_or_path"])
        # Overwrite with the resolved self.config values.
        self._set_hparam_keys()

        # Model is built in setup(), or immediately when dataset_stats is provided.
        self.model: VLAJEPAModel | None = None
        self._preprocessor: VLAJEPAPreprocessor | None = None
        self._postprocessor: VLAJEPAPostprocessor | None = None

        if dataset_stats is not None:
            self._initialize_model(dataset_stats, weights_file)

        self._dataset_stats = dataset_stats

    def _set_hparam_keys(self) -> None:
        """Sync top-level checkpoint hparams from the resolved policy config."""
        for key, value in self.config.__dict__.items():
            if key not in self.hparams:
                continue
            self.hparams[key] = value
        self.hparams["config"] = self.config.to_dict()

    @staticmethod
    def _from_hf(
        pretrained_name_or_path: str | Path,
        config_kwargs: dict[str, Any],
        explicit_args: frozenset[str],
    ) -> tuple[VLAJEPAConfig, dict[str, dict[str, list[float] | str | tuple]] | None, Path | None]:
        """Load a pretrained VLA-JEPA config, dataset stats and weights.

        The checkpoint is the source of truth for every field it records. Architecture fields
        (backbone ids, chunk size, head and predictor geometry, prompt tokens) are baked into its
        tensor shapes and cannot be overridden at all; every other field - training presets,
        inference settings, gripper and relative-action handling - is overridden only when the
        caller passed it explicitly.

        Constructor defaults must not participate in that merge: they describe a from-scratch
        VLA-JEPA, not the published one. Letting them through silently replaced the LIBERO
        checkpoint's `resize_images_to`, `binarize_gripper_action` and `pre_snap_gripper_action`
        with from-scratch values, which fed the backbones the wrong input resolution and emitted
        the gripper in the wrong action space.

        Args:
            pretrained_name_or_path: HuggingFace repo id or local directory.
            config_kwargs: Constructor arguments, keyed like the config fields.
            explicit_args: Names of the constructor arguments the caller actually passed.

        Returns:
            Tuple of (config, dataset_stats, weights_file).
        """
        path = Path(pretrained_name_or_path)
        preprocessor_file: Path | None
        preprocessor_dir: Path | None

        if path.is_dir():
            config_file = path / "config.json"
            weights_file = path / "model.safetensors"
            candidate = path / "policy_preprocessor.json"
            preprocessor_file = candidate if candidate.exists() else None
            preprocessor_dir = path if preprocessor_file is not None else None
        else:
            repo_id = str(pretrained_name_or_path)
            config_file = Path(hf_hub_download(repo_id, "config.json"))  # nosec B615
            weights_file = Path(hf_hub_download(repo_id, "model.safetensors"))  # nosec B615
            preprocessor_file, preprocessor_dir = VLAJEPA._download_preprocessor(repo_id)

        with Path(config_file).open(encoding="utf-8") as f:
            hf_config = json.load(f)

        overrides = {
            key: value
            for key, value in config_kwargs.items()
            if key in explicit_args and key not in _PRETRAINED_ARCHITECTURE_FIELDS
        }
        ignored = sorted(explicit_args & _PRETRAINED_ARCHITECTURE_FIELDS)
        if ignored:
            logger.warning(
                "Ignoring %d explicitly passed argument(s) baked into the shapes of %s: %s",
                len(ignored),
                pretrained_name_or_path,
                ", ".join(ignored),
            )
        if overrides:
            logger.info("Overriding %s config with: %s", pretrained_name_or_path, ", ".join(sorted(overrides)))
        hf_config.update(overrides)

        dataset_stats = extract_dataset_stats(hf_config, preprocessor_file, preprocessor_dir)

        # strict=False: ignore legacy config.json keys not present in VLAJEPAConfig.
        config = VLAJEPAConfig.from_dict(hf_config, strict=False)

        return config, dataset_stats or None, weights_file

    @staticmethod
    def _download_preprocessor(repo_id: str) -> tuple[Path | None, Path | None]:
        """Download the preprocessor pipeline and its referenced state files.

        Args:
            repo_id: HuggingFace repo id.

        Returns:
            Tuple of (preprocessor json path, its directory), both None when the repo has none.
        """
        try:
            preprocessor_file = Path(hf_hub_download(repo_id, "policy_preprocessor.json"))  # nosec B615
            with preprocessor_file.open(encoding="utf-8") as f:
                preproc_data = json.load(f)
            for step in preproc_data.get("steps", []):
                if state_file := step.get("state_file"):
                    hf_hub_download(repo_id, state_file)  # nosec B615
        except Exception:  # noqa: BLE001
            logger.info("No policy_preprocessor.json found in %s; falling back to config.json stats.", repo_id)
            return None, None
        else:
            return preprocessor_file, preprocessor_file.parent

    def _initialize_model(
        self,
        dataset_stats: dict[str, dict[str, list[float] | str | tuple]],
        weights_file: Path | None = None,
    ) -> None:
        """Initialize the model and its pre/postprocessors.

        Called by both the lazy (setup) and eager (checkpoint) paths.

        Args:
            dataset_stats: Dataset normalization statistics.
            weights_file: Optional pretrained weights file.
        """
        self._resolve_feature_dims(dataset_stats)
        self.model = VLAJEPAModel(self.config, dataset_stats)

        if weights_file is not None:
            self._load_pretrained_weights(weights_file)

        self._update_preprocessor_stats(dataset_stats)

    def _load_pretrained_weights(self, weights_file: Path) -> None:
        """Load published VLA-JEPA weights into the model.

        Args:
            weights_file: Path to the ``model.safetensors`` file.
        """
        state_dict = fix_state_dict_keys(
            load_file(str(weights_file)),
            enable_world_model=self.config.enable_world_model,
        )
        current = self.model.state_dict()
        state_dict = drop_unused_module_keys(state_dict, current)
        filtered, reinitialized = filter_reinit_modules(
            state_dict,
            current,
            self.config.reinit_modules,
        )

        missing, unexpected = self.model.load_state_dict(filtered, strict=False)
        reinit_keys = {entry.split(":", 1)[0] for entry in reinitialized}
        missing = [key for key in missing if key not in reinit_keys]
        if missing:
            logger.warning("Missing keys when loading pretrained weights: %d keys", len(missing))
            for key in missing[:10]:
                logger.warning("  - %s", key)
        if unexpected:
            logger.warning("Unexpected keys when loading pretrained weights: %d keys", len(unexpected))
            for key in unexpected[:10]:
                logger.warning("  - %s", key)

    def _resolve_feature_dims(self, dataset_stats: dict[str, dict[str, Any]]) -> None:
        """Derive the action and state dimensionality from the dataset statistics.

        The action head's encoder and decoder are sized from these, so they must be resolved before
        the model is built.

        Args:
            dataset_stats: Dataset normalization statistics.
        """
        for key, stat in dataset_stats.items():
            shape = stat.get("shape")
            if not shape:
                continue
            if key == ACTION:
                object.__setattr__(self.config, "action_dim", int(shape[0]))  # noqa: PLC2801
            elif key == f"observation.{STATE}":
                object.__setattr__(self.config, "state_dim", int(shape[0]))  # noqa: PLC2801

    def _resolve_action_feature_names(self, train_dataset: Any) -> None:  # noqa: ANN401
        """Read the per-dimension action names from the dataset, when it exposes them.

        These resolve the gripper index and the relative-action exclusion mask. Studio's ``Feature``
        does not carry per-dimension names, so they are read from the underlying LeRobot features.

        Args:
            train_dataset: The training dataset.
        """
        if self.config.action_feature_names:
            return
        raw_features = getattr(train_dataset, "raw_features", None)
        if not isinstance(raw_features, dict):
            return
        names = (raw_features.get(ACTION) or {}).get("names")
        if isinstance(names, dict):
            names = next((value for value in names.values() if isinstance(value, list)), None)
        if isinstance(names, list) and names:
            object.__setattr__(  # noqa: PLC2801
                self.config,
                "action_feature_names",
                [str(name) for name in names],
            )

    def _update_preprocessor_stats(self, dataset_stats: dict[str, dict[str, Any]]) -> None:
        """Rebuild the pre- and postprocessors from dataset stats.

        Used on the fine-tuning path to replace pretrained normalization with training-data
        statistics, and by :meth:`_initialize_model` on the lazy path.

        Args:
            dataset_stats: Dataset normalization statistics.
        """
        from .preprocessor import make_vla_jepa_preprocessors  # noqa: PLC0415

        self._preprocessor, self._postprocessor = make_vla_jepa_preprocessors(self.config, dataset_stats)
        self._dataset_stats = dataset_stats
        self.hparams["dataset_stats"] = dataset_stats
        if self.model is not None:
            self.model.set_dataset_stats(dataset_stats)

    def setup(self, stage: str) -> None:
        """Set up the model from the datamodule (lazy initialization path).

        Called by Lightning before fit/validate/test/predict.

        Args:
            stage: Lightning stage (unused, required by the Lightning API).

        Raises:
            TypeError: If the train dataset is not a physicalai Dataset.
        """
        del stage  # Unused argument

        from physicalai.data.dataset import Dataset  # noqa: PLC0415

        datamodule = self.trainer.datamodule  # type: ignore[attr-defined]
        train_dataset = datamodule.train_dataset

        if not isinstance(train_dataset, Dataset):
            msg = f"Expected physicalai Dataset, got {type(train_dataset)}"
            raise TypeError(msg)

        stats_dict = train_dataset.stats
        self._resolve_action_feature_names(train_dataset)

        if self.model is not None:
            self._update_preprocessor_stats(stats_dict)
            reformat_dataset_to_match_policy(self, datamodule)
            return

        # Save to hparams for the checkpoint.
        self.hparams["dataset_stats"] = stats_dict

        self._initialize_model(stats_dict)

        reformat_dataset_to_match_policy(self, datamodule)

    def forward(self, batch: Observation) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        """Run the model on a batch.

        Args:
            batch: An Observation with the input data for the model.

        Returns:
            Training mode: the model output, a tuple of (loss, loss dict).
            Eval mode: the predicted action chunk.

        Raises:
            ValueError: If the model is not initialized during training mode.
        """
        if self.training:
            if self.model is None or self._preprocessor is None:
                msg = "Model is not initialized"
                raise ValueError(msg)

            processed_batch = self._preprocessor(batch.to_dict())
            return self.model(processed_batch)
        return self.predict_action_chunk(batch)

    @torch.no_grad()
    def predict_action_chunk(self, batch: Observation) -> torch.Tensor:
        """Predict a chunk of actions from an observation batch.

        Args:
            batch: An Observation with the input data for action prediction.

        Returns:
            The predicted action chunk, post-processed back into the dataset's action space.

        Raises:
            ValueError: If the model has not been initialized.
        """
        if self.model is None or self._preprocessor is None or self._postprocessor is None:
            msg = "Model is not initialized"
            raise ValueError(msg)

        processed_batch = self._preprocessor(batch.to(self.device).to_dict())
        chunk = self.model.predict_action_chunk(processed_batch)
        return self._postprocessor({ACTION: chunk})[ACTION]

    def training_step(self, batch: Observation, batch_idx: int) -> torch.Tensor:
        """Run a Lightning training step.

        Args:
            batch: Input batch.
            batch_idx: Batch index (unused, required by the Lightning API).

        Returns:
            Loss tensor for backpropagation.
        """
        del batch_idx
        loss, loss_dict = self(batch)

        self.log("train/loss", loss_dict["loss"], prog_bar=True)
        for key in ("action_loss", "wm_loss"):
            if key in loss_dict:
                self.log(f"train/{key}", loss_dict[key])

        return loss

    def compute_val_loss(self, batch: Observation) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        """Compute the validation loss on a batch.

        Args:
            batch: Observation batch (must contain ground-truth actions).

        Returns:
            Tuple of (loss tensor, loss dict).

        Raises:
            ValueError: If the model is not initialized.
        """
        if self.model is None or self._preprocessor is None:
            msg = "Model is not initialized"
            raise ValueError(msg)
        processed_batch = self._preprocessor(batch.to_dict())
        return self.model.compute_val_loss(processed_batch)

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure the optimizer and scheduler.

        Returns:
            Optimizer configuration dict.
        """
        params = [p for p in self.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW(
            params,
            lr=self.config.optimizer_lr,
            weight_decay=self.config.optimizer_weight_decay,
            betas=self.config.optimizer_betas,
            eps=self.config.optimizer_eps,
        )

        scheduler = cosine_decay_with_warmup_scheduler(
            optimizer,
            peak_lr=self.config.optimizer_lr,
            decay_lr=self.config.scheduler_decay_lr,
            num_warmup_steps=self.config.scheduler_warmup_steps,
            num_decay_steps=self.config.scheduler_decay_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        """Clip gradients using the policy config.

        Overrides Lightning's default clipping so the policy's ``optimizer_grad_clip_norm`` applies.

        Args:
            optimizer: The optimizer being used.
            gradient_clip_val: Trainer value; falls back to the config value when None.
            gradient_clip_algorithm: Clipping algorithm; defaults to 'norm'.
        """
        clip_val = gradient_clip_val if gradient_clip_val is not None else self.config.optimizer_grad_clip_norm

        if clip_val and clip_val > 0:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=clip_val,
                gradient_clip_algorithm=gradient_clip_algorithm or "norm",
            )

    @staticmethod
    def get_supported_export_backends() -> list[str | ExportBackend]:
        """Get a list of export backends supported by policy.

        Only the Torch backend is supported: it serializes the Lightning checkpoint and a manifest
        without tracing the graph. The Qwen chat-template tokenization and the variable-length
        vision-token sequence it produces are not traceable, so ONNX, OpenVINO and ExecuTorch are
        not offered.

        Returns:
            list[str | ExportBackend]: A list of supported export backends.
        """
        return [ExportBackend.TORCH]

    @property
    def inputs_schema(self) -> list[InferenceFeature] | None:
        """Describe the policy's expected model inputs for export.

        Returns:
            A list of feature descriptors covering the robot state, the image observations and the
            language task, or ``None`` when the model or the dataset stats are not initialized yet.
        """
        if self.model is None or self._dataset_stats is None:
            return None

        dataset_stats = self._dataset_stats
        schema: list[InferenceFeature] = []

        num_image_features = sum(1 for key in dataset_stats if str(FeatureType.VISUAL) in dataset_stats[key]["type"])

        for feature_id, feature in dataset_stats.items():
            if STATE in feature_id:
                schema.append(
                    InferenceFeature(
                        ftype=InferenceFeatureType.STATE,
                        shape=cast("tuple", feature["shape"]),
                        name=STATE,
                        dtype=InferenceFeatureDtype.FLOAT32,
                    ),
                )
            elif str(FeatureType.VISUAL) in feature["type"]:
                feature_name = (
                    str(feature.get("name", feature_id)).removeprefix("observation.").removeprefix(f"{IMAGES}.")
                )
                name = IMAGES if num_image_features == 1 else f"{IMAGES}.{feature_name}"
                # The preprocessor resizes every camera to `resize_images_to` before the Qwen
                # processor sees it, so the exported schema advertises the resized resolution
                # rather than the dataset's.
                shape = cast("tuple", feature["shape"])
                if self.config.resize_images_to is not None and len(shape) == 3:  # noqa: PLR2004
                    shape = (shape[0], *self.config.resize_images_to)
                schema.append(
                    InferenceFeature(
                        ftype=InferenceFeatureType.VISUAL,
                        shape=shape,
                        name=name,
                        dtype=InferenceFeatureDtype.FLOAT32,
                    ),
                )

        schema.append(
            InferenceFeature(
                ftype=InferenceFeatureType.LANGUAGE,
                # VLA-JEPA tokenizes the chat template with dynamic padding, so there is no fixed
                # token budget to advertise; the task arrives as a single string per sample.
                shape=(),
                name=TASK,
                dtype=InferenceFeatureDtype.STRING,
            ),
        )

        return schema

    @property
    def outputs_schema(self) -> list[InferenceFeature] | None:
        """Describe the policy's model output for export.

        Returns:
            A list with a single ``action`` feature of shape ``(chunk_size, *action_dim)``, or
            ``None`` when the model or the dataset stats are not initialized yet.
        """
        if self.model is None or self._dataset_stats is None:
            return None

        action_shape = cast("tuple", self._dataset_stats[ACTION]["shape"])

        return [
            InferenceFeature(
                ftype=InferenceFeatureType.ACTION,
                shape=(self.config.chunk_size, *action_shape),
                name=ACTION,
                dtype=InferenceFeatureDtype.FLOAT32,
            ),
        ]

    @property
    def extra_export_args(self) -> dict[str, ExportParameters]:
        """Additional export arguments for model conversion.

        The Torch backend reloads this policy from the checkpoint, so its own pre- and
        postprocessors (image resize, normalization, gripper and relative-action steps) run inside
        the model. The manifest therefore only carries the float cast on the way in and the action
        chunk trimmer on the way out.

        Returns:
            dict[str, ExportParameters]: A dictionary mapping backend names to their export
                parameters.

        Raises:
            ValueError: If dataset stats are not available for export argument construction.
        """
        if self._dataset_stats is None:
            msg = (
                "Dataset stats are required for export. Initialize the policy with dataset_stats"
                " or train for at least one epoch to populate them."
            )
            raise ValueError(msg)

        postprocessors_specs = []
        if self.config.chunk_size != self.config.n_action_steps:
            postprocessors_specs.append(
                ComponentSpec(type="action_chunk_trimmer", n_action_steps=self.config.n_action_steps),
            )

        return {
            "torch": TorchExportParameters(
                preprocessors_specs=[ComponentSpec(type="to_float_tensor")],
                postprocessors_specs=postprocessors_specs,
            ),
        }
