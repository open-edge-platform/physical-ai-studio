# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Copyright 2026 The HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0

"""EO-1 Policy - Lightning wrapper for training and inference."""

from __future__ import annotations

import json
import logging
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

from .config import EO1Config
from .model import EO1Model
from .pretrained_utils import drop_tied_missing_keys, extract_dataset_stats, fix_state_dict_keys

if TYPE_CHECKING:
    from physicalai.data import Observation

    from .preprocessor import EO1Postprocessor, EO1Preprocessor

logger = logging.getLogger(__name__)

# Fields that describe the *architecture*: they are baked into the published checkpoint shapes, or
# into what the flow head was trained against, so they are always taken from its config.json rather
# than from the caller's defaults.
_PRETRAINED_ARCHITECTURE_FIELDS = frozenset({
    "action_act",
    "action_dim",
    "chunk_size",
    "max_action_dim",
    "max_period",
    "max_state_dim",
    "min_period",
    "num_action_layers",
    "state_dim",
    "vlm_base",
    "vlm_config",
})


class EO1(ExportablePolicyMixin, Policy):
    """EO-1 Policy - Qwen2.5-VL backbone with a continuous flow-matching action head.

    Lightning wrapper for training and inference with the EO-1 model.

    Uses dual-path initialization:
    - **Lazy path**: `EO1()` + `trainer.fit()` - model built in setup()
    - **Eager path**: `EO1.load_from_checkpoint()` - model built immediately

    Export is limited to the Torch backend. ``to_torch`` serializes the Lightning checkpoint and a
    manifest, so it needs no tracing. Graph-capturing backends (ONNX, OpenVINO, ExecuTorch) are
    intentionally unsupported: the Qwen chat-template tokenization and the variable-length
    vision-token sequence it produces are not traceable.

    Args:
        pretrained_name_or_path: HuggingFace repo id or local path holding a published EO-1
            ``config.json`` / ``model.safetensors``. Architecture fields are read from it; runtime
            and training fields still come from this constructor.
        dataset_stats: Dataset normalization statistics for eager initialization.

    See :class:`~physicalai.policies.eo1.EO1Config` for every other argument.

    Example:
        Training:

        >>> policy = EO1(chunk_size=8, n_action_steps=8)
        >>> trainer = physicalai.Trainer(max_epochs=100)
        >>> trainer.fit(policy, datamodule)

        Inference:

        >>> policy = EO1.load_from_checkpoint("checkpoint.ckpt")
        >>> action = policy.select_action(obs)
    """

    model: Any
    _preprocessor: Any

    def __init__(  # noqa: PLR0913
        self,
        pretrained_name_or_path: str | Path | None = None,
        # Input / output structure.
        n_obs_steps: int = 1,
        chunk_size: int = 8,
        n_action_steps: int = 8,
        action_dim: int = 7,
        state_dim: int = 8,
        *,
        # Backbone.
        vlm_base: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        vlm_config: dict[str, Any] | None = None,
        attn_implementation: str | None = None,
        dtype: str = "auto",
        force_fp32_autocast: bool = True,
        gradient_checkpointing: bool = False,
        # Vision processing.
        image_min_pixels: int | None = 64 * 28 * 28,
        image_max_pixels: int | None = 128 * 28 * 28,
        use_fast_processor: bool = False,
        # Flow head.
        max_state_dim: int = 32,
        max_action_dim: int = 32,
        num_denoise_steps: int = 10,
        num_action_layers: int = 2,
        action_act: str = "linear",
        time_sampling_beta_alpha: float = 1.5,
        time_sampling_beta_beta: float = 1.0,
        time_sampling_scale: float = 0.999,
        time_sampling_offset: float = 0.001,
        min_period: float = 4e-3,
        max_period: float = 4.0,
        supervise_padding_action_dims: bool = True,
        supervise_padding_actions: bool = True,
        # Normalization.
        state_normalization: str = "MEAN_STD",
        action_normalization: str = "MEAN_STD",
        # Training presets.
        optimizer_lr: float = 1e-4,
        optimizer_betas: tuple[float, float] = (0.9, 0.999),
        optimizer_eps: float = 1e-8,
        optimizer_weight_decay: float = 0.1,
        optimizer_grad_clip_norm: float = 1.0,
        scheduler_warmup_steps: int = 900,
        scheduler_decay_steps: int = 30_000,
        scheduler_decay_lr: float = 0.0,
        # Eager initialization (for checkpoint loading).
        dataset_stats: dict[str, dict[str, list[float] | str | tuple]] | None = None,
    ) -> None:
        """Initialize the EO-1 policy.

        Creates an :class:`EO1Config` from the explicit arguments and saves it as hyperparameters.
        """
        super().__init__(n_action_steps=n_action_steps)

        config_kwargs: dict[str, Any] = {
            "n_obs_steps": n_obs_steps,
            "chunk_size": chunk_size,
            "n_action_steps": n_action_steps,
            "action_dim": action_dim,
            "state_dim": state_dim,
            "vlm_base": vlm_base,
            "vlm_config": vlm_config,
            "attn_implementation": attn_implementation,
            "dtype": dtype,
            "force_fp32_autocast": force_fp32_autocast,
            "gradient_checkpointing": gradient_checkpointing,
            "image_min_pixels": image_min_pixels,
            "image_max_pixels": image_max_pixels,
            "use_fast_processor": use_fast_processor,
            "max_state_dim": max_state_dim,
            "max_action_dim": max_action_dim,
            "num_denoise_steps": num_denoise_steps,
            "num_action_layers": num_action_layers,
            "action_act": action_act,
            "time_sampling_beta_alpha": time_sampling_beta_alpha,
            "time_sampling_beta_beta": time_sampling_beta_beta,
            "time_sampling_scale": time_sampling_scale,
            "time_sampling_offset": time_sampling_offset,
            "min_period": min_period,
            "max_period": max_period,
            "supervise_padding_action_dims": supervise_padding_action_dims,
            "supervise_padding_actions": supervise_padding_actions,
            "state_normalization": state_normalization,
            "action_normalization": action_normalization,
            "optimizer_lr": optimizer_lr,
            "optimizer_betas": optimizer_betas,
            "optimizer_eps": optimizer_eps,
            "optimizer_weight_decay": optimizer_weight_decay,
            "optimizer_grad_clip_norm": optimizer_grad_clip_norm,
            "scheduler_warmup_steps": scheduler_warmup_steps,
            "scheduler_decay_steps": scheduler_decay_steps,
            "scheduler_decay_lr": scheduler_decay_lr,
        }

        weights_file = None
        if pretrained_name_or_path is not None:
            self.config, dataset_stats, weights_file = self._from_hf(pretrained_name_or_path, config_kwargs)
        else:
            self.config = EO1Config(**config_kwargs)

        # Save config as hyperparameters for checkpoint restoration.
        self.save_hyperparameters(ignore=["config", "pretrained_name_or_path"])
        # Overwrite with the resolved self.config values.
        self._set_hparam_keys()

        # Model is built in setup(), or immediately when dataset_stats is provided.
        self.model: EO1Model | None = None
        self._preprocessor: EO1Preprocessor | None = None
        self._postprocessor: EO1Postprocessor | None = None

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
    ) -> tuple[EO1Config, dict[str, dict[str, list[float] | str | tuple]] | None, Path | None]:
        """Load a pretrained EO-1 config, dataset stats and weights.

        Architecture fields (backbone id and config, padded feature widths, chunk size, projector
        geometry, timestep-embedding periods) come from the checkpoint. Everything else - training
        presets, inference settings, normalization choices - comes from the caller.

        Args:
            pretrained_name_or_path: HuggingFace repo id or local directory.
            config_kwargs: Constructor arguments, applied on top of the pretrained config.

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
            preprocessor_file, preprocessor_dir = EO1._download_preprocessor(repo_id)

        with Path(config_file).open(encoding="utf-8") as f:
            hf_config = json.load(f)

        hf_config.update({
            key: value for key, value in config_kwargs.items() if key not in _PRETRAINED_ARCHITECTURE_FIELDS
        })

        dataset_stats = extract_dataset_stats(hf_config, preprocessor_file, preprocessor_dir)

        # strict=False: ignore LeRobot config.json keys not present in EO1Config.
        config = EO1Config.from_dict(hf_config, strict=False)

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
        self.model = EO1Model(self.config, dataset_stats)

        if weights_file is not None:
            self._load_pretrained_weights(weights_file)

        self._update_preprocessor_stats(dataset_stats)

    def _load_pretrained_weights(self, weights_file: Path) -> None:
        """Load published EO-1 weights into the model.

        Args:
            weights_file: Path to the ``model.safetensors`` file.
        """
        state_dict = fix_state_dict_keys(load_file(str(weights_file)))
        current = self.model.state_dict()

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        missing = drop_tied_missing_keys(missing, set(state_dict), current)
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

        EO-1 pads both to `max_action_dim` / `max_state_dim` before the flow head, so these do not
        change any tensor shape. They decide how far predicted chunks are cropped back and how wide
        the normalization buffers are.

        Args:
            dataset_stats: Dataset normalization statistics.

        Raises:
            ValueError: If the dataset's action or state is wider than the padded width the flow
                head is built for.
        """
        for key, stat in dataset_stats.items():
            shape = stat.get("shape")
            if not shape:
                continue
            if key == ACTION:
                object.__setattr__(self.config, "action_dim", int(shape[0]))  # noqa: PLC2801
            elif key == f"observation.{STATE}":
                object.__setattr__(self.config, "state_dim", int(shape[0]))  # noqa: PLC2801

        if self.config.action_dim > self.config.max_action_dim:
            msg = (
                f"The dataset's action is {self.config.action_dim}-dimensional but the flow head is "
                f"built for `max_action_dim={self.config.max_action_dim}`. Raise `max_action_dim` "
                f"(note that doing so changes the checkpoint tensor shapes)."
            )
            raise ValueError(msg)
        if self.config.state_dim > self.config.max_state_dim:
            msg = (
                f"The dataset's state is {self.config.state_dim}-dimensional but the state "
                f"projection is built for `max_state_dim={self.config.max_state_dim}`. Raise "
                f"`max_state_dim` (note that doing so changes the checkpoint tensor shapes)."
            )
            raise ValueError(msg)

    def _update_preprocessor_stats(self, dataset_stats: dict[str, dict[str, Any]]) -> None:
        """Rebuild the pre- and postprocessors from dataset stats.

        Used on the fine-tuning path to replace pretrained normalization with training-data
        statistics, and by :meth:`_initialize_model` on the lazy path.

        Args:
            dataset_stats: Dataset normalization statistics.
        """
        from .preprocessor import make_eo1_preprocessors  # noqa: PLC0415

        self._preprocessor, self._postprocessor = make_eo1_preprocessors(self.config, dataset_stats)
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
                # EO-1 does not resize to a fixed resolution: the Qwen image processor rescales each
                # camera to the `image_min_pixels`/`image_max_pixels` budget, so the schema
                # advertises the dataset's own resolution.
                schema.append(
                    InferenceFeature(
                        ftype=InferenceFeatureType.VISUAL,
                        shape=cast("tuple", feature["shape"]),
                        name=name,
                        dtype=InferenceFeatureDtype.FLOAT32,
                    ),
                )

        schema.append(
            InferenceFeature(
                ftype=InferenceFeatureType.LANGUAGE,
                # EO-1 tokenizes the chat template with dynamic padding, so there is no fixed token
                # budget to advertise; the task arrives as a single string per sample.
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
        postprocessors (normalization, denormalization) run inside the model. The manifest therefore
        only carries the float cast on the way in and the action chunk trimmer on the way out.

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
