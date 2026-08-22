# Copyright 2025 2toINF (https://github.com/2toINF) and The HuggingFace Inc. team.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""XVLA policy - Lightning wrapper for training, inference and Torch export.

Only the Torch backend is supported. ``to_torch()`` serializes the policy's state dict and
hyper-parameters, so it needs no traced graph and works for any XVLA the caller can build --
including one whose Florence-2 backbone comes from a caller-supplied ``florence_config``
(``{"vision_config": ..., "text_config": ...}``) rather than a downloaded checkpoint. The
graph backends (ONNX, OpenVINO, ExecuTorch) stay unsupported: the Florence-2 encoder runs
only the camera views a per-batch mask marks valid, which makes the traced shapes depend on
the data rather than on the config.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal, cast

import torch
from physicalai.inference.data import InferenceFeature, InferenceFeatureDtype, InferenceFeatureType
from physicalai.inference.manifest import ComponentSpec

from physicalai.data.dataset import Dataset
from physicalai.data.observation import ACTION, IMAGES, STATE, TASK, FeatureType
from physicalai.export import ExportablePolicyMixin, ExportBackend
from physicalai.export.backends import ExportParameters, TorchExportParameters
from physicalai.policies.base import Policy
from physicalai.train.schedulers import cosine_decay_with_warmup_scheduler
from physicalai.train.utils import reformat_dataset_to_match_policy

from .config import XVLAConfig
from .model import XVLAModel
from .preprocessor import make_xvla_preprocessors, resolve_num_image_views
from .pretrained_utils import (
    detect_normalization_mode,
    extract_dataset_stats,
    extract_domain_id,
    extract_tokenizer_max_length,
    load_config,
    load_xvla_weights,
    read_action_dim,
    resolve_checkpoint,
)

if TYPE_CHECKING:
    from pathlib import Path

    from physicalai.data import Observation

    from .preprocessor import XVLAPostprocessor, XVLAPreprocessor

logger = logging.getLogger(__name__)

_VLM_PARAM_PREFIX = "model.vlm."
_SOFT_PROMPT_MARKER = "soft_prompt"


class XVLA(ExportablePolicyMixin, Policy):
    """XVLA - a cross-embodiment flow-matching vision-language-action policy.

    A Florence-2 encoder conditions a soft-prompted transformer that denoises a whole
    action chunk at once. One checkpoint serves many robots: a per-sample ``domain_id``
    selects the domain-aware projections and soft prompts, and ``action_mode`` picks the
    action space that maps the model's fixed-width action vector onto the embodiment at
    hand.

    Uses the same dual-path initialization as the other Studio policies:

    - **Lazy path**: ``XVLA()`` + ``trainer.fit()`` - the model is built in ``setup()`` from
      the datamodule's statistics, which is also where ``action_mode="auto"`` learns the
      dataset's action width.
    - **Eager path**: ``XVLA(pretrained_name_or_path=...)`` or ``XVLA.load_from_checkpoint(...)``
      - the model is built immediately.

    Export covers the Torch backend only; see
    :meth:`get_supported_export_backends` for why the graph backends are excluded.

    Args:
        pretrained_name_or_path: HuggingFace repo id or local directory of a published XVLA
            checkpoint. Its ``config.json`` supplies the architecture; the arguments below
            override it only where the caller changed them from their defaults.
        florence_config: Architecture of the Florence-2 backbone, in either the native
            ``transformers`` format or the legacy remote-code format used by published
            checkpoints. Empty means the ``transformers`` defaults (Florence-2 base).
        tokenizer_name: HuggingFace tokenizer for the language prompt. Default: "facebook/bart-large".
        tokenizer_max_length: Fixed prompt length every call pads to. When loading a
            pretrained checkpoint, its published preprocessor manifest overrides this value
            rather than the checkpoint's own ``config.json`` field, which is not reliably
            the length its processor pipeline actually used. Default: 64.
        dtype: Model precision. Default: "float32".
        n_obs_steps: Number of observation steps. Default: 1.
        chunk_size: Action steps predicted per forward pass. Default: 32.
        n_action_steps: Action steps executed per chunk. Default: 32.
        hidden_size: Width of the action transformer. Default: 1024.
        depth: Number of transformer blocks. Default: 24.
        num_heads: Number of attention heads. Default: 16.
        mlp_ratio: Feed-forward expansion factor. Default: 4.0.
        num_domains: Embodiments the domain-aware layers can serve. Default: 30.
        len_soft_prompts: Learned prompt tokens per domain. Default: 32.
        dim_time: Width of the timestep features. Default: 32.
        max_len_seq: Longest sequence the positional embedding covers. Default: 512.
        use_hetero_proj: Project the visual streams per domain. Default: False.
        action_mode: Action space name; ``"auto"`` adapts to the dataset. Default: "auto".
        num_denoising_steps: Euler steps used at inference. Default: 10.
        use_proprio: Feed the proprioceptive state to the model. Default: True.
        max_state_dim: Width the state is padded to. Default: 32.
        max_action_dim: Action width the model predicts under ``"auto"``. Default: 20.
        domain_id: Domain index used when the batch carries none. Default: 0.
        domain_feature_key: Batch key holding a per-sample domain index. Default: None.
        resize_imgs_with_padding: Resize cameras to ``(height, width)``. Default: None.
        num_image_views: Camera slots the model expects. Default: None (derive from data).
        empty_cameras: Masked-out camera slots appended to the real ones. Default: 0.
        freeze_vision_encoder: Freeze the Florence-2 vision tower. Default: False.
        freeze_language_encoder: Freeze the Florence-2 text encoder. Default: False.
        train_policy_transformer: Train the action transformer's backbone. Default: True.
        train_soft_prompts: Train the per-domain soft prompts. Default: True.
        normalization_mode: State/action normalization. Default: "IDENTITY".
        optimizer_lr: Base learning rate. Default: 1e-4.
        optimizer_betas: Adam beta coefficients. Default: (0.9, 0.99).
        optimizer_eps: Optimizer epsilon. Default: 1e-8.
        optimizer_weight_decay: Weight decay coefficient. Default: 0.0.
        optimizer_grad_clip_norm: Maximum gradient norm. Default: 10.0.
        optimizer_vlm_lr_scale: Learning-rate multiplier for the Florence-2 parameters.
            Default: 0.1.
        optimizer_soft_prompt_lr_scale: Learning-rate multiplier for the soft prompts.
            Default: 1.0.
        scheduler_warmup_steps: Linear warmup steps. Default: 1000.
        scheduler_decay_steps: Cosine decay horizon; ``None`` auto-scales. Default: 30000.
        scheduler_decay_lr: Final learning rate after decay. Default: 2.5e-6.
        dataset_stats: Dataset statistics for eager initialization (checkpoint restore).

    Example:
        Training:

        >>> policy = XVLA(action_mode="auto")  # doctest: +SKIP
        >>> trainer = physicalai.train.Trainer(max_epochs=10)  # doctest: +SKIP
        >>> trainer.fit(policy, datamodule)  # doctest: +SKIP

        Finetuning a published checkpoint:

        >>> policy = XVLA(pretrained_name_or_path="lerobot/xvla_libero")  # doctest: +SKIP
        >>> action = policy.select_action(observation)  # doctest: +SKIP

        Exporting, including a backbone the caller sized themselves:

        >>> policy = XVLA(  # doctest: +SKIP
        ...     florence_config={"vision_config": {"projection_dim": 32}},
        ...     dataset_stats=dataset_stats,
        ... )
        >>> policy.to_torch("exported/")  # doctest: +SKIP
    """

    def __init__(  # noqa: PLR0913
        self,
        pretrained_name_or_path: str | Path | None = None,
        *,
        # Florence-2 backbone and tokenizer
        florence_config: dict[str, Any] | None = None,
        tokenizer_name: str = "facebook/bart-large",
        tokenizer_max_length: int = 64,
        dtype: Literal["bfloat16", "float32"] = "float32",
        # Input / output structure
        n_obs_steps: int = 1,
        chunk_size: int = 32,
        n_action_steps: int = 32,
        # Action transformer
        hidden_size: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_domains: int = 30,
        len_soft_prompts: int = 32,
        dim_time: int = 32,
        max_len_seq: int = 512,
        use_hetero_proj: bool = False,
        # Action space and proprioception
        action_mode: str = "auto",
        num_denoising_steps: int = 10,
        use_proprio: bool = True,
        max_state_dim: int = 32,
        max_action_dim: int = 20,
        domain_id: int = 0,
        domain_feature_key: str | None = None,
        # Vision preprocessing
        resize_imgs_with_padding: tuple[int, int] | None = None,
        num_image_views: int | None = None,
        empty_cameras: int = 0,
        # Finetuning
        freeze_vision_encoder: bool = False,
        freeze_language_encoder: bool = False,
        train_policy_transformer: bool = True,
        train_soft_prompts: bool = True,
        # Normalization
        normalization_mode: Literal["IDENTITY", "MEAN_STD", "QUANTILES"] = "IDENTITY",
        # Optimizer / scheduler
        optimizer_lr: float = 1e-4,
        optimizer_betas: tuple[float, float] = (0.9, 0.99),
        optimizer_eps: float = 1e-8,
        optimizer_weight_decay: float = 0.0,
        optimizer_grad_clip_norm: float = 10.0,
        optimizer_vlm_lr_scale: float = 0.1,
        optimizer_soft_prompt_lr_scale: float = 1.0,
        scheduler_warmup_steps: int = 1_000,
        scheduler_decay_steps: int | None = 30_000,
        scheduler_decay_lr: float = 2.5e-6,
        # Eager initialization
        dataset_stats: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Initialize the policy, resolving a pretrained checkpoint when one is given."""
        config_kwargs: dict[str, Any] = {
            "florence_config": florence_config or {},
            "tokenizer_name": tokenizer_name,
            "tokenizer_max_length": tokenizer_max_length,
            "dtype": dtype,
            "n_obs_steps": n_obs_steps,
            "chunk_size": chunk_size,
            "n_action_steps": n_action_steps,
            "hidden_size": hidden_size,
            "depth": depth,
            "num_heads": num_heads,
            "mlp_ratio": mlp_ratio,
            "num_domains": num_domains,
            "len_soft_prompts": len_soft_prompts,
            "dim_time": dim_time,
            "max_len_seq": max_len_seq,
            "use_hetero_proj": use_hetero_proj,
            "action_mode": action_mode,
            "num_denoising_steps": num_denoising_steps,
            "use_proprio": use_proprio,
            "max_state_dim": max_state_dim,
            "max_action_dim": max_action_dim,
            "domain_id": domain_id,
            "domain_feature_key": domain_feature_key,
            "resize_imgs_with_padding": resize_imgs_with_padding,
            "num_image_views": num_image_views,
            "empty_cameras": empty_cameras,
            "freeze_vision_encoder": freeze_vision_encoder,
            "freeze_language_encoder": freeze_language_encoder,
            "train_policy_transformer": train_policy_transformer,
            "train_soft_prompts": train_soft_prompts,
            "normalization_mode": normalization_mode,
            "optimizer_lr": optimizer_lr,
            "optimizer_betas": optimizer_betas,
            "optimizer_eps": optimizer_eps,
            "optimizer_weight_decay": optimizer_weight_decay,
            "optimizer_grad_clip_norm": optimizer_grad_clip_norm,
            "optimizer_vlm_lr_scale": optimizer_vlm_lr_scale,
            "optimizer_soft_prompt_lr_scale": optimizer_soft_prompt_lr_scale,
            "scheduler_warmup_steps": scheduler_warmup_steps,
            "scheduler_decay_steps": scheduler_decay_steps,
            "scheduler_decay_lr": scheduler_decay_lr,
        }

        weights_file: Path | None = None
        action_dim: int | None = None
        if pretrained_name_or_path is not None:
            config, dataset_stats, weights_file, action_dim = self._from_hf(
                pretrained_name_or_path,
                config_kwargs,
                dataset_stats,
            )
        else:
            config = XVLAConfig(**config_kwargs)

        super().__init__(n_action_steps=config.n_action_steps)
        self.config = config

        self.save_hyperparameters(ignore=["config", "pretrained_name_or_path"])
        self._set_hparam_keys()

        self.model: XVLAModel | None = None  # pyrefly: ignore[bad-override-mutable-attribute]
        self._preprocessor: XVLAPreprocessor | None = None  # pyrefly: ignore[bad-override-mutable-attribute]
        self._postprocessor: XVLAPostprocessor | None = None
        self._dataset_stats = dataset_stats
        self._action_dim = action_dim

        if dataset_stats is not None or weights_file is not None:
            self._initialize_model(dataset_stats or {}, weights_file)

    # ------------------------------------------------------------------ #
    # Construction                                                        #
    # ------------------------------------------------------------------ #
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
        overrides: dict[str, Any],
        dataset_stats: dict[str, dict[str, Any]] | None,
    ) -> tuple[XVLAConfig, dict[str, dict[str, Any]] | None, Path, int | None]:
        """Resolve a published checkpoint into a config, statistics and a weights file.

        Only arguments the caller actually changed from the constructor defaults override
        the checkpoint's own values, so loading a checkpoint does not silently reshape the
        architecture it was trained with.

        Args:
            pretrained_name_or_path: HuggingFace repo id or local directory.
            overrides: The constructor's configuration arguments.
            dataset_stats: Caller-supplied statistics, which win over the checkpoint's.

        Returns:
            Tuple of ``(config, dataset_stats, weights_file, action_dim)``.
        """
        files = resolve_checkpoint(pretrained_name_or_path)

        defaults = XVLAConfig()
        explicit = {key: value for key, value in overrides.items() if getattr(defaults, key, object()) != value}

        detected = detect_normalization_mode(files.processor_files)
        if detected is not None and "normalization_mode" not in explicit:
            explicit["normalization_mode"] = detected

        # Upstream stores this in a preprocessor step, not config.json: getting it wrong
        # silently selects a different domain's action decoder and soft prompts.
        detected_domain_id = extract_domain_id(files.processor_files)
        if detected_domain_id is not None and "domain_id" not in explicit:
            explicit["domain_id"] = detected_domain_id

        # config.json's own tokenizer_max_length can be a stale value the published
        # processor pipeline never actually used; the preprocessor manifest is authoritative
        # (see extract_tokenizer_max_length -- getting this wrong desyncs every token after
        # the prompt from the position the model was trained to see it at).
        detected_max_length = extract_tokenizer_max_length(files.processor_files)
        if detected_max_length is not None and "tokenizer_max_length" not in explicit:
            explicit["tokenizer_max_length"] = detected_max_length

        config = load_config(files.config_file, explicit)
        if dataset_stats is None:
            dataset_stats = extract_dataset_stats(files.processor_files, files.processor_dir) or None
        return config, dataset_stats, files.weights_file, read_action_dim(files.config_file)

    def _initialize_model(
        self,
        dataset_stats: dict[str, dict[str, Any]],
        weights_file: Path | None = None,
    ) -> None:
        """Build the model and the pre/post-processors.

        Args:
            dataset_stats: Dataset statistics used for normalization, the camera count and
                (under ``action_mode="auto"``) the dataset's action width.
            weights_file: Optional ``model.safetensors`` to load into the model.
        """
        self._action_dim = _action_dim_from_stats(dataset_stats) or self._action_dim
        self.model = XVLAModel(self.config, action_dim=self._action_dim)

        if weights_file is not None:
            load_xvla_weights(self.model, weights_file)
            self.model.apply_dtype()

        self._update_processor_stats(dataset_stats)

    def _update_processor_stats(self, dataset_stats: dict[str, dict[str, Any]]) -> None:
        """Rebuild the pre/post-processors from a new set of statistics.

        Args:
            dataset_stats: Statistics from a checkpoint or a training dataset.
        """
        self._preprocessor, self._postprocessor = make_xvla_preprocessors(
            dataset_stats,
            num_image_views=resolve_num_image_views(
                dataset_stats,
                num_image_views=self.config.num_image_views,
                empty_cameras=self.config.empty_cameras,
            ),
            image_resolution=self.config.resize_imgs_with_padding,
            max_state_dim=self.config.dim_proprio,
            tokenizer_name=self.config.tokenizer_name,
            tokenizer_max_length=self.config.tokenizer_max_length,
            domain_id=self.config.domain_id,
            domain_feature_key=self.config.domain_feature_key,
            normalization_mode=self.config.normalization_mode,
        )
        self._dataset_stats = dataset_stats
        self.hparams["dataset_stats"] = dataset_stats

    def setup(self, stage: str) -> None:
        """Build the model from the datamodule when it was not created eagerly.

        Args:
            stage: The Lightning stage (unused).

        Raises:
            TypeError: If the train dataset is not a physicalai ``Dataset``.
        """
        del stage

        datamodule = self.trainer.datamodule  # type: ignore[attr-defined]
        train_dataset = datamodule.train_dataset

        if not isinstance(train_dataset, Dataset):
            msg = f"Expected physicalai Dataset, got {type(train_dataset)}"
            raise TypeError(msg)

        stats_dict = train_dataset.stats

        if self.model is None:
            self._initialize_model(stats_dict)
        else:
            # Finetuning: keep the pretrained weights but re-fit the processors, and the
            # width of the auto action space, to the new dataset.
            logger.info("Updating XVLA processors for the finetuning dataset")
            action_dim = _action_dim_from_stats(stats_dict)
            if action_dim is not None:
                self._action_dim = action_dim
                self.model.set_action_dim(action_dim)
            self._update_processor_stats(stats_dict)

        reformat_dataset_to_match_policy(self, datamodule)

    @property
    def inner_model(self) -> XVLAModel:
        """The unwrapped XVLA network.

        Raises:
            RuntimeError: If accessed before the model was initialized.
        """
        if self.model is None:
            msg = "inner_model accessed before the model was initialized (setup() has not run yet)."
            raise RuntimeError(msg)
        return self.model

    # ------------------------------------------------------------------ #
    # Training                                                            #
    # ------------------------------------------------------------------ #
    def forward(self, batch: Observation) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        """Run the training loss in training mode, or predict actions in eval mode.

        Args:
            batch: Input observation batch.

        Returns:
            ``(loss, loss_dict)`` while training, else the predicted action chunk.
        """
        if self.training:
            return self.inner_model.compute_loss(self._preprocess(batch))
        return self.predict_action_chunk(batch)

    def training_step(self, batch: Observation, batch_idx: int) -> torch.Tensor:
        """Lightning training step.

        Args:
            batch: Input observation batch.
            batch_idx: Index of the batch (unused).

        Returns:
            The training loss.
        """
        del batch_idx
        loss, loss_dict = self.inner_model.compute_loss(self._preprocess(batch))
        self.log("train/loss", loss_dict["loss"], prog_bar=True)
        for name, value in loss_dict.items():
            if name != "loss":
                self.log(f"train/{name}", value)
        return loss

    def compute_val_loss(self, batch: Observation) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        """Compute the action-prediction MSE on a validation batch.

        Args:
            batch: Observation batch holding ground-truth actions.

        Returns:
            Tuple of ``(loss, loss_dict)``.
        """
        return self.inner_model.compute_val_loss(self._preprocess(batch))

    def configure_optimizers(self) -> dict[str, Any]:  # pyrefly: ignore[bad-override]
        """Configure AdamW with XVLA's differential learning rates.

        The Florence-2 parameters train at ``optimizer_vlm_lr_scale`` of the base rate (a
        tenth, upstream) with the weight decay scaled the same way, the soft prompts at
        ``optimizer_soft_prompt_lr_scale``, and everything else at the full rate. The
        cosine schedule then applies one shared multiplier, so each group keeps its
        relative rate throughout training.

        Returns:
            Dict with the optimizer and its step-interval scheduler.
        """
        groups: dict[str, list[torch.nn.Parameter]] = {"vlm": [], "soft_prompts": [], "other": []}
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith(_VLM_PARAM_PREFIX):
                groups["vlm"].append(param)
            elif _SOFT_PROMPT_MARKER in name:
                groups["soft_prompts"].append(param)
            else:
                groups["other"].append(param)

        base_lr = self.config.optimizer_lr
        weight_decay = self.config.optimizer_weight_decay
        param_groups = [
            {
                "params": groups["vlm"],
                "lr": base_lr * self.config.optimizer_vlm_lr_scale,
                "weight_decay": weight_decay * self.config.optimizer_vlm_lr_scale,
                "name": "vlm",
            },
            {
                "params": groups["soft_prompts"],
                "lr": base_lr * self.config.optimizer_soft_prompt_lr_scale,
                "weight_decay": weight_decay,
                "name": "soft_prompts",
            },
            {
                "params": groups["other"],
                "lr": base_lr,
                "weight_decay": weight_decay,
                "name": "other",
            },
        ]

        optimizer = torch.optim.AdamW(
            [group for group in param_groups if group["params"]],
            lr=base_lr,
            betas=self.config.optimizer_betas,
            eps=self.config.optimizer_eps,
            weight_decay=weight_decay,
        )

        num_training_steps = self.trainer.estimated_stepping_batches
        num_decay_steps = self.config.scheduler_decay_steps
        if num_decay_steps is None:
            num_decay_steps = num_training_steps
            logger.info("scheduler_decay_steps=None, using total training steps: %s", num_decay_steps)

        scheduler = cosine_decay_with_warmup_scheduler(
            optimizer,
            peak_lr=base_lr,
            decay_lr=self.config.scheduler_decay_lr,
            num_warmup_steps=self.config.scheduler_warmup_steps,
            num_decay_steps=num_decay_steps,  # pyrefly: ignore[bad-argument-type]
            num_training_steps=num_training_steps,  # pyrefly: ignore[bad-argument-type]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        """Clip gradients using the policy's configured norm.

        Args:
            optimizer: The optimizer being stepped.
            gradient_clip_val: Trainer-provided clip value, if any.
            gradient_clip_algorithm: Trainer-provided clip algorithm, if any.
        """
        clip_val = gradient_clip_val if gradient_clip_val is not None else self.config.optimizer_grad_clip_norm
        if clip_val and clip_val > 0:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=clip_val,
                gradient_clip_algorithm=gradient_clip_algorithm or "norm",
            )

    # ------------------------------------------------------------------ #
    # Inference                                                           #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict_action_chunk(self, batch: Observation) -> torch.Tensor:
        """Predict one action chunk.

        Args:
            batch: Input observation batch.

        Returns:
            Actions of shape ``[B, chunk_size, D]`` in the dataset's units. ``D`` is the
            action space's emitted width -- the dataset's action width under
            ``action_mode="auto"``. ``select_action`` executes the first ``n_action_steps``
            of the chunk before asking for the next one.

        Raises:
            ValueError: If the postprocessor was not initialized.
        """
        if self._postprocessor is None:
            msg = "Model is not initialized"
            raise ValueError(msg)
        actions = self.inner_model.predict_action_chunk(self._preprocess(batch))
        return self._postprocessor({ACTION: actions})[ACTION]

    def _preprocess(self, batch: Observation) -> dict[str, Any]:
        """Move a batch to the policy device and run it through the preprocessor.

        Args:
            batch: Input observation batch.

        Returns:
            The flattened, preprocessed batch dict.

        Raises:
            ValueError: If the preprocessor was not initialized.
        """
        if self._preprocessor is None:
            msg = "Model is not initialized"
            raise ValueError(msg)
        return self._preprocessor(batch.to(self.device).to_dict())

    # ------------------------------------------------------------------ #
    # Export                                                              #
    # ------------------------------------------------------------------ #
    @staticmethod
    def get_supported_export_backends() -> list[str | ExportBackend]:
        """List the export backends XVLA supports.

        Only Torch: ``to_torch()`` writes the state dict and the hyper-parameters, which
        needs no traced graph, so it works for any XVLA that can be built at all -- a
        published checkpoint, a finetune, or a backbone the caller sized themselves through
        ``florence_config``. The graph backends are excluded because the Florence-2 encoder
        runs only the camera views the per-batch mask marks valid, which makes the traced
        shapes depend on the data rather than on the config.

        Returns:
            The supported backends.
        """
        return [ExportBackend.TORCH]

    @property
    def inputs_schema(self) -> list[InferenceFeature] | None:
        """Describe the observations the exported policy consumes.

        Returns:
            One feature per camera the dataset statistics declare, the proprioceptive state
            (when ``use_proprio`` is set) and the language prompt. ``None`` before the model
            and its statistics exist, since the camera set is only known then.
        """
        if self.model is None or not self._dataset_stats:
            return None

        schema: list[InferenceFeature] = []
        num_cameras = sum(
            1 for stat in self._dataset_stats.values() if str(FeatureType.VISUAL) in str(stat.get("type", ""))
        )

        for feature_id, stat in self._dataset_stats.items():
            if str(FeatureType.VISUAL) in str(stat.get("type", "")):
                camera = str(stat.get("name", feature_id)).removeprefix("observation.").removeprefix(f"{IMAGES}.")
                schema.append(
                    InferenceFeature(
                        ftype=InferenceFeatureType.VISUAL,
                        shape=cast("tuple", tuple(stat["shape"])),
                        name=IMAGES if num_cameras == 1 else f"{IMAGES}.{camera}",
                        dtype=InferenceFeatureDtype.FLOAT32,
                    ),
                )
            elif STATE in feature_id and self.config.use_proprio:
                schema.append(
                    InferenceFeature(
                        ftype=InferenceFeatureType.STATE,
                        shape=cast("tuple", tuple(stat["shape"])),
                        name=STATE,
                        dtype=InferenceFeatureDtype.FLOAT32,
                    ),
                )

        schema.append(
            InferenceFeature(
                ftype=InferenceFeatureType.LANGUAGE,
                shape=(),
                name=TASK,
                dtype=InferenceFeatureDtype.STRING,
            ),
        )
        return schema

    @property
    def outputs_schema(self) -> list[InferenceFeature] | None:
        """Describe the action chunk the exported policy emits.

        Returns:
            A single ``action`` feature of shape ``(chunk_size, action_dim)``, where
            ``action_dim`` is the width the configured action space emits -- the dataset's
            own width under ``action_mode="auto"``. ``None`` before the model exists.
        """
        if self.model is None:
            return None

        action_space = self.inner_model.action_space
        action_dim = getattr(action_space, "real_dim", 0) or self.inner_model.dim_action

        return [
            InferenceFeature(
                ftype=InferenceFeatureType.ACTION,
                shape=(self.config.chunk_size, action_dim),
                name=ACTION,
                dtype=InferenceFeatureDtype.FLOAT32,
            ),
        ]

    @property
    def extra_export_args(self) -> dict[str, ExportParameters]:
        """Describe the pre/post-processing the Torch runner has to reproduce.

        The exported checkpoint carries XVLA's own pre/postprocessor, so the manifest only
        has to declare the camera conversion in front of it and the chunk trimming behind
        it, which the runner applies outside the policy.

        Returns:
            The export parameters, keyed by backend name.
        """
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


def _action_dim_from_stats(stats: dict[str, dict[str, Any]] | None) -> int | None:
    """Read the dataset's action width out of its statistics.

    Args:
        stats: Dataset statistics keyed by feature name.

    Returns:
        The action width, or ``None`` when the statistics declare no action shape.
    """
    if not stats:
        return None
    entry = next(
        (stat for name, stat in stats.items() if name == ACTION or name.rsplit(".", 1)[-1] == ACTION),
        None,
    )
    shape = entry.get("shape") if entry else None
    return int(shape[-1]) if shape else None


__all__ = ["XVLA"]
