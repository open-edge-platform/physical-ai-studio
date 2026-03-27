# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pi05 Policy - Lightning wrapper for training and inference."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from physicalai.data.dataset import Dataset
from physicalai.data.observation import ACTION
from physicalai.export import ExportablePolicyMixin, ExportBackend
from physicalai.policies.base import Policy
from physicalai.train.utils import reformat_dataset_to_match_policy

from .config import Pi05Config
from .model import Pi05Model
from .preprocessor import make_pi05_preprocessors
from .pretrained_utils import extract_dataset_stats as _extract_dataset_stats
from .pretrained_utils import fix_state_dict_keys as _fix_state_dict_keys

if TYPE_CHECKING:
    from physicalai.data import Observation

    from .preprocessor import Pi05Postprocessor, Pi05Preprocessor

logger = logging.getLogger(__name__)


class Pi05(ExportablePolicyMixin, Policy):
    """Pi05 Policy - Physical Intelligence's flow matching VLA model.

    Lightning wrapper for training and inference with Pi05 model.

    Uses dual-path initialization:
    - **Lazy path**: `Pi05()` + `trainer.fit()` - model built in setup()
    - **Eager path**: `Pi05.load_from_checkpoint()` - model built immediately

    Args:
        pretrained_name_or_path: HuggingFace repo ID or local path for pretrained weights and config.
        paligemma_variant: Gemma variant for VLM backbone. Default: "gemma_2b".
        action_expert_variant: Gemma variant for action expert. Default: "gemma_300m".
        dtype: Model precision. Default: "float32".
        n_obs_steps: Number of observation steps. Default: 1.
        chunk_size: Size of action chunks. Default: 50.
        n_action_steps: Number of action steps to execute. Default: 50.
        max_state_dim: Maximum state dimension (padded). Default: 32.
        max_action_dim: Maximum action dimension (padded). Default: 32.
        num_inference_steps: Denoising steps for inference. Default: 10.
        image_resolution: Target image resolution. Default: (224, 224).
        tokenizer_max_length: Maximum tokenizer length. Default: 200.
        gradient_checkpointing: Enable gradient checkpointing. Default: False.
        freeze_vision_encoder: Freeze vision encoder. Default: False.
        train_expert_only: Train only action expert. Default: True.
        optimizer_lr: Learning rate. Default: 2.5e-5.
        dataset_stats: Dataset stats for eager initialization. Default: None.

    Example:
        Training:

        >>> policy = Pi05(optimizer_lr=2.5e-5)
        >>> trainer = physicalai.train.Trainer(max_epochs=100)
        >>> trainer.fit(policy, datamodule)

        Inference:

        >>> policy = Pi05.load_from_checkpoint("checkpoint.ckpt")
        >>> action = policy.select_action(obs)
    """

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        # Pretrained model id
        pretrained_name_or_path: str | Path | None = None,
        # Model architecture
        paligemma_variant: Literal["gemma_300m", "gemma_2b"] = "gemma_2b",
        action_expert_variant: Literal["gemma_300m", "gemma_2b"] = "gemma_300m",
        dtype: Literal["bfloat16", "float32"] = "float32",
        # Input / output structure
        n_obs_steps: int = 1,
        chunk_size: int = 50,
        n_action_steps: int = 50,
        max_state_dim: int = 32,
        max_action_dim: int = 32,
        # Flow matching
        num_inference_steps: int = 10,
        time_sampling_beta_alpha: float = 1.5,
        time_sampling_beta_beta: float = 1.0,
        time_sampling_scale: float = 0.999,
        time_sampling_offset: float = 0.001,
        min_period: float = 4e-3,
        max_period: float = 4.0,
        # Image preprocessing
        image_resolution: tuple[int, int] = (224, 224),
        empty_cameras: int = 0,
        # Tokenizer
        tokenizer_max_length: int = 200,
        # Optimization
        *,
        gradient_checkpointing: bool = False,
        compile_model: bool = False,
        compile_mode: str = "max-autotune",
        # Finetuning
        freeze_vision_encoder: bool = False,
        train_expert_only: bool = True,
        # Optimizer
        optimizer_lr: float = 2.5e-5,
        optimizer_betas: tuple[float, float] = (0.9, 0.95),
        optimizer_eps: float = 1e-8,
        optimizer_weight_decay: float = 0.01,
        optimizer_grad_clip_norm: float = 1.0,
        # Scheduler
        scheduler_warmup_steps: int = 1_000,
        scheduler_decay_steps: int = 30_000,
        scheduler_decay_lr: float = 2.5e-6,
        # Eager initialization
        dataset_stats: dict[str, dict[str, list[float] | str | tuple]] | None = None,
    ) -> None:
        """Initialize Pi05 policy."""
        super().__init__(n_action_steps=n_action_steps)

        weight_file = None
        if pretrained_name_or_path is not None:
            self.config, dataset_stats, weight_file = self._from_hf(
                pretrained_name_or_path,
                n_action_steps=n_action_steps,
                num_inference_steps=num_inference_steps,
                compile_model=compile_model,
                compile_mode=compile_mode,
            )
        else:
            self.config = Pi05Config(
                paligemma_variant=paligemma_variant,
                action_expert_variant=action_expert_variant,
                dtype=dtype,
                n_obs_steps=n_obs_steps,
                chunk_size=chunk_size,
                n_action_steps=n_action_steps,
                max_state_dim=max_state_dim,
                max_action_dim=max_action_dim,
                num_inference_steps=num_inference_steps,
                time_sampling_beta_alpha=time_sampling_beta_alpha,
                time_sampling_beta_beta=time_sampling_beta_beta,
                time_sampling_scale=time_sampling_scale,
                time_sampling_offset=time_sampling_offset,
                min_period=min_period,
                max_period=max_period,
                image_resolution=image_resolution,
                empty_cameras=empty_cameras,
                tokenizer_max_length=tokenizer_max_length,
                gradient_checkpointing=gradient_checkpointing,
                compile_model=compile_model,
                compile_mode=compile_mode,
                freeze_vision_encoder=freeze_vision_encoder,
                train_expert_only=train_expert_only,
                optimizer_lr=optimizer_lr,
                optimizer_betas=optimizer_betas,
                optimizer_eps=optimizer_eps,
                optimizer_weight_decay=optimizer_weight_decay,
                optimizer_grad_clip_norm=optimizer_grad_clip_norm,
                scheduler_warmup_steps=scheduler_warmup_steps,
                scheduler_decay_steps=scheduler_decay_steps,
                scheduler_decay_lr=scheduler_decay_lr,
            )

        self.save_hyperparameters(ignore=["config"])
        self.hparams["config"] = self.config.to_dict()

        self.model: Pi05Model | None = None

        self._preprocessor: Pi05Preprocessor | None = None
        self._postprocessor: Pi05Postprocessor | None = None

        self._dataset_stats = dataset_stats

        if dataset_stats is not None:
            self._initialize_model(dataset_stats, weight_file)

    def _initialize_model(
        self,
        dataset_stats: dict[str, dict[str, list[float] | str | tuple]],
        weights_file: Path | None = None,
    ) -> None:
        """Initialize model and preprocessors.

        Called by both lazy (setup) and eager (checkpoint) paths.
        """
        self.model = Pi05Model(
            dataset_stats,
            paligemma_variant=self.config.paligemma_variant,
            action_expert_variant=self.config.action_expert_variant,
            dtype=self.config.dtype,
            chunk_size=self.config.chunk_size,
            max_action_dim=self.config.max_action_dim,
            n_action_steps=self.config.n_action_steps,
            num_inference_steps=self.config.num_inference_steps,
            time_sampling_beta_alpha=self.config.time_sampling_beta_alpha,
            time_sampling_beta_beta=self.config.time_sampling_beta_beta,
            time_sampling_scale=self.config.time_sampling_scale,
            time_sampling_offset=self.config.time_sampling_offset,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            image_resolution=self.config.image_resolution,
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            compile_model=self.config.compile_model,
        )
        if weights_file is not None:
            # load raw state dict
            original_sd = load_file(str(weights_file))

            # fix keys (same logic as lerobot's _fix_pytorch_state_dict_keys)
            fixed_sd = _fix_state_dict_keys(original_sd)

            # load into model
            missing, unexpected = self.model.load_state_dict(fixed_sd, strict=False, assign=True)
            if missing:
                msg = f"Missing keys when loading pretrained weights: {len(missing)} keys"
                logger.warning(msg)
                for k in missing[:10]:
                    msg = f"  - {k}"
                    logger.warning(msg)
            if unexpected:
                msg = f"Unexpected keys when loading pretrained weights: {len(unexpected)} keys"
                logger.warning(msg)
                for k in unexpected[:10]:
                    msg = f"  - {k}"
                    logger.warning(msg)

            # Apply dtype/precision
            self.model.paligemma_with_expert.to_bfloat16_for_selected_params(self.config.dtype)
            self.model.paligemma_with_expert._set_requires_grad()  # noqa: SLF001

        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self._preprocessor, self._postprocessor = make_pi05_preprocessors(
            max_state_dim=self.config.max_state_dim,
            max_action_dim=self.config.max_action_dim,
            stats=dataset_stats,
            image_resolution=self.config.image_resolution,
            max_token_len=self.config.tokenizer_max_length,
            empty_cameras=self.config.empty_cameras,
        )

        self._dataset_stats = dataset_stats

    def _from_hf(  # noqa: PLR6301
        self,
        pretrained_name_or_path: str | Path,
        *,
        n_action_steps: int | None = 10,
        num_inference_steps: int | None = None,
        compile_model: bool = False,
        compile_mode: str | None = "max-autotune",
        **kwargs: Any,  # noqa: ANN401
    ) -> tuple[Pi05Config, dict[str, dict[str, list[float] | str | tuple]], Path]:
        """Load pretrained Pi05 from a HuggingFace model repo.

        Loads weights from a HuggingFace model ID (e.g. ``lerobot/pi05_libero_finetuned``)
        or a local directory containing ``config.json`` and ``model.safetensors``.

        Handles the key remapping and normalization stat conversion
        from the lerobot QUANTILES format (q01/q99) to MEAN_STD (mean/std).

        Args:
            pretrained_name_or_path: HuggingFace repo ID or local path.
            n_action_steps: Override number of action steps to execute.
            num_inference_steps: Override denoising steps for inference.
            compile_model: Override whether to use torch.compile.
            compile_mode: Override torch compile mode.
            device: Device to place the model on after loading.
            **kwargs: Extra arguments forwarded to ``huggingface_hub.hf_hub_download``.

        Returns:
            Tuple of (config_kwargs, dataset_stats, weights_file).
             - config_kwargs: Dict of arguments to construct Pi05Config.
             - dataset_stats: Dict of dataset stats for preprocessor construction.
             - weights_file: Path to the downloaded weights file.
        """
        path = Path(pretrained_name_or_path)
        is_local = path.is_dir()

        # --- resolve files (local or hub) ---
        if is_local:
            config_file = path / "config.json"
            weights_file = path / "model.safetensors"
            preprocessor_file = path / "policy_preprocessor.json"
            preprocessor_dir = path
        else:
            hub_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k
                in {
                    "cache_dir",
                    "force_download",
                    "resume_download",
                    "proxies",
                    "token",
                    "revision",
                    "local_files_only",
                }
            }
            config_file = Path(hf_hub_download(pretrained_name_or_path, "config.json", **hub_kwargs))  # nosec B615
            weights_file = Path(hf_hub_download(pretrained_name_or_path, "model.safetensors", **hub_kwargs))  # nosec B615
            try:
                preprocessor_file = Path(
                    hf_hub_download(pretrained_name_or_path, "policy_preprocessor.json", **hub_kwargs),  # nosec B615
                )
                preprocessor_dir = preprocessor_file.parent

                # Also download referenced state files
                with Path(preprocessor_file).open(encoding="utf-8") as f:
                    preproc_data = json.load(f)
                for step in preproc_data.get("steps", []):
                    sf = step.get("state_file")
                    if sf:
                        hf_hub_download(pretrained_name_or_path, sf, **hub_kwargs)  # nosec B615
            except Exception:  # noqa: BLE001
                preprocessor_file = None
                preprocessor_dir = None

        # --- parse config.json ---
        with Path(config_file).open(encoding="utf-8") as f:
            hf_config = json.load(f)

        # from_dict skips unknown keys and coerces lists→tuples via type hints
        config_kwargs = Pi05Config.from_dict(hf_config).to_dict()

        # Allow caller overrides
        if n_action_steps is not None:
            config_kwargs["n_action_steps"] = n_action_steps
        if num_inference_steps is not None:
            config_kwargs["num_inference_steps"] = num_inference_steps
        if compile_model is not None:
            config_kwargs["compile_model"] = compile_model
        if compile_mode is not None:
            config_kwargs["compile_mode"] = compile_mode
        config = Pi05Config(**config_kwargs)

        # --- build dataset_stats from HF artefacts ---
        dataset_stats = _extract_dataset_stats(hf_config, preprocessor_file, preprocessor_dir)

        return config, dataset_stats, weights_file

    def setup(self, stage: str) -> None:
        """Set up model from datamodule (lazy initialization path).

        Called by Lightning before fit/validate/test/predict.

        Raises:
            TypeError: If the train dataset is not a physicalai Dataset.
        """
        del stage

        if self.model is not None:
            return

        datamodule = self.trainer.datamodule  # type: ignore[attr-defined]
        train_dataset = datamodule.train_dataset

        if not isinstance(train_dataset, Dataset):
            msg = f"Expected physicalai Dataset, got {type(train_dataset)}"
            raise TypeError(msg)

        stats_dict = train_dataset.stats

        self.hparams["dataset_stats"] = stats_dict

        self._initialize_model(stats_dict)

        reformat_dataset_to_match_policy(self, datamodule)

    def forward(self, batch: Observation) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        """Forward pass through the model.

        Training mode: returns (loss, loss_dict).
        Eval mode: returns action chunk predictions.

        Returns:
            Loss tuple in training mode, or action tensor in eval mode.

        Raises:
            ValueError: If the model is not initialized.
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
        """Predict a chunk of actions from observation.

        Args:
            batch: Input observation batch.

        Returns:
            Action chunk tensor after post-processing.

        Raises:
            ValueError: If the model is not initialized.
        """
        if self.model is None or self._preprocessor is None or self._postprocessor is None:
            msg = "Model is not initialized"
            raise ValueError(msg)

        processed_batch = self._preprocessor(batch.to(self.device).to_dict())
        actions = self.model.predict_action_chunk(processed_batch)

        return self._postprocessor({ACTION: actions})[ACTION]

    def training_step(self, batch: Observation, batch_idx: int) -> torch.Tensor:
        """Lightning training step.

        Returns:
            Training loss tensor.
        """
        del batch_idx
        loss, loss_dict = self(batch)
        self.log("train/loss", loss_dict["loss"], prog_bar=True)
        return loss

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and scheduler.

        Returns:
            Dict with optimizer and lr_scheduler config.
        """
        params = [p for p in self.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW(
            params,
            lr=self.config.optimizer_lr,
            weight_decay=self.config.optimizer_weight_decay,
            betas=self.config.optimizer_betas,
            eps=self.config.optimizer_eps,
        )

        warmup_steps = self.config.scheduler_warmup_steps
        drop_steps = self.config.scheduler_decay_steps
        decay_value = self.config.scheduler_decay_lr

        decay_ratio = decay_value / self.config.optimizer_lr

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            num_drops = (step - warmup_steps) // drop_steps
            return decay_ratio**num_drops

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

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
        """Configure gradient clipping from policy config."""
        clip_val = gradient_clip_val if gradient_clip_val is not None else self.config.optimizer_grad_clip_norm

        if clip_val and clip_val > 0:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=clip_val,
                gradient_clip_algorithm=gradient_clip_algorithm or "norm",
            )

    @property
    def supported_export_backends(self) -> list[str | ExportBackend]:
        """Get a list of export backends supported by policy.

        This method returns a list of supported export backends as strings.

        Returns:
            list[str | ExportBackend]: A list of supported export backends.
        """
        return [ExportBackend.TORCH, ExportBackend.OPENVINO, ExportBackend.ONNX]
