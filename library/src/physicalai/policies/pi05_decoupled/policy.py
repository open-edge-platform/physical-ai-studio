# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0

"""PI05 Policy - Lightning wrapper for training and inference."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from physicalai.data.observation import ACTION, IMAGES
from physicalai.data.dataset import Dataset
from physicalai.policies.base import Policy
from physicalai.train.utils import reformat_dataset_to_match_policy

from .config import PI05Config
from .model import PI05Model
from .preprocessor import make_pi05_preprocessors

if TYPE_CHECKING:
    from physicalai.data import Observation

    from .preprocessor import PI05Postprocessor, PI05Preprocessor

logger = logging.getLogger(__name__)


class PI05(Policy):
    """PI05 Policy - Physical Intelligence's flow matching VLA model.

    Lightning wrapper for training and inference with PI05 model.

    Uses dual-path initialization:
    - **Lazy path**: `PI05()` + `trainer.fit()` - model built in setup()
    - **Eager path**: `PI05.load_from_checkpoint()` - model built immediately

    Args:
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
        train_expert_only: Train only action expert. Default: False.
        optimizer_lr: Learning rate. Default: 2.5e-5.
        dataset_stats: Dataset stats for eager initialization. Default: None.

    Example:
        Training:

        >>> policy = PI05(optimizer_lr=2.5e-5)
        >>> trainer = physicalai.Trainer(max_epochs=100)
        >>> trainer.fit(policy, datamodule)

        Inference:

        >>> policy = PI05.load_from_checkpoint("checkpoint.ckpt")
        >>> action = policy.select_action(obs)
    """

    def __init__(  # noqa: PLR0913
        self,
        # Model architecture
        paligemma_variant: str = "gemma_2b",
        action_expert_variant: str = "gemma_300m",
        dtype: str = "float32",
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
        train_expert_only: bool = False,
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
        super().__init__(n_action_steps=n_action_steps)

        self.config = PI05Config(
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

        self.model: PI05Model | None = None

        self._preprocessor: PI05Preprocessor | None = None
        self._postprocessor: PI05Postprocessor | None = None

        self._dataset_stats = dataset_stats

        if dataset_stats is not None:
            self._initialize_model(dataset_stats)

    def _initialize_model(
        self,
        dataset_stats: dict[str, dict[str, list[float] | str | tuple]],
    ) -> None:
        """Initialize model and preprocessors.

        Called by both lazy (setup) and eager (checkpoint) paths.
        """
        self.model = PI05Model(self.config)

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

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        n_action_steps: int | None = None,
        num_inference_steps: int | None = None,
        compile_model: bool | None = None,
        compile_mode: str | None = None,
        device: str | torch.device = "cpu",
        **kwargs: Any,
    ) -> PI05:
        """Load pretrained PI05 from a HuggingFace model repo.

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
            Initialized PI05 policy with loaded weights.

        Example:
            >>> policy = PI05.from_pretrained("lerobot/pi05_libero_finetuned")
            >>> policy = PI05.from_pretrained(
            ...     "lerobot/pi05_libero_finetuned",
            ...     n_action_steps=10,
            ...     device="cuda",
            ... )
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
                in (
                    "cache_dir",
                    "force_download",
                    "resume_download",
                    "proxies",
                    "token",
                    "revision",
                    "local_files_only",
                )
            }
            config_file = Path(hf_hub_download(pretrained_name_or_path, "config.json", **hub_kwargs))
            weights_file = Path(hf_hub_download(pretrained_name_or_path, "model.safetensors", **hub_kwargs))
            try:
                preprocessor_file = Path(
                    hf_hub_download(pretrained_name_or_path, "policy_preprocessor.json", **hub_kwargs),
                )
                preprocessor_dir = preprocessor_file.parent

                # Also download referenced state files
                with open(preprocessor_file) as f:
                    _preproc = json.load(f)
                for _step in _preproc.get("steps", []):
                    sf = _step.get("state_file")
                    if sf:
                        hf_hub_download(pretrained_name_or_path, sf, **hub_kwargs)
            except Exception:
                preprocessor_file = None
                preprocessor_dir = None

        # --- parse config.json ---
        with open(config_file) as f:
            hf_config = json.load(f)

        config_kwargs: dict[str, Any] = {}
        _FIELD_MAP = {
            "paligemma_variant": "paligemma_variant",
            "action_expert_variant": "action_expert_variant",
            "dtype": "dtype",
            "n_obs_steps": "n_obs_steps",
            "chunk_size": "chunk_size",
            "n_action_steps": "n_action_steps",
            "max_state_dim": "max_state_dim",
            "max_action_dim": "max_action_dim",
            "num_inference_steps": "num_inference_steps",
            "time_sampling_beta_alpha": "time_sampling_beta_alpha",
            "time_sampling_beta_beta": "time_sampling_beta_beta",
            "time_sampling_scale": "time_sampling_scale",
            "time_sampling_offset": "time_sampling_offset",
            "min_period": "min_period",
            "max_period": "max_period",
            "empty_cameras": "empty_cameras",
            "tokenizer_max_length": "tokenizer_max_length",
            "gradient_checkpointing": "gradient_checkpointing",
            "freeze_vision_encoder": "freeze_vision_encoder",
            "train_expert_only": "train_expert_only",
        }
        for hf_key, our_key in _FIELD_MAP.items():
            if hf_key in hf_config:
                config_kwargs[our_key] = hf_config[hf_key]

        if "image_resolution" in hf_config:
            res = hf_config["image_resolution"]
            config_kwargs["image_resolution"] = tuple(res) if isinstance(res, list) else res

        # Allow caller overrides
        if n_action_steps is not None:
            config_kwargs["n_action_steps"] = n_action_steps
        if num_inference_steps is not None:
            config_kwargs["num_inference_steps"] = num_inference_steps
        if compile_model is not None:
            config_kwargs["compile_model"] = compile_model
        if compile_mode is not None:
            config_kwargs["compile_mode"] = compile_mode

        # --- build dataset_stats from HF artefacts ---
        dataset_stats = cls._extract_dataset_stats(hf_config, preprocessor_file, preprocessor_dir)

        # --- create policy with full initialization (no meta device) ---
        config_kwargs["dataset_stats"] = dataset_stats
        policy = cls(**config_kwargs)

        # load raw state dict
        original_sd = load_file(str(weights_file))

        # fix keys (same logic as lerobot's _fix_pytorch_state_dict_keys)
        fixed_sd = cls._fix_state_dict_keys(original_sd)

        # load into model
        missing, unexpected = policy.model.load_state_dict(fixed_sd, strict=False, assign=True)
        if missing:
            logger.warning("Missing keys when loading pretrained weights: %d keys", len(missing))
            for k in missing[:10]:
                logger.warning("  - %s", k)
        if unexpected:
            logger.warning("Unexpected keys when loading pretrained weights: %d keys", len(unexpected))
            for k in unexpected[:10]:
                logger.warning("  - %s", k)

        # Apply dtype/precision
        policy.model.paligemma_with_expert.to_bfloat16_for_selected_params(policy.config.dtype)
        policy.model.paligemma_with_expert._set_requires_grad()

        policy.to(device)
        policy.eval()
        logger.info("Loaded PI05 from %s", pretrained_name_or_path)
        return policy

    # ------------------------------------------------------------------
    # Private helpers for from_pretrained
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_dataset_stats(
        hf_config: dict[str, Any],
        preprocessor_file: Path | None,
        preprocessor_dir: Path | None,
    ) -> dict[str, dict[str, Any]]:
        """Build ``dataset_stats`` dict that ``make_pi05_preprocessors`` expects.

        The stats format expected by the decoupled preprocessor is:
        ``{feature_name: {"name": str, "shape": tuple, "mean": list, "std": list}}``

        Lerobot models use QUANTILES normalization with q01/q99 stats.
        We convert to mean/std that produces an equivalent [-1, 1] mapping:
        ``mean = (q01 + q99) / 2``  and  ``std = (q99 - q01) / 2``.
        """
        stats: dict[str, dict[str, Any]] = {}

        # Try to extract stats from preprocessor config + state file
        if preprocessor_file is not None and preprocessor_file.exists():
            try:
                with open(preprocessor_file) as f:
                    preproc_config = json.load(f)
                stats = PI05._parse_preprocessor_stats(preproc_config, hf_config, preprocessor_dir)
                if stats:
                    return stats
            except Exception:
                logger.debug("Could not parse preprocessor file, falling back to config.json")

        # Fallback: build minimal stats from config.json features
        stats = PI05._parse_config_features(hf_config)
        return stats

    @staticmethod
    def _parse_preprocessor_stats(
        preproc_config: dict[str, Any],
        hf_config: dict[str, Any],
        preprocessor_dir: Path | None,
    ) -> dict[str, dict[str, Any]]:
        """Extract normalization stats from lerobot's policy_preprocessor.json.

        Lerobot stores the stats in a separate safetensors file (referenced
        by ``state_file`` in each pipeline step). The keys are flat:
        ``"observation.state.q01"``, ``"action.q99"``, etc.
        """

        stats: dict[str, dict[str, Any]] = {}

        steps = preproc_config.get("steps", [])
        if isinstance(steps, dict):
            steps = list(steps.values())

        for step in steps:
            step_type = step.get("registry_name", step.get("type", step.get("class_name", "")))
            if "normalizer" not in step_type.lower():
                continue

            state_file = step.get("state_file")
            if not state_file or preprocessor_dir is None:
                continue

            state_path = preprocessor_dir / state_file
            if not state_path.exists():
                logger.warning("Normalizer state file not found: %s", state_path)
                continue

            # Load the flat tensor dict: {"observation.state.q01": tensor, ...}
            tensor_stats = load_file(str(state_path))

            # Group by feature name: "observation.state.q01" → key="observation.state", stat="q01"
            grouped: dict[str, dict[str, list[float]]] = {}
            for flat_key, tensor in tensor_stats.items():
                feat_name, stat_name = flat_key.rsplit(".", 1)
                grouped.setdefault(feat_name, {})[stat_name] = tensor.cpu().tolist()

            # Convert each feature's stats to mean/std
            for feat_name, feat_stats in grouped.items():
                shape = PI05._resolve_feature_shape(feat_name, hf_config, feat_stats)
                mean, std = PI05._convert_normalization_stats(feat_stats)
                if mean is not None and std is not None:
                    stats[feat_name] = {
                        "name": feat_name,
                        "shape": shape,
                        "mean": mean,
                        "std": std,
                    }

        return stats

    @staticmethod
    def _parse_config_features(
        hf_config: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        """Build stats from config.json ``input_features``/``output_features``.

        When no preprocessor stats file is available, build identity-like stats
        from feature shapes in the config.
        """
        stats: dict[str, dict[str, Any]] = {}

        for section in ("input_features", "output_features"):
            features = hf_config.get(section, {})
            if not isinstance(features, dict):
                continue
            for feat_name, feat_info in features.items():
                if not isinstance(feat_info, dict):
                    continue
                shape = feat_info.get("shape", None)
                if shape is None:
                    continue
                shape = tuple(shape)
                dim = shape[0] if shape else 1

                if "state" in feat_name.lower() or feat_name == ACTION or "action" in feat_name.lower():
                    stats[feat_name] = {
                        "name": feat_name,
                        "shape": shape,
                        "mean": [0.0] * dim,
                        "std": [1.0] * dim,
                    }

        return stats

    @staticmethod
    def _resolve_feature_shape(
        feat_name: str,
        hf_config: dict[str, Any],
        feat_stats: dict[str, Any],
    ) -> tuple[int, ...]:
        """Resolve shape for a feature, checking config features then stat tensors."""
        # Check config features
        for section in ("input_features", "output_features"):
            features = hf_config.get(section, {})
            if isinstance(features, dict) and feat_name in features:
                feat_info = features[feat_name]
                if isinstance(feat_info, dict) and "shape" in feat_info:
                    return tuple(feat_info["shape"])

        # Infer from stats tensor lengths
        for key in ("mean", "std", "q01", "q99", "min", "max"):
            val = feat_stats.get(key)
            if isinstance(val, list):
                return (len(val),)

        return (1,)

    @staticmethod
    def _convert_normalization_stats(
        feat_stats: dict[str, Any],
    ) -> tuple[list[float] | None, list[float] | None]:
        """Convert lerobot normalization stats to mean/std.

        Handles QUANTILES (q01/q99), MIN_MAX (min/max), and MEAN_STD (mean/std).
        QUANTILES conversion: mean = (q01+q99)/2, std = (q99-q01)/2.
        MIN_MAX conversion:   mean = (min+max)/2, std = (max-min)/2.
        """
        # Already has mean/std
        if "mean" in feat_stats and "std" in feat_stats:
            mean = feat_stats["mean"]
            std = feat_stats["std"]
            if isinstance(mean, list) and isinstance(std, list):
                return mean, std

        # QUANTILES: q01/q99 → mean/std
        if "q01" in feat_stats and "q99" in feat_stats:
            q01 = feat_stats["q01"]
            q99 = feat_stats["q99"]
            if isinstance(q01, list) and isinstance(q99, list):
                mean = [(a + b) / 2.0 for a, b in zip(q01, q99, strict=False)]
                std = [max((b - a) / 2.0, 1e-8) for a, b in zip(q01, q99, strict=False)]
                return mean, std

        # MIN_MAX: min/max → mean/std
        if "min" in feat_stats and "max" in feat_stats:
            mn = feat_stats["min"]
            mx = feat_stats["max"]
            if isinstance(mn, list) and isinstance(mx, list):
                mean = [(a + b) / 2.0 for a, b in zip(mn, mx, strict=False)]
                std = [max((b - a) / 2.0, 1e-8) for a, b in zip(mn, mx, strict=False)]
                return mean, std

        return None, None

    @staticmethod
    def _fix_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Fix state dict keys to match PI05Model architecture.

        Adapted from lerobot's ``PI05Policy._fix_pytorch_state_dict_keys``.
        """
        fixed: dict[str, torch.Tensor] = {}

        for key, value in state_dict.items():
            new_key = key

            # Strip "model." prefix — HF checkpoint wraps everything
            # under the policy's `self.model`, but we load into PI05Model directly.
            new_key = new_key.removeprefix("model.")

            # Skip adaRMS mismatch keys for expert
            if re.match(
                r"paligemma_with_expert\.gemma_expert\.model\.layers\.\d+\."
                r"(input_layernorm|post_attention_layernorm)\.weight$",
                new_key,
            ):
                # PI05 expert uses adaRMS — skip plain layernorm weights
                continue

            if re.match(r"paligemma_with_expert\.gemma_expert\.model\.norm\.weight$", new_key):
                continue

            # Rename action_time_mlp_* → time_mlp_*
            if new_key.startswith("action_time_mlp_in."):
                new_key = new_key.replace("action_time_mlp_in.", "time_mlp_in.")
            elif new_key.startswith("action_time_mlp_out."):
                new_key = new_key.replace("action_time_mlp_out.", "time_mlp_out.")

            # Skip state_proj (not used in pi05)
            if new_key.startswith("state_proj."):
                continue

            # Copy lm_head → embed_tokens (weight tying)
            if new_key == "paligemma_with_expert.paligemma.lm_head.weight":
                tied_key = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
                fixed[tied_key] = value.clone()

            fixed[new_key] = value

        return fixed

    def setup(self, stage: str) -> None:
        """Set up model from datamodule (lazy initialization path).

        Called by Lightning before fit/validate/test/predict.
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
        """
        if self.training:
            if self.model is None or self._preprocessor is None:
                msg = "Model is not initialized"
                raise ValueError(msg)

            processed_batch = self._preprocessor(batch.to_dict())
            images = processed_batch[IMAGES]
            img_masks = processed_batch["image_masks"]
            tokens = processed_batch["tokenized_prompt"]
            masks = processed_batch["tokenized_prompt_mask"]
            actions = processed_batch[ACTION]

            losses = self.model.forward(images, img_masks, tokens, masks, actions)

            loss = losses.mean()
            loss_dict = {"loss": loss.item()}
            return loss, loss_dict
        return self.predict_action_chunk(batch)

    @torch.no_grad()
    def predict_action_chunk(self, batch: Observation) -> torch.Tensor:
        """Predict a chunk of actions from observation.

        Args:
            batch: Input observation batch.

        Returns:
            Action chunk tensor after post-processing.
        """
        if self.model is None or self._preprocessor is None or self._postprocessor is None:
            msg = "Model is not initialized"
            raise ValueError(msg)

        processed_batch = self._preprocessor(batch.to(self.device).to_dict())
        images = processed_batch[IMAGES]
        img_masks = processed_batch["image_masks"]
        tokens = processed_batch["tokenized_prompt"]
        masks = processed_batch["tokenized_prompt_mask"]

        actions = self.model.sample_actions(images, img_masks, tokens, masks)

        # Unpad actions to actual action dimension
        if self._dataset_stats is not None:
            original_action_dim = int(self._dataset_stats[ACTION]["shape"][-1])
            actions = actions[:, :, :original_action_dim]

        # Clip to n_action_steps so the action queue receives the first N actions,
        # not the last N (deque maxlen silently discards earlier items on extend).
        actions = actions[:, : self._n_action_steps]

        return self._postprocessor({ACTION: actions})[ACTION]

    def training_step(self, batch: Observation, batch_idx: int) -> torch.Tensor:
        """Lightning training step."""
        del batch_idx
        loss, loss_dict = self(batch)
        self.log("train/loss", loss_dict["loss"], prog_bar=True)
        return loss

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and scheduler."""
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

        def lr_lambda(step: int) -> float:
            num_drops = step // drop_steps
            decay_factor = decay_value**num_drops
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            return decay_factor

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
