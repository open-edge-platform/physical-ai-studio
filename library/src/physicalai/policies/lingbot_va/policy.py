# Copyright 2024-2025 The Robbyant Team Authors.
# Copyright 2026 The HuggingFace Inc. team.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""LingBot-VA policy - Lightning wrapper for training and inference.

LingBot-VA predicts future video latents and robot actions in one autoregressive
sequence, so rollouts are stateful in a way most policies are not: every environment
step is observed (not just the ones that trigger a new chunk), and the observed
keyframes are fed back into the transformer's KV cache before the next chunk is
predicted. :meth:`LingBotVA.select_action` therefore overrides the base action-queue
behaviour, which only sees the batch when the queue runs dry.

Only the :attr:`~physicalai.export.ExportBackend.TORCH` backend is supported. The
autoregressive KV cache, the lazily loaded 20 GB frozen VAE/UMT5 stack and the two nested
denoising loops have no meaningful static graph, so the tracing backends (ONNX, OpenVINO,
ExecuTorch) are out of reach; :meth:`~physicalai.export.ExportablePolicyMixin.to_torch`
needs none of them, because it serializes the trainable transformer plus the hyperparameters
that rebuild the policy, and Runtime restores the live Python object from them.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

import torch
from physicalai.inference.data import InferenceFeature, InferenceFeatureDtype, InferenceFeatureType
from physicalai.inference.manifest import ComponentSpec
from torch.optim.lr_scheduler import LambdaLR

from physicalai.data.dataset import Dataset
from physicalai.data.observation import ACTION, IMAGES, TASK, FeatureType
from physicalai.export import ExportablePolicyMixin, ExportBackend
from physicalai.export.backends import ExportParameters, TorchExportParameters
from physicalai.policies.base import Policy
from physicalai.train.utils import reformat_dataset_to_match_policy

from .config import LingBotVAConfig
from .model import LingBotVAModel
from .preprocessor import camera_basename, make_lingbot_va_preprocessors
from .pretrained_utils import (
    detect_normalization_mode,
    extract_action_stats,
    load_config,
    load_transformer_weights,
    resolve_checkpoint,
)

if TYPE_CHECKING:
    from pathlib import Path

    from physicalai.data import Observation

    from .preprocessor import LingBotVAPostprocessor, LingBotVAPreprocessor

logger = logging.getLogger(__name__)


def warmup_constant_scheduler(optimizer: torch.optim.Optimizer, num_warmup_steps: int) -> LambdaLR:
    """Linear warmup followed by a constant learning rate (the upstream LingBot-VA schedule).

    Args:
        optimizer: Optimizer to wrap.
        num_warmup_steps: Number of linear warmup steps. ``0`` disables warmup.

    Returns:
        A ``LambdaLR`` implementing the schedule.
    """

    def lr_lambda(current_step: int) -> float:
        if num_warmup_steps <= 0:
            return 1.0
        return min(1.0, (current_step + 1) / num_warmup_steps)

    return LambdaLR(optimizer, lr_lambda)


class LingBotVA(ExportablePolicyMixin, Policy):
    """LingBot-VA - an autoregressive video-action world model on the Wan2.2 stack.

    Uses the same dual-path initialization as the other Studio policies:

    - **Lazy path**: ``LingBotVA()`` + ``trainer.fit()`` - the model is built in ``setup()``
      from the datamodule's statistics.
    - **Eager path**: ``LingBotVA(pretrained_name_or_path=...)`` or
      ``LingBotVA.load_from_checkpoint(...)`` - the model is built immediately.

    Only the ~5B transformer is trainable and checkpointed; the frozen VAE + UMT5 encoder
    (~20 GB) are pulled from ``wan_pretrained_path`` the first time the model runs.

    Args:
        pretrained_name_or_path: HuggingFace repo id or local directory of a published
            LingBot-VA checkpoint (for example ``"lerobot/lingbot_va_libero_long"``).
            Architecture and inference hyperparameters come from its ``config.json``;
            the arguments below override them.
        patch_size: Latent patch size ``(t, h, w)``.
        num_attention_heads: Number of attention heads.
        attention_head_dim: Per-head dimension.
        in_channels: Video-latent channels consumed by the patch embedder.
        out_channels: Video-latent channels produced by the latent head.
        action_dim: Width of the multi-embodiment action space.
        text_dim: Width of the UMT5 hidden states.
        freq_dim: Width of the sinusoidal timestep features.
        ffn_dim: Feed-forward inner dimension.
        num_layers: Number of transformer blocks.
        cross_attn_norm: Layer-norm before text cross-attention.
        eps: Layer-norm epsilon.
        rope_max_seq_len: Maximum rotary sequence length.
        attn_mode: Attention backend. ``"torch"`` is inference-only; use ``"flex"`` to train.
        wan_pretrained_path: Source of the frozen VAE / text encoder / tokenizer.
        dtype: Model precision.
        text_encoder_device: Device for the frozen UMT5-XXL encoder.
        obs_cam_keys: Camera keys, in the checkpoint's fixed order.
        image_hflip: Horizontally flip camera images before encoding.
        camera_layout: ``"width_concat"`` (LIBERO) or ``"robotwin_tshape"`` (RoboTwin).
        n_obs_steps: Number of observation steps per inference call.
        height: Camera height fed to the VAE.
        width: Camera width fed to the VAE.
        action_per_frame: Action sub-steps predicted per latent frame.
        frame_chunk_size: Latent frames predicted per autoregressive chunk.
        attn_window: Attention window, in chunks, of the streaming KV cache.
        num_inference_steps: Denoising steps for the video-latent stream.
        video_exec_step: Truncate the video denoising loop (``-1`` runs it fully).
        action_num_inference_steps: Denoising steps for the action stream.
        guidance_scale: Classifier-free guidance scale for the video stream.
        action_guidance_scale: Guidance scale for the action stream.
        snr_shift: Flow-matching SNR shift for the video stream.
        action_snr_shift: Flow-matching SNR shift for the action stream.
        max_sequence_length: Padded UMT5 prompt length.
        used_action_channel_ids: Action channels this checkpoint drives; also fixes the
            policy's output action dimension.
        save_predicted_video: Keep the predicted video latents for later VAE decoding.
        normalization_mode: Action (un)normalization method.
        optimizer_lr: Learning rate.
        optimizer_betas: Adam beta coefficients.
        optimizer_eps: Optimizer epsilon.
        optimizer_weight_decay: Weight decay coefficient.
        optimizer_grad_clip_norm: Maximum gradient norm.
        scheduler_warmup_steps: Linear warmup steps before the constant learning rate.
        dataset_stats: Action statistics for eager initialization (checkpoint restore).

    Example:
        Inference from a published checkpoint:

        >>> policy = LingBotVA(pretrained_name_or_path="lerobot/lingbot_va_libero_long")  # doctest: +SKIP
        >>> action = policy.select_action(observation)  # doctest: +SKIP

        Fine-tuning (needs the flex-attention backend):

        >>> policy = LingBotVA(  # doctest: +SKIP
        ...     pretrained_name_or_path="lerobot/lingbot_va_libero_long",
        ...     attn_mode="flex",
        ... )
    """

    def __init__(  # noqa: PLR0913
        self,
        pretrained_name_or_path: str | Path | None = None,
        *,
        # Transformer architecture
        patch_size: tuple[int, int, int] = (1, 2, 2),
        num_attention_heads: int = 24,
        attention_head_dim: int = 128,
        in_channels: int = 48,
        out_channels: int = 48,
        action_dim: int = 30,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 14336,
        num_layers: int = 30,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        rope_max_seq_len: int = 1024,
        attn_mode: Literal["torch", "flashattn", "flex"] = "torch",
        # Frozen sub-models
        wan_pretrained_path: str = "robbyant/lingbot-va-base",
        dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16",
        text_encoder_device: str = "cpu",
        # Cameras
        obs_cam_keys: tuple[str, ...] = ("observation.images.image", "observation.images.image2"),
        image_hflip: bool = False,
        camera_layout: Literal["width_concat", "robotwin_tshape"] = "width_concat",
        # Inference
        n_obs_steps: int = 1,
        height: int = 128,
        width: int = 128,
        action_per_frame: int = 4,
        frame_chunk_size: int = 4,
        attn_window: int = 30,
        num_inference_steps: int = 20,
        video_exec_step: int = -1,
        action_num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        action_guidance_scale: float = 1.0,
        snr_shift: float = 5.0,
        action_snr_shift: float = 0.05,
        max_sequence_length: int = 512,
        # Action space
        used_action_channel_ids: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6),
        save_predicted_video: bool = False,
        normalization_mode: Literal["MEAN_STD", "QUANTILES"] = "QUANTILES",
        # Optimizer / scheduler
        optimizer_lr: float = 1e-5,
        optimizer_betas: tuple[float, float] = (0.9, 0.95),
        optimizer_eps: float = 1e-8,
        optimizer_weight_decay: float = 1e-4,
        optimizer_grad_clip_norm: float = 1.0,
        scheduler_warmup_steps: int = 1_000,
        # Eager initialization
        dataset_stats: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Initialize the policy, resolving a pretrained checkpoint when one is given."""
        config_kwargs: dict[str, Any] = {
            "patch_size": patch_size,
            "num_attention_heads": num_attention_heads,
            "attention_head_dim": attention_head_dim,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "action_dim": action_dim,
            "text_dim": text_dim,
            "freq_dim": freq_dim,
            "ffn_dim": ffn_dim,
            "num_layers": num_layers,
            "cross_attn_norm": cross_attn_norm,
            "eps": eps,
            "rope_max_seq_len": rope_max_seq_len,
            "attn_mode": attn_mode,
            "wan_pretrained_path": wan_pretrained_path,
            "dtype": dtype,
            "text_encoder_device": text_encoder_device,
            "obs_cam_keys": obs_cam_keys,
            "image_hflip": image_hflip,
            "camera_layout": camera_layout,
            "n_obs_steps": n_obs_steps,
            "height": height,
            "width": width,
            "action_per_frame": action_per_frame,
            "frame_chunk_size": frame_chunk_size,
            "attn_window": attn_window,
            "num_inference_steps": num_inference_steps,
            "video_exec_step": video_exec_step,
            "action_num_inference_steps": action_num_inference_steps,
            "guidance_scale": guidance_scale,
            "action_guidance_scale": action_guidance_scale,
            "snr_shift": snr_shift,
            "action_snr_shift": action_snr_shift,
            "max_sequence_length": max_sequence_length,
            "used_action_channel_ids": used_action_channel_ids,
            "save_predicted_video": save_predicted_video,
            "normalization_mode": normalization_mode,
            "optimizer_lr": optimizer_lr,
            "optimizer_betas": optimizer_betas,
            "optimizer_eps": optimizer_eps,
            "optimizer_weight_decay": optimizer_weight_decay,
            "optimizer_grad_clip_norm": optimizer_grad_clip_norm,
            "scheduler_warmup_steps": scheduler_warmup_steps,
        }

        weights_file: Path | None = None
        if pretrained_name_or_path is not None:
            config, dataset_stats, weights_file = self._from_hf(
                pretrained_name_or_path,
                config_kwargs,
                dataset_stats,
            )
        else:
            config = LingBotVAConfig(**config_kwargs)

        # The chunk is executed in full before the next one is predicted.
        super().__init__(n_action_steps=config.n_action_steps)
        self.config = config

        self.save_hyperparameters(ignore=["config", "pretrained_name_or_path"])
        self._set_hparam_keys()

        self.model: LingBotVAModel | None = None  # pyrefly: ignore[bad-override-mutable-attribute]
        self._preprocessor: LingBotVAPreprocessor | None = None  # pyrefly: ignore[bad-override-mutable-attribute]
        self._postprocessor: LingBotVAPostprocessor | None = None
        self._dataset_stats = dataset_stats

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
    ) -> tuple[LingBotVAConfig, dict[str, dict[str, Any]] | None, Path]:
        """Resolve a published checkpoint into a config, statistics and a weights file.

        Only arguments the caller actually changed from the constructor defaults override
        the checkpoint's own values, so loading a checkpoint does not silently reshape its
        architecture.

        Args:
            pretrained_name_or_path: HuggingFace repo id or local directory.
            overrides: The constructor's configuration arguments.
            dataset_stats: Caller-supplied statistics, which win over the checkpoint's.

        Returns:
            Tuple of ``(config, dataset_stats, weights_file)``.
        """
        files = resolve_checkpoint(pretrained_name_or_path)

        defaults = LingBotVAConfig()
        explicit = {key: value for key, value in overrides.items() if getattr(defaults, key, object()) != value}
        detected = detect_normalization_mode(files.postprocessor_file)
        if detected is not None and "normalization_mode" not in explicit:
            explicit["normalization_mode"] = detected

        config = load_config(files.config_file, explicit)
        if dataset_stats is None:
            dataset_stats = extract_action_stats(files.postprocessor_file, files.postprocessor_dir) or None
        return config, dataset_stats, files.weights_file

    def _initialize_model(
        self,
        dataset_stats: dict[str, dict[str, Any]],
        weights_file: Path | None = None,
    ) -> None:
        """Build the model and the pre/post-processors.

        Args:
            dataset_stats: Action statistics used to build the (de)normalizers.
            weights_file: Optional ``model.safetensors`` to load into the transformer.
        """
        self.model = LingBotVAModel(self.config)
        if weights_file is not None:
            load_transformer_weights(self.model, weights_file)
            self.model.to(self.model.dtype)

        self._update_processor_stats(dataset_stats)

    def _update_processor_stats(self, dataset_stats: dict[str, dict[str, Any]]) -> None:
        """Rebuild the pre/post-processors from a new set of statistics.

        Args:
            dataset_stats: Action statistics from a checkpoint or a training dataset.
        """
        self._preprocessor, self._postprocessor = make_lingbot_va_preprocessors(
            dataset_stats,
            used_action_channel_ids=self.config.used_action_channel_ids,
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
            # Fine-tuning: keep the pretrained weights but normalize against the new data.
            logger.info("Updating LingBot-VA action statistics for the fine-tuning dataset")
            self._update_processor_stats(stats_dict)

        reformat_dataset_to_match_policy(self, datamodule)

    @property
    def inner_model(self) -> LingBotVAModel:
        """The unwrapped dual-stream world model.

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
            return self.inner_model(self._preprocess(batch))
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
        self.log("train/latent_loss", loss_dict["latent_loss"])
        self.log("train/action_loss", loss_dict["action_loss"])
        return loss

    def compute_val_loss(self, batch: Observation) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        """Compute the dual-stream flow-matching loss on a validation batch.

        Args:
            batch: Observation batch holding ground-truth actions.

        Returns:
            Tuple of ``(loss, loss_dict)``.
        """
        return self.inner_model.compute_val_loss(self._preprocess(batch))

    def configure_optimizers(self) -> dict[str, Any]:  # pyrefly: ignore[bad-override]
        """Configure AdamW with the upstream linear-warmup-then-constant schedule.

        Returns:
            Dict with the optimizer and its step-interval scheduler.
        """
        optimizer = torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.config.optimizer_lr,
            betas=self.config.optimizer_betas,
            eps=self.config.optimizer_eps,
            weight_decay=self.config.optimizer_weight_decay,
        )
        scheduler = warmup_constant_scheduler(optimizer, self.config.scheduler_warmup_steps)
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
        """Predict one autoregressive chunk of actions.

        This is the entry point the exported Torch artifact is driven through, and it runs
        the world model open-loop: only the episode's **first** call conditions on ``batch``.
        Later calls continue from the KV cache, because a single observation is not the
        keyframe clip a chunk boundary needs. Drive :meth:`select_action` once per
        environment step for the closed-loop behaviour.

        Args:
            batch: Input observation batch; its cameras condition the episode's first chunk.

        Returns:
            Actions of shape ``[B, T, output_action_dim]`` in physical units. ``T`` is
            ``chunk_size``, or ``chunk_size - action_per_frame`` for an episode's first
            chunk, whose first latent frame is the conditioning observation.
        """
        return self._predict_chunk(self._preprocess(batch))

    def _predict_chunk(self, processed: dict[str, Any] | None) -> torch.Tensor:
        """Run one chunk through the model and denormalize it.

        Args:
            processed: A preprocessed observation, or ``None`` to continue the stream from
                the KV cache and the buffered keyframes.

        Returns:
            Actions of shape ``[B, T, output_action_dim]`` in physical units.

        Raises:
            ValueError: If the postprocessor was not initialized.
        """
        if self._postprocessor is None:
            msg = "Model is not initialized"
            raise ValueError(msg)
        actions = self.inner_model.predict_action_chunk(processed)
        return self._postprocessor({ACTION: actions})[ACTION]

    @torch.no_grad()
    def select_action(self, batch: Observation) -> torch.Tensor:
        """Return one action, refilling the chunk and feeding observations back as needed.

        Unlike the base implementation, every call inspects the observation: the first one
        conditions the episode's first chunk, and each later one is a candidate keyframe
        that is replayed into the KV cache when the chunk is exhausted. That closed loop is
        what keeps the world model anchored to what actually happened.

        Args:
            batch: Input observation batch (single environment).

        Returns:
            A single action of shape ``[B, output_action_dim]`` in physical units.
        """
        self.eval()
        model = self.inner_model
        processed = self._preprocess(batch)
        model.ensure_frozen_modules()
        model.maybe_init_prompt(processed)

        if not model.streaming_started:
            # First call: this observation conditions the first chunk; it is not a keyframe.
            model.streaming_started = True
            self._action_queue.clear()
            self._action_queue.extend(self._predict_chunk(processed).transpose(0, 1))
            model.begin_chunk()
        else:
            # This observation is the result of the action just executed.
            model.observe_keyframe(processed)
            if len(self._action_queue) == 0:
                self._action_queue.extend(self._predict_chunk(None).transpose(0, 1))
                model.begin_chunk()

        model.advance_step()
        return self._action_queue.popleft()

    def reset(self) -> None:
        """Clear the action queue and every per-episode streaming state."""
        super().reset()
        if self.model is not None:
            self.model.reset()

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
        """Get a list of export backends supported by policy.

        Returns:
            list[str | ExportBackend]: Only ``torch``. The tracing backends would have to
            capture the streaming KV cache and the two nested denoising loops in a static
            graph, which this architecture does not have.
        """
        return [ExportBackend.TORCH]

    @property
    def inputs_schema(self) -> list[InferenceFeature] | None:
        """Describe the policy's expected inputs for export.

        Returns:
            One visual feature per configured camera, in ``obs_cam_keys`` order, plus the
            ``task`` prompt the world model is conditioned on. LingBot-VA consumes no robot
            state. Returns ``None`` if the model has not been initialized yet.
        """
        if self.model is None:
            return None

        schema: list[InferenceFeature] = [
            InferenceFeature(
                ftype=InferenceFeatureType.VISUAL,
                shape=self._camera_shape(key),
                # Always fully qualified, even for a single camera: ``resolve_camera_keys``
                # matches on the camera's basename, so a bare ``images`` would not resolve.
                name=f"{IMAGES}.{camera_basename(key)}",
                dtype=InferenceFeatureDtype.FLOAT32,
            )
            for key in self.config.obs_cam_keys
        ]
        schema.append(
            InferenceFeature(
                ftype=InferenceFeatureType.LANGUAGE,
                shape=(),
                name=TASK,
                dtype=InferenceFeatureDtype.STRING,
            ),
        )
        return schema

    def _camera_shape(self, camera_key: str) -> tuple[int, ...]:
        """Return the raw ``(C, H, W)`` shape of one configured camera.

        The dataset's own resolution is reported when the statistics carry it; otherwise the
        VAE input resolution is used. Either is accurate enough for the manifest, because the
        model resizes every camera to ``(config.height, config.width)`` itself.

        Args:
            camera_key: A key from ``config.obs_cam_keys``.

        Returns:
            The camera's shape, excluding the batch dimension.
        """
        base = camera_basename(camera_key)
        for name, stat in (self._dataset_stats or {}).items():
            if str(FeatureType.VISUAL) not in str(stat.get("type", "")):
                continue
            if camera_basename(name) == base and stat.get("shape"):
                return tuple(stat["shape"])
        return (3, self.config.height, self.config.width)

    @property
    def outputs_schema(self) -> list[InferenceFeature] | None:
        """Describe the policy's output for export.

        Returns:
            A single ``action`` feature of shape ``(chunk_size, output_action_dim)``. An
            episode's *first* chunk is ``action_per_frame`` steps shorter, because its first
            latent frame is the conditioning observation. Returns ``None`` if the model has
            not been initialized yet.
        """
        if self.model is None:
            return None

        return [
            InferenceFeature(
                ftype=InferenceFeatureType.ACTION,
                shape=(self.config.chunk_size, self.config.output_action_dim),
                name=ACTION,
                dtype=InferenceFeatureDtype.FLOAT32,
            ),
        ]

    @property
    def extra_export_args(self) -> dict[str, ExportParameters]:
        """Additional export arguments for model conversion.

        The exported checkpoint restores the policy itself, so action normalization stays
        inside it: the runtime only has to hand over ``float32`` images in ``[0, 1]``. No
        chunk trimmer is recorded either, since ``n_action_steps == chunk_size``.

        Returns:
            dict[str, ExportParameters]: Export parameters keyed by backend name.
        """
        return {
            "torch": TorchExportParameters(
                preprocessors_specs=[ComponentSpec(type="to_float_tensor")],
                postprocessors_specs=[],
                output_names=[ACTION],
            ),
        }


__all__ = ["LingBotVA", "warmup_constant_scheduler"]
