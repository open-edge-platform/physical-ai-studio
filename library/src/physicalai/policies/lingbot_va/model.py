# Copyright 2024-2025 The Robbyant Team Authors.
# Copyright 2026 The HuggingFace Inc. team.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""LingBot-VA model: an autoregressive video-action world model on the Wan2.2 stack.

The model owns three things:

- the trainable dual-stream transformer (the only sub-module that is checkpointed);
- the frozen Wan VAE + UMT5 text encoder + tokenizer, held *outside* the ``nn.Module``
  registry so they are neither saved nor moved by ``.to()``, and lazily pulled from
  ``config.wan_pretrained_path`` on first use;
- the per-episode streaming state (KV cache, observed-keyframe buffer, flow-matching
  schedulers) that drives autoregressive rollouts.

Inference runs one autoregressive chunk at a time: the video-latent stream is denoised
first (with classifier-free guidance), then the action stream, each with its own
flow-matching scheduler. Real observed keyframes are fed back into the KV cache as the
chunk's actions are executed, which is what makes the rollout closed-loop.

The streaming path is written for single-environment rollouts (batch size 1).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange

from physicalai.data.observation import ACTION, TASK
from physicalai.policies.base import Model

from .components import (
    FlowMatchScheduler,
    WanTransformer3DModel,
    WanVAEStreamingWrapper,
    data_seq_to_patch,
    denormalize_latents,
    encode_prompt,
    get_mesh_id,
    load_text_encoder,
    load_tokenizer,
    load_vae,
    normalize_latents,
    sample_timestep_id,
)
from .preprocessor import resolve_camera_keys

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .config import LingBotVAConfig

logger = logging.getLogger(__name__)

_TORCH_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
_CACHE_NAME = "pos"
_NUM_TRAIN_TIMESTEPS = 1000


class LingBotVAModel(Model):
    """Dual-stream video-action world model.

    Args:
        config: The resolved :class:`~.config.LingBotVAConfig`.

    Example:
        >>> from physicalai.policies.lingbot_va import LingBotVAConfig, LingBotVAModel
        >>> model = LingBotVAModel(LingBotVAConfig(num_layers=1, ffn_dim=64))  # doctest: +SKIP
    """

    def __init__(self, config: LingBotVAConfig) -> None:
        """Build the transformer and initialize the (empty) streaming state."""
        super().__init__()
        self.config = config
        self.dtype = _TORCH_DTYPES[config.dtype]

        self.transformer = WanTransformer3DModel(
            patch_size=tuple(config.patch_size),  # type: ignore[arg-type]
            num_attention_heads=config.num_attention_heads,
            attention_head_dim=config.attention_head_dim,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            action_dim=config.action_dim,
            text_dim=config.text_dim,
            freq_dim=config.freq_dim,
            ffn_dim=config.ffn_dim,
            num_layers=config.num_layers,
            cross_attn_norm=config.cross_attn_norm,
            eps=config.eps,
            rope_max_seq_len=config.rope_max_seq_len,
            attn_mode=config.attn_mode,
        ).to(self.dtype)

        # Frozen modules live outside the nn.Module registry: they are ~20 GB, are not part
        # of the checkpoint, and must not be moved by `.to()`.
        self._frozen: dict[str, Any] = {}
        self._train_schedulers: tuple[FlowMatchScheduler, FlowMatchScheduler] | None = None

        self.last_predicted_latents: torch.Tensor | None = None
        self.reset()

    # ------------------------------------------------------------------ #
    # Delta indices (consumed by the datamodule to build training clips)  #
    # ------------------------------------------------------------------ #
    @property
    def reward_delta_indices(self) -> None:
        """LingBot-VA does not consume rewards."""
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        """Action offsets each training sample must provide."""
        return self.config.action_delta_indices

    @property
    def observation_delta_indices(self) -> list[int]:
        """Frame offsets of the observation clip each training sample must provide."""
        return self.config.observation_delta_indices

    @property
    def device(self) -> torch.device:
        """Device the trainable transformer currently lives on."""
        return next(self.transformer.parameters()).device

    # ------------------------------------------------------------------ #
    # Frozen sub-models                                                   #
    # ------------------------------------------------------------------ #
    def ensure_frozen_modules(self) -> None:
        """Lazily pull the frozen VAE, text encoder and tokenizer.

        The three sub-folders (``vae/``, ``text_encoder/``, ``tokenizer/``) are resolved
        from ``config.wan_pretrained_path``, which may be a HuggingFace repo id or a local
        directory.
        """
        if self._frozen:
            return

        path = self.config.wan_pretrained_path
        device = self.device
        logger.info("Loading frozen LingBot-VA components from %s", path)

        vae = load_vae(path, torch_dtype=self.dtype, torch_device=device, subfolder="vae")
        text_encoder = load_text_encoder(
            path,
            torch_dtype=self.dtype,
            torch_device=self.config.text_encoder_device,
            subfolder="text_encoder",
        )
        self._frozen = {
            "vae": vae.eval(),
            "streaming_vae": WanVAEStreamingWrapper(vae),
            "text_encoder": text_encoder.eval(),
            "tokenizer": load_tokenizer(path, subfolder="tokenizer"),
        }
        if self.config.camera_layout == "robotwin_tshape":
            # The half-resolution wrist cameras need their own causal VAE cache.
            vae_half = load_vae(path, torch_dtype=self.dtype, torch_device=device, subfolder="vae")
            self._frozen["streaming_vae_half"] = WanVAEStreamingWrapper(vae_half.eval())

    @property
    def vae(self) -> Any:  # noqa: ANN401
        """The frozen Wan2.2 VAE (loads it if needed)."""
        self.ensure_frozen_modules()
        return self._frozen["vae"]

    @property
    def streaming_vae(self) -> WanVAEStreamingWrapper:
        """The causal streaming wrapper around the VAE encoder."""
        self.ensure_frozen_modules()
        return self._frozen["streaming_vae"]

    def get_optim_params(self) -> list[torch.nn.Parameter]:
        """Return the trainable parameters (the transformer, or its adapters).

        Returns:
            The transformer parameters that require gradients. The frozen VAE and text
            encoder are never included because they live outside the module registry.
        """
        return [p for p in self.transformer.parameters() if p.requires_grad]

    # ------------------------------------------------------------------ #
    # Streaming state                                                     #
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Reset every per-episode streaming state (KV cache, buffers, schedulers)."""
        config = self.config

        self.streaming_started = False
        self._obs_buffer: list[dict[str, torch.Tensor]] = []
        self._executed_actions: torch.Tensor | None = None
        self._init_latent: torch.Tensor | None = None
        self._first_chunk = True
        self._frame_st_id = 0
        self._exec_step = 0
        self._prev_substep = 0
        self._prompt: str | None = None
        self._prompt_embeds: torch.Tensor | None = None
        self._negative_prompt_embeds: torch.Tensor | None = None
        self.last_predicted_latents = None

        self._use_cfg = (config.guidance_scale > 1) or (config.action_guidance_scale > 1)
        self._scheduler = FlowMatchScheduler(shift=config.snr_shift, sigma_min=0.0, extra_one_step=True)
        self._action_scheduler = FlowMatchScheduler(
            shift=config.action_snr_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )
        self._scheduler.set_timesteps(_NUM_TRAIN_TIMESTEPS, training=True)
        self._action_scheduler.set_timesteps(_NUM_TRAIN_TIMESTEPS, training=True)

        if hasattr(self, "transformer"):
            self.transformer.clear_cache(_CACHE_NAME)
        # Without this the encoder carries the previous episode's temporal state over and
        # corrupts the latent frame counts on the next episode's first encode.
        if self._frozen:
            self._frozen["streaming_vae"].clear_cache()
            if "streaming_vae_half" in self._frozen:
                self._frozen["streaming_vae_half"].clear_cache()

    def begin_chunk(self) -> None:
        """Mark the start of a freshly predicted chunk (resets the sub-step counters)."""
        self._exec_step = 0

    def advance_step(self) -> None:
        """Record that one action of the current chunk has been handed to the env."""
        self._prev_substep = self._exec_step % self.config.action_per_frame
        self._exec_step += 1

    def observe_keyframe(self, batch: dict[str, Any]) -> None:
        """Buffer an observation as a VAE keyframe if it lands on a stride boundary.

        Exactly ``frame_chunk_size * 4`` frames are buffered per chunk, which the VAE's
        temporal downsample collapses into ``frame_chunk_size`` latent frames.

        Args:
            batch: A preprocessed observation dict for the current env step.
        """
        if (self._prev_substep + 1) % self.config.keyframe_stride == 0:
            self._obs_buffer.append(self._extract_raw_obs(batch))

    # ------------------------------------------------------------------ #
    # Prompt handling                                                     #
    # ------------------------------------------------------------------ #
    def maybe_init_prompt(self, batch: dict[str, Any] | None) -> None:
        """Encode the episode's task string once, with its CFG negative counterpart.

        Args:
            batch: A preprocessed batch holding the ``task`` entry, or ``None`` to skip.
        """
        if self._prompt_embeds is not None or batch is None:
            return
        task = batch.get(TASK)
        prompt = task[0] if isinstance(task, list | tuple) else task
        self._prompt = prompt or ""
        self._prompt_embeds = self.encode_task([self._prompt])
        if self._use_cfg:
            self._negative_prompt_embeds = self.encode_task([""])

    @property
    def _prompt_conditioning(self) -> torch.Tensor:
        """The episode's encoded task prompt.

        Returns:
            Prompt embeddings of shape ``[1, max_sequence_length, text_dim]``.

        Raises:
            RuntimeError: If no task has been encoded yet.
        """
        if self._prompt_embeds is None:
            msg = "No task prompt encoded yet; call maybe_init_prompt() with an observation first."
            raise RuntimeError(msg)
        return self._prompt_embeds

    @property
    def _negative_prompt_conditioning(self) -> torch.Tensor:
        """The empty-prompt embeddings used for classifier-free guidance.

        Returns:
            Prompt embeddings of shape ``[1, max_sequence_length, text_dim]``.

        Raises:
            RuntimeError: If classifier-free guidance is active but no prompt was encoded.
        """
        if self._negative_prompt_embeds is None:
            msg = "No negative prompt encoded yet; call maybe_init_prompt() with an observation first."
            raise RuntimeError(msg)
        return self._negative_prompt_embeds

    def encode_task(self, prompts: Sequence[str]) -> torch.Tensor:
        """UMT5-encode task strings into padded prompt embeddings.

        Args:
            prompts: One task string per batch element.

        Returns:
            Prompt embeddings of shape ``[B, max_sequence_length, text_dim]``.
        """
        self.ensure_frozen_modules()
        return encode_prompt(
            list(prompts),
            tokenizer=self._frozen["tokenizer"],
            text_encoder=self._frozen["text_encoder"],
            max_sequence_length=self.config.max_sequence_length,
            dtype=self.dtype,
            device=self.device,
        )

    # ------------------------------------------------------------------ #
    # Training: dual-stream flow-matching loss                            #
    # ------------------------------------------------------------------ #
    def forward(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]] | torch.Tensor:
        """Compute the training loss, or predict an action chunk in eval mode.

        Args:
            batch: Preprocessed batch dict.

        Returns:
            ``(loss, loss_dict)`` in training mode, else the predicted action chunk.
        """
        if self.training:
            return self.compute_loss(batch)
        return self.predict_action_chunk(batch)

    def compute_loss(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        """Dual-stream flow-matching loss (``latent_loss + action_loss``).

        The camera clips are VAE-encoded into video latents and the task is UMT5-encoded;
        both streams are then noised independently and regressed with a timestep-weighted,
        action-masked MSE against the flow-matching velocity target.

        Args:
            batch: Preprocessed batch with camera clips, ``task`` and ``action``.

        Returns:
            Tuple of ``(loss, {"loss", "latent_loss", "action_loss"})``.
        """
        self._require_flex_attention()
        latents, actions, actions_mask, text_emb = self._build_training_streams(batch)
        return self.training_loss_from_streams(latents, actions, actions_mask, text_emb)

    def _require_flex_attention(self) -> None:
        """Fail fast when the model was not built for training.

        Raises:
            ValueError: If ``attn_mode`` is not ``"flex"``.
        """
        if self.config.attn_mode != "flex":
            msg = (
                "LingBot-VA training requires attn_mode='flex' (block-causal flow-matching masks). "
                "Build the policy with attn_mode='flex' for training/fine-tuning."
            )
            raise ValueError(msg)

    def training_loss_from_streams(
        self,
        latents: torch.Tensor,
        actions: torch.Tensor,
        actions_mask: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        """Run the dual-stream loss on already-prepared streams.

        Args:
            latents: Normalized video latents ``[B, in_channels, F, h, w]``.
            actions: Actions in model space ``[B, action_dim, F, action_per_frame, 1]``.
            actions_mask: Mask with the same shape as ``actions``, selecting used channels.
            text_emb: Prompt embeddings ``[B, seq_len, text_dim]``.

        Requires ``attn_mode='flex'``.

        Returns:
            Tuple of ``(loss, {"loss", "latent_loss", "action_loss"})``.
        """
        self._require_flex_attention()

        latent_scheduler, action_scheduler = self._ensure_train_schedulers()
        latent_dict = self._add_noise_stream(
            latents,
            latent_scheduler,
            action_mask=None,
            action_mode=False,
            noisy_cond_prob=0.5,
        )
        action_dict = self._add_noise_stream(
            actions,
            action_scheduler,
            action_mask=actions_mask,
            action_mode=True,
            noisy_cond_prob=0.0,
        )
        latent_dict["text_emb"] = text_emb
        action_dict["text_emb"] = text_emb
        action_dict["actions_mask"] = actions_mask

        input_dict = {
            "latent_dict": latent_dict,
            "action_dict": action_dict,
            # Upstream randomizes the block-causal chunk and attention window per step.
            "chunk_size": int(torch.randint(1, 5, (1,)).item()),
            "window_size": int(torch.randint(4, 65, (1,)).item()),
        }
        prediction = self.transformer(input_dict, train_mode=True)
        latent_loss, action_loss = self._flow_matching_loss(input_dict, prediction)
        loss = latent_loss + action_loss
        return loss, {
            "loss": loss.detach(),
            "latent_loss": latent_loss.detach(),
            "action_loss": action_loss.detach(),
        }

    def _ensure_train_schedulers(self) -> tuple[FlowMatchScheduler, FlowMatchScheduler]:
        """Build (once) the two training schedulers.

        Returns:
            Tuple of ``(latent_scheduler, action_scheduler)``.
        """
        if self._train_schedulers is None:
            latent_scheduler = FlowMatchScheduler(shift=self.config.snr_shift, sigma_min=0.0, extra_one_step=True)
            latent_scheduler.set_timesteps(_NUM_TRAIN_TIMESTEPS, training=True)
            action_scheduler = FlowMatchScheduler(
                shift=self.config.action_snr_shift,
                sigma_min=0.0,
                extra_one_step=True,
            )
            action_scheduler.set_timesteps(_NUM_TRAIN_TIMESTEPS, training=True)
            self._train_schedulers = (latent_scheduler, action_scheduler)
        return self._train_schedulers

    @torch.no_grad()
    def _add_noise_stream(
        self,
        latent: torch.Tensor,
        scheduler: FlowMatchScheduler,
        action_mask: torch.Tensor | None,
        *,
        action_mode: bool,
        noisy_cond_prob: float,
    ) -> dict[str, torch.Tensor]:
        """Flow-matching noising of one stream, plus its conditioning copy and grid ids.

        Args:
            latent: The clean stream ``[B, C, F, H, W]``.
            scheduler: The stream's flow-matching scheduler.
            action_mask: Channel mask for the action stream, or ``None``.
            action_mode: Whether this is the action stream (no spatial patching).
            noisy_cond_prob: Probability of also noising the clean conditioning copy,
                which teaches the model to tolerate imperfect history.

        Returns:
            Dict with ``timesteps``, ``noisy_latents``, ``targets``, ``latent``,
            ``cond_timesteps`` and ``grid_id``.
        """
        device = latent.device
        batch_size, _, num_frames, _, _ = latent.shape
        patch = (1, 1, 1) if action_mode else tuple(self.config.patch_size)

        ts_ids = sample_timestep_id(num_frames, num_train_timesteps=scheduler.num_train_timesteps)
        noise = torch.zeros_like(latent).normal_()
        timesteps = scheduler.timesteps[ts_ids].to(device)
        noisy_latents = scheduler.add_noise(latent, noise, timesteps, t_dim=2)
        targets = scheduler.training_target(latent, noise, timesteps)

        grid_id = (
            get_mesh_id(
                latent.shape[-3] // patch[0],
                latent.shape[-2] // patch[1],
                latent.shape[-1] // patch[2],
                t=1 if action_mode else 0,
                f_w=1,
                f_shift=0,
                action=action_mode,
            )
            .to(device)[None]
            .repeat(batch_size, 1, 1)
        )

        if torch.rand(1).item() < noisy_cond_prob:
            cond_ids = sample_timestep_id(
                num_frames,
                min_timestep_bd=0.5,
                max_timestep_bd=1.0,
                num_train_timesteps=scheduler.num_train_timesteps,
            )
            cond_noise = torch.zeros_like(latent).normal_()
            cond_timesteps = scheduler.timesteps[cond_ids].to(device)
            latent = scheduler.add_noise(latent, cond_noise, cond_timesteps, t_dim=2)
        else:
            cond_timesteps = torch.zeros_like(timesteps)

        if action_mask is not None:
            # Out-of-place: `latent` is the caller's tensor and must not be mutated.
            noisy_latents = noisy_latents * action_mask.float()  # noqa: PLR6104
            targets = targets * action_mask.float()  # noqa: PLR6104
            latent = latent * action_mask.float()  # noqa: PLR6104

        return {
            "timesteps": timesteps[None].repeat(batch_size, 1),
            "noisy_latents": noisy_latents,
            "targets": targets,
            "latent": latent,
            "cond_timesteps": cond_timesteps[None].repeat(batch_size, 1),
            "grid_id": grid_id,
        }

    def _flow_matching_loss(
        self,
        input_dict: dict[str, Any],
        prediction: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Timestep-weighted MSE for both streams.

        Args:
            input_dict: The dict handed to the transformer's training forward.
            prediction: ``(latent_prediction, action_prediction)``.

        Returns:
            Tuple of ``(latent_loss, action_loss)``.
        """
        latent_pred, action_pred = prediction
        latent_stream, action_stream = input_dict["latent_dict"], input_dict["action_dict"]

        action_pred = rearrange(action_pred, "b (f n) c -> b c f n 1", f=action_stream["targets"].shape[-3])
        latent_pred = data_seq_to_patch(
            tuple(self.config.patch_size),  # type: ignore[arg-type]
            latent_pred,
            latent_stream["targets"].shape[-3],
            latent_stream["targets"].shape[-2],
            latent_stream["targets"].shape[-1],
            batch_size=latent_pred.shape[0],
        )

        latent_scheduler, action_scheduler = self._ensure_train_schedulers()
        num_batches, num_frames = latent_stream["timesteps"].shape
        latent_weight = latent_scheduler.training_weight(latent_stream["timesteps"].flatten()).reshape(
            num_batches,
            num_frames,
        )
        action_weight = action_scheduler.training_weight(action_stream["timesteps"].flatten()).reshape(
            num_batches,
            num_frames,
        )

        latent_loss = F.mse_loss(latent_pred.float(), latent_stream["targets"].float().detach(), reduction="none")
        latent_loss = (
            (latent_loss * latent_weight[:, None, :, None, None]).permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1)
        )
        latent_loss = (latent_loss.sum(dim=1) / (torch.ones_like(latent_loss).sum(dim=1) + 1e-6)).mean()

        mask = action_stream["actions_mask"].float()
        action_loss = F.mse_loss(action_pred.float(), action_stream["targets"].float().detach(), reduction="none")
        action_loss = (
            (action_loss * action_weight[:, None, :, None, None] * mask).permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1)
        )
        flat_mask = mask.permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1)
        action_loss = (action_loss.sum(dim=1) / (flat_mask.sum(dim=1) + 1e-6)).mean()
        return latent_loss, action_loss

    @torch.no_grad()
    def _build_training_streams(  # noqa: PLR0914
        self,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build ``(latents, actions, actions_mask, text_emb)`` from a training batch.

        Camera frames are expected as a temporal clip (``[B, C, T, H, W]`` or
        ``[B, T, C, H, W]``) and are VAE-encoded into ``T / 4`` latent frames. Actions
        ``[B, F * action_per_frame, n_used]`` are scattered into the model's ``action_dim``
        space and masked to the used channels.

        Args:
            batch: Preprocessed batch dict.

        Returns:
            Tuple of ``(latents, actions, actions_mask, text_emb)``.

        Raises:
            KeyError: If the batch carries no ground-truth actions.
        """
        config = self.config
        device = self.device

        task = batch.get(TASK)
        if isinstance(task, str):
            task = [task]
        elif task is None:
            task = [""] * self._batch_size(batch)
        text_emb = self.encode_task(list(task))

        latents = self._encode_training_latents(batch)

        if batch.get(ACTION) is None:
            msg = "LingBot-VA training requires ground-truth actions in the batch."
            raise KeyError(msg)

        action = batch[ACTION].to(device)  # [B, F * action_per_frame, n_used]
        batch_size = action.shape[0]
        used = list(config.used_action_channel_ids)
        per_frame, num_frames = config.action_per_frame, config.frame_chunk_size
        action = action[:, : num_frames * per_frame].reshape(batch_size, num_frames, per_frame, len(used))
        action = action.permute(0, 3, 1, 2)  # [B, n_used, F, action_per_frame]

        full = action.new_zeros(batch_size, config.action_dim, num_frames, per_frame)
        index = torch.as_tensor(used, device=device)
        full[:, index] = action
        actions = full.unsqueeze(-1).to(self.dtype)

        mask = torch.zeros(config.action_dim, device=device, dtype=self.dtype)
        mask[index] = 1.0
        actions_mask = mask.view(1, -1, 1, 1, 1).expand_as(actions)
        return latents, actions, actions_mask, text_emb

    @staticmethod
    def _batch_size(batch: dict[str, Any]) -> int:
        """Infer the batch size from the first tensor in the batch.

        Returns:
            The leading dimension of the first tensor found, or 1.
        """
        for value in batch.values():
            if isinstance(value, torch.Tensor) and value.dim() > 0:
                return int(value.shape[0])
        return 1

    @torch.no_grad()
    def _encode_training_latents(self, batch: dict[str, Any]) -> torch.Tensor:
        """VAE-encode the per-camera training clips into normalized video latents.

        Args:
            batch: Preprocessed batch dict holding the camera clips.

        Returns:
            Normalized latents of shape ``[B, z_dim, F, h, w]``.
        """
        vae = self.vae
        vae_device = next(vae.parameters()).device
        keys = resolve_camera_keys(batch, self.config.obs_cam_keys)

        def clip(key: str) -> torch.Tensor:
            x = batch[key].to(vae_device)
            channel_options = (1, 3)
            if x.dim() == 4:  # noqa: PLR2004
                x = x.unsqueeze(2)  # [B, C, H, W] -> single-frame clip
            elif x.shape[1] not in channel_options and x.shape[2] in channel_options:
                x = x.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W] -> [B, C, T, H, W]
            return x.contiguous()

        def encode(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
            batch_size, channels, frames = x.shape[:3]
            x = F.interpolate(x.flatten(0, 1).float(), size=size, mode="bilinear", align_corners=False)
            x = (x.view(batch_size, channels, frames, *size) * 2.0 - 1.0).to(self.dtype)
            mu = vae.encode(x).latent_dist.mode()
            return normalize_latents(mu, vae.config.latents_mean, vae.config.latents_std)

        height, width = self.config.height, self.config.width
        if self.config.camera_layout == "robotwin_tshape":
            head = encode(clip(keys[0]), (height, width))
            left = encode(clip(keys[1]), (height // 2, width // 2))
            right = encode(clip(keys[2]), (height // 2, width // 2))
            return torch.cat([torch.cat([left, right], dim=-1), head], dim=-2).to(self.device)
        per_camera = [encode(clip(key), (height, width)) for key in keys]
        return torch.cat(per_camera, dim=-1).to(self.device)

    # ------------------------------------------------------------------ #
    # Inference: one autoregressive chunk                                 #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Any] | None) -> torch.Tensor:
        """Run one autoregressive chunk.

        Args:
            batch: A preprocessed observation dict for the first chunk of an episode, or
                ``None`` to continue from the KV cache and the buffered keyframes.

        Returns:
            Model-normalized actions of shape ``[B, n_steps, output_action_dim]``, where
            ``n_steps`` is ``chunk_size`` (or ``chunk_size - action_per_frame`` for the
            first chunk, whose first frame is the conditioning observation).

        Raises:
            ValueError: If the episode's first chunk is requested without an observation.
        """
        self.ensure_frozen_modules()
        self.maybe_init_prompt(batch)

        is_first = self._first_chunk
        if is_first:
            if batch is None:
                msg = "The first predicted chunk needs an observation batch to condition on."
                raise ValueError(msg)
            init_latent = self._encode_frames([self._extract_raw_obs(batch)])
            self._init_latent = init_latent
            self._init_streaming_cache()
            # Frame 0 conditions this chunk, so it is not replayed as a keyframe.
            self._obs_buffer = []
            actions, latents = self._infer(init_latent, frame_st_id=0)
            self._first_chunk = False
        else:
            self._commit_kv_cache(self._obs_buffer, self._executed_actions)
            self._obs_buffer = []
            actions, latents = self._infer(None, frame_st_id=self._frame_st_id)

        # Keep the full model-space chunk: it is fed back into the KV cache next time.
        self._executed_actions = actions

        if self.config.save_predicted_video:
            self.last_predicted_latents = latents.detach().to("cpu")

        used = list(self.config.used_action_channel_ids)
        chunk = actions[:, used]  # [B, n_used, F, action_per_frame, 1]
        if is_first:
            chunk = chunk[:, :, 1:]  # drop the conditioning frame's actions
        chunk = chunk.squeeze(-1).flatten(2)  # [B, n_used, n_steps]
        return chunk.transpose(1, 2).contiguous().to(torch.float32)

    def _extract_raw_obs(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Snapshot the configured camera images, kept raw for later VAE encoding.

        Returns:
            Dict mapping the resolved camera keys to detached image tensors.
        """
        resolved = resolve_camera_keys(batch, self.config.obs_cam_keys)
        pairs = zip(self.config.obs_cam_keys, resolved, strict=True)
        return {config_key: batch[batch_key].detach() for config_key, batch_key in pairs}

    def _camera_frame(
        self,
        raw_obs: dict[str, torch.Tensor],
        key: str,
        size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Return one camera frame as ``[1, C, 1, H, W]``, resized and scaled to ``[-1, 1]``.

        Args:
            raw_obs: A snapshot produced by :meth:`_extract_raw_obs`.
            key: Which configured camera to read.
            size: Target ``(height, width)``; defaults to the configured resolution.

        Returns:
            The prepared single-frame clip.
        """
        img = raw_obs[key]
        if img.dim() == 3:  # noqa: PLR2004
            img = img.unsqueeze(0)
        img = img.to(self.device, torch.float32)
        if self.config.image_hflip:
            img = torch.flip(img, dims=[-1])
        if size is None:
            size = (self.config.height, self.config.width)
        img = F.interpolate(img, size=size, mode="bilinear", align_corners=False)
        img = img * 2.0 - 1.0
        return img.unsqueeze(2).to(self.dtype)

    def _normalize_encoder_output(self, enc_out: torch.Tensor) -> torch.Tensor:
        """Take the mean of a VAE encoder output and channel-normalize it.

        Returns:
            Normalized latents.
        """
        mu, _logvar = torch.chunk(enc_out, 2, dim=1)
        return normalize_latents(mu, self.vae.config.latents_mean, self.vae.config.latents_std)

    @torch.no_grad()
    def _encode_frames(self, raw_frames: list[dict[str, torch.Tensor]]) -> torch.Tensor:
        """VAE-encode a clip of observed frames and assemble the per-camera latents.

        The frames of every configured camera are stacked along the temporal axis and
        encoded in a single streaming call, so the VAE's 4x temporal downsample collapses
        ``F`` observed frames into ``F / 4`` latent frames while the causal ``feat_cache``
        carries over from the previous chunk.

        Args:
            raw_frames: One snapshot per observed env sub-step.

        Returns:
            Assembled latents of shape ``[1, z_dim, F/4, h, w]``.
        """
        vae_device = next(self.vae.parameters()).device
        if self.config.camera_layout == "robotwin_tshape":
            return self._encode_frames_tshape(raw_frames, vae_device)

        per_camera = [
            torch.cat([self._camera_frame(frame, key) for frame in raw_frames], dim=2)
            for key in self.config.obs_cam_keys
        ]
        videos = torch.cat(per_camera, dim=0)  # [num_cam, C, F, H, W]
        enc_out = self.streaming_vae.encode_chunk(videos.to(vae_device).to(self.dtype))
        mu_norm = self._normalize_encoder_output(enc_out)
        # Per-camera latents are concatenated along width, in obs_cam_keys order.
        return torch.cat(mu_norm.split(1, dim=0), dim=-1).to(self.device)

    @torch.no_grad()
    def _encode_frames_tshape(
        self,
        raw_frames: list[dict[str, torch.Tensor]],
        vae_device: torch.device,
    ) -> torch.Tensor:
        """RoboTwin "T" layout: full-resolution head below two half-resolution wrists.

        Args:
            raw_frames: One snapshot per observed env sub-step.
            vae_device: Device the VAE lives on.

        Returns:
            Assembled latents of shape ``[1, z_dim, F/4, h, w]``.
        """
        config = self.config
        height, width = config.height, config.width
        head_key, left_key, right_key = config.obs_cam_keys[0], config.obs_cam_keys[1], config.obs_cam_keys[2]

        head = torch.cat([self._camera_frame(f, head_key, size=(height, width)) for f in raw_frames], dim=2)
        left = torch.cat(
            [self._camera_frame(f, left_key, size=(height // 2, width // 2)) for f in raw_frames],
            dim=2,
        )
        right = torch.cat(
            [self._camera_frame(f, right_key, size=(height // 2, width // 2)) for f in raw_frames],
            dim=2,
        )
        wrists = torch.cat([left, right], dim=0)  # [2, C, F, H/2, W/2]

        enc_high = self.streaming_vae.encode_chunk(head.to(vae_device).to(self.dtype))
        enc_wrists = self._frozen["streaming_vae_half"].encode_chunk(wrists.to(vae_device).to(self.dtype))
        enc_out = torch.cat([torch.cat(enc_wrists.split(1, dim=0), dim=-1), enc_high], dim=-2)
        return self._normalize_encoder_output(enc_out).to(self.device)

    # ------------------------------------------------------------------ #
    # KV cache                                                            #
    # ------------------------------------------------------------------ #
    def _init_streaming_cache(self) -> None:
        """Allocate the streaming KV cache for this episode."""
        config = self.config
        latent_h, latent_w = config.latent_hw
        patch = config.patch_size
        latent_token_per_chunk = (config.frame_chunk_size * latent_h * latent_w) // (patch[0] * patch[1] * patch[2])
        action_token_per_chunk = config.frame_chunk_size * config.action_per_frame
        self.transformer.create_empty_cache(
            _CACHE_NAME,
            config.attn_window,
            latent_token_per_chunk,
            action_token_per_chunk,
            device=self.device,
            dtype=self.dtype,
            batch_size=2 if self._use_cfg else 1,
        )

    def _repeat_input_for_cfg(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Duplicate a stream input for classifier-free guidance.

        Args:
            input_dict: A single-stream input dict.

        Returns:
            The same dict, with the batch axis doubled when CFG is active.
        """
        if self._use_cfg:
            input_dict["noisy_latents"] = input_dict["noisy_latents"].repeat(2, 1, 1, 1, 1)
            input_dict["text_emb"] = torch.cat(
                [
                    self._prompt_conditioning.to(self.dtype).clone(),
                    self._negative_prompt_conditioning.to(self.dtype).clone(),
                ],
                dim=0,
            )
            input_dict["grid_id"] = input_dict["grid_id"][None].repeat(2, 1, 1)
            input_dict["timesteps"] = input_dict["timesteps"][None].repeat(2, 1)
        else:
            input_dict["grid_id"] = input_dict["grid_id"][None]
            input_dict["timesteps"] = input_dict["timesteps"][None]
        return input_dict

    def _prepare_stream_inputs(
        self,
        latent_model_input: torch.Tensor | None,
        action_model_input: torch.Tensor | None,
        latent_t: torch.Tensor | float = 0,
        action_t: torch.Tensor | float = 0,
        latent_cond: torch.Tensor | None = None,
        action_cond: torch.Tensor | None = None,
        frame_st_id: int = 0,
    ) -> dict[str, dict[str, Any]]:
        """Build the per-stream transformer inputs for one denoising step.

        Args:
            latent_model_input: Current video latents, or ``None`` to skip that stream.
            action_model_input: Current actions, or ``None`` to skip that stream.
            latent_t: Timestep of the video stream.
            action_t: Timestep of the action stream.
            latent_cond: Clean conditioning latent pinned at frame 0, if any.
            action_cond: Clean conditioning action pinned at frame 0, if any.
            frame_st_id: Absolute frame index of this chunk's first frame.

        Returns:
            Dict with ``latent_res_lst`` and/or ``action_res_lst`` stream inputs.
        """
        config = self.config
        device = self.device
        patch = config.patch_size
        out: dict[str, dict[str, Any]] = {}

        if latent_model_input is not None:
            out["latent_res_lst"] = {
                "noisy_latents": latent_model_input,
                "timesteps": torch.ones([latent_model_input.shape[2]], dtype=torch.float32, device=device) * latent_t,
                "grid_id": get_mesh_id(
                    latent_model_input.shape[-3] // patch[0],
                    latent_model_input.shape[-2] // patch[1],
                    latent_model_input.shape[-1] // patch[2],
                    0,
                    1,
                    frame_st_id,
                ).to(device),
                "text_emb": self._prompt_conditioning.to(self.dtype).clone(),
            }
            if latent_cond is not None:
                out["latent_res_lst"]["noisy_latents"][:, :, 0:1] = latent_cond[:, :, 0:1]
                out["latent_res_lst"]["timesteps"][0:1] *= 0

        if action_model_input is not None:
            out["action_res_lst"] = {
                "noisy_latents": action_model_input,
                "timesteps": torch.ones([action_model_input.shape[2]], dtype=torch.float32, device=device) * action_t,
                "grid_id": get_mesh_id(
                    action_model_input.shape[-3],
                    action_model_input.shape[-2],
                    action_model_input.shape[-1],
                    1,
                    1,
                    frame_st_id,
                    action=True,
                ).to(device),
                "text_emb": self._prompt_conditioning.to(self.dtype).clone(),
            }
            if action_cond is not None:
                out["action_res_lst"]["noisy_latents"][:, :, 0:1] = action_cond[:, :, 0:1]
                out["action_res_lst"]["timesteps"][0:1] *= 0
            out["action_res_lst"]["noisy_latents"][:, ~self._action_channel_mask] *= 0

        return out

    @property
    def _action_channel_mask(self) -> torch.Tensor:
        """Boolean mask over ``action_dim`` selecting the channels this checkpoint drives."""
        mask = torch.zeros([self.config.action_dim], dtype=torch.bool)
        mask[list(self.config.used_action_channel_ids)] = True
        return mask

    def _commit_kv_cache(
        self,
        obs_buffer: list[dict[str, torch.Tensor]],
        executed_actions: torch.Tensor | None,
    ) -> None:
        """Feed the real observed keyframes and the executed actions back into the KV cache.

        Args:
            obs_buffer: Keyframes observed while the previous chunk was executed.
            executed_actions: The previous chunk's actions in model space.
        """
        if not obs_buffer or executed_actions is None:
            return

        self.transformer.clear_pred_cache(_CACHE_NAME)
        latent_model_input = self._encode_frames(obs_buffer)
        if self._frame_st_id == 0 and self._init_latent is not None:
            # Prepend the conditioning latent so the latent and action frame counts align.
            latent_model_input = torch.cat([self._init_latent, latent_model_input], dim=2)

        action_model_input = executed_actions.to(latent_model_input)
        input_dict = self._prepare_stream_inputs(
            latent_model_input,
            action_model_input,
            frame_st_id=self._frame_st_id,
        )
        self.transformer(
            self._repeat_input_for_cfg(input_dict["latent_res_lst"]),
            update_cache=2,
            cache_name=_CACHE_NAME,
            action_mode=False,
        )
        self.transformer(
            self._repeat_input_for_cfg(input_dict["action_res_lst"]),
            update_cache=2,
            cache_name=_CACHE_NAME,
            action_mode=True,
        )
        self._frame_st_id += latent_model_input.shape[2]

    # ------------------------------------------------------------------ #
    # The dual-stream denoising loop                                      #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _infer(self, init_latent: torch.Tensor | None, frame_st_id: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """Denoise one chunk: the video-latent stream first, then the action stream.

        Args:
            init_latent: The conditioning latent for the episode's first chunk.
            frame_st_id: Absolute frame index of this chunk's first frame.

        Returns:
            Tuple of ``(actions, latents)``; actions are ``[1, action_dim, F, apf, 1]``.
        """
        config = self.config
        device = self.device
        latent_h, latent_w = config.latent_hw
        num_frames = config.frame_chunk_size

        latents = torch.randn(
            1,
            config.in_channels,
            num_frames,
            latent_h,
            latent_w,
            device=device,
            dtype=self.dtype,
        )
        actions = torch.randn(
            1,
            config.action_dim,
            num_frames,
            config.action_per_frame,
            1,
            device=device,
            dtype=self.dtype,
        )

        self._scheduler.set_timesteps(config.num_inference_steps)
        self._action_scheduler.set_timesteps(config.action_num_inference_steps)
        timesteps = F.pad(self._scheduler.timesteps, (0, 1), mode="constant", value=0)
        if config.video_exec_step != -1:
            timesteps = timesteps[: config.video_exec_step]
        action_timesteps = F.pad(self._action_scheduler.timesteps, (0, 1), mode="constant", value=0)

        latents = self._denoise_video(latents, timesteps, init_latent, frame_st_id, latent_h, latent_w)
        actions = self._denoise_actions(actions, action_timesteps, frame_st_id)

        actions[:, ~self._action_channel_mask] *= 0
        return actions, latents

    def _denoise_video(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        init_latent: torch.Tensor | None,
        frame_st_id: int,
        latent_h: int,
        latent_w: int,
    ) -> torch.Tensor:
        """Run the video-latent denoising loop for one chunk.

        The final step is run with ``update_cache=1`` so its keys/values are committed to
        the KV cache as this chunk's predicted video context.

        Args:
            latents: Initial noise ``[1, C, F, h, w]``.
            timesteps: Denoising timestep grid (zero-padded by one step).
            init_latent: Conditioning latent for the first chunk, or ``None``.
            frame_st_id: Absolute frame index of this chunk's first frame.
            latent_h: Latent grid height.
            latent_w: Latent grid width.

        Returns:
            The denoised video latents.
        """
        config = self.config
        for i, t in enumerate(timesteps):
            last_step = i == len(timesteps) - 1
            latent_cond = (
                init_latent[:, :, 0:1].to(self.dtype) if frame_st_id == 0 and init_latent is not None else None
            )
            input_dict = self._prepare_stream_inputs(latents, None, t, t, latent_cond, None, frame_st_id=frame_st_id)
            noise_pred = self.transformer(
                self._repeat_input_for_cfg(input_dict["latent_res_lst"]),
                update_cache=1 if last_step else 0,
                cache_name=_CACHE_NAME,
                action_mode=False,
            )
            if not last_step or config.video_exec_step != -1:
                noise_pred = data_seq_to_patch(
                    tuple(config.patch_size),  # type: ignore[arg-type]
                    noise_pred,
                    config.frame_chunk_size,
                    latent_h,
                    latent_w,
                    batch_size=2 if self._use_cfg else 1,
                )
                if config.guidance_scale > 1:
                    noise_pred = noise_pred[1:] + config.guidance_scale * (noise_pred[:1] - noise_pred[1:])
                else:
                    noise_pred = noise_pred[:1]
                latents = self._scheduler.step(noise_pred, t, latents)
            if frame_st_id == 0 and latent_cond is not None:
                latents[:, :, 0:1] = latent_cond
        return latents

    def _denoise_actions(
        self,
        actions: torch.Tensor,
        action_timesteps: torch.Tensor,
        frame_st_id: int,
    ) -> torch.Tensor:
        """Run the action denoising loop for one chunk.

        Args:
            actions: Initial noise ``[1, action_dim, F, action_per_frame, 1]``.
            action_timesteps: Denoising timestep grid (zero-padded by one step).
            frame_st_id: Absolute frame index of this chunk's first frame.

        Returns:
            The denoised actions in model space.
        """
        config = self.config
        for i, t in enumerate(action_timesteps):
            last_step = i == len(action_timesteps) - 1
            action_cond = (
                torch.zeros(
                    [1, config.action_dim, 1, config.action_per_frame, 1],
                    device=self.device,
                    dtype=self.dtype,
                )
                if frame_st_id == 0
                else None
            )
            input_dict = self._prepare_stream_inputs(None, actions, t, t, None, action_cond, frame_st_id=frame_st_id)
            noise_pred = self.transformer(
                self._repeat_input_for_cfg(input_dict["action_res_lst"]),
                update_cache=1 if last_step else 0,
                cache_name=_CACHE_NAME,
                action_mode=True,
            )
            if not last_step:
                noise_pred = rearrange(noise_pred, "b (f n) c -> b c f n 1", f=config.frame_chunk_size)
                if config.action_guidance_scale > 1:
                    noise_pred = noise_pred[1:] + config.action_guidance_scale * (noise_pred[:1] - noise_pred[1:])
                else:
                    noise_pred = noise_pred[:1]
                actions = self._action_scheduler.step(noise_pred, t, actions)
            if frame_st_id == 0 and action_cond is not None:
                actions[:, :, 0:1] = action_cond
        return actions

    # ------------------------------------------------------------------ #
    # Predicted-video decoding (opt-in)                                   #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def decode_predicted_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """VAE-decode predicted latents into a uint8 frame stack.

        Args:
            latents: Predicted latents of shape ``[1, z_dim, F, h, w]``.

        Returns:
            Frames of shape ``[F * 4, H, W, 3]`` as uint8 on CPU.
        """
        vae = self.vae
        vae_device = next(vae.parameters()).device
        latents = latents.to(device=vae_device, dtype=vae.dtype)
        latents = denormalize_latents(latents, vae.config.latents_mean, vae.config.latents_std, vae.config.z_dim)
        video = vae.decode(latents, return_dict=False)[0]  # [B, C, F, H, W] in [-1, 1]
        video = (video.float().clamp(-1, 1) + 1.0) / 2.0
        return (video[0].permute(1, 2, 3, 0) * 255.0).round().to(torch.uint8).cpu()


__all__ = ["LingBotVAModel"]
