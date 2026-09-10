# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Cosmos3 model implementation based on diffusers Cosmos3OmniPipeline."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from diffusers import CosmosActionCondition
from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import (
    _EMBODIMENT_TO_DOMAIN_ID,  # ruff: ignore[import-private-name]
    _EMBODIMENT_TO_RAW_ACTION_DIM,  # ruff: ignore[import-private-name]
)
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from huggingface_hub import file_exists
from PIL import Image

from physicalai.data.observation import ACTION, IMAGES, STATE
from physicalai.policies.base import Model

from .flow_matching import build_action_tokens, build_pack, flow_matching_step
from .pipeline import PolicyPipelineWithState
from .surgery import configure_trainable, init_domain_action_head

if TYPE_CHECKING:
    from .config import Cosmos3Config

logger = logging.getLogger(__name__)

# Register ALOHA embodiment if not already present in diffusers
_EMBODIMENT_TO_DOMAIN_ID.setdefault("aloha", 10)
_EMBODIMENT_TO_RAW_ACTION_DIM.setdefault("aloha", 14)


def _has_pretrained_action_head(pretrained_path: str, domain: str) -> bool:
    """Check whether a model path or repository carries an already trained action head.

    Args:
        pretrained_path: Path to local directory or Hugging Face model repository ID.
        domain: Embodiment domain identifier.

    Returns:
        True if the checkpoint already contains action head weights or policy metadata.
    """
    path_obj = Path(pretrained_path)
    if path_obj.is_dir():
        return (path_obj / f"{domain}_head.pt").is_file() or (path_obj / "checkpoint.json").is_file()

    try:
        return file_exists(repo_id=pretrained_path, filename="checkpoint.json")
    except Exception:  # ruff: ignore[blind-except]
        return False


def _extract_image_tensor(batch: dict[str, Any]) -> torch.Tensor:
    """Extract primary image tensor from a batch dictionary.

    Args:
        batch: Dictionary containing observation fields.

    Returns:
        Extracted image tensor.

    Raises:
        KeyError: If no image tensor can be located in the batch.
    """
    img_val = batch.get(IMAGES)
    if img_val is None:
        for k in ("pixels", "observation.images", "image"):
            if k in batch:
                img_val = batch[k]
                break

    if img_val is None:
        for k, v in batch.items():
            if "image" in k or "pixel" in k:
                img_val = v
                break

    if img_val is None:
        msg = "No image tensor found in batch."
        raise KeyError(msg)

    if isinstance(img_val, dict):
        img_val = next(iter(img_val.values()))

    if not isinstance(img_val, torch.Tensor):
        img_val = torch.as_tensor(img_val)

    return img_val


def _to_pil_image(img_tensor: torch.Tensor) -> Image.Image:
    """Convert a single image tensor of shape (C, H, W) or (H, W, C) to a PIL Image.

    Args:
        img_tensor: Image tensor to convert.

    Returns:
        Converted PIL Image.
    """
    t = img_tensor.detach().cpu()
    if t.ndim == 4:  # ruff: ignore[magic-value-comparison]
        t = t[0]
    if t.shape[0] == 3:  # (3, H, W) -> (H, W, 3) # ruff: ignore[magic-value-comparison]
        t = t.permute(1, 2, 0)
    if t.dtype in {torch.float32, torch.float16, torch.bfloat16}:
        if t.max() <= 1.0:
            t = (t * 255.0).clamp(0, 255)
        t = t.to(torch.uint8)
    arr = t.numpy()
    return Image.fromarray(arr)


def _format_images_sequence(images_tensor: torch.Tensor, target_len: int) -> torch.Tensor:
    """Format images tensor to shape [B, target_len, C, H, W].

    Args:
        images_tensor: Raw input images tensor.
        target_len: Expected temporal length (frames).

    Returns:
        Standardized [B, target_len, C, H, W] images tensor.

    Raises:
        ValueError: If images tensor dimension is unsupported.
    """
    if images_tensor.ndim == 4:  # [B, C, H, W] # ruff: ignore[magic-value-comparison]
        return images_tensor.unsqueeze(1).repeat(1, target_len, 1, 1, 1)
    if images_tensor.ndim == 5:  # [B, T, C, H, W] or [B, C, T, H, W] # ruff: ignore[magic-value-comparison]
        if images_tensor.shape[1] == 3 and images_tensor.shape[2] != 3:  # ruff: ignore[magic-value-comparison]
            images_seq = images_tensor.permute(0, 2, 1, 3, 4)
        else:
            images_seq = images_tensor

        t_len = images_seq.shape[1]
        if t_len < target_len:
            pad = images_seq[:, -1:].repeat(1, target_len - t_len, 1, 1, 1)
            return torch.cat([images_seq, pad], dim=1)
        if t_len > target_len:
            return images_seq[:, :target_len]
        return images_seq

    msg = f"Unexpected images tensor shape: {images_tensor.shape}"
    raise ValueError(msg)


class Cosmos3Model(Model):
    """Cosmos 3 PyTorch Model wrapping diffusers Cosmos3OmniPipeline."""

    def __init__(
        self,
        config: Cosmos3Config,
        pipeline: PolicyPipelineWithState | None = None,
        dataset_stats: dict[str, Any] | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        """Initialize Cosmos3Model.

        Args:
            config: Policy configuration.
            pipeline: Pre-initialized PolicyPipelineWithState instance, or None to load from pretrained.
            dataset_stats: Normalization statistics for action and state space.
            device: Target device for execution.

        Raises:
            ValueError: If the configured domain is unrecognized.
        """
        super().__init__()
        self.config = config

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "float16": torch.float16,
        }
        self.torch_dtype = dtype_map.get(config.dtype, torch.bfloat16)

        # Domain metadata
        if config.domain not in _EMBODIMENT_TO_DOMAIN_ID:
            msg = f"Unknown domain '{config.domain}'. Registered: {list(_EMBODIMENT_TO_DOMAIN_ID.keys())}"
            raise ValueError(msg)

        self.domain_id_val = _EMBODIMENT_TO_DOMAIN_ID[config.domain]
        self.raw_dim = _EMBODIMENT_TO_RAW_ACTION_DIM.get(config.domain, 2)

        # Build or wrap the pipeline
        if pipeline is not None:
            self.pipe = pipeline
        else:
            logger.info("Loading Cosmos3 pipeline from %s", config.pretrained_model_name_or_path)
            self.pipe = PolicyPipelineWithState.from_pretrained(
                config.pretrained_model_name_or_path,
                torch_dtype=self.torch_dtype,
                enable_safety_checker=False,
            )

        if device is not None:
            self.pipe.to(device)

        # Update scheduler flow shift for inference
        if hasattr(self.pipe, "scheduler") and self.pipe.scheduler is not None:
            self.pipe.scheduler = UniPCMultistepScheduler.from_config(
                self.pipe.scheduler.config,
                flow_shift=config.flow_shift,
            )

        # Register submodules for PyTorch parameter tracking and device movement
        self.transformer = self.pipe.transformer
        self.vae = self.pipe.vae

        self.action_dim = getattr(self.transformer.config, "action_dim", 64)

        # Model surgery: freeze backbone and configure trainable layers
        self.vae.requires_grad_(requires_grad=False)
        configure_trainable(
            self.transformer,
            mode=config.mode,
            rank=config.rank,
            alpha_scale=config.alpha_scale,
            dora=config.dora,
        )
        if not _has_pretrained_action_head(config.pretrained_model_name_or_path, config.domain):
            init_domain_action_head(self.transformer, self.domain_id_val)
        else:
            logger.info(
                "Preserving pretrained action head weights from %s for domain '%s'",
                config.pretrained_model_name_or_path,
                config.domain,
            )

        if config.grad_checkpoint:
            self.transformer.enable_gradient_checkpointing()

        # Cache for fixed-shape sequence packs across steps
        self._pack_cache: dict[tuple[str, str, int, int], dict[str, Any]] = {}

        # Action normalization bounds
        self.register_buffer("a_min", -torch.ones(self.raw_dim, dtype=torch.float32))
        self.register_buffer("a_max", torch.ones(self.raw_dim, dtype=torch.float32))
        self.register_buffer(
            "domain_id",
            torch.tensor([self.domain_id_val], dtype=torch.long),
        )

        if dataset_stats is not None:
            self.set_dataset_stats(dataset_stats)

    def set_dataset_stats(self, dataset_stats: dict[str, Any]) -> None:
        """Update normalization bounds from dataset statistics.

        Args:
            dataset_stats: Normalization statistics dictionary.
        """
        act_stat = dataset_stats.get(ACTION, dataset_stats.get("action", {}))
        if "min" in act_stat and "max" in act_stat:
            a_min = torch.as_tensor(act_stat["min"], dtype=torch.float32)
            a_max = torch.as_tensor(act_stat["max"], dtype=torch.float32)
        elif "q01" in act_stat and "q99" in act_stat:
            a_min = torch.as_tensor(act_stat["q01"], dtype=torch.float32)
            a_max = torch.as_tensor(act_stat["q99"], dtype=torch.float32)
        else:
            return

        self.raw_dim = len(a_min)
        self.a_min = a_min.to(self.a_min.device)
        self.a_max = a_max.to(self.a_max.device)

    def to(self, *args: object, **kwargs: object) -> Cosmos3Model:
        """Move the model and underlying pipeline components to the specified device.

        Args:
            *args: Positional device/dtype arguments.
            **kwargs: Keyword device/dtype arguments.

        Returns:
            Self with updated device.
        """
        super().to(*args, **kwargs)
        device = None
        for arg in args:
            if isinstance(arg, (torch.device, str)):
                device = arg
                break
        if "device" in kwargs and isinstance(kwargs["device"], (torch.device, str)):
            device = kwargs["device"]
        if device is not None and hasattr(self, "pipe"):
            self.pipe.to(device)
            self.domain_id = self.domain_id.to(device)
        return self

    def _get_or_build_pack(
        self,
        paradigm: str,
        prompt: str,
        x0_vision: torch.Tensor,
        chunk: int,
        height: int,
        width: int,
        fps: int,
        action_dim: int,
        device: torch.device | str,
    ) -> dict[str, Any]:
        """Retrieve cached sequence pack or construct a new one.

        Args:
            paradigm: Training objective name.
            prompt: Task instruction string.
            x0_vision: Clean vision latents.
            chunk: Action chunk length.
            height: Image height.
            width: Image width.
            fps: Frame rate.
            action_dim: Padded action dimension.
            device: Target execution device.

        Returns:
            Precomputed sequence pack dictionary.
        """
        key = (paradigm, prompt, height, width)
        if key not in self._pack_cache:
            self._pack_cache[key] = build_pack(
                self.pipe,
                paradigm,
                x0_vision,
                prompt,
                chunk,
                height,
                width,
                fps,
                action_dim,
                device,
            )
        return self._pack_cache[key]

    def compute_loss(  # ruff: ignore[too-many-locals]
        self,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        """Compute flow matching loss on the training batch.

        Args:
            batch: Dictionary containing observation fields.

        Returns:
            Tuple of (total_loss, metrics_dict).
        """
        images_tensor = _extract_image_tensor(batch)
        actions_tensor = batch[ACTION]
        state_tensor = batch.get(STATE)

        device = self.transformer.device
        dtype = self.transformer.dtype

        target_len = self.config.chunk_size + 1
        images_seq = _format_images_sequence(images_tensor, target_len)
        batch_size = images_seq.shape[0]

        # Determine paradigm for this step
        if self.config.paradigm == "joint":
            # nosec B311 - non-cryptographic objective sampling
            selected_paradigm = random.choice(("policy", "fd", "id"))  # ruff: ignore[suspicious-non-cryptographic-random-usage]
        else:
            selected_paradigm = self.config.paradigm

        batch_losses = []
        batch_losses_v = []
        batch_losses_a = []

        for b in range(batch_size):
            clip = images_seq[b]  # [T, C, H, W]
            with torch.no_grad():
                # ruff: ignore[private-member-access]
                video_clip, image_size, h, w = self.pipe._prepare_action_video_conditioning(
                    clip,
                    self.config.resolution_tier,
                    target_len,
                    device,
                    dtype,
                )
                x0_vision = self.pipe._remove_action_video_padding_from_latent(  # ruff: ignore[private-member-access]
                    self.pipe._encode_video(video_clip).float(),  # ruff: ignore[private-member-access]
                    image_size,
                )

            act_b = actions_tensor[b].to(device=device, dtype=torch.float32)
            if act_b.ndim == 1:
                act_b = act_b.unsqueeze(0)
            a_min_slice = self.a_min[: self.raw_dim]
            a_max_slice = self.a_max[: self.raw_dim]
            act_norm = 2.0 * (act_b[:, : self.raw_dim] - a_min_slice) / (a_max_slice - a_min_slice + 1e-8) - 1.0

            if state_tensor is not None:
                st_b = state_tensor[b].to(device=device, dtype=torch.float32)
                if st_b.ndim > 1:
                    st_b = st_b[0]  # Take initial frame state
                dim_st = min(len(st_b), self.raw_dim)
                st_min = self.a_min[:dim_st]
                st_max = self.a_max[:dim_st]
                st_norm = 2.0 * (st_b[:dim_st] - st_min) / (st_max - st_min + 1e-8) - 1.0
            else:
                st_norm = torch.zeros(self.raw_dim, device=device, dtype=torch.float32)

            x0_action = build_action_tokens(
                selected_paradigm,
                act_norm,
                st_norm,
                self.config.chunk_size,
                self.action_dim,
                self.raw_dim,
                device,
            )

            pack = self._get_or_build_pack(
                selected_paradigm,
                self.config.prompt,
                x0_vision,
                self.config.chunk_size,
                h,
                w,
                self.config.fps,
                self.action_dim,
                device,
            )

            loss_b, loss_v_b, loss_a_b = flow_matching_step(
                self.transformer,
                pack,
                x0_vision,
                x0_action,
                self.domain_id,
                dtype,
                self.raw_dim,
                self.config.action_weight,
                device,
            )

            batch_losses.append(loss_b)
            batch_losses_v.append(loss_v_b)
            batch_losses_a.append(loss_a_b)

        loss = torch.stack(batch_losses).mean()
        loss_v = torch.stack(batch_losses_v).mean()
        loss_a = torch.stack(batch_losses_a).mean()

        return loss, {"loss": loss, "loss_vision": loss_v, "loss_action": loss_a}

    @torch.no_grad()
    def compute_val_loss(
        self,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        """Compute validation loss on the batch.

        Args:
            batch: Dictionary containing observation fields.

        Returns:
            Tuple of (validation_loss, metrics_dict).
        """
        return self.compute_loss(batch)

    def forward(
        self,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]] | torch.Tensor:
        """Forward pass for training or inference.

        Args:
            batch: Dictionary containing observation fields.

        Returns:
            Loss tuple in training mode, or action chunk predictions during eval.
        """
        if self.training:
            return self.compute_loss(batch)
        return self.predict_action_chunk(batch)

    @torch.no_grad()
    def predict_action_chunk(  # ruff: ignore[too-many-locals]
        self,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        """Predict a chunk of actions for the given observation.

        Args:
            batch: Batch dictionary containing image and optional state.

        Returns:
            Predicted action tensor of shape (B, chunk_size, raw_dim).
        """
        img_tensor = _extract_image_tensor(batch)
        state_tensor = batch.get(STATE)

        device = self.transformer.device
        batch_size = img_tensor.shape[0] if img_tensor.ndim in {4, 5} else 1

        preds = []
        for b in range(batch_size):
            cur_img = img_tensor[b] if img_tensor.ndim in {4, 5} else img_tensor
            if cur_img.ndim == 4:  # [T, C, H, W] -> take latest frame # ruff: ignore[magic-value-comparison]
                cur_img = cur_img[-1]
            pil_img = _to_pil_image(cur_img)

            if state_tensor is not None:
                cur_state = state_tensor[b] if state_tensor.ndim > 1 else state_tensor
                cur_state = cur_state.to(device=device, dtype=torch.float32)
                if cur_state.ndim > 1:
                    cur_state = cur_state[-1]
                dim_st = min(len(cur_state), self.raw_dim)
                st_min = self.a_min[:dim_st].to(device)
                st_max = self.a_max[:dim_st].to(device)
                norm_state = 2.0 * (cur_state[:dim_st] - st_min) / (st_max - st_min + 1e-8) - 1.0
                self.pipe.current_state = norm_state
            else:
                self.pipe.current_state = None

            condition = CosmosActionCondition(
                mode="policy",
                chunk_size=self.config.chunk_size,
                domain_name=self.config.domain,
                resolution_tier=self.config.resolution_tier,
                image=pil_img,
                view_point=None,
            )

            result = self.pipe(
                prompt=self.config.prompt,
                action=condition,
                fps=self.config.fps,
                num_inference_steps=self.config.num_inference_steps,
                guidance_scale=self.config.guidance_scale,
                use_system_prompt=False,
                output_type="latent",
            )

            # In policy mode with current_state, result.action[0] has length chunk_size + 1 (first row is state)
            if self.pipe.current_state is not None and result.action[0].shape[0] > self.config.chunk_size:
                action_chunk = result.action[0][1 : self.config.chunk_size + 1, : self.raw_dim]
            else:
                action_chunk = result.action[0][: self.config.chunk_size, : self.raw_dim]

            # Denormalize to dataset space
            a_min_slice = self.a_min[: self.raw_dim].to(action_chunk)
            a_max_slice = self.a_max[: self.raw_dim].to(action_chunk)
            denorm_action = (action_chunk + 1.0) / 2.0 * (a_max_slice - a_min_slice) + a_min_slice

            preds.append(denorm_action)

        return torch.stack(preds, dim=0)

    @property
    def reward_delta_indices(self) -> list | None:
        """Reward delta indices (not implemented).

        Returns:
            None.
        """
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        """Action prediction horizons relative to current frame.

        Returns:
            List of integer relative horizons.
        """
        return list(range(self.config.chunk_size))

    @property
    def observation_delta_indices(self) -> list[int] | None:
        """Observation video window offsets relative to current frame.

        Returns:
            List of integer relative frame offsets.
        """
        return list(range(self.config.chunk_size + 1))
