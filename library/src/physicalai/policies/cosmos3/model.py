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
    _ACTION_VIEWPOINT_TEMPLATES,  # ruff: ignore[import-private-name]
    _EMBODIMENT_TO_DOMAIN_ID,  # ruff: ignore[import-private-name]
    _EMBODIMENT_TO_RAW_ACTION_DIM,  # ruff: ignore[import-private-name]
)
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from huggingface_hub import file_exists
from PIL import Image

from physicalai.data.observation import ACTION, IMAGES, STATE
from physicalai.policies.base import Model

from .flow_matching import build_action_tokens, build_pack, flow_matching_step
from .normalization import load_stats_file, resolve_affine
from .pipeline import PolicyPipelineWithState
from .preprocessor import Cosmos3Preprocessor
from .representation import (
    embodiment_gripper_flipped,
    embodiment_normalization,
    flip_gripper_last_channel,
    resolve_action_space,
)
from .surgery import configure_trainable, init_domain_action_head

if TYPE_CHECKING:
    from .config import Cosmos3Config

logger = logging.getLogger(__name__)

# Register ALOHA embodiment if not already present in diffusers
_EMBODIMENT_TO_DOMAIN_ID.setdefault("aloha", 10)
_EMBODIMENT_TO_RAW_ACTION_DIM.setdefault("aloha", 14)

# The released DROID policy checkpoints (e.g. nvidia/cosmos3-edge-policy-droid, model card:
# 8D DROID action) emit an 8D ``joint_pos`` action ``[joint(7), gripper(1)]``, not the 10D
# ee_pose the stock diffusers table assumes. Pin DROID to 8 so the head output is sliced
# and conditioned at its true width.
_EMBODIMENT_TO_RAW_ACTION_DIM["droid_lerobot"] = 8

# Register top_down_2d_view in diffusers action viewpoint templates if not present
_ACTION_VIEWPOINT_TEMPLATES.setdefault(
    "top_down_2d_view",
    "This video is captured from a top-down view of a flat 2D scene.",
)


def _has_pretrained_action_head(pretrained_path: str, embodiment: str) -> bool:
    """Check whether a model path or repository carries an already trained action head.

    Args:
        pretrained_path: Path to local directory or Hugging Face model repository ID.
        embodiment: Embodiment identifier.

    Returns:
        True if the checkpoint already contains action head weights or policy metadata.
    """
    path_obj = Path(pretrained_path)
    if path_obj.is_dir():
        return (path_obj / f"{embodiment}_head.pt").is_file() or (path_obj / "checkpoint.json").is_file()

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


def _materialize_meta_parameters(module: torch.nn.Module | None) -> None:
    """Materialize any uninitialized (meta) parameters on CPU to avoid device-transfer errors.

    Args:
        module: PyTorch module whose meta parameters will be instantiated with zeros.
    """
    if module is None:
        return
    for name, param in list(module.named_parameters()):
        if param.is_meta:
            p_name, _, c_name = name.rpartition(".")
            parent = module.get_submodule(p_name) if p_name else module
            setattr(
                parent,
                c_name,
                torch.nn.Parameter(torch.zeros(param.shape, dtype=param.dtype, device="cpu")),
            )


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
            ValueError: If the configured embodiment or action_space is unrecognized.
        """
        super().__init__()
        self.config = config

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "float16": torch.float16,
        }
        self.torch_dtype = dtype_map.get(config.dtype, torch.bfloat16)

        # Embodiment metadata
        if config.embodiment not in _EMBODIMENT_TO_DOMAIN_ID:
            msg = f"Unknown embodiment '{config.embodiment}'. Registered: {list(_EMBODIMENT_TO_DOMAIN_ID.keys())}"
            raise ValueError(msg)

        self.domain_id_val = _EMBODIMENT_TO_DOMAIN_ID[config.embodiment]
        if config.embodiment not in _EMBODIMENT_TO_RAW_ACTION_DIM:
            # No silent fallback: an unregistered embodiment would slice actions to the wrong width.
            msg = (
                f"Embodiment '{config.embodiment}' has no registered raw action width. "
                f"Registered: {sorted(_EMBODIMENT_TO_RAW_ACTION_DIM)}. Add its canonical action "
                "width to _EMBODIMENT_TO_RAW_ACTION_DIM before using it."
            )
            raise ValueError(msg)
        self.raw_dim = _EMBODIMENT_TO_RAW_ACTION_DIM[config.embodiment]

        # Action space: resolved from the embodiment, overridable via config.action_space.
        # Only ``identity`` (pusht/aloha) and ``joint_pos`` (droid_lerobot) are supported; both
        # pass the raw action column through, joint_pos additionally flips the gripper. The
        # normalization method (none/minmax/...) mirrors the cosmos-framework dataset defaults.
        self.action_space = resolve_action_space(config.embodiment, getattr(config, "action_space", None))
        if self.action_space not in {"identity", "joint_pos"}:
            msg = (
                f"Unsupported action_space '{self.action_space}' for embodiment '{config.embodiment}'. "
                "Supported: 'identity', 'joint_pos'."
            )
            raise ValueError(msg)
        self.norm_method = embodiment_normalization(config.embodiment)

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
            _materialize_meta_parameters(self.pipe.transformer)
            _materialize_meta_parameters(self.pipe.vae)

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
        if not _has_pretrained_action_head(config.pretrained_model_name_or_path, config.embodiment):
            init_domain_action_head(self.transformer, self.domain_id_val)
        else:
            logger.info(
                "Preserving pretrained action head weights from %s for embodiment '%s'",
                config.pretrained_model_name_or_path,
                config.embodiment,
            )

        if config.grad_checkpoint:
            self.transformer.enable_gradient_checkpointing()

        # Input preprocessor for view composition and viewpoint metadata
        self.preprocessor = Cosmos3Preprocessor(
            embodiment=config.embodiment,
            view_point=getattr(config, "view_point", None),
        )

        # Cache for fixed-shape sequence packs across steps
        self._pack_cache: dict[tuple[str, str, int, int, str | None], dict[str, Any]] = {}

        # Action normalization as an affine (offset, scale): normalize=(x-offset)/scale,
        # denormalize=x*scale+offset. Default identity ("none"); populated from an explicit
        # stats file or the dataset statistics below. Persisted with the domain head.
        self.register_buffer("norm_offset", torch.zeros(self.raw_dim, dtype=torch.float32))
        self.register_buffer("norm_scale", torch.ones(self.raw_dim, dtype=torch.float32))
        self.register_buffer(
            "domain_id",
            torch.tensor([self.domain_id_val], dtype=torch.long),
        )

        stats_path = getattr(config, "normalizer_stats_path", None)
        if stats_path is not None:
            self._load_normalizer_stats_file(stats_path)
        elif dataset_stats is not None:
            self.set_dataset_stats(dataset_stats)

    def _set_affine(self, offset: torch.Tensor, scale: torch.Tensor) -> None:
        """Store the normalization affine, broadcasting scalars to ``raw_dim``."""
        offset = offset.reshape(-1).float()
        scale = scale.reshape(-1).float()
        if offset.numel() == 1:
            offset = offset.expand(self.raw_dim).clone()
        if scale.numel() == 1:
            scale = scale.expand(self.raw_dim).clone()
        self.raw_dim = offset.numel()
        self.norm_offset = offset.to(self.norm_offset.device)
        self.norm_scale = scale.to(self.norm_scale.device)

    def _load_normalizer_stats_file(self, path: str) -> None:
        """Load an explicit cosmos-format stats file and set the normalization affine.

        The embodiment's registered method is used; a ``none`` embodiment (e.g. DROID) is
        promoted to ``quantile`` since supplying a stats file signals normalized actions.

        Raises:
            ValueError: If the stats width does not match the raw action dim.
        """
        method = self.norm_method if self.norm_method != "none" else "quantile"
        stats = load_stats_file(path, method)
        offset, scale = resolve_affine(method, stats)
        if offset.numel() != self.raw_dim:
            msg = (
                f"Normalizer stats width {offset.numel()} from {path!r} does not match the "
                f"raw action dim {self.raw_dim} for embodiment '{self.config.embodiment}'."
            )
            raise ValueError(msg)
        self.norm_method = method
        self._set_affine(offset, scale)
        logger.info("Loaded %s normalizer stats for embodiment '%s' from %s", method, self.config.embodiment, path)

    def set_dataset_stats(self, dataset_stats: dict[str, Any]) -> None:
        """Update the normalization affine from dataset statistics.

        Only ``identity`` embodiments consume dataset statistics: their raw action column is
        the model action space. ``joint_pos`` (DROID) keeps actions un-normalized, so its
        raw-column dataset stats are ignored — supply ``normalizer_stats_path`` to normalize.
        """
        if self.action_space != "identity":
            logger.info(
                "Embodiment '%s' uses the %s action space; dataset stats are ignored "
                "(pass normalizer_stats_path to normalize). Actions stay un-normalized.",
                self.config.embodiment,
                self.action_space,
            )
            return

        act_stat = dataset_stats.get(ACTION, dataset_stats.get("action", {}))
        stats = {
            k: torch.as_tensor(v, dtype=torch.float32) for k, v in act_stat.items() if isinstance(v, (list, tuple))
        }
        method = self.norm_method
        if method in {"quantile", "quantile_rot"} and not {"q01", "q99"} <= stats.keys():
            method = "minmax"  # dataset lacks quantiles; fall back to min-max
        if method == "minmax" and not {"min", "max"} <= stats.keys():
            return
        offset, scale = resolve_affine(method, stats)
        self._set_affine(offset, scale)

    def _normalize_action(self, x: torch.Tensor) -> torch.Tensor:
        """Affine-normalize the leading ``raw_dim`` channels: ``(x - offset) / scale``.

        Returns:
            The normalized tensor.
        """
        d = min(x.shape[-1], self.raw_dim)
        return (x[..., :d] - self.norm_offset[:d].to(x)) / self.norm_scale[:d].to(x)

    def _denormalize_action(self, y: torch.Tensor) -> torch.Tensor:
        """Invert :meth:`_normalize_action`: ``y * scale + offset``.

        Returns:
            The denormalized tensor.
        """
        d = min(y.shape[-1], self.raw_dim)
        return y[..., :d] * self.norm_scale[:d].to(y) + self.norm_offset[:d].to(y)

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
        view_point: str | None = None,
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
            view_point: Optional camera viewpoint tag for prompt framing.

        Returns:
            Precomputed sequence pack dictionary.
        """
        key = (paradigm, prompt, height, width, view_point)
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
                view_point=view_point,
            )
        return self._pack_cache[key]

    def compute_loss(
        self,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        """Compute flow matching loss on the training batch.

        Args:
            batch: Dictionary containing observation fields.

        Returns:
            Tuple of (total_loss, metrics_dict).
        """
        preprocessed = self.preprocessor(batch)
        images_tensor = preprocessed[IMAGES]
        view_point = preprocessed.get("view_point")
        actions_tensor = preprocessed.get(ACTION)
        state_tensor = preprocessed.get(STATE)

        # State arrives as a single combined ``[B, (T,) raw_dim]`` column (the datamodule owns
        # composing split dataset sub-columns into one state vector).
        state_seq_batch = state_tensor

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

            # Identity (pusht/aloha) and joint_pos (DROID) action spaces: the raw action column
            # is already the model action space. joint_pos additionally inverts the DROID gripper
            # (cosmos-framework ``_is_gripper_action_flipped``) on both the action chunk and the
            # prepended state token.
            act_b = actions_tensor[b].to(device=device, dtype=torch.float32)
            if act_b.ndim == 1:
                act_b = act_b.unsqueeze(0)
            act_raw = act_b[:, : self.raw_dim]
            if state_seq_batch is not None:
                st_b = state_seq_batch[b].to(device=device, dtype=torch.float32)
                if st_b.ndim > 1:
                    st_b = st_b[0]  # Take initial frame state
                st_raw = st_b[: self.raw_dim]
            else:
                st_raw = None
            if self.action_space == "joint_pos" and embodiment_gripper_flipped(self.config.embodiment):
                act_raw = flip_gripper_last_channel(act_raw)
                if st_raw is not None:
                    st_raw = flip_gripper_last_channel(st_raw)

            # Apply the embodiment's normalization affine uniformly to the state token and chunk.
            act_norm = self._normalize_action(act_raw)
            st_norm = (
                self._normalize_action(st_raw)
                if st_raw is not None
                else torch.zeros(self.raw_dim, device=device, dtype=torch.float32)
            )

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
                view_point=view_point,
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

    def _log_action_channel_stats(self, action_chunk: torch.Tensor) -> None:
        """Log per-channel min/max/mean/std of the raw action-head output at DEBUG level."""
        stats = action_chunk.detach().float().cpu()
        mins = stats.amin(dim=0).tolist()
        maxs = stats.amax(dim=0).tolist()
        means = stats.mean(dim=0).tolist()
        stds = stats.std(dim=0).tolist()
        rows = "\n".join(
            f"  ch{i:02d}: min={mn:+.4f} max={mx:+.4f} mean={mu:+.4f} std={sd:.4f}"
            for i, (mn, mx, mu, sd) in enumerate(zip(mins, maxs, means, stds, strict=False))
        )
        logger.debug(
            "Cosmos3 raw action output (embodiment=%s, action_space=%s, norm_method=%s):\n%s",
            self.config.embodiment,
            self.action_space,
            self.norm_method,
            rows,
        )

    @torch.no_grad()
    def predict_action_chunk(
        self,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        """Predict a chunk of actions for the given observation.

        Args:
            batch: Batch dictionary containing image and optional state.

        Returns:
            Predicted action tensor of shape (B, chunk_size, raw_dim).
        """
        preprocessed = self.preprocessor(batch)
        img_tensor = preprocessed[IMAGES]
        view_point = preprocessed.get("view_point")
        state_tensor = preprocessed.get(STATE)
        # State arrives as a single combined column (the datamodule owns composing split
        # dataset sub-columns into one state vector).
        state_field = state_tensor

        device = self.transformer.device
        batch_size = img_tensor.shape[0] if img_tensor.ndim in {4, 5} else 1

        preds = []
        for b in range(batch_size):
            cur_img = img_tensor[b] if img_tensor.ndim in {4, 5} else img_tensor
            if cur_img.ndim == 4:  # [T, C, H, W] -> take latest frame # ruff: ignore[magic-value-comparison]
                cur_img = cur_img[-1]
            pil_img = _to_pil_image(cur_img)

            if state_field is not None:
                cur_state = state_field[b] if state_field.ndim > 1 else state_field
                cur_state = cur_state.to(device=device, dtype=torch.float32)
                if cur_state.ndim > 1:
                    cur_state = cur_state[-1]
                if self.action_space == "identity":
                    native_state = cur_state[: self.raw_dim]
                else:
                    # joint_pos (DROID): raw joint+gripper state token; invert the DROID gripper
                    # to the model's convention and anchor the decoded chunk on it (use_state parity).
                    native_state = cur_state[: self.raw_dim]
                    if embodiment_gripper_flipped(self.config.embodiment):
                        native_state = flip_gripper_last_channel(native_state)
                cond_state = self._normalize_action(native_state)
                self.pipe.current_state = cond_state
            else:
                self.pipe.current_state = None

            condition = CosmosActionCondition(
                mode="policy",
                chunk_size=self.config.chunk_size,
                domain_name=self.config.embodiment,
                resolution_tier=self.config.resolution_tier,
                image=pil_img,
                view_point=view_point,
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

            # Raw action-head output before any denorm; reveals per-channel calibration
            # (e.g. rot6d/gripper in [-1, 1] vs translation scale) for the current checkpoint.
            if logger.isEnabledFor(logging.DEBUG):
                self._log_action_channel_stats(action_chunk)

            # Invert the normalization affine back into the model action space.
            native_chunk = self._denormalize_action(action_chunk)

            # Identity/joint_pos: the denormalized chunk is already in the raw dataset space.
            out_action = native_chunk

            preds.append(out_action)

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
