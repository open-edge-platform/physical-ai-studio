# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: PLR2004

"""Ordered observations, Qwen prompts, and dataset-owned normalization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from tokenizers import Tokenizer
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812
from transformers import (
    PreTrainedTokenizerFast,
    Qwen2VLImageProcessor,
    Qwen3VLProcessor,
    Qwen3VLVideoProcessor,
    VJEPA2VideoProcessor,
)

from physicalai.data import Feature, FeatureType
from physicalai.policies.utils.normalization import FeatureNormalizeTransform, NormalizationType

if TYPE_CHECKING:
    from .config import VLAJEPAConfig


def _batch_key(feature: Feature) -> str:
    name = str(feature.name)
    if feature.ftype == FeatureType.VISUAL and not name.startswith("images."):
        return f"images.{name}"
    return name


def _normalizer(config: VLAJEPAConfig, *, inverse: bool = False) -> FeatureNormalizeTransform:
    features = {
        _batch_key(feature): feature
        for feature in config.input_features + config.output_features
        if feature.ftype in {FeatureType.STATE, FeatureType.ACTION}
    }
    modes = {
        FeatureType.STATE: NormalizationType(config.state_normalization),
        FeatureType.ACTION: NormalizationType(config.action_normalization),
    }
    required = {
        NormalizationType.MEAN_STD: ("mean", "std"),
        NormalizationType.MIN_MAX: ("min", "max"),
        NormalizationType.QUANTILES: ("q01", "q99"),
        NormalizationType.IDENTITY: (),
    }
    for feature in features.values():
        mode = modes[FeatureType.STATE] if feature.ftype == FeatureType.STATE else modes[FeatureType.ACTION]
        for stat in required[mode]:
            value = getattr(feature.normalization_data, stat, None)
            if value is None:
                msg = f"Feature {feature.name!r} requires {stat} statistics for {mode} normalization."
                raise ValueError(msg)
            tensor = torch.as_tensor(value)
            if tuple(tensor.shape) != feature.shape or not torch.isfinite(tensor).all():
                msg = f"Invalid {stat} statistics for feature {feature.name!r}."
                raise ValueError(msg)
        stats = feature.normalization_data
        bounds = required[mode]
        if stats is not None and bounds:
            lower, upper = bounds
            if mode == NormalizationType.MEAN_STD:
                valid = torch.as_tensor(stats.std) >= 0
            else:
                valid = torch.as_tensor(getattr(stats, upper)) >= torch.as_tensor(getattr(stats, lower))
            if not valid.all():
                msg = f"Inconsistent normalization bounds for {feature.name!r}."
                raise ValueError(msg)
    return FeatureNormalizeTransform(features, modes, inverse=inverse)


class VLAJEPAPreprocessor(nn.Module):
    """Prepare ordered images and action-token prompts without downloading assets."""

    def __init__(self, config: VLAJEPAConfig) -> None:
        """Reconstruct local processors and feature normalizers."""
        super().__init__()
        self.config = config
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer.from_str(config.tokenizer_json),
            eos_token="<|im_end|>",  # noqa: S106 - tokenizer vocabulary, not a credential.
            pad_token="<|endoftext|>",  # noqa: S106 - tokenizer vocabulary, not a credential.
            padding_side="left",
        )
        # Transformers' generic processor stub does not describe multimodal keyword arguments.
        self.processor: Any = Qwen3VLProcessor(
            image_processor=Qwen2VLImageProcessor(**config.image_processor_config),
            tokenizer=tokenizer,
            chat_template=config.chat_template,
            video_processor=Qwen3VLVideoProcessor(),
        )
        self.video_processor: Any = (
            VJEPA2VideoProcessor(**config.video_processor_config) if config.enable_world_model else None
        )
        latent_ids = [
            tokenizer.convert_tokens_to_ids(f"<|action_{index}|>") for index in range(config.num_latent_tokens)
        ]
        self.register_buffer("_latent_ids", torch.tensor(latent_ids, dtype=torch.long), persistent=False)
        self.normalizer = _normalizer(config)
        self.image_keys = [
            _batch_key(feature) for feature in config.input_features if feature.ftype == FeatureType.VISUAL
        ]
        self.state_keys = [
            _batch_key(feature) for feature in config.input_features if feature.ftype == FeatureType.STATE
        ]
        self.action_key = next(
            _batch_key(feature) for feature in config.output_features if feature.ftype == FeatureType.ACTION
        )
        self.latent_prompt = "".join(
            f"<|action_{index}|>" * config.num_action_tokens_per_timestep for index in range(config.num_latent_tokens)
        )
        self.embodied_prompt = "<|embodied_action|>" * config.num_embodied_action_tokens

    def _images(self, batch: dict[str, Any]) -> list[Tensor]:
        frames = []
        for key in self.image_keys:
            image = batch.get(key, batch.get("images") if len(self.image_keys) == 1 else None)
            if not isinstance(image, Tensor):
                msg = f"Missing image tensor {key!r}."
                raise TypeError(msg)
            if image.ndim == 5:
                image = image[:, 0 if self.config.enable_world_model else -1]
            # Runtime camera frames arrive channels-last (B, H, W, 3).
            if image.ndim == 4 and image.shape[1] != 3 and image.shape[-1] == 3:
                image = image.permute(0, 3, 1, 2)
            if image.ndim != 4 or image.shape[1] != 3 or image.shape[0] < 1:
                msg = f"Image {key!r} must be batched RGB, channels-first or channels-last."
                raise ValueError(msg)
            image = image.float() / 255 if image.dtype == torch.uint8 else image.float()
            if not torch.isfinite(image).all() or image.amin() < 0 or image.amax() > 1:
                msg = f"Image {key!r} must contain finite values in [0, 1] or uint8 pixels."
                raise ValueError(msg)
            frames.append(F.interpolate(image, size=self.config.image_resolution, mode="area"))
        return frames

    def _videos(self, batch: dict[str, Any]) -> tuple[Tensor, Tensor]:  # noqa: PLR0914
        videos = []
        frame_masks = []
        for key in self.image_keys[: self.config.world_model_num_views]:
            clip = batch.get(key, batch.get("images") if len(self.image_keys) == 1 else None)
            if not isinstance(clip, Tensor) or clip.ndim != 5 or clip.shape[1:3] != (self.config.num_video_frames, 3):
                msg = "JEPA-assisted training requires (batch, num_video_frames, 3, height, width) camera clips."
                raise ValueError(msg)
            clip = clip.float() / 255 if clip.dtype == torch.uint8 else clip.float()
            if not clip.isfinite().all() or clip.amin() < 0 or clip.amax() > 1:
                msg = "Video pixels must be finite and in [0, 1]."
                raise ValueError(msg)
            videos.append(clip)
            camera = key.removeprefix("images.")
            candidates = [
                f"images.{camera}_is_pad",
                f"extra.observation.images.{camera}_is_pad",
                f"extra.observation.image.{camera}_is_pad",
            ]
            if len(self.image_keys) == 1:
                candidates.append("extra.observation.image_is_pad")
            padding = next((batch[name] for name in candidates if name in batch), None)
            frame_valid = torch.ones(clip.shape[:2], dtype=torch.bool, device=clip.device)
            if padding is not None:
                if not isinstance(padding, Tensor) or padding.dtype != torch.bool or padding.shape != clip.shape[:2]:
                    msg = "Video padding masks must match the temporal clip shape."
                    raise ValueError(msg)
                frame_valid = ~padding
            frame_masks.append(frame_valid)
        valid = torch.stack(frame_masks, dim=1).all(dim=1)
        stacked = torch.stack(videos, dim=1)
        missing = self.config.world_model_num_views - stacked.shape[1]
        if missing:
            stacked = torch.cat((stacked, stacked[:, :1].expand(-1, missing, -1, -1, -1, -1)), dim=1)
        batch_size, views, frames, channels, height, width = stacked.shape
        pixels = self.video_processor(
            videos=list(stacked.reshape(batch_size * views, frames, channels, height, width)),
            return_tensors="pt",
            device=stacked.device,
            do_rescale=False,
        )["pixel_values_videos"]
        tubelet = self.config.world_model_config["tubelet_size"]
        valid = valid.view(batch_size, frames // tubelet, tubelet).all(dim=-1).cumprod(dim=1).bool()
        return pixels.reshape(batch_size, views, *pixels.shape[1:]), valid[:, 1:]

    def forward(self, batch: dict[str, Any], *, include_video: bool = False) -> dict[str, Tensor]:  # noqa: C901, PLR0912, PLR0914, PLR0915
        """Return tensor-only network inputs, with current-frame image conditioning.

        Raises:
            ValueError: If instructions or tensor shapes violate the feature contract.
        """
        normalized = self.normalizer(dict(batch))
        frames = self._images(normalized)
        batch_size = frames[0].shape[0]
        if any(frame.shape[0] != batch_size for frame in frames):
            msg = "All camera tensors must have the same batch size."
            raise ValueError(msg)
        tasks = normalized.get("task")
        if isinstance(tasks, str):
            tasks = [tasks] * batch_size
        if (
            not isinstance(tasks, (list, tuple))
            or len(tasks) != batch_size
            or not all(isinstance(task, str) for task in tasks)
        ):
            msg = "Provide one task string per observation."
            raise ValueError(msg)
        messages = []
        images = []
        for index, task in enumerate(tasks):
            views = [frame[index] for frame in frames]
            images.extend(views)
            prompt = self.config.prompt_template.format(
                instruction=task,
                actions=self.latent_prompt,
                e_actions=self.embodied_prompt,
            )
            content = [{"type": "image"} for _ in views] + [{"type": "text", "text": prompt}]
            messages.append([{"role": "user", "content": content}])
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        processed = self.processor(
            text=text,
            images=images,
            do_rescale=False,
            padding=True,
            return_tensors="pt",
            device=frames[0].device,
        ).to(frames[0].device)
        result = dict(processed)
        matches = result["input_ids"] == self.config.embodied_action_token_id
        if not torch.all(matches.sum(dim=1) == self.config.num_embodied_action_tokens):
            msg = "The task/prompt contains an invalid number of embodied-action tokens."
            raise ValueError(msg)
        result["embodied_positions"] = matches.nonzero(as_tuple=True)[1].view(batch_size, -1)
        if include_video and self.config.enable_world_model:
            latent_matches = torch.isin(result["input_ids"], self._latent_ids)
            if not torch.all(
                latent_matches.sum(dim=1) == self.config.num_latent_tokens * self.config.num_action_tokens_per_timestep,
            ):
                msg = "The task/prompt contains an invalid number of latent-action tokens."
                raise ValueError(msg)
            result["latent_positions"] = latent_matches.nonzero(as_tuple=True)[1].view(batch_size, -1)
            result["videos"], result["video_valid"] = self._videos(normalized)
        if self.state_keys:
            state = normalized.get(self.state_keys[0])
            if (
                not isinstance(state, Tensor)
                or state.ndim not in {2, 3}
                or state.shape[-1] != self.config.state_dim
                or not state.isfinite().all()
            ):
                msg = "State tensor does not match the configured feature."
                raise ValueError(msg)
            if state.ndim == 2:
                state = state[:, None]
            if state.ndim != 3 or state.shape[0] != batch_size:
                msg = "State must have shape (batch, state_dim) or (batch, history, state_dim)."
                raise ValueError(msg)
            index = 0 if self.config.enable_world_model else state.shape[1] - 1
            result["state"] = state[:, index : index + 1].float()
        actions = normalized.get(self.action_key)
        if actions is not None:
            if (
                not isinstance(actions, Tensor)
                or actions.shape != (batch_size, self.config.chunk_size, self.config.action_dim)
                or not actions.isfinite().all()
            ):
                msg = "Actions must have shape (batch, chunk_size, action_dim)."
                raise ValueError(msg)
            result["action"] = actions.float()
        if "extra.action_is_pad" in normalized:
            mask = normalized["extra.action_is_pad"]
            if (
                not isinstance(mask, Tensor)
                or mask.dtype != torch.bool
                or mask.shape != (batch_size, self.config.chunk_size)
            ):
                msg = "Action padding mask does not match the action horizon."
                raise ValueError(msg)
            result["extra.action_is_pad"] = mask.bool()
        return result


class VLAJEPAPostprocessor(nn.Module):
    """Restore dataset action units, with opt-in checkpoint gripper conventions."""

    def __init__(self, config: VLAJEPAConfig) -> None:
        """Initialize action denormalization and checkpoint gripper options."""
        super().__init__()
        self.config = config
        self.unnormalizer = _normalizer(config, inverse=True)
        self.action_key = next(
            _batch_key(feature) for feature in config.output_features if feature.ftype == FeatureType.ACTION
        )

    def forward(self, actions: Tensor) -> Tensor:
        """Return a denormalized full chunk without assuming a robot gripper layout."""
        if self.config.clip_normalized_actions and self.config.action_normalization in {"MIN_MAX", "QUANTILES"}:
            clipped = actions.clamp(-1, 1)
            mask = self.unnormalizer.buffers_lookup[self.action_key].get("mask")
            actions = clipped if mask is None else torch.where(mask, clipped, actions)
        if self.config.pre_snap_gripper:
            actions = actions.clone()
            actions[..., self.config.gripper_dim] = (
                actions[..., self.config.gripper_dim] >= self.config.gripper_threshold
            ).to(actions)
        actions = self.unnormalizer({self.action_key: actions})[self.action_key]
        if self.config.binarize_gripper:
            actions = actions.clone()
            actions[..., self.config.gripper_dim] = 1 - 2 * (
                actions[..., self.config.gripper_dim] > self.config.gripper_threshold
            ).to(actions)
        return actions
