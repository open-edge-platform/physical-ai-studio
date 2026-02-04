# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2025 Physical Intelligence
# SPDX-License-Identifier: Apache-2.0

"""Preprocessor and postprocessor for Pi0/Pi0.5 models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from getiaction.data.observation import ACTION, STATE, TASK, Observation

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)


@dataclass
class NormStats:
    """Normalization statistics for state/action preprocessing."""

    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    q01: np.ndarray | None = None
    q99: np.ndarray | None = None


class Pi0Preprocessor(torch.nn.Module):
    """Preprocess Pi0 inputs to model-ready tensors."""

    def __init__(
        self,
        *,
        max_state_dim: int = 32,
        max_action_dim: int = 32,
        chunk_size: int = 50,
        image_resolution: tuple[int, int] = (224, 224),
        use_quantile_norm: bool = False,
        stats: dict[str, NormStats] | None = None,
        tokenizer_name: str = "google/paligemma-3b-pt-224",
        max_token_len: int = 48,
    ) -> None:
        """Initialize Pi0Preprocessor with normalization and tokenization settings."""
        super().__init__()
        self.max_state_dim = max_state_dim
        self.max_action_dim = max_action_dim
        self.chunk_size = chunk_size
        self.image_resolution = image_resolution
        self.use_quantile_norm = use_quantile_norm
        self.stats = stats
        self.tokenizer_name = tokenizer_name
        self.max_token_len = max_token_len
        self._tokenizer: Any = None

    @property
    def tokenizer(self) -> Any:  # noqa: ANN401
        """Return the tokenizer, lazily initializing from HuggingFace if needed."""  # noqa: DOC501
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer  # noqa: PLC0415

                self._tokenizer = AutoTokenizer.from_pretrained(  # nosec B615
                    self.tokenizer_name,
                    revision="main",
                )
            except ImportError as e:
                msg = "Tokenizer requires transformers. Install with: pip install transformers"
                raise ImportError(msg) from e
        return self._tokenizer

    def forward(self, batch: Any) -> dict[str, Any]:  # noqa: ANN401
        """Preprocess batch into model-ready tensors."""  # noqa: DOC201
        if isinstance(batch, Observation):
            return self._from_observation(batch)
        return self._from_mapping(batch)

    def _from_observation(self, batch: Observation) -> dict[str, Any]:
        batch_dict = batch.to_dict(flatten=True)
        return self._from_mapping(batch_dict)

    def _from_mapping(self, batch: Any) -> dict[str, Any]:  # noqa: ANN401
        batch_dict = {str(key): value for key, value in batch.items()}
        result: dict[str, Any] = {}

        images, image_masks = self._process_images(batch_dict)
        result["images"] = images
        result["image_masks"] = image_masks

        state = batch_dict.get("observation.state")
        if state is None:
            state = batch_dict.get("state")
        if isinstance(state, (torch.Tensor, np.ndarray)):
            result["state"] = self._process_state(state)

        task = batch_dict.get(TASK) or batch_dict.get("task") or batch_dict.get("prompt", "")
        expected_batch = self._infer_batch_size(result.get("state"), images)
        task = self._ensure_newline(task, expected_batch=expected_batch)
        tokens, masks = self._tokenize(task)
        result["tokenized_prompt"] = tokens
        result["tokenized_prompt_mask"] = masks

        actions = batch_dict.get(ACTION)
        if actions is None:
            actions = batch_dict.get("action")
        if isinstance(actions, (torch.Tensor, np.ndarray)):
            result["actions"] = self._process_actions(actions)

        return result

    @staticmethod
    def _ensure_newline(task: Any, *, expected_batch: int | None = None) -> str | list[str]:  # noqa: ANN401, PLR0911
        if task is None:
            return "\n"
        if isinstance(task, str):
            return task if task.endswith("\n") else f"{task}\n"
        if isinstance(task, list) and all(isinstance(t, str) for t in task):
            tasks = [t if t.endswith("\n") else f"{t}\n" for t in task]
            if expected_batch is None:
                return tasks
            if len(tasks) == expected_batch:
                return tasks
            if expected_batch == 1:
                return tasks[0]
            if len(tasks) == 1:
                return tasks * expected_batch
            if len(tasks) > expected_batch:
                return tasks[:expected_batch]
            return tasks + [tasks[-1]] * (expected_batch - len(tasks))
        return "\n"

    @staticmethod
    def _infer_batch_size(state: torch.Tensor | None, images: dict[str, torch.Tensor]) -> int | None:
        if isinstance(state, torch.Tensor) and state.ndim > 0:
            return state.shape[0]
        for tensor in images.values():
            if isinstance(tensor, torch.Tensor) and tensor.ndim > 0:
                return tensor.shape[0]
        return None

    def _process_images(
        self,
        batch: Mapping[str, Any],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        images: dict[str, torch.Tensor] = {}
        image_masks: dict[str, torch.Tensor] = {}

        image_keys = [k for k in batch if "image" in k.lower() and "mask" not in k.lower() and not k.startswith("_")]

        for key in image_keys:
            img = batch[key]
            if img is None:
                continue
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img)
            if not isinstance(img, torch.Tensor):
                continue
            if img.ndim == 3:  # noqa: PLR2004
                img = img.unsqueeze(0)

            img = img.to(dtype=torch.float32)
            is_channels_first = img.shape[1] == 3  # noqa: PLR2004
            if is_channels_first:
                img = img.permute(0, 2, 3, 1)

            target_h, target_w = self.image_resolution
            if img.shape[1:3] != (target_h, target_w):
                img = self._resize_with_pad(img, target_h, target_w)

            if img.dtype == torch.uint8:
                img = img.float() / 255.0
            img = img * 2.0 - 1.0

            if is_channels_first:
                img = img.permute(0, 3, 1, 2)

            clean_name = key.replace("observation.images.", "").replace("observation.", "")
            images[clean_name] = img
            image_masks[clean_name] = torch.ones(img.shape[0], dtype=torch.bool, device=img.device)

        return images, image_masks

    @staticmethod
    def _resize_with_pad(images: torch.Tensor, height: int, width: int) -> torch.Tensor:  # noqa: PLR0914
        import torch.nn.functional as F  # noqa: N812, PLC0415

        if images.shape[-1] <= 4:  # noqa: PLR2004
            channels_last = True
            if images.dim() == 3:  # noqa: PLR2004
                images = images.unsqueeze(0)
            images = images.permute(0, 3, 1, 2)
        else:
            channels_last = False
            if images.dim() == 3:  # noqa: PLR2004
                images = images.unsqueeze(0)

        _, _, h, w = images.shape
        ratio = max(w / width, h / height)
        new_h = int(h / ratio)
        new_w = int(w / ratio)
        resized = F.interpolate(images, size=(new_h, new_w), mode="bilinear", align_corners=False)

        pad_h0, rem_h = divmod(height - new_h, 2)
        pad_h1 = pad_h0 + rem_h
        pad_w0, rem_w = divmod(width - new_w, 2)
        pad_w1 = pad_w0 + rem_w

        padded = F.pad(resized, (pad_w0, pad_w1, pad_h0, pad_h1), mode="constant", value=-1.0)

        if channels_last:
            padded = padded.permute(0, 2, 3, 1)

        return padded

    def _process_state(self, state: torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)
        state = state.float()

        if self.stats is not None and STATE in self.stats:
            state = self._normalize(state, self.stats[STATE])
        elif self.stats is not None and "state" in self.stats:
            state = self._normalize(state, self.stats["state"])

        orig_dim = state.shape[-1]
        if orig_dim < self.max_state_dim:
            pad_size = self.max_state_dim - orig_dim
            padding = torch.zeros(*state.shape[:-1], pad_size, dtype=state.dtype, device=state.device)
            state = torch.cat([state, padding], dim=-1)
        elif orig_dim > self.max_state_dim:
            state = state[..., : self.max_state_dim]

        return state

    def _process_actions(self, actions: torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)
        actions = actions.float()

        if actions.ndim == 2:  # noqa: PLR2004
            actions = actions.unsqueeze(1)

        if self.stats is not None and ACTION in self.stats:
            actions = self._normalize(actions, self.stats[ACTION])
        elif self.stats is not None and "actions" in self.stats:
            actions = self._normalize(actions, self.stats["actions"])

        orig_dim = actions.shape[-1]
        if orig_dim < self.max_action_dim:
            pad_size = self.max_action_dim - orig_dim
            padding = torch.zeros(*actions.shape[:-1], pad_size, dtype=actions.dtype, device=actions.device)
            actions = torch.cat([actions, padding], dim=-1)
        elif orig_dim > self.max_action_dim:
            actions = actions[..., : self.max_action_dim]

        if actions.shape[1] < self.chunk_size:
            pad_size = self.chunk_size - actions.shape[1]
            padding = torch.zeros(
                actions.shape[0],
                pad_size,
                actions.shape[2],
                dtype=actions.dtype,
                device=actions.device,
            )
            actions = torch.cat([actions, padding], dim=1)
        elif actions.shape[1] > self.chunk_size:
            actions = actions[:, : self.chunk_size]

        return actions

    def _normalize(self, x: torch.Tensor, stats: NormStats) -> torch.Tensor:
        if self.use_quantile_norm and stats.q01 is not None and stats.q99 is not None:
            q01 = torch.tensor(stats.q01, dtype=x.dtype, device=x.device)
            q99 = torch.tensor(stats.q99, dtype=x.dtype, device=x.device)
            dim = x.shape[-1]
            q01 = q01[..., :dim]
            q99 = q99[..., :dim]
            return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        if stats.mean is not None and stats.std is not None:
            mean = torch.tensor(stats.mean, dtype=x.dtype, device=x.device)
            std = torch.tensor(stats.std, dtype=x.dtype, device=x.device)
            dim = x.shape[-1]
            mean = mean[..., :dim]
            std = std[..., :dim]
            return (x - mean) / (std + 1e-6)
        return x

    def _tokenize(self, text: str | list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(text, str):
            text = [text]

        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_token_len,
            return_tensors="pt",
        )

        return encoded["input_ids"], encoded["attention_mask"].bool()


class Pi0Postprocessor(torch.nn.Module):
    """Postprocess Pi0 actions back to dataset space."""

    def __init__(
        self,
        *,
        action_dim: int,
        max_action_dim: int = 32,
        use_quantile_norm: bool = False,
        stats: dict[str, NormStats] | None = None,
    ) -> None:
        """Initialize Pi0Postprocessor with denormalization settings."""
        super().__init__()
        self.action_dim = action_dim
        self.max_action_dim = max_action_dim
        self.use_quantile_norm = use_quantile_norm
        self.stats = stats

    def forward(self, outputs: dict[str, Any]) -> dict[str, Any]:
        """Postprocess model outputs by denormalizing and trimming actions."""  # noqa: DOC201
        result = dict(outputs)
        if ACTION in result:
            actions = result[ACTION]
            actions = actions[..., : self.action_dim]
            if self.stats is not None and ACTION in self.stats:
                actions = self._denormalize(actions, self.stats[ACTION])
            elif self.stats is not None and "actions" in self.stats:
                actions = self._denormalize(actions, self.stats["actions"])
            result[ACTION] = actions
        return result

    def _denormalize(self, x: torch.Tensor, stats: NormStats) -> torch.Tensor:
        if self.use_quantile_norm and stats.q01 is not None and stats.q99 is not None:
            q01 = torch.tensor(stats.q01, dtype=x.dtype, device=x.device)
            q99 = torch.tensor(stats.q99, dtype=x.dtype, device=x.device)
            dim = x.shape[-1]
            q01 = q01[..., :dim]
            q99 = q99[..., :dim]
            return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
        if stats.mean is not None and stats.std is not None:
            mean = torch.tensor(stats.mean, dtype=x.dtype, device=x.device)
            std = torch.tensor(stats.std, dtype=x.dtype, device=x.device)
            dim = x.shape[-1]
            mean = mean[..., :dim]
            std = std[..., :dim]
            return x * (std + 1e-6) + mean
        return x


def make_pi0_preprocessors(
    *,
    max_state_dim: int = 32,
    max_action_dim: int = 32,
    chunk_size: int = 50,
    env_action_dim: int | None = None,
    stats: dict[str, dict[str, Any]] | None = None,
    use_quantile_norm: bool = False,
    image_resolution: tuple[int, int] = (224, 224),
    tokenizer_name: str = "google/paligemma-3b-pt-224",
    max_token_len: int = 48,
) -> tuple[Pi0Preprocessor, Pi0Postprocessor]:
    """Create paired preprocessor and postprocessor for Pi0 model."""  # noqa: DOC201
    norm_stats: dict[str, NormStats] | None = None
    if stats is not None:
        norm_stats = {}
        for key, stat_dict in stats.items():
            norm_stats[key] = NormStats(
                mean=np.array(stat_dict.get("mean")) if "mean" in stat_dict else None,
                std=np.array(stat_dict.get("std")) if "std" in stat_dict else None,
                q01=np.array(stat_dict.get("q01")) if "q01" in stat_dict else None,
                q99=np.array(stat_dict.get("q99")) if "q99" in stat_dict else None,
            )

    preprocessor = Pi0Preprocessor(
        max_state_dim=max_state_dim,
        max_action_dim=max_action_dim,
        chunk_size=chunk_size,
        image_resolution=image_resolution,
        use_quantile_norm=use_quantile_norm,
        stats=norm_stats,
        tokenizer_name=tokenizer_name,
        max_token_len=max_token_len,
    )

    postprocessor = Pi0Postprocessor(
        action_dim=env_action_dim or max_action_dim,
        max_action_dim=max_action_dim,
        use_quantile_norm=use_quantile_norm,
        stats=norm_stats,
    )

    return preprocessor, postprocessor
