# Copyright (C) 2026 Xiaomi Corporation.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Preprocessor / postprocessor for the XR0 model.

* **Prompt & vision** -- a Qwen3-VL multi-view chat prompt is assembled (one
  ``<|vision_start|><|image_pad|><|vision_end|>`` block per camera view present
  in the observation) and tokenized with the stock ``Qwen3VLProcessor`` (via
  ``AutoProcessor``) Images are resized with :func:`_resize_batch` -- a batched
  ``torch.nn.functional.interpolate`` that runs on the observation's device --
  and passed to the processor as ``uint8`` tensors with ``do_resize=False``.
* **State** -- padded into the 32-dim bimanual layout and shaped ``(B, 1, D)``,
  matching the source ``state.view(1, 1, -1)``.
* **Action** -- normalized with the source ``normalize_action`` mean/std
  convention (:func:`normalize_action`), padded to
  ``max_action_dim``, with a validity ``action_mask``.

The postprocessor inverts the action normalization
(:func:`denormalize_action`).
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F  # noqa: N812
from torchvision.transforms import functional as vision_f

from physicalai.data import Feature, FeatureType
from physicalai.data.observation import ACTION, IMAGES, STATE, TASK, Observation

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

logger = logging.getLogger(__name__)

_MAX_ASPECT_RATIO = 200
ACTION_DIM = 32
ACTION_MASK = "action_mask"
STATE_DIM = 32
ACTION_EPS = 1e-6
_TEMPORAL_STATE_NDIM = 3
_TEMPORAL_IMAGE_NDIM = 5
_BATCHED_IMAGE_NDIM = 4
_STATS_NDIM = 2

# Pinned commit SHA for the default Qwen3-VL processor download. A concrete
# revision keeps the fetched tokenizer/processor reproducible and avoids the
# supply-chain risk of resolving to a moving HEAD (see library security rule 9).
_PROCESSOR_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"


# --------------------------------------------------------------------------- #
# Qwen3-VL chat-prompt text                                                   #
# --------------------------------------------------------------------------- #
# Prompt text pieces used to assemble the structured chat message fed to the
# real ``Qwen3VLProcessor`` (see ``_build_message``).
_MULTI_VIEW_HEADER = "The following observations are captured from multiple views.\n"
_TASK_TEMPLATE = "Generate robot actions for the task:\n{instruction} /no_cot"
_ASSISTANT_PRIMER = "<cot></cot>"

# View titles the model was trained with (Xiaomi reference ``VIEW_TITLES`` in
# tools/lerobot_convert.py), e.g. "wrist_left" -> "Left-Wrist" so the prompt
# reads "# Left-Wrist View".
_VIEW_TITLES = {
    "ego": "Ego",
    "base": "Base",
    "wrist_left": "Left-Wrist",
    "wrist_right": "Right-Wrist",
}

# Canonical prompt order of the XR0 view names.
_VIEW_ORDER = ("ego", "base", "wrist_left", "wrist_right")


def _validate_view_map(image_key_view_map: Mapping[str, str] | None) -> dict[str, str]:
    """Validate a mapping from exact batch image keys to canonical views.

    Returns:
        The validated mapping, keyed by batch image key.

    Raises:
        ValueError: If a value is not a canonical XR0 view name, or if two keys
            map onto the same view.
    """
    mapping = dict(image_key_view_map or {})

    unknown = sorted({view for view in mapping.values() if view not in _VIEW_TITLES})
    if unknown:
        msg = f"image_key_view_map values must be canonical XR0 view names {list(_VIEW_ORDER)}, got {unknown}."
        raise ValueError(msg)

    duplicates = sorted(view for view, count in Counter(mapping.values()).items() if count > 1)
    if duplicates:
        msg = f"image_key_view_map values must be unique, got duplicates {duplicates}."
        raise ValueError(msg)

    return mapping


def _view_title(view: str) -> str:
    """Human-readable view title matching the reference prompt.

    Known views use the reference eval's exact titles (e.g. ``"wrist_left"`` ->
    ``"Left-Wrist"``); unknown views fall back to a capitalized join.

    Returns:
        The human-readable view title.
    """
    key = view.replace("-", "_")
    if key in _VIEW_TITLES:
        return _VIEW_TITLES[key]
    return " ".join(word.capitalize() for word in key.split("_"))


def _to_chw_float(images: torch.Tensor) -> torch.Tensor:
    """Normalize a batched image tensor to ``(B, 3, H, W)`` float in ``[0, 255]``.

    Channels-last input is permuted, single-channel input is expanded to RGB and
    floating-point input is assumed to be in ``[0, 1]``. The device is preserved
    so the subsequent resize stays where the observation already lives.

    Returns:
        A ``(B, 3, H, W)`` float tensor with values in ``[0, 255]``.

    Raises:
        ValueError: If ``images`` is not a 4D batched image tensor.
    """
    if images.ndim != _BATCHED_IMAGE_NDIM:
        msg = f"expected a batched (B, C, H, W) or (B, H, W, C) image tensor, got shape {tuple(images.shape)}"
        raise ValueError(msg)
    images = images.detach()
    if images.shape[1] not in {1, 3} and images.shape[-1] in {1, 3}:  # channels-last
        images = images.permute(0, 3, 1, 2)
    images = images.float() if images.dtype == torch.uint8 else images.float().clamp(0.0, 1.0) * 255.0
    if images.shape[1] == 1:
        images = images.expand(-1, 3, -1, -1)
    return images


def _target_size(height: int, width: int, factor: int, max_pixels: int) -> tuple[int, int]:
    """Compute the patch-aligned ``(height, width)`` within the area budget.

    Both sides are rounded up to a multiple of ``factor`` (so the area is never
    below ``factor ** 2``, i.e. one merged vision patch) and the total area is
    capped at ``max_pixels``, preserving aspect ratio for the VLM vision encoder.

    Returns:
        The target ``(height, width)``.

    Raises:
        ValueError: If the image aspect ratio exceeds ``_MAX_ASPECT_RATIO``.
    """
    ratio = max(height, width) / min(height, width)
    if ratio > _MAX_ASPECT_RATIO:
        msg = f"absolute aspect ratio must be smaller than 200, got {ratio}"
        raise ValueError(msg)

    new_height = max(factor, round(height / factor) * factor)
    new_width = max(factor, round(width / factor) * factor)

    if new_height * new_width > max_pixels:
        scale = math.sqrt(height * width / max_pixels)
        new_height = max(factor, math.floor(height / scale / factor) * factor)
        new_width = max(factor, math.floor(width / scale / factor) * factor)

    return new_height, new_width


def _resize_batch(images: torch.Tensor, factor: int, max_pixels: int) -> torch.Tensor:
    """Resize a batch of images to patch-aligned dimensions within an area budget.

    The whole ``(B, C, H, W)`` view is resized in a single
    ``torch.nn.functional.interpolate`` call on the input device, so large
    training batches never round-trip through per-image CPU resizing. ``bicubic``
    + ``antialias`` matches PIL's default ``Image.resize`` filter.

    Returns:
        A ``(B, 3, new_height, new_width)`` ``uint8`` tensor on the input device.
    """
    images = _to_chw_float(images)
    height, width = int(images.shape[-2]), int(images.shape[-1])
    new_height, new_width = _target_size(height, width, factor, max_pixels)
    if (new_height, new_width) != (height, width):
        images = F.interpolate(
            images,
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
    return images.clamp(0.0, 255.0).round().to(torch.uint8)


def _augment_images(images: list[torch.Tensor]) -> list[torch.Tensor]:
    """Apply Xiaomi's training crop and shared per-sample color jitter to resized views.

    Returns:
        Augmented RGB uint8 views at their original resized dimensions.
    """
    draws = torch.rand(6, device=images[0].device).tolist()
    ops = (
        (vision_f.adjust_brightness, 1.0 + (2 * draws[0] - 1) * 32.0 / 255.0),
        (vision_f.adjust_contrast, 0.5 + draws[1]),
        (vision_f.adjust_saturation, 0.5 + draws[2]),
    )
    augmented = []
    jitter_probability = 0.5
    for source in images:
        height, width = source.shape[-2:]
        crop_h, crop_w = int(height * 0.95), int(width * 0.95)
        top = int(torch.randint(height - crop_h + 1, (), device=source.device).item())
        left = int(torch.randint(width - crop_w + 1, (), device=source.device).item())
        image = vision_f.resize(vision_f.crop(source, top, left, crop_h, crop_w), [height, width])
        for draw, (op, factor) in zip(draws[3:], ops, strict=True):
            if draw < jitter_probability:
                image = op(image, factor)
        augmented.append(image)
    return augmented


def _normalize_action(action: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Standardize ``action`` to zero-mean/unit-scale using ``(action - mean) / (std + eps)``.

    Returns:
        The normalized action array.
    """
    return (action - mean) / (std + ACTION_EPS)


def _denormalize_action(action: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Invert :func:`normalize_action`, mapping a normalized action back to raw units.

    Returns:
        The denormalized action array.
    """
    return action * (std + ACTION_EPS) + mean


def _to_chunk_stats(
    mean: Sequence[float] | torch.Tensor,
    std: Sequence[float] | torch.Tensor,
    *,
    chunk_size: int,
    dim: int,
    name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert ``mean`` / ``std`` into per-timestep ``(chunk_size, dim)`` buffers.

    Stats need one row per timestep; only a single-timestep chunk accepts a
    1D array. Pad missing dimensions with mean 0 and std 1 so they stay
    unchanged, or truncate extra dimensions. Already full-width stats are copied.

    Returns:
        A ``(mean, std)`` tuple of ``(chunk_size, dim)`` float32 tensors.

    Raises:
        ValueError: If ``mean`` / ``std`` disagree in shape, or if the statistics
            are not per-timestep with the expected temporal chunk_size.
    """
    t_mean = torch.as_tensor(mean, dtype=torch.float32)
    t_std = torch.as_tensor(std, dtype=torch.float32)
    if t_mean.shape != t_std.shape:
        msg = f"{name} mean/std must have the same shape, got {tuple(t_mean.shape)} and {tuple(t_std.shape)}"
        raise ValueError(msg)
    if t_mean.ndim == 1 and chunk_size == 1:
        t_mean = t_mean.unsqueeze(0)
        t_std = t_std.unsqueeze(0)
    if t_mean.ndim != _STATS_NDIM or int(t_mean.shape[0]) != chunk_size:
        msg = (
            f"{name} stats must be per-timestep with shape ({chunk_size}, D), got {tuple(t_mean.shape)}. "
            "Per-dimension stats are not broadcast over time; compute per-timestep stats with "
            "physicalai.policies.xr0.compute_action_chunk_stats."
        )
        raise ValueError(msg)

    out_mean = torch.zeros(chunk_size, dim)
    out_std = torch.ones(chunk_size, dim)
    width = min(dim, int(t_mean.shape[-1]))
    out_mean[:, :width] = t_mean[:, :width]
    out_std[:, :width] = t_std[:, :width]
    return out_mean, out_std


class XR0Preprocessor(torch.nn.Module):
    """Transform framework observations into the XR0 model batch.

    Produces the keys consumed by ``XR0Model`` (``input_ids``,
    ``attention_mask``, ``pixel_values``, ``image_grid_thw``, ``state`` and --
    during training -- ``action`` / ``action_mask``).

    Args:
        max_state_dim: State dimension after padding.
        max_action_dim: Action dimension after padding.
        features: Optional feature map (from dataset stats) used to normalize the
            action with the source mean/std convention. When ``None`` the action
            is passed through unnormalized.
        image_factor: Patch-alignment factor for :func:`_resize_batch`.
        image_max_pixels: Maximum image area for :func:`_resize_batch`.
        processor_name: HuggingFace id of the Qwen3-VL processor.
        max_token_len: Fixed prompt length the OpenVINO tokenizer pads to at export
            (matches the graph's baked ``tokenizer_max_length``).
        image_key_view_map: Optional mapping from dataset image key to canonical
            XR0 view name (one of ``"ego"``, ``"base"``, ``"wrist_left"``,
            ``"wrist_right"``). Keys must match the flattened batch image keys
            (e.g. ``"images.top_camera"``). When set, the mapping must cover all
            batch image keys and the prompt sections use canonical view order.
            When empty, the view name is the image key without ``"images."``
            and the prompt follows the dataset's key order.
        chunk_size: Number of predicted action timesteps; the temporal length of
            the action normalization buffers.
        state_len: Number of state timesteps; the temporal length of the state
            normalization buffers.
        normalize_state: When True, normalize the state with per-timestep
            mean/std (from ``features`` or explicit ``state_mean`` / ``state_std``).
            Defaults to False (raw state, matching the upstream recipe).
        state_mean: Optional explicit ``(state_len, max_state_dim)`` state mean
            overriding the feature-derived value (used to reload the exported
            normalization).
        state_std: Optional explicit ``(state_len, max_state_dim)`` state std
            overriding the feature-derived value (used to reload the exported
            normalization).
        action_mode: ``"absolute"`` normalizes raw actions; ``"delta"``
            normalizes actions relative to the current state.
        action_mean: Optional explicit ``(chunk_size, max_action_dim)`` action
            mean overriding the feature-derived value.
        action_std: Optional explicit ``(chunk_size, max_action_dim)`` action std
            overriding the feature-derived value.
    """

    action_mean: torch.Tensor
    action_std: torch.Tensor
    state_mean: torch.Tensor
    state_std: torch.Tensor

    def __init__(  # noqa: PLR0913
        self,
        max_state_dim: int = 32,
        max_action_dim: int = 32,
        features: dict[str, Feature] | None = None,
        image_factor: int = 32,
        image_max_pixels: int = 90000,
        processor_name: str = "Qwen/Qwen3-VL-4B-Instruct",
        max_token_len: int = 256,
        *,
        image_key_view_map: Mapping[str, str] | None = None,
        chunk_size: int = 30,
        state_len: int = 1,
        normalize_state: bool = False,
        state_mean: Sequence[float] | torch.Tensor | None = None,
        state_std: Sequence[float] | torch.Tensor | None = None,
        action_mode: str = "absolute",
        action_mean: Sequence[float] | torch.Tensor | None = None,
        action_std: Sequence[float] | torch.Tensor | None = None,
    ) -> None:
        """Initialize the XR0 preprocessor."""
        super().__init__()
        self.max_state_dim = max_state_dim
        self.max_action_dim = max_action_dim
        self.chunk_size = int(chunk_size)
        self.state_len = int(state_len)
        self.image_factor = image_factor
        self.image_max_pixels = image_max_pixels
        self.processor_name = processor_name
        self.max_token_len = int(max_token_len)
        self.image_key_view_map = _validate_view_map(image_key_view_map)
        self.normalize_state = bool(normalize_state)
        self.action_mode = str(action_mode)
        self._processor: Any = None

        # Explicit ``action_mean`` / ``action_std`` take precedence over the
        # feature-derived stats. Both paths yield ``(chunk_size,
        # max_action_dim)`` buffers.
        if action_mean is not None and action_std is not None:
            raw_mean, raw_std = action_mean, action_std
        else:
            raw_mean, raw_std = self._action_stats(features)
        mean, std = _to_chunk_stats(
            raw_mean,
            raw_std,
            chunk_size=self.chunk_size,
            dim=self.max_action_dim,
            name=ACTION,
        )
        self.register_buffer("action_mean", mean, persistent=False)
        self.register_buffer("action_std", std, persistent=False)

        # Explicit ``state_mean`` / ``state_std``
        # take precedence over feature-derived stats so the
        # exported graph reproduces the training normalization exactly.
        if state_mean is not None and state_std is not None:
            raw_mean, raw_std = state_mean, state_std
        else:
            raw_mean, raw_std = self._state_stats(features)
        s_mean, s_std = _to_chunk_stats(
            raw_mean,
            raw_std,
            chunk_size=self.state_len,
            dim=self.max_state_dim,
            name=STATE,
        )
        self.register_buffer("state_mean", s_mean, persistent=False)
        self.register_buffer("state_std", s_std, persistent=False)

    def _action_stats(self, features: dict[str, Feature] | None) -> tuple[torch.Tensor, torch.Tensor]:
        """Get action mean/std from features, or identity stats when unavailable.

        Returns:
            The raw feature stats or ``(chunk_size, max_action_dim)`` identity stats.
        """
        for feature in (features or {}).values():
            if feature.ftype != FeatureType.ACTION or feature.normalization_data is None:
                continue
            norm = feature.normalization_data
            if norm.mean is None or norm.std is None:
                continue
            return torch.as_tensor(norm.mean), torch.as_tensor(norm.std)
        return torch.zeros(self.chunk_size, self.max_action_dim), torch.ones(self.chunk_size, self.max_action_dim)

    def _state_stats(self, features: dict[str, Feature] | None) -> tuple[torch.Tensor, torch.Tensor]:
        """Get state mean/std from features, or identity stats when unavailable.

        Returns identity stats (mean 0, std 1) when state normalization is
        disabled or no state stats are available, so raw-state checkpoints
        stay bit-for-bit unchanged.

        Returns:
            The raw feature stats or ``(state_len, max_state_dim)`` identity stats.
        """
        if self.normalize_state:
            for feature in (features or {}).values():
                if feature.ftype != FeatureType.STATE or feature.normalization_data is None:
                    continue
                norm = feature.normalization_data
                if norm.mean is None or norm.std is None:
                    continue
                return torch.as_tensor(norm.mean), torch.as_tensor(norm.std)
        return torch.zeros(self.state_len, self.max_state_dim), torch.ones(self.state_len, self.max_state_dim)

    @property
    def processor(self) -> Any:  # noqa: ANN401
        """Lazy-load the Qwen3-VL processor.

        Raises:
            ImportError: If transformers is not installed.
        """
        if self._processor is None:
            try:
                from transformers import AutoProcessor  # noqa: PLC0415
            except ImportError as exc:
                msg = "XR0 preprocessing requires transformers. Install with: uv pip install transformers"
                raise ImportError(msg) from exc
            self._processor = AutoProcessor.from_pretrained(self.processor_name, revision=_PROCESSOR_REVISION)
            self._processor.tokenizer.padding_side = "right"
        return self._processor

    @property
    def tokenizer(self) -> Any:  # noqa: ANN401
        """The Qwen3-VL tokenizer (used for the OpenVINO tokenizer export).

        Returns:
            The processor's underlying HuggingFace tokenizer.
        """
        return self.processor.tokenizer

    @staticmethod
    def _build_message(
        instruction: str,
        views: list[str],
        images: Sequence[torch.Tensor],
    ) -> list[dict[str, Any]]:
        """Assemble the Qwen3-VL multi-view chat message for one sample.

        Returns:
            The Qwen3-VL chat message (user + assistant primer) for one sample.
        """
        content: list[dict[str, Any]] = [{"type": "text", "text": _MULTI_VIEW_HEADER}]
        for view, image in zip(views, images, strict=False):
            content.extend((
                {"type": "text", "text": f"# {_view_title(view)} View\n"},
                {"type": "image", "image": image},
                {"type": "text", "text": "\n"},
            ))
        content.append({"type": "text", "text": _TASK_TEMPLATE.format(instruction=instruction)})
        return [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": _ASSISTANT_PRIMER}]},
        ]

    def _resolve_views(self, image_keys: list[str]) -> tuple[list[str], list[str]]:
        """Map the batch image keys onto canonical view names in canonical order.

        Returns:
            An ``(image_keys, views)`` tuple, both ordered by :data:`_VIEW_ORDER`
            when ``image_key_view_map`` is set and left in the batch's own key
            order otherwise.

        Raises:
            ValueError: If ``image_key_view_map`` is set and its keys do not match
                the batch image keys exactly.
        """
        if not self.image_key_view_map:
            return image_keys, [key.removeprefix(f"{IMAGES}.") for key in image_keys]

        if set(self.image_key_view_map) != set(image_keys):
            msg = (
                "image_key_view_map keys must match the batch image keys exactly. "
                f"Expected {sorted(self.image_key_view_map)}, got {sorted(image_keys)}."
            )
            raise ValueError(msg)

        ordered_keys = sorted(image_keys, key=lambda key: _VIEW_ORDER.index(self.image_key_view_map[key]))
        return ordered_keys, [self.image_key_view_map[key] for key in ordered_keys]

    def _extract_view_images(
        self,
        batch: dict[str, Any],
        *,
        augment_images: bool = False,
    ) -> tuple[list[str], list[list[torch.Tensor]]]:
        """Return the ordered view names and, per sample, the resized images.

        Returns:
            A ``(views, images)`` tuple: the ordered view names and, per sample,
            the list of resized ``(3, H, W)`` ``uint8`` tensors (one per view).

        Raises:
            ValueError: If the batch contains no image observation.
        """
        image_keys = [key for key in Observation.get_flattened_keys(batch, IMAGES) if "is_pad" not in key]
        if not image_keys:
            msg = "XR0Preprocessor requires at least one image observation"
            raise ValueError(msg)
        image_keys, views = self._resolve_views(image_keys)

        per_view: list[torch.Tensor] = []
        for key in image_keys:
            tensor = batch[key]
            if tensor.ndim == _TEMPORAL_IMAGE_NDIM:  # (B, T, C, H, W) -> last frame
                tensor = tensor[:, -1]
            # One batched resize per view, on the observation's device. The tensors
            # stay there: the Qwen-VL image processor is torchvision-backed, so its
            # rescale / normalize / patchify run on the same device.
            per_view.append(
                _resize_batch(
                    tensor,
                    factor=self.image_factor,
                    max_pixels=self.image_max_pixels,
                ),
            )

        batch_size = per_view[0].shape[0]
        images = [[view[sample] for view in per_view] for sample in range(batch_size)]
        if augment_images:
            images = [_augment_images(sample) for sample in images]
        return views, images

    def _prepare_state(self, batch: dict[str, Any], device: torch.device) -> torch.Tensor:
        """Pad the state into ``(B, 1, max_state_dim)`` (source ``state.view(1, 1, -1)``).

        Returns:
            The padded state tensor of shape ``(B, 1, max_state_dim)``.
        """
        state = batch[STATE]
        if state.ndim == _TEMPORAL_STATE_NDIM:  # (B, T, D) -> last frame
            state = state[:, -1, :]
        state = state.to(torch.float32)
        state = F.pad(state, (0, max(0, self.max_state_dim - state.shape[-1])))[:, : self.max_state_dim]
        state = state.unsqueeze(1)  # (B, 1, max_state_dim)
        if self.normalize_state:
            # ``(B, 1, D)`` against the per-timestep ``(state_len, D)`` buffers.
            mean = self.state_mean.to(state.device)
            std = self.state_std.to(state.device)
            state = _normalize_action(state, mean, std)
        return state.to(device)

    def _prepare_action(
        self,
        action: torch.Tensor,
        device: torch.device,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize (source convention) + pad the action, and build its validity mask.

        When ``action_mode == "delta"`` the raw current-frame ``state`` is
        subtracted from the action target (on the overlapping leading channels)
        before padding/normalization, so the flow head predicts
        ``action[t] - state`` against the per-timestep delta stats.

        Returns:
            A ``(action, mask)`` tuple of padded action and its validity mask.

        Raises:
            ValueError: If ``action_mode == 'delta'`` but no ``state`` is provided,
                or if the action chunk length does not match ``chunk_size``.
        """
        action = action.to(torch.float32).clone()  # Clone tensor to avoid mutating the input action
        if action.shape[-2] != self.chunk_size:
            msg = (
                f"expected an action chunk of {self.chunk_size} timesteps to match the per-timestep "
                f"normalization stats, got {action.shape[-2]}"
            )
            raise ValueError(msg)
        if self.action_mode == "delta":
            if state is None:
                msg = "action_mode='delta' requires the current state to form the delta target."
                raise ValueError(msg)
            raw_state = state
            if raw_state.ndim == _TEMPORAL_STATE_NDIM:  # (B, T, D) -> current (last) frame
                raw_state = raw_state[:, -1, :]
            raw_state = raw_state.to(torch.float32)
            overlap = min(action.shape[-1], raw_state.shape[-1])
            current = raw_state[..., :overlap].unsqueeze(1)  # (B, 1, overlap)
            if overlap < action.shape[-1]:
                head = action[..., :overlap] - current
                action = torch.cat([head, action[..., overlap:]], dim=-1)
            else:
                action -= current
        real_dim = min(action.shape[-1], self.max_action_dim)
        action = F.pad(action, (0, max(0, self.max_action_dim - action.shape[-1])))[..., : self.max_action_dim]
        action = _normalize_action(action, self.action_mean, self.action_std)

        mask = torch.zeros_like(action, dtype=torch.int32)
        mask[..., :real_dim] = 1
        return action.to(device), mask.to(device)

    def forward(self, batch: dict[str, Any], *, augment_images: bool = False) -> dict[str, torch.Tensor]:
        """Process a batch into the XR0 model input.

        Args:
            batch: Dict with STATE, TASK, image keys and optionally ACTION.
            augment_images: Apply training crop and color jitter to images.

        Returns:
            Dict with ``input_ids`` / ``attention_mask`` / ``pixel_values`` /
            ``image_grid_thw`` / ``state`` and optionally ``action`` /
            ``action_mask``.
        """
        batch = dict(batch)
        device = batch[STATE].device

        views, images = self._extract_view_images(batch, augment_images=augment_images)
        batch_size = len(images)

        task = batch.get(TASK)
        if task is None:
            task = [""] * batch_size
        elif isinstance(task, str):
            task = [task]

        messages = [self._build_message(str(task[i]).strip(), views, images[i]) for i in range(batch_size)]
        encoded = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True, "images_kwargs": {"do_resize": False}},
        )

        out: dict[str, torch.Tensor] = {
            "input_ids": encoded["input_ids"].to(device),
            "attention_mask": encoded["attention_mask"].to(device),
            "pixel_values": encoded["pixel_values"].to(device),
            "image_grid_thw": encoded["image_grid_thw"].to(device),
            "state": self._prepare_state(batch, device),
        }

        if ACTION in batch and batch[ACTION] is not None:
            action, action_mask = self._prepare_action(batch[ACTION], device, state=batch[STATE])
            out[ACTION] = action
            out[ACTION_MASK] = action_mask

        return out


class XR0Postprocessor(torch.nn.Module):
    """Invert the XR0 action normalization.

    Denormalizes predicted actions with the source
    :func:`denormalize_action` convention and slices
    back to the original (unpadded) action dimension when known.

    Args:
        max_action_dim: Padded action dimension used by the preprocessor.
        features: Optional feature map used to recover the action mean/std and
            the original action dimension.
        chunk_size: Number of predicted action timesteps; the temporal length of
            the action normalization buffers.
        action_mode: ``"absolute"`` or ``"delta"``.
        action_mean: Optional explicit ``(chunk_size, max_action_dim)`` action
            mean overriding the feature-derived value.
        action_std: Optional explicit ``(chunk_size, max_action_dim)`` action std
            overriding the feature-derived value.
    """

    action_mean: torch.Tensor
    action_std: torch.Tensor

    def __init__(
        self,
        max_action_dim: int = 32,
        features: dict[str, Feature] | None = None,
        *,
        chunk_size: int = 30,
        action_mode: str = "absolute",
        action_mean: Sequence[float] | torch.Tensor | None = None,
        action_std: Sequence[float] | torch.Tensor | None = None,
    ) -> None:
        """Initialize the XR0 postprocessor."""
        super().__init__()
        self.max_action_dim = max_action_dim
        self.chunk_size = int(chunk_size)
        self.action_dim: int | None = None
        self.action_mode = str(action_mode)

        raw_mean: Sequence[float] | torch.Tensor = torch.zeros(self.chunk_size, max_action_dim)
        raw_std: Sequence[float] | torch.Tensor = torch.ones(self.chunk_size, max_action_dim)
        for feature in (features or {}).values():
            if feature.ftype != FeatureType.ACTION or feature.normalization_data is None:
                continue
            norm = feature.normalization_data
            if norm.mean is None or norm.std is None:
                continue
            if action_mean is None or action_std is None:
                raw_mean, raw_std = torch.as_tensor(norm.mean), torch.as_tensor(norm.std)
            # Prefer the feature's declared width: explicit stats may be padded
            # to ``max_action_dim``. Fall back to the per-timestep stats' last
            # axis ``(chunk_size, D)`` when the shape is unknown.
            feat_dim = int(feature.shape[-1]) if feature.shape else int(torch.as_tensor(norm.mean).shape[-1])
            self.action_dim = min(max_action_dim, feat_dim)
            break

        # Explicit per-timestep stats (e.g. delta stats computed over the
        # training set) override the feature-derived denormalization mean/std;
        # the unpadded ``action_dim`` is still recovered from ``features``.
        if action_mean is not None and action_std is not None:
            raw_mean, raw_std = action_mean, action_std
        mean, std = _to_chunk_stats(
            raw_mean,
            raw_std,
            chunk_size=self.chunk_size,
            dim=max_action_dim,
            name=ACTION,
        )

        self.register_buffer("action_mean", mean, persistent=False)
        self.register_buffer("action_std", std, persistent=False)

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Denormalize and unpad the predicted actions.

        In ``action_mode="delta"`` the denormalized prediction is a delta and the
        current-frame ``state`` is re-added (on the overlapping leading channels)
        to recover the absolute action.

        Returns:
            Batch dict with the denormalized action.

        Raises:
            ValueError: If ``action_mode == 'delta'`` but no ``state`` is provided,
                or if the action chunk length does not match ``chunk_size``.
        """
        batch = dict(batch)
        if ACTION in batch and batch[ACTION] is not None:
            action = batch[ACTION].to(torch.float32)
            if action.shape[-2] != self.chunk_size:
                msg = (
                    f"expected an action chunk of {self.chunk_size} timesteps to match the per-timestep "
                    f"normalization stats, got {action.shape[-2]}"
                )
                raise ValueError(msg)
            mean = self.action_mean.to(action.device)
            std = self.action_std.to(action.device)
            action = _denormalize_action(action, mean, std)
            if self.action_mode == "delta":
                state = batch.get(STATE)
                if state is None:
                    msg = "action_mode='delta' requires the current state to invert the delta prediction."
                    raise ValueError(
                        msg,
                    )
                current = state.to(torch.float32).to(action.device)
                if current.ndim == _TEMPORAL_STATE_NDIM:  # (B, T, D) -> current (last) frame
                    current = current[:, -1, :]
                overlap = min(action.shape[-1], current.shape[-1])
                current = current[..., :overlap].unsqueeze(1)  # (B, 1, overlap)
                if overlap < action.shape[-1]:
                    head = action[..., :overlap] + current
                    action = torch.cat([head, action[..., overlap:]], dim=-1)
                else:
                    action += current
            if self.action_dim is not None:
                action = action[..., : self.action_dim]
            batch[ACTION] = action
        return batch


def make_xr0_preprocessors(
    max_state_dim: int = 32,
    max_action_dim: int = 32,
    stats: dict[str, dict[str, Any]] | None = None,
    *,
    chunk_size: int = 30,
    state_len: int = 1,
    image_factor: int = 32,
    image_max_pixels: int = 90000,
    processor_name: str = "Qwen/Qwen3-VL-4B-Instruct",
    image_key_view_map: Mapping[str, str] | None = None,
    normalize_state: bool = False,
    action_mode: str = "absolute",
    action_mean: Sequence[float] | torch.Tensor | None = None,
    action_std: Sequence[float] | torch.Tensor | None = None,
) -> tuple[XR0Preprocessor, XR0Postprocessor]:
    """Create the XR0 preprocessor / postprocessor pair from dataset stats.

    Args:
        max_state_dim: Padded state dimension.
        max_action_dim: Padded action dimension.
        stats: Dataset statistics as nested dicts (LeRobot format). The action
            entry must carry per-timestep ``(chunk_size, D)`` mean/std;
            :meth:`XR0.setup` collects them during training.
        chunk_size: Number of predicted action timesteps.
        state_len: Number of state timesteps.
        image_factor: Patch-alignment factor for image resizing.
        image_max_pixels: Maximum image area for image resizing.
        processor_name: HuggingFace id of the Qwen3-VL processor.
        image_key_view_map: Optional mapping from dataset image key to canonical
            XR0 view name, used to rename and order the prompt's view sections.
            Keys must match the flattened batch keys (e.g. ``"images.top_camera"``).
        normalize_state: When True, normalize the state with the dataset's
            mean/std. Defaults to False (raw state).
        action_mode: ``"absolute"`` (default) or ``"delta"``. In delta mode the
            action target is ``action[t] - state`` and the postprocessor re-adds
            the state.
        action_mean: Per-timestep action mean (``(chunk_size,
            max_action_dim)``) overriding the stats-derived value. Applies to
            both action modes.
        action_std: Per-timestep action std, same shape as ``action_mean``.

    Returns:
        Tuple of (preprocessor, postprocessor).
    """
    from physicalai.data import NormalizationParameters  # noqa: PLC0415

    features: dict[str, Feature] = {}
    if stats is not None:
        for key, stat in stats.items():
            if ACTION in key:
                feature_type = FeatureType.ACTION
            elif STATE in key:
                feature_type = FeatureType.STATE
            else:
                continue
            raw_name = str(stat.get("name", key))
            mapped_name = raw_name.rsplit("observation.", maxsplit=1)[-1] if "observation." in raw_name else raw_name
            features[mapped_name] = Feature(
                name=mapped_name,
                ftype=feature_type,
                shape=tuple(stat["shape"]),
                normalization_data=NormalizationParameters(
                    mean=stat.get("mean"),
                    std=stat.get("std"),
                    q01=stat.get("q01"),
                    q99=stat.get("q99"),
                ),
            )

    override_mean: torch.Tensor | None = None
    override_std: torch.Tensor | None = None
    if action_mean is not None and action_std is not None:
        override_mean = torch.as_tensor(action_mean, dtype=torch.float32)
        override_std = torch.as_tensor(action_std, dtype=torch.float32)

    preprocessor = XR0Preprocessor(
        max_state_dim=max_state_dim,
        max_action_dim=max_action_dim,
        features=features,
        image_factor=image_factor,
        image_max_pixels=image_max_pixels,
        processor_name=processor_name,
        image_key_view_map=image_key_view_map,
        chunk_size=chunk_size,
        state_len=state_len,
        normalize_state=normalize_state,
        action_mode=action_mode,
        action_mean=override_mean,
        action_std=override_std,
    )
    postprocessor = XR0Postprocessor(
        max_action_dim=max_action_dim,
        features=features,
        chunk_size=chunk_size,
        action_mode=action_mode,
        action_mean=override_mean,
        action_std=override_std,
    )
    return preprocessor, postprocessor
