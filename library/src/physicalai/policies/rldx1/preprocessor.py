# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Vendored from RLWRLD/RLDX-1 (Apache-2.0)

"""Transforms for the RLDX-1 policy inputs and outputs.

This module provides ``nn.Module``-based transforms that bridge the Studio
:class:`~physicalai.data.observation.Observation` format and the vendored RLDX
network (``components/core_rldx.py``):

- :class:`Rldx1Preprocessor`: ``Observation`` -> RLDX ``inputs`` dict
- :class:`Rldx1Postprocessor`: RLDX ``action_pred`` -> environment action space

Both run the PAS-native pipeline in ``preprocessing.py`` -- batched torch,
library normalization, torchvision image geometry, HF Qwen tokenization. The
native pipeline is parity-tested against the original vendored processor; see
``tests/unit/policies/test_rldx1_preprocessing.py`` for the numerical-parity
proofs. The preprocessor produces the exact keys consumed by
:meth:`RLDX.forward` / :meth:`RLDX.get_action`:

Backbone (Qwen3-VL) keys:
    ``input_ids``, ``attention_mask``, ``pixel_values``, ``image_grid_thw``,
    ``image_wise_encoding``, ``num_views``, ``num_frames``.

Action-head keys:
    ``state`` ``(B, 1, max_state_dim)``, ``action``
    ``(B, action_horizon, max_action_dim)``, ``embodiment_id`` ``(B,)``.

VTC video: each camera view can carry ``num_frames`` temporal frames per step
(the ``delta_timestamps`` video window ``[-6, -4, -2, 0]``). Frames are ordered
frame-major / view-inner into the Qwen conversation, matching upstream
``_get_vlm_inputs``; ``num_frames`` is reported per sample. The same
deterministic geometry (:mod:`augmentations`) is used at train and eval time;
there is no train-time image augmentation.

Scope (v1): absolute actions, no relative actions / motion / memory / physics
streams.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from PIL import Image
from torch import nn
from transformers import AutoProcessor

from physicalai.data.observation import ACTION, IMAGES, STATE, TASK, Observation
from physicalai.policies.utils.normalization import FeatureNormalizeTransform

from .augmentations import AspectAreaResizeAndCrop
from .preprocessing import (
    build_qwen_conversation,
    build_state_action_degenerate_masks,
    build_state_action_features,
    build_state_action_norm_map,
    clip_state_action,
    formalize_language,
    pad_state_action,
    tokenize_vlm_batch,
    zero_degenerate_state_action,
)

if TYPE_CHECKING:
    from transformers import ProcessorMixin

    from physicalai.data import Feature

logger = logging.getLogger(__name__)


def _module_device(module: nn.Module) -> torch.device:
    """Return a module's device from its parameters, then buffers, else CPU.

    ``next(iter(module.parameters()))`` raises ``StopIteration`` on a
    parameter-less module (e.g. ``nn.Identity`` or a buffer-only normalizer).
    Inside ``training_step`` that exception is silently caught by Lightning's
    data-fetch loop and misread as end-of-epoch, skipping training entirely.
    """
    for param in module.parameters():
        return param.device
    for buf in module.buffers():
        return buf.device
    return torch.device("cpu")


# ============================================================================ #
# Constants                                                                    #
# ============================================================================ #

# LeRobot-format keys (accepted alongside the native Observation keys).
OBSERVATION_STATE = "observation.state"
OBSERVATION_IMAGES_PREFIX = "observation.images."
OBSERVATION_IMAGE = "observation.image"

# Transform output keys (RLDX inputs dict).
INPUT_IDS = "input_ids"
ATTENTION_MASK = "attention_mask"
PIXEL_VALUES = "pixel_values"
IMAGE_GRID_THW = "image_grid_thw"
MM_TOKEN_TYPE_IDS = "mm_token_type_ids"  # noqa: S105 # nosec B105 -- dict key name, not a secret
IMAGE_WISE_ENCODING = "image_wise_encoding"
NUM_VIEWS = "num_views"
NUM_FRAMES = "num_frames"
ACTION_MASK = "action_mask"

# Default padded dimensions (match RLDX PT config.json).
MAX_STATE_DIM = 64
MAX_ACTION_DIM = 64
ACTION_HORIZON = 16

# Default embodiment projector slot. 0 = general_embodiment, the slot RLDX-1
# reserves and pre-conditions for downstream / new-robot fine-tunes (every
# released FT used it). Released benchmark checkpoints use the slot they were
# trained on (see EMBODIMENT_TAG_TO_PROJECTOR_INDEX in components.embodiments):
# 0 = general_embodiment (FT-ROBOCASA/RC365/LIBERO/GR1),
# 1 = fractal20220817_data (FT-SIMPLER-GOOGLE), 3 = bridge_orig (FT-SIMPLER-WIDOWX).
# (35 = new_embodiment is the legacy GR00T slot, superseded by general_embodiment.)
DEFAULT_EMBODIMENT_ID = 0

# Backbone / processor defaults. RLWRLD/RLDX-1-VLM ships the exact Qwen3-VL
# processor used to train the released checkpoints.
DEFAULT_MODEL_NAME = "RLWRLD/RLDX-1-VLM"
DEFAULT_TASK = "Perform the task."

# Fixed upstream RLDX-1 recipe (FT config.json ground truth): text-first prompt
# order, lowercase + strip punctuation.
FORMALIZE_LANGUAGE = True
CONVERSATION_IMAGE_FIRST = False


# ============================================================================ #
# Preprocessor                                                                 #
# ============================================================================ #


class Rldx1Preprocessor(nn.Module):
    """Preprocessor for RLDX-1 policy inputs.

    Converts a Studio :class:`~physicalai.data.observation.Observation` (or an
    equivalent flat dict) into the exact ``inputs`` dict consumed by
    :meth:`RLDX.forward` / :meth:`RLDX.get_action`.

    Runs the PAS-native pipeline (:meth:`forward`): library state/action
    normalization, ``cv2`` aspect-area resize + center crop,
    ``formalize_language``, text-first Qwen3-VL chat template, and HF
    tokenization. The native pipeline is parity-tested against the original
    vendored processor (see ``tests/unit/policies/test_rldx1_preprocessing.py``).
    The HF Qwen3-VL processor is loaded lazily on the first call.

    Scope (v1): absolute actions, single observation step. Relative actions are
    not supported.

    Args:
        max_state_dim: Padded state dimension (shorter states zero-padded).
        max_action_dim: Padded action dimension (shorter actions zero-padded).
        action_horizon: Number of action steps predicted per chunk.
        model_name: HuggingFace ID / local path of the Qwen3-VL processor.
        revision: Pinned git commit SHA for the processor download.
        normalize: Retained for API compatibility; normalization is always
            applied from ``stats``.
        use_percentiles: Use 1st/99th percentile bounds instead of min/max.
        clip_outliers: Clip normalized state/action to ``[-1, 1]`` (upstream
            ``clip_outliers``). ``True`` (default) matches upstream: the forward
            pass clamps train targets; the postprocessor clamps before the
            inverse. ``False`` (Pi05-style) skips both clamps, preserving
            out-of-percentile action tails for wide-range tasks (e.g. PushT).
        image_max_area: Target max area (pixels) for aspect-preserving resize.
        image_min_area: Minimum pixel-area floor forwarded to the Qwen vision
            tiler as ``min_pixels``. Frames below this area are upscaled to it
            (aspect-preserving) before encoding. ``None`` (default) leaves the
            tiler's own floor untouched. Set to e.g. ``65536`` (256x256) to lift
            tiny frames such as 96x96 PushT into a richer visual token sequence.
        image_resize_m: Alignment multiple for resized/cropped dimensions.
        default_task: Fallback instruction when an observation has no task.
        embodiment_id: Per-embodiment projector slot in the MSAT action head.
            Default 0 (general_embodiment) for a fresh new-robot fine-tune; set
            the slot a released checkpoint was trained on to reuse it.
        stats: Dataset statistics ``{key: {min, max, mean, std, q01, q99}}``.

    Examples:
        >>> pre = Rldx1Preprocessor(stats=dataset.stats)
        >>> inputs = pre(observation)
        >>> out = model.net.get_action(inputs)
    """

    def __init__(
        self,
        *,
        max_state_dim: int = MAX_STATE_DIM,
        max_action_dim: int = MAX_ACTION_DIM,
        action_horizon: int = ACTION_HORIZON,
        model_name: str = DEFAULT_MODEL_NAME,
        revision: str | None = None,
        normalize: bool = True,
        use_percentiles: bool = True,
        clip_outliers: bool = True,
        image_max_area: int = 65536,
        image_min_area: int | None = None,
        image_resize_m: int = 32,
        default_task: str = DEFAULT_TASK,
        embodiment_id: int = DEFAULT_EMBODIMENT_ID,
        stats: dict[str, dict[str, list[float]]] | None = None,
    ) -> None:
        """Initialize the preprocessor and build the normalization blocks."""
        super().__init__()

        self.max_state_dim = max_state_dim
        self.max_action_dim = max_action_dim
        self.action_horizon = action_horizon
        self.model_name = model_name
        self.revision = revision
        self.normalize = normalize
        self.use_percentiles = use_percentiles
        self.clip_outliers = clip_outliers
        self.image_max_area = image_max_area
        self.image_min_area = image_min_area
        self.image_resize_m = image_resize_m
        self.default_task = default_task
        self.embodiment_id = embodiment_id

        # Deterministic image geometry, shared by train and eval (no stochastic
        # augmentation).
        self._image_transform = AspectAreaResizeAndCrop(
            target_area=image_max_area,
            m_alignment=image_resize_m,
            min_area=image_min_area,
        )

        # PAS-native state/action normalizer (Stage 1). An nn.Module submodule so
        # its min/max/q01/q99 buffers move with `.to(device)` and export cleanly.
        sa_features = build_state_action_features(stats)
        self._has_sa_features = bool(sa_features)
        if sa_features:
            norm_map = build_state_action_norm_map(use_percentiles=use_percentiles)
            self._state_action_normalizer: nn.Module = FeatureNormalizeTransform(sa_features, norm_map)
        else:
            self._state_action_normalizer = nn.Identity()

        # Degenerate (constant) channel masks. Upstream masks q99==q01 channels to
        # 0; PAS QUANTILES/MIN_MAX map them to -1. Zero them post-normalize for
        # parity. persistent=False: derived from stats, recomputed on reload.
        degenerate = build_state_action_degenerate_masks(sa_features, use_percentiles=use_percentiles)
        self.register_buffer("_degenerate_mask_state", degenerate.get(STATE), persistent=False)
        self.register_buffer("_degenerate_mask_action", degenerate.get(ACTION), persistent=False)

        # Lazily-loaded HF Qwen processor for the PAS-native VLM path (Stage 4).
        self._vlm_processor_cache: ProcessorMixin | None = None

    # -- native VLM processor ---------------------------------------------- #
    @property
    def _vlm_processor(self) -> ProcessorMixin:
        """Lazily load the HF Qwen processor for the PAS-native VLM path.

        Returns:
            The cached HF processor with left padding (Flash-Attention
            compatible), loaded with ``trust_remote_code=False`` and the pinned
            ``revision``.
        """
        if self._vlm_processor_cache is None:
            # lib.security: never trust_remote_code; pin the processor revision.
            # revision is passed as an explicit keyword (not via a conditionally
            # built kwargs dict) so the pin is auditable to static scanners.
            processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=False,
                use_fast=True,
                revision=self.revision,
            )
            processor.tokenizer.padding_side = "left"
            self._vlm_processor_cache = processor
        return self._vlm_processor_cache

    # -- image geometry ------------------------------------------------------ #
    def _transform_frames(self, frames: list[np.ndarray]) -> list[Image.Image]:
        """Resize a sample's frames into PIL images via the torchvision geometry.

        Args:
            frames: ``(H, W, 3)`` uint8 frames for one view of one sample.

        Returns:
            List of resized/cropped PIL images, one per input frame.
        """
        pil_frames = []
        for frame in frames:
            tensor = torch.from_numpy(frame).permute(2, 0, 1)
            transformed = self._image_transform(tensor)
            pil_frames.append(Image.fromarray(np.ascontiguousarray(transformed.permute(1, 2, 0).numpy())))
        return pil_frames

    # -- native forward ---------------------------------------------------- #
    def _normalize_pad_state_action(
        self,
        batch_dict: dict[str, Any],
        *,
        has_action: bool,
    ) -> dict[str, torch.Tensor]:
        """Normalize, clip and pad state/action with the PAS-native blocks.

        Returns:
            Dict with padded ``state`` ``(B, 1, max_state_dim)`` and, when an
            action is present, ``action`` ``(B, action_horizon, max_action_dim)``
            plus ``action_mask``.
        """
        # Run on the normalizer's buffer device (falls back to CPU for Identity).
        sa_device = _module_device(self._state_action_normalizer)

        state_raw = batch_dict.get(OBSERVATION_STATE, batch_dict.get(STATE))
        state_tensor = self._as_float_tensor(state_raw)
        # When observation_delta_indices returns the VTC video window, the dataset
        # delivers state with shape (B, T, D). The RLDX backbone uses only the
        # current-step state, so slice to the last frame.
        if state_tensor.ndim == 3:  # noqa: PLR2004
            state_tensor = state_tensor[:, -1:, :]
        sa_batch: dict[str, Any] = {STATE: state_tensor.to(sa_device)}
        if has_action:
            sa_batch[ACTION] = self._as_float_tensor(batch_dict[ACTION]).to(sa_device)

        sa_batch = self._state_action_normalizer(sa_batch)
        if self._has_sa_features and self.clip_outliers:
            sa_batch = clip_state_action(sa_batch)
        if self._has_sa_features:
            # Match upstream: constant (q99==q01) channels normalize to 0, not -1.
            sa_batch = zero_degenerate_state_action(
                sa_batch,
                {
                    STATE: cast("torch.Tensor", self._degenerate_mask_state),
                    ACTION: cast("torch.Tensor", self._degenerate_mask_action),
                },
            )
        sa_batch = pad_state_action(
            sa_batch,
            max_state_dim=self.max_state_dim,
            max_action_dim=self.max_action_dim,
            max_action_horizon=self.action_horizon,
        )

        dtype = torch.get_default_dtype()
        return {key: value.to(dtype) for key, value in sa_batch.items()}

    def _build_conversations(
        self,
        batch_dict: dict[str, Any],
        view_keys: list[str],
        tasks: list[str],
        batch_size: int,
    ) -> tuple[list[list[dict[str, Any]]], int]:
        """Build one Qwen conversation per sample and report the frame count.

        Each view can carry ``num_frames`` temporal frames (the VTC video
        window). Frames are resized then flattened frame-major / view-inner --
        ``[t0v0, t0v1, ..., t1v0, ...]`` -- matching upstream ``_get_vlm_inputs``.

        Returns:
            ``(conversations, num_frames)`` -- ``batch_size`` conversations and
            the per-sample temporal frame count.
        """
        conversations: list[list[dict[str, Any]]] = []
        num_frames = 1
        for index in range(batch_size):
            per_view_frames: list[list[Image.Image]] = []
            for view in view_keys:
                frames = self._frames_for_view(batch_dict[view], index)
                per_view_frames.append(self._transform_frames(frames))

            num_frames = len(per_view_frames[0])
            # Frame-major, view-inner ordering (upstream: for t: for view).
            images = [per_view_frames[view_idx][t] for t in range(num_frames) for view_idx in range(len(view_keys))]

            language = formalize_language(tasks[index]) if FORMALIZE_LANGUAGE else tasks[index]
            conversations.append(
                build_qwen_conversation(images, language, image_first=CONVERSATION_IMAGE_FIRST),
            )
        return conversations, num_frames

    def _frames_for_view(self, view_value: Any, index: int) -> list[np.ndarray]:  # noqa: ANN401
        """Return the ``(H, W, 3)`` uint8 frames for one sample and view.

        Handles both single-frame (``(C, H, W)`` / ``(H, W, C)``) and multi-frame
        (``(T, C, H, W)`` / ``(T, H, W, C)``) layouts; ``delta_timestamps`` adds a
        leading temporal axis when a video window is requested.

        Returns:
            A list of per-frame ``(H, W, 3)`` uint8 arrays.
        """
        sample = self._index(view_value, index)
        return [self._to_hwc_uint8(frame) for frame in self._split_frames(sample)]

    @staticmethod
    def _split_frames(sample: Any) -> list[np.ndarray]:  # noqa: ANN401
        """Split a per-sample view array into a list of single-frame arrays.

        Returns:
            One array per temporal frame. A 4-D input is split along its leading
            (temporal) axis; a 3-D input is returned as a single frame.
        """
        arr = np.asarray(sample)
        if arr.ndim == 4:  # noqa: PLR2004 - (T, C, H, W) or (T, H, W, C)
            return [arr[t] for t in range(arr.shape[0])]
        return [arr]

    def forward(self, batch: Observation | dict[str, Any]) -> dict[str, torch.Tensor]:
        """Preprocess an observation batch into RLDX ``inputs``.

        Runs the PAS-native pipeline: state/action normalization + padding,
        torchvision image geometry, and Qwen3-VL tokenization (see
        ``preprocessing.py``). Supports the v1 configuration only: absolute
        actions, single observation step, deterministic image geometry.

        Args:
            batch: Input as :class:`Observation` or a flat dict with keys
                ``state``, ``images.*`` / ``observation.images.*``, ``task``,
                and optionally ``action`` (for the training loss).

        Returns:
            Dict of tensors consumed by :meth:`RLDX.forward` /
            :meth:`RLDX.get_action`.

        Raises:
            ValueError: If the observation carries no camera image.
        """
        batch_dict = batch.to_dict(flatten=True) if isinstance(batch, Observation) else dict(batch)
        view_keys = self._image_keys(batch_dict)
        if not view_keys:
            msg = "RLDX-1 preprocessor requires at least one camera image."
            raise ValueError(msg)

        batch_size, device = self._infer_batch_info(batch_dict)
        tasks = self._task_strings(batch_dict.get(TASK), batch_size)
        has_action = batch_dict.get(ACTION) is not None

        sa_inputs = self._normalize_pad_state_action(batch_dict, has_action=has_action)
        conversations, num_frames = self._build_conversations(batch_dict, view_keys, tasks, batch_size)
        vlm = tokenize_vlm_batch(self._vlm_processor, conversations)

        inputs: dict[str, torch.Tensor] = {
            INPUT_IDS: vlm[INPUT_IDS],
            ATTENTION_MASK: vlm[ATTENTION_MASK],
            PIXEL_VALUES: vlm[PIXEL_VALUES],
            IMAGE_GRID_THW: vlm[IMAGE_GRID_THW],
            # vtc scalars: bool image-wise flag, per-sample view/frame counts.
            IMAGE_WISE_ENCODING: torch.tensor([True] * batch_size),
            NUM_VIEWS: torch.tensor([len(view_keys)] * batch_size),
            NUM_FRAMES: torch.tensor([num_frames] * batch_size),
            STATE: sa_inputs[STATE],
            "embodiment_id": torch.tensor([self.embodiment_id] * batch_size),
        }
        if MM_TOKEN_TYPE_IDS in vlm:
            inputs[MM_TOKEN_TYPE_IDS] = vlm[MM_TOKEN_TYPE_IDS]
        if has_action:
            inputs[ACTION] = sa_inputs[ACTION]
            inputs[ACTION_MASK] = sa_inputs[ACTION_MASK]

        return {key: (value.to(device) if isinstance(value, torch.Tensor) else value) for key, value in inputs.items()}

    @staticmethod
    def _as_float_tensor(value: Any) -> torch.Tensor:  # noqa: ANN401
        """Coerce a state/action value to a float32 tensor.

        Returns:
            A ``float32`` tensor view of ``value``.
        """
        if isinstance(value, torch.Tensor):
            return value.to(torch.float32)
        return torch.as_tensor(np.asarray(value, dtype=np.float32))

    @staticmethod
    def _index(value: Any, index: int) -> np.ndarray | None:  # noqa: ANN401
        """Return the ``index``-th element of a batched tensor/array as numpy.

        Returns:
            The selected sample as a numpy array, or ``None`` when ``value`` is None.
        """
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            sample = value[index].detach().cpu()
            # Upcast only floating tensors (numpy has no bf16); keep uint8 as
            # uint8 so _to_hwc_uint8 doesn't misread it as float [0, 1].
            if sample.is_floating_point():
                sample = sample.to(torch.float32)
            return sample.numpy()
        return np.asarray(value)[index]

    @staticmethod
    def _to_hwc_uint8(arr: np.ndarray) -> np.ndarray:
        """Convert an image array to contiguous ``(H, W, 3)`` uint8 layout.

        Returns:
            A contiguous ``(H, W, 3)`` uint8 array.
        """
        arr = np.asarray(arr)
        # CHW -> HWC when the leading axis is the channel axis.
        if arr.ndim == 3 and arr.shape[0] in {1, 3} and arr.shape[-1] not in {1, 3}:  # noqa: PLR2004
            arr = np.transpose(arr, (1, 2, 0))
        if np.issubdtype(arr.dtype, np.floating):
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
        if arr.ndim == 3 and arr.shape[-1] == 1:  # noqa: PLR2004
            arr = np.repeat(arr, 3, axis=-1)
        return np.ascontiguousarray(arr)

    @staticmethod
    def _infer_batch_info(batch: dict[str, Any]) -> tuple[int, torch.device]:
        """Infer batch size and device from the first tensor in ``batch``.

        Returns:
            Tuple of ``(batch_size, device)``.
        """
        for value in batch.values():
            if isinstance(value, torch.Tensor):
                return value.shape[0], value.device
        return 1, torch.device("cpu")

    @staticmethod
    def _image_keys(batch: dict[str, Any]) -> list[str]:
        """Find sorted image keys, supporting LeRobot and Observation formats.

        Excludes LeRobot's per-key ``*_is_pad`` padding masks: with a VTC video
        window, ``reformat_dataset_to_match_policy`` sets ``delta_indices`` on
        every ``observation.*`` key, and the dataset attaches a sibling
        ``{key}_is_pad`` bool mask for each -- which shares the
        ``observation.images.`` prefix and would otherwise be misread as a camera.

        Returns:
            Sorted list of image keys (possibly empty).
        """
        keys = sorted(k for k in batch if k.startswith(OBSERVATION_IMAGES_PREFIX) and "is_pad" not in k)
        if not keys:
            keys = sorted(k for k in batch if k.startswith(f"{IMAGES}.") and k != IMAGES and "is_pad" not in k)
        if not keys and OBSERVATION_IMAGE in batch:
            keys = [OBSERVATION_IMAGE]
        if not keys and isinstance(batch.get(IMAGES), torch.Tensor):
            keys = [IMAGES]
        return keys

    @staticmethod
    def _task_strings(task: Any, batch_size: int) -> list[str]:  # noqa: ANN401
        """Normalize the task field into a list of ``batch_size`` strings.

        Returns:
            List of task strings, one per batch element.
        """
        if task is None:
            return [DEFAULT_TASK] * batch_size
        if isinstance(task, str):
            return [task] * batch_size
        if isinstance(task, (list, tuple)):
            tasks = [str(t) for t in task]
            if len(tasks) == 1:
                return tasks * batch_size
            return tasks
        return [str(task)] * batch_size


# ============================================================================ #
# Postprocessor                                                                #
# ============================================================================ #


class Rldx1Postprocessor(nn.Module):
    """Postprocessor for RLDX-1 policy outputs.

    Decodes the RLDX ``action_pred`` ``(B, action_horizon, max_action_dim)``
    back to the environment action space using its own inverse
    :class:`~physicalai.policies.utils.normalization.FeatureNormalizeTransform`.

    Args:
        action_feature: Action :class:`~physicalai.data.Feature` carrying the
            normalization statistics.  ``None`` skips denormalization.
        use_percentiles: Use 1st/99th percentile bounds instead of min/max
            (must match the paired preprocessor).
        clip_outliers: Clamp normalized action to ``[-1, 1]`` before the
            inverse transform (must match the paired preprocessor).
        action_horizon: Number of action steps per chunk.
        env_action_dim: Original (unpadded) action dimension. ``0`` keeps the
            full decoded width.

    Examples:
        >>> pre, post = make_rldx1_transforms(stats=dataset.stats, env_action_dim=7)
        >>> inputs = pre(observation)
        >>> action = post(model.net.get_action(inputs))
    """

    def __init__(
        self,
        *,
        action_feature: Feature | None = None,
        use_percentiles: bool = True,
        clip_outliers: bool = True,
        action_horizon: int = ACTION_HORIZON,
    ) -> None:
        """Initialize the postprocessor."""
        super().__init__()
        self.action_horizon = action_horizon
        self.clip_outliers = clip_outliers

        if action_feature is not None and action_feature.shape is not None:
            self._action_dim = int(action_feature.shape[0])
            norm_map = build_state_action_norm_map(use_percentiles=use_percentiles)
            self._action_denormalizer: nn.Module = FeatureNormalizeTransform(
                {ACTION: action_feature},
                norm_map,
                inverse=True,
            )
        else:
            self._action_dim = 0
            self._action_denormalizer = nn.Identity()

    def forward(
        self,
        action: torch.Tensor,
        state: dict[str, np.ndarray] | None = None,  # noqa: ARG002 - reserved for relative-action decoding
    ) -> torch.Tensor:
        """Decode the predicted action chunk to the environment action space.

        Args:
            action: RLDX prediction ``(B, T, D)`` or ``(B, D)``.
            state: Reserved for future relative-action decoding; ignored.

        Returns:
            Action chunk ``(B, T, env_action_dim)`` (or ``(B, env_action_dim)``
            when the input was 2-D) in the original environment space.
        """
        squeeze = action.dim() == 2  # noqa: PLR2004
        if squeeze:
            action = action.unsqueeze(1)

        if self._action_dim > 0:
            sliced = action[..., : self.action_horizon, : self._action_dim]
            if self.clip_outliers:
                sliced = sliced.clamp(-1.0, 1.0)
            denorm_device = sliced.device
            for buf in self._action_denormalizer.buffers():
                denorm_device = buf.device
                break
            batch = {ACTION: sliced.to(denorm_device)}
            batch = self._action_denormalizer(batch)
            out_t = batch[ACTION].to(action.device, action.dtype)
        else:
            out_t = action

        return out_t.squeeze(1) if squeeze else out_t


# ============================================================================ #
# Factory                                                                      #
# ============================================================================ #


def make_rldx1_transforms(
    *,
    stats: dict[str, dict[str, list[float]]] | None,
    max_state_dim: int = MAX_STATE_DIM,
    max_action_dim: int = MAX_ACTION_DIM,
    action_horizon: int = ACTION_HORIZON,
    model_name: str = DEFAULT_MODEL_NAME,
    revision: str | None = None,
    normalize: bool = True,
    use_percentiles: bool = True,
    clip_outliers: bool = True,
    image_max_area: int = 65536,
    image_min_area: int = 50176,
    image_resize_m: int = 32,
    embodiment_id: int = DEFAULT_EMBODIMENT_ID,
) -> tuple[Rldx1Preprocessor, Rldx1Postprocessor]:
    """Build the matched RLDX-1 preprocessor / postprocessor pair.

    Both share the same normalization statistics, so the forward and inverse
    transforms are guaranteed consistent.

    Args:
        stats: Dataset statistics for (de)normalization.
        env_action_dim: Original environment action dimension for decoding.
        max_state_dim: Padded state dimension.
        max_action_dim: Padded action dimension.
        action_horizon: Number of action steps per chunk.
        model_name: HuggingFace ID / local path of the Qwen3-VL processor.
        revision: Pinned git commit SHA for the processor download.
        normalize: Retained for API compatibility (normalization always applied).
        use_percentiles: Prefer 1st/99th percentile bounds over min/max.
        clip_outliers: Clip normalized state/action to ``[-1, 1]`` at train and
            inference (upstream default ``True``; ``False`` = Pi05-style no clip).
        image_max_area: Target max area (pixels) for aspect-preserving resize.
        image_min_area: Minimum pixel-area floor forwarded to the Qwen vision
            tiler as ``min_pixels`` (frames below it are upscaled, aspect-
            preserving). ``None`` (default) leaves the tiler's floor untouched;
            e.g. ``65536`` lifts 96x96 PushT frames to 256x256.
        image_resize_m: Alignment multiple for resized/cropped dimensions.
        embodiment_id: Per-embodiment projector slot in the MSAT action head
            (default 0 = general_embodiment for a fresh new-robot fine-tune).

    Returns:
        Tuple of ``(preprocessor, postprocessor)``.
    """
    preprocessor = Rldx1Preprocessor(
        max_state_dim=max_state_dim,
        max_action_dim=max_action_dim,
        action_horizon=action_horizon,
        model_name=model_name,
        revision=revision,
        normalize=normalize,
        use_percentiles=use_percentiles,
        clip_outliers=clip_outliers,
        image_max_area=image_max_area,
        image_min_area=image_min_area,
        image_resize_m=image_resize_m,
        embodiment_id=embodiment_id,
        stats=stats,
    )
    action_feature = build_state_action_features(stats).get(ACTION)
    postprocessor = Rldx1Postprocessor(
        action_feature=action_feature,
        use_percentiles=use_percentiles,
        clip_outliers=clip_outliers,
        action_horizon=action_horizon,
    )
    return preprocessor, postprocessor
