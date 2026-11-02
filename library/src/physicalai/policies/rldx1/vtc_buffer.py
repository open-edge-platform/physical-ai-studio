# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Vendored from RLWRLD/RLDX-1 (Apache-2.0)

"""VTC (Video Temporal Context) frame buffer for RLDX-1 rollout.

The :class:`VtcWindowBuffer` is a stateful helper owned by :class:`Rldx1`.
It manages the per-episode ring buffer of camera frames so that
``predict_action_chunk`` can assemble the same multi-frame temporal window
the model was trained on (e.g. offsets ``[-6, -4, -2, 0]`` for
``video_length=4, video_stride=2``).

During training, ``delta_timestamps`` already delivers ``(B, T, C, H, W)``
batches, so :meth:`VtcWindowBuffer.prepare` is a no-op on those inputs.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from physicalai.data import Observation as Obs

from .preprocessor import Rldx1Preprocessor

if TYPE_CHECKING:
    from physicalai.data import Observation


class VtcWindowBuffer:
    """Stateful per-view frame ring buffer for VTC temporal windowing.

    Args:
        video_length: Number of frames in the temporal window (``T``).
        video_stride: Env-step stride between sampled frames.
    """

    def __init__(self, video_length: int, video_stride: int) -> None:
        """Initialize the temporal frame buffer.

        Args:
            video_length: Number of frames in the temporal window.
            video_stride: Env-step stride between sampled frames.
        """
        self._video_length = video_length
        self._video_stride = video_stride
        self._history: dict[str, deque[torch.Tensor]] | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all per-view frame history.

        Call at every episode boundary so the next episode builds its temporal
        window from scratch.
        """
        self._history = None

    # ------------------------------------------------------------------
    # State inspection
    # ------------------------------------------------------------------

    @property
    def is_warming_up(self) -> bool:
        """True while the buffer is filling and inference should be withheld.

        Returns ``False`` when:
        - The buffer has never been written (no camera views found, or
          ``video_length <= 1`` — no windowing needed).
        - The buffer is fully populated and a complete window is available.

        Returns ``True`` only while history is initialized but not yet full,
        i.e. during the warmup phase at the start of an episode.
        """
        if self._history is None:
            return False
        target = (self._video_length - 1) * self._video_stride + 1
        return any(len(buf) < target for buf in self._history.values())

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def record(self, batch: Observation | dict[str, Any]) -> None:
        """Append the current frame to each per-view ring buffer.

        No-op when ``video_length <= 1``, when no camera keys are found, or
        when the batch already carries a temporal axis (training path).

        Args:
            batch: Current single-frame observation.

        Raises:
            RuntimeError: If the buffer is not initialized by ``_ensure_history``
        """
        if self._video_length <= 1:
            return
        batch_dict = _to_flat_dict(batch)
        view_keys = _image_keys(batch_dict)
        if not view_keys or _is_multiframe(batch_dict, view_keys):
            return
        self._ensure_history(view_keys)
        if self._history is None:
            msg = "History should have been initialized by _ensure_history"
            raise RuntimeError(msg)
        for key in view_keys:
            history = self._history[key]
            frame = _as_frame_tensor(batch_dict[key])
            if not history:
                window_size = (self._video_length - 1) * self._video_stride + 1
                history.extend([frame] * window_size)
            else:
                history.append(frame)

    def prepare(self, batch: Observation | dict[str, Any]) -> Observation | dict[str, Any]:
        """Return the batch with the VTC window applied.

        Passes through unchanged when:
        - ``video_length <= 1`` (single-frame model),
        - no camera keys are found, or
        - the batch already has a temporal axis (training / validation path).

        When the buffer has not been seeded yet (direct call that bypassed
        ``select_action``), seeds the history with the current frame so the
        window is well-defined.

        Args:
            batch: Current observation — single frame per view or already
                multi-frame.

        Returns:
            The original batch, or a flat dict with ``(B, T, C, H, W)`` views.
        """
        if self._video_length <= 1:
            return batch
        batch_dict = _to_flat_dict(batch)
        view_keys = _image_keys(batch_dict)
        if not view_keys or _is_multiframe(batch_dict, view_keys):
            return batch
        # Seed if bypassed select_action (e.g. direct eval ``forward`` call).
        if self._history is None or not all(k in self._history for k in view_keys):
            self.record(batch_dict)
        return self._apply_window(batch_dict, view_keys)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_history(self, view_keys: list[str]) -> None:
        if self._history is None:
            span = (self._video_length - 1) * self._video_stride
            self._history = {key: deque(maxlen=span + 1) for key in view_keys}

    def _apply_window(
        self,
        batch_dict: dict[str, Any],
        view_keys: list[str],
    ) -> dict[str, Any]:
        """Stack each view's history into a ``(B, T, C, H, W)`` VTC window.

        Samples the history at offsets ``[(i - (L-1)) * S for i in range(L)]``,
        clamping to the oldest available frame when the episode is shorter than
        the window span — matching upstream reset-fill behaviour.

        Returns:
            A shallow copy of ``batch_dict`` with the view arrays replaced by
            their stacked multi-frame windows.
        """
        vl = self._video_length
        vs = self._video_stride
        offsets = [(i - (vl - 1)) * vs for i in range(vl)]
        assert self._history is not None  # noqa: S101
        out = dict(batch_dict)
        for key in view_keys:
            buffer = self._history[key]
            count = len(buffer)
            frames = [buffer[max(0, count - 1 + offset)] for offset in offsets]
            out[key] = torch.stack(frames, dim=1)  # (B, T, C, H, W)
        return out


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _to_flat_dict(batch: Observation | dict[str, Any]) -> dict[str, Any]:
    """Return a flat observation dict for both ``Observation`` and dict inputs."""
    if isinstance(batch, Obs):
        return batch.to_dict(flatten=True)
    return dict(batch)


def _image_keys(batch_dict: dict[str, Any]) -> list[str]:
    """Return the camera view keys detected by the preprocessor."""
    return Rldx1Preprocessor._image_keys(batch_dict)  # noqa: SLF001


def _as_frame_tensor(value: Any) -> torch.Tensor:  # noqa: ANN401
    """Coerce a single-frame view value to a ``(B, C, H, W)`` tensor.

    Returns:
        Tensor of shape ``(B, C, H, W)``.
    """
    if isinstance(value, torch.Tensor):
        return value
    return torch.as_tensor(np.asarray(value))


def _is_multiframe(batch_dict: dict[str, Any], view_keys: list[str]) -> bool:
    """Return ``True`` when the views already carry a temporal axis.

    A batched single frame is ``(B, C, H, W)`` (4-D); a batched VTC window is
    ``(B, T, C, H, W)`` (5-D). A 5-D view means the ``delta_timestamps`` path
    already produced the window and no history stacking is needed.
    """
    value = batch_dict[view_keys[0]]
    ndim = value.dim() if isinstance(value, torch.Tensor) else np.asarray(value).ndim
    return ndim == 5  # noqa: PLR2004 - (B, T, C, H, W)
