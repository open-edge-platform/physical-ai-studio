# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Real-Time Chunking (RTC) async inference runner.

Runs inference in a background daemon thread while the main thread
pops actions from a dual-track queue at robot Hz. Designed for
flow-matching / diffusion policies exported with RTC denoising baked
into the graph (e.g. Pi05RTCWrapper → OpenVINO).

Unlike :class:`~physicalai.inference.runners.ActionChunking` (synchronous,
blocks on empty queue), this runner never blocks the control loop after
the first chunk arrives.
"""

from __future__ import annotations

import logging
import time
from copy import deepcopy
from threading import Event, Lock, Thread
from typing import TYPE_CHECKING, Any

import numpy as np

from physicalai.inference.constants import ACTION
from physicalai.inference.rtc.action_queue import RTCActionQueue
from physicalai.inference.rtc.latency_tracker import LatencyTracker
from physicalai.inference.runners.base import InferenceRunner

if TYPE_CHECKING:
    from physicalai.inference.adapters.base import RuntimeAdapter
    from physicalai.inference.postprocessors.base import Postprocessor

logger = logging.getLogger(__name__)

# How long the RTC loop sleeps when paused, idle, or backpressured by a full queue.
_RTC_IDLE_SLEEP_S: float = 0.01
# Backoff between transient inference errors (per consecutive failure).
_RTC_ERROR_RETRY_DELAY_S: float = 0.5
# Consecutive transient errors tolerated before giving up and re-raising.
_RTC_MAX_CONSECUTIVE_ERRORS: int = 10
# Hard timeout for joining the RTC thread on stop().
_RTC_JOIN_TIMEOUT_S: float = 3.0


class RTCActionChunking(InferenceRunner):
    """Async RTC runner with background inference thread.

    On the first ``run()`` call, spawns a daemon thread that continuously
    predicts action chunks and merges them into a thread-safe queue.
    Subsequent ``run()`` calls publish the latest observation and pop one
    action from the queue — no blocking inference on the main thread.

    The background thread injects RTC-specific inputs:
    - ``noise``: random noise for the denoising process
    - ``prev_chunk_left_over``: unconsumed tail of the previous chunk
    - ``inference_delay``: integer scalar derived from measured latency

    Args:
        runner: Inner runner (typically :class:`SinglePass`) that calls
            the adapter.
        chunk_size: Number of actions per chunk (model output dim).
        execution_horizon: How many fresh actions to execute per chunk.
        fps: Robot control frequency in Hz.
        max_action_dim: Model's internal latent action dimension
            (used for noise and prev_chunk_left_over shapes). This is
            NOT the robot output DOF — e.g. 32 for Pi05, even though
            output is 6 for SO-101.
        queue_threshold: Re-infer when queue drops to this level.
            Defaults to 30.
        action_key: Key for the action in the output dict returned to
            callers (e.g. ``"action"``).
        postprocessors: Optional postprocessor pipeline to apply inside
            the RTC thread (e.g. denormalization). These run in the
            background thread so the queue stores both raw and processed
            actions.
    """

    def __init__(
        self,
        runner: InferenceRunner,
        chunk_size: int = 50,
        execution_horizon: int = 10,
        fps: float = 30.0,
        max_action_dim: int = 32,
        queue_threshold: int | None = None,
        action_key: str = ACTION,
        postprocessors: list[Postprocessor] | None = None,
        max_guidance_weight: float = 10.0,
    ) -> None:
        self._inner = runner
        self._chunk_size = chunk_size
        self._execution_horizon = execution_horizon
        self._fps = fps
        self._max_action_dim = max_action_dim
        self._queue_threshold = queue_threshold if queue_threshold is not None else 30
        self._action_key = action_key
        self._postprocessors: list[Postprocessor] = postprocessors or []
        self._max_guidance_weight = max_guidance_weight

        self._queue = RTCActionQueue()
        self._latency_tracker = LatencyTracker()

        self._obs_lock = Lock()
        self._obs_holder: dict[str, np.ndarray] | None = None

        self._shutdown = Event()
        self._active = Event()
        self._first_chunk_ready = Event()
        self._thread: Thread | None = None
        self._adapter: RuntimeAdapter | None = None

    @property
    def runner_provided_keys(self) -> set[str]:
        """RTC-specific inputs injected by the background thread."""
        return {"noise", "prev_chunk_left_over", "inference_delay", "max_guidance_weight", "execution_horizon"}

    def run(
        self,
        adapter: RuntimeAdapter,
        inputs: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Publish observation and pop one action from the queue.

        On the first call, starts the background thread and blocks
        until the first chunk is ready. Subsequent calls are non-blocking.

        Args:
            adapter: The loaded runtime adapter.
            inputs: Pre-processed model inputs (observation).

        Returns:
            Dict with one action under ``self._action_key``.
        """
        # Start thread on first call
        if self._thread is None:
            self._start(adapter)

        # Publish latest observation
        with self._obs_lock:
            self._obs_holder = deepcopy(inputs)

        # Wait for first chunk (only blocks on very first call)
        if not self._first_chunk_ready.is_set():
            self._first_chunk_ready.wait()

        # Pop one action
        action = self._queue.get()
        if action is None:
            # Queue drained between threshold check and here — return zeros
            action = np.zeros(self._max_action_dim, dtype=np.float32)

        return {self._action_key: action}

    def reset(self) -> None:
        """Stop background thread and clear all state."""
        self._stop()
        self._queue.reset()
        self._latency_tracker.reset()
        self._first_chunk_ready.clear()

    def _start(self, adapter: RuntimeAdapter) -> None:
        """Launch the background inference thread."""
        self._adapter = adapter
        self._shutdown.clear()
        self._active.set()
        self._thread = Thread(
            target=self._rtc_loop,
            name="rtc-inference",
            daemon=True,
        )
        self._thread.start()
        logger.info("RTC background thread started (fps=%.1f, chunk=%d, horizon=%d)",
                    self._fps, self._chunk_size, self._execution_horizon)

    def _stop(self) -> None:
        """Signal shutdown and wait for the thread to exit."""
        if self._thread is not None:
            self._shutdown.set()
            self._thread.join(timeout=_RTC_JOIN_TIMEOUT_S)
            if self._thread.is_alive():
                logger.warning("RTC thread did not join within %.1fs", _RTC_JOIN_TIMEOUT_S)
            self._thread = None
            self._adapter = None
            logger.info("RTC background thread stopped")

    def _rtc_loop(self) -> None:
        """Background loop: infer chunks and merge into queue."""
        assert self._adapter is not None  # noqa: S101
        consecutive_errors = 0

        while not self._shutdown.is_set():
            if not self._active.is_set():
                self._active.wait(timeout=_RTC_IDLE_SLEEP_S)
                continue

            # Only re-infer when queue is running low
            if self._queue.qsize() > self._queue_threshold:
                time.sleep(_RTC_IDLE_SLEEP_S)
                continue

            # 1. Snapshot observation
            with self._obs_lock:
                if self._obs_holder is None:
                    time.sleep(_RTC_IDLE_SLEEP_S)
                    continue
                inputs = deepcopy(self._obs_holder)

            # 2. Get leftover from queue (raw model output, possibly trimmed)
            prev_chunk = self._queue.get_left_over()
            if prev_chunk is None:
                prev_chunk = np.zeros(
                    (1, self._chunk_size, self._max_action_dim),
                    dtype=np.float32,
                )
            else:
                # prev_chunk shape: (remaining, output_dim) or (remaining, action_dim)
                remaining = prev_chunk.shape[0]
                out_dim = prev_chunk.shape[-1]

                # Pad action dim back to model's max_action_dim if output was trimmed
                if out_dim < self._max_action_dim:
                    prev_chunk = np.pad(
                        prev_chunk,
                        ((0, 0), (0, self._max_action_dim - out_dim)),
                    )

                # Reshape to (1, remaining, action_dim) and pad time to chunk_size
                prev_chunk = prev_chunk.reshape(1, remaining, self._max_action_dim)
                pad_len = self._chunk_size - remaining
                if pad_len > 0:
                    prev_chunk = np.pad(prev_chunk, ((0, 0), (0, pad_len), (0, 0)))

            # 3. Compute delay from worst-case latency
            latency = self._latency_tracker.max()
            max_delay = self._chunk_size - self._execution_horizon
            delay = int(np.ceil(latency * self._fps)) if latency > 0 else 0
            delay = min(delay, max_delay)

            # 4. Inject RTC-specific inputs
            inputs["prev_chunk_left_over"] = prev_chunk
            inputs["inference_delay"] = np.int64(delay)  # scalar, shape []
            inputs["noise"] = np.random.randn(1, self._chunk_size, self._max_action_dim).astype(np.float32)
            inputs["max_guidance_weight"] = np.float32(self._max_guidance_weight)
            inputs["execution_horizon"] = np.int64(self._execution_horizon)

            # Snapshot cursor before inference for delay cross-check
            action_index_before = self._queue.get_action_index()

            # 5. Run inference and measure latency
            try:
                t0 = time.perf_counter()
                outputs = self._inner.run(self._adapter, inputs)
                elapsed = time.perf_counter() - t0
                self._latency_tracker.add(elapsed)
                consecutive_errors = 0
            except Exception:
                consecutive_errors += 1
                logger.exception("RTC inference error (%d/%d)",
                                 consecutive_errors, _RTC_MAX_CONSECUTIVE_ERRORS)
                if consecutive_errors >= _RTC_MAX_CONSECUTIVE_ERRORS:
                    logger.error("Too many consecutive RTC errors — shutting down thread")
                    self._shutdown.set()
                    return
                time.sleep(_RTC_ERROR_RETRY_DELAY_S)
                continue

            # 6. Extract raw actions: (1, chunk_size, action_dim) → (chunk_size, action_dim)
            raw_actions = outputs[self._action_key]
            if raw_actions.ndim == 3:
                raw_actions = raw_actions[0]

            # 7. Postprocess (denormalize) for robot
            processed_actions = self._postprocess(raw_actions)

            # 8. Compute real delay and merge
            real_delay = int(np.ceil(elapsed * self._fps))
            real_delay = min(real_delay, max(0, len(raw_actions) - self._execution_horizon))

            self._queue.merge(
                raw_actions, processed_actions, real_delay,
                action_index_before_inference=action_index_before,
            )
            self._first_chunk_ready.set()

            logger.debug(
                "RTC chunk: latency=%.3fs delay=%d qsize=%d",
                elapsed, real_delay, self._queue.qsize(),
            )

    def _postprocess(self, actions: np.ndarray) -> np.ndarray:
        """Apply postprocessors to raw actions.

        Args:
            actions: Shape ``(chunk_size, action_dim)``.

        Returns:
            Postprocessed actions, same shape.
        """
        if not self._postprocessors:
            return actions.copy()

        # Wrap in dict for postprocessor interface
        outputs: dict[str, Any] = {self._action_key: actions}
        for pp in self._postprocessors:
            outputs = pp(outputs)
        return outputs[self._action_key]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"runner={self._inner!r}, "
            f"chunk_size={self._chunk_size}, "
            f"execution_horizon={self._execution_horizon}, "
            f"fps={self._fps})"
        )
