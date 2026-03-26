# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Action-chunking inference runner (decorator).

Wraps any ``InferenceRunner`` to add temporal action buffering. The inner
runner produces an output dict whose action value has shape
``(batch, horizon, action_dim)``.  This wrapper queues the individual
timesteps, dispensing one per call. Only invokes the inner runner again
when the queue is exhausted.

This is the GoF Decorator pattern: ``ActionChunking`` *is* an
``InferenceRunner`` and *has* an ``InferenceRunner``.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import numpy as np

from physicalai.inference.runners.base import InferenceRunner

if TYPE_CHECKING:
    from physicalai.inference.adapters.base import RuntimeAdapter


class ActionChunking(InferenceRunner):
    """Wrap a runner with temporal action buffering.

    On the first call (or when the queue is empty), delegates to the
    inner runner which returns an output dict containing an action with
    shape ``(batch, chunk_size, action_dim)``.  All chunk steps are
    enqueued and one is returned.  Subsequent calls pop from the queue
    without running inference.

    Args:
        runner: The inner runner to delegate inference to.
        chunk_size: Number of actions per chunk. Must match the inner
            runner's output temporal dimension.

    Examples:
        Wrap a single-pass runner with action chunking:

        >>> runner = ActionChunking(SinglePass(), chunk_size=10)
        >>> outputs = runner.run(adapter, inputs)  # runs inference, queues 10 actions
        >>> outputs = runner.run(adapter, inputs)  # pops from queue, no inference

        Compose with any runner (e.g. future flow-matching):

        >>> runner = ActionChunking(FlowMatching(num_steps=20), chunk_size=5)
    """

    def __init__(self, runner: InferenceRunner, chunk_size: int = 1) -> None:
        """Initialize with an inner runner and chunk size.

        Args:
            runner: The inner runner to wrap.
            chunk_size: Number of actions per chunk.
        """
        self.runner = runner
        self.chunk_size = chunk_size
        self._action_queue: deque[np.ndarray] = deque()

    def run(
        self,
        adapter: RuntimeAdapter,
        inputs: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Return the next action, running inference only when the queue is empty.

        Args:
            adapter: The loaded runtime adapter.
            inputs: Pre-processed model inputs.

        Returns:
            Output dict with a single action of shape
            ``(batch_size, action_dim)``.
        """
        if len(self._action_queue) > 0:
            return {"action": self._action_queue.popleft()}

        outputs = self.runner.run(adapter, inputs)
        actions = outputs["action"]

        batch_actions = np.transpose(actions, (1, 0, 2))
        self._action_queue.extend(batch_actions)

        return {"action": self._action_queue.popleft()}

    def reset(self) -> None:
        """Clear the action queue and reset the inner runner."""
        self._action_queue.clear()
        self.runner.reset()

    def __repr__(self) -> str:
        """Return string representation of the runner."""
        return f"{self.__class__.__name__}(runner={self.runner!r}, chunk_size={self.chunk_size})"
