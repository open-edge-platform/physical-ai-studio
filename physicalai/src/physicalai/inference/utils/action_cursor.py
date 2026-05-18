# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""ActionCursor: action-chunk buffer for temporal action dispensing."""

from __future__ import annotations

from collections import deque

import numpy as np


class ActionCursor:
    """Buffer that queues an action chunk and dispenses one timestep per call.

    Call :meth:`push_chunk` with the full action output from the runner
    (shape ``(batch, T, action_dim)``), then call :meth:`pop` repeatedly
    to retrieve individual timestep actions (shape ``(batch, action_dim)``).
    When the buffer is exhausted, :attr:`empty` is ``True`` and a new
    chunk should be pushed.

    Examples:
            >>> cursor = ActionCursor()
            >>> cursor.empty
            True
            >>> chunk = np.random.randn(1, 10, 7)  # batch=1, T=10, action_dim=7
            >>> cursor.push_chunk(chunk)
            >>> cursor.empty
            False
            >>> action = cursor.pop()  # shape (1, 7)
            >>> cursor.reset()
            >>> cursor.empty
            True
    """

    def __init__(self) -> None:
        """Initialize an empty ActionCursor with no buffered actions."""
        self._queue: deque[np.ndarray] = deque()

    @property
    def empty(self) -> bool:
        """True when there are no buffered actions remaining."""
        return len(self._queue) == 0

    def push_chunk(self, chunk: np.ndarray) -> None:
        """Queue all timestep slices from an action chunk.

        Args:
                chunk: Action array with shape ``(batch, T, action_dim)`` or
                        ``(T, action_dim)``.  Each of the ``T`` timestep slices is
                        enqueued individually.
        """
        min_batched_action_dim = 2
        if chunk.ndim == min_batched_action_dim:
            # (T, action_dim) - no batch dimension
            self._queue.extend(chunk)
        else:
            # (batch, T, action_dim) - transpose to (T, batch, action_dim)
            self._queue.extend(np.transpose(chunk, (1, 0, 2)))

    def pop(self) -> np.ndarray:
        """Return and remove the next buffered action.

        Returns:
                Action array for the current timestep, shape ``(batch, action_dim)``
                or ``(action_dim,)`` depending on what was pushed.

        Raises:
                IndexError: If the queue is empty.
        """
        if self.empty:
            msg = "ActionCursor is empty; call push_chunk before pop."
            raise IndexError(msg)
        return self._queue.popleft()

    def reset(self) -> None:
        """Clear all buffered actions."""
        self._queue.clear()

    def __repr__(self) -> str:
        """Return string representation."""
        return f"ActionCursor(buffered={len(self._queue)})"
