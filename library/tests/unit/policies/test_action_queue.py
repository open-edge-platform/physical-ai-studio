# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for action chunking queue behavior in base Policy.

Regression test for commit 663c2f16: ``_queue_actions`` must truncate the
predicted chunk to ``n_action_steps`` before extending the queue. Otherwise the
bounded ``deque(maxlen=n_action_steps)`` silently drops the first items.
"""

from __future__ import annotations

import torch

from physicalai.data.observation import Observation
from physicalai.policies.base.policy import Policy


class _ChunkPolicy(Policy):
    """Concrete Policy returning a fixed action chunk for testing."""

    def __init__(self, chunk: torch.Tensor, n_action_steps: int) -> None:
        super().__init__(n_action_steps=n_action_steps)
        self._chunk = chunk
        self.predict_calls = 0

    def forward(self, batch: Observation) -> torch.Tensor:  # pragma: no cover - unused
        return self.predict_action_chunk(batch)

    def predict_action_chunk(self, batch: Observation) -> torch.Tensor:
        self.predict_calls += 1
        return self._chunk


def _dummy_batch() -> Observation:
    return Observation(state=torch.zeros(1, 1))


class TestQueueActions:
    """Regression tests for Policy._queue_actions / select_action."""

    def test_select_action_returns_first_chunk_item_unbatched(self) -> None:
        """First select_action call must return action at index 0, not 1.

        With chunk_size=8 and n_action_steps=4, before the fix the unbounded
        ``extend`` overflowed the deque and dropped indices [0..3], so the
        first action returned was index 4.
        """
        chunk = torch.arange(8, dtype=torch.float32).unsqueeze(-1)  # (T=8, D=1)
        policy = _ChunkPolicy(chunk=chunk, n_action_steps=4)

        first = policy.select_action(_dummy_batch())
        assert torch.equal(first, chunk[0])

    def test_select_action_returns_first_chunk_item_batched(self) -> None:
        """Same regression check for (B, T, D) shaped chunks."""
        # (B=2, T=8, D=1): values along T are 0..7 so we can identify the index.
        t = torch.arange(8, dtype=torch.float32).unsqueeze(-1)
        chunk = torch.stack([t, t + 100.0], dim=0)  # (2, 8, 1)
        policy = _ChunkPolicy(chunk=chunk, n_action_steps=4)

        first = policy.select_action(_dummy_batch())
        # First call should be t=0 across the batch.
        assert torch.equal(first, chunk[:, 0])

    def test_select_action_consumes_n_steps_then_repredicts(self) -> None:
        """Queue should yield exactly n_action_steps items before a new predict."""
        chunk = torch.arange(8, dtype=torch.float32).unsqueeze(-1)
        n = 4
        policy = _ChunkPolicy(chunk=chunk, n_action_steps=n)

        for i in range(n):
            action = policy.select_action(_dummy_batch())
            assert torch.equal(action, chunk[i]), f"step {i}"
        assert policy.predict_calls == 1

        # n+1-th call should trigger a new prediction.
        action = policy.select_action(_dummy_batch())
        assert policy.predict_calls == 2
        assert torch.equal(action, chunk[0])

    def test_reset_clears_queue(self) -> None:
        chunk = torch.arange(8, dtype=torch.float32).unsqueeze(-1)
        policy = _ChunkPolicy(chunk=chunk, n_action_steps=4)

        policy.select_action(_dummy_batch())
        assert len(policy._action_queue) == 3

        policy.reset()
        assert len(policy._action_queue) == 0
