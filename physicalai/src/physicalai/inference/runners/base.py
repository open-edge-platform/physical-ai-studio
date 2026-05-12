# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base inference runner interface.

Runners encapsulate *how* inference runs — the execution procedure —
separate from *what* runs (the adapter/backend) and *where* (the device).

A runner is a composable unit: runners can wrap other runners to add
behavior (e.g. action chunking wraps any base runner to add temporal
buffering — the GoF Decorator pattern).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from physicalai.inference.adapters.base import RuntimeAdapter


class InferenceRunner(ABC):
    """Abstract base class for inference runners.

    A runner defines *how* a single call to the inference model
    translates into one or more adapter calls and how outputs are
    post-processed.

    Runners are composable: a runner can wrap another runner to add
    behavior without modifying the inner runner's logic. For example,
    ``ActionChunking(SinglePass())`` adds temporal buffering around
    a single forward pass.

    Subclasses must implement:
    - ``run`` — execute inference given an adapter and prepared inputs
    - ``reset`` — clear any internal state between episodes

    Examples:
        >>> runner = SinglePass()
        >>> outputs = runner.run(adapter, inputs)
        >>> action = outputs["action"]
        >>> runner.reset()  # new episode
    """

    @abstractmethod
    def run(
        self,
        adapter: RuntimeAdapter,
        inputs: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Execute one inference step and return model outputs.

        Args:
            adapter: The loaded runtime adapter to call ``predict`` on.
            inputs: Pre-processed model inputs (flat dict of numpy arrays).

        Returns:
            Dict mapping output names to numpy arrays.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state for a new episode.

        Runners that maintain state between calls (e.g. action queues)
        must clear it here. Stateless runners can no-op.
        """

    def __repr__(self) -> str:
        """Return string representation of the runner."""
        return f"{self.__class__.__name__}()"
