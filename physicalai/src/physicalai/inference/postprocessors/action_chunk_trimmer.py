# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Action chunk trimmer postprocessor.

Some policies are trained with longer chunk, than it's used un inference.
Training with a longer chunk makes action sequence smoother, but
on inference tail of a long chunk is mostly useless.
This postprocessor trims action chunk to a specified length,
following common `n_action_steps` notation from policies configs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from physicalai.inference.postprocessors.base import Postprocessor

if TYPE_CHECKING:
    import numpy as np

_NDIM_WITH_TEMPORAL = 3


class ActionChunkTrimmer(Postprocessor):
    """Trim action chunk to a specified length.

    Args:
        action_key: Explicit adapter output key to treat as the action.
            When ``None`` (default), uses the first key if ``"action"``
            is not already present.

    Examples:
        >>> trimmer = ActionChunkTrimmer(n_action_steps=10)
        >>> trimmer({"actions": np.zeros((1, 50, 6))}).shape
        [1, 10, 6]
    """

    def __init__(self, n_action_steps: int) -> None:
        """Initialize with the number of action steps.

        Args:
            n_action_steps: Number of action steps to trim the action chunk to.
        """
        self._n_action_steps = n_action_steps

    @override
    def __call__(self, actions: np.ndarray) -> np.ndarray:
        if actions.ndim == _NDIM_WITH_TEMPORAL and actions.shape[1] > self._n_action_steps:
            actions = actions[:, : self._n_action_steps, :]
        return actions

    def __repr__(self) -> str:
        """Return string representation."""
        return f"{self.__class__.__name__}(n_action_steps={self._n_action_steps})"
