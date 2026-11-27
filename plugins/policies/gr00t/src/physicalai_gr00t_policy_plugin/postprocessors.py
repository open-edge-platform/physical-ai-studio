# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Runtime postprocessors that GR00T packages declare in their manifest."""

from __future__ import annotations

from typing import override

import numpy as np
from physicalai.inference.constants import ACTION
from physicalai.inference.postprocessors.base import Postprocessor


class BinaryGripperPostprocessor(Postprocessor):
    """Turn one action column's gripper prediction into a binary open or close command.

    Predictions above *threshold* become *open_command*, those below it *close_command*, and one
    exactly at it the midpoint of the two, as NVIDIA's LIBERO evaluation does with ``sign()``.

    Args:
        index: Column of the gripper in the flat Runtime action.
        threshold: Boundary between closed and open predictions.
        open_command: Command sent for an open prediction.
        close_command: Command sent for a closed prediction.

    Examples:
        >>> gripper = BinaryGripperPostprocessor(index=1, threshold=0.5, open_command=-1.0, close_command=1.0)
        >>> gripper({"action": np.array([[[0.3, 0.9], [0.3, 0.2]]])})["action"]
        array([[[ 0.3, -1. ],
                [ 0.3,  1. ]]])
    """

    def __init__(self, index: int, threshold: float, open_command: float, close_command: float) -> None:
        """Configure the column and the two commands."""
        self._index = index
        self._threshold = threshold
        self._open = open_command
        self._close = close_command

    @override
    def __call__(self, outputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        actions = np.array(outputs[ACTION], copy=True)
        value = actions[..., self._index]
        midpoint = (self._open + self._close) / 2.0
        actions[..., self._index] = np.where(
            value > self._threshold,
            self._open,
            np.where(value < self._threshold, self._close, midpoint),
        )
        outputs[ACTION] = actions
        return outputs

    def __repr__(self) -> str:
        """Return the configuration."""
        return (
            f"{type(self).__name__}(index={self._index}, threshold={self._threshold}, "
            f"open_command={self._open}, close_command={self._close})"
        )
