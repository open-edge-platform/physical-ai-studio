# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Action normalization postprocessor.

Maps adapter-specific output keys to a canonical ``"action"`` key and
squeezes a temporal dimension of size 1 when present.  This is the
domain-specific bridge between generic inference runners and the
PhysicalAI ``select_action()`` API.
"""

from __future__ import annotations

from typing import override

import numpy as np

from physicalai.inference.constants import ACTION
from physicalai.inference.postprocessors.base import Postprocessor

_NDIM_WITH_TEMPORAL = 3


class ActionNormalizer(Postprocessor):
    """Normalize runner output to a canonical ``"action"`` key.

    Adapters may return action tensors under varying keys (``"actions"``,
    ``"pred_actions"``, ``"output"``, etc.).  This postprocessor finds
    the action tensor, renames it to ``"action"``, and squeezes an
    optional temporal dimension of size 1.

    Key resolution order:

    1. ``"action"`` — already canonical, no rename needed.
    2. Explicit ``action_key`` passed to the constructor.
    3. First key in the output dict (fallback heuristic).

    Args:
        action_key: Explicit adapter output key to treat as the action.
            When ``None`` (default), uses the first key if ``"action"``
            is not already present.

    Examples:
        Default auto-detection:

        >>> normalizer = ActionNormalizer()
        >>> normalizer({"actions": np.zeros((1, 4))})
        {"action": array([[0., 0., 0., 0.]])}

        Explicit key:

        >>> normalizer = ActionNormalizer(action_key="pred_actions")
        >>> normalizer({"pred_actions": np.zeros((1, 4)), "extra": ...})
        {"action": array([[0., 0., 0., 0.]]), "extra": ...}
    """

    def __init__(self, action_key: str | None = None) -> None:
        """Initialize with an optional explicit action key.

        Args:
            action_key: Adapter output key to treat as the action tensor.
                When ``None``, falls back to ``"action"`` or the first key.
        """
        self._action_key = action_key

    @override
    def __call__(self, outputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        outputs = dict(outputs)

        if ACTION in outputs:
            action_key = ACTION
        elif self._action_key is not None:
            action_key = self._action_key
        else:
            action_key = next(iter(outputs))

        actions = outputs.pop(action_key)

        if actions.ndim == _NDIM_WITH_TEMPORAL and actions.shape[1] == 1:
            actions = np.squeeze(actions, axis=1)

        outputs[ACTION] = actions
        return outputs

    def __repr__(self) -> str:
        """Return string representation."""
        if self._action_key is not None:
            return f"{self.__class__.__name__}(action_key={self._action_key!r})"
        return f"{self.__class__.__name__}()"
