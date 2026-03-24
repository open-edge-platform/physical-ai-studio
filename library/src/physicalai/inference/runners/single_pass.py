# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Single-pass inference runner.

The simplest execution pattern: call the adapter once per inference step
and return the result directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, override

import numpy as np

from physicalai.inference.runners.base import InferenceRunner

if TYPE_CHECKING:
    from physicalai.inference.adapters.base import RuntimeAdapter

_NDIM_WITH_TEMPORAL = 3


class SinglePass(InferenceRunner):
    """Execute a single forward pass and return the action directly.

    Handles the common case where ``adapter.predict()`` returns an action
    tensor with an optional temporal dimension of size 1 that needs to be
    squeezed away.

    This runner is stateless — ``reset()`` is a no-op.
    """

    @override
    def run(
        self,
        adapter: RuntimeAdapter,
        inputs: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Run a single forward pass through the adapter.

        Args:
            adapter: The loaded runtime adapter.
            inputs: Pre-processed model inputs.

        Returns:
            Action array with shape ``(batch_size, action_dim)``. If the
            raw output has a temporal dimension of size 1, it is squeezed.
        """
        outputs = adapter.predict(inputs)

        action_key = _get_action_output_key(outputs)
        actions: np.ndarray = outputs[action_key]

        if actions.ndim == _NDIM_WITH_TEMPORAL and actions.shape[1] == 1:
            actions = np.squeeze(actions, axis=1)

        return actions

    def reset(self) -> None:
        """No-op — single-pass runner is stateless."""


def _get_action_output_key(outputs: dict[str, np.ndarray]) -> str:
    """Determine which output key contains the actions.

    Args:
        outputs: Model output dictionary.

    Returns:
        The key containing action data.
    """
    for key in ("actions", "action", "output", "pred_actions"):
        if key in outputs:
            return key
    return next(iter(outputs))
