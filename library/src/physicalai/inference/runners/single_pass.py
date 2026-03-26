# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Single-pass inference runner.

The simplest execution pattern: call the adapter once per inference step
and return the resulting output dict with a canonical ``"action"`` key.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, override

import numpy as np

from physicalai.inference.runners.base import InferenceRunner

if TYPE_CHECKING:
    from physicalai.inference.adapters.base import RuntimeAdapter

_NDIM_WITH_TEMPORAL = 3


class SinglePass(InferenceRunner):
    """Execute a single forward pass and return the adapter output.

    Handles the common case where ``adapter.predict()`` returns an output
    dict whose primary action tensor has an optional temporal dimension of
    size 1 that needs to be squeezed away.

    This runner is stateless — ``reset()`` is a no-op.
    """

    @override
    def run(
        self,
        adapter: RuntimeAdapter,
        inputs: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Run a single forward pass through the adapter.

        Args:
            adapter: The loaded runtime adapter.
            inputs: Pre-processed model inputs.

        Returns:
            Adapter output dict with the primary action stored under
            ``"action"``. If the action tensor has a temporal dimension
            of size 1, it is squeezed.
        """
        outputs = dict(adapter.predict(inputs))
        if "action" in outputs:
            actions: np.ndarray = outputs["action"]
        else:
            raw_action_key = adapter.output_names[0] if adapter.output_names else next(iter(outputs))
            actions = outputs.pop(raw_action_key)

        if actions.ndim == _NDIM_WITH_TEMPORAL and actions.shape[1] == 1:
            actions = np.squeeze(actions, axis=1)

        outputs["action"] = actions
        return outputs

    def reset(self) -> None:
        """No-op — single-pass runner is stateless."""
