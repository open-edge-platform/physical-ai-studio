# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Single-pass inference runner.

The simplest execution pattern: call the adapter once per inference step
and return the output dict unchanged.  This runner is completely generic
— it knows nothing about the domain (robotics, vision, etc.).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from physicalai.inference.runners.base import InferenceRunner

if TYPE_CHECKING:
    import numpy as np

    from physicalai.inference.adapters.base import RuntimeAdapter


class SinglePass(InferenceRunner):
    """Execute a single forward pass and return the adapter output.

    This is a pure passthrough runner: it calls ``adapter.predict()``
    and returns the result unchanged.  Any domain-specific transforms
    (key normalization, temporal squeezing, etc.) belong in
    postprocessors.

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
            Adapter output dict, returned as-is.
        """
        return dict(adapter.predict(inputs))

    def reset(self) -> None:
        """No-op — single-pass runner is stateless."""
