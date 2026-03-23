# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Runner factory for selecting inference runners from metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from physicalai.inference.runners.action_chunking import ActionChunking
from physicalai.inference.runners.single_pass import SinglePass

if TYPE_CHECKING:
    from physicalai.inference.runners.base import InferenceRunner


def get_runner(metadata: dict[str, Any]) -> InferenceRunner:
    """Select and instantiate a runner from export metadata.

    Args:
        metadata: Export metadata dict (from ``metadata.yaml`` or ``manifest.json``).

    Returns:
        Configured runner instance. Returns ``ActionChunking(SinglePass())``
        when ``use_action_queue`` is truthy, ``SinglePass()`` otherwise.
    """
    if metadata.get("use_action_queue"):
        chunk_size = metadata.get("chunk_size", 1)
        return ActionChunking(runner=SinglePass(), chunk_size=chunk_size)
    return SinglePass()
