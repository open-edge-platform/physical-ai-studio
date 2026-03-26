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
    """Select and instantiate a runner from export metadata or manifest.

    Supports two formats:

    1. **Manifest** (``manifest.json``): If ``metadata`` contains a
       ``"runner"`` key with ``class_path`` + ``init_args``, the runner
       is instantiated dynamically via :class:`ComponentSpec`.
    2. **Legacy** (``metadata.yaml``): Falls back to reading flat keys
       ``use_action_queue`` and ``chunk_size``.

    Args:
        metadata: Export metadata dict (from ``metadata.yaml`` or
            ``manifest.json``).

    Returns:
        Configured runner instance.
    """
    runner_spec = metadata.get("runner")
    if isinstance(runner_spec, dict) and "class_path" in runner_spec:
        from physicalai.inference.component_factory import instantiate_component  # noqa: PLC0415
        from physicalai.inference.manifest import ComponentSpec  # noqa: PLC0415

        return instantiate_component(ComponentSpec.model_validate(runner_spec))

    if metadata.get("use_action_queue"):
        chunk_size = metadata.get("chunk_size", 1)
        return ActionChunking(runner=SinglePass(), chunk_size=chunk_size)
    return SinglePass()
