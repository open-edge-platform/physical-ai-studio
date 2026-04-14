# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Runner factory for selecting inference runners from manifest or metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from physicalai.inference.runners.action_chunking import ActionChunking
from physicalai.inference.runners.single_pass import SinglePass

if TYPE_CHECKING:
    from physicalai.inference.manifest import Manifest
    from physicalai.inference.runners.base import InferenceRunner


def get_runner(source: Manifest | dict[str, Any]) -> InferenceRunner:
    """Select and instantiate a runner from a manifest or legacy metadata.

    Supports three formats:

    1. **Manifest object** — reads ``source.model.runner`` and
       instantiates via :func:`instantiate_component`.
    2. **Dict with runner spec** — raw manifest dict containing a
       ``"model"`` section with a runner component spec.
    3. **Legacy dict** — falls back to flat ``use_action_queue``
       and ``chunk_size`` keys.

    Args:
        source: A :class:`Manifest` instance or a raw metadata dict.

    Returns:
        Configured runner instance.
    """
    from physicalai.inference.manifest import Manifest  # noqa: PLC0415

    if isinstance(source, Manifest):
        if source.model.runner is not None:
            from physicalai.inference.component_factory import instantiate_component  # noqa: PLC0415

            return instantiate_component(source.model.runner)
        return SinglePass()

    runner_spec = _extract_runner_spec(source)
    if runner_spec is not None:
        from physicalai.inference.component_factory import instantiate_component  # noqa: PLC0415
        from physicalai.inference.manifest import ComponentSpec  # noqa: PLC0415

        return instantiate_component(ComponentSpec.model_validate(runner_spec))

    if source.get("use_action_queue"):
        chunk_size = source.get("chunk_size", 1)
        return ActionChunking(runner=SinglePass(), chunk_size=chunk_size)
    return SinglePass()


def _extract_runner_spec(metadata: dict[str, Any]) -> dict[str, Any] | None:
    """Extract a runner spec dict from nested or flat metadata.

    Returns:
        Runner spec dict if found, otherwise ``None``.
    """
    model_section = metadata.get("model", {})
    if isinstance(model_section, dict):
        runner = model_section.get("runner")
        if isinstance(runner, dict) and ("class_path" in runner or "type" in runner):
            return runner

    runner = metadata.get("runner")
    if isinstance(runner, dict) and ("class_path" in runner or "type" in runner):
        return runner

    return None
