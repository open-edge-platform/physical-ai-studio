# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Helpers shared by the GR00T worker scripts.

Workers run inside the pinned GR00T environments (Python 3.10 or 3.12), not in
the host. They import this module as a top-level module because the worker
directory is the script's ``sys.path[0]``.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import select
import sys
import threading
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = 1
_PROTOCOL_FD_ENV = "PHYSICALAI_WORKER_PROTOCOL_FD"
_PROTOCOL_PREFIX = bytes([30]) + b"physicalai-worker "
_PROJECT_ENV = "PHYSICALAI_WORKER_PROJECT"
_AUTO_MODULES = (
    "transformers.models.auto.configuration_auto",
    "transformers.models.auto.auto_factory",
    "transformers.models.auto.tokenization_auto",
    "transformers.models.auto.processing_auto",
    "transformers.models.auto.image_processing_auto",
    "transformers.models.auto.feature_extraction_auto",
    "transformers.models.auto.video_processing_auto",
)


class CancelledError(Exception):
    """Raised after the host asked the worker to stop."""

    def __init__(self, step: int | None) -> None:
        super().__init__(f"Cancelled at step {step}.")
        self.step = step


def source_dir() -> Path:
    """Return the Isaac-GR00T checkout this environment was built from.

    Core sets ``PHYSICALAI_WORKER_PROJECT`` for managed environments; users of
    their own interpreter set it themselves.

    Returns:
        The pinned (and possibly patched) checkout.

    Raises:
        RuntimeError: If the variable is missing.
    """
    configured = os.environ.get(_PROJECT_ENV)
    if not configured:
        msg = f"{_PROJECT_ENV} must name the Isaac-GR00T checkout of this environment."
        raise RuntimeError(msg)
    return Path(configured)


def identity() -> dict[str, Any]:
    """Describe this worker for the core handshake.

    Returns:
        Framework, package, and the installed ``gr00t`` version.
    """
    from importlib.metadata import version

    return {
        "framework": os.environ.get("PHYSICALAI_WORKER_FRAMEWORK", "gr00t"),
        "package": "gr00t",
        "version": version("gr00t"),
    }


class EventChannel:
    """Tagged JSON-lines event stream multiplexed onto worker stderr."""

    def __init__(self) -> None:
        """Bind to the protocol descriptor when the host provides one."""
        configured = os.environ.get(_PROTOCOL_FD_ENV)
        self._fd = int(configured) if configured else None

    @property
    def enabled(self) -> bool:
        """Whether a host is listening."""
        return self._fd is not None

    def send(self, message: dict[str, Any]) -> None:
        """Send one event; a no-op when no host is listening."""
        if self._fd is None:
            return
        payload = _PROTOCOL_PREFIX + (json.dumps(_finite(message), separators=(",", ":")) + "\n").encode("utf-8")
        while payload:
            payload = payload[os.write(self._fd, payload) :]


def start_cancel_listener() -> threading.Event:
    """Watch standard input for the host's cancellation request.

    Returns:
        An event that becomes set when the host requests cancellation.
    """
    cancelled = threading.Event()

    def listen() -> None:
        while not cancelled.is_set():
            try:
                readable, _, _ = select.select([sys.stdin], [], [], 0.2)
            except (OSError, ValueError):
                return
            if not readable:
                continue
            line = sys.stdin.buffer.readline()
            if not line:
                return
            try:
                control = json.loads(line)
            except ValueError:
                continue
            if isinstance(control, dict) and control.get("type") == "cancel":
                cancelled.set()

    threading.Thread(target=listen, name="gr00t-worker-cancel", daemon=True).start()
    return cancelled


def disable_remote_code() -> None:
    """Refuse to execute Python code downloaded with a model repository.

    Upstream GR00T requests ``trust_remote_code=True`` by default. The pinned
    model repositories contain no custom code, so this guard changes nothing for
    them; it prevents a different or modified repository from executing code.
    Every remote-code path in Transformers goes through
    ``get_class_from_dynamic_module``, which this replaces in each module that
    imported it.
    """
    dynamic = importlib.import_module("transformers.dynamic_module_utils")
    original = dynamic.__dict__.get("get_class_from_dynamic_module")
    for name in _AUTO_MODULES:
        try:
            importlib.import_module(name)
        except ImportError:
            continue

    def refuse(*_args: object, **_kwargs: object) -> None:
        msg = "Loading Python code from a model repository is disabled by the Physical AI GR00T plugin."
        raise RuntimeError(msg)

    for module in list(sys.modules.values()):
        namespace = getattr(module, "__dict__", None)
        if not isinstance(namespace, dict) or not str(namespace.get("__name__", "")).startswith("transformers"):
            continue
        if namespace.get("get_class_from_dynamic_module") is original:
            namespace["get_class_from_dynamic_module"] = refuse


def _finite(value: Any) -> Any:  # ruff: ignore[any-type]  # JSON-shaped values are dynamically typed.
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    return value
