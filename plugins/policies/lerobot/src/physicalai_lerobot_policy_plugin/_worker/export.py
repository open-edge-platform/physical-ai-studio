# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Run ``lerobot-export`` inside the isolated export worker environment."""

from __future__ import annotations

import importlib.metadata
import json
import os
import select
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

_PROTOCOL_VERSION = 1
_PROTOCOL_PREFIX = bytes([30]) + b"physicalai-worker "
_ACK_TIMEOUT_SECONDS = 60.0
_ENABLEMENT_OV_DIR_ENV = "PHYSICALAI_LEROBOT_ENABLEMENT_OV_DIR"
_ENABLEMENT_OV_PACKAGE = "lerobot_openvino_enablement"


def _send(message: dict[str, object]) -> None:
    payload = _PROTOCOL_PREFIX + json.dumps(message, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
    file_descriptor = int(os.environ["PHYSICALAI_WORKER_PROTOCOL_FD"])
    while payload:
        payload = payload[os.write(file_descriptor, payload) :]


def _wait_for_host_acknowledgement() -> str:
    """Block until the host accepts the handshake, so a rejected worker never starts work."""
    deadline = time.monotonic() + _ACK_TIMEOUT_SECONDS
    while (remaining := deadline - time.monotonic()) > 0:
        try:
            readable, _, _ = select.select([sys.stdin], [], [], min(remaining, 0.5))
        except (OSError, ValueError):
            return "closed"
        if not readable:
            continue
        line = sys.stdin.buffer.readline()
        if not line:
            return "closed"
        try:
            control = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(control, dict) and control.get("type") in {"ack", "cancel"}:
            return str(control["type"])
    return "closed"


def _stage_openvino_enablement(staging_root: Path) -> str | None:
    """Copy the bundled OpenVINO enablement into an importable package directory."""
    source_raw = os.environ.get(_ENABLEMENT_OV_DIR_ENV)
    if not source_raw:
        return None
    source = Path(source_raw).expanduser().resolve()
    if not source.is_dir():
        msg = f"OpenVINO enablement directory not found: {source}"
        raise FileNotFoundError(msg)
    for item in source.rglob("*"):
        if item.is_symlink():
            msg = f"OpenVINO enablement contains symbolic link: {item}"
            raise ValueError(msg)
    destination = staging_root / _ENABLEMENT_OV_PACKAGE
    shutil.copytree(source, destination)
    return str(staging_root)


def _needs_openvino_plugin(arguments: list[str]) -> bool:
    for index, arg in enumerate(arguments):
        if arg == "--backend.type=openvino":
            return True
        if arg == "--backend.type" and index + 1 < len(arguments) and arguments[index + 1] == "openvino":
            return True
    return False


def _has_backend_plugin_arg(arguments: list[str]) -> bool:
    return any(
        arg == "--backend.discover_packages_path" or arg.startswith("--backend.discover_packages_path=")
        for arg in arguments
    )


def _run_export(arguments: list[str]) -> Path:
    from lerobot.scripts import lerobot_export
    from lerobot.utils import init_logging

    init_logging()
    export_args = list(arguments)
    previous_pythonpath = os.environ.get("PYTHONPATH")
    added_sys_path: str | None = None
    with tempfile.TemporaryDirectory(prefix="physicalai-lerobot-enablement-") as staging:
        staged_parent = _stage_openvino_enablement(Path(staging))
        if staged_parent is not None:
            entries = [staged_parent]
            if previous_pythonpath:
                entries.append(previous_pythonpath)
            os.environ["PYTHONPATH"] = os.pathsep.join(entries)
            sys.path.insert(0, staged_parent)
            added_sys_path = staged_parent
            if _needs_openvino_plugin(export_args) and not _has_backend_plugin_arg(export_args):
                export_args.append(f"--backend.discover_packages_path={_ENABLEMENT_OV_PACKAGE}")
        try:
            return lerobot_export.export(export_args)
        finally:
            if previous_pythonpath is None:
                os.environ.pop("PYTHONPATH", None)
            else:
                os.environ["PYTHONPATH"] = previous_pythonpath
            if added_sys_path is not None and added_sys_path in sys.path:
                sys.path.remove(added_sys_path)


def _run(arguments: list[str]) -> int:
    framework = os.environ["PHYSICALAI_WORKER_FRAMEWORK"]
    package = os.environ["PHYSICALAI_WORKER_PACKAGE"]
    _send(
        {
            "type": "hello",
            "protocol": _PROTOCOL_VERSION,
            "framework": framework,
            "package": package,
            "version": importlib.metadata.version(package),
        },
    )
    answer = _wait_for_host_acknowledgement()
    if answer == "cancel":
        _send({"type": "cancelled", "step": None})
        return 130
    if answer != "ack":
        _send({"type": "error", "error": "WorkerHandshakeError", "message": "The host did not accept the handshake."})
        return 1

    try:
        bundle_dir = _run_export(arguments)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        if code == 0:
            _send({"type": "result", "output_dir": None, "checkpoint": None})
            return 0
        _send({"type": "error", "error": "SystemExit", "message": str(exc.code)})
        return code
    except Exception as exc:  # noqa: BLE001  # Send export failures over the worker protocol.
        _send({"type": "error", "error": type(exc).__name__, "message": str(exc)})
        traceback.print_exc(file=sys.stderr)
        return 1
    else:
        _send({"type": "result", "output_dir": str(bundle_dir), "checkpoint": str(bundle_dir)})
        return 0


def main() -> int:
    """Execute ``lerobot-export`` with the host/worker protocol enabled."""
    return _run(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
