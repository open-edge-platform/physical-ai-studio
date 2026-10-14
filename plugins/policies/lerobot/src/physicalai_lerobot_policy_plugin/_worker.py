# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Run LeRobot's native CLI inside the isolated worker environment."""

from __future__ import annotations

import importlib.metadata
import json
import os
import select
import sys
import threading
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

    from lerobot.configs.train import TrainPipelineConfig

_PROTOCOL_VERSION = 1
_NON_METRIC_FIELDS = frozenset({"steps", "samples", "episodes", "epochs"})
_CANCEL_EVENT = threading.Event()
_CANCEL_LISTENER_STOP = threading.Event()


def _send(message: dict[str, object]) -> None:
    payload = json.dumps(message, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
    file_descriptor = int(os.environ["PHYSICALAI_WORKER_PROTOCOL_FD"])
    while payload:
        payload = payload[os.write(file_descriptor, payload) :]


def _listen_for_cancellation() -> None:
    while not _CANCEL_LISTENER_STOP.is_set():
        try:
            readable, _, _ = select.select([sys.stdin], [], [], 0.1)
        except (OSError, ValueError):
            return
        if not readable:
            continue
        line = sys.stdin.buffer.readline()
        if not line:
            return
        try:
            control = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(control, dict) and control.get("type") == "cancel":
            _CANCEL_EVENT.set()
            return


class _Hooks:
    @staticmethod
    def should_stop() -> bool:
        return _CANCEL_EVENT.is_set()

    @staticmethod
    def on_step(step: int, total: int | None, metrics: Mapping[str, float]) -> None:
        _send({"type": "progress", "step": step, "total": total, "metrics": dict(metrics)})


class _WorkerCancelledError(Exception):
    """Internal signal raised after the host requests training cancellation."""

    def __init__(self, step: int | None) -> None:
        self.step = step


@contextmanager
def _capture_config(total_steps: list[int | None]) -> Iterator[list[TrainPipelineConfig]]:
    from lerobot.configs.train import TrainPipelineConfig  # ruff: ignore[import-outside-top-level]

    original_validate = TrainPipelineConfig.validate
    configs: list[TrainPipelineConfig] = []

    def capture(config: TrainPipelineConfig) -> None:
        original_validate(config)
        configs.append(config)
        steps = getattr(config, "steps", None)
        total_steps[0] = steps if isinstance(steps, int) and not isinstance(steps, bool) else None

    TrainPipelineConfig.validate = capture
    try:
        yield configs
    finally:
        TrainPipelineConfig.validate = original_validate


class _NativeTrainModule(Protocol):
    update_policy: Callable[..., object]


@contextmanager
def _report_steps(train_module: _NativeTrainModule, hooks: _Hooks, total_steps: list[int | None]) -> Iterator[None]:
    original_update = train_module.update_policy

    def report_update(*args: object, **kwargs: object) -> object:
        tracker = args[0] if args else kwargs.get("tracker")
        current_step = getattr(tracker, "steps", None)
        if hooks.should_stop():
            raise _WorkerCancelledError(current_step if isinstance(current_step, int) else None)

        result = original_update(*args, **kwargs)
        if not isinstance(result, tuple) or not result:
            msg = "LeRobot update_policy must return a non-empty tuple."
            raise TypeError(msg)
        updated_tracker = result[0]
        updated_step = getattr(updated_tracker, "steps", None)
        if not isinstance(updated_step, int):
            msg = "LeRobot update_policy returned a tracker without an integer 'steps' counter."
            raise TypeError(msg)
        step = updated_step + 1
        hooks.on_step(step, total_steps[0], _read_metrics(updated_tracker))
        if hooks.should_stop():
            raise _WorkerCancelledError(step)
        return result

    train_module.update_policy = report_update
    try:
        yield
    finally:
        train_module.update_policy = original_update


def _read_metrics(tracker: object) -> dict[str, float]:
    to_dict = getattr(tracker, "to_dict", None)
    if not callable(to_dict):
        msg = "LeRobot update tracker does not expose to_dict()."
        raise TypeError(msg)
    values = to_dict(use_avg=False)
    if not isinstance(values, dict):
        msg = "LeRobot tracker to_dict() must return a dictionary."
        raise TypeError(msg)
    return {
        name: float(value)
        for name, value in values.items()
        if isinstance(name, str)
        and name not in _NON_METRIC_FIELDS
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    }


def _train_native() -> dict[str, str | None]:
    from lerobot.scripts import lerobot_train  # ruff: ignore[import-outside-top-level]

    hooks = _Hooks()
    total_steps: list[int | None] = [None]
    with _capture_config(total_steps) as configs, _report_steps(lerobot_train, hooks, total_steps):
        lerobot_train.main()
    config = configs[-1] if configs else None
    output_dir = getattr(config, "output_dir", None)
    output_path = str(output_dir) if output_dir is not None else None
    checkpoint = None
    if output_dir is not None:
        candidate = Path(output_dir) / "checkpoints" / "last" / "pretrained_model"
        if candidate.is_dir():
            checkpoint = str(candidate)
    return {"output_dir": output_path, "checkpoint": checkpoint}


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
    _CANCEL_EVENT.clear()
    _CANCEL_LISTENER_STOP.clear()
    cancellation_listener = threading.Thread(target=_listen_for_cancellation, name="worker-control", daemon=True)
    cancellation_listener.start()
    original_argv = sys.argv
    sys.argv = ["lerobot-train", *arguments]
    try:
        result = _train_native()
    except _WorkerCancelledError as exc:
        _send({"type": "cancelled", "step": exc.step})
        return 130
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        if code == 0:
            _send({"type": "result", "output_dir": None, "checkpoint": None})
            return 0
        _send({"type": "error", "error": "SystemExit", "message": str(exc.code)})
        return code
    except Exception as exc:  # ruff: ignore[blind-except]  # Send native training failures over the worker protocol.
        _send({"type": "error", "error": type(exc).__name__, "message": str(exc)})
        traceback.print_exc(file=sys.stderr)
        return 1
    else:
        _send({"type": "result", **result})
        return 0
    finally:
        sys.argv = original_argv
        _CANCEL_LISTENER_STOP.set()
        cancellation_listener.join(timeout=1.0)


def main() -> int:
    """Execute LeRobot's native train command with the worker protocol enabled.

    Returns:
        LeRobot's exit code, or the worker cancellation status.
    """
    return _run(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
