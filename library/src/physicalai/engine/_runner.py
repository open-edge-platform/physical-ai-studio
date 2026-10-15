# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared process runner for framework-owned worker scripts."""

from __future__ import annotations

import json
import logging
import os
import queue
import selectors
import subprocess  # ruff: ignore[suspicious-subprocess-import]  # Workers are launched with an argv list and shell=False.
import threading
import time
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any

from packaging.version import InvalidVersion, Version

from ._environment import interpreter_for, locked_version, supported_versions
from ._types import Run, TrainHooks, TrainingCancelledError, WorkerEnv

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import BinaryIO

logger = logging.getLogger(__name__)
_PROTOCOL_VERSION = 1
_STARTUP_TIMEOUT_SECONDS = 60.0
_CANCEL_GRACE_SECONDS = 5.0
_TERMINATE_GRACE_SECONDS = 5.0
_STREAM_CHUNK_ENCODING = "utf-8"


class WorkerProtocolError(RuntimeError):
    """Raised when a worker violates the host/worker protocol."""


class _NullHooks:
    @staticmethod
    def on_step(step: int, total: int | None, metrics: Mapping[str, float]) -> None:
        del step, total, metrics

    @staticmethod
    def should_stop() -> bool:
        return False


class _ProtocolEOF:
    pass


_PROTOCOL_EOF = _ProtocolEOF()


@dataclass
class _WorkerState:
    env: WorkerEnv
    hooks: TrainHooks
    expected_version: Version | None
    process: subprocess.Popen[bytes]
    events: queue.Queue[dict[str, Any] | BaseException | _ProtocolEOF]
    started_at: float = field(default_factory=time.monotonic)
    hello_received: bool = False
    protocol_closed: bool = False
    result: dict[str, Any] | None = None
    worker_error: tuple[str, str] | None = None
    cancelled: bool = False
    cancelled_step: int | None = None
    last_step: int | None = None
    cancellation_sent_at: float | None = None
    termination_sent_at: float | None = None
    threads: tuple[threading.Thread, ...] = ()
    protocol_reader_stop: threading.Event = field(default_factory=threading.Event)

    def consume(self, event: dict[str, Any] | BaseException | _ProtocolEOF) -> None:
        if event is _PROTOCOL_EOF:
            self.protocol_closed = True
            return
        if isinstance(event, BaseException):
            msg = "Engine worker sent an invalid protocol message."
            raise WorkerProtocolError(msg) from event
        if not isinstance(event, dict):
            msg = "Engine worker sent a non-object protocol event."
            raise WorkerProtocolError(msg)
        if not self.hello_received:
            _validate_hello(self.env, event, self.expected_version)
            self.hello_received = True
            return
        self._consume_message(event)

    def _consume_message(self, message: dict[str, Any]) -> None:
        message_type = message.get("type")
        if message_type == "progress":
            self.last_step = _report_progress(self.hooks, message)
        elif message_type == "result":
            if self.result is not None:
                msg = "Engine worker sent more than one result message."
                raise WorkerProtocolError(msg)
            self.result = message
        elif message_type == "error":
            self.worker_error = (_message_string(message, "error"), _message_string(message, "message"))
        elif message_type == "cancelled":
            self.cancelled = True
            self.cancelled_step = _optional_step(message.get("step"))
        else:
            msg = f"Engine worker sent unknown protocol message type {message_type!r}."
            raise WorkerProtocolError(msg)

    def check_control(self, now: float) -> None:
        if not self.hello_received and now - self.started_at > _STARTUP_TIMEOUT_SECONDS and self.process.poll() is None:
            msg = f"Engine worker for {self.env.framework!r} did not complete its version handshake."
            raise TimeoutError(msg)
        if self.hello_received and self.cancellation_sent_at is None and self.hooks.should_stop():
            _send_cancel(self.process)
            self.cancellation_sent_at = now
        if self.cancellation_sent_at is None or self.process.poll() is not None:
            return
        if self.termination_sent_at is None and now - self.cancellation_sent_at >= _CANCEL_GRACE_SECONDS:
            logger.warning("Stopping unresponsive %s worker after cancellation grace period.", self.env.framework)
            self.process.terminate()
            self.termination_sent_at = now
        elif self.termination_sent_at is not None and now - self.termination_sent_at >= _TERMINATE_GRACE_SECONDS:
            self.process.kill()
            self.termination_sent_at = float("inf")


def run_worker(
    env: WorkerEnv,
    arguments: Sequence[str],
    *,
    device: str,
    hooks: TrainHooks | None = None,
    accelerator: str = "auto",
    interpreter: str | Path | None = None,
    expected_version: str | Version | None = None,
) -> Run:
    """Run a framework worker with JSON-lines progress and cancellation.

    The framework's stdout and stderr are drained into the host logger. Worker
    control and progress use a separate inherited pipe, so native framework
    output never has to be rewritten or multiplexed with protocol messages.

    Returns:
        A :class:`Run` with the worker's native output and checkpoint paths.

    Raises:
        NotImplementedError: If the host cannot pass the protocol file descriptor.
        TypeError: If any worker argument is not a string.
    """
    if not all(isinstance(argument, str) for argument in arguments):
        msg = "Worker arguments must be strings."
        raise TypeError(msg)
    if os.name == "nt":
        msg = "Managed engine workers currently require an OS that supports inherited file descriptors."
        raise NotImplementedError(msg)

    python = interpreter_for(env, accelerator=accelerator, interpreter=interpreter)
    if expected_version is None and interpreter is None:
        expected_version = locked_version(env)
    expected = Version(str(expected_version)) if expected_version is not None else None

    with resources.as_file(env.worker) as worker_path:
        command = [str(python), str(worker_path), *arguments]
        return _run_process(env, command, device=device, hooks=hooks, expected_version=expected)


def _run_process(
    env: WorkerEnv,
    command: list[str],
    *,
    device: str,
    hooks: TrainHooks | None,
    expected_version: Version | None,
) -> Run:
    state = _start_worker(env, command, device, hooks, expected_version)
    try:
        return_code = _wait_for_worker(state)
    finally:
        if state.process.poll() is None:
            _stop_process(state.process)
        _close_worker(state)
    return _worker_result(state, return_code)


def _start_worker(
    engine_env: WorkerEnv,
    command: list[str],
    device: str,
    hooks: TrainHooks | None,
    expected_version: Version | None,
) -> _WorkerState:
    read_fd, write_fd = os.pipe()
    inherit_protocol_fd = True
    os.set_inheritable(write_fd, inherit_protocol_fd)
    child_env = {
        **os.environ,
        "PHYSICALAI_WORKER_PROTOCOL_FD": str(write_fd),
        "PHYSICALAI_WORKER_FRAMEWORK": engine_env.framework,
        "PHYSICALAI_WORKER_PACKAGE": engine_env.package,
        "PHYSICALAI_WORKER_DEVICE": device,
    }
    try:
        process = subprocess.Popen(  # ruff: ignore[subprocess-without-shell-equals-true]  # argv is intentionally forwarded; shell execution is disabled.
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=child_env,
            pass_fds=(write_fd,),
            shell=False,
        )
    except BaseException:
        os.close(read_fd)
        os.close(write_fd)
        raise
    os.close(write_fd)

    events: queue.Queue[dict[str, Any] | BaseException | _ProtocolEOF] = queue.Queue()
    state = _WorkerState(
        env=engine_env,
        hooks=hooks if hooks is not None else _NullHooks(),
        expected_version=expected_version,
        process=process,
        events=events,
    )
    protocol_thread = threading.Thread(
        target=_read_protocol,
        args=(read_fd, events, state.protocol_reader_stop),
        name=f"{engine_env.framework}-worker-protocol",
        daemon=True,
    )
    stdout_thread = threading.Thread(
        target=_forward_stream,
        args=(process.stdout, logging.INFO, f"{engine_env.framework} worker stdout"),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_forward_stream,
        args=(process.stderr, logging.INFO, f"{engine_env.framework} worker stderr"),
        daemon=True,
    )
    state.threads = (protocol_thread, stdout_thread, stderr_thread)
    protocol_thread.start()
    stdout_thread.start()
    stderr_thread.start()
    return state


def _wait_for_worker(state: _WorkerState) -> int:
    while True:
        try:
            event = state.events.get(timeout=0.05)
        except queue.Empty:
            event = None
        if event is not None:
            state.consume(event)
        state.check_control(time.monotonic())
        terminal_message_received = state.result is not None or state.worker_error is not None or state.cancelled
        return_code = state.process.poll()
        if (
            return_code is not None
            and (state.protocol_closed or terminal_message_received or return_code != 0)
            and state.events.empty()
        ):
            return state.process.wait()


def _close_worker(state: _WorkerState) -> None:
    state.protocol_reader_stop.set()
    if state.process.stdin is not None:
        state.process.stdin.close()
    for stream in (state.process.stdout, state.process.stderr):
        if stream is not None:
            stream.close()
    for thread in state.threads:
        thread.join(timeout=1.0)


def _worker_result(state: _WorkerState, return_code: int) -> Run:
    if not state.hello_received:
        msg = f"{state.env.framework} worker exited before completing its version handshake."
        raise WorkerProtocolError(msg)
    if state.cancelled or (
        state.cancellation_sent_at is not None and state.result is None and state.worker_error is None
    ):
        raise TrainingCancelledError(state.cancelled_step if state.cancelled_step is not None else state.last_step)
    if state.worker_error is not None:
        error_type, error_message = state.worker_error
        msg = f"{state.env.framework} worker raised {error_type}: {error_message}"
        raise RuntimeError(msg)
    if return_code != 0:
        msg = f"{state.env.framework} worker exited with status {return_code}."
        raise RuntimeError(msg)
    if state.result is None:
        return Run(framework=state.env.framework, output_dir=None, checkpoint=None)
    return Run(
        framework=state.env.framework,
        output_dir=_optional_path(state.result, "output_dir"),
        checkpoint=_optional_path(state.result, "checkpoint"),
    )


def _read_protocol(
    file_descriptor: int,
    events: queue.Queue[dict[str, Any] | BaseException | _ProtocolEOF],
    stop: threading.Event,
) -> None:
    pending = bytearray()
    with selectors.DefaultSelector() as selector:
        selector.register(file_descriptor, selectors.EVENT_READ)
        try:
            while not stop.is_set():
                if not selector.select(timeout=0.1):
                    continue
                chunk = os.read(file_descriptor, 8192)
                if not chunk:
                    break
                pending.extend(chunk)
                while b"\n" in pending:
                    line, _, rest = pending.partition(b"\n")
                    pending = bytearray(rest)
                    if not _enqueue_protocol_line(bytes(line), events):
                        return
            if pending:
                _enqueue_protocol_line(bytes(pending), events)
        except OSError as exc:
            events.put(exc)
        finally:
            os.close(file_descriptor)
            events.put(_PROTOCOL_EOF)


def _enqueue_protocol_line(
    line: bytes,
    events: queue.Queue[dict[str, Any] | BaseException | _ProtocolEOF],
) -> bool:
    try:
        message = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        events.put(exc)
        return False
    if not isinstance(message, dict):
        events.put(TypeError("Protocol lines must contain JSON objects."))
        return False
    events.put(message)
    return True


def _forward_stream(stream: BinaryIO | None, level: int, label: str) -> None:
    if stream is None:
        return
    for line in iter(stream.readline, b""):
        text = line.decode(_STREAM_CHUNK_ENCODING, errors="replace").rstrip("\r\n")
        if text:
            logger.log(level, "%s: %s", label, text)


def _validate_hello(env: WorkerEnv, message: dict[str, Any], expected_version: Version | None) -> None:
    if message.get("type") != "hello":
        msg = "The first protocol message must be a hello handshake."
        raise WorkerProtocolError(msg)
    if message.get("protocol") != _PROTOCOL_VERSION:
        msg = f"Unsupported engine worker protocol {message.get('protocol')!r}."
        raise WorkerProtocolError(msg)
    if message.get("framework") != env.framework:
        msg = f"Worker framework {message.get('framework')!r} does not match {env.framework!r}."
        raise WorkerProtocolError(msg)
    package = message.get("package")
    if not isinstance(package, str) or _normalize_name(package) != _normalize_name(env.package):
        msg = f"Worker reported an unexpected package {package!r}."
        raise WorkerProtocolError(msg)
    raw_version = message.get("version")
    if not isinstance(raw_version, str):
        msg = "Worker handshake is missing a package version."
        raise TypeError(msg)
    try:
        version = Version(raw_version)
    except InvalidVersion as exc:
        msg = f"Worker reported an invalid {env.package} version {raw_version!r}."
        raise WorkerProtocolError(msg) from exc

    supported = supported_versions(env)
    if version not in supported:
        msg = f"Worker {env.package} version {version} is outside the supported range {supported}."
        raise WorkerProtocolError(msg)
    if expected_version is not None and version != expected_version:
        msg = f"Worker {env.package} version {version} does not match the required version {expected_version}."
        raise WorkerProtocolError(msg)


def _report_progress(hooks: TrainHooks, message: dict[str, Any]) -> int:
    step = _required_step(message.get("step"))
    total = _optional_step(message.get("total"))
    raw_metrics = message.get("metrics")
    if not isinstance(raw_metrics, dict):
        msg = "Worker progress metrics must be a JSON object."
        raise TypeError(msg)
    metrics: dict[str, float] = {}
    for name, value in raw_metrics.items():
        if not isinstance(name, str) or isinstance(value, bool) or not isinstance(value, (int, float)):
            msg = "Worker progress metrics must map names to numeric values."
            raise TypeError(msg)
        metrics[name] = float(value)
    hooks.on_step(step, total, metrics)
    return step


def _send_cancel(process: subprocess.Popen[bytes]) -> None:
    if process.stdin is None:
        return
    try:
        process.stdin.write(b'{"type":"cancel"}\n')
        process.stdin.flush()
    except BrokenPipeError:
        logger.debug("Worker closed its control pipe before cancellation was delivered.")


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=_TERMINATE_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _required_step(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        msg = f"Worker reported an invalid step value {value!r}."
        raise TypeError(msg)
    return value


def _optional_step(value: object) -> int | None:
    if value is None:
        return None
    return _required_step(value)


def _message_string(message: dict[str, Any], key: str) -> str:
    value = message.get(key)
    if not isinstance(value, str):
        msg = f"Worker error message has an invalid {key!r} field."
        raise TypeError(msg)
    return value


def _optional_path(message: dict[str, Any], key: str) -> Path | None:
    value = message.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        msg = f"Worker result has an invalid {key!r} path."
        raise TypeError(msg)
    return Path(value)


def _normalize_name(value: str) -> str:
    return value.replace("_", "-").lower()


__all__ = ["WorkerProtocolError", "run_worker"]
