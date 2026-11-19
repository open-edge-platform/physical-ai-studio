# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared process runner for framework-owned worker scripts."""

from __future__ import annotations

import json
import logging
import os
import queue
import selectors
import subprocess  # ruff: ignore[suspicious-subprocess-import]  # Unmanaged workers use argv lists with shell=False.
import threading
import time
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any

from packaging.version import InvalidVersion, Version

from physicalai import isolate

from ._environment import locked_version, prepare_environment, supported_versions
from ._types import Run, TrainHooks, TrainingCancelledError, WorkerEnv

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from importlib.resources.abc import Traversable
    from typing import BinaryIO

logger = logging.getLogger(__name__)
_PROTOCOL_VERSION = 1
_PROTOCOL_PREFIX = bytes([30]) + b"physicalai-worker "
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
    process: isolate.Session | subprocess.Popen[bytes]
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
    worker: Path | Traversable | None = None,
    extra_env: Mapping[str, str] | None = None,
) -> Run:
    """Run a framework worker with JSON-lines progress and cancellation.

    The framework's stdout and stderr are drained into the host logger. Worker
    events are tagged on stderr so the runner can use Runtime's portable process
    API without requiring an inherited file descriptor.

    A managed environment is prepared on demand; its exact locked package
    version is then required, and ``PHYSICALAI_WORKER_PROJECT`` tells the worker
    where the project it was built from lives (for example, a framework checkout).

    Args:
        env: Worker environment.
        arguments: Arguments passed to the worker script.
        device: Device string exposed to the worker.
        hooks: Progress and cancellation hooks.
        accelerator: Accelerator profile of a managed environment.
        interpreter: User-managed interpreter that bypasses provisioning.
        expected_version: Exact package version the worker must report.
        worker: Worker script to run instead of ``env.worker``.
        extra_env: Additional environment variables for the worker.

    Returns:
        A :class:`Run` with the worker's native output and checkpoint paths.

    Raises:
        TypeError: If any worker argument is not a string.
    """
    if not all(isinstance(argument, str) for argument in arguments):
        msg = "Worker arguments must be strings."
        raise TypeError(msg)
    if os.name == "nt":
        msg = "Managed engine workers currently require POSIX pipe semantics for cancellation."
        raise NotImplementedError(msg)

    variables = dict(extra_env or {})
    entrypoint = worker if worker is not None else env.worker
    isolate_prepared: isolate.PreparedEnvironment | None = None
    if interpreter is None:
        prepared = prepare_environment(env, accelerator)
        isolate_prepared = prepared.isolate
        variables["PHYSICALAI_WORKER_PROJECT"] = str(prepared.project)
        if expected_version is None:
            expected_version = locked_version(env, prepared.project)
    else:
        python = Path(interpreter).expanduser()
    expected = Version(str(expected_version)) if expected_version is not None else None

    if isolate_prepared is not None:

        def launch_managed(child_env: Mapping[str, str]) -> isolate.Session:
            return isolate.start(
                isolate_prepared,
                entrypoint=entrypoint,
                args=arguments,
                env=child_env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

        return _run_process(
            env,
            launch_managed,
            device=device,
            hooks=hooks,
            expected_version=expected,
            variables=variables,
        )

    with resources.as_file(entrypoint) as entrypoint_path:

        def launch_unmanaged(child_env: Mapping[str, str]) -> subprocess.Popen[bytes]:
            return subprocess.Popen(  # ruff: ignore[subprocess-without-shell-equals-true]  # Explicit argv, no shell.
                [str(python), str(entrypoint_path), *arguments],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=dict(child_env),
                shell=False,
            )

        return _run_process(
            env,
            launch_unmanaged,
            device=device,
            hooks=hooks,
            expected_version=expected,
            variables=variables,
        )


def _run_process(
    env: WorkerEnv,
    launch: Callable[[Mapping[str, str]], isolate.Session | subprocess.Popen[bytes]],
    *,
    device: str,
    hooks: TrainHooks | None,
    expected_version: Version | None,
    variables: Mapping[str, str],
) -> Run:
    state = _start_worker(env, launch, device, hooks, expected_version, variables)
    try:
        return_code = _wait_for_worker(state)
    finally:
        if state.process.poll() is None:
            _stop_process(state.process)
        _close_worker(state)
    return _worker_result(state, return_code)


def _start_worker(
    engine_env: WorkerEnv,
    launch: Callable[[Mapping[str, str]], isolate.Session | subprocess.Popen[bytes]],
    device: str,
    hooks: TrainHooks | None,
    expected_version: Version | None,
    variables: Mapping[str, str],
) -> _WorkerState:
    child_env = {
        **os.environ,
        **variables,
        "PHYSICALAI_WORKER_PROTOCOL_FD": "2",
        "PHYSICALAI_WORKER_FRAMEWORK": engine_env.framework,
        "PHYSICALAI_WORKER_PACKAGE": engine_env.package,
        "PHYSICALAI_WORKER_DEVICE": device,
    }
    process = launch(child_env)

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
        args=(
            process.stderr,
            events,
            state.protocol_reader_stop,
            f"{engine_env.framework} worker stderr",
        ),
        name=f"{engine_env.framework}-worker-protocol",
        daemon=True,
    )
    stdout_thread = threading.Thread(
        target=_forward_stream,
        args=(process.stdout, logging.INFO, f"{engine_env.framework} worker stdout"),
        daemon=True,
    )
    state.threads = (protocol_thread, stdout_thread)
    protocol_thread.start()
    stdout_thread.start()
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
    if state.process.stdin is not None:
        state.process.stdin.close()
    for thread in state.threads:
        thread.join(timeout=1.0)
    state.protocol_reader_stop.set()
    for thread in state.threads:
        thread.join(timeout=1.0)
    if isinstance(state.process, isolate.Session):
        state.process.close()
    else:
        for stream in (state.process.stdout, state.process.stderr):
            if stream is not None:
                stream.close()


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
    stream: BinaryIO | None,
    events: queue.Queue[dict[str, Any] | BaseException | _ProtocolEOF],
    stop: threading.Event,
    label: str,
) -> None:
    try:
        if stream is None:
            return
        file_descriptor = stream.fileno()
        pending = bytearray()
        with selectors.DefaultSelector() as selector:
            selector.register(file_descriptor, selectors.EVENT_READ)
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
                    if not _read_protocol_line(bytes(line), events, label):
                        return
            if pending and not stop.is_set():
                _read_protocol_line(bytes(pending), events, label)
    except (OSError, ValueError) as exc:
        events.put(exc)
    finally:
        events.put(_PROTOCOL_EOF)


def _read_protocol_line(
    line: bytes,
    events: queue.Queue[dict[str, Any] | BaseException | _ProtocolEOF],
    label: str,
) -> bool:
    # Console progress may share stderr without a newline before the tagged event.
    protocol_offset = line.find(_PROTOCOL_PREFIX)
    if protocol_offset >= 0:
        text = line[:protocol_offset].decode(_STREAM_CHUNK_ENCODING, errors="replace").rstrip("\r")
        if text:
            logger.info("%s: %s", label, text)
        payload = line[protocol_offset + len(_PROTOCOL_PREFIX) :].rstrip(b"\r")
        return _enqueue_protocol_line(payload, events)
    text = line.decode(_STREAM_CHUNK_ENCODING, errors="replace").rstrip("\r")
    if text:
        logger.info("%s: %s", label, text)
    return True


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
    if supported is not None and version not in supported:
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


def _send_cancel(process: isolate.Session | subprocess.Popen[bytes]) -> None:
    if process.stdin is None:
        return
    try:
        process.stdin.write(b'{"type":"cancel"}\n')
        process.stdin.flush()
    except BrokenPipeError:
        logger.debug("Worker closed its control pipe before cancellation was delivered.")


def _stop_process(process: isolate.Session | subprocess.Popen[bytes]) -> None:
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
