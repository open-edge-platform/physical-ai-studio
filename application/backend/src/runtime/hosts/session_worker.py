"""Detached RuntimeSession worker started by RuntimeProcessHost."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import sys
import threading
import traceback
from typing import TYPE_CHECKING
from uuid import uuid4

from loguru import logger

from core.logging import setup_logging
from exceptions import BaseException as AppBaseException
from runtime.contract import ErrorEvent
from runtime.session import RuntimeSession
from runtime.transport.server import RuntimeZenohServer

if TYPE_CHECKING:
    from types import FrameType

_WATCHDOG_INTERVAL_S = 0.1


def _watch_parent(parent_pid: int, stop: threading.Event) -> None:
    """Stop when B1's spawning API exits; B2 replaces this with subscriber presence."""
    while not stop.wait(_WATCHDOG_INTERVAL_S):
        if os.getppid() != parent_pid:
            logger.warning("Parent process {} exited; stopping the runtime session", parent_pid)
            stop.set()
            return


def suppress_stdout() -> int:
    """Redirect fd 1 while startup code may write outside Python's stdout object."""
    saved_fd = os.dup(1)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 1)
    os.close(devnull_fd)
    return saved_fd


def restore_stdout(saved_fd: int) -> None:
    """Restore fd 1 so the worker can send its one-line parent handshake."""
    os.dup2(saved_fd, 1)
    os.close(saved_fd)
    sys.stdout = os.fdopen(1, "w")


def _error_event(exc: Exception) -> ErrorEvent:
    if isinstance(exc, AppBaseException):
        return ErrorEvent(message=exc.message, error_code=exc.error_code)
    return ErrorEvent(
        message=str(exc) or "Failed to connect to the robot.",
        error_code="robot_connection_failed",
    )


def signal_ready() -> None:
    """Send the successful startup handshake and close stdout permanently."""
    sys.stdout.write("READY\n")
    sys.stdout.flush()
    sys.stdout.close()


def signal_error(exc: Exception, *, phase: str | None = None) -> None:
    """Send one structured startup failure handshake and close stdout permanently."""
    event = _error_event(exc)
    payload: dict[str, str] = {
        "msg": event.message,
        "error_code": event.error_code,
        "traceback": traceback.format_exc(),
    }
    if phase is not None:
        payload["phase"] = phase
    sys.stdout.write(f"ERROR:{json.dumps(payload)}\n")
    sys.stdout.flush()
    sys.stdout.close()


def _publish_fatal(server: RuntimeZenohServer | None, exc: Exception) -> None:
    """Report a failure after READY through the session's Zenoh error endpoint."""
    if server is not None and server.is_open:
        server.emit(_error_event(exc), fatal=True)


def _stop_on_sigterm(_signum: int, _frame: FrameType | None, stop: threading.Event) -> None:
    stop.set()


def _teardown(
    loop: asyncio.AbstractEventLoop,
    session: RuntimeSession | None,
    server: RuntimeZenohServer | None,
    *,
    report_errors: bool,
) -> bool:
    """Release session resources even if startup stopped partway through."""
    failed = False
    if session is not None:
        try:
            loop.run_until_complete(session.teardown())
        except Exception as exc:
            failed = True
            logger.exception("Runtime session teardown failed")
            if report_errors:
                _publish_fatal(server, exc)
    if server is not None:
        try:
            server.close()
        except Exception:
            failed = True
            logger.exception("Runtime transport teardown failed")
    with contextlib.suppress(Exception):
        loop.run_until_complete(loop.shutdown_asyncgens())
    loop.close()
    return failed


def main() -> int:
    """Run one runtime session configured by JSON stdin and acknowledged on stdout."""
    raw = sys.stdin.read()
    sys.stdin.close()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    stop = threading.Event()
    saved_fd = suppress_stdout()
    server: RuntimeZenohServer | None = None
    session: RuntimeSession | None = None
    phase = "invalid_config"

    try:
        payload = json.loads(raw)
        session_name = payload["session_name"]
        document = payload["document"]
        follower_name = payload.get("follower_name")
        leader_name = payload.get("leader_name")
        parent_pid = payload["parent_pid"]
        if not isinstance(session_name, str) or not isinstance(document, dict) or not isinstance(parent_pid, int):
            raise TypeError("Runtime session startup payload has invalid field types")

        setup_logging()
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, lambda signum, frame: _stop_on_sigterm(signum, frame, stop))
        threading.Thread(
            target=_watch_parent,
            args=(parent_pid, stop),
            name="runtime-parent-watchdog",
            daemon=True,
        ).start()

        instance_id = uuid4().hex
        server = RuntimeZenohServer(session_name, instance_id=instance_id)
        session = RuntimeSession(
            document,
            event_sink=server,
            follower_name=follower_name,
            leader_name=leader_name,
        )
        phase = "endpoint_collision"
        server.open(session.apply)
        server.wait_for_client()
        phase = "setup_failed"
        loop.run_until_complete(session.setup())
    except Exception as exc:
        restore_stdout(saved_fd)
        signal_error(exc, phase=phase)
        _teardown(loop, session, server, report_errors=False)
        return 1

    restore_stdout(saved_fd)
    signal_ready()

    completed = False
    try:
        session.run(stop)
        completed = True
    except Exception as exc:
        logger.exception("Runtime session failed")
        _publish_fatal(server, exc)
    finally:
        if _teardown(loop, session, server, report_errors=True):
            completed = False
    return 0 if completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
