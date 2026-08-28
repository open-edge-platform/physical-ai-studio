"""Detached RuntimeSession worker started by RuntimeProcessHost."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from loguru import logger

from core.logging import setup_logging
from exceptions import BaseException as AppBaseException
from exceptions import RuntimeSessionBusyError
from runtime.config_builder import runtime_camera_keys, runtime_identity_digest
from runtime.contract import ErrorEvent, LifecycleData, LifecycleEvent, SetFollowerSourceCommand
from runtime.session import RuntimeSession
from runtime.transport.lock import SessionNameLock, live_session_pid
from runtime.transport.server import RuntimeZenohServer

if TYPE_CHECKING:
    from types import FrameType

_IDLE_POLL_INTERVAL_S = 0.1


def _watch_subscribers(
    server: RuntimeZenohServer,
    session: RuntimeSession,
    idle_timeout_s: float,
    stop: threading.Event,
) -> None:
    """Hold the arm and commit the recording when the last subscriber leaves, then idle-exit.

    Started only after wait_for_client() returned, so a subscriber has already
    matched at least once. Losing every subscriber is the worst abandonment
    state — an arm following a leader with nobody watching — so the session
    latches a target before the countdown starts.

    The recording is finalized at the same moment rather than at process exit.
    The process itself stays alive for the idle window so a returning client
    keeps the hardware connection, but the dataset must not wait that long: the
    user navigates straight back to the dataset page expecting their episodes.
    """
    subscribers_present = True
    idle_since: float | None = None

    while not stop.wait(_IDLE_POLL_INTERVAL_S):
        if server.has_matching_subscribers():
            subscribers_present = True
            idle_since = None
            continue
        if subscribers_present:
            subscribers_present = False
            logger.warning("Runtime session lost its last subscriber; switching follower to hold")
            try:
                session.apply(SetFollowerSourceCommand(follower_source="hold"))
            except Exception:
                logger.exception("Failed to switch runtime session to hold after losing subscribers")
            # Latch the arm first, then commit. Waiting for the idle exit would
            # leave saved episodes invisible on the dataset page for the whole
            # countdown, and an abandoned open episode would pause it forever.
            try:
                session.finalize_recording()
            except Exception:
                logger.exception("Failed to finalize the recording after losing subscribers")
        now = time.monotonic()
        if idle_since is None:
            idle_since = now
        elif now - idle_since > idle_timeout_s:
            if server.has_matching_subscribers():
                # A client attached on the deadline. Without this re-check they
                # would lose the session and pay a full hardware reconnect.
                idle_since = None
                continue
            logger.info("Runtime session idle for {}s; shutting down", idle_timeout_s)
            server.emit(LifecycleEvent(data=LifecycleData(event="shutdown", reason="idle_timeout")))
            # Setting stop ends session.run(...), which returns through
            # session.teardown() in main(), releasing the devices. The recording
            # was already committed when the last subscriber left, so nothing
            # depends on reaching this point.
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
    """Send the startup handshake, then make the closed stdout safe to write to.

    The handshake pipe must close so the parent stops waiting, but a closed
    ``sys.stdout`` makes every later write raise ``ValueError``. tqdm flushes
    stdout while building a progress bar, so ``datasets`` crashed
    ``save_episode()``. The ``/dev/null`` handle covers any library that writes
    to stdout and is held for the process lifetime, never closed.

    Disabling progress bars is a separate concern: tqdm renders to *stderr*,
    which is the session's log sink, so leaving them on spams the logs with
    ``Map: 100%|...`` on every episode.
    """
    from datasets.utils import disable_progress_bars

    sys.stdout.write("READY\n")
    sys.stdout.flush()
    sys.stdout.close()
    sys.stdout = open(os.devnull, "w")  # noqa: SIM115
    disable_progress_bars()


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


@dataclass(frozen=True, slots=True)
class _StartupPayload:
    session_name: str
    document: dict[str, Any]
    follower_name: str | None
    leader_name: str | None
    idle_timeout_s: float


def _parse_startup_payload(raw: str) -> _StartupPayload:
    payload = json.loads(raw)
    session_name = payload["session_name"]
    document = payload["document"]
    follower_name = payload.get("follower_name")
    leader_name = payload.get("leader_name")
    idle_timeout_s = payload["idle_timeout_s"]
    if (
        not isinstance(session_name, str)
        or not isinstance(document, dict)
        or isinstance(idle_timeout_s, bool)
        or not isinstance(idle_timeout_s, int | float)
        or (follower_name is not None and not isinstance(follower_name, str))
        or (leader_name is not None and not isinstance(leader_name, str))
    ):
        raise TypeError("Runtime session startup payload has invalid field types")
    return _StartupPayload(session_name, document, follower_name, leader_name, float(idle_timeout_s))


def _open_locked_session(
    payload: _StartupPayload,
    stop: threading.Event,
    loop: asyncio.AbstractEventLoop,
    phase: list[str],
) -> tuple[SessionNameLock, RuntimeZenohServer, RuntimeSession]:
    """Take the name lock, declare endpoints, and set the session up.

    ``phase`` is a one-element list so the handshake can report where startup
    failed after this function raises.
    """
    lock = SessionNameLock(payload.session_name)
    phase[0] = "name_lock_contention"
    if not lock.acquire():
        raise RuntimeSessionBusyError(robot_name=payload.follower_name, pid=live_session_pid(payload.session_name))

    instance_id = uuid4().hex
    server = RuntimeZenohServer(payload.session_name, instance_id=instance_id)
    server.update_metadata(
        identity_digest=runtime_identity_digest(payload.document),
        camera_keys=runtime_camera_keys(payload.document),
        pid=os.getpid(),
        started_at=time.time(),
        idle_timeout_s=payload.idle_timeout_s,
    )
    session = RuntimeSession(
        payload.document,
        event_sink=server,
        follower_name=payload.follower_name,
        leader_name=payload.leader_name,
    )
    phase[0] = "endpoint_collision"
    server.open(session.apply, session.handle_request)
    server.wait_for_client()
    phase[0] = "setup_failed"
    loop.run_until_complete(session.setup())
    threading.Thread(
        target=_watch_subscribers,
        args=(server, session, payload.idle_timeout_s, stop),
        name="runtime-subscriber-watch",
        daemon=True,
    ).start()
    return lock, server, session


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
    lock: SessionNameLock | None = None
    phase = ["invalid_config"]

    try:
        try:
            payload = _parse_startup_payload(raw)
            setup_logging()
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, lambda signum, frame: _stop_on_sigterm(signum, frame, stop))
            lock, server, session = _open_locked_session(payload, stop, loop, phase)
        except Exception as exc:
            restore_stdout(saved_fd)
            signal_error(exc, phase=phase[0])
            _teardown(loop, session, server, report_errors=False)
            return 1

        restore_stdout(saved_fd)
        signal_ready()

        completed = False
        try:
            # session.run() returns through session.teardown() below, which
            # finalizes any open recording mutation so saved episodes survive.
            session.run(stop)
            completed = True
        except Exception as exc:
            logger.exception("Runtime session failed")
            _publish_fatal(server, exc)
        finally:
            if _teardown(loop, session, server, report_errors=True):
                completed = False
        return 0 if completed else 1
    finally:
        if lock is not None:
            lock.release()


if __name__ == "__main__":
    raise SystemExit(main())
