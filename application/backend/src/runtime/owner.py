"""Parent-side policy: probe, attach or spawn, and resolve the spawn race."""

from __future__ import annotations

import contextlib
import os
import signal
import time
from typing import TYPE_CHECKING, Any

from loguru import logger

from exceptions import BaseException as AppBaseException
from exceptions import RuntimeSessionBusyError
from runtime.config_builder import runtime_camera_keys, runtime_identity_digest
from runtime.hosts.process_host import RuntimeProcessHost
from runtime.transport.codec import decode_metadata
from runtime.transport.ids import metadata_key, runtime_session_name
from runtime.transport.lock import live_session_pid
from runtime.transport.session import open_session

if TYPE_CHECKING:
    from uuid import UUID

    from runtime.transport.client import RuntimeSessionClient

_PROBE_TIMEOUT = 1.0
_RACE_RETRY_TIMEOUT = 5.0
_STOP_TIMEOUT = 5.0


def probe_session_metadata(session_name: str, timeout: float = _PROBE_TIMEOUT) -> dict[str, Any] | None:
    """Query ``/metadata`` once without declaring telemetry subscribers.

    Used by discovery (deletion guards, the record-path check) so a miss does
    not reset a session's idle countdown.
    """
    session = open_session(session_name, listen=False)
    try:
        replies = session.get(metadata_key(session_name), timeout=timeout)
        for reply in replies:
            sample = reply.ok
            if sample is not None:
                return decode_metadata(sample.payload.to_bytes())
    except Exception:
        logger.debug("Runtime metadata probe failed for {}", session_name, exc_info=True)
    finally:
        session.close()
    return None


def runtime_session_holder(follower_id: UUID | str, *, timeout: float = _PROBE_TIMEOUT) -> dict[str, Any] | None:
    """Return metadata for a live session driving this follower, or ``None``.

    Reads the on-disk lock registry first — a miss is the common case and must
    not open a Zenoh session — and only probes ``/metadata`` on a hit. A live
    lock with a metadata miss is still treated as held: the worker may have
    the flock and the arm before ``/metadata`` answers, and a probe timeout
    must not look like an idle robot.
    """
    name = runtime_session_name(follower_id)
    pid = live_session_pid(name)
    if pid is None:
        return None
    metadata = probe_session_metadata(name, timeout=timeout)
    if metadata is not None:
        return metadata
    logger.warning(
        "Runtime session {} holds the lock (pid {}) but metadata did not answer",
        name,
        pid,
    )
    return {"pid": pid}


def stop_runtime_session(session_name: str, *, timeout: float = _STOP_TIMEOUT) -> None:
    """Terminate the live session for ``session_name``, if any. Blocking.

    Sends SIGTERM to the lock holder, waits for the name lock to drop, then
    SIGKILL. Does not attach as a subscriber, so a winding-down session is
    not kept alive by the stop itself.
    """
    pid = live_session_pid(session_name)
    if pid is None:
        metadata = probe_session_metadata(session_name, timeout=min(1.0, timeout))
        raw = None if metadata is None else metadata.get("pid")
        pid = raw if isinstance(raw, int) and raw > 0 else None
    if pid is None:
        return

    logger.info("Stopping runtime session {} (pid {})", session_name, pid)
    _signal_pid(pid, signal.SIGTERM)
    if _wait_until_session_gone(session_name, pid, timeout):
        return

    logger.warning("Runtime session {} (pid {}) did not stop within {}s, killing", session_name, pid, timeout)
    _signal_pid(pid, signal.SIGKILL)
    _wait_until_session_gone(session_name, pid, 1.0)


def _signal_pid(pid: int, sig: signal.Signals) -> None:
    with contextlib.suppress(OSError):
        os.kill(pid, sig)


def _wait_until_session_gone(session_name: str, pid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if live_session_pid(session_name) is None:
            return True
        try:
            os.kill(pid, 0)
        except OSError:
            return True
        time.sleep(0.05)
    return live_session_pid(session_name) is None


class RuntimeSessionOwner:
    """Attach to a live runtime session, or spawn one if none answers."""

    def __init__(
        self,
        client: RuntimeSessionClient,
        *,
        session_name: str,
        document: dict[str, Any],
        follower_name: str | None,
        leader_name: str | None,
        idle_timeout_s: float,
    ) -> None:
        self._client = client
        self._session_name = session_name
        self._document = document
        self._follower_name = follower_name
        self._leader_name = leader_name
        self._idle_timeout_s = idle_timeout_s
        self._host: RuntimeProcessHost | None = None
        self._metadata: dict[str, Any] | None = None
        self._spawned = False

    @property
    def spawned(self) -> bool:
        """Whether this owner started the child, rather than attaching to one."""
        return self._spawned

    @property
    def metadata(self) -> dict[str, Any]:
        """Session metadata adopted during ``connect``."""
        if self._metadata is None:
            raise RuntimeError("Runtime session owner is not connected")
        return self._metadata

    @property
    def host(self) -> RuntimeProcessHost | None:
        """The spawned child handle, or ``None`` when this owner attached."""
        return self._host

    @property
    def error(self) -> AppBaseException | None:
        """A startup failure reported by a child this owner spawned."""
        return None if self._host is None else self._host.error

    def is_alive(self) -> bool:
        """Return whether the session process is still running."""
        if self._spawned:
            return self._host is not None and self._host.is_alive()
        pid = None if self._metadata is None else self._metadata.get("pid")
        if not isinstance(pid, int) or pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    def exited_cleanly(self) -> bool:
        """Return whether the session ended through a clean shutdown path."""
        if self._spawned:
            return self._host is not None and self._host.exited_cleanly
        return self._client.shutdown_received

    def stop(self) -> None:
        """Terminate the child. Only meaningful for a session this owner spawned."""
        if self._host is not None:
            self._host.stop()

    def stop_abandoned_spawn(self) -> None:
        """Stop a child this owner started that other clients cannot attach to yet.

        A websocket close after ``/metadata`` is up is a detach: another client
        may already be attached, and a refresh must be able to rejoin. Only an
        in-flight spawn that has not published metadata is private to this owner.
        """
        if self._host is None or self._spawned or self._metadata is not None:
            return
        if self._client.probe(timeout=_PROBE_TIMEOUT) is not None:
            return
        self.stop()

    def connect(self, *, replace: bool = False) -> None:
        """Attach to a live session, or spawn one. Blocking.

        When ``replace`` is true, any live session for this name is stopped
        first and this owner always tries to spawn with its current document.
        """
        if replace:
            stop_runtime_session(self._session_name)

        skip_probe = replace
        while True:
            metadata = None if skip_probe else self._client.probe()
            skip_probe = False
            if metadata is not None and self._try_attach(metadata):
                return

            self._host = RuntimeProcessHost(
                self._session_name,
                self._document,
                follower_name=self._follower_name,
                leader_name=self._leader_name,
                idle_timeout_s=self._idle_timeout_s,
            )
            try:
                self._host.start()
            except AppBaseException as exc:
                self._host = None
                if getattr(exc, "phase", None) != "name_lock_contention":
                    raise
                # The winner may not have published metadata yet.
                metadata = self._client.probe_with_retry(_RACE_RETRY_TIMEOUT)
                if metadata is None:
                    raise
                if self._try_attach(metadata):
                    return
                continue

            self._spawned = True
            try:
                metadata = self._wait_for_spawned_metadata(_RACE_RETRY_TIMEOUT)
                if metadata is None:
                    raise AppBaseException(
                        message=f"Runtime session {self._session_name} reported READY but its metadata is unreachable.",
                        error_code="robot_connection_failed",
                        http_status=500,
                    )
                self._attach_to(metadata)
            except Exception:
                self.stop()
                self._host = None
                self._spawned = False
                raise
            return

    def _wait_for_spawned_metadata(self, timeout: float) -> dict[str, Any] | None:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            if self._host is not None and not self._host.is_alive():
                if self._host.error is not None:
                    raise self._host.error
                raise AppBaseException(
                    message="Runtime session stopped before answering metadata",
                    error_code="robot_connection_failed",
                    http_status=500,
                )
            metadata = self._client.probe(timeout=min(1.0, remaining))
            if metadata is not None:
                return metadata
            time.sleep(min(0.05, remaining))

    def _try_attach(self, metadata: dict[str, Any]) -> bool:
        """Attach to ``metadata`` if it can serve this client.

        Returns True after attaching. Returns False after stopping a session
        that matches identity but lacks cameras this client needs, so the
        caller can spawn. Raises RuntimeSessionBusyError on an identity mismatch.

        Identity is checked first: a client asking for a different arm must get
        423, not a silent takeover because it also happens to need more cameras.
        """
        self._reject_if_different_rig(metadata)
        if self._needs_more_cameras(metadata):
            logger.info(
                "Runtime session {} is missing cameras this client needs; restarting",
                self._session_name,
            )
            stop_runtime_session(self._session_name)
            return False
        self._attach_to(metadata)
        return True

    def _attach_to(self, metadata: dict[str, Any]) -> None:
        self._reject_if_different_rig(metadata)
        self._client.attach(metadata)
        self._metadata = metadata

    def _reject_if_different_rig(self, metadata: dict[str, Any]) -> None:
        expected = runtime_identity_digest(self._document)
        if metadata.get("identity_digest") == expected:
            return
        pid = metadata.get("pid")
        raise RuntimeSessionBusyError(
            robot_name=self._follower_name,
            pid=pid if isinstance(pid, int) else None,
        )

    def _needs_more_cameras(self, metadata: dict[str, Any]) -> bool:
        """Whether the running session lacks a camera this client needs.

        Cameras are outside the session identity, so a session started by a client
        that needed none — the environment form preview — is a valid attach target
        for its own identity but cannot serve a client that has to read frames.
        Restarting is correct rather than rude: the displaced client reattaches to
        the superset, because its identity still matches.
        """
        running = set(metadata.get("camera_keys") or [])
        return not set(runtime_camera_keys(self._document)).issubset(running)
