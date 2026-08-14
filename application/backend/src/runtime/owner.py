"""Parent-side policy: probe, attach or spawn, and resolve the spawn race."""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any

from loguru import logger

from exceptions import BaseException as AppBaseException
from exceptions import RuntimeSessionBusyError
from runtime.config_builder import runtime_config_digest
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
    not open a Zenoh session — and only probes ``/metadata`` on a hit.
    """
    name = runtime_session_name(follower_id)
    if live_session_pid(name) is None:
        return None
    return probe_session_metadata(name, timeout=timeout)


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

    def connect(self) -> None:
        """Attach to a live session, or spawn one. Blocking."""
        metadata = self._client.probe()
        if metadata is not None:
            self._attach_to(metadata)
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
            metadata = self._client.probe_with_retry(_RACE_RETRY_TIMEOUT)
            if metadata is None:
                raise
            self._attach_to(metadata)
            return

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
            raise
        self._spawned = True

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

    def _attach_to(self, metadata: dict[str, Any]) -> None:
        self._reject_if_different_rig(metadata)
        self._client.attach(metadata)
        self._metadata = metadata

    def _reject_if_different_rig(self, metadata: dict[str, Any]) -> None:
        expected = runtime_config_digest(self._document)
        actual = metadata.get("config_digest")
        if actual == expected:
            return
        pid = metadata.get("pid")
        raise RuntimeSessionBusyError(
            robot_name=self._follower_name,
            pid=pid if isinstance(pid, int) else None,
        )
