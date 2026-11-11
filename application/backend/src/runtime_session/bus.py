"""Commands, events and requests between the processes of one session, over zenoh.

The session process :meth:`Bus.host`\\ s a zenoh router listening on a Unix
socket; its worker processes :meth:`Bus.join` by connecting to that socket.
Nothing listens on the network and sessions cannot hear each other. Keys are
relative to ``pai/<session_id>/``: ``cmd/<component>`` and ``evt/<component>``
for pub/sub, ``rpc/<component>/<op>`` for requests. Payloads are pydantic
models as JSON. Joints and frames go through the :mod:`store`, not the bus.

See the "Event bus" section of docs/feature_plans/runtime-inhousing-plan.md.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import zenoh
from loguru import logger
from pydantic import BaseModel, TypeAdapter, ValidationError

from schemas.runtime_session import ComponentState, StatusMessage

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

_ROOT = "pai"

EVENTS = "evt/session"
"""The session's general channel, which its socket forwards: ``SessionEvent`` messages from the session."""


STATUSES = "evt/status/**"
"""Every part's and device's ``StatusMessage``, which the socket forwards."""


def status_key(component: str) -> str:
    """Where ``component`` publishes its ``StatusMessage``."""
    return f"evt/status/{component}"


def publish_status(bus: Bus, component: str, state: ComponentState, error: str | None = None) -> None:
    """Say how ``component`` is, on its status key."""
    bus.publish(status_key(component), StatusMessage(component=component, state=state, error=error))


def command_key(command: str) -> str:
    """Where a client command is served: ``rpc/<command>``, by whichever part handles it.

    A command names what to do, not who does it, so a socket can forward
    commands without knowing which part of the session answers them.
    """
    return f"rpc/{command}"


class BusRequestError(RuntimeError):
    """The handler of a request raised; carries its message and error code."""

    def __init__(self, message: str, error_code: str, data: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.data = data or {}
        """Whatever the handler's exception carried as ``data``, for the client."""


class NoResponderError(LookupError):
    """Nothing serves the requested key right now, for example a command for a part that is not loaded."""


def socket_path(session_id: str) -> Path:
    """Where the hosting session listens."""
    # shortcut: a fixed temp-dir path assumes one backend per host (like store blocks); use a per-backend dir otherwise.
    return Path(tempfile.gettempdir()) / f"pai-{session_id}.sock"


class Declaration:
    """A subscriber or queryable on the bus; close it, or use it as a context manager, to undeclare it.

    Whatever is still open when the bus closes is undeclared with it.
    """

    def __init__(self, open_declarations: set[Declaration]) -> None:
        self._entity: zenoh.Subscriber | zenoh.Queryable | None = None
        self._open = open_declarations
        self._open.add(self)

    def close(self) -> None:
        self._open.discard(self)
        # Taken out first, so of two closers (the owner, and the bus closing) only one undeclares it.
        entity, self._entity = self._entity, None
        if entity is not None:
            entity.undeclare()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class Subscription[M](Declaration):
    """Messages on one key, handed to the event loop that subscribed."""

    def __init__(self, model: Any, loop: asyncio.AbstractEventLoop, open_declarations: set[Declaration]) -> None:
        super().__init__(open_declarations)
        self._adapter: TypeAdapter[M] = TypeAdapter(model)
        self._loop = loop
        self._queue: asyncio.Queue[M] = asyncio.Queue()

    def _on_sample(self, sample: zenoh.Sample) -> None:  # zenoh thread
        try:
            message = self._adapter.validate_json(sample.payload.to_bytes())
        except ValueError:
            logger.exception("Dropped a malformed message on {}", sample.key_expr)
            return
        self._loop.call_soon_threadsafe(self._queue.put_nowait, message)

    async def next(self) -> M:
        return await self._queue.get()

    def poll(self) -> M | None:
        """The oldest undelivered message, or ``None``; for fixed-rate loops."""
        try:
            return self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return None


class Bus:
    """One process's zenoh session, scoped to a runtime session."""

    def __init__(self, session_id: str, config: zenoh.Config) -> None:
        self.session_id = session_id
        self._prefix = f"{_ROOT}/{session_id}/"
        self._session = zenoh.open(config)
        self._declared: set[Declaration] = set()

    @classmethod
    def host(cls, session_id: str) -> Bus:
        """The session process's bus: listens on the session's socket."""
        path = socket_path(session_id)
        path.unlink(missing_ok=True)  # left behind by a killed session
        return cls(session_id, _config(listen=path))

    @classmethod
    def join(cls, session_id: str) -> Bus:
        """A worker process's bus: connects to the hosting session."""
        return cls(session_id, _config(connect=socket_path(session_id)))

    def publish(self, key: str, message: BaseModel) -> None:
        self._session.put(self._prefix + key, message.model_dump_json())

    def subscribe[M](self, key: str, model: type[M] | Any) -> Subscription[M]:
        """Receive messages on ``key``; call from the event loop that will consume them.

        ``model`` is a pydantic model or a discriminated union of them. Messages
        published before the subscription has reached the publisher's peer are
        lost; use :meth:`request` when delivery must be confirmed.
        """
        subscription: Subscription[M] = Subscription(model, asyncio.get_running_loop(), self._declared)
        subscription._entity = self._session.declare_subscriber(self._prefix + key, subscription._on_sample)
        return subscription

    def listen[M](self, key: str, model: type[M] | Any, handler: Callable[[M], Awaitable[None] | None]) -> Declaration:
        """Call ``handler`` with each message on ``key``; call from the event loop it runs on.

        Each message gets its own call, started in the order they arrive. A handler that
        raises is logged. Like :meth:`subscribe`, it misses what was published before the
        subscription reached the publisher.
        """
        loop = asyncio.get_running_loop()
        adapter: TypeAdapter[M] = TypeAdapter(model)

        async def handle(message: M) -> None:
            try:
                result = handler(message)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.exception("Handling a message on {} failed", key)

        def on_sample(sample: zenoh.Sample) -> None:  # zenoh thread
            try:
                message = adapter.validate_json(sample.payload.to_bytes())
            except ValueError:
                logger.exception("Dropped a malformed message on {}", sample.key_expr)
                return
            asyncio.run_coroutine_threadsafe(handle(message), loop)

        listened = Declaration(self._declared)
        listened._entity = self._session.declare_subscriber(self._prefix + key, on_sample)
        return listened

    async def request[R](
        self,
        key: str,
        message: BaseModel | str,
        response: type[R] | Any,
        *,
        timeout: float,  # noqa: ASYNC109 - zenoh enforces it
    ) -> R:
        """Send ``message`` (a model, or JSON as is) to the handler serving ``key`` and return its reply.

        Raises:
            BusRequestError: The handler raised; carries its message and error code.
            NoResponderError: Nothing serves ``key``.
            TimeoutError: The handler did not reply in time.
        """
        return await asyncio.to_thread(self.request_sync, key, message, response, timeout=timeout)

    def request_sync[R](self, key: str, message: BaseModel | str, response: type[R] | Any, *, timeout: float) -> R:
        """:meth:`request`, blocking the calling thread; for threads outside the event loop, like a model's executor.

        Raises:
            BusRequestError: The handler raised; carries its message and error code.
            NoResponderError: Nothing serves ``key``.
            TimeoutError: The handler did not reply in time.
        """
        payload = message if isinstance(message, str) else message.model_dump_json()
        replies = list(self._session.get(self._prefix + key, payload=payload, timeout=timeout))
        # zenoh reports its own timeout as an error reply without a replier
        answered = [reply for reply in replies if reply.ok is not None or reply.replier_id is not None]
        if not answered:
            if replies:
                raise TimeoutError(f"No reply on {key} within {timeout}s")
            raise NoResponderError(f"Nothing serves {key}")
        reply = answered[0]
        if reply.ok is None:
            error = json.loads(reply.err.payload.to_string()) if reply.err else {}
            raise BusRequestError(
                error.get("message", ""), error.get("error_code", "command_failed"), error.get("data")
            )
        adapter: TypeAdapter[R] = TypeAdapter(response)
        return adapter.validate_json(reply.ok.payload.to_bytes())

    def serve[M](
        self, key: str, model: type[M] | Any, handler: Callable[[M], BaseModel | Awaitable[BaseModel]]
    ) -> Declaration:
        """Answer requests on ``key``; call from the event loop async handlers run on.

        An exception in ``handler`` is sent back and raised in the requester as
        :class:`BusRequestError`, with the exception's ``error_code`` if it has
        one (``invalid_command`` for a payload that does not validate).
        """
        loop = asyncio.get_running_loop()
        adapter: TypeAdapter[M] = TypeAdapter(model)

        async def answer(query: zenoh.Query) -> None:
            try:
                result = handler(adapter.validate_json(query.payload.to_bytes() if query.payload else b""))
                if inspect.isawaitable(result):
                    result = await result
                query.reply(query.key_expr, result.model_dump_json())
            except Exception as error:  # every failure goes back to the requester
                logger.opt(exception=error).debug("Request on {} failed", key)
                data = getattr(error, "data", None)
                reply = {
                    "message": str(error),
                    "error_code": _error_code(error),
                    "data": data if isinstance(data, dict) else None,
                }
                query.reply_err(json.dumps(reply))
            finally:
                query.drop()

        def on_query(query: zenoh.Query) -> None:  # zenoh thread
            asyncio.run_coroutine_threadsafe(answer(query), loop)

        served = Declaration(self._declared)
        served._entity = self._session.declare_queryable(self._prefix + key, on_query)
        return served

    def close(self) -> None:
        for declared in list(self._declared):
            declared.close()
        self._session.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


async def until_serving[R](bus: Bus, key: str, message: BaseModel, response: type[R] | Any, *, within: float) -> R:
    """The first reply on ``key`` from a worker that just started: its queryables take a moment to reach the bus.

    Raises:
        NoResponderError: Nothing served ``key`` within ``within`` seconds.
        TimeoutError: Neither did anything reply.
    """
    deadline = time.monotonic() + within
    while True:
        try:
            return await bus.request(key, message, response, timeout=1.0)
        except (NoResponderError, TimeoutError):
            if time.monotonic() > deadline:
                raise
            await asyncio.sleep(0.05)


def _error_code(error: Exception) -> str:
    if isinstance(error, ValidationError):
        return "invalid_command"
    code = getattr(error, "error_code", None)
    return code if isinstance(code, str) else "command_failed"


def _config(*, listen: Path | None = None, connect: Path | None = None) -> zenoh.Config:
    config = zenoh.Config()
    # The host routes, so workers that joined hear each other and not only the host.
    config.insert_json5("mode", '"router"' if listen else '"client"')
    config.insert_json5("scouting/multicast/enabled", "false")
    config.insert_json5("listen/endpoints", json.dumps([f"unixsock-stream/{listen}"] if listen else []))
    config.insert_json5("connect/endpoints", json.dumps([f"unixsock-stream/{connect}"] if connect else []))
    return config
