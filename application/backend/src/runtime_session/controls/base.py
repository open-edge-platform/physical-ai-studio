"""What drives the follower: one control at a time, writing ``action``.

A control is a thread in the session's process, ticking at its own rate. It
borrows what the session owns (the store, the leader) and never connects or
disconnects anything, so switching controls is cheap and leaves every device
as it was. The robot worker follows whatever ``action`` holds, at its own rate.

Switching still leaves a gap, as the old control is joined before the next
starts. What changes while a control runs (teleop's slowdown, for example) is
therefore a command the control serves on the session's bus itself, between
its ticks, rather than a new control.

See the "Controls" section of docs/feature_plans/runtime-inhousing-plan.md.
"""

from __future__ import annotations

from abc import abstractmethod
from contextlib import ExitStack, asynccontextmanager
from typing import TYPE_CHECKING, Any

from runtime_session.bus import command_key
from workers.base import ThreadWorker
from workers.rate import Rate

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

    from pydantic import BaseModel

    from runtime_session.bus import Bus
    from runtime_session.store import Store
    from schemas.runtime_session import ControlConfig


type Commands = dict[str, tuple[type[BaseModel], Callable[[Any], BaseModel | Awaitable[BaseModel]]]]
"""Client commands by name, each with its payload and its handler, as :meth:`Bus.serve` takes them."""

CHANGED = "evt/control"
"""Where a running control publishes its config after one of its commands changed it."""


class Control(ThreadWorker[None]):
    """Calls :meth:`on_start` once, then :meth:`on_tick` at ``hz`` until stopped.

    Both run on the control's thread. An exception in ``on_start`` fails the
    start (the session then holds); one in ``on_tick`` ends the control.
    """

    def __init__(self, store: Store, config: ControlConfig, *, name: str, hz: float, bus: Bus | None = None) -> None:
        super().__init__(name=name)
        self.store = store
        self.hz = hz
        self.config = config
        """What was asked for, and changed since by its commands; the session reports it in its state."""
        self.bus = bus
        """Where :meth:`commands` are served; without one, nothing is."""

    @asynccontextmanager
    async def lifecycle(self) -> AsyncIterator[None]:
        self.on_start()
        try:
            with ExitStack() as served:
                if self.bus is not None:
                    for name, (model, handler) in self.commands().items():
                        # Served on this thread's loop, so a handler runs between ticks, never during one.
                        served.enter_context(self.bus.serve(command_key(name), model, handler))
                yield
        finally:
            self.on_stop()

    async def run_loop(self, context: None) -> None:  # noqa: ARG002
        rate = Rate(self.hz, name=self.name)
        while not self.should_stop():
            self.on_tick()
            await rate.sleep()

    def on_start(self) -> None:
        """Capture what the control starts from, before its first tick. Does nothing by default."""

    def on_stop(self) -> None:
        """Release what :meth:`on_start` set up, on the control's thread. Does nothing by default."""

    def commands(self) -> Commands:
        """The client commands this control serves while it runs, by name, like the session's own. None by default."""
        return {}

    def changed(self, config: ControlConfig) -> None:
        """Take ``config`` as what this control now does, and tell the session so it reports it."""
        self.config = config
        if self.bus is not None:
            self.bus.publish(CHANGED, config)

    @abstractmethod
    def on_tick(self) -> None:
        """Decide and write ``action``."""
