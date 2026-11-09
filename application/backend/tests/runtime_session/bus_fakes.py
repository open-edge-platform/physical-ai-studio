"""A worker process on the bus, for the bus tests; module-level so a spawned child can import it."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, Field

from runtime_session.bus import Bus
from workers.base import ProcessWorker

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class Say(BaseModel):
    kind: Literal["say"] = "say"
    text: str


class Stop(BaseModel):
    kind: Literal["stop"] = "stop"


Command = Annotated[Say | Stop, Field(discriminator="kind")]


class Heard(BaseModel):
    text: str


def upper(message: Say) -> Heard:
    if not message.text:
        raise ValueError("nothing to say")
    return Heard(text=message.text.upper())


class EchoWorker(ProcessWorker[Bus]):
    """Publishes every ``say`` command back as an event and serves ``upper``."""

    def __init__(self, session_id: str) -> None:
        super().__init__(name="EchoWorker")
        self.session_id = session_id

    @asynccontextmanager
    async def lifecycle(self) -> AsyncIterator[Bus]:
        with Bus.join(self.session_id) as bus:
            yield bus

    async def run_loop(self, context: Bus) -> None:
        commands = context.subscribe("cmd/echo", Command)
        context.serve("rpc/echo/upper", Say, upper)
        while not self.should_stop():
            match await commands.next():
                case Say(text=text):
                    context.publish("evt/echo", Heard(text=text))
                case Stop():
                    return
