from __future__ import annotations

import asyncio
import uuid

import pytest

from runtime_session.bus import Bus, BusRequestError, NoResponderError, socket_path

from .bus_fakes import Command, EchoWorker, Heard, Say, Stop, upper


@pytest.fixture
def session_id() -> str:
    return uuid.uuid4().hex


async def _until_heard(bus: Bus, events, text: str) -> Heard:
    """Resend until the event comes back: a subscription takes a moment to reach the other peer."""
    for _ in range(100):
        bus.publish("cmd/echo", Say(text=text))
        try:
            return await asyncio.wait_for(events.next(), 0.05)
        except TimeoutError:
            continue
    raise AssertionError("never heard back")


async def test_pubsub_in_one_process_decodes_the_union(session_id: str) -> None:
    with Bus.host(session_id) as bus:
        commands = bus.subscribe("cmd/echo", Command)
        assert commands.poll() is None
        bus.publish("cmd/echo", Say(text="hi"))
        bus.publish("cmd/echo", Stop())
        assert await asyncio.wait_for(commands.next(), 1) == Say(text="hi")
        assert await asyncio.wait_for(commands.next(), 1) == Stop()


async def test_request_replies_and_reports_handler_errors(session_id: str) -> None:
    async def slow(message: Say) -> Heard:
        await asyncio.sleep(1)
        return upper(message)

    with Bus.host(session_id) as bus:
        bus.serve("rpc/echo/upper", Say, upper)
        bus.serve("rpc/echo/slow", Say, slow)
        assert await bus.request("rpc/echo/upper", Say(text="hi"), Heard, timeout=1) == Heard(text="HI")
        with pytest.raises(BusRequestError, match="nothing to say") as failed:
            await bus.request("rpc/echo/upper", Say(text=""), Heard, timeout=1)
        assert failed.value.error_code == "command_failed"
        with pytest.raises(BusRequestError) as invalid:
            await bus.request("rpc/echo/upper", '{"kind": "say"}', Heard, timeout=1)
        assert invalid.value.error_code == "invalid_command"
        with pytest.raises(TimeoutError):
            await bus.request("rpc/echo/slow", Say(text="hi"), Heard, timeout=0.2)
        with pytest.raises(NoResponderError):
            await bus.request("rpc/nobody/home", Say(text="hi"), Heard, timeout=0.2)


async def test_worker_process_on_the_bus(session_id: str) -> None:
    with Bus.host(session_id) as bus:
        events = bus.subscribe("evt/echo", Heard)
        async with EchoWorker(session_id) as worker:
            assert await _until_heard(bus, events, "hello") == Heard(text="hello")
            assert await bus.request("rpc/echo/upper", Say(text="hi"), Heard, timeout=5) == Heard(text="HI")
            bus.publish("cmd/echo", Stop())
            await asyncio.wait_for(worker.wait_stopped(), 5)
    assert not socket_path(session_id).exists()


async def test_sessions_do_not_hear_each_other(session_id: str) -> None:
    other = uuid.uuid4().hex
    with Bus.host(session_id) as bus, Bus.host(other) as other_bus:
        heard = bus.subscribe("cmd/echo", Command)
        bus.serve("rpc/echo/upper", Say, upper)
        other_bus.publish("cmd/echo", Say(text="hi"))
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(heard.next(), 0.3)
        with pytest.raises(NoResponderError):
            await other_bus.request("rpc/echo/upper", Say(text="hi"), Heard, timeout=0.2)


async def test_closing_a_declaration_undeclares_only_it(session_id: str) -> None:
    with Bus.host(session_id) as bus:
        kept = bus.subscribe("cmd/echo", Command)
        with bus.subscribe("cmd/echo", Command) as dropped, bus.serve("rpc/echo/upper", Say, upper):
            assert await bus.request("rpc/echo/upper", Say(text="hi"), Heard, timeout=1) == Heard(text="HI")
        bus.publish("cmd/echo", Say(text="hi"))
        assert await asyncio.wait_for(kept.next(), 1) == Say(text="hi")
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(dropped.next(), 0.2)
        with pytest.raises(NoResponderError):
            await bus.request("rpc/echo/upper", Say(text="hi"), Heard, timeout=0.2)
