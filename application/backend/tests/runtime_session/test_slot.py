from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import pytest

from exceptions import RuntimeSessionCommandError
from runtime_session.slot import Slot

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@asynccontextmanager
async def _part(log: list[str], name: str, *, fail: bool = False) -> AsyncIterator[str]:
    log.append(f"enter {name}")
    if fail:
        raise RuntimeError("would not connect")
    try:
        yield name
    finally:
        log.append(f"exit {name}")


async def test_load_holds_the_value_until_unload() -> None:
    log: list[str] = []
    slot: Slot[str] = Slot("environment")
    assert await slot.load(_part(log, "bench")) == "bench"
    assert slot.value == slot.require("load one") == "bench"
    await slot.unload()
    await slot.unload()  # a no-op when empty
    assert slot.value is None
    assert log == ["enter bench", "exit bench"]


async def test_one_at_a_time_and_require_says_what_to_do() -> None:
    slot: Slot[str] = Slot("environment")
    with pytest.raises(RuntimeSessionCommandError, match="Load an environment first") as missing:
        slot.require("Load an environment first.")
    assert missing.value.error_code == "no_environment"
    await slot.load(_part([], "bench"))
    with pytest.raises(RuntimeSessionCommandError) as twice:
        await slot.load(_part([], "other"))
    assert twice.value.error_code == "environment_already_loaded"


async def test_a_failed_load_leaves_the_slot_empty() -> None:
    slot: Slot[str] = Slot("environment")
    with pytest.raises(RuntimeError):
        await slot.load(_part([], "bench", fail=True))
    assert slot.value is None
    assert await slot.load(_part([], "bench")) == "bench"
