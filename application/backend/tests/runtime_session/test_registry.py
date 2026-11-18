from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from uuid import uuid4

import pytest

from exceptions import ResourceInUseError
from runtime_session.registry import RuntimeSessionRegistry


def _closing_session(session_id: str) -> SimpleNamespace:
    closed, finished = asyncio.Event(), asyncio.Event()
    closed.set()
    return SimpleNamespace(id=session_id, label=session_id, closed=closed, finished=finished)


async def test_a_load_waits_only_for_the_closing_session_that_holds_its_robot() -> None:
    parking, other = uuid4(), uuid4()
    registry = RuntimeSessionRegistry()
    session = _closing_session("parking")
    registry.add(session)  # type: ignore[arg-type]
    registry.claim_robots(session, {parking: "reBot"})  # type: ignore[arg-type]

    started = time.monotonic()
    await registry.until_closed(within=5.0, holding=[other])
    assert time.monotonic() - started < 0.5

    asyncio.get_running_loop().call_later(0.3, session.finished.set)  # the park ends
    started = time.monotonic()
    await registry.until_closed(within=5.0, holding=[parking])
    assert 0.25 < time.monotonic() - started < 2.0


def test_export_readers_are_reference_counted_and_scoped() -> None:
    registry = RuntimeSessionRegistry()
    model_id = uuid4()
    with registry.reading_model_export(model_id, "openvino", "original"):
        with registry.reading_model_export(model_id, "openvino", "original"):
            assert registry._model_exports[(model_id, "openvino", "original")] == 2
            with pytest.raises(ResourceInUseError), registry.deleting_model_export(model_id, "openvino"):
                pass
        with pytest.raises(ResourceInUseError), registry.deleting_model_export(model_id, "openvino", "original"):
            pass
        with registry.deleting_model_export(model_id, "openvino", "int8"):
            pass
        with registry.deleting_model_export(model_id, "torch"):
            pass
        with registry.deleting_model_export(uuid4(), "openvino"):
            pass
    with registry.deleting_model_export(model_id, "openvino"):
        pass


@pytest.mark.parametrize("scope", [None, "original"])
def test_deletion_blocks_overlapping_readers_and_removals_and_releases_after_failure(scope) -> None:
    registry = RuntimeSessionRegistry()
    model_id = uuid4()
    with pytest.raises(RuntimeError, match="failed"), registry.deleting_model_export(model_id, "openvino", scope):
        with pytest.raises(ResourceInUseError), registry.reading_model_export(model_id, "openvino", "original"):
            pass
        with pytest.raises(ResourceInUseError), registry.deleting_model_export(model_id, "openvino"):
            pass
        raise RuntimeError("failed")
    with registry.reading_model_export(model_id, "openvino", "original"):
        pass
