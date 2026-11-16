import asyncio
from unittest.mock import MagicMock, call

import pytest

from workers.robots.so101_setup_worker import SO101SetupWorker


@pytest.mark.parametrize("disconnect_fails", [False, True])
async def test_cleanup_disconnects_with_public_bus_api(disconnect_fails: bool) -> None:
    worker = SO101SetupWorker.__new__(SO101SetupWorker)
    bus = MagicMock(spec=["disconnect"])
    if disconnect_fails:
        bus.disconnect.side_effect = [ConnectionError("Torque disable failed"), None]
    worker.bus = bus
    worker._bus_lock = asyncio.Lock()
    worker._device_claim = MagicMock()

    await worker._cleanup()

    expected_calls = [call(), call(disable_torque=False)] if disconnect_fails else [call()]
    assert bus.disconnect.call_args_list == expected_calls
    assert worker.bus is None
    worker._device_claim.close.assert_called_once()


async def test_cleanup_releases_device_claim_when_both_disconnect_attempts_fail() -> None:
    worker = SO101SetupWorker.__new__(SO101SetupWorker)
    bus = MagicMock(spec=["disconnect"])
    bus.disconnect.side_effect = ConnectionError("Port unavailable")
    worker.bus = bus
    worker._bus_lock = asyncio.Lock()
    worker._device_claim = MagicMock()

    await worker._cleanup()

    assert bus.disconnect.call_args_list == [call(), call(disable_torque=False)]
    assert worker.bus is None
    worker._device_claim.close.assert_called_once()
