from __future__ import annotations

import asyncio
import os
import signal
import time
from typing import TYPE_CHECKING

import pytest

from workers.base import WorkerStartupError
from workers.rate import Rate

from .fakes import FakeProcessWorker, FakeThreadWorker

if TYPE_CHECKING:
    from pathlib import Path

    from .fakes import Mode

WORKER_TYPES = [FakeThreadWorker, FakeProcessWorker]


def _build(worker_type: type[FakeThreadWorker | FakeProcessWorker], tmp_path: Path, mode: Mode = "ok"):
    return worker_type(tmp_path, mode)


@pytest.mark.parametrize("worker_type", WORKER_TYPES)
async def test_context_reaches_run_loop_and_is_released_on_exit(worker_type, tmp_path: Path) -> None:
    worker = _build(worker_type, tmp_path)

    async with worker:
        assert (tmp_path / "acquired").exists()
        assert not (tmp_path / "released").exists()
        await _wait_for(tmp_path / "context")

    assert (tmp_path / "context").read_text() == f"device-of-{worker.name}"
    assert (tmp_path / "released").exists()
    assert not worker.is_alive()
    assert worker.failure is None


@pytest.mark.parametrize("worker_type", WORKER_TYPES)
async def test_failure_to_acquire_raises_in_the_parent(worker_type, tmp_path: Path) -> None:
    worker = _build(worker_type, tmp_path, "fail_acquire")

    with pytest.raises(WorkerStartupError, match="no device"):
        async with worker:
            pytest.fail("the block must not run when acquiring failed")

    # The lifecycle's own cleanup still ran.
    assert (tmp_path / "released").exists()
    assert not worker.is_alive()


@pytest.mark.parametrize("worker_type", WORKER_TYPES)
async def test_failure_while_running_is_reported_not_raised(worker_type, tmp_path: Path) -> None:
    worker = _build(worker_type, tmp_path, "fail_run")

    async with worker:
        await asyncio.wait_for(worker.wait_stopped(), timeout=10)
        assert worker.failure is not None
        assert "lost device" in worker.failure

    assert (tmp_path / "released").exists()


async def test_process_that_will_not_stop_is_killed(tmp_path: Path) -> None:
    worker = FakeProcessWorker(tmp_path, "ignore_stop")
    worker.stop_timeout_s = 0.2

    async with worker:
        await _wait_for(tmp_path / "context")

    assert not worker.is_alive()
    # Killed mid-loop, so its lifecycle never got to release.
    assert not (tmp_path / "released").exists()


async def test_thread_that_will_not_stop_is_left_behind(tmp_path: Path) -> None:
    worker = FakeThreadWorker(tmp_path, "ignore_stop")
    worker.stop_timeout_s = 0.2

    started = time.monotonic()
    async with worker:
        await _wait_for(tmp_path / "context")

    # The parent gave up waiting instead of hanging on the thread.
    assert time.monotonic() - started < 1.5
    assert worker.is_alive()
    await asyncio.to_thread(worker.join, 5)


async def test_process_that_dies_before_ready_fails_fast(tmp_path: Path) -> None:
    worker = FakeProcessWorker(tmp_path, "exit_hard")

    started = time.monotonic()
    with pytest.raises(WorkerStartupError, match="exited with code 3"):
        async with worker:
            pass

    assert time.monotonic() - started < worker.ready_timeout_s / 2


@pytest.mark.parametrize("worker_type", WORKER_TYPES)
async def test_startup_times_out_and_stops_the_worker(worker_type, tmp_path: Path) -> None:
    worker = _build(worker_type, tmp_path, "slow_ready")
    worker.ready_timeout_s = 0.2

    with pytest.raises(WorkerStartupError, match="not ready within"):
        async with worker:
            pass

    assert not worker.is_alive()
    assert (tmp_path / "released").exists()


@pytest.mark.parametrize("worker_type", WORKER_TYPES)
async def test_cancelled_startup_stops_the_worker(worker_type, tmp_path: Path) -> None:
    worker = _build(worker_type, tmp_path, "slow_ready")

    async def enter() -> None:
        async with worker:
            pytest.fail("the block must not run when startup was cancelled")

    task = asyncio.create_task(enter())
    await _wait_for(tmp_path / "acquired")
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert not worker.is_alive()
    assert (tmp_path / "released").exists()


async def test_sigterm_stops_a_process_gracefully(tmp_path: Path) -> None:
    worker = FakeProcessWorker(tmp_path)

    async with worker:
        await _wait_for(tmp_path / "context")
        assert worker.pid is not None
        os.kill(worker.pid, signal.SIGTERM)
        await asyncio.wait_for(worker.wait_stopped(), timeout=10)

    assert (tmp_path / "released").exists()
    assert worker.failure is None
    assert worker.exitcode == 0


class _FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


async def test_rate_keeps_a_fixed_schedule(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _FakeClock()
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock.now += seconds

    monkeypatch.setattr("workers.rate.asyncio.sleep", fake_sleep)
    rate = Rate(10.0, name="test", clock=clock)

    for work in (0.02, 0.05, 0.0):
        clock.now += work
        await rate.sleep()

    # Each tick sleeps whatever is left of its 100ms, so the work does not add up.
    assert sleeps == pytest.approx([0.08, 0.05, 0.1])


async def test_rate_restarts_the_schedule_after_an_overrun(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _FakeClock()
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock.now += seconds

    monkeypatch.setattr("workers.rate.asyncio.sleep", fake_sleep)
    rate = Rate(10.0, name="test", clock=clock)

    clock.now += 0.35  # three periods late
    await rate.sleep()
    clock.now += 0.01
    await rate.sleep()

    # No burst of catch-up ticks: the overrun yields once, then a normal period follows.
    assert sleeps == pytest.approx([0.0, 0.09])


def test_rate_rejects_a_non_positive_frequency() -> None:
    with pytest.raises(ValueError, match="positive"):
        Rate(0, name="test")


async def _wait_for(path: Path, limit_s: float = 10.0) -> None:
    deadline = time.monotonic() + limit_s
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"{path.name} was never written")
        await asyncio.sleep(0.01)
