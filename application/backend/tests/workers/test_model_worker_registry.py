import asyncio
import multiprocessing as mp
from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest

from workers.model_worker_registry import ModelWorkerRegistry


def _make_mock_worker() -> MagicMock:
    """Return a mock ModelWorker with a real mp.Event for model_loaded_event."""
    worker = MagicMock()
    worker.model_loaded_event = mp.Event()
    return worker


@pytest.fixture
def stop_event():
    return mp.Event()


@pytest.fixture
def registry(stop_event):
    """Registry with 2 pre-spawned mock workers (no real processes)."""
    with patch("workers.model_worker_registry.ModelWorker", side_effect=lambda **_: _make_mock_worker()):
        return ModelWorkerRegistry(max_workers=2, stop_event=stop_event)


@pytest.fixture
def single_worker_registry(stop_event):
    with patch("workers.model_worker_registry.ModelWorker", side_effect=lambda **_: _make_mock_worker()):
        return ModelWorkerRegistry(max_workers=1, stop_event=stop_event)


class TestModelWorkerRegistryInit:
    def test_pre_spawns_correct_number_of_workers(self, registry):
        assert len(registry._workers) == 2

    def test_all_workers_start_idle(self, registry):
        assert len(registry._idle) == 2
        assert len(registry._busy) == 0

    def test_workers_are_started(self, registry):
        for worker in registry._workers.values():
            worker.start.assert_called_once()


class TestModelWorkerRegistryAcquire:
    def test_acquire_returns_worker_id_and_worker(self, registry, test_model):
        worker_id, worker = asyncio.run(registry.acquire(test_model, "torch"))
        assert isinstance(worker_id, UUID)
        assert worker is registry._workers[worker_id]

    def test_acquire_moves_worker_to_busy(self, registry, test_model):
        worker_id, _ = asyncio.run(registry.acquire(test_model, "torch"))
        assert worker_id in registry._busy
        assert worker_id not in registry._idle

    def test_acquire_calls_load_model_on_worker(self, registry, test_model):
        _, worker = asyncio.run(registry.acquire(test_model, "torch"))
        worker.load_model.assert_called_once_with(test_model, "torch")

    def test_acquire_two_workers_exhausts_pool(self, registry, test_model):
        asyncio.run(registry.acquire(test_model, "torch"))
        asyncio.run(registry.acquire(test_model, "torch"))
        assert len(registry._idle) == 0
        assert len(registry._busy) == 2

    def test_acquire_raises_when_no_idle_workers(self, single_worker_registry, test_model):
        asyncio.run(single_worker_registry.acquire(test_model, "torch"))
        with pytest.raises(ValueError, match="No idle model workers available"):
            asyncio.run(single_worker_registry.acquire(test_model, "torch"))


class TestModelWorkerRegistryRelease:
    def test_release_returns_worker_to_idle(self, single_worker_registry, test_model):
        worker_id, _ = asyncio.run(single_worker_registry.acquire(test_model, "torch"))
        asyncio.run(single_worker_registry.release(worker_id))
        assert worker_id in single_worker_registry._idle
        assert worker_id not in single_worker_registry._busy

    def test_release_calls_unload_model(self, single_worker_registry, test_model):
        worker_id, worker = asyncio.run(single_worker_registry.acquire(test_model, "torch"))
        asyncio.run(single_worker_registry.release(worker_id))
        worker.unload_model.assert_called_once()

    def test_released_worker_can_be_acquired_again(self, single_worker_registry, test_model):
        worker_id, _ = asyncio.run(single_worker_registry.acquire(test_model, "torch"))
        asyncio.run(single_worker_registry.release(worker_id))
        new_id, _ = asyncio.run(single_worker_registry.acquire(test_model, "onnx"))
        assert new_id == worker_id

    def test_release_unknown_id_is_noop(self, registry):
        from uuid import uuid4

        asyncio.run(registry.release(uuid4()))  # should not raise


class TestModelWorkerRegistryGet:
    def test_get_returns_correct_worker(self, registry):
        worker_id = next(iter(registry._workers))
        assert registry.get(worker_id) is registry._workers[worker_id]

    def test_get_unknown_id_returns_none(self, registry):
        from uuid import uuid4

        assert registry.get(uuid4()) is None


class TestModelWorkerRegistryShutdown:
    def test_shutdown_all_terminates_workers(self, registry):
        workers = list(registry._workers.values())
        asyncio.run(registry.shutdown_all())
        for worker in workers:
            worker.terminate.assert_called_once()

    def test_shutdown_all_clears_state(self, registry):
        asyncio.run(registry.shutdown_all())
        assert len(registry._workers) == 0
        assert len(registry._idle) == 0
        assert len(registry._busy) == 0

    def test_context_manager_calls_shutdown_all(self, stop_event):
        with patch("workers.model_worker_registry.ModelWorker", side_effect=lambda **_: _make_mock_worker()):

            async def run():
                async with ModelWorkerRegistry(max_workers=1, stop_event=stop_event) as reg:
                    return list(reg._workers.values())

            workers = asyncio.run(run())

        for worker in workers:
            worker.terminate.assert_called_once()
