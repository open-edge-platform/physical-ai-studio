import asyncio
import multiprocessing as mp
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from workers.model_worker import ModelWorker


@pytest.fixture
def stop_event():
    return mp.Event()


class TestModelWorker:
    def test_worker_starts_idle(self, stop_event):
        worker = ModelWorker(stop_event=stop_event)
        assert not worker.is_loaded

    def test_load_model_puts_command_on_queue(self, stop_event, test_model):
        worker = ModelWorker(stop_event=stop_event)
        worker.load_model(test_model, backend="torch")

        cmd = worker.command_queue.get(timeout=2)
        assert cmd[0] == "load"
        assert cmd[1].name == test_model.name
        assert cmd[2] == "torch"

        worker.command_queue.cancel_join_thread()
        worker.observation_queue.cancel_join_thread()
        worker.output_queue.cancel_join_thread()

    def test_unload_model_sets_unload_event(self, stop_event):
        worker = ModelWorker(stop_event=stop_event)
        assert not worker.unload_event.is_set()
        worker.unload_model()
        assert worker.unload_event.is_set()

    def test_is_loaded_reflects_model_loaded_event(self, stop_event):
        worker = ModelWorker(stop_event=stop_event)
        assert not worker.is_loaded
        worker.model_loaded_event.set()
        assert worker.is_loaded
        worker.model_loaded_event.clear()
        assert not worker.is_loaded

    def test_full_load_unload_cycle(self, stop_event, test_model):
        """Worker loads a model, signals loaded, then unloads on request.

        Runs ``run_loop()`` in a thread (same process) so that
        ``unittest.mock.patch`` is visible — ``mp.Process.start()`` uses
        *spawn* on macOS which re-imports the module in a fresh interpreter
        and loses the mock.
        """
        fake_output = MagicMock()
        fake_output.shape = (6,)
        fake_inference_model = MagicMock()
        fake_inference_model.select_action.return_value = [fake_output]

        with patch("workers.model_worker.load_inference_model", return_value=fake_inference_model):
            worker = ModelWorker(stop_event=stop_event)
            worker.load_model(test_model, backend="torch")

            loop = asyncio.new_event_loop()
            t = threading.Thread(target=loop.run_until_complete, args=(worker.run_loop(),), daemon=True)
            t.start()

            try:
                loaded = worker.model_loaded_event.wait(timeout=10)
                assert loaded, "Model did not load within timeout"
                assert worker.is_loaded

                worker.unload_model()
                for _ in range(50):
                    time.sleep(0.1)
                    if not worker.model_loaded_event.is_set():
                        break
                else:
                    pytest.fail("Worker did not clear model_loaded_event after unload")

                assert not worker.is_loaded
            finally:
                stop_event.set()
                t.join(timeout=5)
                loop.close()
                worker.command_queue.cancel_join_thread()
                worker.observation_queue.cancel_join_thread()
                worker.output_queue.cancel_join_thread()
