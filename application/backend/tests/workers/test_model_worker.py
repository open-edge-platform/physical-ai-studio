import multiprocessing as mp
from unittest.mock import MagicMock, patch

import pytest

from schemas.hardware import InferenceBackend, InferenceDevice
from workers.model_worker import ModelWorker


@pytest.fixture
def stop_event():
    return mp.Event()


class TestModelWorker:
    def test_worker_starts_idle(self, stop_event, test_model):
        inference_device = InferenceDevice(backend=InferenceBackend.TORCH, device="xpu:0")
        worker = ModelWorker(model=test_model, inference_device=inference_device, stop_event=stop_event)
        assert not worker.is_loaded

    def test_is_loaded_reflects_model_loaded_event(self, stop_event, test_model):
        inference_device = InferenceDevice(backend=InferenceBackend.TORCH, device="xpu:0")
        worker = ModelWorker(model=test_model, inference_device=inference_device, stop_event=stop_event)
        assert not worker.is_loaded
        worker.model_loaded_event.set()
        assert worker.is_loaded
        worker.model_loaded_event.clear()
        assert not worker.is_loaded

    def test_model_loads_on_process_start(self, stop_event, test_model):
        """Worker loads the model in setup() when the process starts."""
        fake_inference_model = MagicMock()

        with patch("models.utils.load_inference_model", return_value=fake_inference_model):
            inference_device = InferenceDevice(backend=InferenceBackend.TORCH, device="xpu:0")
            worker = ModelWorker(model=test_model, inference_device=inference_device, stop_event=stop_event)
            worker.start()

            try:
                loaded = worker.model_loaded_event.wait(timeout=10)
                assert loaded, "Model did not load within timeout"
                assert worker.is_loaded
            finally:
                stop_event.set()
                worker.join(timeout=5)
                worker.observation_queue.cancel_join_thread()
                worker.output_queue.cancel_join_thread()
