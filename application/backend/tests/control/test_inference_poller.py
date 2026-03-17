import time
from multiprocessing import Queue

import numpy as np
import pytest
from physicalai.data import Observation

from control.inference_poller import InferencePoller
from control.inference_result import InferenceResult


@pytest.fixture
def poller():
    input_queue = Queue()
    output_queue = Queue()
    yield InferencePoller(input_queue, output_queue)
    input_queue.close()
    input_queue.cancel_join_thread()
    output_queue.close()
    output_queue.cancel_join_thread()


class TestInferencePoller:
    def test_is_busy_when_inference_active(self, poller):
        assert not poller.busy
        assert poller.run_inference(Observation())
        assert poller.busy

    def test_getting_results_gets_result(self, poller: InferencePoller):
        assert poller.run_inference(Observation())
        poller.output_queue.put_nowait(InferenceResult(time=0, data=np.array([1, 2, 3, 4])))
        time.sleep(0.01)  # let feeder thread flush to pipe
        result = poller.get_result()
        assert result.time == 0
        assert result.data.tolist() == [1, 2, 3, 4]

    def test_getting_results_sets_busy_to_false(self, poller: InferencePoller):
        assert poller.run_inference(Observation())
        poller.output_queue.put_nowait(InferenceResult(time=0, data=np.array([])))
        time.sleep(0.01)  # let feeder thread flush to pipe
        poller.get_result()
        assert not poller.busy

    def test_reset(self, poller: InferencePoller):
        assert poller.run_inference(Observation())
        poller.output_queue.put_nowait(InferenceResult(time=0, data=np.array([])))
        time.sleep(0.01)  # let feeder thread flush to pipe
        assert poller.has_result()
        poller.reset()
        assert not poller.has_result()

    def test_reset_with_pending_inference(self, poller: InferencePoller):
        """Resetting with pending inference will allow pending inference to come in."""
        assert poller.run_inference(Observation())
        poller.reset()
        assert not poller.has_result()
        assert poller.busy
        poller.output_queue.put_nowait(InferenceResult(time=0, data=np.array([])))
        time.sleep(0.01)  # let feeder thread flush to pipe
        assert poller.has_result()
