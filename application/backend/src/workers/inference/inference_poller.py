from multiprocessing import Queue

from physicalai.data import Observation

from workers.inference.inference_result import InferenceResult


class InferencePoller:
    """An inference poller class that keeps makes sure that only one inference call is added to queue at once."""

    busy: bool = False
    observation_queue: Queue
    output_queue: Queue

    def __init__(self, observation_queue: Queue, output_queue: Queue):
        self.observation_queue = observation_queue
        self.output_queue = output_queue

    def run_inference(self, observation: Observation) -> bool:
        if self.busy:
            return False
        self.observation_queue.put(observation)
        self.busy = True
        return True

    def has_result(self) -> bool:
        """Check output queue for results..."""
        return not self.output_queue.empty()

    def get_result(self) -> InferenceResult:
        self.busy = False
        return self.output_queue.get_nowait()
