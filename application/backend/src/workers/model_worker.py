import asyncio
import multiprocessing as mp
import queue
import time
from multiprocessing.synchronize import Event as EventClass

from loguru import logger
from physicalai.inference import InferenceModel

from models.utils import load_inference_model
from schemas import Model
from workers.inference.inference_result import InferenceResult

from .base import BaseProcessWorker


class ModelWorker(BaseProcessWorker):
    ROLE: str = "ModelWorker"

    backend: str
    model: Model
    inference_model: InferenceModel
    observation_queue: mp.Queue
    output_queue: mp.Queue
    model_loaded_event: EventClass

    def __init__(self, model: Model, backend: str, stop_event: EventClass):
        self.observation_queue = mp.Queue()
        self.output_queue = mp.Queue()
        super().__init__(stop_event=stop_event, queues_to_cancel=[self.observation_queue, self.output_queue])
        self.model = model
        self.backend = backend
        self.close_event = mp.Event()
        self.model_loaded_event = mp.Event()

    async def setup(self) -> None:
        """Load model."""
        self.inference_model = load_inference_model(self.model, backend=self.backend)
        logger.info("Model loaded.")
        self.model_loaded_event.set()

    async def wait_for_loading_to_complete(self) -> None:
        await asyncio.to_thread(self.model_loaded_event.wait)

    def stop(self) -> None:
        """Stop model worker."""
        self.close_event.set()

    async def run_loop(self) -> None:
        """Run inference."""
        logger.info("Inference-Run loop")
        # Should stop is a check for global shutdown, close event for worker shutdown.
        while not self.should_stop() and not self.close_event.is_set():
            try:
                observation = self.observation_queue.get(timeout=1)
                start_time = time.perf_counter()
                # batch size is 1, so taking first batch 0
                output = self.inference_model.select_action(observation)[0].detach().cpu().numpy()
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                logger.debug(f"Inference: ({elapsed_time}): {output.shape}")
                self.output_queue.put(InferenceResult(time=elapsed_time, data=output))
            except queue.Empty:
                continue

        logger.info("Inference stopped")

    async def teardown(self) -> None:
        self.observation_queue.close()
        self.output_queue.close()
        await super().teardown()
