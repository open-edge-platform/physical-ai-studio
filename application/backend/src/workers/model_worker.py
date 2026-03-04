from workers.inference.inference_result import InferenceResult
import queue
import time
from models.utils import load_inference_model
from schemas import Model
from loguru._logger import Logger as LoguruLogger
from physicalai.inference import InferenceModel

from .base import BaseProcessWorker

from multiprocessing.synchronize import Event as EventClass
import multiprocessing as mp
from loguru import logger


class ModelWorker(BaseProcessWorker):
    ROLE: str = "ModelWorker"

    backend: str
    model: Model
    inference_model: InferenceModel

    def __init__(self,
                 model: Model,
        backend: str,
        stop_event: EventClass,
        #logger_: LoguruLogger
    ):
        self.observation_queue = mp.Queue()
        self.output_queue = mp.Queue()
        super().__init__(stop_event=stop_event, queues_to_cancel=[
            self.observation_queue,
            self.output_queue
        ])
        self.model = model
        self.backend = backend
        self.close_event = mp.Event()

    def setup(self) -> None:
        """Load model."""
        self.inference_model = load_inference_model(self.model, backend=self.backend)
        logger.info("Inference model loaded")

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
                logger.info("Trying to run inference")
                start_time = time.perf_counter()
                output = self.inference_model.select_action(observation)[0].detach().cpu()
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                logger.info(f"Done running inference ({elapsed_time}): {output.shape}")
                self.output_queue.put(InferenceResult(time=elapsed_time, data=output))
            except queue.Empty:
                continue

        logger.info("Inference stopped")

    def teardown(self) -> None:
        self.observation_queue.close()
        self.output_queue.close()
