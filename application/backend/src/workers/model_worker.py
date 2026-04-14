import asyncio
import multiprocessing as mp
import queue
import time
from multiprocessing.synchronize import Event as EventClass

from loguru import logger
from physicalai.inference import InferenceModel

from control.inference_result import InferenceResult
from models.utils import load_inference_model
from schemas import Model

from .base import BaseProcessWorker


class ModelWorker(BaseProcessWorker):
    ROLE: str = "ModelWorker"

    inference_model: InferenceModel
    command_queue: mp.Queue
    observation_queue: mp.Queue
    output_queue: mp.Queue
    model_loaded_event: EventClass
    unload_event: EventClass

    def __init__(self, stop_event: EventClass):
        self.command_queue = mp.Queue()
        self.observation_queue = mp.Queue()
        self.output_queue = mp.Queue()
        super().__init__(
            stop_event=stop_event,
            queues_to_cancel=[self.command_queue, self.observation_queue, self.output_queue],
        )
        self.model_loaded_event = mp.Event()
        self.unload_event = mp.Event()

    @property
    def is_loaded(self) -> bool:
        return self.model_loaded_event.is_set()

    def load_model(self, model: Model, backend: str) -> None:
        """Send a load command to the worker process."""
        self.command_queue.put(("load", model, backend))

    def unload_model(self) -> None:
        """Signal the worker to stop inference and return to idle."""
        self.unload_event.set()

    async def wait_for_loading_to_complete(self) -> None:
        await asyncio.to_thread(self.model_loaded_event.wait)

    async def run_loop(self) -> None:
        """Idle → load → inference → idle cycle."""
        while not self.should_stop():
            # Wait for a load command
            try:
                cmd = self.command_queue.get(timeout=1)
            except queue.Empty:
                continue

            if cmd[0] != "load":
                continue

            _, model, backend = cmd
            logger.info(f"Loading model: {model.name} ({backend})")
            self.inference_model = load_inference_model(model, backend=backend)
            logger.info("Model loaded.")
            self.model_loaded_event.set()

            # Inference loop until unload is requested
            while not self.should_stop() and not self.unload_event.is_set():
                try:
                    observation = self.observation_queue.get(timeout=1)
                    start_time = time.perf_counter()
                    output = self.inference_model.select_action(observation)[0]
                    elapsed_time = time.perf_counter() - start_time
                    logger.debug(f"Inference: ({elapsed_time}): {output.shape}")
                    self.output_queue.put(InferenceResult(time=elapsed_time, data=output))
                except queue.Empty:
                    continue

            logger.info("Inference stopped, unloading model.")
            self.unload_event.clear()
            self.model_loaded_event.clear()
            del self.inference_model

    async def teardown(self) -> None:
        self.command_queue.close()
        self.observation_queue.close()
        self.output_queue.close()
        await super().teardown()
