from schemas.hardware import InferenceDevice
import asyncio
import multiprocessing as mp
import queue
import time
from multiprocessing.synchronize import Event as EventClass

from loguru import logger

from control.inference_result import InferenceResult
from schemas import Model

from .base import BaseProcessWorker


class ModelWorker(BaseProcessWorker):
    ROLE: str = "ModelWorker"

    observation_queue: mp.Queue
    output_queue: mp.Queue
    model_loaded_event: EventClass
    unload_event: EventClass

    def __init__(self, model: Model, inference_device: InferenceDevice, stop_event: EventClass):
        self.observation_queue = mp.Queue()
        self.output_queue = mp.Queue()
        super().__init__(
            stop_event=stop_event,
            queues_to_cancel=[self.observation_queue, self.output_queue],
        )
        self.model = model
        self.inference_device = inference_device
        self.model_loaded_event = mp.Event()

    @property
    def is_loaded(self) -> bool:
        return self.model_loaded_event.is_set()

    def unload_model(self) -> None:
        """Signal the worker to stop inference and return to idle."""
        self.unload_event.set()

    async def wait_for_loading_to_complete(self) -> None:
        await asyncio.to_thread(self.model_loaded_event.wait)

    async def setup(self) -> None:
        from models.utils import load_inference_model

        logger.info(f"Loading model: {self.model.name} ({self.inference_device})")
        self.inference_model = load_inference_model(self.model, inference_device=self.inference_device)
        logger.info("Model loaded.")
        self.model_loaded_event.set()

    async def run_loop(self) -> None:
        """Idle → load → inference → idle cycle."""
        while not self.should_stop():
            # Inference loop until unload is requested
            try:
                observation = self.observation_queue.get(timeout=1)
                start_time = time.perf_counter()
                output = self.inference_model.predict_action_chunk(observation.to_numpy().to_dict(flatten=False))[0]
                elapsed_time = time.perf_counter() - start_time
                logger.debug(f"Inference: ({elapsed_time}): {output.shape}")
                self.output_queue.put(InferenceResult(time=elapsed_time, data=output))
            except queue.Empty:
                continue

        logger.info("Inference stopped, unloading model.")
        del self.inference_model

    async def teardown(self) -> None:
        self.observation_queue.close()
        self.output_queue.close()
        await super().teardown()
