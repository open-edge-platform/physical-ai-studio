from schemas.hardware import InferenceDevice
import asyncio
import ctypes
import multiprocessing as mp
import time
from multiprocessing.synchronize import Event as EventClass

from loguru import logger
from control.data_registry import EnvironmentDataRegistry
from control.utils import format_observation_for_model, get_observation_from_manifest
from schemas.model import Model
from workers.base import BaseProcessWorker, run_at_frequency


class ModelIntegration(BaseProcessWorker):
    ROLE = "ModelIntegrationWorker"

    _child_workers: list[BaseProcessWorker] = []

    def __init__(
        self,
        model: Model,
        inference_device: InferenceDevice,
        data_manifest: EnvironmentDataRegistry,
        mp_terminate_event: EventClass,
        event_queue: mp.Queue,
    ):
        super().__init__(stop_event=mp_terminate_event, queues_to_cancel=[])
        self.loaded_event = mp.Event()
        self.data_manifest = data_manifest
        self.inference_device = inference_device
        self.model = model
        self.event_queue = event_queue
        self.model_integration = None
        self.is_running = False
        self.fps = 30  # TODO FPS
        self._task_buf = mp.Array(ctypes.c_char, 256)
        self._start_task_event = mp.Event()
        self._stop_task_event = mp.Event()
        self.chunk: list[list[float]] = []

    async def setup(self) -> None:
        from models.utils import load_inference_model

        self.inference_model = load_inference_model(self.model, self.inference_device)
        self.loaded_event.set()

    def start_task(self, task: str) -> None:
        self._task_buf.get_obj().value = task.encode("utf-8")[:255]  # type: ignore[misc, assignment]
        self._start_task_event.set()

    def stop_task(self) -> None:
        self._stop_task_event.set()

    async def run_loop(self) -> None:
        while not self.should_stop():
            async with run_at_frequency(self.fps):
                await asyncio.gather(
                    self._handle_start_task(),
                    self._handle_stop_task(),
                )

                if self.inference_model and self.is_running:
                    if len(self.chunk) == 0:
                        obs = get_observation_from_manifest(self.data_manifest)
                        observation = format_observation_for_model(obs, self.data_manifest, self.get_task())
                        start = time.perf_counter()
                        self.chunk = list(self.inference_model.predict_action_chunk(observation.to_numpy().to_dict(flatten=False)))
                        elapsed = time.perf_counter() - start
                        logger.info(f"Inference: ({elapsed})")

                    action = self.chunk.pop(0)
                    with self.data_manifest.robot.actions.get_lock():
                        self.data_manifest.robot.actions.get_obj()[:] = action

    def get_task(self) -> str:
        return bytes(self._task_buf.get_obj()).rstrip(b"\x00").decode()

    async def teardown(self) -> None:
        for worker in self._child_workers:
            worker.stop()

        if self.model_integration:
            self.model_integration.teardown()

    async def _handle_start_task(self) -> None:
        if self._start_task_event.is_set():
            self._start_task_event.clear()
            self.chunk.clear()
            self.is_running = True
            self.event_queue.put_nowait(
                {
                    "event": "start_task",
                    "state": {"is_running": True},
                }
            )

    async def _handle_stop_task(self) -> None:
        if self._stop_task_event.is_set():
            self._stop_task_event.clear()
            self.is_running = False
            self.event_queue.put_nowait(
                {
                    "event": "stop_task",
                    "state": {"is_running": False},
                }
            )
