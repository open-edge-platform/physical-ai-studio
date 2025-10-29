from loguru import logger
import multiprocessing as mp
from schemas import Dataset, Model
from schemas.jobs import TrainJobPayload

class TrainingManager:
    worker: TrainingWorker | None = None
    queue: mp.Queue()

    def __init__(self, stop_event: mp.Event()):
        self.stop_event = stop_event
        self.queue = mp.Queue()

    def train(self, payload: TrainJobPayload):
        if self.worker is not None:
            raise ValueError("Cannot start another training.")

        settings = get_settings()

        id = uuid4()
        self.model = Model(
            id=id,
            project_id=payload.project_id,
            dataset_id=payload.dataset_id,
            path=str(settings.models_dir / str(id)),
            name=payload.model_name,
            policy=payload.policy,
            properties={},
        )

        dataset = Dataset(
            id="b55bf695-8927-4ad9-96a1-88a6a0f33326",
            name="block-placement",
            path="/home/ronald/.cache/geti_action/datasets/block-placement",
            project_id="cf8062f8-7354-4643-8899-9be8d57ab2ce",
        )

        self.worker = TrainingWorker(
            stop_event=self.stop_event,
            model=self.model,
            dataset=dataset,
            queue=self.queue,
        )
        self.worker.start()

    async def event_processor(self):
        try:
            while not self.stop_event.is_set():
                try:
                    event, payload = self.queue.get_nowait()
                    logger.info(f"{event} {payload}")
                    if event == EventType.CHECKPOINT:
                        self.model.path = payload
                        # Save model
                    if event == EventType.DONE:
                        # Update job?
                        pass
                    if event == EventType.PROGRESS:
                        # Update job with progress
                        pass
                except mp.queues.Empty:
                    await asyncio.sleep(0.05)
        except Exception as e:
            print(f"Outgoing task stopped: {e}")
