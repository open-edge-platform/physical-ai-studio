from multiprocessing.synchronize import Event as EventClass

from loguru import logger

from schemas import Model
from workers.inference.queue_mixer import QueueMixer
from workers.inference.inference_poller import InferencePoller
from workers.model_worker import ModelWorker

from physicalai.data import Observation

class SyncMixedModelIntegration:
    model_worker: ModelWorker
    queue_mixer: QueueMixer
    inference_poller: InferencePoller
    fps: int

    def __init__(self, model: Model, backend: str, stop_event: EventClass, fps: int):
        self.model_worker = ModelWorker(
            model=model,
            backend=backend,
            stop_event=stop_event,
        )
        self.fps = fps

        # Communication layer to model worker. It ensures no queue.
        self.inference_poller = InferencePoller(self.model_worker.observation_queue, self.model_worker.output_queue)

        # Queue mixer to move to new inference result while still executing previous.
        self.queue_mixer = QueueMixer(lerp_duration=12)
        # TODO: Remove hardcode and use running average of inference time?

    def select_action(self, observation: Observation) -> list[list[float]] | None:
        if self.inference_poller.has_result():
            inference_result = self.inference_poller.get_result()
            offset = int(inference_result.time * self.fps)
            logger.debug(
                f"Got inference from inference_poller: {inference_result.data.shape} with offset {offset}"
            )
            self.queue_mixer.add(inference_result.data, offset)
            self.queue_mixer.lerp_duration = offset  # inference time should be a good guide for now.

        if not self.inference_poller.busy:
            self.inference_poller.run_inference(observation)

        if not self.queue_mixer.empty():
            return self.queue_mixer.pop().tolist()

        return None

    def reset(self) -> None:
       self.inference_poller.reset()

    async def setup(self) -> None:
        self.model_worker.start()

    def teardown(self) -> None:
        self.model_worker.stop()
        self.model_worker.join(timeout=5)
