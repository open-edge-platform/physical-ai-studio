from multiprocessing.synchronize import Event as EventClass

from physicalai.data import Observation

from schemas import Model
from workers.inference.inference_poller import InferencePoller
from workers.inference.queue_mixer import QueueMixer
from workers.model_worker import ModelWorker


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
        self.queue_mixer = QueueMixer(lerp_duration=self.fps)

    def select_action(self, observation: Observation) -> list[list[float]] | None:
        if self.inference_poller.has_result():
            inference_result = self.inference_poller.get_result()
            offset = int(inference_result.time * self.fps)
            self.queue_mixer.add(inference_result.data, offset + 1)
            self.queue_mixer.lerp_duration = max(offset, 10)  # inference time should be a good guide for now.

        if not self.inference_poller.busy:
            self.inference_poller.run_inference(observation)

        if not self.queue_mixer.empty():
            return self.queue_mixer.pop().tolist()

        return None

    def reset(self) -> None:
        self.inference_poller.reset()

    async def setup(self) -> None:
        self.model_worker.start()
        await self.model_worker.wait_for_loading_to_complete()

    def teardown(self) -> None:
        self.model_worker.stop()
        self.model_worker.join(timeout=5)
