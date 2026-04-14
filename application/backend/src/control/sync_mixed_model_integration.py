from physicalai.data import Observation

from control.inference_poller import InferencePoller
from control.queue_mixer import QueueMixer
from workers.model_worker import ModelWorker


class SyncMixedModelIntegration:
    model_worker: ModelWorker
    queue_mixer: QueueMixer
    inference_poller: InferencePoller
    fps: int

    def __init__(self, model_worker: ModelWorker, fps: int):
        self.model_worker = model_worker
        self.fps = fps

        # Communication layer to model worker. It ensures no queue.
        self.inference_poller = InferencePoller(self.model_worker.observation_queue, self.model_worker.output_queue)

        # Queue mixer to move to new inference result while still executing previous.
        self.queue_mixer = QueueMixer(lerp_duration=self.fps)

    def select_action(self, observation: Observation) -> list[list[float]] | None:
        if self.inference_poller.has_result():
            inference_result = self.inference_poller.get_result()
            offset = int(inference_result.time * self.fps)
            self.queue_mixer.add(inference_result.data, offset)
            self.queue_mixer.lerp_duration = max(offset, 1)

        if not self.inference_poller.busy:
            self.inference_poller.run_inference(observation)

        if not self.queue_mixer.empty():
            return self.queue_mixer.pop().tolist()

        return None

    def reset(self) -> None:
        self.inference_poller.reset()

    async def setup(self) -> None:
        # Worker is already running (pre-spawned); just wait for model to load.
        await self.model_worker.wait_for_loading_to_complete()

    def teardown(self) -> None:
        self.inference_poller.reset()
