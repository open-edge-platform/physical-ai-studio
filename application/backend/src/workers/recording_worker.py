import asyncio
import ctypes
import multiprocessing as mp
from multiprocessing.synchronize import Event as EventClass
from pathlib import Path

from control.data_registry import EnvironmentDataRegistry
from control.utils import build_lerobot_dataset_features, format_observation_for_dataset, get_observation_from_manifest
from internal_datasets.mutations.recording_mutation import RecordingMutation
from schemas.dataset import Dataset
from workers.base import BaseProcessWorker, run_at_frequency

RECORDING_FPS = 30


class RecordingWorker(BaseProcessWorker):
    ROLE = "RecordingWorker"

    recording_mutation: RecordingMutation | None = None

    def __init__(
        self,
        dataset: Dataset,
        data_manifest: EnvironmentDataRegistry,
        mp_terminate_event: EventClass,
        event_queue: mp.Queue,
    ):
        super().__init__(stop_event=mp_terminate_event, queues_to_cancel=[])
        self.loaded_event = mp.Event()
        self.dataset_config = dataset
        self.event_queue = event_queue
        self.data_manifest = data_manifest
        self._start_event = mp.Event()
        self._save_event = mp.Event()
        self._discard_event = mp.Event()
        self._task_buf = mp.Array(ctypes.c_char, 256)
        self._is_recording = False
        self._episodes_recorded = 0

    def start_episode(self, task: str) -> None:
        self._task_buf.get_obj().value = task.encode("utf-8")[:255]  # type: ignore[misc, assignment]
        self._start_event.set()

    def save_episode(self) -> None:
        self._save_event.set()

    def discard_episode(self) -> None:
        self._discard_event.set()

    async def setup(self) -> None:
        from internal_datasets.lerobot.lerobot_dataset import InternalLeRobotDataset

        self.dataset = InternalLeRobotDataset(Path(self.dataset_config.path))
        features = build_lerobot_dataset_features(self.data_manifest)
        self.fps = self.dataset.get_fps() or RECORDING_FPS
        self.recording_mutation = self.dataset.start_recording_mutation(
            fps=self.fps,
            features=features,
            robot_type=self.data_manifest.robot.type,
        )
        self.loaded_event.set()

    async def run_loop(self) -> None:
        while not self.should_stop():
            async with run_at_frequency(self.fps):
                await asyncio.gather(
                    self._handle_start_recording(),
                    self._handle_save_episode(),
                    self._handle_discard_episode(),
                )

                if self._is_recording and self.recording_mutation:
                    obs = get_observation_from_manifest(self.data_manifest)
                    dataset_observation, actions = format_observation_for_dataset(obs, self.data_manifest)
                    self.recording_mutation.add_frame(dataset_observation, actions, self.get_task())

    async def teardown(self) -> None:
        if self.recording_mutation:
            self.recording_mutation.teardown()

    def get_task(self) -> str:
        return bytes(self._task_buf.get_obj()).rstrip(b"\x00").decode()

    async def _handle_start_recording(self) -> None:
        if self._start_event.is_set():
            # say(f"Start episode {self.state.episodes_recorded + 1}")
            self._is_recording = True
            self.event_queue.put_nowait({"event": "start_recording", "state": {"is_recording": True}})
            self._start_event.clear()

    async def _handle_save_episode(self) -> None:
        if self._save_event.is_set() and self.recording_mutation:
            # say(f"Saving episode {self.state.episodes_recorded + 1}")
            self._save_event.clear()
            self.recording_mutation.save_episode()
            self._is_recording = False
            self._episodes_recorded += 1
            self.event_queue.put_nowait(
                {
                    "event": "save_episode",
                    "state": {"is_recording": False, "episodes_recorded": self._episodes_recorded},
                }
            )

    async def _handle_discard_episode(self) -> None:
        if self._discard_event.is_set() and self.recording_mutation:
            # say("Discard episode")
            self._discard_event.clear()
            self.recording_mutation.discard_buffer()
            self._is_recording = False
            self.event_queue.put_nowait(
                {
                    "event": "discard_episode",
                    "state": {"is_recording": False},
                }
            )
