import asyncio
import base64
import time
import traceback
from multiprocessing import Event, Queue
from multiprocessing.synchronize import Event as EventClass
from pathlib import Path

import cv2
import numpy as np
from frame_source.video_capture_base import VideoCaptureBase
from lerobot.utils.robot_utils import precise_sleep
from loguru import logger
from pydantic import BaseModel

from internal_datasets.dataset_client import DatasetClient
from internal_datasets.lerobot.lerobot_dataset import InternalLeRobotDataset
from internal_datasets.mutations.recording_mutation import RecordingMutation
from robots.robot_client import RobotClient
from robots.robot_client_factory import RobotClientFactory
from schemas import TeleoperationConfig
from schemas.dataset import Episode
from services.robot_calibration_service import RobotCalibrationService
from utils.async_camera_capture import AsyncCameraCapture
from utils.dataset import build_lerobot_dataset_features
from utils.serial_robot_tools import RobotConnectionManager
from workers.camera_worker import create_frames_source_from_camera

from .base import BaseThreadWorker


class CameraFrameProcessor:
    @staticmethod
    def process(frame: np.ndarray) -> np.ndarray:
        """Post process camera frame."""
        if isinstance(frame, np.ndarray):
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None


class TeleoperateState(BaseModel):
    initialized: bool = False
    is_recording: bool = False
    error: bool = False


class TeleoperateWorker(BaseThreadWorker):
    ROLE: str = "TeleoperateWorker"

    events: dict[str, EventClass]
    queue: Queue
    state: TeleoperateState

    config: TeleoperationConfig
    fps: float = 30
    robot_client_factory: RobotClientFactory

    action_keys: list[str] = []
    camera_keys: list[str] = []

    dataset: DatasetClient | None = None
    mutation: RecordingMutation | None = None
    leader: RobotClient | None = None
    follower: RobotClient | None = None
    cameras: dict[str, VideoCaptureBase] = {}
    recording_mutation = None

    def __init__(
        self,
        stop_event: EventClass,
        queue: Queue,
        config: TeleoperationConfig,
        calibration_service: RobotCalibrationService,
        robot_manager: RobotConnectionManager,
    ):
        super().__init__(stop_event=stop_event)

        self.state = TeleoperateState()
        self.config = config
        self.queue = queue
        self.robot_client_factory = RobotClientFactory(robot_manager, calibration_service)
        self.frame_captures = {}

        self.events = {
            "stop": Event(),
            "reset": Event(),
            "save": Event(),
            "start": Event(),
        }

    def stop(self) -> None:
        """Stop teleoperation and stop loop."""
        self.events["stop"].set()

    def start_recording(self) -> None:
        """Start recording observations to dataset buffer."""
        self.events["start"].set()

    def save(self) -> None:
        """Save current dataset recording buffer."""
        self.events["save"].set()

    def reset(self) -> None:
        """Reset the dataset recording buffer."""
        self.events["reset"].set()

    @staticmethod
    def _camera_frame_postprocessing(frame: np.ndarray) -> np.ndarray:
        """Post process camera frame."""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    async def setup_environment(self) -> None:
        """Setup environment."""
        robot = self.config.environment.robots[0]  # Assume 1 arm for now.
        if robot.tele_operator.type == "none" or robot.tele_operator.robot is None:
            raise ValueError("No teleoperator given.")
        self.follower = await self.robot_client_factory.build(robot.robot)
        self.leader = await self.robot_client_factory.build(robot.tele_operator.robot)

        self.frame_captures = {}
        for cam_cfg in self.config.environment.cameras:
            cam_id = str(cam_cfg.id)
            cam = create_frames_source_from_camera(cam_cfg)  # gives you the object with connect/read

            cap = AsyncCameraCapture(
                camera=cam,
                fps=cam_cfg.payload.fps,
                process_fn=CameraFrameProcessor.process,  # BGR->RGB, etc.
                use_cached_on_failure=True,
            )
            await cap.start()
            self.frame_captures[cam_id] = cap

        await self.follower.connect()
        await self.leader.connect()

        await asyncio.sleep(1)  #  warmup cameras

    def setup(self) -> None:
        """Set up robots, cameras and dataset."""
        try:
            logger.info("connect to robot, cameras and setup dataset")
            if self.loop is None:
                raise RuntimeError("The event loop must be set.")
            self.loop.run_until_complete(self.setup_environment())
            self.dataset = InternalLeRobotDataset(Path(self.config.dataset.path))

            if self.leader is None or self.follower is None or self.dataset is None:
                raise RuntimeError("Environment setup failed.")

            self.action_keys = self.follower.features()
            self.camera_keys = [str(camera.id) for camera in self.config.environment.cameras]
            features = self.loop.run_until_complete(
                build_lerobot_dataset_features(self.config.environment, self.robot_client_factory)
            )

            self.recording_mutation = self.dataset.start_recording_mutation(
                fps=self.fps,  # TODO: Implement in Environment
                features=features,
                robot_type=self.follower.name,
            )

            logger.info("dataset loaded")

            self.state.initialized = True
            logger.info("teleoperation all setup, reporting state")
        except Exception as e:
            self.error = True
            self._report_error(e)
        self._report_state()

    def _report_state(self):
        state = {"event": "state", "data": self.state.model_dump()}
        logger.info(f"teleoperation state: {state}")
        self.queue.put(state)

    def _report_error(self, error: BaseException):
        data = {
            "event": "error",
            "data": str(error),
        }
        logger.exception(f"error: {data}")
        self.queue.put(data)

    def _report_observation(self, observation: dict, timestamp: float):
        """Report observation to queue."""

        camera_images = {
            key: self._base_64_encode_observation(cv2.cvtColor(observation[key], cv2.COLOR_RGB2BGR))
            for key in self.camera_keys
            if key in observation
        }
        self.queue.put(  # Mimicing the dataset features format.
            {
                "event": "observations",
                "data": {
                    "actions": {key: observation[key] for key in self.action_keys},
                    "cameras": camera_images,
                    "timestamp": timestamp,
                },
            }
        )

    def _report_episode(self, episode: Episode):
        self.queue.put(
            {
                "event": "episode",
                "data": episode.model_dump(),
            }
        )

    async def run_loop(self) -> None:
        """Teleoperation loop."""
        logger.info("run loop")
        try:
            self.events["reset"].clear()
            self.events["save"].clear()
            self.events["stop"].clear()
            self.events["start"].clear()

            self.start_episode_t = time.perf_counter()
            self.state.is_recording = False

            if self.leader is None or self.follower is None or self.dataset is None:
                raise RuntimeError("Environment setup failed.")

            while not self.should_stop() and not self.events["stop"].is_set():
                start_loop_t = time.perf_counter()
                if self.events["start"].is_set():
                    self._on_start()

                if self.events["save"].is_set():
                    self._on_save()

                if self.events["reset"].is_set():
                    self._on_reset()

                # Trossen notes:
                # Add force feedback
                actions = (await self.leader.read_state())["state"]
                observations = (await self.follower.read_state())["state"]
                await self.follower.set_joints_state(actions, 1 / self.fps)
                forces = (await self.follower.read_forces())["state"]
                if forces is not None:
                    await self.leader.set_forces(forces)

                for cam_id, cap in self.frame_captures.items():
                    frame, t_perf, ok, err, seq = await cap.get_latest()
                    if ok and frame is not None:
                        observations[cam_id] = frame

                timestamp = time.perf_counter() - self.start_episode_t
                self._report_observation(observations, timestamp)
                if self.state.is_recording and self.recording_mutation is not None:
                    self.recording_mutation.add_frame(
                        self._to_lerobot_observations(observations), actions, self.config.task
                    )

                dt_s = time.perf_counter() - start_loop_t
                wait_time = 1 / (self.fps * 2) - dt_s
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                else:
                    await asyncio.sleep(0)
        except Exception as e:
            self.error = True
            traceback.print_exception(e)
            self._report_error(e)

    def _to_lerobot_observations(self, observations: dict) -> dict:
        """Remap camera observations from camera ID keys to lowercase camera name keys."""
        lerobot_observations = dict(observations)
        for camera in self.config.environment.cameras:
            lerobot_observations[camera.name.lower()] = lerobot_observations.pop(str(camera.id))
        return lerobot_observations

    def _on_start(self) -> None:
        logger.info("start")
        self.events["start"].clear()
        self.state.is_recording = True
        self._report_state()

    def _on_save(self) -> None:
        logger.info("save")
        self.events["save"].clear()
        precise_sleep(0.3)  # TODO check if neccesary
        if self.recording_mutation is not None:
            new_episode = self.recording_mutation.save_episode(self.config.task)
            self._report_episode(new_episode)
        self.state.is_recording = False
        self._report_state()

    def _on_reset(self) -> None:
        logger.info("reset")
        self.events["reset"].clear()
        precise_sleep(0.3)  # TODO check if neccesary
        if self.recording_mutation is not None:
            self.recording_mutation.discard_buffer()
        self.state.is_recording = False
        self._report_state()

    async def teardown(self) -> None:
        """Disconnect robots and close queue."""
        logger.info("Teardown")
        try:
            self.queue.cancel_join_thread()
        except Exception as e:
            logger.warning(f"Failed cancelling queue join thread: {e}")

        if self.recording_mutation:
            self.recording_mutation.teardown()

        if self.follower is not None:
            try:
                await self.follower.disconnect()
            except Exception:
                logger.info(f"Failed disconnecting follower: {self.follower}")

        if self.leader is not None:
            try:
                await self.leader.disconnect()
            except Exception:
                logger.info(f"Failed disconnecting leader: {self.leader}")

        for cap in self.frame_captures.values():
            try:
                await cap.stop()
            except Exception:
                logger.info("Failed stopping a camera thread. Ignoring")

        # Wait for .5 seconds before closing queue to allow messages through
        await asyncio.sleep(0.5)

        self.queue.close()

        import threading

        logger.error("THREADS AFTER TELEOP TEARDOWN:\n" + "\n".join(t.name for t in threading.enumerate()))

    def _base_64_encode_observation(self, observation: np.ndarray | None) -> str:
        if observation is None:
            return ""
        _, imagebytes = cv2.imencode(".jpg", observation)
        return base64.b64encode(imagebytes).decode()
