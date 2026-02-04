import asyncio
import base64
import time
from multiprocessing import Event, Queue
from multiprocessing.synchronize import Event as EventClass

import cv2
import numpy as np
from frame_source.video_capture_base import VideoCaptureBase
from lerobot.utils.robot_utils import precise_sleep
from loguru import logger
from pydantic import BaseModel

from internal_datasets.dataset_client import DatasetClient
from internal_datasets.utils import get_internal_dataset
from robots.robot_client import RobotClient
from robots.utils import get_robot_client
from schemas import TeleoperationConfig
from schemas.dataset import Episode
from services.robot_calibration_service import RobotCalibrationService
from utils.dataset import build_lerobot_dataset_features
from utils.serial_robot_tools import RobotConnectionManager
from workers.camera_worker import create_frames_source_from_camera

from .base import BaseThreadWorker

OBSERVATION_IMAGES_PREFIX = "observation.images."


class CameraFrameProcessor:
    @staticmethod
    def process(frame: np.ndarray) -> np.ndarray:
        """Post process camera frame."""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


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
    robot_manager: RobotConnectionManager

    action_keys: list[str] = []
    camera_keys: list[str] = []

    dataset: DatasetClient | None = None
    leader: RobotClient
    follower: RobotClient
    cameras: dict[str, VideoCaptureBase]

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
        self.robot_manager = robot_manager
        self.calibration_service = calibration_service
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
        if robot.tele_operator.robot is None:
            raise ValueError("No teleoperator given.")
        self.follower = await get_robot_client(robot.robot, self.robot_manager, self.calibration_service)
        self.leader = await get_robot_client(robot.tele_operator.robot, self.robot_manager, self.calibration_service)

        self.cameras = {
            camera.name: create_frames_source_from_camera(camera) for camera in self.config.environment.cameras
        }
        for camera in self.cameras.values():
            # camera.attach_processor(CameraFrameProcessor()) # TODO Not working. Fix in framesource
            camera.connect()

        await self.follower.connect()
        await self.leader.connect()

    def setup(self) -> None:
        """Set up robots, cameras and dataset."""
        try:
            logger.info("connect to robot, cameras and setup dataset")
            asyncio.run(self.setup_environment())

            self.action_keys = self.follower.features()
            self.camera_keys = [
                f"{OBSERVATION_IMAGES_PREFIX}{camera.name.lower()}" for camera in self.config.environment.cameras
            ]
            print(self.camera_keys)
            self.dataset = get_internal_dataset(self.config.dataset)
            if not self.dataset.exists_on_disk:
                features = asyncio.run(
                    build_lerobot_dataset_features(
                        self.config.environment, self.robot_manager, self.calibration_service
                    )
                )
                self.dataset.create(
                    fps=30,  # TODO: Implement in Environment
                    features=features,
                    robot_type=self.follower.name,
                )

            self.dataset.prepare_for_writing(number_of_threads=4 * len(self.cameras))

            logger.info("dataset loaded, starting cameras")
            for camera in self.cameras.values():
                camera.start_async()

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
        self.queue.put(
            {
                "event": "observations",
                "data": {
                    "actions": {key: observation["action"][index] for index, key in enumerate(self.action_keys)},
                    "cameras": {
                        key.removeprefix(OBSERVATION_IMAGES_PREFIX): self._base_64_encode_observation(
                            cv2.cvtColor(observation[key], cv2.COLOR_RGB2BGR)
                        )
                        for key in self.camera_keys
                        if key in observation
                    },
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
                action = (await self.leader.read_state())["state"]
                state = (await self.follower.read_state())["state"]
                forces = (await self.follower.read_forces())["state"]
                await self.follower.set_joints_state(action)
                if forces is not None:
                    await self.leader.set_forces(forces)

                frame = {
                    "task": self.config.task,
                    "observation.state": np.array(list(state.values()), dtype=np.float32),
                    "action": np.array(list(action.values()), dtype=np.float32),
                }

                actions = {f"{key}.pos": value for key, value in action.items()}
                observations = {f"{key}.pos": value for key, value in state.items()}
                print(observations)
                for camera_name, camera in self.cameras.items():
                    _success, camera_frame = camera.get_latest_frame()  # HWC
                    if camera_frame is None:
                        raise Exception("Camera frame is None")
                    processed_frame = CameraFrameProcessor.process(camera_frame)
                    frame[f"observation.images.{camera_name}"] = processed_frame
                    observations[camera_name] = processed_frame

                timestamp = time.perf_counter() - self.start_episode_t
                self._report_observation(frame, timestamp)
                if self.state.is_recording and self.dataset is not None:
                    self.dataset.add_frame(observations, actions, self.config.task)

                dt_s = time.perf_counter() - start_loop_t
                wait_time = 1 / 30 - dt_s
                precise_sleep(wait_time)
        except Exception as e:
            self.error = True
            self._report_error(e)

    def _on_start(self) -> None:
        logger.info("start")
        self.events["start"].clear()
        self.state.is_recording = True
        self._report_state()

    def _on_save(self) -> None:
        logger.info("save")
        self.events["save"].clear()
        precise_sleep(0.3)  # TODO check if neccesary
        if self.dataset is not None:
            new_episode = self.dataset.save_episode(self.config.task)
            self._report_episode(new_episode)
        self.state.is_recording = False
        self._report_state()

    def _on_reset(self) -> None:
        logger.info("reset")
        self.events["reset"].clear()
        precise_sleep(0.3)  # TODO check if neccesary
        if self.dataset is not None:
            self.dataset.discard_buffer()
        self.state.is_recording = False
        self._report_state()

    def teardown(self) -> None:
        """Disconnect robots and close queue."""
        logger.info("Teardown")
        try:
            self.queue.cancel_join_thread()
        except Exception as e:
            logger.warning(f"Failed cancelling queue join thread: {e}")

        if self.dataset:
            self.dataset.teardown()

        if self.follower is not None:
            try:
                asyncio.run(self.follower.disconnect())
            except Exception:
                logger.info(f"Failed disconnecting follower: {self.follower}")

        if self.leader is not None:
            try:
                asyncio.run(self.leader.disconnect())
            except Exception:
                logger.info(f"Failed disconnecting leader: {self.leader}")

        for camera in self.cameras.values():
            camera.disconnect()

        # Wait for .5 seconds before closing queue to allow messages thru
        asyncio.run(asyncio.sleep(0.5))

        self.queue.close()

    def _base_64_encode_observation(self, observation: np.ndarray | None) -> str:
        if observation is None:
            return ""
        _, imagebytes = cv2.imencode(".jpg", observation)
        return base64.b64encode(imagebytes).decode()
