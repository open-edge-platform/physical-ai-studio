import base64
import shutil
import time
import copy
from multiprocessing import Event, Queue
from multiprocessing.synchronize import Event as EventClass

import cv2
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features, combine_feature_dicts
from lerobot.robots.utils import make_robot_from_config
from lerobot.teleoperators.utils import make_teleoperator_from_config
from lerobot.utils.robot_utils import busy_wait
from lerobot.processor import make_default_processors
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from loguru import logger

from schemas import TeleoperationConfig
from schemas.dataset import Episode, EpisodeVideo
from utils.camera import build_camera_config
from utils.dataset import check_repository_exists, get_episode_actions
from utils.framesource_bridge import FrameSourceCameraBridge
from utils.robot import make_lerobot_robot_config_from_robot, make_lerobot_teleoperator_config_from_robot

from .base import BaseThreadWorker


class TeleoperateWorker(BaseThreadWorker):
    ROLE: str = "TeleoperateWorker"

    events: dict[str, EventClass]
    queue: Queue
    is_recording = False

    action_keys: list[str] = []
    camera_keys: list[str] = []

    def __init__(self, stop_event: EventClass, queue: Queue, config: TeleoperationConfig):
        super().__init__(stop_event=stop_event)
        self.config = config
        self.queue = queue
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

    def setup(self) -> None:
        """Set up robots, cameras and dataset."""
        logger.info("connect to robot, cameras and setup dataset")
        cameras = {camera.name: build_camera_config(camera) for camera in self.config.cameras}
        follower_config = make_lerobot_robot_config_from_robot(self.config.follower, cameras)
        leader_config = make_lerobot_teleoperator_config_from_robot(self.config.leader)

        self.robot = make_robot_from_config(follower_config)
        self.teleoperator = make_teleoperator_from_config(leader_config)

        self.teleop_action_processor, self.robot_action_processor, self.robot_observation_processor = make_default_processors()
        self.dataset_features = combine_feature_dicts(
            aggregate_pipeline_dataset_features(
                pipeline=self.teleop_action_processor,
                initial_features=create_initial_features(
                    action=self.robot.action_features
                ),
                use_videos=True,
            ),
            aggregate_pipeline_dataset_features(
                pipeline=self.robot_observation_processor,
                initial_features=create_initial_features(observation=self.robot.observation_features),
                use_videos=True,
            ),
        )


        # After setting up the robot, instantiate the FrameSource using a bridge
        # This can be done directly once switched over to LeRobotDataset V3.
        # We do need to first instantiate using the lerobot dict because a follower requires cameras.
        self.robot.cameras = {camera.name: FrameSourceCameraBridge(camera) for camera in self.config.cameras}

        if check_repository_exists(self.config.dataset.path):
            self.dataset = LeRobotDataset(
                repo_id=self.config.dataset.name, root=self.config.dataset.path, batch_encoding_size=1
            )
        else:
            self.dataset = LeRobotDataset.create(
                repo_id=self.config.dataset.name,
                root=self.config.dataset.path,
                fps=self.config.fps,
                features=self.dataset_features,
                robot_type=self.robot.name,
                use_videos=True,
                image_writer_threads=4,
            )

        self.dataset.start_image_writer(
            num_processes=0,
            num_threads=4 * len(self.robot.cameras),
        )

        logger.info("before robot connection")
        self.robot.connect()
        logger.info("between robot and teleoperator connection")
        self.teleoperator.connect()
        logger.info("after robot connection")

        self.action_keys = [f"{key}.pos" for key in self.robot.bus.sync_read("Present_Position")]
        self.camera_keys = [camera.name for camera in self.config.cameras]

        logger.info("teleoperation all setup, reporting state")
        self._report_state()

    def _report_state(self):
        state = {"event": "state", "data": {"initialized": True, "is_recording": self.is_recording}}
        logger.info(f"teleoperation state: {state}")
        self.queue.put(state)

    def _report_observation(self, observation: dict, timestamp: float):
        """Report observation to queue."""
        self.queue.put(
            {
                "event": "observations",
                "data": {
                    "actions": {key: observation.get(key, 0) for key in self.action_keys},
                    "cameras": {
                        key: self._base_64_encode_observation(cv2.cvtColor(observation[key], cv2.COLOR_RGB2BGR))
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
        self.events["reset"].clear()
        self.events["save"].clear()
        self.events["stop"].clear()
        self.events["start"].clear()

        start_episode_t = time.perf_counter()
        self.is_recording = False
        while not self.should_stop() and not self.events["stop"].is_set():
            start_loop_t = time.perf_counter()
            if self.events["save"].is_set():
                logger.info("save")
                self.events["save"].clear()
                busy_wait(0.3)  # TODO check if neccesary
                new_episode = self._build_episode_from_buffer(self.dataset.meta.latest_episode)
                if new_episode is not None:
                    self._report_episode(new_episode)
                self.dataset.save_episode()
                self.is_recording = False
                self._report_state()

            if self.events["reset"].is_set():
                logger.info("reset")
                self.events["reset"].clear()
                busy_wait(0.3)  # TODO check if neccesary
                self.dataset.clear_episode_buffer()
                self.is_recording = False
                self._report_state()

            if self.events["start"].is_set():
                logger.info("start")
                self.events["start"].clear()
                self.is_recording = True
                start_episode_t = time.perf_counter()
                self._report_state()

            action = self.teleoperator.get_action()
            self.robot.send_action(action)
            observation = self.robot.get_observation()
            actions_processed = self.teleop_action_processor((action, observation))
            obs_processed = self.robot_observation_processor(observation)

            timestamp = time.perf_counter() - start_episode_t
            self._report_observation(observation, timestamp)
            if self.is_recording:
                observation_frame = build_dataset_frame(self.dataset.features, obs_processed, prefix="observation")
                action_frame = build_dataset_frame(self.dataset.features, actions_processed, prefix="action")
                frame = {**observation_frame, **action_frame, "task": self.config.task}
                self.dataset.add_frame(frame)

            dt_s = time.perf_counter() - start_loop_t
            wait_time = 1 / self.config.fps - dt_s

            busy_wait(wait_time)

    def teardown(self) -> None:
        """Disconnect robots and close queue."""

        try:
            self.queue.cancel_join_thread()
        except Exception as e:
            logger.warning(f"Failed cancelling queue join thread: {e}")

        # must happen
        self.dataset.finalize()

        # Ensure the dataset is removed if there are episodes
        # This is because lerobot dataset needs episodes otherwise it will be in an invalid state
        if self.dataset.num_episodes == 0:
            shutil.rmtree(self.dataset.root)

        self.robot.disconnect()
        self.teleoperator.disconnect()
        self.queue.close()

    def _base_64_encode_observation(self, observation: np.ndarray | None) -> str:
        if observation is None:
            return ""
        _, imagebytes = cv2.imencode(".jpg", observation)
        return base64.b64encode(imagebytes).decode()

    def _build_episode_from_buffer(self, episode: dict | None) -> Episode | None:
        """Build Episode object from buffer and episode dict."""
        data = self._build_episode_data_from_buffer()
        if data is None:
            return None

        end = data["timestamp"][-1]
        video_timestamps = {video_key: EpisodeVideo(start=0, end=end) for video_key in self.dataset.meta.video_keys}
        if episode is not None:
            for video_key in self.dataset.meta.video_keys:
                offset = episode[f"videos/{video_key}/to_timestamp"][-1]
                video_timestamps[video_key].start += offset
                video_timestamps[video_key].end += offset

        return Episode(
            episode_index=data["episode_index"].tolist()[0],
            length=len(data["frame_index"]),
            fps=self.dataset.fps,
            tasks=[self.config.task],
            actions=data["action"].tolist(),
            videos=video_timestamps,
            modification_timestamp=int(time.time()),
        )


    def _build_episode_data_from_buffer(self) -> dict | None:
        """Build episode data from the buffer.

        LeRobotDataset V3 doesnt update episode data on save.
        In order to get the episode data we duplicate the actions that happen inside.
        """

        episode_buffer = copy.deepcopy(self.dataset.episode_buffer)
        if episode_buffer is not None:
            episode_length = episode_buffer.pop("size")
            tasks = episode_buffer.pop("task")
            episode_tasks = list(set(tasks))
            episode_index = episode_buffer["episode_index"]

            episode_buffer["index"] = np.arange(self.dataset.meta.total_frames, self.dataset.meta.total_frames + episode_length)
            episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

            # Update tasks and task indices with new tasks if any
            self.dataset.meta.save_episode_tasks(episode_tasks)

            # Given tasks in natural language, find their corresponding task indices
            episode_buffer["task_index"] = np.array([self.dataset.meta.get_task_index(task) for task in tasks])

            for key, ft in self.dataset.features.items():
                # index, episode_index, task_index are already processed above, and image and video
                # are processed separately by storing image path and frame info as meta data
                if key in ["index", "episode_index", "task_index"] or ft["dtype"] in ["image", "video"]:
                    continue
                episode_buffer[key] = np.stack(episode_buffer[key])

            return episode_buffer
