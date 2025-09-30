from .base import BaseProcessWorker, BaseThreadWorker
import time

from multiprocessing import Event, Queue
from multiprocessing.synchronize import Event as EventClass
from schemas import TeleoperationConfig
from lerobot.utils.robot_utils import busy_wait
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from utils.camera import build_camera_config
from utils.robot import make_lerobot_robot_config_from_robot, make_lerobot_teleoperator_config_from_robot
from lerobot.teleoperators.utils import make_teleoperator_from_config
from lerobot.robots.utils import make_robot_from_config
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features

class TeleoperateWorker(BaseProcessWorker):
    ROLE: str = "TeleoperateWorker"

    events: dict[str, EventClass]
    queue: Queue
    is_recording = False

    def __init__(self, stop_event: EventClass, queue: Queue, config: TeleoperationConfig):
        super().__init__(stop_event=stop_event, queues_to_cancel=[queue])
        self.config = config
        self.queue = queue
        self.events = {
            "stop": Event(),
            "reset": Event(),
            "save": Event(),
            "start": Event(),
        }

    def stop(self):
        self.events["stop"].set()

    def start_recording(self):
        self.events["start"].set()

    def save(self):
        self.events["save"].set()

    def reset(self):
        self.events["reset"].set()

    def setup(self):
        print("connect to robot, cameras and setup dataset")
        cameras = {camera.name: build_camera_config(camera) for camera in self.config.cameras}
        follower_config = make_lerobot_robot_config_from_robot(self.config.follower, cameras)
        leader_config = make_lerobot_teleoperator_config_from_robot(self.config.leader)

        self.robot = make_robot_from_config(follower_config)
        self.teleoperator = make_teleoperator_from_config(leader_config)
        self.dataset = LeRobotDataset(
            repo_id=self.config.dataset.name,
            root=self.config.dataset.path,
            batch_encoding_size=1
        )

        self.dataset.start_image_writer(
            num_processes=0,
            num_threads=4 * len(self.robot.cameras),
        )
        self.report_state()

    def report_state(self):
        self.queue.put({
            "event": "state",
            "data": {
                "initialized": True,
                "is_recording": self.is_recording
            }
        })


    def run_loop(self):
        print("run loop")
        self.events["reset"].clear()
        self.events["save"].clear()
        self.events["stop"].clear()
        self.events["start"].clear()

        self.robot.connect()
        self.teleoperator.connect()

        start_loop_t = time.perf_counter()
        self.is_recording = False
        while not self.should_stop() and not self.events["stop"].is_set():
            if self.events["save"].is_set():
                print("save")
                self.events["save"].clear()
                busy_wait(0.3) #TODO check if neccesary
                self.dataset.save_episode()
                self.is_recording = False
                self.report_state()

            if self.events["reset"].is_set():
                print("reset")
                self.events["reset"].clear()
                busy_wait(0.3) #TODO check if neccesary
                self.dataset.clear_episode_buffer()
                self.is_recording = False
                self.report_state()

            if self.events["start"].is_set():
                print("start")
                self.events["start"].clear()
                self.is_recording = True
                self.report_state()

            action = self.teleoperator.get_action()
            sent_action = self.robot.send_action(action)
            if self.is_recording:
                observation = self.robot.get_observation()
                observation_frame = build_dataset_frame(self.dataset.features, observation, prefix="observation")
                action_frame = build_dataset_frame(self.dataset.features, sent_action, prefix="action")
                frame = {**observation_frame, **action_frame}
                self.dataset.add_frame(frame, task=self.config.task)

            dt_s = time.perf_counter() - start_loop_t
            wait_time = 1 / self.config.fps - dt_s

            busy_wait(wait_time)

    def teardown(self):
        self.robot.disconnect()
        self.teleoperator.disconnect()
        self.queue.close()
