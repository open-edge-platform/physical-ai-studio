import time
import asyncio
from uuid import UUID
from pathlib import Path

from getiaction.data import Observation
from getiaction.policies import ACT, ACTModel
from lerobot.utils.robot_utils import busy_wait
import torch
from lerobot.robots import make_robot_from_config
from pydantic import BaseModel

from utils.camera import build_camera_config, initialize_camera
from schemas import CameraConfig, RobotConfig
from services import DatasetService, ModelService
from utils.framesource_bridge import FrameSourceCameraBridge
from utils.robot import make_lerobot_robot_config_from_robot


class InferenceConfig(BaseModel):
    task: str
    device: str
    model_id: str
    fps: int
    cameras: list[CameraConfig]
    robot: RobotConfig


config = InferenceConfig(
    task="grab block",
    model_id="d97557f2-7dcc-41d8-80e5-acfd5a6d6086",
    device="cuda",
    fps=30,
    cameras=[
        {
            "id": "fe99defd-d699-4bbf-99ef-3c806ef11d19",
            "port_or_device_id": "/dev/video4",
            "name": "top",
            "driver": "webcam",
            "width": 640,
            "height": 480,
            "fps": 30,
            "use_depth": False,
        },
        {
            "id": "169f3090-ec64-45da-8434-8e419f0c7f1d",
            "port_or_device_id": "/dev/video6",
            "name": "grabber",
            "driver": "webcam",
            "width": 640,
            "height": 480,
            "fps": 30,
            "use_depth": False,
        },
    ],
    robot={
        "id": "khaos",
        "robot_type": "so101_follower",
        "serial_id": "5AA9017083",
        "port": "/dev/ttyACM2",
        "type": "follower",
    },
)


async def main():
    model = await ModelService.get_model_by_id(UUID(config.model_id))
    #dataset = await DatasetService().get_dataset_by_id(UUID("222d929a-39e7-4376-95fd-3d8809b7d9fb"))
    #if dataset is None:
    #    raise ValueError("dataset not found")

    # lerobot_dataset = LeRobotDataset(dataset.name, dataset.path)
    # episode_index = 0
    # from_idx = lerobot_dataset.episode_data_index["from"][episode_index].item()
    # to_idx = lerobot_dataset.episode_data_index["to"][episode_index].item()
    # actions = lerobot_dataset.hf_dataset["action"][from_idx:to_idx]
    # print(actions)
    # FormatConverter.to_observation(lerobot_dataset.episodes[0])

    # return
    if model is None:
        raise ValueError("Model not found")
    # path = "/home/ronald/intel/geti-action/application/backend/experiments/lightning_logs/version_8/checkpoints/epoch=1-step=100.ckpt"
    act_model = ACTModel.load_from_checkpoint(Path(model.path).expanduser())
    ## act = ACT.load_checkpoint(path)
    policy = ACT(act_model)
    policy.setup("predict")

    cameras = {camera.name: initialize_camera(build_camera_config(camera)) for camera in config.cameras}
    follower_config = make_lerobot_robot_config_from_robot(config.robot, {})

    robot = make_robot_from_config(follower_config)
    robot.connect()
    for camera in cameras.values():
        camera.connect()

    # obs = FormatConverter.to_observation(robot.get_observation())


    while True:
        start_loop_t = time.perf_counter()

        robot_observation = robot.get_observation()
        state = torch.tensor(list(robot_observation.values())).unsqueeze(0)
        images: dict = {}
        for name, camera in cameras.items():
            frame = camera.read()
            key = name #f"observation.images.{name}"
            print(frame.shape)

            images[key] = torch.from_numpy(frame)
            images[key] = images[key].float()
            images[key] = images[key].permute(2, 0, 1).contiguous()
            images[key] = images[key].unsqueeze(0)

        observation = Observation(
            state=state,
            images=images,
        )

        actions = policy.select_action(observation)
        formatted_actions = dict(zip(robot_observation.keys(), actions[0].tolist()))
        robot.send_action(formatted_actions)

        dt_s = time.perf_counter() - start_loop_t
        wait_time = 1 / config.fps - dt_s

        busy_wait(wait_time)


    for camera in cameras.values():
        camera.disconnect()

    robot.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
