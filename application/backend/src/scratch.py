#!/usr/bin/env python3

import asyncio
from uuid import UUID

from getiaction.data.lerobot.converters import FormatConverter
from getiaction.data.observation import Observation
from getiaction.policies.act.model import ACT
from getiaction.policies.act.policy import ACT as ACTPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots import make_robot_from_config
from pydantic import BaseModel
import torch
from utils.framesource_bridge import FrameSourceCameraBridge
from utils.camera import build_camera_config
from utils.robot import make_lerobot_robot_config_from_robot
from schemas import CameraConfig, RobotConfig, Model
from services import ModelService, DatasetService

from pathlib import Path


class InferenceConfig(BaseModel):
    task: str
    device: str
    model_id: str
    cameras: list[CameraConfig]
    robot: RobotConfig


config = InferenceConfig(
    task="grab block",
    model_id="72e255b1-bc0f-486f-b495-bd51dbc6a844",
    device="cuda",
    cameras=[
        {
            "id": "fe99defd-d699-4bbf-99ef-3c806ef11d19",
            "port_or_device_id": "/dev/video4:4",
            "name": "top",
            "driver": "webcam",
            "width": 640,
            "height": 480,
            "fps": 30,
            "use_depth": False,
        },
        {
            "id": "169f3090-ec64-45da-8434-8e419f0c7f1d",
            "port_or_device_id": "/dev/video6:6",
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
    dataset = await DatasetService().get_dataset_by_id(UUID("cf8062f8-7354-4643-8899-9be8d57ab2ce"))
    if dataset is None:
        raise ValueError("dataset not found")

    #lerobot_dataset = LeRobotDataset(dataset.name, dataset.path)
    #episode_index = 0
    #from_idx = lerobot_dataset.episode_data_index["from"][episode_index].item()
    #to_idx = lerobot_dataset.episode_data_index["to"][episode_index].item()
    #actions = lerobot_dataset.hf_dataset["action"][from_idx:to_idx]
    #print(actions)
    #FormatConverter.to_observation(lerobot_dataset.episodes[0])

    #return
    if model is None:
        raise ValueError("Model not found")
    # path = "/home/ronald/intel/geti-action/application/backend/experiments/lightning_logs/version_8/checkpoints/epoch=1-step=100.ckpt"
    act = ACT.load_checkpoint(Path(model.path).expanduser())
    # act = ACT.load_checkpoint(path)
    policy = ACTPolicy(act)
    policy.setup("predict")

    cameras = {camera.name: FrameSourceCameraBridge(camera) for camera in config.cameras}
    follower_config = make_lerobot_robot_config_from_robot(config.robot, {})

    robot = make_robot_from_config(follower_config)
    robot.connect()
    for camera in cameras.values():
        camera.connect()

    #obs = FormatConverter.to_observation(robot.get_observation())
    actions = torch.tensor(list(robot.get_observation().values()))

    images: dict = {}
    for name, camera in cameras.items():
        frame = camera.read()
        key = f"observation.images.{name}"

        images[key] = torch.from_numpy(frame)
        images[key] = images[key].type(torch.float32) / 255
        images[key] = images[key].permute(2, 0, 1).contiguous()

    task = "grab block and place on square"

    observation = Observation(
        action=actions,
        images=images,
        task=task,
    )

    policy.select_action(observation)


    for camera in cameras.values():
        camera.disconnect()

    robot.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
