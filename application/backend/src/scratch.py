from getiaction.data.lerobot.datamodule import LeRobotDataModule
from torch.utils.data import DataLoader
import yaml
from pathlib import Path
import time
import asyncio
from uuid import UUID
from pathlib import Path

from getiaction.train import Trainer
from getiaction.data import Observation
from getiaction.policies import ACT, ACTModel, ACTConfig
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
        "port": "/dev/ttyACM0",
        "type": "follower",
    },
)


async def main():
    model_path = "/home/ronald/intel/geti-action/application/backend/place-block-model.ckpt"
    #model = await ModelService.get_model_by_id(UUID(config.model_id))
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
    #if model is None:
    #    raise ValueError("Model not found")
    # path = "/home/ronald/intel/geti-action/application/backend/experiments/lightning_logs/version_8/checkpoints/epoch=1-step=100.ckpt"
    act_model = ACTModel.load_from_checkpoint(model_path)
    act_model.eval()
    ## act = ACT.load_checkpoint(path)
    policy = ACT(act_model)
    policy.eval()

    cameras = {camera.name: initialize_camera(build_camera_config(camera)) for camera in config.cameras}
    follower_config = make_lerobot_robot_config_from_robot(config.robot, {})

    robot = make_robot_from_config(follower_config)
    robot.connect()
    for camera in cameras.values():
        camera.connect()


    print(robot.bus.sync_read("Present_Position"))
    root_position_action = {'shoulder_pan.pos': -2.271006813020435, 'shoulder_lift.pos': -98.08027923211169, 'elbow_flex.pos': 99.37527889335118, 'wrist_flex.pos': 67.34527687296418, 'wrist_roll.pos': -13.406593406593402, 'gripper.pos': 27.128953771289538}
    robot.send_action(root_position_action)



    while True:
        start_loop_t = time.perf_counter()

        robot_observation = robot.get_observation()
        action_keys =  [f"{key}.pos" for key in robot.bus.sync_read("Present_Position")]
        state = torch.tensor(list(robot_observation.values())).unsqueeze(0)
        images: dict = {}
        for name, camera in cameras.items():
            frame = camera.read()
            key = name #f"observation.images.{name}"
            print(frame.shape)

            images[key] = torch.from_numpy(frame)
            images[key] = images[key].float() / 255
            images[key] = images[key].permute(2, 0, 1).contiguous()
            images[key] = images[key].unsqueeze(0)

        observation = Observation(
            state=state,
            images=images,
        )

        #print(observation.to_dict())
        #print(act_model(observation.to_dict()))
        #break
        actions = policy.select_action(observation)
        print(actions.shape)
        formatted_actions = dict(zip(robot_observation.keys(), actions[0].tolist()))
        print(formatted_actions)
        robot.send_action(formatted_actions)

        dt_s = time.perf_counter() - start_loop_t
        wait_time = 1 / config.fps - dt_s

        break
        busy_wait(wait_time)


    for camera in cameras.values():
        camera.disconnect()

    robot.disconnect()


def export_existing_model():
    donor_model_path = "/home/ronald/intel/geti-action/application/backend/config_donor_model.ckpt"
    donor_data = torch.load(donor_model_path, map_location="cpu", weights_only=True)
    print(donor_data["model_config"])
    model_path = "/home/ronald/intel/geti-action/application/backend/lightning_logs/version_3/checkpoints/epoch=748-step=47187.ckpt"

    data = torch.load(model_path, map_location="cpu", weights_only=True)
    return
    data["model_config"] = donor_data["model_config"]
    act_model = ACTModel.load_from_checkpoint(data)
    ACT(model=act_model).to_torch("./place-block-model.ckpt")

def build_donor_model():
    l_dm = LeRobotDataModule(
        repo_id="rhecker/place-block",
        train_batch_size=8,
    )
    lib_model = ACTModel(
        input_features=l_dm.train_dataset.observation_features,
        output_features=l_dm.train_dataset.action_features,
    )

    trainer = Trainer(
        max_steps=1,
    )

    policy = ACT(model=lib_model)

    trainer.fit(model=policy, datamodule=l_dm)
    policy.to_torch("./config_donor_model.ckpt")

def second():
    l_dm = LeRobotDataModule(
        repo_id="rhecker/place-block",
        train_batch_size=1,
    )
    lib_model = ACTModel(
        input_features=l_dm.train_dataset.observation_features,
        output_features=l_dm.train_dataset.action_features,
    )
    policy = ACT(model=lib_model)

    l_dm.val_dataset = l_dm.train_dataset
    test_loader = l_dm.val_dataloader()
    #test_loader = l_dm.train_dataloader()
    i = 0
    for data in test_loader:
        print(data)
        i += 1
        if data.episode_index.tolist() > 0:
            break



    #trainer = Trainer(
    #    callbacks=[
    #        #checkpoint_callback,
    #        TrainingTrackingCallback(
    #            shutdown_event=self._stop_event,
    #            interrupt_event=self.interrupt_event,
    #            dispatcher=dispatcher,
    #        ),
    #    ],
    #    max_steps=10000,
    #)
    #


from dataclasses import fields
from typing import TYPE_CHECKING, Any
import numpy as np
def _collate_observations(batch: list[Observation]) -> Observation:
    """Collate a batch of Observations into a single batched Observation.

    Args:
        batch (list[Observation]): A list containing Observations.

    Returns:
        Observation: A single Observation with batched tensors.
    """
    if not batch:
        return Observation()

    collated_data: dict[str, Any] = {}

    # Iterate through all fields defined in the Observation dataclass
    for field in fields(Observation):
        key = field.name
        values = [getattr(elem, key) for elem in batch]

        # Filter out None values to determine the data type
        non_none_values = [v for v in values if v is not None]

        if not non_none_values:
            collated_data[key] = None
            continue

        first_non_none = non_none_values[0]

        # Handle tensors and NumPy arrays
        if isinstance(first_non_none, (torch.Tensor, np.ndarray)):
            # Convert NumPy arrays to PyTorch tensors before stacking
            tensors_to_stack = [torch.from_numpy(v) if isinstance(v, np.ndarray) else v for v in non_none_values]
            collated_data[key] = torch.stack(tensors_to_stack, dim=0)

        # Handle nested dictionaries, such as the `images` field
        elif isinstance(first_non_none, dict):
            collated_inner_dict = {}
            for inner_key in first_non_none:
                inner_values = [d.get(inner_key) for d in values if d is not None]
                if inner_values:
                    first_inner_value = inner_values[0]
                    # Only stack if the values are tensors or arrays
                    if isinstance(first_inner_value, (torch.Tensor, np.ndarray)):
                        tensors_to_stack = [
                            torch.from_numpy(v) if isinstance(v, np.ndarray) else v for v in inner_values
                        ]
                        collated_inner_dict[inner_key] = torch.stack(tensors_to_stack, dim=0)
                    else:
                        # For non-tensor values (like strings), just keep them as a list
                        collated_inner_dict[inner_key] = inner_values
            collated_data[key] = collated_inner_dict

        # Handle primitive types like booleans, integers, and floats
        elif isinstance(first_non_none, (bool, int, float)):
            collated_data[key] = torch.tensor(non_none_values)

        # Fallback for other types, like strings
        else:
            collated_data[key] = values

    return Observation(**collated_data)

def dataset_check():
    """Check the dataset if the action matches the selected action via policy."""
    model_path = "/home/ronald/intel/geti-action/application/backend/act_policy_real_data.pt"
    repo_id = "rhecker/place-block"
    l_dm = LeRobotDataModule(
        repo_id=repo_id
    )

    dl =  DataLoader(
        l_dm.train_dataset,
        batch_size=1,
        collate_fn=_collate_observations,  # type: ignore[arg-type]
        shuffle=True,
    )

    act_model = ACTModel.load_from_checkpoint(model_path)
    act_model.eval()
    policy = ACT(model=act_model)
    policy.eval()

    for batch in dl:
        print(batch.state.shape) # [1, 6]
        print(batch.images["top"].shape) # [1, 3, 480, 640]
        obs = Observation(
            state=batch.state,
            images=batch.images,
        )
        print(obs)
        action = policy.select_action(obs)
        # for some random sample the following values
        print(batch.action) # tensor([[ -2.0124, -91.5318,  83.9262,  65.9940,  -7.6163,  23.2374]])
        print(action)       # tensor([[-14.0694, -16.2433, -40.7977,  65.4488,   2.0635,  19.5631]])

        break


def export_to_backend():
    pass

if __name__ == "__main__":
    export_to_backend()
    #dataset_check()
    #export_existing_model()
    #second()
    #asyncio.run(main())
