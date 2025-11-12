import inspect
import sys
import time

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.robots.utils import make_robot_from_config
from lerobot.utils.control_utils import predict_action
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import get_safe_torch_device
from pydantic import BaseModel

from schemas import CameraConfig, Dataset, RobotConfig
from utils.camera import build_camera_config
from utils.robot import make_lerobot_robot_config_from_robot

checkpoint_path = sys.argv[1]


class InferenceConfig(BaseModel):
    task: str
    device: str
    policy: str
    cameras: list[CameraConfig]
    robot: RobotConfig
    dataset: Dataset


config = InferenceConfig(
    task="Place block on paper",
    policy="rhecker/duplo_act_policy",
    device="cuda",
    dataset={
        "id": "c574e5a0-f57e-4103-b823-4f2ef41a1969",
        "name": "eval_place-block",
        "path": "/home/ronald/.cache/huggingface/lerobot/rhecker/eval_place-block",
        "project_id": "a7ec475b-3094-4a8e-9479-2bea8157a90b",
    },
    cameras=[
        {
            "id": "e55489e2-5761-4c4b-8bc3-845bdfae6c50",
            "port_or_device_id": "/dev/video4",
            "name": "top",
            "driver": "webcam",
            "width": 640,
            "height": 480,
            "fps": 30,
            "use_depth": False,
        },
        {
            "id": "e55489e2-5761-4c4b-8bc3-845bdfae6c50",
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


def push_to_hub_own(policy):
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from huggingface_hub import HfApi

    api = HfApi()
    repo_id = api.create_repo(repo_id="duplo-bright", private=True, exist_ok=True).repo_id

    # Push the files to the repo in a single commit
    with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        saved_path = Path(tmp) / repo_id

        policy.save_pretrained(saved_path)  # Calls _save_pretrained and stores model tensors

        card = policy.generate_model_card("duplo-bright", policy.config.type, policy.config.license, policy.config.tags)
        card.save(str(saved_path / "README.md"))

        policy.save_pretrained(saved_path)  # Calls _save_pretrained and stores train config

        commit_info = api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=saved_path,
            commit_message="Upload policy weights, train config and readme",
            allow_patterns=["*.safetensors", "*.json", "*.yaml", "*.md"],
            ignore_patterns=["*.tmp", "*.log"],
        )

        print(f"Model pushed to {commit_info.repo_url.url}")


sig = inspect.signature(ACTConfig.from_pretrained)


cameras = {camera.name: build_camera_config(camera) for camera in config.cameras}
follower_config = make_lerobot_robot_config_from_robot(config.robot, cameras)

robot = make_robot_from_config(follower_config)
robot.connect()

cli_overrides = {
    "type": "act",
}
# policy_path = "/home/ronald/projects/lerobot/outputs/train/2025-10-06/11-57-35_act/checkpoints/last/pretrained_model"
# policy_config = PreTrainedConfig.from_pretrained(policy_path)

dataset = LeRobotDataset(config.dataset.name, config.dataset.path)
# policy = make_policy(
#    policy_config,
#    ds_meta=dataset.meta,
# )
# push_to_hub_own(policy)

# "/home/ronald/projects/lerobot/outputs/train/duplo-top/checkpoints/0200000/pretrained_model/"
# policy = SmolVLAPolicy.from_pretrained(checkpoint_path)
policy = ACTPolicy.from_pretrained( checkpoint_path)

timestamp = 0
start_episode_t = time.perf_counter()
timeout = 60
while timestamp < timeout:
    start_loop_t = time.perf_counter()
    observation = robot.get_observation()
    observation_frame = build_dataset_frame(dataset.features, observation, prefix="observation")

    action_values = predict_action(
        observation_frame,
        policy,
        get_safe_torch_device(policy.config.device),
        policy.config.use_amp,
        task=config.task,
        robot_type=robot.robot_type,
    )
    action = {key: action_values[i].item() for i, key in enumerate(robot.action_features)}
    sent_action = robot.send_action(action)

    dt_s = time.perf_counter() - start_loop_t
    busy_wait(1 / dataset.fps - dt_s)

robot.disconnect()
