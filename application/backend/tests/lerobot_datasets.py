"""Record small but real LeRobot v3 datasets, video included, for tests that read or rewrite one."""

from pathlib import Path

import numpy as np

from internal_datasets.access_mode import DatasetAccessMode
from internal_datasets.lerobot.lerobot_dataset import InternalLeRobotDataset
from internal_datasets.lerobot.video_encoding_settings import VideoEncodingSettings

FRAMES_PER_EPISODE = 3
FEATURES = {
    "action": {"dtype": "float32", "shape": (2,), "names": ["shoulder.pos", "gripper.pos"]},
    "observation.state": {"dtype": "float32", "shape": (2,), "names": ["shoulder.pos", "gripper.pos"]},
    "observation.images.camera": {
        "dtype": "video",
        "shape": (32, 32, 3),
        "names": ["height", "width", "channels"],
    },
}


def record_dataset(root: Path, episode_tasks: list[str]) -> Path:
    """Record one episode per entry of ``episode_tasks``, each with that task, into ``root``."""
    dataset = InternalLeRobotDataset(
        root,
        recording_video_encoding_settings=VideoEncodingSettings(
            vcodec="libsvtav1", streaming_encoding=False, encoder_threads=1
        ),
        access_mode=DatasetAccessMode.RECORDING_MUTATION,
    )
    dataset.create(fps=10, features=FEATURES, robot_type="test")
    dataset.prepare_for_writing()
    for episode_index, task in enumerate(episode_tasks):
        for frame in range(FRAMES_PER_EPISODE):
            dataset._dataset.add_frame(
                {
                    "action": np.full(2, episode_index, dtype=np.float32),
                    "observation.state": np.full(2, frame, dtype=np.float32),
                    "observation.images.camera": np.full((32, 32, 3), 10 * episode_index, dtype=np.uint8),
                    "task": task,
                }
            )
        dataset.save_episode()
    dataset.finalize()
    return root
