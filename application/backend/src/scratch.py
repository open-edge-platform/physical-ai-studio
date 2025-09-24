#!/usr/bin/env python3

from lerobot.constants import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

from utils.dataset import get_local_repository_ids

home = HF_LEROBOT_HOME
ids = get_local_repository_ids(home)

meta = LeRobotDatasetMetadata(ids[0], home / ids[0])

print(meta.features["observation.images.front"])
print(
    [
        {
            "name": name.split(".")[-1],
            "width": feature["info"]["video.width"],
            "height": feature["info"]["video.height"],
            "fps": feature["info"]["video.fps"],
            "type": "OpenCV",
            "use_depth": False,
            "port_or_id": "",
        }
        for name, feature in meta.features.items()
        if feature["dtype"] == "video"
    ]
)
