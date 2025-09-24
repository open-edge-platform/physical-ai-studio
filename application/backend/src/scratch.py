#!/usr/bin/env python3

from pathlib import Path
from lerobot.constants import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from utils.dataset import get_local_repository_ids

home = HF_LEROBOT_HOME
ids = get_local_repository_ids(home)

print(str(LeRobotDatasetMetadata(ids[0], home / ids[0]).root))
