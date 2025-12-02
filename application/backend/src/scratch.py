#!/usr/bin/env python3
import torch

from datasets.arrow_dataset import Column
from lerobot.datasets.lerobot_dataset import LeRobotDataset


repo_id = "place-block"
root = "/home/ronald/.cache/geti_action/datasets/place-block"
dataset = LeRobotDataset(repo_id, root)
hf_dataset = dataset.load_hf_dataset()

for ep_idx in hf_dataset.unique("episode_index"):
    print(ep_idx)

#data = hf_dataset.filter(lambda x: x["episode_index"] == 0)
episode = dataset.meta.episodes[1]
print(list(dataset.meta.tasks.to_dict()["task_index"].keys()))
#print({video_key: episode[f"videos/{video_key}/from_timestamp"] for video_key in dataset.meta.video_keys})
#print(episode)
#from_idx = episode["dataset_from_index"]
#to_idx = episode["dataset_to_index"]
##print()
#print(torch.stack(hf_dataset["action"][from_idx:to_idx]).shape)
#print(column.__contains__)
