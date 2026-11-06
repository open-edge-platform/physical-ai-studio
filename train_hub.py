# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Train standard PatchPolicy on Hub PushT224 with Studio's224 observation gym."""

import sys
from pathlib import Path

import torch
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lightning import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from physicalai.data.lerobot import LeRobotDataModule
from physicalai.policies import PatchPolicy
from physicalai.train import Trainer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pusht_history import PushTHistoryGym  # noqa: E402

REPO_ID = "alfieroddan/pushT-224x224"
REVISION = "9ba1652264d306cb9fa657e169b4f269e9bac1c5"


def main() -> None:
    seed_everything(42, workers=True)
    output = Path(__file__).resolve().parent / "hub_inhouse"
    output.mkdir(exist_ok=True)
    count = LeRobotDatasetMetadata(REPO_ID, revision=REVISION).total_episodes
    episodes = torch.randperm(
        count, generator=torch.Generator().manual_seed(42)
    ).tolist()
    policy = PatchPolicy(compile_model=True)
    gym = PushTHistoryGym(
        n_obs_steps=policy.n_obs_steps,
        observation_width=224,
        observation_height=224,
    )
    try:
        data = LeRobotDataModule(
            repo_id=REPO_ID,
            revision=REVISION,
            episodes=episodes[: int(0.9 * count)],
            train_batch_size=8,
            num_workers=0,
            val_gym=gym,
            num_rollouts_val=10,
        )
        trainer = Trainer(
            default_root_dir=output,
            max_epochs=400,
            accelerator="gpu",
            devices=1,
            precision="32-true",
            check_val_every_n_epoch=10,
            log_every_n_steps=100,
            logger=WandbLogger(
                project="patch-policy-pusht",
                name="dinov2-vqbet-hub224-inhouse",
                save_dir=output,
            ),
            callbacks=[ModelCheckpoint(save_last=True, save_top_k=0, every_n_epochs=1)],
        )
        trainer.fit(policy, datamodule=data)
    finally:
        gym.close()


if __name__ == "__main__":
    main()
