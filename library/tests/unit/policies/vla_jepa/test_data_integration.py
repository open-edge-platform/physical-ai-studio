# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: S101

"""Temporal dataset configuration before Lightning starts loader workers."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from physicalai.data import DataModule, Feature, FeatureType, Observation
from physicalai.data.lerobot.dataset import _LeRobotDatasetAdapter  # noqa: PLC2701
from physicalai.policies.vla_jepa import VLAJEPA, VLAJEPAConfig
from physicalai.train import Trainer


class TemporalDataset(_LeRobotDatasetAdapter):
    """Generate frames using the real adapter's reader-offset interface."""

    def __init__(self, config: VLAJEPAConfig) -> None:
        """Store an in-memory reader with unset temporal offsets."""
        self.config = config
        self._action_columns = None
        self._lerobot_dataset = SimpleNamespace(
            reader=SimpleNamespace(delta_indices=None),
            fps=30,
            tolerance_s=1e-4,
            features={"observation.images.front": {}, "observation.state": {}, "action": {}},
        )

    def __len__(self) -> int:
        """Return one batch of two samples."""
        return 2

    @property
    def observation_features(self) -> dict[str, Feature]:
        """The configured observation contract."""
        return {str(feature.name): feature for feature in self.config.input_features}

    @property
    def action_features(self) -> dict[str, Feature]:
        """The configured action contract."""
        return {str(feature.name): feature for feature in self.config.output_features}

    def __getitem__(self, index: int) -> Observation:
        """Return a sample using the worker's copy of the reader offsets."""
        del index
        images = {}
        for feature in self.config.input_features:
            if feature.ftype == FeatureType.VISUAL:
                key = f"observation.images.{feature.name}"
                shape = feature.shape
                if key in self.delta_indices:
                    shape = (len(self.delta_indices[key]), *shape)
                images[str(feature.name)] = torch.zeros(shape)
        action_steps = self.delta_indices.get("action")
        action_shape = (len(action_steps), self.config.action_dim) if action_steps else (self.config.action_dim,)
        return Observation(
            images=images,
            state=torch.zeros(self.config.state_dim),
            action=torch.zeros(action_shape),
            task="pick up the object",
        )


@pytest.mark.parametrize("num_workers", [0, 1])
@pytest.mark.parametrize("enable_world_model", [False, True])
def test_temporal_offsets_precede_worker_start(
    jepa_config: VLAJEPAConfig,
    monkeypatch: pytest.MonkeyPatch,
    num_workers: int,
    *,
    enable_world_model: bool,
) -> None:
    """The first prefetched batch contains JEPA clips and complete action chunks."""
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")
    config = replace(jepa_config, enable_world_model=enable_world_model)
    train = TemporalDataset(config)
    validation = TemporalDataset(config)
    datamodule = DataModule(
        train_dataset=train,
        val_eval_dataset=validation,
        train_batch_size=2,
        num_workers=num_workers,
    )
    policy = VLAJEPA(pretrained_name_or_path=None)
    policy.config = config
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        limit_val_batches=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    trainer.fit(policy, datamodule=datamodule)
    assert trainer.global_step == 1
    for dataset in (train, validation):
        if enable_world_model:
            assert dataset.delta_indices["observation.images.front"] == list(range(config.num_video_frames))
        else:
            assert "observation.images.front" not in dataset.delta_indices
        assert dataset.delta_indices["action"] == list(range(config.chunk_size))
