# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Exercise PATCH preparation through real Lightning training lifecycles."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import torch
from datasets import Dataset as HFDataset
from lerobot.datasets.dataset_reader import DatasetReader
from lerobot.datasets.io_utils import hf_transform_to_torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lightning.pytorch import Trainer as LightningTrainer
from lightning.pytorch.callbacks import Callback
from physicalai.data import DataModule, Observation
from physicalai.data.lerobot.dataset import _LeRobotDatasetAdapter
from physicalai.policies.patch_policy import PatchPolicy, preparation
from physicalai.policies.patch_policy.components.heads.behavior_transformer import ResidualActionQuantizer
from physicalai.train import Trainer
from physicalai.train.callbacks import PolicyDatasetInteraction, ProgressReportingCallback
from physicalai.train.utils import reformat_dataset_to_match_policy


@pytest.fixture(autouse=True)
def small_cpu_workload():
    """Avoid large thread-pool overhead for tiny CPU training batches."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(7)
        yield
    torch.set_num_threads(threads)


@pytest.fixture
def training_data(tmp_path: Path):
    """Build real LeRobot readers and Studio adapters without files or downloads."""
    camera = "observation.images.front"
    features = {
        camera: {"dtype": "image", "shape": (3, 32, 32)},
        "action": {"dtype": "float32", "shape": (1,)},
    }
    stats = {
        key: {"mean": np.zeros(size), "std": np.ones(size), "min": -np.ones(size), "max": np.ones(size)}
        for key, size in ((camera, 3), ("action", 1))
    }
    meta = SimpleNamespace(
        camera_keys=[camera], video_keys=[], depth_keys=[], image_keys=[camera],
        features=features, stats=stats, fps=10, total_frames=6,
        episodes={
            0: {"dataset_from_index": 0, "dataset_to_index": 4},
            1: {"dataset_from_index": 4, "dataset_to_index": 6},
        },
        tasks=pd.DataFrame(index=["test task"]),
    )
    datasets = []
    for episode, indexes in ((0, list(range(4))), (1, [4, 5])):
        rows = {
            "index": indexes,
            "episode_index": [episode] * len(indexes),
            "frame_index": list(range(len(indexes))),
            "task_index": [0] * len(indexes),
            "timestamp": [idx / 10 for idx in range(len(indexes))],
            "action": [[idx / 10 if episode == 0 else 10.0 + idx] for idx in indexes],
            camera: [np.full((3, 32, 32), idx / 10, dtype=np.float32) for idx in indexes],
        }
        reader = DatasetReader(meta, tmp_path, [episode], 1e-4, "pyav", None, None)
        reader.hf_dataset = HFDataset.from_dict(rows).with_transform(hf_transform_to_torch)
        reader._build_index_mapping()
        source = LeRobotDataset.__new__(LeRobotDataset)
        source.reader = reader
        source.meta = meta
        source.writer = None
        source.image_transforms = None
        source.tolerance_s = 1e-4
        datasets.append(_LeRobotDatasetAdapter.from_lerobot(source))
    return datasets


def make_policy(action_head: str, *, n_obs_steps: int = 1) -> PatchPolicy:
    return PatchPolicy(
        encoder_name="resnet18", encoder_output="average", encoder_pretrained=False,
        image_size=(32, 32), action_head=action_head,
        n_obs_steps=n_obs_steps, chunk_size=2, n_action_steps=1,
        hidden_dim=8, num_layers=1, num_heads=2, dropout=0.0,
        num_train_timesteps=2, num_inference_steps=2,
        vq_latent_dim=4, vq_num_embeddings=2, vq_num_groups=2,
        vqvae_iters=1, vqvae_batch_size=2,
    )


def make_data(training_data) -> DataModule:
    train, validation = training_data
    return DataModule(
        train, train_batch_size=2, val_eval_dataset=validation, val_batch_size=2,
        num_workers=0, pin_memory=False,
    )


def make_trainer(tmp_path: Path, callbacks=(), *, epochs: int = 1, trainer_class=Trainer):
    # Leave max_steps unset: estimating steps must actually initialize the loader.
    callbacks = list(callbacks)
    if trainer_class is LightningTrainer:
        callbacks.insert(0, PolicyDatasetInteraction())
    return trainer_class(
        accelerator="cpu", devices=1, precision="32-true", max_epochs=epochs,
        default_root_dir=tmp_path, callbacks=callbacks, num_sanity_val_steps=1,
        logger=False, enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False, limit_val_batches=1,
    )


class FitProbe(Callback):
    """Inspect the real optimization, encoder, and validation paths."""

    def __init__(self, data: DataModule, *, expected_steps: int = 2) -> None:
        self.data = data
        self.raw = data.train_dataset
        self.validation = data.val_eval_dataset
        self.expected_steps = expected_steps
        self.encoder_calls = []
        self.train_batches = 0
        self.validation_batches = 0
        self.tokenizer_state = None
        self.started = False

    def on_validation_start(self, trainer, pl_module):
        if self.started:
            return
        self.started = True
        assert self.data.train_dataset is self.raw
        assert self.data.val_eval_dataset is self.validation
        head = pl_module.model.action_head
        if hasattr(head, "quantizer"):
            assert head.quantizer.fitted
            assert not head.quantizer.training
            self.tokenizer_state = {key: value.clone() for key, value in head.quantizer.state_dict().items()}
        optimizer_ids = {id(parameter) for group in trainer.optimizers[0].param_groups for parameter in group["params"]}
        assert optimizer_ids == {id(parameter) for parameter in head.parameters() if parameter.requires_grad}
        assert optimizer_ids == {id(parameter) for parameter in pl_module.parameters() if parameter.requires_grad}
        assert optimizer_ids.isdisjoint(id(parameter) for parameter in pl_module.model.encoder.parameters())
        if self.tokenizer_state is not None:
            assert optimizer_ids.isdisjoint(id(parameter) for parameter in head.quantizer.parameters())
        self.head_weights = {
            name: parameter.detach().clone() for name, parameter in head.named_parameters() if parameter.requires_grad
        }
        self.encoder_state = {key: value.clone() for key, value in pl_module.model.encoder.state_dict().items()}

        def encoder_forward(encoder, _inputs, output):
            assert not encoder.backbone.training
            assert not output.requires_grad
            self.encoder_calls.append("train" if trainer.training else "validation")

        self.handle = pl_module.model.encoder.register_forward_hook(encoder_forward)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        assert trainer.estimated_stepping_batches == self.expected_steps
        assert trainer.train_dataloader.dataset is self.raw
        assert isinstance(batch, Observation)
        assert batch.images["front"].shape == (2, pl_module.n_obs_steps, 3, 32, 32)
        assert batch.action.shape == (2, pl_module.n_obs_steps + pl_module.chunk_size - 1, 1)
        self.train_batches += 1

    def on_after_backward(self, trainer, pl_module):
        assert all(
            not parameter.requires_grad and parameter.grad is None for parameter in pl_module.model.encoder.parameters()
        )
        assert any(parameter.grad is not None for parameter in pl_module.model.action_head.parameters())

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        assert isinstance(batch, Observation)
        assert batch.images is not None
        self.validation_batches += 1

    def on_fit_end(self, trainer, pl_module):
        assert self.train_batches == 2
        assert self.validation_batches == 2  # Sanity check and end-of-epoch validation.
        assert self.encoder_calls.count("validation") == self.validation_batches
        assert self.encoder_calls.count("train") == self.train_batches
        head = pl_module.model.action_head
        assert any(
            not torch.equal(parameter, self.head_weights[name])
            for name, parameter in head.named_parameters() if parameter.requires_grad
        )
        for key, value in pl_module.model.encoder.state_dict().items():
            torch.testing.assert_close(value, self.encoder_state[key], rtol=0, atol=0)
        if self.tokenizer_state is not None:
            for key, value in head.quantizer.state_dict().items():
                torch.testing.assert_close(value, self.tokenizer_state[key], rtol=0, atol=0)
            assert all(
                not parameter.requires_grad and parameter.grad is None for parameter in head.quantizer.parameters()
            )
        self.handle.remove()


@pytest.mark.parametrize("action_head", ["diffusion", "behavior_transformer"])
@pytest.mark.parametrize("trainer_class", [Trainer, LightningTrainer])
@pytest.mark.parametrize("workers", [0, 1])
def test_fit_prepares_before_step_estimation_and_keeps_validation_raw(
    training_data, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action_head: str, trainer_class, workers: int,
) -> None:
    data = make_data(training_data)
    data.num_workers = workers
    policy = make_policy(action_head, n_obs_steps=2)
    probe = FitProbe(data)
    original_loader = data.train_dataloader

    def loader():
        assert data.train_dataset is training_data[0]
        assert data.train_dataset.delta_indices["observation.images.front"] == [-1, 0]
        assert data.train_dataset.delta_indices["action"] == [-1, 0, 1]
        return original_loader()

    monkeypatch.setattr(data, "train_dataloader", loader)
    tokenizer_batches = []
    original_loss = ResidualActionQuantizer.compute_loss

    def tokenizer_loss(tokenizer, actions):
        tokenizer_batches.append(actions.detach().clone())
        return original_loss(tokenizer, actions)

    monkeypatch.setattr(ResidualActionQuantizer, "compute_loss", tokenizer_loss)
    progress = ProgressReportingCallback(report=lambda *_args: None, should_stop=lambda: False)
    trainer = make_trainer(tmp_path, [progress, probe], trainer_class=trainer_class)
    trainer.fit(policy, datamodule=data)

    assert trainer.global_step == 2
    assert data.train_dataset.delta_indices["observation.images.front"] == [-1, 0]
    assert data.train_dataset.delta_indices["action"] == [-1, 0, 1]
    assert data.train_dataset is training_data[0]
    assert data.val_eval_dataset is training_data[1]
    if action_head == "behavior_transformer":
        targets = torch.stack([training_data[0][idx].action for idx in range(4)])
        expected = targets.unfold(1, policy.chunk_size, 1).transpose(-1, -2).reshape(-1, policy.chunk_size, 1)
        actual = torch.cat(tokenizer_batches)
        assert len(tokenizer_batches) == (len(expected.unique(dim=0)) + 1) // 2
        torch.testing.assert_close(actual.unique(dim=0), expected.unique(dim=0))
    else:
        assert not tokenizer_batches


@pytest.mark.parametrize("trainer_class", [Trainer, LightningTrainer])
def test_checkpoint_resume_skips_ready_tokenizer(
    training_data, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, trainer_class,
) -> None:
    data = make_data(training_data)
    policy = make_policy("behavior_transformer")
    trainer = make_trainer(tmp_path, trainer_class=trainer_class)
    trainer.fit(policy, datamodule=data)
    path = tmp_path / "resume.ckpt"
    trainer.save_checkpoint(path)
    tokenizer_state = {key: value.clone() for key, value in policy.model.action_head.quantizer.state_dict().items()}
    fit_tokenizer = Mock(side_effect=AssertionError("A restored, fitted tokenizer must not be fitted again."))
    monkeypatch.setattr(ResidualActionQuantizer, "compute_loss", fit_tokenizer)
    restored = make_policy("behavior_transformer")
    probe = FitProbe(data, expected_steps=4)
    resumed = make_trainer(tmp_path, [probe], epochs=2, trainer_class=trainer_class)

    resumed.fit(restored, datamodule=data, ckpt_path=path)

    assert resumed.global_step == 4
    fit_tokenizer.assert_not_called()
    for key, value in restored.model.action_head.quantizer.state_dict().items():
        torch.testing.assert_close(value, tokenizer_state[key], rtol=0, atol=0)
    assert data.train_dataset is training_data[0]


@pytest.mark.parametrize("trainer_class", [Trainer, LightningTrainer])
def test_tokenizer_failure_leaves_registered_quantizer_unchanged(
    training_data, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, trainer_class,
) -> None:
    data = make_data(training_data)
    policy = make_policy("behavior_transformer")
    policy.input_features = list(data.train_dataset.observation_features.values())
    policy.output_features = list(data.train_dataset.action_features.values())
    policy.configure_model()
    quantizer = policy.model.action_head.quantizer
    original_state = {key: value.clone() for key, value in quantizer.state_dict().items()}
    original_loss = ResidualActionQuantizer.compute_loss
    fitted_tokenizers = []

    def fail_tokenizer(tokenizer, actions):
        assert tokenizer is not quantizer
        fitted_tokenizers.append(tokenizer)
        if len(fitted_tokenizers) == 2:
            assert any(
                not torch.equal(value, original_state[key]) for key, value in tokenizer.state_dict().items()
            )
            raise RuntimeError("injected tokenizer failure")
        return original_loss(tokenizer, actions)

    monkeypatch.setattr(ResidualActionQuantizer, "compute_loss", fail_tokenizer)
    trainer = make_trainer(tmp_path, trainer_class=trainer_class)
    with pytest.raises(RuntimeError, match="injected tokenizer failure"):
        trainer.fit(policy, datamodule=data)

    assert len(fitted_tokenizers) == 2
    assert fitted_tokenizers[0] is fitted_tokenizers[1]
    assert policy.model.action_head.quantizer is quantizer
    assert not quantizer.fitted
    for key, value in quantizer.state_dict().items():
        torch.testing.assert_close(value, original_state[key], rtol=0, atol=0)
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in quantizer.parameters())


@pytest.mark.parametrize("trainer_class", [Trainer, LightningTrainer])
def test_step_estimation_precedes_tokenizer_fitting_without_callback_reordering(
    training_data, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, trainer_class,
) -> None:
    data = make_data(training_data)
    policy = make_policy("behavior_transformer")
    progress = ProgressReportingCallback(report=lambda *_args: None, should_stop=lambda: False)
    trainer = make_trainer(tmp_path, [progress], trainer_class=trainer_class)
    callbacks = list(trainer.callbacks)
    original_fit = preparation.fit_tokenizer

    def fit(policy):
        assert trainer.train_dataloader is not None
        assert trainer.train_dataloader.dataset is data.train_dataset
        assert not policy.model.action_head.quantizer.fitted
        original_fit(policy)

    monkeypatch.setattr("physicalai.policies.patch_policy.policy.fit_tokenizer", fit)

    trainer.fit(policy, datamodule=data)

    assert trainer.callbacks == callbacks
    assert policy.model.action_head.quantizer.fitted
    assert data.train_dataset is training_data[0]


@pytest.mark.parametrize("n_obs_steps", [1, 2])
def test_raw_action_targets_match_reader_windows_without_decoding_images(
    training_data, monkeypatch: pytest.MonkeyPatch, n_obs_steps: int,
) -> None:
    data = make_data(training_data)
    policy = make_policy("behavior_transformer", n_obs_steps=n_obs_steps)
    policy.input_features = list(data.train_dataset.observation_features.values())
    policy.output_features = list(data.train_dataset.action_features.values())
    policy.configure_model()
    reformat_dataset_to_match_policy(policy, data)
    dataset = data.train_dataset
    expected = [dataset[index].action.clone() for index in range(len(dataset))]
    deltas = copy.deepcopy(dataset.delta_indices)
    reader = Mock(side_effect=AssertionError("Action-only preparation must not decode images."))
    monkeypatch.setattr(DatasetReader, "get_item", reader)

    actual = list(preparation._action_targets(dataset))

    assert len(actual) == len(expected)
    for target, wanted in zip(actual, expected, strict=True):
        assert target.shape == (n_obs_steps + policy.chunk_size - 1, 1)
        torch.testing.assert_close(target, wanted, rtol=0, atol=0)
    policy.trainer = SimpleNamespace(
        strategy=preparation.SingleDeviceStrategy(device="cpu"), world_size=1, callbacks=[], datamodule=data,
    )
    preparation.fit_tokenizer(policy)
    assert policy.model.action_head.quantizer.fitted
    reader.assert_not_called()
    assert dataset.delta_indices == deltas
