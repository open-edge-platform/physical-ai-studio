# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tokenizer preparation using real tables, adapters, and quantizers."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset as HFDataset
from lightning.pytorch.callbacks import BatchSizeFinder, LearningRateFinder
from lightning.pytorch.strategies import SingleDeviceStrategy

from physicalai.data import Feature, FeatureType
from physicalai.data.lerobot.dataset import _LeRobotDatasetAdapter
from physicalai.policies.patch_policy import preparation
from physicalai.policies.patch_policy.components.heads import BehaviourTransformerHead
from physicalai.policies.patch_policy.components.heads.behavior_transformer import ResidualActionQuantizer
from physicalai.policies.patch_policy.config import PatchPolicyConfig


class _ActionSource:
    """In-memory LeRobot storage; preparation must never decode observation rows."""

    def __init__(self, table: HFDataset, episodes: list[dict[str, int]]) -> None:
        self.hf_dataset = table
        self.meta = SimpleNamespace(episodes=episodes)
        self.reader = SimpleNamespace(delta_indices={})

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, index: int) -> None:
        raise AssertionError("Tokenizer preparation should read the action table, not decode images.")


def _dataset(*, split_columns: bool = False) -> _LeRobotDatasetAdapter:
    rows: dict[str, list[int] | list[list[float]]] = {
        "index": [0, 1, 2, 3, 4],
        "episode_index": [0, 0, 0, 1, 1],
    }
    if split_columns:
        rows.update({
            "action.arm": [[0.0], [1.0], [2.0], [3.0], [4.0]],
            "action.gripper": [[10.0], [11.0], [12.0], [13.0], [14.0]],
        })
    else:
        rows["action"] = [[0.0], [1.0], [2.0], [3.0], [4.0]]
    source = _ActionSource(
        HFDataset.from_dict(rows),
        [
            {"dataset_from_index": 0, "dataset_to_index": 3},
            {"dataset_from_index": 3, "dataset_to_index": 5},
        ],
    )
    columns = ["action.gripper", "action.arm"] if split_columns else None
    adapter = _LeRobotDatasetAdapter.from_lerobot(source, action_columns=columns)
    adapter.delta_indices = dict.fromkeys(columns or ["action"], [-1, 0, 1])
    return adapter


def _config() -> PatchPolicyConfig:
    return PatchPolicyConfig(
        input_features=[Feature(name="front", ftype=FeatureType.VISUAL, shape=(3, 8, 8))],
        output_features=[Feature(name="action", ftype=FeatureType.ACTION, shape=(1,))],
        action_dim=1,
        n_obs_steps=2,
        chunk_size=2,
    )


def _quantizer() -> ResidualActionQuantizer:
    return ResidualActionQuantizer(1, 2, 4, 2, 1, action_scale=1.0)


def _policy() -> SimpleNamespace:
    head = BehaviourTransformerHead(
        4,
        1,
        2,
        n_obs_steps=2,
        n_patches=1,
        hidden_dim=8,
        num_layers=1,
        num_heads=1,
        latent_dim=4,
        num_embeddings=2,
        groups=1,
        action_scale=1.0,
    )
    trainer = SimpleNamespace(
        strategy=SingleDeviceStrategy(device="cpu"),
        world_size=1,
        callbacks=[],
        datamodule=SimpleNamespace(train_dataset=_dataset()),
    )
    return SimpleNamespace(
        model=SimpleNamespace(action_head=head),
        trainer=trainer,
        config=_config(),
        device=torch.device("cpu"),
        vqvae_iters=1,
        vqvae_batch_size=2,
    )


def test_action_reader_only_accepts_lerobot_adapters() -> None:
    with pytest.raises(TypeError, match="only supports LeRobot datasets"):
        list(preparation._lerobot_action_targets(HFDataset.from_dict({"action": [[1.0]]})))


def test_action_reader_clamps_at_episode_boundaries_and_preserves_split_column_order() -> None:
    targets = list(preparation._lerobot_action_targets(_dataset(split_columns=True)))

    expected = torch.tensor([
        [[10.0, 0.0], [10.0, 0.0], [11.0, 1.0]],
        [[10.0, 0.0], [11.0, 1.0], [12.0, 2.0]],
        [[11.0, 1.0], [12.0, 2.0], [12.0, 2.0]],
        [[13.0, 3.0], [13.0, 3.0], [14.0, 4.0]],
        [[13.0, 3.0], [14.0, 4.0], [14.0, 4.0]],
    ])
    torch.testing.assert_close(torch.stack(targets), expected)
    assert all(target.device.type == "cpu" for target in targets)


@pytest.mark.parametrize("indices", [[0, 2, 3, 4], [0, 1]])
def test_action_reader_rejects_missing_frames_inside_and_beyond_the_training_split(indices: list[int]) -> None:
    dataset = _dataset()
    dataset._lerobot_dataset.hf_dataset = dataset._lerobot_dataset.hf_dataset.select(indices)

    with pytest.raises(ValueError, match="outside the training split"):
        list(preparation._lerobot_action_targets(dataset))


def test_collection_returns_unique_overlapping_windows_including_boundary_padding() -> None:
    windows = preparation._collect_action_windows(_dataset(), _config())

    expected = torch.tensor([
        [[0.0], [0.0]],
        [[0.0], [1.0]],
        [[1.0], [2.0]],
        [[2.0], [2.0]],
        [[3.0], [3.0]],
        [[3.0], [4.0]],
        [[4.0], [4.0]],
    ])
    torch.testing.assert_close(windows, expected)
    assert windows.shape == (7, 2, 1)
    assert windows.dtype == torch.float32
    assert windows.device.type == "cpu"


def test_collection_validates_target_shapes_and_empty_training_data() -> None:
    dataset = _dataset()
    with pytest.raises(ValueError, match="action targets shaped"):
        preparation._collect_action_windows(dataset, replace(_config(), n_obs_steps=3))
    dataset._lerobot_dataset.hf_dataset = dataset._lerobot_dataset.hf_dataset.select([])
    with pytest.raises(ValueError, match="empty training dataset"):
        preparation._collect_action_windows(dataset, _config())


def test_preparation_validates_device_tuning_precision_and_training_dataset() -> None:
    policy = _policy()
    tokenizer = policy.model.action_head.quantizer
    assert preparation._validate_preparation(policy, tokenizer) is policy.trainer.datamodule.train_dataset
    policy.trainer.world_size = 2
    with pytest.raises(ValueError, match="single-device"):
        preparation._validate_preparation(policy, tokenizer)
    policy.trainer.world_size = 1
    for callback in [BatchSizeFinder(), LearningRateFinder()]:
        policy.trainer.callbacks = [callback]
        with pytest.raises(ValueError, match="tuning"):
            preparation._validate_preparation(policy, tokenizer)
    policy.trainer.callbacks = []
    with pytest.raises(ValueError, match="FP32 parameters"):
        preparation._validate_preparation(policy, tokenizer.double())
    tokenizer.float()
    policy.trainer.datamodule = None
    with pytest.raises(ValueError, match="train_dataset"):
        preparation._validate_preparation(policy, tokenizer)


def test_tiny_real_tokenizer_training_updates_parameters_and_ema_even_under_no_grad() -> None:
    torch.manual_seed(7)
    tokenizer = _quantizer().train()
    before_weight = tokenizer.encoder.fc.weight.detach().clone()
    before_codebook = tokenizer.codebook.clone()
    windows = torch.tensor([[[0.0], [1.0]], [[1.0], [2.0]], [[2.0], [2.0]]], dtype=torch.float64)

    with torch.no_grad():
        preparation._train_tokenizer(tokenizer, windows, device=torch.device("cpu"), num_passes=2, batch_size=2)

    assert not torch.equal(tokenizer.encoder.fc.weight, before_weight)
    assert not torch.equal(tokenizer.codebook, before_codebook)
    assert not bool(tokenizer.fitted)
    assert all(torch.isfinite(value).all() for value in tokenizer.state_dict().values())
    loss, _ = tokenizer.compute_loss(windows.float())
    assert torch.isfinite(loss)


def test_real_tokenizer_training_rejects_nonfinite_loss() -> None:
    tokenizer = _quantizer().train()
    with pytest.raises(RuntimeError, match="non-finite loss"):
        preparation._train_tokenizer(
            tokenizer, torch.full((2, 2, 1), float("nan")), device=torch.device("cpu"), num_passes=1, batch_size=2
        )
    assert not bool(tokenizer.fitted)


@pytest.mark.parametrize("fail", [False, True])
def test_fitting_trains_a_copy_and_commits_only_after_success(monkeypatch: pytest.MonkeyPatch, fail: bool) -> None:
    policy = _policy()
    original = policy.model.action_head.quantizer
    before = {key: value.clone() for key, value in original.state_dict().items()}

    def train_copy(tokenizer: ResidualActionQuantizer, windows: torch.Tensor, **kwargs: object) -> None:
        assert tokenizer is not original
        assert tokenizer.training
        assert all(parameter.requires_grad for parameter in tokenizer.parameters())
        assert all(parameter.dtype == torch.float32 for parameter in tokenizer.parameters())
        assert windows.shape == (7, 2, 1)
        assert kwargs == {"device": torch.device("cpu"), "num_passes": 1, "batch_size": 2}
        with torch.no_grad():
            tokenizer.encoder.fc.weight.add_(1.0)
            tokenizer.codebook.add_(2.0)
        for key, value in original.state_dict().items():
            torch.testing.assert_close(value, before[key])
        if fail:
            raise RuntimeError("fitting interrupted")

    monkeypatch.setattr(preparation, "_train_tokenizer", train_copy)
    if fail:
        with pytest.raises(RuntimeError, match="fitting interrupted"):
            preparation.fit_tokenizer(policy)
        for key, value in original.state_dict().items():
            torch.testing.assert_close(value, before[key])
    else:
        preparation.fit_tokenizer(policy)
        torch.testing.assert_close(original.encoder.fc.weight, before["encoder.fc.weight"] + 1.0)
        torch.testing.assert_close(original.codebook, before["codebook"] + 2.0)
        assert bool(original.fitted)
    assert policy.model.action_head.quantizer is original
    assert not original.training
    assert all(not parameter.requires_grad for parameter in original.parameters())


def test_fitting_skips_non_behavior_heads_and_fitted_tokenizers_before_accessing_trainer() -> None:
    with pytest.raises(ValueError, match="initialized model"):
        preparation.fit_tokenizer(SimpleNamespace(model=None))
    preparation.fit_tokenizer(SimpleNamespace(model=SimpleNamespace(action_head=torch.nn.Identity())))
    policy = _policy()
    tokenizer = policy.model.action_head.quantizer
    tokenizer.fitted.fill_(True)
    before = {key: value.clone() for key, value in tokenizer.state_dict().items()}
    del policy.trainer

    preparation.fit_tokenizer(policy)

    for key, value in tokenizer.state_dict().items():
        torch.testing.assert_close(value, before[key])
