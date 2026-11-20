# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Runtime adapter trust-boundary tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import Mock

import numpy as np
import pytest
import torch

if TYPE_CHECKING:
    from pathlib import Path

from physicalai_lerobot_policy_plugin.runtime import LeRobotRuntimeAdapter, _to_lerobot_batch  # ruff: ignore[import-private-name]


def test_uint8_channel_last_camera_frames_become_unit_float_channel_first() -> None:
    """Match LeRobot's own camera preprocessing: uint8 HWC -> float32 CHW in [0, 1]."""
    frame = np.arange(6 * 5 * 3, dtype=np.uint8).reshape(6, 5, 3) * 2

    batch = _to_lerobot_batch({"images": {"top": frame}}, "cpu")

    image = batch["observation.images.top"]
    assert isinstance(image, torch.Tensor)
    assert image.dtype == torch.float32
    assert image.shape == (3, 6, 5)
    expected = torch.from_numpy(frame).permute(2, 0, 1).to(torch.float32) / 255
    torch.testing.assert_close(image, expected, rtol=0, atol=0)


@pytest.mark.parametrize("key", ["image", "images.wrist", "observation.images.top"])
def test_every_image_key_form_scales_uint8_frames(key: str) -> None:
    """Flat, dotted and native image keys all get the same uint8 conversion."""
    frame = np.full((2, 3, 3), 255, dtype=np.uint8)

    batch = _to_lerobot_batch({key: frame}, "cpu")

    (image,) = batch.values()
    assert isinstance(image, torch.Tensor)
    assert image.dtype == torch.float32
    assert torch.equal(image, torch.ones(3, 2, 3))


def test_float_camera_frames_keep_their_unit_scale() -> None:
    """Float frames are already in LeRobot's [0, 1] range and must not be rescaled."""
    frame = np.linspace(0, 1, 2 * 3 * 3, dtype=np.float64).reshape(3, 2, 3)

    batch = _to_lerobot_batch({"images": {"top": frame}}, "cpu")

    image = batch["observation.images.top"]
    assert isinstance(image, torch.Tensor)
    assert image.dtype == torch.float32
    torch.testing.assert_close(image, torch.from_numpy(frame).to(torch.float32))


def test_non_uint8_integer_camera_frames_are_rejected() -> None:
    """Integer frames other than uint8 have no defined [0, 1] scale, so fail loudly."""
    frame = np.zeros((2, 3, 3), dtype=np.uint16)

    with pytest.raises(TypeError, match=r"uint16.*observation\.images\.depth"):
        _to_lerobot_batch({"images": {"depth": frame}}, "cpu")


@pytest.mark.parametrize("dtype", [np.float64, np.float16, np.int64])
def test_numeric_state_is_float32(dtype: type[np.generic]) -> None:
    """Match LeRobot's ``.float()`` on state so float32 policies never see float64 or ints."""
    state = np.array([1, -2, 3], dtype=dtype)

    batch = _to_lerobot_batch({"state": state, "observation.environment_state": state}, "cpu")

    for key in ("observation.state", "observation.environment_state"):
        value = batch[key]
        assert isinstance(value, torch.Tensor)
        assert value.dtype == torch.float32
        torch.testing.assert_close(value, torch.tensor([1.0, -2.0, 3.0]))


def test_uint8_camera_actions_match_native_float_preprocessing(tmp_path: Path) -> None:
    """A real ACT package gives the same actions for raw uint8 frames as for LeRobot's float frames.

    Before uint8 scaling, LeRobot's normalizer cast its image statistics to uint8 and
    produced different (wrapped) values, so this compares full pre/post-processing.
    """
    from lerobot.configs import FeatureType, PolicyFeature  # ruff: ignore[import-outside-top-level]
    from lerobot.policies import make_pre_post_processors  # ruff: ignore[import-outside-top-level]
    from lerobot.policies.act.configuration_act import ACTConfig  # ruff: ignore[import-outside-top-level]
    from lerobot.policies.act.modeling_act import ACTPolicy  # ruff: ignore[import-outside-top-level]

    torch.manual_seed(0)
    config = ACTConfig(
        input_features={
            "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        device="cpu",
        chunk_size=2,
        n_action_steps=2,
        n_encoder_layers=1,
        n_decoder_layers=1,
        n_heads=2,
        dim_model=16,
        dim_feedforward=32,
        use_vae=False,
        pretrained_backbone_weights=None,
    )
    dataset_stats = {
        "observation.images.top": {
            "mean": torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1),
            "std": torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1),
        },
        "observation.state": {"mean": torch.zeros(2), "std": torch.ones(2)},
        "action": {"mean": torch.zeros(2), "std": torch.ones(2)},
    }
    native_dir = tmp_path / "pretrained_model"
    native_dir.mkdir()
    policy = ACTPolicy(config)
    preprocessor, postprocessor = make_pre_post_processors(policy_cfg=config, dataset_stats=dataset_stats)
    policy.save_pretrained(native_dir)
    preprocessor.save_pretrained(native_dir)
    postprocessor.save_pretrained(native_dir)

    rng = np.random.default_rng(0)
    camera_frame = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    camera_state = np.array([0.25, -0.5], dtype=np.float64)
    native_observation = {
        "observation.images.top": torch.from_numpy(camera_frame).permute(2, 0, 1).to(torch.float32) / 255,
        "observation.state": torch.from_numpy(camera_state).to(torch.float32),
    }

    adapter = LeRobotRuntimeAdapter(device="cpu")
    adapter.load(native_dir)
    policy.eval()
    for _ in range(3):  # Crosses the 2-step action chunk boundary.
        with torch.inference_mode():
            native_action = postprocessor(policy.select_action(preprocessor(native_observation)))
        runtime_action = adapter.predict({"images": {"top": camera_frame}, "state": camera_state})["action"]
        np.testing.assert_allclose(runtime_action, native_action.numpy(), rtol=1e-5, atol=1e-6)


def test_groot_package_is_rejected_before_loading_remote_policy_code(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reject unreviewed Groot before the policy factory can load its model class."""
    from lerobot.configs.policies import PreTrainedConfig  # ruff: ignore[import-outside-top-level]
    from lerobot.policies import factory  # ruff: ignore[import-outside-top-level]
    from lerobot.utils import import_utils  # ruff: ignore[import-outside-top-level]

    artifact = tmp_path / "groot-model"
    artifact.mkdir()
    (artifact / "config.json").write_text('{"type":"groot"}', encoding="utf-8")

    def fake_config_load(_cls: type, *_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(type="groot")

    monkeypatch.setattr(PreTrainedConfig, "from_pretrained", classmethod(fake_config_load))
    monkeypatch.setattr(import_utils, "register_third_party_plugins", lambda: None)
    get_policy_class = Mock(side_effect=AssertionError("Groot model factory must not run"))
    monkeypatch.setattr(factory, "get_policy_class", get_policy_class)

    with pytest.raises(ValueError, match=r"Groot.*remote code.*reviewed and pinned"):
        LeRobotRuntimeAdapter().load(artifact)

    get_policy_class.assert_not_called()
