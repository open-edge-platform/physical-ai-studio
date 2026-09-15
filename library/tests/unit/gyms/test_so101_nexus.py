# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the SO101-Nexus gym adapter."""

import math
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from physicalai.gyms import SO101NexusGym
from physicalai.gyms import so101_nexus as adapter_module


class _FakeActionSpace:
    low = np.array([-math.pi] * 5 + [0.0], dtype=np.float32)
    high = np.array([math.pi] * 5 + [1.0], dtype=np.float32)

    @staticmethod
    def sample() -> np.ndarray:
        return np.zeros(6, dtype=np.float32)


class _FakeEnv:
    action_space = _FakeActionSpace()
    unwrapped = SimpleNamespace(task_description="Pick up the red cube.")

    def __init__(self) -> None:
        self.last_action: np.ndarray | None = None

    def step(self, action: np.ndarray):
        self.last_action = action
        return _raw_observation(), 1.0, False, False, {"success": False}


def _raw_observation() -> dict[str, np.ndarray]:
    return {
        "state": np.array([0.0, 0.1, -0.2, 0.3, -0.4, 0.5], dtype=np.float32),
        "wrist_camera": np.full((8, 8, 3), 255, dtype=np.uint8),
        "overhead_camera": np.zeros((8, 8, 3), dtype=np.uint8),
    }


def _install_fake_so101(monkeypatch) -> None:
    module = ModuleType("so101_nexus")

    def to_dataset(values, *, gripper_limits_rad):
        values = np.asarray(values).copy()
        values[..., :5] = np.rad2deg(values[..., :5])
        low, high = gripper_limits_rad
        values[..., 5] = (values[..., 5] - low) / (high - low) * 100.0
        return values

    def to_sim(values, *, gripper_limits_rad):
        values = np.asarray(values).copy()
        values[..., :5] = np.deg2rad(values[..., :5])
        low, high = gripper_limits_rad
        values[..., 5] = low + values[..., 5] / 100.0 * (high - low)
        return values

    setattr(module, "sim_qpos_to_dataset_row", to_dataset)
    setattr(module, "dataset_row_to_sim_qpos", to_sim)
    monkeypatch.setitem(sys.modules, "so101_nexus", module)
    monkeypatch.setattr(adapter_module, "so101_nexus", module)
    monkeypatch.setattr(adapter_module, "_SO101_NEXUS_AVAILABLE", True)


def _make_adapter() -> SO101NexusGym:
    adapter = SO101NexusGym.__new__(SO101NexusGym)
    adapter._env = _FakeEnv()
    adapter._device = torch.device("cpu")
    adapter._is_vectorized = False
    adapter._gripper_limits_rad = (0.0, 1.0)
    adapter._task_description_override = None
    adapter._camera_key_map = {
        "wrist_camera": "wrist",
        "overhead_camera": "overhead",
    }
    return adapter


def _replace_gymnasium_init(monkeypatch) -> dict:
    captured_kwargs = {}

    def fake_init(adapter, **_kwargs) -> None:
        captured_kwargs.update(_kwargs)
        adapter._env = _FakeEnv()
        adapter._device = torch.device("cpu")
        adapter._is_vectorized = False

    monkeypatch.setattr(adapter_module.GymnasiumGym, "__init__", fake_init)
    return captured_kwargs


def test_converts_visual_observation_to_runtime_contract(monkeypatch) -> None:
    _install_fake_so101(monkeypatch)
    adapter = _make_adapter()

    observation = adapter.to_observation(adapter._normalize_raw_obs(_raw_observation()))

    assert observation.state is not None
    dataset_state = np.array(
        [*np.rad2deg([0.0, 0.1, -0.2, 0.3, -0.4]), 50.0],
        dtype=np.float32,
    )
    expected_state = np.array(
        [0.0, 5.6636405, -11.839708, 17.028671, -13.606686, 49.965446],
        dtype=np.float32,
    )
    torch.testing.assert_close(
        observation.state,
        torch.tensor(expected_state).unsqueeze(0),
    )
    assert isinstance(observation.images, dict)
    assert observation.images["wrist"].shape == (1, 3, 8, 8)
    assert observation.images["wrist"].max() == 1.0
    assert observation.images["overhead"].max() == 0.0
    assert observation.task == ["Pick up the red cube."]


def test_converts_policy_action_to_simulator_radians(monkeypatch) -> None:
    _install_fake_so101(monkeypatch)
    adapter = _make_adapter()

    runtime_action = np.array([75.0, -75.0, 45.0, 0.0, 50.0, 25.0], dtype=np.float32)
    runtime_ticks = np.array([3079, 1174, 2499, 1922, 3015, 2407], dtype=np.int32)
    expected_dataset_action = adapter._ticks_to_dataset(runtime_ticks)
    adapter.step(torch.from_numpy(runtime_action))

    assert adapter._env.last_action is not None
    expected_sim_action = np.deg2rad(expected_dataset_action)
    expected_sim_action[-1] = expected_dataset_action[-1] / 100.0
    np.testing.assert_allclose(adapter._env.last_action, expected_sim_action, rtol=1e-6)


def test_runtime_conversion_clips_like_hardware() -> None:
    """Runtime coordinates retain the hardware driver's bounded ranges."""
    adapter = _make_adapter()
    dataset = np.array([1000.0, -1000.0, 0.0, 0.0, 0.0, 150.0], dtype=np.float32)

    runtime = adapter._dataset_to_runtime(
        dataset,
    )

    np.testing.assert_allclose(runtime[[0, 1, 5]], [100.0, -100.0, 100.0])


def test_converts_configured_camera_subset(monkeypatch) -> None:
    """A custom visual config can expose any available RGB camera subset."""
    _install_fake_so101(monkeypatch)
    adapter = _make_adapter()
    adapter._camera_key_map = {"overhead_camera": "workspace"}

    observation = adapter.to_observation(adapter._normalize_raw_obs(_raw_observation()))

    assert isinstance(observation.images, dict)
    assert set(observation.images) == {"workspace"}
    assert observation.images["workspace"].shape == (1, 3, 8, 8)


def test_constructor_infers_cameras_from_custom_visual_config(monkeypatch) -> None:
    """Custom task configs determine the policy-facing RGB image set."""
    _replace_gymnasium_init(monkeypatch)
    monkeypatch.setattr(adapter_module, "_get_so101_nexus", lambda: SimpleNamespace())
    config = SimpleNamespace(
        obs_mode="visual",
        observations=[
            SimpleNamespace(name="joint_positions"),
            SimpleNamespace(name="overhead_camera", modalities=("rgb",)),
        ],
    )

    adapter = SO101NexusGym(gym_id="MuJoCoTouch-v1", config=config)

    assert adapter._camera_key_map == {"overhead_camera": "overhead"}


def test_constructor_forwards_explicit_render_mode(monkeypatch) -> None:
    """An explicit render mode must not collide with the adapter default."""
    captured_kwargs = _replace_gymnasium_init(monkeypatch)
    monkeypatch.setattr(adapter_module, "_get_so101_nexus", lambda: SimpleNamespace())
    config = SimpleNamespace(
        obs_mode="visual",
        observations=[SimpleNamespace(name="wrist_camera", modalities=("rgb",))],
    )

    SO101NexusGym(config=config, render_mode=None)

    assert captured_kwargs["render_mode"] is None


def test_vectorization_is_rejected() -> None:
    """The adapter must not inherit unsupported vectorized construction."""
    with pytest.raises(NotImplementedError, match="does not support vectorized"):
        SO101NexusGym.vectorize("MuJoCoPickLift-v1", num_envs=2)


def test_missing_dependency_has_install_guidance(monkeypatch) -> None:
    """The optional dependency failure explains how to install it."""
    monkeypatch.setattr(adapter_module, "_SO101_NEXUS_AVAILABLE", False)
    monkeypatch.setattr(adapter_module, "_SO101_NEXUS_IMPORT_ERROR", "not installed")

    with pytest.raises(ImportError, match="uv sync --extra so101-nexus"):
        adapter_module._check_so101_nexus_available()


def test_data_first_import_does_not_cycle() -> None:
    """Importing gyms must not load the optional SO101 hardware driver."""
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import builtins; original = builtins.__import__; "
            "builtins.__import__ = lambda name, *args, **kwargs: "
            "(_ for _ in ()).throw(ModuleNotFoundError(name)) "
            "if name == 'scservo_sdk' else original(name, *args, **kwargs); "
            "from physicalai.data import Feature; from physicalai.gyms import SO101NexusGym",
        ],
        check=True,
    )
