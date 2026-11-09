# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Small history-wrapper and rollout checks."""

import copy
import pickle
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from jsonargparse import ArgumentParser

from physicalai.data import Observation
from physicalai.eval import rollout
from physicalai.eval.rollout.functional import _collect_frame
from physicalai.gyms import Gym, GymnasiumGym, HistoryGym
from physicalai.gyms.step_limit import StepLimit


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("n_obs_steps", [1, 3, 4])
@pytest.mark.parametrize("stride", [1, 2, 3])
def test_history_values_and_reset(batch_size, n_obs_steps, stride):
    """Selected keys advance in order, snapshots survive backend buffer reuse."""
    frame = torch.zeros(batch_size, 3, 4, 4)
    state = np.zeros((batch_size, 2), dtype=np.float32)
    observation = Observation(images={"image": frame, "image1": frame + 10}, state=state)
    gym = Mock()
    gym.reset.return_value = observation, {"reset": True}
    gym.step.return_value = observation, 1.0, False, False, {"step": True}
    env = HistoryGym(gym, n_obs_steps=n_obs_steps, keys={"images.image", "state"}, stride=stride)

    initial, info = env.reset(seed=7, episode_index=2)
    gym.reset.assert_called_once_with(seed=7, episode_index=2)
    assert info == {"reset": True}
    assert initial.images["image"].shape == (batch_size, n_obs_steps, 3, 4, 4)
    assert initial.images["image"][0, :, 0, 0, 0].tolist() == [0] * n_obs_steps
    assert initial.state.shape == (batch_size, n_obs_steps, 2)
    assert initial.state[0, :, 0].tolist() == [0] * n_obs_steps
    assert initial.images["image1"] is observation.images["image1"]
    assert observation.images["image"].shape == (batch_size, 3, 4, 4)

    for step in range(1, (n_obs_steps - 1) * stride + 4):
        frame.fill_(step)
        state.fill(step)
        current, reward, terminated, truncated, info = env.step(torch.zeros(batch_size, 2))
        expected = [max(0, step - offset * stride) for offset in reversed(range(n_obs_steps))]
        assert current.images["image"][0, :, 0, 0, 0].tolist() == expected
        assert current.state[0, :, 0].tolist() == expected
        assert (reward, terminated, truncated, info) == (1.0, False, False, {"step": True})
    assert torch.count_nonzero(initial.images["image"]) == 0
    assert np.count_nonzero(initial.state) == 0
    frame.fill_(100)
    state.fill(100)
    reset, _ = env.reset()
    assert reset.images["image"][0, :, 0, 0, 0].tolist() == [100] * n_obs_steps
    assert reset.state[0, :, 0].tolist() == [100] * n_obs_steps
    frame.fill_(101)
    state.fill(101)
    current, *_ = env.step(torch.zeros(batch_size, 2))
    assert current.images["image"][0, :, 0, 0, 0].tolist() == [100] * (n_obs_steps - 1) + [101]
    assert current.state[0, :, 0].tolist() == [100] * (n_obs_steps - 1) + [101]


@pytest.mark.parametrize("stride", [0, -1, 1.5, True, False, "2", None])
def test_invalid_stride(stride):
    with pytest.raises(ValueError, match="stride must be a positive integer"):
        HistoryGym(Mock(), n_obs_steps=4, keys={"state"}, stride=stride)


@pytest.mark.parametrize(
    ("n_obs_steps", "keys"),
    [(0, {"state"}), (1.5, {"state"}), (2, set()), (2, {"unknown"}), (2, {"images", "images.top"})],
)
def test_invalid_config(n_obs_steps, keys):
    with pytest.raises(ValueError):
        HistoryGym(Mock(), n_obs_steps=n_obs_steps, keys=keys)


@pytest.mark.parametrize("key", ["images.missing", "state", "task"])
def test_invalid_observation(key):
    gym = Mock()
    gym.reset.return_value = Observation(images={"top": torch.zeros(1, 3, 4, 4)}), {}
    with pytest.raises(ValueError, match=key):
        HistoryGym(gym, n_obs_steps=2, keys={key}).reset()


@pytest.mark.parametrize("outer_limit", [True, False])
def test_nested_parser_and_rollout(outer_limit):
    """CLI construction and real rollout work with either wrapper ordering."""
    parser = ArgumentParser(exit_on_error=False)
    parser.add_argument("--gym", type=Gym)
    config = {
        "class_path": "physicalai.gyms.GymnasiumGym",
        "init_args": {"gym_id": "CartPole-v1", "render_mode": None},
    }
    wrappers = [
        ("physicalai.gyms.HistoryGym", {"n_obs_steps": 2, "keys": ["state"], "stride": 2}),
        ("physicalai.gyms.step_limit.StepLimit", {"max_steps": 3}),
    ]
    for class_path, init_args in wrappers if outer_limit else reversed(wrappers):
        config = {"class_path": class_path, "init_args": {"gym": config, **init_args}}
    env = parser.instantiate(parser.parse_object({"gym": config})).gym
    policy = Mock(spec=["select_action", "reset"])

    def select_action(observation):
        assert observation.state.shape == (1, 2, 4)
        return torch.zeros(1, 1, dtype=torch.int64)

    policy.select_action.side_effect = select_action
    try:
        assert isinstance(env, StepLimit if outer_limit else HistoryGym)
        assert isinstance(env.gym.gym, GymnasiumGym)
        assert (env.gym if outer_limit else env).stride == 2
        assert isinstance(env, Gym)
        result = rollout(env, policy, seed=42, return_observations=True)
        assert result["episode_length"] == 3
        assert result["observation"]["state"].shape == (4, 1, 2, 4)
        policy.reset.assert_called_once()
    finally:
        env.close()


@pytest.mark.parametrize("frame_key", ["image", ["image", "image1"]])
def test_rollout_history_frames(frame_key):
    """A history-aware policy and video recorder see the expected shapes."""
    frame = torch.zeros(1, 3, 4, 4)
    gym = Mock()
    gym.get_max_episode_steps.return_value = 2
    gym.reset.return_value = Observation(images={"image": frame, "image1": frame}), {}

    def step(action):
        assert action.shape == (2,)
        frame.add_(1)
        return Observation(images={"image": frame, "image1": frame}), 1.0, False, False, {}

    gym.step.side_effect = step
    env = HistoryGym(gym, n_obs_steps=2, keys={"images.image", "images.image1"})
    policy = Mock(spec=["select_action", "reset"])

    def select_action(observation):
        assert observation.batch_size == 1
        assert observation.images["image"].shape == (1, 2, 3, 4, 4)
        assert observation.images["image1"].shape == (1, 2, 3, 4, 4)
        return torch.zeros(1, 2)

    policy.select_action.side_effect = select_action
    recorder = Mock(frame_key=frame_key, caption="test")
    result = rollout(env, policy, return_frames=True, frame_key=frame_key, video_recorder=recorder)
    assert result["episode_length"] == 2
    width = 4 if isinstance(frame_key, str) else 8
    assert result["frames"].shape == (2, 4, width, 3)
    np.testing.assert_array_equal(result["frames"][:, 0, 0, 0], [0, 1])
    assert recorder.record_frame.call_count == 2
    np.testing.assert_array_equal(recorder.record_frame.call_args.args[0], result["frames"][-1])


def test_collect_numpy_history_frame():
    images = np.zeros((1, 2, 3, 4, 4), dtype=np.float32)
    images[:, -1] = 1
    frame = _collect_frame(Observation(images=images), "image")
    np.testing.assert_array_equal(frame, np.ones((4, 4, 3)))


def test_backend_conversion_and_metadata_are_delegated():
    gym = Mock(task_description="current task")
    env = HistoryGym(gym, n_obs_steps=2, keys={"state"})
    raw_obs = object()
    assert env.to_observation(raw_obs) is gym.to_observation.return_value
    gym.to_observation.assert_called_once_with(raw_obs)
    env.render()
    gym.render.assert_called_once_with()
    assert env.sample_action() is gym.sample_action.return_value
    assert env.task_description == "current task"
    env.close()
    gym.close.assert_called_once_with()


@pytest.mark.parametrize("batched", [False, True])
def test_step_outputs_are_passed_through(batched):
    """Only the observation changes; all other outputs retain their identity."""
    gym = Mock()
    observation = Observation(state=torch.zeros(1, 2))
    info = [{"success": True}] if batched else {"success": True}
    reward, terminated, truncated = ([1.0], [True], [False]) if batched else (1.0, True, False)
    gym.reset.return_value = observation, info
    gym.step.return_value = observation, reward, terminated, truncated, info
    env = HistoryGym(gym, n_obs_steps=2, keys={"state"})
    _, reset_info = env.reset()
    assert reset_info is info
    result = env.step(torch.zeros(1, 2))
    for actual, expected in zip(result[1:], gym.step.return_value[1:], strict=True):
        assert actual is expected


@pytest.mark.parametrize("reconstruct", ["deepcopy", "pickle"])
@pytest.mark.parametrize("stride", [1, 2])
def test_reconstruction_preserves_history(reconstruct, stride):
    """Copy and unpickle wrappers before gym has been restored without recursion."""
    env = HistoryGym(GymnasiumGym("CartPole-v1", render_mode=None), n_obs_steps=2, keys={"state"}, stride=stride)
    restored = None
    try:
        env.reset(seed=42)
        observation, *_ = env.step(torch.tensor(0))
        if reconstruct == "deepcopy":
            restored = copy.deepcopy(env)
        else:
            # Trusted bytes produced locally from this test's wrapper, not external input.
            restored = pickle.loads(pickle.dumps(env))
        assert restored.gym is not env.gym
        assert restored.keys == env.keys
        assert restored.stride == stride
        torch.testing.assert_close(torch.stack(tuple(restored._history["state"])[::stride], dim=1), observation.state)
        expected, *_ = env.step(torch.tensor(0))
        actual, *_ = restored.step(torch.tensor(0))
        torch.testing.assert_close(actual.state, expected.state)
        with pytest.raises(AttributeError):
            getattr(restored, "missing_backend_attribute")
    finally:
        env.close()
        if restored is not None:
            restored.close()
