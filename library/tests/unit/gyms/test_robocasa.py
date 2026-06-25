# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit Tests - RoboCasa Kitchen Gym.

These run without spinning up MuJoCo. The ``RoboCasaGym`` constructor
only calls ``_setup_spaces`` and defers env creation to ``_ensure_env``,
so observation/action spaces are introspectable without any simulator.
"""

from __future__ import annotations

import numpy as np
import pytest

# Skip the whole module if the dedicated venv has not been provisioned
# (see library/scripts/benchmark/install_robocasa.sh).
pytest.importorskip("robocasa")
pytest.importorskip("robosuite")


from physicalai.gyms.robocasa import (  # noqa: E402
    ACTION_DIM,
    DEFAULT_CAMERAS,
    OBS_STATE_DIM,
    RoboCasaGym,
    _resolve_tasks,
    convert_action,
    create_robocasa_gyms,
)


class TestResolveTasks:
    """Test task-group resolution (``_resolve_tasks``)."""

    def test_resolves_task_group_to_split(self):
        """``atomic_seen`` returns the v1.0 atomic task list under split=target."""
        names, split = _resolve_tasks("atomic_seen")
        assert split == "target"
        # Spot-check membership of known-stable v1.0 names rather than the
        # exact count so an upstream task-mix update doesn't immediately
        # break the test.
        assert "CloseFridge" in names
        assert "OpenCabinet" in names
        assert "OpenDrawer" in names

    def test_resolves_single_task_keeps_split_none(self):
        """A bare task name resolves to itself with no split override."""
        names, split = _resolve_tasks("CloseFridge")
        assert names == ["CloseFridge"]
        assert split is None

    def test_resolves_comma_separated_tasks(self):
        """Comma-separated lists are split and stripped."""
        names, split = _resolve_tasks("CloseFridge, OpenDrawer ,TurnOnStove")
        assert names == ["CloseFridge", "OpenDrawer", "TurnOnStove"]
        assert split is None

    def test_rejects_unknown_task_group(self):
        """An unknown name is treated as a single task, not an error."""
        # The regex/split path should accept any non-empty string; only
        # robocasa will later complain that the env doesn't exist.
        names, split = _resolve_tasks("definitely_not_a_group")
        assert names == ["definitely_not_a_group"]
        assert split is None

    def test_rejects_empty_task_string(self):
        """Empty string raises ValueError."""
        with pytest.raises(ValueError, match="at least one"):
            _resolve_tasks("")


class TestConvertAction:
    """Test the flat-vector to action-dict converter."""

    def test_convert_action_shapes(self):
        """``convert_action`` returns the five RoboCasa action keys with the right slice widths."""
        action_dict = convert_action(np.zeros(ACTION_DIM, dtype=np.float32))

        expected_widths = {
            "action.base_motion": 4,
            "action.control_mode": 1,
            "action.end_effector_position": 3,
            "action.end_effector_rotation": 3,
            "action.gripper_close": 1,
        }
        assert set(action_dict.keys()) == set(expected_widths.keys())
        for key, width in expected_widths.items():
            assert action_dict[key].shape == (width,), f"{key} has wrong shape"

    def test_convert_action_preserves_values(self):
        """Slice values match the source array element-wise."""
        flat = np.arange(ACTION_DIM, dtype=np.float32)
        action_dict = convert_action(flat)
        np.testing.assert_array_equal(action_dict["action.base_motion"], flat[0:4])
        np.testing.assert_array_equal(action_dict["action.control_mode"], flat[4:5])
        np.testing.assert_array_equal(action_dict["action.end_effector_position"], flat[5:8])
        np.testing.assert_array_equal(action_dict["action.end_effector_rotation"], flat[8:11])
        np.testing.assert_array_equal(action_dict["action.gripper_close"], flat[11:12])


class TestObservationSpace:
    """Test ``_setup_spaces`` runs without MuJoCo."""

    def test_observation_space_pixels_agent_pos(self):
        """``pixels_agent_pos`` exposes images and a 16-D agent_pos Box."""
        gym = RoboCasaGym(
            task="CloseFridge",
            obs_type="pixels_agent_pos",
            observation_height=128,
            observation_width=128,
        )
        try:
            assert gym._env is None  # lazy — no MuJoCo here
            assert OBS_STATE_DIM == 16
            obs_space = gym.observation_space
            assert "pixels" in obs_space.spaces
            assert "agent_pos" in obs_space.spaces
            assert obs_space.spaces["agent_pos"].shape == (OBS_STATE_DIM,)
            for cam in DEFAULT_CAMERAS:
                assert cam in obs_space.spaces["pixels"].spaces
                assert obs_space.spaces["pixels"].spaces[cam].shape == (128, 128, 3)
        finally:
            gym.close()

    def test_observation_space_pixels_only(self):
        """``pixels`` omits agent_pos."""
        gym = RoboCasaGym(
            task="CloseFridge",
            obs_type="pixels",
            observation_height=128,
            observation_width=128,
        )
        try:
            assert gym._env is None
            obs_space = gym.observation_space
            assert "pixels" in obs_space.spaces
            assert "agent_pos" not in obs_space.spaces
        finally:
            gym.close()

    def test_invalid_obs_type_raises(self):
        """Unknown ``obs_type`` is rejected."""
        with pytest.raises(ValueError, match="Unsupported obs_type"):
            RoboCasaGym(task="CloseFridge", obs_type="not_a_type")

    def test_action_space_shape(self):
        """Action space is a 12-D Box in [-1, 1]."""
        gym = RoboCasaGym(task="CloseFridge")
        try:
            assert gym.action_space.shape == (ACTION_DIM,)
            assert float(gym.action_space.low.min()) == -1.0
            assert float(gym.action_space.high.max()) == 1.0
        finally:
            gym.close()


class TestCreateRobocasaGyms:
    """Test the multi-task factory (``create_robocasa_gyms``)."""

    def test_create_from_atomic_seen_group(self):
        """``atomic_seen`` produces lazy gyms for every v1.0 atomic task."""
        gyms = create_robocasa_gyms("atomic_seen", observation_height=128, observation_width=128)
        try:
            assert len(gyms) > 0
            assert all(g._env is None for g in gyms), "constructor must not eagerly build MuJoCo envs"
            # Spot-check membership rather than exact count so an upstream
            # task-mix update doesn't immediately break the test.
            task_names = {g.task for g in gyms}
            assert "CloseFridge" in task_names
            assert "OpenCabinet" in task_names
        finally:
            for g in gyms:
                g.close()

    def test_create_from_explicit_list(self):
        """An explicit list of task names is honored verbatim."""
        gyms = create_robocasa_gyms(
            ["CloseFridge", "OpenDrawer"],
            observation_height=128,
            observation_width=128,
        )
        try:
            assert len(gyms) == 2
            assert [g.task for g in gyms] == ["CloseFridge", "OpenDrawer"]
        finally:
            for g in gyms:
                g.close()

    def test_create_from_empty_list_raises(self):
        """An empty list raises before instantiating any gym."""
        with pytest.raises(ValueError, match="at least one"):
            create_robocasa_gyms([])


class TestStepValidation:
    """Validate ``RoboCasaGym.step`` action-shape checks."""

    def test_step_rejects_wrong_length_1d_action(self):
        """A 1-D action with incorrect length raises ``ValueError``."""

        class _SentinelEnv:
            def step(self, _action):
                msg = "step() should not be called for invalid action shapes"
                raise AssertionError(msg)

            def close(self):
                return None

        gym = RoboCasaGym(task="CloseFridge")
        try:
            # Avoid constructing MuJoCo; shape validation should fail first.
            gym._env = _SentinelEnv()
            with pytest.raises(ValueError, match=r"Expected 1-D action shape"):
                gym.step(np.zeros(ACTION_DIM - 1, dtype=np.float32))
        finally:
            gym.close()
