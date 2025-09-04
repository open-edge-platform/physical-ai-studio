# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit Tests - Base Gym Testing"""

import pytest
import gymnasium as gym


class BaseTestGym:
    """
    A base class for testing Gym environment wrappers.
    """
    env = None # The gym environment instance

    @pytest.fixture(autouse=True)
    def setup_and_teardown_env(self):
        """This fixture is automatically run for every test method."""
        self.setup_env()
        yield
        if self.env:
            self.env.close()
            self.env = None

    def setup_env(self):
        """
        Placeholder for environment setup.
        Subclasses MUST override this method to instantiate `self.env`.
        """
        raise NotImplementedError("Subclasses must implement setup_env")

    def test_env_creation(self):
        """Tests if the environment is created successfully."""
        assert self.env is not None
        assert hasattr(self.env, 'env') # Check for the wrapped env
        assert isinstance(self.env.env, gym.Env)

    def test_observation_and_action_spaces(self):
        """Tests if observation and action spaces are valid gym spaces."""
        assert isinstance(self.env.observation_space, gym.Space)
        assert isinstance(self.env.action_space, gym.Space)

    def test_reset_api(self):
        """Tests the `reset` method's return signature and types."""
        obs, info = self.env.reset()

        assert self.env.observation_space.contains(obs)
        assert isinstance(info, dict)

    def test_step_api(self):
        """Tests the `step` method's return signature and types."""
        self.env.reset()
        # Take a random action from the action space
        action = self.env.action_space.sample()

        result = self.env.step(action)

        # Check that step returns the correct 5-tuple
        assert isinstance(result, tuple) and len(result) == 5, "step() must return a 5-tuple"

        obs, reward, terminated, truncated, info = result

        # Check types
        assert self.env.observation_space.contains(obs)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
