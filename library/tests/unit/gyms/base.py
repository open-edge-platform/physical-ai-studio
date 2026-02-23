# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit Tests - Base Gym Testing"""

import pytest
from physicalai.data import Observation


class BaseTestGym:
    """
    A base class for testing Gym environment wrappers.
    """

    env = None  # The gym environment instance

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
        assert hasattr(self.env, "_env")  # Check for the wrapped env

    def test_reset_api(self):
        """Tests the `reset` method's return signature and types."""
        obs, info = self.env.reset()

        assert isinstance(obs, Observation)
        assert isinstance(info, dict)

    def test_step_api(self):
        """Tests the `step` method's return signature and types."""
        self.env.reset()
        # Take a random action from the action space
        action = self.env.sample_action()

        result = self.env.step(action)

        # Check that step returns the correct 5-tuple
        assert isinstance(result, tuple) and len(result) == 5, (
            "step() must return a 5-tuple"
        )

        obs, reward, terminated, truncated, info = result

        def is_single_or_batch(x, typ):
            return isinstance(x, typ) or (
                isinstance(x, (list, tuple)) and all(isinstance(v, typ) for v in x)
            )

        # Check types
        assert isinstance(obs, Observation)
        assert is_single_or_batch(reward, float)
        assert is_single_or_batch(terminated, bool)
        assert is_single_or_batch(truncated, bool)
        assert is_single_or_batch(info, dict)
