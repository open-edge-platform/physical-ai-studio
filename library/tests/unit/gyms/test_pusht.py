# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit Tests - MVTecLoco Datamodule."""

from getiaction.gyms import PushTGym
from tests.unit.gyms.base import BaseTestGym


class TestPushTGym(BaseTestGym):
    """
    Tests the specific implementation of the PushTGym wrapper.
    """

    def setup_env(self):
        """Sets up the PushTGym environment for testing."""
        self.env = PushTGym()

    def test_pushtgym_default_parameters(self):
        """
        Tests if PushTGym initializes with the correct default parameters.
        """
        # The env is already created by the setup_env fixture
        assert self.env._gym_id == "gym_pusht/PushT-v0"

    def test_pushtgym_custom_parameters(self):
        """
        Tests if PushTGym can be initialized with custom parameters.
        """
        self.env.close()
        self.env = PushTGym(
            obs_type="state",
        )
