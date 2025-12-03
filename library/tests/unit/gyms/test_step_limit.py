# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit Tests - PushT Gym."""

from getiaction.gyms import GymnasiumGym
from getiaction.gyms.step_limit import with_step_limit


class TestStepLimit:
    """Tests for the StepLimit environment wrapper."""

    def test_terminates_after_limit(self):
        """StepLimit must truncate after max_steps."""
        env = with_step_limit(GymnasiumGym("CartPole-v1"), max_steps=3)
        obs, _ = env.reset()
        action = env.sample_action()

        _, _, _, trunc1, _ = env.step(action)  #  step 1
        _, _, _, trunc2, _ = env.step(action)  #  step 2
        _, _, _, trunc3, info3 = env.step(action)  #  step 3

        assert trunc1 is False
        assert trunc2 is False
        assert trunc3 is True
        assert "TimeLimit.truncated" in info3

        env.close()

    def test_step_counter_resets_on_reset(self):
        """Step count resets when env.reset() is called."""
        env = with_step_limit(GymnasiumGym("CartPole-v1"), max_steps=2)
        obs, _ = env.reset()
        action = env.sample_action()

        env.step(action)
        env.reset()
        _, _, _, trunc, _ = env.step(action)

        assert trunc is False  # step counter should be fresh

        env.close()

    def test_vectorized_step_limit(self):
        """StepLimit must batch truncate for vectorized envs."""
        base = GymnasiumGym.vectorize("CartPole-v1", num_envs=3)
        env = with_step_limit(base, max_steps=2)
        obs, _ = env.reset()
        action = env.sample_action()

        env.step(action)  # step 1
        obs, reward, terminated, truncated, info = env.step(action)  # step 2 => truncate

        assert isinstance(truncated, list)
        assert truncated == [True, True, True]
        assert "TimeLimit.truncated" in info

        env.close()

    def test_passthrough_methods(self):
        """Wrapper forwards methods to underlying env."""
        import torch
        base = GymnasiumGym("CartPole-v1")
        env = with_step_limit(base, max_steps=5)
        obs, _ = env.reset()

        assert base.sample_action().shape == env.sample_action().shape

        env.close()
