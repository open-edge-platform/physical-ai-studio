# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for policy factory functions."""

import pytest

from physicalai.policies import get_policy
from physicalai.policies.lerobot import get_lerobot_policy


class TestGetPolicy:
    """Tests for get_policy factory function."""

    def test_physicalai_source(self):
        """Test creating first-party policies."""
        policy = get_policy("act", source="physicalai")
        assert policy.__class__.__name__ == "ACT"

    def test_lerobot_source(self):
        """Test creating LeRobot policies."""
        pytest.importorskip("lerobot")
        policy = get_policy("act", source="lerobot")
        assert policy is not None

    def test_default_source(self):
        """Test default source is physicalai."""
        policy = get_policy("act")
        assert policy.__class__.__name__ == "ACT"

    def test_case_insensitive_source(self):
        """Test source parameter is case-insensitive."""
        policy = get_policy("act", source="PHYSICALAI")
        assert policy is not None

    def test_unknown_policy_raises_error(self):
        """Test unknown policy name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown physicalai policy"):
            get_policy("nonexistent", source="physicalai")

    def test_unknown_source_raises_error(self):
        """Test unknown source raises ValueError."""
        with pytest.raises(ValueError, match="Unknown source.*Supported sources"):
            get_policy("act", source="invalid")


class TestGetLeRobotPolicy:
    """Tests for get_lerobot_policy factory function."""

    def test_explicit_wrappers(self):
        """Test creating policies with explicit wrappers."""
        pytest.importorskip("lerobot")
        for policy_name in ["act", "diffusion"]:
            policy = get_lerobot_policy(policy_name)
            assert policy is not None

    def test_universal_wrapper(self):
        """Test creating policies via universal wrapper escape hatch (best-effort, warns)."""
        pytest.importorskip("lerobot")
        from physicalai.policies.lerobot import policy as policy_module  # noqa: PLC0415

        policy_module._WARNED_UNSUPPORTED_NAMES.discard("vqbet")
        with pytest.warns(UserWarning, match="not in physicalai's supported set"):
            policy = get_lerobot_policy("vqbet")
        assert policy is not None

    def test_case_insensitive(self):
        """Test policy name is case-insensitive."""
        pytest.importorskip("lerobot")
        policy = get_lerobot_policy("ACT")
        assert policy is not None

    def test_unknown_policy_warns_and_defers_validation(self):
        """Unknown policy names go through the escape hatch with a warning; LeRobot validates on eager construction."""
        pytest.importorskip("lerobot")
        from physicalai.policies.lerobot import policy as policy_module  # noqa: PLC0415

        policy_module._WARNED_UNSUPPORTED_NAMES.discard("nonexistent")
        with pytest.warns(UserWarning, match="not in physicalai's supported set"):
            policy = get_lerobot_policy("nonexistent")
        assert policy is not None
        assert policy.policy_name == "nonexistent"
        assert policy._config is None

    def test_lerobot_not_installed(self, monkeypatch):
        """Test ImportError when lerobot not installed."""
        # Mock LEROBOT_AVAILABLE to False
        import physicalai.policies.lerobot as lerobot_module

        monkeypatch.setattr(lerobot_module, "LEROBOT_AVAILABLE", False)
        with pytest.raises(ImportError, match="LeRobot is not installed"):
            get_lerobot_policy("act")
