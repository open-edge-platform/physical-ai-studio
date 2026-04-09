# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the factory and discovery functions."""

import pytest

from physicalai.capture.discovery import discover_all
from physicalai.capture.errors import MissingDependencyError
from physicalai.capture.factory import create_camera


class TestCreateCamera:
    """create_camera() driver dispatch."""

    def test_unknown_driver_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown camera type"):
            create_camera("nonexistent")

    def test_case_insensitive(self) -> None:
        # Ensure camera type dispatch is case-insensitive.
        from physicalai.capture.cameras.uvc import UVCCamera

        cam = create_camera("UVC", backend="v4l2")
        assert isinstance(cam, UVCCamera)


class TestDiscoverAll:
    """discover_all() aggregation."""

    def test_returns_dict(self) -> None:
        # Without any camera SDKs installed, backends will fail to
        # import and be silently skipped.
        result = discover_all()
        assert isinstance(result, dict)

    def test_missing_backends_skipped(self) -> None:
        # Should not raise even when no backends are installed.
        result = discover_all()
        # Result may be empty or partial; the key invariant is no exception.
        assert isinstance(result, dict)
