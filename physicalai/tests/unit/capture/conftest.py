# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pytest configuration for capture tests."""

from __future__ import annotations

import importlib.util
import sys
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

    from physicalai.capture.transport._spec import CameraSpec

HAS_ICEORYX2 = importlib.util.find_spec("iceoryx2") is not None


@pytest.fixture
def fake_camera_spec() -> CameraSpec:
    from physicalai.capture.transport._spec import CameraSpec  # noqa: PLC0415

    return CameraSpec(camera_type="fake", camera_kwargs={})


@pytest.fixture
def publisher_service(fake_camera_spec: CameraSpec) -> Generator[str, None, None]:
    from uuid import uuid4  # noqa: PLC0415

    from physicalai.capture.transport._publisher import CameraPublisher  # noqa: PLC0415

    service_name = f"physicalai/test/{uuid4().hex[:8]}/frame"
    publisher = CameraPublisher(
        fake_camera_spec,
        service_name,
        _factory_override="tests.unit.capture.fake:FakeCamera",
    )
    publisher.start(timeout=10.0)
    try:
        yield service_name
    finally:
        publisher.stop()
