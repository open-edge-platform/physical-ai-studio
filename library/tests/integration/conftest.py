# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

@pytest.fixture
def device(request: pytest.FixtureRequest) -> str:
    """Get device from CLI parameter.

    Returns:
        str: Device to use for training/inference (e.g., 'cpu', 'cuda', 'xpu').
    """
    return request.config.getoption("--device", default="gpu")


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add custom command line options.

    Args:
        parser: Pytest parser for adding command line options.
    """
    parser.addoption(
        "--device",
        action="store",
        default="gpu",
        help="Device to use for training/inference (e.g., 'cpu', 'cuda', 'xpu')",
    )
