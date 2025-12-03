# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Dummy policy config."""

from dataclasses import dataclass

from getiaction.config import Config


@dataclass(frozen=True)
class DummyConfig(Config):
    """Dummy policy config."""

    action_shape: list | tuple
