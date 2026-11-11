# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from exceptions import AwsFeatureDisabledError
from settings import get_settings

from .ssh_network_exposure import (
    SshFeatureAvailability,
    evaluate_ssh_feature_availability,
    get_ssh_feature_availability,
    is_loopback_host,
)

__all__ = [
    "SshFeatureAvailability",
    "evaluate_ssh_feature_availability",
    "get_ssh_feature_availability",
    "is_loopback_host",
    "require_aws_feature_active",
]


def require_aws_feature_active() -> None:
    """Require the AWS integration to be explicitly enabled."""
    if not get_settings().aws_enabled:
        raise AwsFeatureDisabledError
