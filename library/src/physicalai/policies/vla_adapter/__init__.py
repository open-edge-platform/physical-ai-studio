# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""VLA-Adapter Policy - tiny-scale VLA with a Bridge-Attention action head."""

from physicalai.policies.vla_adapter.config import VLAAdapterConfig
from physicalai.policies.vla_adapter.model import VLAAdapterModel
from physicalai.policies.vla_adapter.policy import VLAAdapter

__all__ = ["VLAAdapter", "VLAAdapterConfig", "VLAAdapterModel"]
