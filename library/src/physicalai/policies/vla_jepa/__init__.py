# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""First-party VLA-JEPA policy."""

from .config import VLAJEPAConfig
from .model import VLAJEPAModel
from .policy import VLAJEPA

__all__ = ["VLAJEPA", "VLAJEPAConfig", "VLAJEPAModel"]
