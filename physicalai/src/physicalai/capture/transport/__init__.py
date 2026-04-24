# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared-memory camera transport via iceoryx2.

Provides :func:`create_shared_camera` and :class:`SharedCamera`
as the public entry points for multi-process camera sharing.

Requires the ``transport`` extra::

    pip install physicalai[transport]
"""

from __future__ import annotations

from ._shared_camera import SharedCamera

__all__ = ["SharedCamera"]
