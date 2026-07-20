# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Storage capacity reporting for the trainer service.

Reports free/total space on the volume backing the trainer's storage
directory so the studio backend can surface how much room a remote trainer
has for datasets and model artifacts.
"""

from __future__ import annotations

import shutil

from trainer.schemas import StorageInfo
from trainer.settings import get_settings


def get_storage_info() -> StorageInfo:
    """Report total and free space on the trainer's storage volume."""
    settings = get_settings()
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(settings.storage_dir)
    return StorageInfo(total_bytes=usage.total, free_bytes=usage.free)
