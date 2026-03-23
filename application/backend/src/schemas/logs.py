# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class LogSource(BaseModel):
    """Describes an available log source.

    Types:
        - application: The main app log (catch-all for non-worker logs)
        - worker: Per-class worker logs (training, inference, etc.)
        - job: Per-job logs created during training runs
    """

    id: str
    name: str
    type: Literal["application", "worker", "job"]
    created_at: datetime | None = None
