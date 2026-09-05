# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Schemas for the backend health check endpoint."""

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Current health of the backend process.

    ``instance_id`` identifies the running process and changes after a restart;
    ``restart_required`` signals that installed plugin changes need a process
    replacement before they become active.
    """

    status: Literal["healthy"] = Field(description="Always ``healthy`` while the process is alive")
    instance_id: str = Field(description="Unique identifier of the current backend process instance")
    restart_required: bool = Field(
        description="True when installed plugin changes require a server restart to take effect"
    )