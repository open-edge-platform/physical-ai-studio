# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Schemas for backend-discovered recording presets."""

from pydantic import BaseModel


class VideoRecordingPreset(BaseModel):
    """One recording configuration the current backend has verified."""

    id: str
    name: str
    description: str


class VideoRecordingPresets(BaseModel):
    """The recording presets available on the current backend."""

    presets: list[VideoRecordingPreset]
