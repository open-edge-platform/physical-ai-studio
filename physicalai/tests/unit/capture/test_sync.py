# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for multi-camera synchronisation."""

from __future__ import annotations

import asyncio

from physicalai.capture.multi import SyncedFrames, read_cameras
from tests.unit.capture.fake import FakeCamera


class TestReadCameras:
    """read_cameras() with multiple FakeCamera instances."""

    def test_returns_synced_frames(self) -> None:
        cameras = {
            "left": FakeCamera(device_name="left"),
            "right": FakeCamera(device_name="right"),
        }
        for cam in cameras.values():
            cam.connect()
        try:
            synced = read_cameras(cameras)
            assert isinstance(synced, SyncedFrames)
            assert "left" in synced.frames
            assert "right" in synced.frames
        finally:
            for cam in cameras.values():
                cam.disconnect()

    def test_skew_is_small(self) -> None:
        cameras = {
            "a": FakeCamera(device_name="a"),
            "b": FakeCamera(device_name="b"),
        }
        for cam in cameras.values():
            cam.connect()
        try:
            synced = read_cameras(cameras)
            # FakeCamera reads are near-instant so skew should be tiny
            assert synced.max_skew_ms < 100.0
        finally:
            for cam in cameras.values():
                cam.disconnect()

    def test_single_camera(self) -> None:
        cam = FakeCamera(device_name="solo")
        cam.connect()
        try:
            synced = read_cameras({"solo": cam})
            assert synced.max_skew_ms == 0.0
            assert synced.frames["solo"].sequence == 0
        finally:
            cam.disconnect()


class TestAsyncReadCameras:
    """async_read_cameras() with FakeCamera instances."""

    def test_async_returns_synced_frames(self) -> None:
        from physicalai.capture.multi import async_read_cameras

        async def _run() -> SyncedFrames:
            cameras = {
                "left": FakeCamera(device_name="left"),
                "right": FakeCamera(device_name="right"),
            }
            for cam in cameras.values():
                cam.connect()
            try:
                return await async_read_cameras(cameras)
            finally:
                for cam in cameras.values():
                    cam.disconnect()

        synced = asyncio.run(_run())
        assert "left" in synced.frames
        assert "right" in synced.frames
