# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for VideoRecorder."""

from pathlib import Path

import numpy as np
import pytest

try:
    import imageio.v3  # noqa: F401
    _IMAGEIO_AVAILABLE = True
except ImportError:
    _IMAGEIO_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _IMAGEIO_AVAILABLE, reason="imageio not available")


@pytest.fixture
def frames():
    return [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(5)]


class TestVideoRecorder:
    def test_init(self, tmp_path: Path):
        from physicalai.eval.video import VideoRecorder
        r = VideoRecorder(output_dir=tmp_path / "new")
        assert (tmp_path / "new").exists() and r.fps == 30 and r.record_mode == "all"
        r.close()

    @pytest.mark.parametrize("mode,save_success,save_failure", [
        ("all", True, True),
        ("failures", False, True),
        ("successes", True, False),
        ("none", False, False),
    ])
    def test_record_modes(self, tmp_path: Path, frames, mode, save_success, save_failure):
        from physicalai.eval.video import VideoRecorder
        with VideoRecorder(output_dir=tmp_path, record_mode=mode) as r:
            r.start_episode("s")
            for f in frames:
                r.record_frame(f)
            assert (r.finish_episode(success=True) is not None) == save_success

            r.start_episode("f")
            for f in frames:
                r.record_frame(f)
            assert (r.finish_episode(success=False) is not None) == save_failure

    def test_chw_and_float_formats(self, tmp_path: Path):
        from physicalai.eval.video import VideoRecorder
        with VideoRecorder(output_dir=tmp_path) as r:
            r.start_episode("chw")
            r.record_frame(np.random.randint(0, 255, (3, 64, 64), dtype=np.uint8))
            assert r.finish_episode(success=True) is not None

            r.start_episode("float")
            r.record_frame(np.random.rand(64, 64, 3).astype(np.float32))
            assert r.finish_episode(success=True) is not None

    def test_filename_suffix(self, tmp_path: Path, frames):
        from physicalai.eval.video import VideoRecorder
        with VideoRecorder(output_dir=tmp_path) as r:
            r.start_episode("t1")
            for f in frames:
                r.record_frame(f)
            result1 = r.finish_episode(success=True)
            assert result1 is not None and "success" in result1.name

            r.start_episode("t2")
            for f in frames:
                r.record_frame(f)
            result2 = r.finish_episode(success=False)
            assert result2 is not None and "failure" in result2.name
