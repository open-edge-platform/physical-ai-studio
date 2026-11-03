import base64
import importlib
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import torch
from lerobot.datasets.video_utils import decode_video_frames_torchcodec

from internal_datasets.lerobot.lerobot_dataset import InternalLeRobotDataset


@pytest.fixture(autouse=True)
def thumbnail_cache():
    InternalLeRobotDataset._cached_thumbnail_png_bytes.cache_clear()
    with patch.object(cv2, "VideoCapture", side_effect=AssertionError("Do not decode thumbnails with OpenCV")):
        yield
    InternalLeRobotDataset._cached_thumbnail_png_bytes.cache_clear()


@pytest.fixture(scope="session")
def native_video_decoder():
    try:
        importlib.import_module("torchcodec.decoders")
    except (ImportError, RuntimeError, OSError) as exc:
        pytest.skip(f"Native TorchCodec decoding unavailable: {str(exc).splitlines()[0]}")


def _write_video(video_path: Path, colors: list[tuple[int, int, int]]) -> None:
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter.fourcc(*"mp4v"), 2, (640, 480))
    assert writer.isOpened()
    for color in colors:
        writer.write(np.full((480, 640, 3), color, dtype=np.uint8))
    writer.release()


def _make_dataset(tmp_path: Path, video_path: Path) -> InternalLeRobotDataset:
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/info.json").touch()
    episode = {"episode_index": 1, "videos/observation.images.main/from_timestamp": 0.5}
    meta = SimpleNamespace(root=tmp_path, get_video_file_path=MagicMock(return_value=video_path.name))
    dataset = InternalLeRobotDataset.__new__(InternalLeRobotDataset)
    dataset.path = tmp_path
    dataset._dataset = SimpleNamespace(meta=meta, tolerance_s=1e-4)
    dataset._find_episode_metadata = MagicMock(return_value=episode)
    return dataset


@pytest.mark.parametrize("codec", ["mpeg4", "av1"])
@pytest.mark.usefixtures("native_video_decoder")
def test_thumbnail_decodes_only_requested_video_and_caches_by_file_version(tmp_path: Path, codec: str) -> None:
    video_path = tmp_path / "chunk.mp4"
    if codec == "av1":
        # libaom-av1, 32x24, 2 fps: red at 0s, blue at 0.5s. No encoder needed at test time.
        fixture = Path(__file__).parent / "fixtures/thumbnail_av1.mp4.base64"
        video_path.write_bytes(base64.b64decode(fixture.read_bytes()))
    else:
        # OpenCV's writer takes BGR, unlike LeRobot's decoded RGB tensors.
        _write_video(video_path, [(0, 0, 255), (255, 0, 0)])
    dataset = _make_dataset(tmp_path, video_path)

    with patch(
        "internal_datasets.lerobot.lerobot_dataset.decode_video_frames_torchcodec",
        wraps=decode_video_frames_torchcodec,
    ) as decode:
        first = dataset.get_episode_thumbnail_png(1, "observation.images.main", 64, 48)
    assert decode.call_count == 1
    assert decode.call_args.args == (str(video_path), [0.5], 1e-4)
    assert decode.call_args.kwargs["return_uint8"] is True
    second = dataset.get_episode_thumbnail_png(1, "observation.images.main", 64, 48)

    assert first is not None and second == first
    decoded = cv2.imdecode(np.frombuffer(first[0], dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape == (48, 64, 3)
    assert decoded[:, :, 0].mean() > 230  # Blue frame at the episode's video offset, in BGR.
    assert decoded[:, :, 1:].mean() < 20
    assert dataset._cached_thumbnail_png_bytes.cache_info().hits == 1
    stat = video_path.stat()
    os.utime(video_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
    assert dataset.get_episode_thumbnail_png(1, "observation.images.main", 64, 48) == first
    assert dataset._cached_thumbnail_png_bytes.cache_info().misses == 2
    assert dataset.get_episode_thumbnail_png(1, "observation.images.main", 321, 241) is not None
    assert dataset._cached_thumbnail_png_bytes.cache_info().misses == 2


def test_thumbnail_caches_successful_decode_by_file_version(tmp_path: Path) -> None:
    video_path = tmp_path / "chunk.mp4"
    video_path.touch()
    dataset = _make_dataset(tmp_path, video_path)
    frames = torch.zeros((1, 3, 8, 8), dtype=torch.uint8)
    with patch(
        "internal_datasets.lerobot.lerobot_dataset.decode_video_frames_torchcodec", return_value=frames
    ) as decode:
        first = dataset.get_episode_thumbnail_png(1, "observation.images.main", 64, 48)
        assert first is not None
        assert dataset.get_episode_thumbnail_png(1, "observation.images.main", 64, 48) == first
        assert decode.call_count == 1

        stat = video_path.stat()
        os.utime(video_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
        assert dataset.get_episode_thumbnail_png(1, "observation.images.main", 64, 48) == first
        assert decode.call_count == 2

        for _ in range(2):
            assert dataset.get_episode_thumbnail_png(1, "observation.images.main", 321, 241) is not None
        assert decode.call_count == 4
    assert dataset._cached_thumbnail_png_bytes.cache_info().misses == 2


@pytest.mark.usefixtures("native_video_decoder")
def test_thumbnail_reloads_replaced_video(tmp_path: Path) -> None:
    video_path = tmp_path / "chunk.mp4"
    _write_video(video_path, [(0, 0, 0), (255, 0, 0)])
    dataset = _make_dataset(tmp_path, video_path)
    first = dataset.get_episode_thumbnail_png(1, "observation.images.main", 64, 48)
    assert first is not None

    replacement = tmp_path / "replacement.mp4"
    _write_video(replacement, [(0, 0, 0), (0, 255, 0)])
    stat = video_path.stat()
    replacement.replace(video_path)
    os.utime(video_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
    second = dataset.get_episode_thumbnail_png(1, "observation.images.main", 64, 48)

    assert second is not None and second != first
    decoded = cv2.imdecode(np.frombuffer(second[0], dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded[:, :, 1].mean() > 230
    assert decoded[:, :, (0, 2)].mean() < 20
    assert dataset._cached_thumbnail_png_bytes.cache_info().misses == 2


def test_thumbnail_retries_transient_decode_failure_and_closes_decoder(tmp_path: Path) -> None:
    video_path = tmp_path / "chunk.mp4"
    video_path.touch()
    dataset = _make_dataset(tmp_path, video_path)
    frames = torch.zeros((1, 3, 8, 8), dtype=torch.uint8)
    with (
        patch("internal_datasets.lerobot.lerobot_dataset.VideoDecoderCache") as cache,
        patch(
            "internal_datasets.lerobot.lerobot_dataset.decode_video_frames_torchcodec",
            side_effect=[RuntimeError("Temporary decoding failure"), frames],
        ),
    ):
        assert dataset.get_episode_thumbnail_png(1, "observation.images.main", 64, 48) is None
        cache.return_value.clear.assert_called_once()
        assert dataset._cached_thumbnail_png_bytes.cache_info().currsize == 0
        assert dataset.get_episode_thumbnail_png(1, "observation.images.main", 64, 48) is not None
        assert cache.return_value.clear.call_count == 2


@pytest.mark.usefixtures("native_video_decoder")
def test_thumbnail_handles_corrupt_video(tmp_path: Path) -> None:
    video_path = tmp_path / "chunk.mp4"
    video_path.write_bytes(b"not a video")
    dataset = _make_dataset(tmp_path, video_path)
    assert dataset.get_episode_thumbnail_png(1, "observation.images.main", 64, 48) is None


def test_thumbnail_handles_missing_video(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path, tmp_path / "missing.mp4")
    assert dataset.get_episode_thumbnail_png(1, "observation.images.main", 64, 48) is None
