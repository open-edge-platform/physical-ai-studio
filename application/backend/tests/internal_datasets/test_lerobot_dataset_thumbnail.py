import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import cv2
import numpy as np

from internal_datasets.lerobot.lerobot_dataset import InternalLeRobotDataset


def test_thumbnail_decodes_only_requested_video_and_caches_by_file_version(tmp_path: Path) -> None:
    video_path = tmp_path / "chunk.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 2, (640, 480))
    assert writer.isOpened()
    for color in (0, 255):
        writer.write(np.full((480, 640, 3), color, dtype=np.uint8))
    writer.release()

    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/info.json").touch()
    episode = {"episode_index": 1, "videos/observation.images.main/from_timestamp": 0.5}
    meta = SimpleNamespace(root=tmp_path, get_video_file_path=lambda *_: video_path.name)
    dataset = InternalLeRobotDataset.__new__(InternalLeRobotDataset)
    dataset.path = tmp_path
    dataset._dataset = SimpleNamespace(meta=meta)
    dataset._find_episode_metadata = MagicMock(return_value=episode)
    dataset._build_thumbnail_png_bytes.cache_clear()

    first = dataset.get_episode_thumbnail_png(1, "observation.images.main", 64, 48)
    second = dataset.get_episode_thumbnail_png(1, "observation.images.main", 64, 48)

    assert first is not None and second == first
    decoded = cv2.imdecode(np.frombuffer(first[0], dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (48, 64, 3)
    assert decoded.mean() > 230  # The second frame is at the episode's video offset.
    assert dataset._build_thumbnail_png_bytes.cache_info().hits == 1
    stat = video_path.stat()
    os.utime(video_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
    assert dataset.get_episode_thumbnail_png(1, "observation.images.main", 64, 48) == first
    assert dataset._build_thumbnail_png_bytes.cache_info().misses == 2
    dataset._build_thumbnail_png_bytes.cache_clear()
