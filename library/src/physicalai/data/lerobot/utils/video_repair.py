# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Video repair utilities for LeRobotDataset.

This module provides functionality to detect and repair corrupt video frames in
LeRobot datasets. Corrupt frames are replaced with the previous valid frame during
repair.
"""

import contextlib
from pathlib import Path
from shutil import copyfile, rmtree
from typing import TYPE_CHECKING

from lightning_utilities import module_available
from tqdm import tqdm

if TYPE_CHECKING or module_available("lerobot"):
    from lerobot.datasets.image_writer import write_image
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.video_utils import decode_video_frames, encode_video_frames
else:
    write_image = None
    LeRobotDataset = None
    decode_video_frames = None
    encode_video_frames = None

if TYPE_CHECKING or module_available("torchcodec"):
    from torchcodec.decoders import VideoDecoder
else:
    VideoDecoder = None


def repair_corrupt_video(video_path: Path | str) -> None:
    """Repair a corrupt video by recoding it and duplicating corrupt frames.

    Decodes all frames from the video. If a frame cannot be decoded, the previous
    frame is used instead. The original video is backed up with a `.bak` extension.

    Args:
        video_path: Path to the video file to repair.

    Raises:
        ImportError: If required dependencies (torchcodec or lerobot) are not available.
        RuntimeError: If backup video cannot be created.
        ValueError: If video file does not exist.

    """
    video_path = Path(video_path)
    if not video_path.exists():
        msg = f"Video file does not exist: {video_path}"
        raise ValueError(msg)

    if VideoDecoder is None:
        msg = "torchcodec is required for video repair but is not available"
        raise ImportError(msg)

    if write_image is None or encode_video_frames is None:
        msg = "lerobot is required for video repair but is not available"
        raise ImportError(msg)

    backup_path = video_path.with_suffix(video_path.suffix + ".bak")
    copyfile(video_path, backup_path)
    if not backup_path.exists():
        msg = f"Failed to create backup video from {video_path}"
        raise RuntimeError(msg)

    decoder = VideoDecoder(video_path, device="cpu", seek_mode="approximate")
    output_folder = video_path.with_suffix("").with_name(
        video_path.stem + "_tmp",
    )
    output_folder.mkdir(parents=True, exist_ok=True)

    frame = decoder[0]
    for frame_index in range(len(decoder)):
        output_frame = output_folder / f"frame_{frame_index:06d}.png"
        with contextlib.suppress(Exception):
            frame = decoder[frame_index]
        write_image(frame.cpu().numpy(), output_frame)
    average_fps = decoder.metadata.average_fps

    encode_video_frames(output_folder, video_path, int(average_fps), overwrite=True)
    rmtree(output_folder)


def find_videos_with_corrupt_frame(dataset: LeRobotDataset, frame_index: int) -> list[Path]:
    """Find video files that contain corrupt frames at the given frame index.

    Args:
        dataset: LeRobotDataset object.
        frame_index: Index of the frame to check.

    Returns:
        List of paths to video files that contain corrupt frames at this index.

    Raises:
        ImportError: If lerobot is not available.
        RuntimeError: If there is no video data in the dataset.

    """
    if decode_video_frames is None:
        msg = "lerobot is required but is not available"
        raise ImportError(msg)

    item = dataset.hf_dataset[frame_index]
    episode_index = item["episode_index"].item()
    corrupt_videos = []

    if len(dataset.meta.video_keys) == 0:
        msg = "No video data present in dataset"
        raise RuntimeError(msg)

    current_timestamp = item["timestamp"].item()
    query_timestamps = dataset._get_query_timestamps(current_timestamp, None)  # noqa: SLF001

    for video_key, query_timestamp in query_timestamps.items():
        video_path = dataset.root / dataset.meta.get_video_file_path(episode_index, video_key)
        try:
            decode_video_frames(video_path, query_timestamp, dataset.tolerance_s, dataset.video_backend)
        except Exception:  # noqa: BLE001
            corrupt_videos.append(video_path)

    return corrupt_videos


def find_corrupt_videos_in_dataset(dataset: LeRobotDataset) -> list[Path]:
    """Scan all frames in the dataset to find videos with corrupt frames.

    Args:
        dataset: LeRobotDataset object.

    Returns:
        List of unique paths to video files that contain corrupt frames.

    """
    all_corrupt_videos = []
    for frame_index in tqdm(range(len(dataset))):
        corrupt_videos = find_videos_with_corrupt_frame(dataset, frame_index)
        all_corrupt_videos.extend(corrupt_videos)
    return list(set(all_corrupt_videos))


def repair_corrupt_videos_in_dataset(dataset: LeRobotDataset) -> None:
    """Automatically find and repair all corrupt videos in a LeRobot dataset.

    Args:
        dataset: LeRobotDataset object.

    """
    corrupt_videos = find_corrupt_videos_in_dataset(dataset)
    for corrupt_video in corrupt_videos:
        repair_corrupt_video(corrupt_video)
