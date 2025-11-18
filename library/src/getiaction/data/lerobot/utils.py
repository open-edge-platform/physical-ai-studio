# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Utility methods for checking and automatically fixing videos in a LeRobotDataset."""

from typing import TYPE_CHECKING

from lightning_utilities import module_available

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

import contextlib
from pathlib import Path
from shutil import copyfile, rmtree

from tqdm import tqdm


def recode_torch_codec_video(video_path: Path | str) -> None:
    """Recode a video and duplicate frames when they are corrupt.

    The original video is stored under the .bak.mp4 extension.

    Args:
        video_path (str): Path of the video file.

    Raises:
        RuntimeError: If backup video cannot be created.

    """
    backup_path = Path(str(video_path).replace(".mp4", ".bak.mp4"))
    copyfile(video_path, backup_path)
    if not backup_path.exists():
        msg = f"Failed to create backup video from {video_path}"
        raise RuntimeError(msg)

    # initialize video decoder
    decoder = VideoDecoder(video_path, device="cpu", seek_mode="approximate")
    output_folder = Path(str(Path(video_path).with_suffix("")) + "_tmp")
    Path.mkdir(output_folder, parents=True)

    frame = decoder[0]
    for fr_idx in range(len(decoder)):
        output_frame = output_folder / f"frame_{fr_idx:06d}.png"
        with contextlib.suppress(Exception):
            frame = decoder[fr_idx]
        write_image(frame.cpu().numpy(), output_frame)
    average_fps = decoder.metadata.average_fps

    encode_video_frames(output_folder, video_path, int(average_fps), overwrite=True)
    rmtree(output_folder)


def check_frame(ds: LeRobotDataset, fr_idx: int) -> list[str]:
    """Try to decode a video frame and return the video file if the frame is corrupt.

    Args:
        ds (LeRobotDataset): LeRobotDataset object
        fr_idx (int): index of the frame.

    Returns:
        list[str]: Video files for which the frame is corrupt.

    Raises:
        RuntimeError: If there is no video data in the dataset.

    """
    item = ds.hf_dataset[fr_idx]
    ep_idx = item["episode_index"].item()
    corrupt_videos = []
    if len(ds.meta.video_keys) > 0:
        current_ts = item["timestamp"].item()
        # noinspection PyProtectedMember
        query_timestamps = ds._get_query_timestamps(current_ts, None)  # noqa: SLF001

        for vid_key, query_ts in query_timestamps.items():

            video_path = ds.root / ds.meta.get_video_file_path(ep_idx, vid_key)
            # noinspection PyBroadException
            try:
                decode_video_frames(video_path, query_ts, ds.tolerance_s, ds.video_backend)
            except Exception:  # noqa: BLE001
                corrupt_videos.append(video_path)
    else:
        msg = "No frame data present in dataset"
        raise RuntimeError(msg)

    return corrupt_videos


def check_dataset(ds: LeRobotDataset) -> list[str]:
    """Try to decode all frames of the dataset.

    Args:
        ds (LeRobotDataset): LeRobotDataset object

    Returns:
        list[str]: Video files which contain corrupt frames.

    """
    all_corrupt_videos = []
    for fr_idx in tqdm(range(len(ds))):
        corrupt_videos = check_frame(ds, fr_idx)
        all_corrupt_videos.extend(corrupt_videos)
    return list(set(all_corrupt_videos))


def auto_fix_dataset(ds: LeRobotDataset) -> None:
    """Automatically fix corrupt video frames in a LeRobot dataset.

    Args:
        ds (LeRobotDataset): LeRobotDataset object

    """
    corrupt_videos = check_dataset(ds)
    for corrupt_video in corrupt_videos:
        recode_torch_codec_video(corrupt_video)
