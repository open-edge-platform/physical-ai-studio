# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Video recording utilities for policy evaluation.

This module provides the `VideoRecorder` class for recording episode
videos during policy evaluation. It integrates with:

- `rollout()` function via `video_recorder` parameter
- `Rollout` metric via `video_recorder` attribute
- `Benchmark` class via `video_dir`/`record_mode` args

Example:
    >>> from getiaction.eval.video import VideoRecorder

    >>> recorder = VideoRecorder(
    ...     output_dir="./videos",
    ...     fps=30,
    ...     record_mode="failures",
    ... )

    >>> # Pass to rollout function
    >>> result = rollout(env, policy, video_recorder=recorder)
    >>> recorder.close()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal, Self

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

# Optional imageio import
_IMAGEIO_AVAILABLE = False
_IMAGEIO_ERROR: str | None = None

try:
    import imageio.v3 as iio

    _IMAGEIO_AVAILABLE = True
except ImportError as e:
    _IMAGEIO_ERROR = str(e)
    iio = None  # type: ignore[assignment]


def _check_imageio_available() -> None:
    """Check if imageio is available for video encoding.

    Raises:
        ImportError: If imageio is not installed.
    """
    if not _IMAGEIO_AVAILABLE:
        msg = (
            "imageio is required for video recording. Install with:\n"
            "  uv pip install imageio[ffmpeg]\n"
            f"\nOriginal error: {_IMAGEIO_ERROR}"
        )
        raise ImportError(msg)


RecordMode = Literal["all", "failures", "successes", "none"]


class VideoRecorder:
    """Video recorder with explicit control for rollout recording.

    Use this when you need fine-grained control over recording,
    or when integrating with custom evaluation loops.

    Args:
        output_dir: Directory to save videos.
        fps: Frames per second for output video.
        codec: Video codec (h264, libx264, etc.).
        record_mode: When to save videos.
            - "all": Save all episodes
            - "failures": Only save failed episodes (for debugging)
            - "successes": Only save successful episodes
            - "none": Disable recording

    Attributes:
        output_dir: Directory where videos are saved.
        fps: Video frame rate.
        record_mode: Current recording mode.
        episode_count: Number of episodes completed.

    Examples:
        With rollout function:

            from getiaction.eval.video import VideoRecorder
            from getiaction.eval.rollout import rollout

            recorder = VideoRecorder(output_dir="./videos", record_mode="failures")
            result = rollout(env, policy, video_recorder=recorder)
            recorder.close()

        With Rollout metric:

            from getiaction.eval.rollout import Rollout
            from getiaction.eval.video import VideoRecorder

            metric = Rollout(
                max_steps=300,
                video_recorder=VideoRecorder("./videos", record_mode="all"),
            )
            metric.update(env, policy)
            metric.video_recorder.close()

        Manual control in custom loop:

            recorder = VideoRecorder(
                output_dir="./videos",
                fps=30,
                record_mode="failures",
            )

            for episode in range(10):
                recorder.start_episode(name=f"episode_{episode}")
                obs = gym.reset()
                done = False

                while not done:
                    recorder.record_frame(obs["image"])
                    action = policy.select_action(obs)
                    obs, reward, done, truncated, info = gym.step(action)

                # Finish decides whether to save based on record_mode
                recorder.finish_episode(success=info["success"])

            recorder.close()
    """

    def __init__(
        self,
        output_dir: Path | str,
        fps: int = 30,
        codec: str = "h264",
        record_mode: RecordMode = "all",
    ) -> None:
        """Initialize video recorder.

        Args:
            output_dir: Directory to save videos.
            fps: Frames per second for output video.
            codec: Video codec for encoding.
            record_mode: When to save videos.
        """
        _check_imageio_available()

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.codec = codec
        self.record_mode = record_mode

        self._frames: list[np.ndarray] = []
        self._current_episode_name: str | None = None
        self.episode_count = 0

        logger.debug(
            "VideoRecorder initialized: output_dir=%s, fps=%d, mode=%s",
            self.output_dir,
            fps,
            record_mode,
        )

    def start_episode(self, name: str) -> None:
        """Start recording a new episode.

        Args:
            name: Name for this episode (used in filename).
        """
        if self.record_mode == "none":
            return

        self._frames = []
        self._current_episode_name = name
        logger.debug("Started recording episode: %s", name)

    # Standard channel counts for CHW detection
    _CHW_CHANNEL_COUNTS: ClassVar[set[int]] = {1, 3, 4}
    _NDIM_3D: ClassVar[int] = 3

    def record_frame(self, frame: np.ndarray) -> None:
        """Record a single frame.

        Args:
            frame: Image array (H, W, C) in RGB format, uint8.
                Can also be (C, H, W) format which will be transposed.
        """
        import numpy as np  # noqa: PLC0415

        if self.record_mode == "none" or self._current_episode_name is None:
            return

        # Handle CHW -> HWC conversion
        if frame.ndim == self._NDIM_3D and frame.shape[0] in self._CHW_CHANNEL_COUNTS:
            frame = np.transpose(frame, (1, 2, 0))

        # Ensure uint8
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)

        self._frames.append(frame.copy())

    def finish_episode(self, *, success: bool) -> Path | None:
        """Finish recording and save based on record_mode.

        Args:
            success: Whether the episode was successful.

        Returns:
            Path to saved video, or None if not saved.
        """
        if self.record_mode == "none" or not self._frames:
            self._frames = []
            self._current_episode_name = None
            return None

        # Decide whether to save based on record_mode
        should_save = (
            self.record_mode == "all"
            or (self.record_mode == "failures" and not success)
            or (self.record_mode == "successes" and success)
        )

        if not should_save:
            logger.debug(
                "Skipping save for episode %s (mode=%s, success=%s)",
                self._current_episode_name,
                self.record_mode,
                success,
            )
            self._frames = []
            self._current_episode_name = None
            return None

        # Add success/failure suffix
        suffix = "success" if success else "failure"
        video_path = self.output_dir / f"{self._current_episode_name}_{suffix}.mp4"

        # Write video
        self._encode_video(self._frames, video_path)

        logger.info(
            "Saved video: %s (%d frames)",
            video_path,
            len(self._frames),
        )

        self._frames = []
        self._current_episode_name = None
        self.episode_count += 1

        return video_path

    def _encode_video(self, frames: list[np.ndarray], path: Path) -> None:
        """Encode frames to video file.

        Args:
            frames: List of RGB frames (H, W, C).
            path: Output video path.
        """
        import imageio.v3 as iio  # noqa: PLC0415

        # Stack frames and write
        iio.imwrite(
            path,
            frames,
            fps=self.fps,
            codec=self.codec,
        )

    def close(self) -> None:
        """Clean up resources."""
        # Save any pending episode
        if self._frames and self._current_episode_name:
            logger.warning(
                "Closing with unsaved frames for episode %s",
                self._current_episode_name,
            )

        self._frames = []
        self._current_episode_name = None

        logger.debug(
            "VideoRecorder closed: %d episodes recorded",
            self.episode_count,
        )

    def __enter__(self) -> Self:
        """Context manager entry.

        Returns:
            Self for use in with statement.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Context manager exit."""
        self.close()


__all__ = ["RecordMode", "VideoRecorder"]
