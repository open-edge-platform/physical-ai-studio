from pathlib import Path
from unittest.mock import MagicMock, patch

import av
import numpy as np
import pytest
from lerobot.configs import RGBEncoderConfig
from lerobot.configs.video import DepthEncoderConfig
from lerobot.datasets import pyav_utils

from internal_datasets.access_mode import DatasetAccessMode
from internal_datasets.lerobot import video_encoding_settings
from internal_datasets.lerobot.lerobot_dataset import InternalLeRobotDataset
from internal_datasets.lerobot.video_encoding_settings import (
    StudioRGBEncoderConfig,
    VideoEncodingSettings,
    video_encoding_presets,
)


def test_video_recording_settings_translate_to_lerobot_kwargs() -> None:
    settings = VideoEncodingSettings(
        streaming_encoding=True,
        vcodec="h264",
        encoder_threads=4,
        encoder_queue_maxsize=60,
    )

    kwargs = settings.to_lerobot_write_kwargs()

    assert kwargs["streaming_encoding"] is True
    assert kwargs["encoder_threads"] == 4
    assert kwargs["encoder_queue_maxsize"] == 60
    assert isinstance(kwargs["rgb_encoder"], RGBEncoderConfig)
    assert kwargs["rgb_encoder"].vcodec == "h264"
    assert kwargs["rgb_encoder"].g == 2
    assert "vcodec" not in kwargs


def test_unused_depth_encoder_does_not_resolve_or_validate_a_codec() -> None:
    with (
        patch.object(DepthEncoderConfig, "resolve_vcodec", side_effect=AssertionError("Unused depth codec resolved")),
        patch.object(DepthEncoderConfig, "validate", side_effect=AssertionError("Unused depth codec validated")),
    ):
        encoder = video_encoding_settings._UnsupportedDepthEncoderConfig()

    with pytest.raises(NotImplementedError, match="Depth video recording is not supported"):
        encoder.get_codec_options()


@pytest.mark.parametrize("streaming_encoding", [True, False])
def test_rgb_recording_create_and_resume_without_usable_depth_encoder(tmp_path: Path, streaming_encoding: bool) -> None:
    settings = VideoEncodingSettings(vcodec="libsvtav1", streaming_encoding=streaming_encoding, encoder_threads=1)
    features = {
        "action": {"dtype": "float32", "shape": (1,), "names": ["joint.pos"]},
        "observation.images.camera": {
            "dtype": "video",
            "shape": (64, 64, 3),
            "names": ["height", "width", "channels"],
        },
    }
    codec_video_formats = pyav_utils._get_codec_video_formats

    def docker_codec_video_formats(vcodec: str) -> tuple[str, ...]:
        # The LGPL Docker build resolves generic HEVC to QSV, not libx265.
        return ("nv12", "p010le", "qsv") if vcodec == "hevc" else codec_video_formats(vcodec)

    with patch.object(pyav_utils, "_get_codec_video_formats", side_effect=docker_codec_video_formats):
        # Prove that the unmodified LeRobot default cannot work in this build.
        with pytest.raises(ValueError, match=r"gray12le.*not supported"):
            DepthEncoderConfig()

        dataset = InternalLeRobotDataset(
            tmp_path / "dataset",
            recording_video_encoding_settings=settings,
            access_mode=DatasetAccessMode.RECORDING_MUTATION,
        )
        dataset.create(fps=30, features=features, robot_type="test")
        for expected_episodes in (1, 2):
            if expected_episodes == 2:
                dataset.resume_dataset()
            dataset.prepare_for_writing()
            for _ in range(2):
                dataset._dataset.add_frame(
                    {
                        "action": np.zeros(1, dtype=np.float32),
                        "observation.images.camera": np.zeros((64, 64, 3), dtype=np.uint8),
                        "task": "test",
                    }
                )
            dataset.save_episode()
            dataset.finalize()
            assert dataset._dataset.num_episodes == expected_episodes

    decoded_frames = 0
    for video_path in dataset.path.rglob("*.mp4"):
        with av.open(str(video_path)) as container:
            decoded_frames += sum(1 for _ in container.decode(video=0))
    assert decoded_frames == 4


def test_create_uses_rgb_encoder_and_not_vcodec(tmp_path: Path) -> None:
    settings = VideoEncodingSettings(
        streaming_encoding=True,
        vcodec="h264",
        encoder_threads=2,
        encoder_queue_maxsize=60,
    )
    dataset = InternalLeRobotDataset.__new__(InternalLeRobotDataset)
    dataset.path = tmp_path / "dataset"
    dataset._recording_video_encoding_settings = settings
    dataset._access_mode = DatasetAccessMode.READ_ONLY

    with (
        patch.object(InternalLeRobotDataset, "_check_repository_exists", return_value=False),
        patch(
            "internal_datasets.lerobot.lerobot_dataset.LeRobotDataset.create", return_value=MagicMock()
        ) as create_mock,
    ):
        dataset.create(fps=30, features={}, robot_type="so100")

    kwargs = create_mock.call_args.kwargs
    assert isinstance(kwargs["rgb_encoder"], RGBEncoderConfig)
    assert kwargs["rgb_encoder"].vcodec == "h264"
    assert kwargs["rgb_encoder"].g == 2
    assert isinstance(kwargs["depth_encoder"], video_encoding_settings._UnsupportedDepthEncoderConfig)
    assert kwargs["streaming_encoding"] is True
    assert kwargs["encoder_threads"] == 2
    assert kwargs["encoder_queue_maxsize"] == 60
    assert "vcodec" not in kwargs


def test_load_dataset_is_read_only_and_does_not_pass_write_kwargs(tmp_path: Path) -> None:
    settings = VideoEncodingSettings(
        streaming_encoding=True,
        vcodec="h264",
        encoder_threads=2,
        encoder_queue_maxsize=60,
    )
    dataset = InternalLeRobotDataset.__new__(InternalLeRobotDataset)
    dataset.path = tmp_path / "dataset"
    dataset._recording_video_encoding_settings = settings
    dataset._access_mode = DatasetAccessMode.READ_ONLY

    with (
        patch.object(InternalLeRobotDataset, "_check_repository_exists", return_value=True),
        patch(
            "internal_datasets.lerobot.lerobot_dataset.LeRobotDataset",
            return_value=MagicMock(num_episodes=1),
        ) as init_mock,
    ):
        dataset.load_dataset()

    kwargs = init_mock.call_args.kwargs
    assert "rgb_encoder" not in kwargs
    assert "depth_encoder" not in kwargs
    assert "streaming_encoding" not in kwargs
    assert "encoder_threads" not in kwargs
    assert "encoder_queue_maxsize" not in kwargs
    assert "vcodec" not in kwargs


def test_resume_dataset_uses_write_kwargs_and_not_vcodec(tmp_path: Path) -> None:
    settings = VideoEncodingSettings(
        streaming_encoding=True,
        vcodec="h264",
        encoder_threads=2,
        encoder_queue_maxsize=60,
    )
    dataset = InternalLeRobotDataset.__new__(InternalLeRobotDataset)
    dataset.path = tmp_path / "dataset"
    dataset._recording_video_encoding_settings = settings
    dataset._access_mode = DatasetAccessMode.RECORDING_MUTATION

    with (
        patch.object(InternalLeRobotDataset, "_check_repository_exists", return_value=True),
        patch(
            "internal_datasets.lerobot.lerobot_dataset.LeRobotDataset.resume",
            return_value=MagicMock(num_episodes=1),
        ) as resume_mock,
    ):
        dataset.resume_dataset()

    kwargs = resume_mock.call_args.kwargs
    assert isinstance(kwargs["rgb_encoder"], RGBEncoderConfig)
    assert kwargs["rgb_encoder"].vcodec == "h264"
    assert kwargs["rgb_encoder"].g == 2
    assert isinstance(kwargs["depth_encoder"], video_encoding_settings._UnsupportedDepthEncoderConfig)
    assert kwargs["streaming_encoding"] is True
    assert kwargs["encoder_threads"] == 2
    assert kwargs["encoder_queue_maxsize"] == 60
    assert "vcodec" not in kwargs


def test_resume_dataset_raises_in_read_only_mode(tmp_path: Path) -> None:
    settings = VideoEncodingSettings(
        streaming_encoding=True,
        vcodec="h264",
        encoder_threads=2,
        encoder_queue_maxsize=60,
    )
    dataset = InternalLeRobotDataset.__new__(InternalLeRobotDataset)
    dataset.path = tmp_path / "dataset"
    dataset._recording_video_encoding_settings = settings
    dataset._access_mode = DatasetAccessMode.READ_ONLY

    with patch.object(InternalLeRobotDataset, "_resume_for_writing") as resume_mock:
        try:
            dataset.resume_dataset()
            assert False, "Expected ValueError"
        except ValueError as exc:
            assert "RECORDING_MUTATION" in str(exc)
    resume_mock.assert_not_called()


def test_video_recording_settings_translate_extra_encoding_fields() -> None:
    settings = VideoEncodingSettings(vcodec="h264", pix_fmt="yuv420p", g=10, crf=23, preset="medium")

    rgb_encoder = settings.to_lerobot_write_kwargs()["rgb_encoder"]

    assert (rgb_encoder.pix_fmt, rgb_encoder.g, rgb_encoder.crf, rgb_encoder.preset) == ("yuv420p", 10, 23, "medium")


def test_hardware_codec_defaults_to_nv12_pix_fmt() -> None:
    with patch.object(video_encoding_settings, "_is_vcodec_usable", return_value=True):
        rgb_encoder = VideoEncodingSettings(vcodec="h264_nvenc").to_lerobot_write_kwargs()["rgb_encoder"]

    assert rgb_encoder.pix_fmt == "nv12"


def test_vcodec_candidates_prefer_hardware_and_exclude_native_codecs() -> None:
    candidates = video_encoding_settings.vcodec_candidates()

    assert candidates[:3] == ["av1_qsv", "hevc_qsv", "h264_qsv"]
    assert "h264" not in candidates
    assert "hevc" not in candidates


def test_explicit_unusable_vcodec_is_rejected() -> None:
    with (
        patch.object(video_encoding_settings, "_is_vcodec_usable", return_value=False),
        pytest.raises(RuntimeError, match=r"av1_qsv.*unavailable"),
    ):
        StudioRGBEncoderConfig(vcodec="av1_qsv").resolve_vcodec()


def test_named_numeric_encoder_preset_is_normalized() -> None:
    with patch.object(video_encoding_settings, "_numeric_preset_choices", return_value={"veryfast": 7}):
        assert video_encoding_settings._normalise_preset("h264_qsv", "veryfast") == 7


def test_video_encoding_presets_include_apple_videotoolbox() -> None:
    presets = {preset.id: preset for preset in video_encoding_presets()}

    assert presets["apple-videotoolbox-h264"].settings.vcodec == "h264_videotoolbox"
    assert presets["apple-videotoolbox-hevc"].settings.vcodec == "hevc_videotoolbox"


def test_explicit_unusable_vcodec_raises_when_nothing_usable() -> None:
    with (
        patch.object(video_encoding_settings, "_is_vcodec_usable", return_value=False),
        pytest.raises(RuntimeError, match="No usable video encoder"),
    ):
        StudioRGBEncoderConfig(vcodec="auto").resolve_vcodec()
