from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from lerobot.configs import RGBEncoderConfig

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
