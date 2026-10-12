from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from lerobot.configs import RGBEncoderConfig
from lerobot.configs.video import VALID_VIDEO_CODECS

from internal_datasets.access_mode import DatasetAccessMode
from internal_datasets.lerobot.lerobot_dataset import InternalLeRobotDataset
from internal_datasets.lerobot.streaming_encoding_settings import StreamingEncodingSettings, _resolve_vcodec


@pytest.fixture
def fresh_vcodec_cache():
    _resolve_vcodec.cache_clear()
    yield
    _resolve_vcodec.cache_clear()


def test_recording_checks_video_encoder_before_creating_cache(tmp_path: Path, fresh_vcodec_cache) -> None:
    dataset = InternalLeRobotDataset.__new__(InternalLeRobotDataset)
    dataset.path = tmp_path / "dataset"
    dataset._streaming_encoding_settings = StreamingEncodingSettings()

    with (
        patch.object(StreamingEncodingSettings, "_vcodec_candidates", return_value=["h264"]),
        patch.object(StreamingEncodingSettings, "_is_vcodec_usable", return_value=False),
        patch("internal_datasets.lerobot.lerobot_dataset.get_settings") as settings_mock,
        pytest.raises(RuntimeError, match="No usable video encoder"),
    ):
        dataset.start_recording_mutation(fps=30, features={}, robot_type="so100")

    settings_mock.assert_not_called()
    assert not (tmp_path / "dataset").exists()


def test_recording_does_not_require_ffmpeg_executable(tmp_path: Path, fresh_vcodec_cache) -> None:
    dataset = InternalLeRobotDataset.__new__(InternalLeRobotDataset)
    dataset.path = tmp_path / "dataset"
    dataset._streaming_encoding_settings = StreamingEncodingSettings()

    with (
        patch("shutil.which", return_value=None),
        patch.object(StreamingEncodingSettings, "_vcodec_candidates", return_value=["h264"]),
        patch.object(StreamingEncodingSettings, "_is_vcodec_usable", return_value=True),
        patch("internal_datasets.lerobot.lerobot_dataset.get_settings") as settings_mock,
        patch("internal_datasets.lerobot.lerobot_dataset.InternalLeRobotDataset") as cache_dataset,
        patch("internal_datasets.lerobot.lerobot_dataset.RecordingMutation") as mutation,
    ):
        settings_mock.return_value.cache_dir = tmp_path
        result = dataset.start_recording_mutation(fps=30, features={}, robot_type="so100")

    cache_dataset.return_value.create.assert_called_once()
    assert result is mutation.return_value


def test_auto_codec_candidates_are_accepted_by_lerobot() -> None:
    assert set(StreamingEncodingSettings._vcodec_candidates()) <= VALID_VIDEO_CODECS


def test_streaming_settings_translate_to_lerobot_kwargs() -> None:
    settings = StreamingEncodingSettings(
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
    assert kwargs["rgb_encoder"].g is None
    assert "vcodec" not in kwargs


def test_create_uses_rgb_encoder_and_not_vcodec(tmp_path: Path) -> None:
    settings = StreamingEncodingSettings(
        streaming_encoding=True,
        vcodec="h264",
        encoder_threads=2,
        encoder_queue_maxsize=60,
    )
    dataset = InternalLeRobotDataset.__new__(InternalLeRobotDataset)
    dataset.path = tmp_path / "dataset"
    dataset._streaming_encoding_settings = settings
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
    assert kwargs["rgb_encoder"].g is None
    assert kwargs["streaming_encoding"] is True
    assert kwargs["encoder_threads"] == 2
    assert kwargs["encoder_queue_maxsize"] == 60
    assert "vcodec" not in kwargs


def test_load_dataset_is_read_only_and_does_not_pass_write_kwargs(tmp_path: Path) -> None:
    settings = StreamingEncodingSettings(
        streaming_encoding=True,
        vcodec="h264",
        encoder_threads=2,
        encoder_queue_maxsize=60,
    )
    dataset = InternalLeRobotDataset.__new__(InternalLeRobotDataset)
    dataset.path = tmp_path / "dataset"
    dataset._streaming_encoding_settings = settings
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
    settings = StreamingEncodingSettings(
        streaming_encoding=True,
        vcodec="h264",
        encoder_threads=2,
        encoder_queue_maxsize=60,
    )
    dataset = InternalLeRobotDataset.__new__(InternalLeRobotDataset)
    dataset.path = tmp_path / "dataset"
    dataset._streaming_encoding_settings = settings
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
    assert kwargs["rgb_encoder"].g is None
    assert kwargs["streaming_encoding"] is True
    assert kwargs["encoder_threads"] == 2
    assert kwargs["encoder_queue_maxsize"] == 60
    assert "vcodec" not in kwargs


def test_resume_dataset_raises_in_read_only_mode(tmp_path: Path) -> None:
    settings = StreamingEncodingSettings(
        streaming_encoding=True,
        vcodec="h264",
        encoder_threads=2,
        encoder_queue_maxsize=60,
    )
    dataset = InternalLeRobotDataset.__new__(InternalLeRobotDataset)
    dataset.path = tmp_path / "dataset"
    dataset._streaming_encoding_settings = settings
    dataset._access_mode = DatasetAccessMode.READ_ONLY

    with patch.object(InternalLeRobotDataset, "_resume_for_writing") as resume_mock:
        try:
            dataset.resume_dataset()
            assert False, "Expected ValueError"
        except ValueError as exc:
            assert "RECORDING_MUTATION" in str(exc)
    resume_mock.assert_not_called()
