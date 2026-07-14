from pathlib import Path
from unittest.mock import MagicMock, patch

from lerobot.configs import RGBEncoderConfig

from internal_datasets.access_mode import DatasetAccessMode
from internal_datasets.lerobot.lerobot_dataset import InternalLeRobotDataset
from internal_datasets.lerobot.streaming_encoding_settings import StreamingEncodingSettings


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
