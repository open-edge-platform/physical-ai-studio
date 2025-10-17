"""Tests for video callback functionality."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from getiaction.data import Observation
from getiaction.gyms import Gym
from getiaction.train.callbacks import VideoLogger


class TestVideoLogger:
    """Test VideoLogger callback."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        return tempfile.mkdtemp()

    @pytest.fixture
    def video_logger(self, temp_dir):
        """Create VideoLogger instance for tests."""
        return VideoLogger(
            output_dir=temp_dir,
            fps=10,
            log_every_n_batches=1,
            max_videos_per_phase=3,
            phases=["train", "test"]
        )

    @pytest.fixture
    def observation_batch(self):
        """Create test Observation batch."""
        return Observation(
            state=torch.randn(2, 10),
            action=torch.randn(2, 4),
            images=torch.randn(2, 3, 64, 64)  # Batch of images
        )

    @pytest.fixture
    def observation_batch_dict_images(self):
        """Create test Observation batch with dict images."""
        return Observation(
            state=torch.randn(2, 10),
            action=torch.randn(2, 4),
            images={"camera1": torch.randn(2, 3, 64, 64), "camera2": torch.randn(2, 3, 32, 32)}
        )

    def test_initialization(self, video_logger, temp_dir):
        """Test VideoLogger initialization."""
        assert video_logger.output_dir == Path(temp_dir)
        assert video_logger.fps == 10
        assert video_logger.log_every_n_batches == 1
        assert video_logger.max_videos_per_phase == 3
        assert video_logger.phases == ["train", "test"]
        assert video_logger.video_counts == {"train": 0, "test": 0}

    def test_extract_video_from_observation_tensor(self, video_logger, observation_batch):
        """Test video extraction from Observation with tensor images."""
        frames = video_logger._extract_video_from_observation(observation_batch)
        assert frames is not None
        assert frames.shape == (2, 3, 64, 64)

    def test_extract_video_from_observation_dict(self, video_logger, observation_batch_dict_images):
        """Test video extraction from Observation with dict images."""
        frames = video_logger._extract_video_from_observation(observation_batch_dict_images)
        assert frames is not None
        # Should get the first camera's images
        assert frames.shape == (2, 3, 64, 64) or frames.shape == (2, 3, 32, 32)

    def test_extract_video_from_observation_no_images(self, video_logger):
        """Test video extraction when no images present."""
        observation = Observation(
            state=torch.randn(2, 10),
            action=torch.randn(2, 4),
            images=None
        )
        frames = video_logger._extract_video_from_observation(observation)
        assert frames is None

    def test_should_log_video(self, video_logger):
        """Test video logging condition checks."""
        # Should log for valid phase and batch
        assert video_logger._should_log_video("train", 0) is True

        # Should not log for invalid phase
        assert video_logger._should_log_video("invalid", 0) is False

        # Should not log when max videos reached
        video_logger.video_counts["train"] = 3
        assert video_logger._should_log_video("train", 0) is False

        # Should not log when batch index doesn't match interval
        video_logger.video_counts["train"] = 0
        video_logger.log_every_n_batches = 5
        assert video_logger._should_log_video("train", 2) is False
        assert video_logger._should_log_video("train", 5) is True

    def test_save_video_formats(self, video_logger):
        """Test video saving with different tensor formats."""
        # Test with 4D tensor (channels first)
        frames = torch.rand(3, 3, 32, 32)  # [T, C, H, W]
        video_logger._save_video(frames, "test_channels_first.mp4")

        # Test with 4D tensor (channels last)
        frames = torch.rand(3, 32, 32, 3)  # [T, H, W, C]
        video_logger._save_video(frames, "test_channels_last.mp4")

        # Test with normalized values [0, 1]
        frames = torch.rand(3, 32, 32, 3)
        video_logger._save_video(frames, "test_normalized.mp4")

        # Check files were created
        output_dir = video_logger.output_dir
        assert (output_dir / "test_channels_first.mp4").exists()
        assert (output_dir / "test_channels_last.mp4").exists()
        assert (output_dir / "test_normalized.mp4").exists()

    def test_on_train_batch_end(self, video_logger, observation_batch):
        """Test training batch end callback."""
        # Mock trainer and module
        trainer = MagicMock()
        trainer.current_epoch = 5
        pl_module = MagicMock()

        # Call callback
        video_logger.on_train_batch_end(trainer, pl_module, None, observation_batch, 0)

        # Check video was saved and count incremented
        expected_file = video_logger.output_dir / "train" / "epoch_005_batch_0000.mp4"
        assert expected_file.exists()
        assert video_logger.video_counts["train"] == 1

    def test_on_validation_batch_end(self, video_logger, observation_batch):
        """Test validation batch end callback."""
        # Add val to phases
        video_logger.phases.append("val")
        video_logger.video_counts["val"] = 0

        # Mock trainer and module
        trainer = MagicMock()
        trainer.current_epoch = 3
        pl_module = MagicMock()

        # Call callback
        video_logger.on_validation_batch_end(trainer, pl_module, None, observation_batch, 1)

        # Check video was saved and count incremented
        expected_file = video_logger.output_dir / "val" / "epoch_003_batch_0001.mp4"
        assert expected_file.exists()
        assert video_logger.video_counts["val"] == 1

    def test_on_test_batch_end(self, video_logger):
        """Test test batch end callback."""
        # Mock trainer and module
        trainer = MagicMock()
        pl_module = MagicMock()

        # Create mock gym environment
        gym_env = MagicMock(spec=Gym)

        # Call callback - should not crash but won't save video yet
        video_logger.on_test_batch_end(trainer, pl_module, None, gym_env, 2)

        # Currently this is a placeholder that doesn't save videos
        # Count should remain 0 since gym video capture is not implemented
        assert video_logger.video_counts["test"] == 0

    def test_max_videos_per_phase_limit(self, video_logger, observation_batch):
        """Test that max videos per phase is respected."""
        trainer = MagicMock()
        trainer.current_epoch = 0
        pl_module = MagicMock()

        # Log up to the maximum
        for i in range(video_logger.max_videos_per_phase):
            video_logger.on_train_batch_end(trainer, pl_module, None, observation_batch, i)

        # Should have saved max_videos_per_phase videos
        assert video_logger.video_counts["train"] == video_logger.max_videos_per_phase

        # Try to log one more - should be ignored
        initial_count = video_logger.video_counts["train"]
        video_logger.on_train_batch_end(trainer, pl_module, None, observation_batch, 999)
        assert video_logger.video_counts["train"] == initial_count  # No increment

    def test_phase_filtering(self, video_logger, observation_batch):
        """Test that only specified phases are logged."""
        video_logger.phases = ["test"]  # Only test phase

        trainer = MagicMock()
        trainer.current_epoch = 0
        pl_module = MagicMock()

        # Try to log training - should be ignored
        video_logger.on_train_batch_end(trainer, pl_module, None, observation_batch, 0)
        assert video_logger.video_counts.get("train", 0) == 0

        # Test phase uses gym environment, so count won't increment yet
        gym_env = MagicMock(spec=Gym)
        video_logger.on_test_batch_end(trainer, pl_module, None, gym_env, 0)
        # Still 0 because gym video capture is not implemented
        assert video_logger.video_counts["test"] == 0

    def test_device_compatibility(self, video_logger):
        """Test that VideoLogger works with tensors on different devices."""
        # Test CPU device (always available)
        cpu_observation = Observation(
            state=torch.randn(2, 10),
            action=torch.randn(2, 4),
            images=torch.randn(2, 3, 32, 32)  # CPU tensor
        )

        frames = video_logger._extract_video_from_observation(cpu_observation)
        assert frames is not None
        assert frames.device.type == "cpu"

        # Test saving video from CPU tensor
        video_logger._save_video(frames, "test_cpu_device.mp4")
        assert (video_logger.output_dir / "test_cpu_device.mp4").exists()

        # Test with CUDA if available
        if torch.cuda.is_available():
            cuda_observation = Observation(
                state=torch.randn(2, 10).cuda(),
                action=torch.randn(2, 4).cuda(),
                images=torch.randn(2, 3, 32, 32).cuda()  # CUDA tensor
            )

            frames = video_logger._extract_video_from_observation(cuda_observation)
            assert frames is not None
            assert frames.device.type == "cuda"

            # Test saving video from CUDA tensor - should work via .cpu()
            video_logger._save_video(frames, "test_cuda_device.mp4")
            assert (video_logger.output_dir / "test_cuda_device.mp4").exists()

        # Test with MPS if available (Apple Silicon)
        if torch.backends.mps.is_available():
            mps_observation = Observation(
                state=torch.randn(2, 10).to("mps"),
                action=torch.randn(2, 4).to("mps"),
                images=torch.randn(2, 3, 32, 32).to("mps")  # MPS tensor
            )

            frames = video_logger._extract_video_from_observation(mps_observation)
            assert frames is not None
            assert frames.device.type == "mps"

            # Test saving video from MPS tensor - should work via .cpu()
            video_logger._save_video(frames, "test_mps_device.mp4")
            assert (video_logger.output_dir / "test_mps_device.mp4").exists()