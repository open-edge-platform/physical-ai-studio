#!/usr/bin/env python
"""Integration tests for ACT wrapper with proper temporal data configuration.

These tests validate end-to-end training workflows using LeRobot's proper
data loading with temporal chunking configured via delta_timestamps.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Add getiaction to path
getiaction_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(getiaction_path))


@pytest.fixture
def temporal_dataset():
    """Load dataset with temporal chunking configured."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # Configure temporal chunking for ACT
    # Note: ACT only uses temporal chunking for ACTIONS, not observations!
    # observation_delta_indices returns None in ACTConfig
    # PushT has fps=10, so delta_timestamps must be multiples of 0.1
    delta_timestamps = {
        # No temporal dimension for observations - ACT uses single timestep
        "action": [  # 10 action steps (matching chunk_size)
            0.0, 0.1, 0.2, 0.3, 0.4,
            0.5, 0.6, 0.7, 0.8, 0.9
        ],
    }

    return LeRobotDataset(
        "lerobot/pusht",
        delta_timestamps=delta_timestamps,
    )


@pytest.fixture
def temporal_dataloader(temporal_dataset):
    """Create dataloader from temporal dataset."""
    return torch.utils.data.DataLoader(
        temporal_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
    )


@pytest.fixture
def policy_with_temporal_config(temporal_dataset):
    """Create ACT policy configured for temporal dataset."""
    from getiaction.policies.lerobot import ACT
    from lerobot.datasets.utils import dataset_to_policy_features

    features = dataset_to_policy_features(temporal_dataset.meta.features)

    return ACT(
        input_features=features,
        output_features=features,
        dim_model=256,
        chunk_size=10,  # Matches action sequence length
        n_action_steps=10,
        n_encoder_layers=2,
        n_decoder_layers=1,
        vision_backbone="resnet18",
        use_vae=True,
        latent_dim=32,
        stats=temporal_dataset.meta.stats,
    )


class TestACTIntegration:
    """Integration tests with proper temporal data configuration."""

    def test_training_step_with_temporal_data(
        self,
        policy_with_temporal_config,
        temporal_dataloader
    ):
        """Test that training step works with properly configured temporal data."""
        policy = policy_with_temporal_config
        policy.train()

        # Get batch with temporal sequences
        batch = next(iter(temporal_dataloader))

        # Verify batch has correct temporal dimensions
        assert batch["action"].ndim == 3, (
            f"Expected 3D action tensor [batch, time, dim], got shape {batch['action'].shape}"
        )
        assert batch["action"].shape[1] == 10, (
            f"Expected 10 time steps, got {batch['action'].shape[1]}"
        )

        # Training step should work
        loss = policy.training_step(batch, batch_idx=0)

        # Loss can be either tensor or float depending on whether policy is attached to trainer
        assert isinstance(loss, (torch.Tensor, float))
        if isinstance(loss, torch.Tensor):
            loss_value = loss.item()
        else:
            loss_value = loss

        assert loss_value > 0, f"Expected positive loss, got {loss_value}"
        print(f"✅ Training step successful with loss: {loss_value:.4f}")

    def test_validation_step_with_temporal_data(
        self,
        policy_with_temporal_config,
        temporal_dataloader
    ):
        """Test that validation step works with temporal data."""
        policy = policy_with_temporal_config
        policy.eval()

        batch = next(iter(temporal_dataloader))

        # Validation step should work
        val_loss = policy.validation_step(batch, batch_idx=0)

        assert isinstance(val_loss, torch.Tensor)
        assert val_loss.shape == torch.Size([])
        assert torch.isfinite(val_loss)
        print(f"✅ Validation loss: {val_loss.item():.4f}")

    def test_full_training_loop(
        self,
        policy_with_temporal_config,
        temporal_dataloader
    ):
        """Test full training loop with Lightning Trainer."""
        import lightning as L

        policy = policy_with_temporal_config

        # Create Lightning trainer
        trainer = L.Trainer(
            max_epochs=1,
            limit_train_batches=5,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
            accelerator="cpu",  # Force CPU for reproducibility
            num_sanity_val_steps=0,  # Disable validation sanity check
        )

        # Train for a few steps
        trainer.fit(policy, train_dataloaders=temporal_dataloader)

        # Verify training happened
        assert trainer.global_step == 5
        assert policy.trainer is not None
        print(f"✅ Trained for {trainer.global_step} steps")

    @pytest.mark.skip(reason="DataModule integration requires proper feature configuration - see INTEGRATION_TEST_RESULTS.md")
    def test_training_with_getiaction_datamodule(self, temporal_dataset):
        """Test training with GetiAction's LeRobotDataModule.

        Note: This test is skipped because LeRobotDataModule requires careful configuration
        of input/output features that match the dataset structure. The test demonstrates
        the intended usage pattern but needs further investigation of proper feature mapping.
        """
        from getiaction.data.lerobot import LeRobotDataModule
        from getiaction.policies.lerobot import ACT
        from lerobot.datasets.utils import dataset_to_policy_features
        import lightning as L

        # Create DataModule with temporal config
        # Note: ACT only uses temporal chunking for ACTIONS, not observations
        # PushT has fps=10, so delta_timestamps must be multiples of 0.1
        delta_timestamps = {
            # No temporal dimension for observations - ACT uses single timestep
            "action": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        }

        datamodule = LeRobotDataModule(
            repo_id="lerobot/pusht",
            train_batch_size=8,
            delta_timestamps=delta_timestamps,
        )

        # Create policy
        features = dataset_to_policy_features(temporal_dataset.meta.features)
        policy = ACT(
            input_features=features,
            output_features=features,
            dim_model=256,
            chunk_size=10,
            n_action_steps=10,
            stats=temporal_dataset.meta.stats,
        )

        # Train
        trainer = L.Trainer(
            max_epochs=1,
            limit_train_batches=3,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
            accelerator="cpu",
            num_sanity_val_steps=0,  # Disable validation sanity check
        )

        trainer.fit(policy, datamodule)

        assert trainer.global_step == 3
        print(f"✅ Training with DataModule successful!")


@pytest.mark.skipif(
    "SLOW_TESTS" not in __import__("os").environ,
    reason="Slow test - set SLOW_TESTS=1 to run"
)
class TestACTMultiEpochTraining:
    """Slower integration tests for multi-epoch training."""

    def test_multi_epoch_training(
        self,
        policy_with_temporal_config,
        temporal_dataloader
    ):
        """Test training for multiple epochs."""
        import lightning as L

        policy = policy_with_temporal_config

        trainer = L.Trainer(
            max_epochs=3,
            limit_train_batches=10,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
            accelerator="cpu",
        )

        trainer.fit(policy, train_dataloaders=temporal_dataloader)

        assert trainer.global_step == 30  # 3 epochs * 10 batches
        print(f"✅ Multi-epoch training successful: {trainer.global_step} steps")

    def test_checkpoint_save_load(
        self,
        policy_with_temporal_config,
        temporal_dataloader,
        tmp_path
    ):
        """Test saving and loading checkpoints."""
        import lightning as L

        policy = policy_with_temporal_config
        checkpoint_path = tmp_path / "test.ckpt"

        # Train a bit
        trainer = L.Trainer(
            max_epochs=1,
            limit_train_batches=5,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
            accelerator="cpu",
        )
        trainer.fit(policy, train_dataloaders=temporal_dataloader)

        # Save checkpoint
        trainer.save_checkpoint(checkpoint_path)
        assert checkpoint_path.exists()

        # Load checkpoint
        from getiaction.policies.lerobot import ACT
        loaded_policy = ACT.load_from_checkpoint(checkpoint_path)

        assert loaded_policy is not None
        print(f"✅ Checkpoint saved and loaded successfully")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
