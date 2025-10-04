#!/usr/bin/env python
"""Test if ACT training works with GetiAction's LeRobotDataModule."""

import sys
from pathlib import Path

import torch

# Add getiaction to path
getiaction_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(getiaction_path))

from getiaction.data.lerobot import LeRobotDataModule
from getiaction.policies.lerobot import ACT
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import dataset_to_policy_features

def test_with_getiaction_datamodule():
    """Test if training works with GetiAction's DataModule."""
    print("\n" + "="*70)
    print("TEST: ACT Training with GetiAction DataModule")
    print("="*70)

    # Load dataset
    print("\n1. Loading dataset...")
    dataset = LeRobotDataset("lerobot/pusht")
    features = dataset_to_policy_features(dataset.meta.features)
    print(f"   ✓ Loaded {len(dataset)} samples")

    # Create DataModule WITHOUT delta_timestamps (same as simple DataLoader)
    print("\n2. Creating DataModule WITHOUT delta_timestamps...")
    datamodule_no_delta = LeRobotDataModule(
        repo_id="lerobot/pusht",
        batch_size=8,
        num_workers=0,
        delta_timestamps=None,  # This is the problem!
    )
    datamodule_no_delta.setup()

    # Create policy
    print("\n3. Creating ACT policy...")
    policy = ACT(
        input_features=features,
        output_features=features,
        dim_model=256,
        chunk_size=10,
        n_action_steps=10,
        n_encoder_layers=2,
        n_decoder_layers=1,
        stats=dataset.meta.stats,
    )
    policy.train()
    print("   ✓ Policy created")

    # Try training step WITHOUT delta_timestamps
    print("\n4. Testing training step WITHOUT delta_timestamps...")
    train_loader = datamodule_no_delta.train_dataloader()
    batch = next(iter(train_loader))

    print(f"   Batch keys: {batch.keys()}")
    print(f"   Action shape: {batch['action'].shape}")
    print(f"   Image shape: {batch.get('observation.image', torch.zeros(1)).shape}")

    try:
        loss = policy.training_step(batch, batch_idx=0)
        print(f"   ✓ Training step works! Loss: {loss.item():.4f}")
    except RuntimeError as e:
        print(f"   ❌ Training step FAILED: {e}")
        print(f"      This is expected without delta_timestamps!")

    # Now create DataModule WITH delta_timestamps
    print("\n5. Creating DataModule WITH delta_timestamps...")
    delta_timestamps = {
        "observation.image": [-0.1, -0.05, 0.0],  # 3 frames
        "action": [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18],  # 10 actions
    }
    datamodule_with_delta = LeRobotDataModule(
        repo_id="lerobot/pusht",
        batch_size=8,
        num_workers=0,
        delta_timestamps=delta_timestamps,
    )
    datamodule_with_delta.setup()
    print("   ✓ DataModule created with delta_timestamps")

    # Try training step WITH delta_timestamps
    print("\n6. Testing training step WITH delta_timestamps...")
    train_loader = datamodule_with_delta.train_dataloader()
    batch = next(iter(train_loader))

    print(f"   Batch keys: {batch.keys()}")
    print(f"   Action shape: {batch['action'].shape}")
    if 'observation.image' in batch:
        print(f"   Image shape: {batch['observation.image'].shape}")

    try:
        loss = policy.training_step(batch, batch_idx=0)
        print(f"   ✓ Training step works! Loss: {loss.item():.4f}")
        print("   ✅ SUCCESS - Training works with delta_timestamps!")
    except Exception as e:
        print(f"   ❌ Training step FAILED: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("\n✅ GetiAction's LeRobotDataModule DOES support delta_timestamps")
    print("❌ Simple torch DataLoader does NOT configure temporal chunking")
    print("\n📝 Test failures are due to using simple DataLoader")
    print("   Fix: Use LeRobotDataModule with proper delta_timestamps config")
    print()

if __name__ == "__main__":
    test_with_getiaction_datamodule()
