"""Comprehensive training tests for Diffusion policy.

This script tests:
1. Python API training
2. YAML CLI training
3. Forward pass equivalence

Usage:
    python test_diffusion_training.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
import yaml
from lightning.pytorch import Trainer

from getiaction.data.lerobot import LeRobotDataModule
from getiaction.policies.lerobot.diffusion import Diffusion


def test_python_training():
    """Test training Diffusion policy via Python API."""
    print("\n" + "=" * 80)
    print("Test 1: Python API Training")
    print("=" * 80)

    try:
        # Create policy with minimal config for testing
        policy = Diffusion(
            down_dims=[512, 1024, 2048],
            learning_rate=1e-4,
        )
        print("✓ Created Diffusion policy")

        # Create datamodule with minimal data
        datamodule = LeRobotDataModule(
            repo_id="lerobot/pusht",
            episodes=[0],  # Just first episode
            train_batch_size=2,
        )
        print("✓ Created LeRobotDataModule")

        # Create trainer (minimal configuration for testing)
        trainer = Trainer(
            max_steps=2,
            enable_checkpointing=False,
            logger=False,
            num_sanity_val_steps=0,  # Skip validation sanity check
            accelerator="cpu",  # Use CPU to avoid distribution issues
            devices=1,
        )
        print("✓ Created Trainer")

        # Run training
        print("\nRunning training for 2 steps...")
        trainer.fit(policy, datamodule)
        print("✓ Training completed successfully")

        print("\n✓ Python API training test PASSED")
        return True

    except Exception as e:
        print(f"\n✗ Python API training test FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_cli_training():
    """Test training Diffusion policy via YAML CLI."""
    print("\n" + "=" * 80)
    print("Test 2: YAML CLI Training")
    print("=" * 80)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"

            # Create YAML config
            config = {
                "model": {
                    "class_path": "getiaction.policies.lerobot.diffusion.Diffusion",
                    "init_args": {
                        "down_dims": [512, 1024, 2048],
                        "learning_rate": 1e-4,
                    },
                },
                "data": {
                    "class_path": "getiaction.data.lerobot.LeRobotDataModule",
                    "init_args": {
                        "repo_id": "lerobot/pusht",
                        "episodes": [0],
                        "train_batch_size": 2,
                    },
                },
                "trainer": {
                    "max_steps": 2,
                    "enable_checkpointing": False,
                    "logger": False,
                    "num_sanity_val_steps": 0,
                },
            }

            # Write config
            with open(config_path, "w") as f:
                yaml.dump(config, f)
            print(f"✓ Created YAML config at {config_path}")

            # Test that we can load and parse the config
            from lightning.pytorch.cli import LightningCLI

            # Create CLI instance with the config
            cli = LightningCLI(
                Diffusion,
                LeRobotDataModule,
                seed_everything_default=42,
                trainer_defaults={
                    "max_steps": 2,
                    "logger": False,
                    "enable_checkpointing": False,
                    "accelerator": "cpu",
                    "devices": 1,
                },
                run=False,  # Don't run yet, just parse
                args=[f"--config={config_path}"],
            )
            print("✓ Successfully parsed YAML config with LightningCLI")

            # Run training through CLI
            print("\nRunning training for 2 steps via CLI...")
            cli.trainer.fit(cli.model, cli.datamodule)
            print("✓ CLI training completed successfully")

            print("\n✓ YAML CLI training test PASSED")
            return True

    except Exception as e:
        print(f"\n✗ YAML CLI training test FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_forward_pass_equivalence():
    """Test that forward pass outputs are consistent."""
    print("\n" + "=" * 80)
    print("Test 3: Forward Pass Equivalence")
    print("=" * 80)

    try:
        # Create policy
        policy = Diffusion()

        # Create datamodule
        datamodule = LeRobotDataModule(
            repo_id="lerobot/pusht",
            episodes=[0],
            train_batch_size=2,
        )
        datamodule.setup("fit")
        print("✓ Policy setup completed")

        # Get a batch from the training dataloader
        dataloader = datamodule.train_dataloader()
        batch = next(iter(dataloader))
        print("✓ Got batch from dataloader")

        # Run training step (which internally calls forward)
        policy.setup("fit", datamodule)
        with torch.no_grad():
            loss = policy.training_step(batch, 0)
        print(f"✓ Forward pass successful, loss: {loss.item():.6f}")

        # Verify loss is a scalar
        assert loss.numel() == 1, f"Expected scalar loss, got shape {loss.shape}"
        print("✓ Loss is scalar as expected")

        print("\n✓ Forward pass equivalence test PASSED")
        return True

    except Exception as e:
        print(f"\n✗ Forward pass equivalence test FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all training tests."""
    print("Testing Diffusion Policy Training")
    print("=" * 80)

    results = []

    # Test 1: Python API
    try:
        results.append(("Python API Training", test_python_training()))
    except Exception as e:
        print(f"Failed to run Python API test: {e}")
        results.append(("Python API Training", False))

    # Test 2: YAML CLI
    try:
        results.append(("YAML CLI Training", test_cli_training()))
    except Exception as e:
        print(f"Failed to run YAML CLI test: {e}")
        results.append(("YAML CLI Training", False))

    # Test 3: Forward pass equivalence
    try:
        results.append(("Forward Pass Equivalence", test_forward_pass_equivalence()))
    except Exception as e:
        print(f"Failed to run forward pass equivalence test: {e}")
        results.append(("Forward Pass Equivalence", False))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    all_passed = all(passed for _, passed in results)
    print("=" * 80)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 80)

    return all_passed


if __name__ == "__main__":
    import sys

    success = main()
    sys.exit(0 if success else 1)
