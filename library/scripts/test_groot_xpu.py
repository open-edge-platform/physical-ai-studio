# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Test script for Groot model on Intel XPU.

Run this script to verify XPU setup is working correctly:

    # With render group (required for XPU access)
    sg render -c "python scripts/test_groot_xpu.py"

    # Or if already in render group
    python scripts/test_groot_xpu.py
"""

import torch


def test_xpu_availability() -> None:
    """Test basic XPU availability."""
    print("=" * 60)
    print("Intel XPU Availability Test")
    print("=" * 60)

    print(f"\nPyTorch version: {torch.__version__}")

    if not hasattr(torch, "xpu"):
        print("\n❌ PyTorch does not have XPU support.")
        print("   Install PyTorch from XPU wheel index:")
        print("   uv pip install torch --index-url https://download.pytorch.org/whl/xpu")
        return False

    if not torch.xpu.is_available():
        print("\n❌ XPU is not available.")
        print("   Possible causes:")
        print("   - User not in 'render' group (run: sudo usermod -aG render $USER)")
        print("   - Level Zero runtime not installed")
        print("   - Intel GPU driver not loaded")
        return False

    print(f"\n✓ XPU available: {torch.xpu.is_available()}")
    print(f"✓ XPU device count: {torch.xpu.device_count()}")

    for i in range(torch.xpu.device_count()):
        props = torch.xpu.get_device_properties(i)
        print(f"\nDevice {i}: {torch.xpu.get_device_name(i)}")
        print(f"  Total memory: {props.total_memory / 1024**3:.2f} GB")

    return True


def test_xpu_compute() -> None:
    """Test basic compute operations on XPU."""
    print("\n" + "=" * 60)
    print("XPU Compute Test")
    print("=" * 60)

    # Matrix multiplication
    print("\n1. Matrix multiplication (1000x1000)...")
    x = torch.randn(1000, 1000, device="xpu")
    y = torch.randn(1000, 1000, device="xpu")
    z = x @ y
    print(f"   ✓ Result shape: {z.shape}, dtype: {z.dtype}")

    # BFloat16 support
    print("\n2. BFloat16 support...")
    x_bf16 = torch.randn(1000, 1000, device="xpu", dtype=torch.bfloat16)
    y_bf16 = torch.randn(1000, 1000, device="xpu", dtype=torch.bfloat16)
    z_bf16 = x_bf16 @ y_bf16
    print(f"   ✓ Result shape: {z_bf16.shape}, dtype: {z_bf16.dtype}")

    # Autocast
    print("\n3. Autocast support...")
    with torch.autocast(device_type="xpu", dtype=torch.bfloat16):
        a = torch.randn(1000, 1000, device="xpu")
        b = torch.randn(1000, 1000, device="xpu")
        c = a @ b
    print(f"   ✓ Autocast result dtype: {c.dtype}")

    # Simple model test
    print("\n4. Simple nn.Module test...")
    model = torch.nn.Sequential(
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128),
    ).to("xpu")

    x = torch.randn(32, 512, device="xpu")
    with torch.autocast(device_type="xpu", dtype=torch.bfloat16):
        y = model(x)
    print(f"   ✓ Output shape: {y.shape}, dtype: {y.dtype}")

    print("\n✓ All compute tests passed!")


def test_groot_device_utils() -> None:
    """Test Groot device utility functions."""
    print("\n" + "=" * 60)
    print("Groot Device Utilities Test")
    print("=" * 60)

    try:
        from getiaction.policies.groot import get_best_device, is_xpu_available

        print(f"\n✓ is_xpu_available(): {is_xpu_available()}")
        print(f"✓ get_best_device(): {get_best_device()}")

    except ImportError as e:
        print(f"\n⚠ Could not import Groot utilities: {e}")
        print("   Make sure getiaction is installed:")
        print("   uv pip install -e .")


def main() -> None:
    """Run all XPU tests."""
    if not test_xpu_availability():
        print("\n" + "=" * 60)
        print("XPU setup incomplete. Please fix the issues above.")
        print("=" * 60)
        return

    test_xpu_compute()
    test_groot_device_utils()

    print("\n" + "=" * 60)
    print("✓ All tests passed! XPU is ready for Groot model.")
    print("=" * 60)
    print("\nExample usage:")
    print("  from getiaction.policies.groot import GrootModel, get_best_device")
    print("  device = get_best_device()  # Returns 'xpu' on this machine")
    print("  model = GrootModel.from_pretrained().to(device)")


if __name__ == "__main__":
    main()
