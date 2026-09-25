#!/usr/bin/env python3
"""Check if environment is ready for backend comparison.

Verifies that all dependencies are installed and configured.
"""

import sys
from importlib import import_module
from importlib.metadata import version


def check_module(name: str, package: str | None = None, required: bool = True) -> bool:
    """Check if module is available.

    Args:
        name: Module name to import
        package: Display name (if different from import name)
        required: Whether module is required or optional

    Returns:
        True if module is available
    """
    display_name = package or name
    try:
        mod = import_module(name)
        ver = version(package or name.split(".")[0])
        status = "✓" if required else "✓ (optional)"
        print(f"  {status} {display_name:<25} version {ver}")
        return True
    except (ImportError, ModuleNotFoundError):
        status = "✗ MISSING" if required else "⚠ missing (optional)"
        print(f"  {status} {display_name:<25}")
        return False


def check_policy(policy_path: str) -> bool:
    """Check if policy class is available.

    Args:
        policy_path: Full path to policy class

    Returns:
        True if policy is available
    """
    try:
        module_path, class_name = policy_path.rsplit(".", 1)
        module = import_module(module_path)
        policy_cls = getattr(module, class_name)

        # Check export support
        policy = policy_cls()
        backends = policy.get_supported_export_backends()

        status = "✓"
        print(f"  {status} {class_name:<25} backends: {', '.join(backends)}")
        return True
    except Exception as e:
        print(f"  ✗ {policy_path:<25} error: {e}")
        return False


def main():
    """Run all checks."""
    print("\n" + "=" * 70)
    print("Backend Accuracy Comparison - Environment Check")
    print("=" * 70)

    all_ok = True

    # Core dependencies
    print("\n1. Core Dependencies (Required):")
    checks = [
        ("torch", "torch", True),
        ("numpy", "numpy", True),
        ("physicalai", "physicalai-train", True),
    ]

    for mod, pkg, req in checks:
        if not check_module(mod, pkg, req):
            all_ok = False

    # Export backends
    print("\n2. Export Backends:")
    checks = [
        ("openvino", "openvino", True),
        ("onnx", "onnx", False),
    ]

    for mod, pkg, req in checks:
        if not check_module(mod, pkg, req):
            if req:
                all_ok = False

    # Closed-loop dependencies
    print("\n3. Closed-Loop Simulation (for closed_loop_benchmark.py):")
    checks = [
        ("libero.libero", "hf-libero", True),
        ("robosuite", "robosuite", True),
        ("mujoco", "mujoco", True),
    ]

    libero_ok = True
    for mod, pkg, req in checks:
        if not check_module(mod, pkg, req):
            libero_ok = False
            if req:
                all_ok = False

    if not libero_ok:
        print("\n  ⚠ Closed-loop benchmark will not work without LIBERO.")
        print("    Install with: uv pip install hf-libero")

    # LeRobot (optional)
    print("\n4. LeRobot Integration (Optional - for LeRobot policies):")
    lerobot_ok = check_module("lerobot", "lerobot", required=False)

    if not lerobot_ok:
        print("\n  Note: LeRobot policies (from HuggingFace) require lerobot package.")
        print("        Install with: uv pip install lerobot")

    # Policies
    print("\n5. Available Policies:")
    policy_paths = [
        "physicalai.policies.ACT",
        "physicalai.policies.Pi0",
        "physicalai.policies.Pi05",
        "physicalai.policies.Groot",
        "physicalai.policies.SmolVLA",
    ]

    for policy_path in policy_paths:
        check_policy(policy_path)

    if lerobot_ok:
        print("\n   LeRobot Policies:")
        lerobot_policies = [
            "physicalai.policies.lerobot.ACT",
            "physicalai.policies.lerobot.PI05",
            "physicalai.policies.lerobot.Diffusion",
        ]
        for policy_path in lerobot_policies:
            check_policy(policy_path)

    # GPU check
    print("\n6. Hardware Acceleration:")
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("  ⚠ CUDA not available (CPU only - slower)")

        # Check OpenVINO GPU
        try:
            import openvino as ov

            core = ov.Core()
            devices = core.available_devices()
            gpu_devices = [d for d in devices if "GPU" in d]
            if gpu_devices:
                print(f"  ✓ OpenVINO GPU available: {', '.join(gpu_devices)}")
            else:
                print("  ⚠ OpenVINO GPU not available (CPU only)")
        except Exception:
            pass

    except Exception as e:
        print(f"  ✗ Could not check GPU: {e}")

    # Summary
    print("\n" + "=" * 70)
    if all_ok:
        print("✅ Environment is READY!")
        print("\nYou can run:")
        print("  • numerical_comparison.py (dataset-based)")
        if libero_ok:
            print("  • closed_loop_benchmark.py (simulator-based)")
        else:
            print("  ⚠ closed_loop_benchmark.py requires LIBERO (install hf-libero)")
    else:
        print("❌ Environment has MISSING dependencies")
        print("\nInstall missing packages with:")
        print("  uv pip install 'physicalai-train[export]'")
        if not libero_ok:
            print("  uv pip install hf-libero")
        print("\nSee requirements.txt for full list")

    print("=" * 70 + "\n")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
