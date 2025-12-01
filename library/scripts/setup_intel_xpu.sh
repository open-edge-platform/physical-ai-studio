#!/bin/bash
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Intel XPU Setup Script for Ubuntu 24.04
# This script installs the Intel compute runtime, Level Zero, and PyTorch with XPU support
#
# Prerequisites:
#   - Ubuntu 24.04
#   - Intel Arc GPU (B580, A770, etc.) or Intel Data Center GPU
#   - Python 3.12+ with uv or pip
#
# Usage:
#   chmod +x scripts/setup_intel_xpu.sh
#   ./scripts/setup_intel_xpu.sh
#
# After installation:
#   - Log out and back in (or run: newgrp render)
#   - Test with: sg render -c "python scripts/test_groot_xpu.py"

set -e

echo "======================================"
echo "Intel XPU Setup for Ubuntu 24.04"
echo "======================================"

# Check if running as root for system packages
check_root() {
    if [ "$EUID" -ne 0 ]; then
        echo "Some commands require sudo. You may be prompted for your password."
        SUDO="sudo"
    else
        SUDO=""
    fi
}

# Add Intel GPU repository
setup_intel_repo() {
    echo ""
    echo "[1/5] Setting up Intel GPU repository..."

    # Install prerequisites
    $SUDO apt-get update
    $SUDO apt-get install -y gpg-agent wget

    # Add Intel GPG key
    wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
        $SUDO gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg

    # Add Intel repository for Ubuntu 24.04
    echo "deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu noble client" | \
        $SUDO tee /etc/apt/sources.list.d/intel-gpu-noble.list > /dev/null

    $SUDO apt-get update

    echo "✓ Intel GPU repository configured"
}

# Install Intel compute runtime and Level Zero
install_compute_runtime() {
    echo ""
    echo "[2/5] Installing Intel compute runtime and Level Zero..."

    # Check if Kobuk PPA is already providing Level Zero (newer versions)
    if dpkg -l | grep -q "libze1"; then
        echo "Level Zero already installed from PPA"
        dpkg -l | grep -E "libze|intel-opencl"
    else
        # Install compute runtime packages from Intel repo
        $SUDO apt-get install -y \
            intel-opencl-icd \
            libze1 \
            libze-dev \
            libze-intel-gpu1
    fi

    # Optional: Install additional media packages if available
    $SUDO apt-get install -y \
        intel-media-va-driver-non-free \
        libmfx1 \
        libmfxgen1 \
        libvpl2 2>/dev/null || true

    echo "✓ Intel compute runtime installed"
}

# Add user to render group
setup_user_groups() {
    echo ""
    echo "[3/5] Setting up user permissions..."

    CURRENT_USER=${SUDO_USER:-$USER}

    # Add user to render and video groups
    if ! groups $CURRENT_USER | grep -q '\brender\b'; then
        $SUDO usermod -aG render $CURRENT_USER
        echo "Added $CURRENT_USER to 'render' group"
    fi

    if ! groups $CURRENT_USER | grep -q '\bvideo\b'; then
        $SUDO usermod -aG video $CURRENT_USER
        echo "Added $CURRENT_USER to 'video' group"
    fi

    echo "✓ User permissions configured"
    echo "  NOTE: You may need to log out and back in for group changes to take effect"
}

# Install PyTorch with XPU support
install_pytorch_xpu() {
    echo ""
    echo "[4/5] Installing PyTorch with XPU support..."

    # Check if uv is available
    if command -v uv &> /dev/null; then
        echo "Using uv for package installation..."

        # Install PyTorch XPU from Intel's index
        uv pip install torch torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/xpu

    elif command -v pip &> /dev/null; then
        echo "Using pip for package installation..."

        pip install torch torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/xpu
    else
        echo "ERROR: Neither uv nor pip found. Please install a package manager."
        exit 1
    fi

    echo "✓ PyTorch XPU installed"
}

# Verify installation
verify_installation() {
    echo ""
    echo "[5/5] Verifying installation..."

    echo ""
    echo "Checking Level Zero devices:"
    if command -v sycl-ls &> /dev/null; then
        sycl-ls
    else
        echo "  sycl-ls not available, checking via Python..."
    fi

    echo ""
    echo "Checking PyTorch XPU support:"
    python3 << 'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")

if hasattr(torch, 'xpu'):
    print(f"XPU available: {torch.xpu.is_available()}")
    if torch.xpu.is_available():
        print(f"XPU device count: {torch.xpu.device_count()}")
        for i in range(torch.xpu.device_count()):
            props = torch.xpu.get_device_properties(i)
            print(f"  Device {i}: {props.name}")
            print(f"    Total memory: {props.total_memory / 1024**3:.2f} GB")
    else:
        print("  XPU is not available. Possible reasons:")
        print("  - Level Zero runtime not installed")
        print("  - User not in 'render' group (try: newgrp render)")
        print("  - Intel GPU driver not loaded")
else:
    print("XPU support not available in this PyTorch build")
    print("Make sure you installed from the XPU index:")
    print("  pip install torch --index-url https://download.pytorch.org/whl/xpu")
EOF

    echo ""
    echo "======================================"
    echo "Setup complete!"
    echo "======================================"
    echo ""
    echo "If XPU is not detected, try:"
    echo "  1. Log out and back in (for group membership changes)"
    echo "  2. Run: newgrp render"
    echo "  3. Reboot the system"
    echo ""
}

# Main
check_root
setup_intel_repo
install_compute_runtime
setup_user_groups
install_pytorch_xpu
verify_installation
