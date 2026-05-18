# Installation

This guide covers all installation methods for PhysicalAI Library.

## Quick Install

```bash
pip install physicalai-train
```

That's it! You're ready to [train your first policy](quickstart.md).

## Prerequisites

### Python

PhysicalAI requires **Python 3.12 or higher**.

Check your version:

```bash
python --version
```

### FFMPEG

FFMPEG is required for video processing (used by LeRobot datasets):

```bash
# Ubuntu/Debian
sudo apt-get install -y ffmpeg

# macOS
brew install ffmpeg

# Windows (with Chocolatey)
choco install ffmpeg
```

Verify installation:

```bash
ffmpeg -version
```

## Installation Methods

### Method 1: pip (Recommended)

For most users, pip install is the simplest option:

```bash
pip install physicalai-train
```

To install with specific backend support:

```bash
# With PI0 policy support
pip install physicalai-train[pi0]

# With SmolVLA policy support
pip install physicalai-train[smolvla]

# With all optional dependencies
pip install physicalai-train[all]
```

### Method 2: From Source (Development)

For contributors or users who need the latest features:

```bash
# Clone repository
git clone https://github.com/open-edge-platform/physical-ai-studio.git
cd physical-ai-studio/library

# Create virtual environment with uv (recommended)
uv venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install with all development dependencies
uv sync --all-extras
```

## Verify Installation

Run a quick test to ensure everything is working:

```python test="skip" reason="requires full physicalai install with dependencies"
import physicalai
print(physicalai.__version__)

# Test imports
from physicalai.policies import ACT
from physicalai.data import LeRobotDataModule
from physicalai.train import Trainer

print("Installation successful!")
```

Or from the command line:

```bash
physicalai --help
```

You should see the CLI help menu with available commands.

## GPU Support

PhysicalAI uses PyTorch Lightning, which automatically detects and uses available GPUs.

### Intel GPUs

Ensure you have the correct XPU version for your PyTorch installation:

```bash
# Check PyTorch XPU support
python -c "import torch; print(f'XPU available: {torch.xpu.is_available()}')"
```

### NVIDIA GPUs

Ensure you have the correct CUDA version for your PyTorch installation:

```bash
# Check PyTorch CUDA support
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Troubleshooting

### ImportError: No module named 'physicalai'

Ensure you're in the correct virtual environment:

```bash
which python  # Should point to your venv
pip list | grep physicalai
```

### FFMPEG not found

LeRobot datasets require FFMPEG. Install it using your system package manager (see Prerequisites above).

### XPU/CUDA out of memory

Reduce batch size in your training config:

```bash
physicalai fit --config your_config.yaml --data.train_batch_size 8
```

### Permission errors on Linux

If you encounter permission issues with pip:

```bash
pip install --user physicalai
```

Or use a virtual environment (recommended).

### LIBERO Installation Fails (CMake Error)

**Problem:** When installing `hf-libero` (required for LIBERO benchmarks), you see:

```
CMake Error at CMakeLists.txt:1 (cmake_minimum_required):
  Compatibility with CMake < 3.5 has been removed from CMake.
```

**Cause:** The `egl-probe` package (dependency of hf-libero) requests CMake 2.8.12, but modern CMake (3.5+) refuses to build projects requesting versions < 3.5.

**Solution:**

First, install system dependencies:

```bash
# Ubuntu/Debian
sudo apt-get install -y cmake libclang-dev libegl1-mesa-dev \
  libgl1-mesa-dev libgles2-mesa-dev libglew-dev libglfw3-dev \
  libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev

# macOS
brew install cmake llvm glew glfw
```

Then install egl-probe from patched source:

```bash
# Download and patch egl-probe
cd /tmp
pip download --no-binary :all: egl-probe
tar xzf egl_probe-*.tar.gz
cd egl_probe-*/

# Patch CMakeLists.txt to use CMake 3.5+
sed -i 's/cmake_minimum_required(VERSION 2\.8\.12)/cmake_minimum_required(VERSION 3.5)/g' \
  egl_probe/CMakeLists.txt

# Install from patched source
pip install .

# Clean up
cd / && rm -rf /tmp/egl_probe*
```

Finally, install hf-libero:

```bash
pip install hf-libero
```

**Affected components:**
- LIBERO benchmark notebooks
- Backend accuracy comparison tools
- Any workflow requiring `hf-libero`

This is a known upstream issue and will be fixed when egl-probe updates its CMake requirements.

## Next Steps

- [Quickstart](quickstart.md) - Train your first policy in 5 minutes
- [First Benchmark](first-benchmark.md) - Evaluate your trained policy
- [First Deployment](first-deployment.md) - Export and deploy to production
