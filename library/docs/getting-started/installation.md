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

## Next Steps

- [Quickstart](quickstart.md) - Train your first policy in 5 minutes
- [First Benchmark](first-benchmark.md) - Evaluate your trained policy
- [First Deployment](first-deployment.md) - Export and deploy to production
