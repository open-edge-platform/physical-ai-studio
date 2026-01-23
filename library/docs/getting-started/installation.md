# Installation

This guide covers all installation methods for Geti Action Library.

## Quick Install

```bash
pip install getiaction
```

That's it! You're ready to [train your first policy](quickstart.md).

## Prerequisites

### Python

Geti Action requires **Python 3.10 or higher**.

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
pip install getiaction
```

To install with specific backend support:

```bash
# With OpenVINO export support
pip install getiaction[openvino]

# With ONNX export support
pip install getiaction[onnx]

# With all optional dependencies
pip install getiaction[all]
```

### Method 2: From Source (Development)

For contributors or users who need the latest features:

```bash
# Clone repository
git clone https://github.com/open-edge-platform/geti-action.git
cd geti-action/library

# Create virtual environment with uv (recommended)
uv venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install with all development dependencies
uv sync --all-extras
```

### Method 3: Docker

Coming soon.

## Verify Installation

Run a quick test to ensure everything is working:

```python test="skip" reason="requires full getiaction install with dependencies"
import getiaction
print(getiaction.__version__)

# Test imports
from getiaction.policies import ACT
from getiaction.data import LeRobotDataModule
from getiaction.train import Trainer

print("Installation successful!")
```

Or from the command line:

```bash
getiaction --help
```

You should see the CLI help menu with available commands.

## GPU Support

Geti Action uses PyTorch Lightning, which automatically detects and uses available GPUs.

### NVIDIA GPUs

Ensure you have the correct CUDA version for your PyTorch installation:

```bash
# Check PyTorch CUDA support
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Intel GPUs (with OpenVINO)

For inference on Intel hardware:

```bash
pip install openvino
```

## Troubleshooting

### ImportError: No module named 'getiaction'

Ensure you're in the correct virtual environment:

```bash
which python  # Should point to your venv
pip list | grep getiaction
```

### FFMPEG not found

LeRobot datasets require FFMPEG. Install it using your system package manager (see Prerequisites above).

### CUDA out of memory

Reduce batch size in your training config:

```bash
getiaction fit --config your_config.yaml --data.train_batch_size 8
```

### Permission errors on Linux

If you encounter permission issues with pip:

```bash
pip install --user getiaction
```

Or use a virtual environment (recommended).

## Next Steps

- [Quickstart](quickstart.md) - Train your first policy in 5 minutes
- [First Benchmark](first-benchmark.md) - Evaluate your trained policy
- [First Deployment](first-deployment.md) - Export and deploy to production
