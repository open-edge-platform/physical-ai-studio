# Inference Module

Production-ready inference API for exported GetiAction policies.

## Overview

The inference module provides a unified, production-ready interface for running robot policies across different backends (OpenVINO, ONNX, TorchScript). It automatically handles backend detection, device selection, and maintains API compatibility with PyTorch training policies.

## Key Features

- **Unified API**: Same interface regardless of backend
- **Auto-detection**: Automatically detects backend, device, and policy configuration
- **Stateful Policies**: Handles action queues and episode resets
- **Multi-backend**: Supports OpenVINO, ONNX, TorchScript
- **Production-ready**: Optimized for deployment on edge devices and servers

## Quick Start

### Basic Usage

```python
from getiaction.inference import InferenceModel

# Load exported policy (auto-detects everything)
policy = InferenceModel.load("./exports/act_policy")

# Use exactly like PyTorch policy
policy.reset()  # Start new episode
action = policy.select_action(observation)  # Get action
```

### Explicit Configuration

```python
# Specify backend and device explicitly
policy = InferenceModel(
    export_dir="./exports",
    policy_name="act",
    backend="openvino",  # or "onnx", "torch"
    device="CPU"  # or "GPU", "cuda", etc.
)
```

## Architecture

### Module Structure

```text
inference/
├── __init__.py          # Public API
├── model.py             # InferenceModel class
├── adapters/            # Backend adapters
│   ├── __init__.py      # Adapter factory
│   ├── base.py          # Base adapter interface
│   ├── openvino.py      # OpenVINO adapter
│   ├── onnx.py          # ONNX Runtime adapter
│   └── torchscript.py   # TorchScript adapter
└── README.md            # This file
```

### Design Principles

1. **Separation of Concerns**: Backend-specific logic is isolated in adapters
2. **Factory Pattern**: `get_adapter()` creates appropriate adapter based on backend
3. **Adapter Pattern**: All backends implement common `RuntimeAdapter` interface
4. **Auto-detection**: Smart defaults minimize configuration needs

## API Reference

### InferenceModel

Main class for production inference.

#### Constructor

```python
InferenceModel(
    export_dir: str | Path,
    policy_name: str | None = None,
    backend: str | ExportBackend | "auto" = "auto",
    device: str = "auto",
    **adapter_kwargs
)
```

**Parameters:**

- `export_dir`: Directory containing exported policy files
- `policy_name`: Policy name (auto-detected from files if None)
- `backend`: Backend to use ("openvino", "onnx", "torch", or "auto")
- `device`: Device for inference (backend-specific, or "auto")
- `**adapter_kwargs`: Backend-specific options

#### Class Method: load()

```python
@classmethod
def load(cls, export_dir: str | Path, **kwargs) -> InferenceModel
```

Convenience method that auto-detects all parameters.

#### Method: select_action()

```python
def select_action(self, observation: Observation) -> torch.Tensor
```

Select action for given observation. Matches PyTorch policy API.

**For chunked policies (chunk_size > 1):**

- Automatically manages action queue
- Returns one action at a time
- Only calls model when queue is empty

**Returns:**

- Action tensor: `(batch_size, action_dim)` or `(action_dim,)`

#### Method: reset()

```python
def reset(self) -> None
```

Reset policy state for new episode. Clears action queues and internal state.

**Always call at episode start!**

### Backend Adapters

All adapters implement the `RuntimeAdapter` interface:

```python
class RuntimeAdapter(ABC):
    def load(self, model_path: Path) -> None
    def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]

    @property
    def input_names(self) -> list[str]

    @property
    def output_names(self) -> list[str]
```

#### OpenVINOAdapter

**Device options:** `"CPU"`, `"GPU"`, `"NPU"`, `"AUTO"`

```python
from getiaction.inference.adapters import OpenVINOAdapter

adapter = OpenVINOAdapter(device="CPU")
adapter.load(Path("model.xml"))
outputs = adapter.predict(inputs)
```

#### ONNXAdapter

**Device options:** `"cpu"`, `"cuda"`, `"tensorrt"`

```python
from getiaction.inference.adapters import ONNXAdapter

adapter = ONNXAdapter(device="cuda")
adapter.load(Path("model.onnx"))
outputs = adapter.predict(inputs)
```

#### TorchScriptAdapter

**Device options:** `"cpu"`, `"cuda"`

```python
from getiaction.inference.adapters import TorchScriptAdapter

adapter = TorchScriptAdapter(device="cuda")
adapter.load(Path("model.pt"))
outputs = adapter.predict(inputs)
```

## Usage Examples

### Episode Loop

```python
from getiaction.inference import InferenceModel

# Load policy
policy = InferenceModel.load("./exports/act_policy")

# Run episodes
for episode in range(num_episodes):
    policy.reset()  # Important: reset at episode start
    obs = env.reset()
    done = False

    while not done:
        action = policy.select_action(obs)
        obs, reward, done = env.step(action)
```

### Backend Selection

```python
# OpenVINO (best for Intel hardware)
policy = InferenceModel.load("./exports", backend="openvino", device="CPU")

# ONNX (cross-platform)
policy = InferenceModel.load("./exports", backend="onnx", device="cuda")

# TorchScript (PyTorch ecosystem)
policy = InferenceModel.load("./exports", backend="torch", device="cuda")
```

### Chunked Policies

```python
# ACT with chunk_size=100
policy = InferenceModel.load("./exports/act_policy")

# Action queue is managed automatically
policy.reset()
for t in range(1000):
    action = policy.select_action(obs)  # Returns single action
    obs, reward, done = env.step(action)

    if done:
        policy.reset()  # Clear action queue
        obs = env.reset()
```

## Auto-Detection

The inference module automatically detects:

### 1. Backend

From file extensions in export directory:

- `.xml` → OpenVINO
- `.onnx` → ONNX
- `.pt` → TorchScript

Or from `metadata.yaml`:

```yaml
backend: openvino
```

### 2. Device

**OpenVINO:** Defaults to `"CPU"` (most compatible)

**ONNX/TorchScript:**

- Uses `"cuda"` if available
- Falls back to `"cpu"`

### 3. Policy Configuration

From `metadata.yaml`:

```yaml
backend: openvino
chunk_size: 100
use_action_queue: true
policy_class: getiaction.policies.act.ACT
```

## Metadata Format

Export creates `metadata.yaml` with policy configuration:

```yaml
backend: openvino # Backend used for export
chunk_size: 100 # Action chunk size
use_action_queue: true # Whether to use action queue
policy_class: getiaction.policies.act.ACT # Policy class path
n_action_steps: 100 # Number of action steps (optional)
```

## Performance Tips

### OpenVINO

- **CPU**: Best compatibility, good performance on Intel CPUs
- **GPU**: Faster inference on Intel integrated/discrete GPUs
- **NPU**: Ultra-low latency on Intel NPUs (e.g., Meteor Lake)
- **AUTO**: Let OpenVINO choose best device

```python
# Optimize for throughput
policy = InferenceModel.load(
    "./exports",
    backend="openvino",
    device="CPU",
    performance_hint="throughput"  # or "latency"
)
```

### ONNX Runtime

- **CPU**: Use for development/testing
- **CUDA**: GPU acceleration
- **TensorRT**: Optimized for NVIDIA GPUs

```python
# Use TensorRT for best NVIDIA GPU performance
policy = InferenceModel.load(
    "./exports",
    backend="onnx",
    device="tensorrt"
)
```

### TorchScript

- Generally slower than OpenVINO/ONNX
- Use for PyTorch ecosystem integration
- Best for rapid prototyping

## Error Handling

```python
from getiaction.inference import InferenceModel

try:
    policy = InferenceModel.load("./exports")
except FileNotFoundError as e:
    print(f"Export directory not found: {e}")
except ValueError as e:
    print(f"Cannot detect backend or policy: {e}")
except ImportError as e:
    print(f"Backend runtime not installed: {e}")
```

## Testing

Run tests with:

```bash
uv run pytest tests/unit/inference/
```

## Dependencies

### Core

- `torch`: For tensor operations
- `numpy`: For array operations
- `pyyaml`: For metadata parsing

### Backend-specific (optional)

- `openvino`: For OpenVINO backend
- `onnxruntime` or `onnxruntime-gpu`: For ONNX backend
- `torch`: Already included for TorchScript

Install with:

```bash
# OpenVINO
uv pip install openvino

# ONNX (CPU)
uv pip install onnxruntime

# ONNX (GPU)
uv pip install onnxruntime-gpu
```

## See Also

- [Export Best Practices](../../../tmp/inference_api_docs/EXPORT_BEST_PRACTICES.md)
- [Model Architecture](../../../tmp/inference_api_docs/MODEL_ARCHITECTURE.md)
- [Complete Workflow](../../../tmp/inference_api_docs/WORKFLOW.md)
