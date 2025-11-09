# Export & Inference Guide

Export trained policies and deploy them to production.

## Quick Start

```python
from getiaction.policies.act import ACT
from getiaction.train import Trainer
from getiaction.inference import InferenceModel

# 1. Train
policy = ACT(...)
trainer.fit(policy, datamodule)

# 2. Export
policy.export("./exports", backend="openvino")

# 3. Inference
inference_model = InferenceModel.load("./exports")
action = inference_model.select_action(observation)
```

## Backends

| Backend             | Best For                    | Install                   | Support |
| ------------------- | --------------------------- | ------------------------- | ------- |
| **OpenVINO**        | Intel CPUs/GPUs/NPUs        | `pip install openvino`    | ✅      |
| **ONNX**            | NVIDIA GPUs, cross-platform | `pip install onnxruntime` | ✅      |
| **TorchScript**     | PyTorch ecosystem           | Built-in                  | ✅      |
| **Torch Export IR** | Edge/mobile devices         | Built-in                  | ✅      |

## Export

```python
# From checkpoint
policy = ACT.load_from_checkpoint("checkpoints/best.ckpt")
policy.export("./exports", backend="openvino")

# LeRobot policies (same API)
from getiaction.policies.lerobot import ACT
policy = ACT(...)
policy.export("./exports", backend="openvino")
```

**Output:**

```text
exports/
├── model.xml / model.onnx / model.pt
├── metadata.yaml
└── metadata.json
```

## Inference

```python
from getiaction.inference import InferenceModel

# Load (auto-detects backend)
policy = InferenceModel.load("./exports")

# Run episode
obs = env.reset()
policy.reset()
while not done:
    action = policy.select_action(obs)
    obs, reward, done, _ = env.step(action)
```

## Choosing a Backend

- **OpenVINO**: Intel hardware (CPU/GPU/NPU), edge devices
- **ONNX**: NVIDIA GPUs (TensorRT), cross-platform, cloud
- **TorchScript**: PyTorch ecosystem, mobile, debugging
- **Torch Export IR**: Edge/mobile deployment, resource-constrained devices

## Performance Tips

1. **Match backend to hardware** (OpenVINO for Intel, ONNX for NVIDIA)
2. **Use action queuing** (chunked policies run model once, return `chunk_size` actions)
3. **Warm-up model** (first inference is slower due to compilation)
4. **Reuse policy instance** (avoid loading model repeatedly)

```python
# Benchmark
import time
policy = InferenceModel.load("./exports")
policy.reset()

start = time.time()
for _ in range(1000):
    action = policy.select_action(obs)
print(f"{(time.time()-start)/1000*1000:.2f}ms per action")
```

## Troubleshooting

### Export Errors

| Error                                   | Cause                     | Solution                     |
| --------------------------------------- | ------------------------- | ---------------------------- |
| `RuntimeError: Policy not initialized`  | Exporting before training | Call `trainer.fit()` first   |
| `ValueError: Cannot create dummy input` | Missing input shapes      | Train with proper datamodule |

### Inference Errors

| Error                                 | Cause                   | Solution                                |
| ------------------------------------- | ----------------------- | --------------------------------------- |
| `ImportError: openvino not installed` | Backend missing         | `pip install openvino/onnxruntime`      |
| `ValueError: Cannot detect backend`   | No model file           | Verify export completed                 |
| `RuntimeError: Input shape mismatch`  | Wrong observation shape | Check `policy.metadata["input_shapes"]` |

### Numerical Differences

Small differences (rtol=0.15) between training and inference are normal due to backend optimizations.

## Advanced

### Compare Backends

```python
# Export and benchmark all backends
for backend in ["openvino", "onnx", "torch"]:
    policy.export(f"./{backend}", backend=backend)
    p = InferenceModel.load(f"./{backend}")
    # ... benchmark ...
```

### Access Metadata

```python
policy = InferenceModel.load("./exports")
print(policy.backend)           # ExportBackend.OPENVINO
print(policy.chunk_size)        # 100
print(policy.metadata)          # Full config dict
```

## See Also

- [Inference Design](../design/inference/overview.md) - Architecture details
- [Export Design](../design/export/README.md) - Export system design
- [CLI Guide](cli.md) - Command-line interface
