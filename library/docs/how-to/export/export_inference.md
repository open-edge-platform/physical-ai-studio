# Export & Inference Guide

Export trained policies and deploy them to production.

## Quick Start

```python test="skip" reason="requires checkpoint"
from getiaction.policies.act import ACT
from getiaction.inference import InferenceModel

# Train (or load checkpoint)
policy = ACT.load_from_checkpoint("checkpoints/best.ckpt")

# Export
policy.export("./exports", backend="openvino")

# Deploy
model = InferenceModel.load("./exports")
action = model.select_action(observation)
```

## Backends

| Backend             | Best For                     | Install                |
| ------------------- | ---------------------------- | ---------------------- |
| **OpenVINO**        | Intel hardware (CPU/GPU/NPU) | `pip install openvino` |
| **ONNX**            | Cross-platform               | `pip install onnx`     |
| **Torch Export IR** | Edge/mobile devices          | Built-in               |

## Export

```python test="skip" reason="requires checkpoint"
from getiaction.policies import ACT

policy = ACT.load_from_checkpoint("checkpoints/best.ckpt")
policy.export("./exports", backend="openvino")
```

**Output structure:**

```text
exports/
├── model.xml / model.onnx / model.pt
├── metadata.yaml
└── metadata.json
```

## Inference

```python test="skip" reason="requires exported model and environment"
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

## Performance Tips

1. **Match backend to hardware** - OpenVINO for Intel, ONNX for NVIDIA
2. **Use action queuing** - Chunked policies return multiple actions per inference
3. **Warm-up model** - First inference is slower due to compilation
4. **Reuse policy instance** - Avoid loading model repeatedly

### Benchmarking Latency

```python test="skip" reason="requires exported model"
import time

policy = InferenceModel.load("./exports")
policy.reset()

start = time.time()
for _ in range(1000):
    action = policy.select_action(obs)
print(f"{(time.time()-start)/1000*1000:.2f}ms per action")
```

## See Also

- [Inference Design](../design/inference/README.md) - Architecture details
- [Export Design](../design/export/README.md) - Export system design
