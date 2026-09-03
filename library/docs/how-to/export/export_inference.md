# Export & Inference Guide

Export trained policies and deploy them to production.

## Quick Start

```python test="skip" reason="requires checkpoint"
from physicalai.policies.act import ACT
from physicalai.inference import InferenceModel

# Train (or load checkpoint)
policy = ACT.load_from_checkpoint("checkpoints/best.ckpt")

# Export
policy.export("./exports", backend="openvino")

# Deploy
model = InferenceModel("./exports")
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
from physicalai.policies import ACT

policy = ACT.load_from_checkpoint("checkpoints/best.ckpt")
policy.export("./exports", backend="openvino")
```

The same export contract is available from the shared CLI host:

```bash
physicalai export \
    --policy physicalai.policies.ACT \
    --ckpt_path checkpoints/best.ckpt \
    --backend openvino \
    --output_dir ./exports
```

**Output structure:**

```text
exports/
├── model.xml / model.onnx / model.pt
├── metadata.yaml
└── metadata.json
```

## Post-Export Hooks

`export(...)` accepts `post_export_hooks`, a list of callables run after the model
is written. Each hook receives the exported model file path (a `str`) and can
apply post-processing such as quantization or weight compression in place.

```python test="skip" reason="requires checkpoint"
def my_hook(model_path: str) -> None:
    ...  # inspect or rewrite the exported artifact at model_path

policy.export("./exports", backend="openvino", post_export_hooks=[my_hook])
```

### INT8 Weight Compression (OpenVINO)

`compress_weights_openvino_int8_sym` is a built-in hook that compresses an
exported OpenVINO IR to INT8 symmetric weights using
[NNCF](https://github.com/openvinotoolkit/nncf), shrinking the model and
speeding up inference on Intel hardware. It rewrites the `.xml`/`.bin` in place.

Install the optional dependency first:

```bash
pip install physicalai-train[nncf]
```

Then pass the hook to `export(...)`:

```python test="skip" reason="requires checkpoint and nncf"
from physicalai.policies import Pi05
from physicalai.export import compress_weights_openvino_int8_sym

policy = Pi05.load_from_checkpoint("checkpoints/best.ckpt")
policy.export(
    "./exports",
    backend="openvino",
    post_export_hooks=[compress_weights_openvino_int8_sym],
)
```

The exported `model.xml` / `model.bin` are replaced with their INT8-compressed
counterparts and load transparently through `InferenceModel("./exports")`.

## Inference

```python test="skip" reason="requires exported model and environment"
from physicalai.inference import InferenceModel

# Load (auto-detects backend)
policy = InferenceModel("./exports")

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

policy = InferenceModel("./exports")
policy.reset()

start = time.time()
for _ in range(1000):
    action = policy.select_action(obs)
print(f"{(time.time()-start)/1000*1000:.2f}ms per action")
```

## See Also

- [Inference Design](../../explanation/inference/README.md) - Architecture details
- [Export Design](../../explanation/export/README.md) - Export system design
- [CLI Guide](../training/cli.md) - Training via command line
