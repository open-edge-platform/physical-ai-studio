# VLA Evaluation Harness

This integration benchmarks Physical AI Studio policies with AllenAI's
[`vla-evaluation-harness`](https://github.com/allenai/vla-evaluation-harness).
The model server runs in the Physical AI Studio environment; the harness
isolates benchmark simulators, normally in Docker.

## Prerequisite

Docker is required to run the benchmark environments.

## Installation

From `library/`, install the policy dependencies and the published harness in
the same virtual environment. Select the backend extra appropriate for the
machine.

```bash
uv sync --extra cu128 --extra pi05
uv pip install vla-eval
source .venv/bin/activate
```

## Model Server

Run commands from `library/benchmarks/vla-evaluation-harness`.

The general server loads the policy declaration inline from one YAML:

```bash
python model_servers/physicalai.py \
  --config configs/physicalai_pi05_libero.yaml
```

The maintained Pi0.5/LIBERO subclass provides the same mapping without YAML:

```bash
python model_servers/libero_pi05.py --port 8000
```

Both paths expose the server at `ws://localhost:8000`. The general server is
preferred when policy construction, camera mapping, state mapping, and action
chunking can be represented in config. Add a subclass only for custom loading
or stable model-benchmark defaults.

## LIBERO Evaluation

In a second terminal, run the one-task smoke test first:

```bash
cd library/benchmarks/vla-evaluation-harness
uv run --no-sync vla-eval run --config configs/benchmarks/libero/smoke_test.yaml
```

Then run LIBERO-10 (10 tasks, 50 episodes per task):

```bash
uv run --no-sync vla-eval run --config configs/benchmarks/libero/10.yaml
```

Results are written to `results/`. `--no-sync` is required because the CUDA
extra was selected during installation; a bare `uv run` synchronizes the
project without that extra and may replace the CUDA-enabled Torch wheel.
Model-server commands continue running with plain `python` in the active
policy environment.

## Files

```text
configs/
├── physicalai_pi05_libero.yaml
├── physicalai_pi05_libero_openvino.yaml
├── physicalai_pi05_libero_torch.yaml
└── benchmarks/libero/
    ├── smoke_test.yaml
    └── 10.yaml
model_servers/
├── physicalai.py
└── libero_pi05.py
shard_libero_10.sh
```

The adapter remains outside the `physicalai` package so external harness and
simulator dependencies do not become part of the public package API.

The classes follow vla-eval's naming convention: `PhysicalAIModelServer` for
the package adapter and `LiberoPi05ModelServer` for the benchmark-specific
server. Their filenames stay concise because they already live under
`model_servers/`.

## More Examples

### OpenVINO Pi0.5 on LIBERO

1. Export the model with the OpenVINO backend from
   `library/benchmarks/vla-evaluation-harness`:

```python
from physicalai.policies.pi05 import Pi05

if __name__ == "__main__":
    model = Pi05(pretrained_name_or_path="lerobot/pi05_libero_finetuned_v044").eval()
    model.export("pi05_libero_openvino", backend="openvino")
```

1. Start the general model server with `InferenceModel`:

```bash
python model_servers/physicalai.py \
  --config configs/physicalai_pi05_libero_openvino.yaml
```

The config constructs the exported model directly:

```yaml
policy:
  class_path: physicalai.inference.InferenceModel
  init_args:
    export_dir: pi05_libero_openvino
    device: auto
```

`InferenceModel` reads the export manifest and selects the OpenVINO backend.
Set `device` to a specific OpenVINO target when needed.

1. Run LIBERO in a second terminal:

```bash
uv run --no-sync vla-eval run --config configs/benchmarks/libero/10.yaml
```

### Torch Pi0.5 on LIBERO

1. Export the model with the Torch backend from
   `library/benchmarks/vla-evaluation-harness`:

```python
from physicalai.policies.pi05 import Pi05

if __name__ == "__main__":
    model = Pi05(pretrained_name_or_path="lerobot/pi05_libero_finetuned_v044").eval()
    model.export("pi05_libero_torch", backend="torch")
```

1. Start the general model server with the exported Torch model:

```bash
python model_servers/physicalai.py \
  --config configs/physicalai_pi05_libero_torch.yaml
```

The config constructs `InferenceModel` with the Torch backend explicitly:

```yaml
policy:
  class_path: physicalai.inference.InferenceModel
  init_args:
    export_dir: pi05_libero_torch
    backend: torch
    device: auto
```

Set `device` to `cuda`, `xpu`, or `cpu` to select a specific Torch device.

1. Run LIBERO in a second terminal:

```bash
uv run --no-sync vla-eval run --config configs/benchmarks/libero/10.yaml
```

### LIBERO with Episode Sharding

Start the model server as described above. In a second terminal,
launch four evaluation shards from
`library/benchmarks/vla-evaluation-harness`:

```bash
./shard_libero_10.sh 4
```

Episodes are assigned to shards round-robin. All shards connect to the same model server and write to one evaluation recording; the final command
materializes the merged per-episode and aggregate results. The optional
argument sets the number of shards and defaults to `4`. Increase it only after
confirming the model server and benchmark host can sustain the additional
concurrent requests.

## Future Development

We aim to introduce the batchified model prediction for actions.
