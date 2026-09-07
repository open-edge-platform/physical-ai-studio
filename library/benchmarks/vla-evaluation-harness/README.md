# VLA Evaluation Harness

This integration evaluates Physical AI Studio policies with AllenAI's
[`vla-evaluation-harness`](https://github.com/allenai/vla-evaluation-harness).
The policy runs on the host through `PhysicalAIModelServer`; the harness runs
benchmark simulators in isolated Docker containers and exchanges observations
and actions with the server over WebSocket.

## Contents

- [Results](#results)
- [Installation](#installation)
- [Result Artifacts](#result-artifacts)
- [Configs](#configs)
- [Model Servers](#model-servers)
- [Reproduction](#reproduction)
- [Sharding](#sharding)

## Results

### LIBERO Results

The standard protocol evaluates Spatial, Object, Goal, and LIBERO-10 with 10
tasks per suite and 50 episodes per task.

| Model                                                   | Backend | Spatial | Object |  Goal | LIBERO-10 | Average |       Runtime |
| ------------------------------------------------------- | ------- | ------: | -----: | ----: | --------: | ------: | ------------: |
| [`lerobot/pi05_libero_finetuned`](#pi05-libero-pytorch) | PyTorch |   97.8% |  99.6% | 96.8% |     95.8% |   97.5% | PyTorch Eager |

The four-suite average is the arithmetic mean of the suite success rates.

## Installation

Docker is required for the LIBERO environment. From `library/`, install the
policy dependencies and published harness using the backend extra appropriate
for the machine:

```bash
uv sync --extra cu128 --extra pi05
uv pip install vla-eval
source .venv/bin/activate
```

Run the remaining commands from `library/benchmarks/vla-evaluation-harness`.
The examples use `uv run --no-sync` because `uv run` may resynchronize
the project without the selected accelerator extra.

## Result Artifacts

Results are written below the config's `output_dir`, or the directory supplied
with `--output-dir`. Recording creates `recording-<eval-id>.sqlite`; merge
materializes per-episode JSONL and aggregate JSON files.

## Configs

Model-server configs describe policy construction and benchmark-to-policy
field mapping. Benchmark configs describe simulator tasks and episode counts.

```text
configs/
├── physicalai_pi05_libero.yaml          # supported live PyTorch policy
├── physicalai_pi05_libero_openvino.yaml # optional exported OpenVINO example
├── physicalai_pi05_libero_torch.yaml    # optional exported Torch example
└── benchmarks/libero/
    ├── smoke_test.yaml                   # one task, one episode
    ├── 10.yaml                           # LIBERO-10, 500 episodes
    └── libero.yaml                       # all standard suites, 2,000 episodes
```

The exported-model configs are retained as examples but are not part of the
current supported results matrix.

## Model Servers

```text
model_servers/
├── physicalai.py  # reusable, config-driven Physical AI Studio adapter
└── libero_pi05.py # optional subclass with maintained LIBERO defaults
```

Use `physicalai.py` whenever policy construction, camera mapping, state
mapping, device placement, and action chunking fit in YAML. Add or use a
subclass only when custom loading or protocol behavior cannot be represented
by the general config.

The adapter remains outside the `physicalai` package so vla-eval and simulator
dependencies do not become part of the public package API.

## Reproduction

### Pi05 LIBERO PyTorch

Model server:

```bash
python model_servers/physicalai.py \
  --config configs/physicalai_pi05_libero.yaml
```

Benchmark:

```bash
uv run --no-sync vla-eval run \
  --config configs/benchmarks/libero/libero.yaml
```

## Sharding

Run the same benchmark across four parallel simulator processes:

```bash
./shard.sh \
  --config configs/benchmarks/libero/libero.yaml \
  --shards 4 \
  --output-dir results/pi05-pytorch-libero
```

`shard.sh` assigns episodes to shards, gives every shard the same evaluation
ID, waits for completion, and merges the results. Use `--eval-id <id>` to set
the evaluation ID explicitly.

`PhysicalAIModelServer` currently implements `predict()`, not
`predict_batch()`. Sharding parallelizes simulator work, but inference requests
are not batchified, so the shared model server may limit scaling.
