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

| Model                                                   | Backend | Spatial | Object |  Goal | LIBERO-10 | Average | Harness version |
| ------------------------------------------------------- | ------- | ------: | -----: | ----: | --------: | ------: | --------------: |
| [`lerobot/pi05_libero_finetuned`](#pi05-libero-pytorch) | PyTorch |   97.8% |  99.6% | 96.8% |     95.8% |   97.5% |           0.4.0 |

The four-suite average is the arithmetic mean of the suite success rates.

### LIBERO-Plus Results

The LIBERO-Plus protocol evaluates the same four suites with one episode for
each task entry, for 10,030 episodes in total. The suites are kept separate
because each one is substantially longer than its standard LIBERO counterpart.

| Model                                                        | Backend | Spatial | Object |   Goal | LIBERO-10 | Average | Harness version |
| ------------------------------------------------------------ | ------- | ------: | -----: | -----: | --------: | ------: | --------------: |
| [`lerobot/pi05_libero_finetuned`](#pi05-libero-plus-pytorch) | PyTorch |  88.05% | 87.49% | 80.70% |    81.02% |  84.32% |           0.4.0 |

The four-suite average is the arithmetic mean of the suite success rates. The
results contain 2,402 Spatial, 2,518 Object, 2,591 Goal, and 2,519 LIBERO-10
episodes.

## Installation

Docker is required for the LIBERO and LIBERO-Plus environments. From
`library/`, install the policy dependencies and published harness using the
backend extra appropriate for the machine:

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
├── physicalai_pi05_libero.yaml            # standard LIBERO PyTorch policy
├── physicalai_pi05_libero_plus.yaml       # LIBERO-Plus PyTorch policy
├── physicalai_pi05_libero_openvino.yaml   # optional exported OpenVINO example
├── physicalai_pi05_libero_torch.yaml      # optional exported Torch example
├── benchmarks/libero/
│   ├── smoke_test.yaml                     # one task, one episode
│   ├── 10.yaml                             # LIBERO-10, 500 episodes
│   └── libero.yaml                         # all standard suites, 2,000 episodes
└── benchmarks/libero-plus/
  ├── spatial.yaml                        # Spatial, 2,402 episodes
  ├── object.yaml                         # Object, 2,518 episodes
  ├── goal.yaml                           # Goal, 2,591 episodes
  └── 10.yaml                             # LIBERO-10, 2,519 episodes
```

The exported-model configs are retained as examples but are not part of the
current supported results matrix. Standard LIBERO sends policy state as
`observation.state`; LIBERO-Plus sends it as `states`, so the live policy has a
separate model-server config for each protocol.

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

### Pi05 LIBERO-Plus PyTorch

Start the Pi05 finetuned policy server:

```bash
python model_servers/physicalai.py \
  --config configs/physicalai_pi05_libero_plus.yaml
```

Run each long suite independently in a second terminal. Each command launches
four simulator shards and merges that suite's shared recording when all shards
complete:

```bash
./shard.sh \
  --config configs/benchmarks/libero-plus/spatial.yaml \
  --shards 4

./shard.sh \
  --config configs/benchmarks/libero-plus/object.yaml \
  --shards 4

./shard.sh \
  --config configs/benchmarks/libero-plus/goal.yaml \
  --shards 4

./shard.sh \
  --config configs/benchmarks/libero-plus/10.yaml \
  --shards 4
```

The configs use separate output directories, so suites can be scheduled and
rerun independently. Run multiple suites concurrently only if the model server
and benchmark host can sustain their combined inference and simulator load.

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
