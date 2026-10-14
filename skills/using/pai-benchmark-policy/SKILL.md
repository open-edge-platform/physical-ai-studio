---
name: pai-benchmark-policy
description: Benchmarks a trained Physical AI Studio policy in a simulation gym and reports success metrics. Use when running physicalai benchmark, selecting a benchmark config for a checkpoint or export, tuning episode/env settings, recording videos, or interpreting results.json and results.csv. DO NOT USE FOR implementing a new Benchmark class in Studio source.
license: Apache-2.0
---

# Benchmarking a Studio Policy

Benchmarking evaluates a trained policy by rolling it out in a gym and scoring success. `PushTBenchmark` and `LiberoBenchmark` are available through the Python API, alongside `BenchmarkResults` and `TaskResult`. Use an installed `physicalai-train` package or the [Studio benchmark examples](https://github.com/open-edge-platform/physical-ai-studio/tree/main/library/configs/benchmark) for CLI configs.

## Python API invocation

Use this path for notebooks, tests, custom scripts, or direct library integrations.

```python
from physicalai.benchmark.gyms import PushTBenchmark
from physicalai.policies import ACT

policy = ACT.load_from_checkpoint("experiments/act/version_0/checkpoints/last.ckpt")
benchmark = PushTBenchmark(num_episodes=1)
results = benchmark.evaluate(policy)
print(results.summary())
results.to_json("results/benchmark/results.json")
results.to_csv("results/benchmark/results.csv")
```

For exported artifacts, load the Runtime-facing model first:

```python
from physicalai.benchmark.gyms import PushTBenchmark
from physicalai.inference import InferenceModel

model = InferenceModel("./exports/act_policy")
results = PushTBenchmark(num_episodes=1).evaluate(model)
```

## CLI invocation

```bash
physicalai benchmark \
  --config <benchmark-config.yaml> \
  --policy physicalai.policies.ACT \
  --ckpt_path experiments/act/version_0/checkpoints/last.ckpt \
  --output_dir ./results/benchmark
```

- `--policy` — policy class path.
- `--ckpt_path` — a `.ckpt` **or** an export directory.
- `--config` — a benchmark config copied into your project from the Studio examples, selecting the `Benchmark` class and its settings.
- `--output_dir` — defaults to `./results/benchmark`.

Override benchmark settings on the CLI, e.g. `--benchmark.num_episodes 10 --benchmark.num_envs 8`.

## Output

- Prints `results.summary()` to stdout.
- Writes `results.json` and `results.csv` into `--output_dir`.
- Optional video via config `video_dir` + `record_mode` (`all` | `failures` | `successes` | `none`).

## Workflow

1. **Choose API or CLI deliberately.** Use the Python API for code-level tasks; use CLI for config/docs/entry-point tasks.
   - Done when: the selected path matches the user's requested surface area.
2. **Confirm the policy loads** from the checkpoint/export before a full sweep:
   ```bash
   physicalai benchmark --config <benchmark-config.yaml> --policy <ClassPath> --ckpt_path <path> --benchmark.num_episodes 1
   ```
   - Done when: one episode runs end-to-end and a summary prints.
3. **Run the full benchmark** with the intended episode/env counts.
   - Done when: `results.json` and `results.csv` are written and the success metric is populated.
4. **Interpret results** via `BenchmarkResults`/`TaskResult` fields; compare against a baseline checkpoint on the same config.
5. **Record videos** for qualitative review when a task regresses (`record_mode: failures`).

## Required checks

- The policy runs from **both** a `.ckpt` and an export dir if both are supported paths.
- The Python API path (`Benchmark(...).evaluate(...)`) and CLI wrapper agree on supported inputs for the run.
- Success/episode metrics are populated (not zero/NaN by accident) and reproducible across runs.
- Env/episode counts match hardware; large `num_envs` fits memory.
- Install the optional gym extra needed by the chosen environment before running it.

## Related skills

- `pai-train-policy` — to produce the checkpoint being benchmarked.
- `pai-export-policy` — when benchmarking an exported artifact for deployment parity.
