---
name: physicalai-train-benchmarking-a-policy
description: Benchmarks a trained Physical AI Studio policy in a simulation gym and reports success metrics. Use when running physicalai benchmark, editing configs under library/configs/benchmark, adding or changing a Benchmark class in physicalai.benchmark, tuning rollout/episode/env settings, recording rollout videos, or interpreting results.json / results.csv.
license: Apache-2.0
---

# Benchmarking a Studio Policy

Benchmarking evaluates a trained policy by rolling it out in a gym and scoring success. Benchmark classes live in `library/src/physicalai/benchmark/gyms/benchmark.py` (`Benchmark`, `PushTBenchmark`, `LiberoBenchmark`); results types in `benchmark/gyms/results.py` (`BenchmarkResults`, `TaskResult`); rollout logic in `library/src/physicalai/eval/rollout.py` (`evaluate_policy`). The library supports both direct Python API use and the `physicalai benchmark` CLI wrapper (`library/src/physicalai/cli/benchmark.py`).

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
  --config configs/benchmark/pusht.yaml \
  --policy physicalai.policies.ACT \
  --ckpt_path experiments/act/version_0/checkpoints/last.ckpt \
  --output_dir ./results/benchmark
```

- `--policy` — policy class path.
- `--ckpt_path` — a `.ckpt` **or** an export directory.
- `--config` — a benchmark config (`configs/benchmark/pusht.yaml`, `configs/benchmark/libero.yaml`) selecting the `Benchmark` class and its settings.
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
   physicalai benchmark --config configs/benchmark/<suite>.yaml --policy <ClassPath> --ckpt_path <path> --benchmark.num_episodes 1
   ```
   - Done when: one episode runs end-to-end and a summary prints.
3. **Run the full benchmark** with the intended episode/env counts.
   - Done when: `results.json` and `results.csv` are written and the success metric is populated.
4. **Interpret results** via `BenchmarkResults`/`TaskResult` fields; compare against a baseline checkpoint on the same config.
5. **Record videos** for qualitative review when a task regresses (`record_mode: failures`).

## Adding or changing a Benchmark

1. Subclass `Benchmark` in `benchmark/gyms/` (study `PushTBenchmark` / `LiberoBenchmark`); the gym itself comes from `physicalai.gyms` (`pusht.py`, `libero.py`, …).
2. Add a matching config in `library/configs/benchmark/`.
3. Add tests under `library/tests/unit/benchmark/`.
   - Done when: `uv run pytest tests/unit/benchmark` passes and a 1-episode run succeeds.

## Required checks

- The policy runs from **both** a `.ckpt` and an export dir if both are supported paths.
- The Python API path (`Benchmark(...).evaluate(...)`) and CLI wrapper agree on supported inputs for user-facing benchmark changes.
- Success/episode metrics are populated (not zero/NaN by accident) and reproducible across runs.
- Env/episode counts match hardware; large `num_envs` fits memory.
- Heavy gym deps (e.g. `libero`, `robocasa`) are gated behind their optional extras and imported lazily.

## Related skills

- `physicalai-train-training-a-policy` — to produce the checkpoint being benchmarked.
- `physicalai-train-exporting-and-validating` — when benchmarking an exported artifact for deployment parity.
