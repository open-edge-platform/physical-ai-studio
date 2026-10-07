---
name: pai-add-benchmark
description: Adds or changes a first-party simulation Benchmark in the Physical AI Studio library. Use when implementing a Benchmark class under library/src/physicalai/benchmark, adding a library benchmark config or gym integration, or changing benchmark tests. DO NOT USE FOR running a benchmark on an existing policy (use pai-benchmark-policy).
license: Apache-2.0
---

# Add a Studio Benchmark

`Benchmark`, `PushTBenchmark`, and `LiberoBenchmark` live under `library/src/physicalai/benchmark/gyms/`. Use `pai-benchmark-policy` to run an existing suite; this skill is for changing the library's implementation.

## Workflow

1. **Inspect a matching suite.** Study its `Benchmark` subclass, `library/src/physicalai/eval/rollout.py`, and results types. Identify the gym and success metric before modifying code.
   - Done when: the rollout inputs and expected `BenchmarkResults` fields are clear.
2. **Add or adapt the suite** under `library/src/physicalai/benchmark/gyms/`. Keep heavy gym dependencies behind optional extras and import them lazily.
   - Done when: a one-episode evaluation can produce a summary without unrelated gym packages installed.
3. **Add a first-party config** in `library/configs/benchmark/` only if the suite needs a CLI entry. Check the direct Python API path and the `physicalai benchmark` wrapper with the same checkpoint and episode count.
   - Done when: both entry points produce matching `results.json` / `results.csv` metrics.
4. **Test and document** the suite in `library/tests/unit/benchmark/` and the relevant benchmark docs. Compare to an unchanged baseline and verify optional video recording when supported.
   - Done when: focused tests and a one-episode smoke test pass.

## Verify

```bash
# from library/
uv run --no-sync pytest tests/unit/benchmark
physicalai benchmark --config configs/benchmark/<suite>.yaml --policy <ClassPath> --ckpt_path <path> --benchmark.num_episodes 1
```
