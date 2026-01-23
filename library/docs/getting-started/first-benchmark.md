# First Benchmark

Evaluate your trained policy on standardized simulation environments.

## What You'll Do

1. Load your trained checkpoint
2. Run evaluation on LIBERO benchmark
3. Interpret the results

## Prerequisites

- [Trained policy checkpoint](quickstart.md)
- LIBERO benchmark environment (installed automatically)

## Step 1: Run Benchmark with CLI

Evaluate your policy on the LIBERO-10 benchmark:

```bash
getiaction benchmark \
    --benchmark getiaction.benchmark.LiberoBenchmark \
    --benchmark.task_suite libero_10 \
    --benchmark.num_episodes 20 \
    --policy getiaction.policies.ACT \
    --ckpt_path experiments/lightning_logs/version_0/checkpoints/last.ckpt
```

This runs 20 episodes per task across 10 LIBERO tasks.

## Step 2: Run Benchmark with Python API

For more control:

```python test="skip" reason="requires checkpoint and libero"
from getiaction.benchmark import LiberoBenchmark
from getiaction.policies import ACT

# Load trained policy
policy = ACT.load_from_checkpoint(
    "experiments/lightning_logs/version_0/checkpoints/last.ckpt"
)
policy.eval()

# Create benchmark
benchmark = LiberoBenchmark(
    task_suite="libero_10",
    num_episodes=20,
)

# Run evaluation
results = benchmark.evaluate(policy)

# Print summary
print(results.summary())
```

## Step 3: Interpret Results

### Console Output

```text
================================================================================
                           BENCHMARK RESULTS SUMMARY
================================================================================
Benchmark: LiberoBenchmark
Tasks: 10 | Episodes per task: 20 | Total episodes: 200
--------------------------------------------------------------------------------

Task Results:
  libero_10_0                    85.0% success    reward: 0.85 ± 0.36
  libero_10_1                    70.0% success    reward: 0.70 ± 0.46
  libero_10_2                    90.0% success    reward: 0.90 ± 0.30
  ...

--------------------------------------------------------------------------------
AGGREGATE:  75.5% success rate    mean reward: 0.76 ± 0.43
================================================================================
```

### What the Metrics Mean

| Metric           | Description                                |
| ---------------- | ------------------------------------------ |
| **Success Rate** | % of episodes where the task was completed |
| **Mean Reward**  | Average reward across episodes             |
| **± value**      | Standard deviation (consistency measure)   |

### Save Results

```python test="skip" reason="requires results object from benchmark"
# Export to files
results.to_json("results.json")
results.to_csv("results.csv")

# Access individual task results
for task in results.task_results:
    print(f"{task.task_id}: {task.success_rate:.1f}%")
```

## Record Videos

Debug failures by recording episodes:

```bash
getiaction benchmark \
    --benchmark getiaction.benchmark.LiberoBenchmark \
    --benchmark.task_suite libero_10 \
    --benchmark.num_episodes 20 \
    --benchmark.video_dir ./videos \
    --benchmark.record_mode failures \
    --policy getiaction.policies.ACT \
    --ckpt_path experiments/lightning_logs/version_0/checkpoints/last.ckpt
```

Recording modes:

| Mode        | Records                  |
| ----------- | ------------------------ |
| `all`       | Every episode            |
| `failures`  | Failed episodes only     |
| `successes` | Successful episodes only |
| `none`      | No videos (default)      |

## Available Benchmarks

### LIBERO Suites

| Suite            | Tasks | Description                                    |
| ---------------- | ----- | ---------------------------------------------- |
| `libero_spatial` | 10    | Spatial reasoning tasks                        |
| `libero_object`  | 10    | Object manipulation                            |
| `libero_goal`    | 10    | Goal-conditioned tasks                         |
| `libero_10`      | 10    | Mixed evaluation (recommended for quick tests) |
| `libero_90`      | 90    | Full benchmark                                 |

### PushT

2D pushing task (faster to run):

```python test="skip" reason="requires policy object"
from getiaction.benchmark import PushTBenchmark

benchmark = PushTBenchmark(num_episodes=50)
results = benchmark.evaluate(policy)
```

## Quick Test Run

Test a single task before full evaluation:

```bash
getiaction benchmark \
    --benchmark getiaction.benchmark.LiberoBenchmark \
    --benchmark.task_suite libero_10 \
    --benchmark.task_ids "[0]" \
    --benchmark.num_episodes 1 \
    --policy getiaction.policies.ACT \
    --ckpt_path experiments/lightning_logs/version_0/checkpoints/last.ckpt
```

## Using Config Files

For reproducible evaluations:

```yaml
# configs/benchmark/my_eval.yaml
benchmark:
  class_path: getiaction.benchmark.LiberoBenchmark
  init_args:
    task_suite: libero_10
    num_episodes: 50
    video_dir: ./results/videos
    record_mode: failures

policy: getiaction.policies.ACT
ckpt_path: ./experiments/lightning_logs/version_0/checkpoints/last.ckpt
output_dir: ./results/benchmark
```

```bash
getiaction benchmark --config configs/benchmark/my_eval.yaml
```

## What's Next?

- Happy with results? [Deploy your policy](first-deployment.md)
- Results poor? Train longer or tune hyperparameters
- [How-To Guides](../how-to/) for advanced benchmark configurations

## Troubleshooting

**Benchmark takes too long**: Reduce `--benchmark.num_episodes` or test fewer tasks with `--benchmark.task_ids`

**LIBERO import error**: Ensure LIBERO is installed: `pip install libero`

**Video recording fails**: Check FFMPEG is installed and `video_dir` is writable
