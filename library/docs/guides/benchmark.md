# Benchmark Guide

Evaluate trained policies on standardized environments.

## Quick Start

### CLI

```bash
getiaction benchmark \
    --benchmark getiaction.benchmark.LiberoBenchmark \
    --benchmark.task_suite libero_10 \
    --policy getiaction.policies.ACT \
    --ckpt_path ./checkpoints/model.ckpt
```

### Python API

```python
from getiaction.benchmark import LiberoBenchmark
from getiaction.policies import ACT

policy = ACT.load_from_checkpoint("./checkpoints/model.ckpt")
policy.eval()

benchmark = LiberoBenchmark(task_suite="libero_10", num_episodes=20)
results = benchmark.evaluate(policy)

print(results.summary())
results.to_json("results.json")
```

## CLI Reference

| Argument        | Required | Description                                        |
| --------------- | -------- | -------------------------------------------------- |
| `--benchmark`   | Yes      | Benchmark class path                               |
| `--benchmark.*` | -        | Benchmark-specific options                         |
| `--policy`      | Yes      | Policy class path                                  |
| `--ckpt_path`   | Yes      | Checkpoint file or export directory                |
| `--output_dir`  | No       | Results directory (default: `./results/benchmark`) |

## Python API

### LiberoBenchmark

```python
benchmark = LiberoBenchmark(
    task_suite="libero_10",       # Task suite name
    task_ids=[0, 1, 2],           # Optional: subset of tasks
    num_episodes=20,              # Episodes per task
    max_steps=300,                # Max steps per episode
    seed=42,                      # Random seed
    video_dir="./videos",         # Video output directory
    record_mode="failures",       # Video recording mode
)
```

### BenchmarkResults

```python
results = benchmark.evaluate(policy)

# Metrics
results.success_rate    # Overall success rate
results.mean_reward     # Mean reward
results.n_tasks         # Number of tasks
results.n_episodes      # Total episodes

# Per-task results
for task in results.results:
    print(f"{task.task_id}: {task.success_rate:.1%}")

# Export
results.to_json("results.json")
results.to_csv("results.csv")
```

## LIBERO Task Suites

| Suite            | Tasks | Focus              |
| ---------------- | ----- | ------------------ |
| `libero_spatial` | 10    | Spatial reasoning  |
| `libero_object`  | 10    | Object recognition |
| `libero_goal`    | 10    | Goal specification |
| `libero_10`      | 10    | Mixed evaluation   |
| `libero_90`      | 90    | Full benchmark     |

## Video Recording

| Mode        | Saves                    |
| ----------- | ------------------------ |
| `all`       | Every episode            |
| `failures`  | Failed episodes only     |
| `successes` | Successful episodes only |
| `none`      | No videos                |

Use `failures` mode during debugging to save disk space.

## Config Files

```yaml
# configs/benchmark/my_eval.yaml
benchmark:
  class_path: getiaction.benchmark.LiberoBenchmark
  init_args:
    task_suite: libero_10
    num_episodes: 20
    video_dir: ./results/videos
    record_mode: failures

policy: getiaction.policies.ACT
ckpt_path: ./checkpoints/model.ckpt
output_dir: ./results/benchmark
```

```bash
getiaction benchmark --config configs/benchmark/my_eval.yaml
```

## Output

### Console

```
================================================================================
                           BENCHMARK RESULTS SUMMARY
================================================================================
Benchmark: LiberoBenchmark
Tasks: 10 | Episodes per task: 20 | Total episodes: 200
--------------------------------------------------------------------------------

Task Results:
  libero_10_0                    85.0% success    reward: 0.85 ± 0.36
  libero_10_1                    70.0% success    reward: 0.70 ± 0.46
  ...

--------------------------------------------------------------------------------
AGGREGATE:  75.5% success rate    mean reward: 0.76 ± 0.43
================================================================================
```

### Files

| File           | Content                         |
| -------------- | ------------------------------- |
| `results.json` | Full results with metadata      |
| `results.csv`  | Per-task metrics table          |
| `videos/*.mp4` | Episode recordings (if enabled) |

## Examples

### Quick Test (Single Task)

```bash
getiaction benchmark \
    --benchmark getiaction.benchmark.LiberoBenchmark \
    --benchmark.task_suite libero_10 \
    --benchmark.task_ids "[0]" \
    --benchmark.num_episodes 1 \
    --policy getiaction.policies.ACT \
    --ckpt_path ./checkpoints/model.ckpt
```

### Full Evaluation with Videos

```bash
getiaction benchmark \
    --benchmark getiaction.benchmark.LiberoBenchmark \
    --benchmark.task_suite libero_90 \
    --benchmark.num_episodes 50 \
    --benchmark.video_dir ./results/videos \
    --benchmark.record_mode all \
    --policy getiaction.policies.ACT \
    --ckpt_path ./checkpoints/model.ckpt
```

### Debug Failed Episodes

```bash
getiaction benchmark \
    --benchmark getiaction.benchmark.LiberoBenchmark \
    --benchmark.task_suite libero_10 \
    --benchmark.video_dir ./debug_videos \
    --benchmark.record_mode failures \
    --policy getiaction.policies.ACT \
    --ckpt_path ./checkpoints/model.ckpt
```
