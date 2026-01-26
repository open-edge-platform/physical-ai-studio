# Benchmark Guide

After training a policy, you need to measure how well it performs. The benchmark module runs your policy through standardized environments, collecting success rates, rewards, and optional video recordings. Use the CLI for quick evaluation or the Python API for custom workflows.

## Quick Start

### CLI

```bash
getiaction benchmark \
    --benchmark getiaction.benchmark.LiberoBenchmark \
    --benchmark.task_suite libero_10 \
    --policy getiaction.policies.ACT \
    --ckpt_path ./lightning_logs/version_0/checkpoints/epoch=99.ckpt
```

### Python API

```python
from getiaction.benchmark import LiberoBenchmark
from getiaction.policies import ACT

policy = ACT.load_from_checkpoint("./checkpoints/epoch=99.ckpt")
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
| `--ckpt_path`   | Yes      | Path to checkpoint file or export directory        |
| `--output_dir`  | No       | Results directory (default: `./results/benchmark`) |

## Python API Reference

### LiberoBenchmark

```python
from getiaction.benchmark import LiberoBenchmark

benchmark = LiberoBenchmark(
    task_suite="libero_10",       # Task suite name
    task_ids=[0, 1, 2],           # Optional: subset of tasks
    num_episodes=20,              # Episodes per task
    max_steps=300,                # Max steps per episode
    seed=42,                      # Random seed
    observation_height=256,       # Image height
    observation_width=256,        # Image width
    video_dir="./videos",         # Video output directory
    record_mode="failures",       # Video recording mode
)
```

### BenchmarkResults

```python
results = benchmark.evaluate(policy)

# Aggregate metrics
results.success_rate          # Overall success rate
results.mean_reward           # Mean reward across all episodes
results.std_reward            # Reward standard deviation
results.n_tasks               # Number of tasks
results.n_episodes            # Total episodes

# Per-task results
for task_result in results.results:
    print(f"{task_result.task_id}: {task_result.success_rate:.1%}")

# Export
results.to_json("results.json")
results.to_csv("results.csv")
print(results.summary())

# Load previous results
from getiaction.benchmark import BenchmarkResults
old_results = BenchmarkResults.from_json("results.json")
```

### VideoRecorder (Advanced)

```python
from getiaction.benchmark import VideoRecorder

recorder = VideoRecorder(
    output_dir="./videos",
    fps=30,
    record_mode="failures",  # all | failures | successes | none
)

# Used internally by benchmark, or manually:
with recorder:
    recorder.start_episode("task_0_ep_0")
    for frame in frames:
        recorder.add_frame(frame)
    recorder.end_episode(success=False)  # Saves if mode matches
```

## LIBERO Options

LIBERO provides multiple task suites for different evaluation needs. For quick iteration, use `libero_10`. For comprehensive evaluation, use `libero_90`.

### Task Suites

| Suite            | Tasks | Focus              |
| ---------------- | ----- | ------------------ |
| `libero_spatial` | 10    | Spatial reasoning  |
| `libero_object`  | 10    | Object recognition |
| `libero_goal`    | 10    | Goal specification |
| `libero_10`      | 10    | Mixed evaluation   |
| `libero_90`      | 90    | Full benchmark     |

### Video Recording Modes

Recording every episode fills disk fast. Use `failures` mode during debugging to capture only what went wrong.

| Mode        | Saves                    |
| ----------- | ------------------------ |
| `all`       | Every episode            |
| `failures`  | Failed episodes only     |
| `successes` | Successful episodes only |
| `none`      | No videos                |

## Config Files

Store benchmark configurations in YAML for reproducibility:

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
ckpt_path: ./checkpoints/act_libero.ckpt
output_dir: ./results/benchmark
```

```bash
getiaction benchmark --config configs/benchmark/my_eval.yaml
```

## Output

### Console Summary

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
  ...

--------------------------------------------------------------------------------
AGGREGATE:  75.5% success rate    mean reward: 0.76 ± 0.43
================================================================================
```

### Exported Files

| File           | Content                         |
| -------------- | ------------------------------- |
| `results.json` | Full results with metadata      |
| `results.csv`  | Per-task metrics table          |
| `videos/*.mp4` | Episode recordings (if enabled) |

## Examples

### Quick Test

CLI:

```bash
getiaction benchmark \
    --benchmark getiaction.benchmark.LiberoBenchmark \
    --benchmark.task_suite libero_10 \
    --benchmark.task_ids "[0]" \
    --benchmark.num_episodes 1 \
    --policy getiaction.policies.ACT \
    --ckpt_path ./checkpoints/act.ckpt
```

Python:

```python
benchmark = LiberoBenchmark(task_suite="libero_10", task_ids=[0], num_episodes=1)
results = benchmark.evaluate(policy)
```

### Full Evaluation with Videos

CLI:

```bash
getiaction benchmark \
    --benchmark getiaction.benchmark.LiberoBenchmark \
    --benchmark.task_suite libero_90 \
    --benchmark.num_episodes 50 \
    --benchmark.video_dir ./results/videos \
    --benchmark.record_mode all \
    --policy getiaction.policies.ACT \
    --ckpt_path ./checkpoints/act.ckpt
```

Python:

```python
benchmark = LiberoBenchmark(
    task_suite="libero_90",
    num_episodes=50,
    video_dir="./results/videos",
    record_mode="all",
)
results = benchmark.evaluate(policy)
results.to_json("./results/libero_90_full/results.json")
```

### Debug Failed Episodes

CLI:

```bash
getiaction benchmark \
    --benchmark getiaction.benchmark.LiberoBenchmark \
    --benchmark.task_suite libero_10 \
    --benchmark.video_dir ./debug_videos \
    --benchmark.record_mode failures \
    --policy getiaction.policies.ACT \
    --ckpt_path ./checkpoints/act.ckpt
```

Python:

```python
benchmark = LiberoBenchmark(
    task_suite="libero_10",
    video_dir="./debug_videos",
    record_mode="failures",
)
results = benchmark.evaluate(policy)

# Check which tasks failed
for r in results.results:
    if r.success_rate < 1.0:
        print(f"{r.task_id}: {r.success_rate:.1%} - check videos")
```
