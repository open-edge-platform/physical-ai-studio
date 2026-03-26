# LIBERO Benchmark

Evaluates policies on the [LIBERO](https://libero-project.github.io/) robotic manipulation benchmark suites.

`LiberoBenchmark` wraps `Benchmark` and auto-creates a `LiberoGym` for every task in a suite. It runs your policy through each environment, collects success rates and rewards, and optionally records episode videos.

## Installation

Requires the LIBERO optional dependency:

```bash
uv sync --extra libero
```

## Task Suites

| Suite            | Tasks | Max Steps | Focus                   |
| ---------------- | ----- | --------- | ----------------------- |
| `libero_spatial` | 10    | 280       | Spatial reasoning       |
| `libero_object`  | 10    | 280       | Object generalization   |
| `libero_goal`    | 10    | 300       | Goal-conditioned tasks  |
| `libero_10`      | 10    | 520       | Mixed long-horizon      |
| `libero_90`      | 90    | 400       | Large-scale evaluation  |


## End-to-End Example with Pi0.5

The following example loads a pretrained Pi0.5 checkpoint and runs the full LIBERO-10 benchmark.

We use the LIBERO imitation learning training set provided by the LeRobot found [here](https://huggingface.co/datasets/HuggingFaceVLA/libero). Please see the LeRobot implementation [here](https://huggingface.co/docs/lerobot/libero).

```bash
uv sync --extra libero --extra pi0
```

## Code

```python
from physicalai.benchmark import LiberoBenchmark
from physicalai.data import LeRobotDataModule
from physicalai.policies import Pi05
from physicalai.train import Trainer

# Imitation dataset of LIBERO
datamodule = LeRobotDataModule(repo_id="HuggingFaceVLA/libero")
model = Pi05()
trainer = Trainer(max_epochs=100)
trainer.fit(model=model, datamodule=datamodule)

# Load best checkpoint
policy = Pi05.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
policy.eval()

# Benchmark
benchmark = LiberoBenchmark(
    task_suite="libero_10",
    num_episodes=20,         # 20 episodes per task
    seed=42,
    video_dir="./videos",
    record_mode="failures",  # Record only failed episodes
)

results = benchmark.evaluate(policy)
print(results.summary())
```

or from the lerobot pre-trained policy:

```python
import torch
from physicalai.policies.pi05 import Pi05
from physicalai.benchmark import LiberoBenchmark

device = "cuda" if torch.cuda.is_available() else "cpu"

# LeRobot Weights
policy = Pi05(pretrained_name_or_path="lerobot/pi05_libero_finetuned_v044")
policy.eval()

# Benchmark
benchmark = LiberoBenchmark(
    task_suite="libero_10",
    num_episodes=20,
    seed=42,
    video_dir="./videos",
    record_mode="failures",
)

results = benchmark.evaluate(policy)
print(results.summary())
```

## CLI

Use the `physicalai benchmark` subcommand for one-liner evaluation without writing Python:

```bash
physicalai benchmark \
    --benchmark physicalai.benchmark.LiberoBenchmark \
    --benchmark.task_suite libero_10 \
    --benchmark.num_episodes 20 \
    --benchmark.video_dir ./results/videos \
    --benchmark.record_mode failures \
    --policy physicalai.policies.pi05.Pi05 \
    --ckpt_path ./checkpoints/pi05_libero.ckpt
```
