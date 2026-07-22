# Rollout Evaluation

Policy evaluation system with two APIs: functional `rollout()` for scripts
and stateful `Rollout` metric for Lightning training.

## Design

The system separates rollout execution from metric aggregation:

- **`rollout()`**: Stateless function that executes a single episode
- **`Rollout`**: Stateful torchmetrics.Metric that accumulates results and
  handles distributed synchronization

Both share the same underlying rollout logic but serve different use cases.

## Architecture

```text
┌────────────────────────────────────────────┐
│          Evaluation System                 │
├────────────────────────────────────────────┤
│                                            │
│  rollout()              Rollout            │
│  ├─ Stateless          ├─ Stateful         │
│  ├─ Single episode     ├─ Accumulation     │
│  ├─ Immediate return   ├─ Deferred compute │
│  └─ No framework       └─ Lightning + DDP  │
│                                            │
│         ↓                      ↓           │
│    ┌────────────────────────────┐          │
│    │  Gym Environment + Policy  │          │
│    └────────────────────────────┘          │
└────────────────────────────────────────────┘
```

### File Structure

```text
getiaction/eval/
├── rollout.py    # rollout() function
└── metrics.py    # Rollout metric class

getiaction/policies/base/
└── policy.py     # Policy with integrated Rollout metrics
```

## API

### `rollout()` - Functional API

```python
def rollout(
    env: Gym,
    policy: Policy,
    seed: int | None = None,
    max_steps: int | None = None,
    return_observations: bool = False,
) -> dict[str, Any]:
    """Execute one episode, return results immediately."""
```

Returns: `sum_reward`, `max_reward`, `episode_length`,
optionally `observations`/`actions`/`rewards`

### `Rollout` - Metric Class

```python
class Rollout(torchmetrics.Metric):
    def update(self, env: Gym, policy: Policy, seed: int | None = None):
        """Run episode and accumulate state."""

    def compute(self) -> dict[str, torch.Tensor]:
        """Compute aggregated metrics across all episodes."""

    def reset(self):
        """Clear state (automatic between epochs)."""
```

Returns: `avg_sum_reward`, `avg_max_reward`, `avg_episode_length`,
`pc_success`, `n_episodes`

## When to Use What

| Use Case               | Use `rollout()`           | Use `Rollout`         |
| ---------------------- | ------------------------- | --------------------- |
| **Standalone scripts** | ✅ Simple, direct         | ❌ Overkill           |
| **Custom aggregation** | ✅ Full control           | ❌ Limited stats      |
| **Lightning training** | ❌ Manual boilerplate     | ✅ Native integration |
| **Multi-GPU**          | ❌ Manual sync needed     | ✅ Automatic sync     |
| **Debugging/analysis** | ✅ Return full trajectory | ❌ Only aggregates    |
| **Non-Lightning**      | ✅ Framework agnostic     | ❌ Requires Lightning |

### Decision Tree

```text
Need evaluation?
├─ In Lightning training loop?
│  ├─ Yes → Use Rollout metric (automatic aggregation + DDP)
│  └─ No → Continue
└─ Need custom statistics or trajectory data?
   ├─ Yes → Use rollout() (full control)
   └─ No → Use rollout() anyway (simpler)
```

## Key Features

### Distributed Training (Rollout metric only)

Multi-GPU support via torchmetrics state synchronization:

```python
# Each GPU runs rollouts independently
# Metrics automatically synchronized at epoch end
trainer = Trainer(devices=4, strategy="ddp")
trainer.fit(policy, datamodule)
```

State variables use `dist_reduce_fx` for aggregation:

- `dist_reduce_fx="sum"`: Sum values across GPUs (counts, totals)
- `dist_reduce_fx="cat"`: Concatenate lists across GPUs (per-episode data)

### Policy Integration

Base `Policy` class includes validation/test Rollout metrics:

```python
class Policy(L.LightningModule):
    def __init__(self):
        self.val_rollout = Rollout()
        self.test_rollout = Rollout()

    def validation_step(self, batch: Gym, batch_idx: int):
        return self.evaluate_gym(batch, batch_idx, stage="val")

    def on_validation_epoch_end(self):
        metrics = self.val_rollout.compute()
        self.log_dict({f"val/gym/{k}": v for k, v in metrics.items()})
        self.val_rollout.reset()
```

Two-level logging:

- **Per-episode**: `on_step=True` - immediate feedback during validation
- **Aggregated**: `on_epoch=True` - epoch summary for comparison

## Examples

### Example 1: Evaluation Script with `rollout()`

```python
from getiaction.eval import rollout
from getiaction.policies.act import ACT
from getiaction.gyms import PushTGym

policy = ACT.load_from_checkpoint("model.ckpt")
env = PushTGym()

# Run 50 episodes
results = [rollout(env, policy, seed=i, max_steps=500) for i in range(50)]

# Compute statistics
avg_reward = sum(r['sum_reward'] for r in results) / len(results)
print(f"Success: {success_rate*100:.1f}%, Reward: {avg_reward:.2f}")
```

### Example 2: Lightning Training with `Rollout`

```python
from lightning.pytorch import Trainer
from getiaction.data import DataModule
from getiaction.policies.act import ACT

datamodule = DataModule(
    train_dataset=dataset,
    val_gym=PushTGym(),
    num_rollouts_val=10,
    max_episode_steps=500,
)

conf = ACTConfig(
        input_features=datamodule.train_dataset.observation_features,
        output_features=datamodule.train_dataset.action_features,
    )
model = ACTModel.from_config(conf)
act_policy = ACT(model=model)
trainer = Trainer(max_epochs=100)
trainer.fit(policy, datamodule)

# Metrics automatically logged:
# - val/gym/episode/sum_reward (per episode)
# - val/gym/avg_sum_reward (aggregated)
```

### Example 3: Multi-GPU Training

```python
# Same code, add devices and strategy
trainer = Trainer(
    devices=4,
    strategy="ddp",
    max_epochs=100,
)
trainer.fit(policy, datamodule)
# Rollout metrics automatically synchronized across GPUs
```

### Example 4: Custom Aggregation

```python
from getiaction.eval import rollout
import numpy as np

results = [rollout(env, policy, seed=i) for i in range(100)]
rewards = [r['sum_reward'] for r in results]

# Custom statistics
print(f"Mean: {np.mean(rewards):.2f}")
print(f"Median: {np.median(rewards):.2f}")
print(f"95th percentile: {np.percentile(rewards, 95):.2f}")
```

### Example 5: Trajectory Analysis

```python
# Get full episode trajectory
result = rollout(env, policy, seed=42, return_observations=True)

# Analyze
import matplotlib.pyplot as plt
plt.plot(result['rewards'])
plt.xlabel('Step')
plt.ylabel('Reward')
plt.title(f"Total: {result['sum_reward']:.2f}")
plt.show()
```

### Example 6: Multiple Environments

```python
from getiaction.eval import Rollout

test_envs = [
    PushTGym(),
    PushTGym(difficulty="hard"),
    PushTGym(object_type="T"),
]

for i, env in enumerate(test_envs):
    metric = Rollout(max_steps=500)
    for seed in range(20):
        metric.update(env, policy, seed=seed)

    results = metric.compute()
    print(f"Env {i}: {results['max_reward']:.1f} highest reward")
```

## Extension

### Custom Metrics

```python
from getiaction.eval import Rollout

class ExtendedRollout(Rollout):
    def __init__(self, max_steps=None):
        super().__init__(max_steps)
        self.add_state("min_reward", default=torch.tensor(float('inf')), dist_reduce_fx="min")
        self.add_state("max_reward", default=torch.tensor(float('-inf')), dist_reduce_fx="max")

    def update(self, env, policy, seed=None):
        super().update(env, policy, seed)
        latest_reward = self.all_sum_rewards[-1]
        self.min_reward = torch.min(self.min_reward, latest_reward)
        self.max_reward = torch.max(self.max_reward, latest_reward)

    def compute(self):
        metrics = super().compute()
        metrics.update({
            "min_reward": self.min_reward,
            "max_reward": self.max_reward,
            "reward_range": self.max_reward - self.min_reward,
        })
        return metrics
```

### Custom Policy Integration

```python
from getiaction.policies.base import Policy

class MyPolicy(Policy):
    def evaluate_gym(self, batch, batch_idx, stage):
        # Pre-processing
        self.eval()

        # Standard evaluation
        metrics = super().evaluate_gym(batch, batch_idx, stage)

        # Post-processing
        if self.should_log_extra(batch_idx):
            self.log_extra_info(batch)

        return metrics
```

## Summary

| Aspect      | `rollout()`       | `Rollout`      |
| ----------- | ----------------- | -------------- |
| Type        | Function          | Metric Class   |
| State       | Stateless         | Stateful       |
| Use Case    | Scripts, analysis | Training loops |
| Framework   | Agnostic          | Lightning      |
| Multi-GPU   | Manual            | Automatic      |
| Aggregation | Manual            | Automatic      |

Choose based on use case:

- Standalone evaluation, debugging, custom stats → `rollout()`
- Lightning training, multi-GPU → `Rollout`
