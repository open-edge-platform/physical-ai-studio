# Evaluation Implementation Summary

This document describes the elegant evaluation solution implemented for getiaction, inspired by LeRobot's evaluation framework but adapted for getiaction's architecture.

## Overview

The evaluation system enables policies to be tested in gym environments during training, providing real-world performance metrics (success rates, rewards, episode lengths) alongside training losses.

## Architecture

### 1. **Evaluation Utilities** (`getiaction/eval/rollout.py`)

Core evaluation functions adapted from LeRobot's approach:

- **`rollout()`**: Executes a complete policy rollout in a gym environment
  - Resets policy state before starting
  - Uses `Gym.get_max_episode_steps()` to determine episode length
  - Properly handles `Gym.reset()` return tuple `(observation, info)`
  - Collects actions, rewards, success flags, and optionally observations
  - Returns comprehensive metrics including sum_reward, max_reward, is_success, episode_length
  - Handles proper action conversion (Tensor → numpy) and observation formatting

- **`evaluate_policy()`**: Runs multiple evaluation episodes
  - Aggregates metrics across episodes (average rewards, success rate)
  - Supports seeded evaluation for reproducibility
  - Returns per-episode and aggregated statistics

### 2. **Lightning Callback** (`getiaction/train/callbacks/evaluation.py`)

**`GymEvaluation`**: Integrates gym evaluation into Lightning's validation loop

- Triggers during validation steps when gym environments are provided
- Runs rollout() for each gym environment batch
- Logs metrics to trainer's logger with prefix `val/gym/`
- Metrics: episode_length, sum_reward, max_reward, success

### 3. **Base Policy Updates** (`getiaction/policies/base/policy.py`)

Added two critical methods:

- **`reset()`**: Abstract method that must be implemented by all policies
  - Clears action queues, observation histories, and stateful components
  - Critical for proper episode initialization in gym environments
  - Called automatically by rollout() before each episode

- **`validation_step()`**: Handles two validation types:
  - Dataset validation: standard loss computation (must be overridden by subclasses)
  - Gym validation: returns None, evaluation handled by GymEvaluation
  - Detects gym batches by checking for `'env'` key in batch dict

### 4. **LeRobot Policy Wrappers**

Both ACT and Universal policy wrappers now include:

- **`reset()`**: Forwards to underlying LeRobot policy's reset()
- **`select_action()`**: Properly handles observation format conversion

## Key Design Decisions

### 1. **Separation of Concerns**

- Rollout logic in dedicated module (`eval/rollout.py`)
- Lightning integration via callback (no coupling to policy internals)
- Base Policy defines interface, wrappers implement forwarding

### 2. **LeRobot Compatibility**

- Rollout structure mirrors LeRobot's but adapted for getiaction's Observation dataclass
- Preserves LeRobot policy behavior through proper forwarding
- Format conversion handled transparently

### 3. **Lightning-First Design**

- Uses Lightning's callback system (no custom training loops)
- Leverages Lightning's logging infrastructure
- Works seamlessly with existing Lightning features (checkpointing, early stopping, etc.)

### 4. **Flexible Validation**

- Supports both dataset validation (loss-based) and gym validation (rollout-based)
- Automatic detection via batch structure (`'env'` key presence)
- Callback handles gym evaluation transparently

## Usage

### Basic Setup

```python
from getiaction.policies.lerobot import ACT
from getiaction.data import DataModule
from getiaction.gyms import PushTGym
from getiaction.cli import Trainer
from getiaction.train.callbacks import GymEvaluation

# Create policy
policy = ACT(...)

# Create datamodule with eval gym
eval_gym = PushTGym()
datamodule = DataModule(
    train_dataset=train_dataset,
    val_gyms=eval_gym,
    num_rollouts_val=10
)

# Create trainer with callback
callback = GymEvaluation(max_steps=300)
trainer = Trainer(callbacks=[callback])

# Train (evaluation happens automatically during validation)
trainer.fit(policy, datamodule)
```

### Standalone Evaluation

```python
from getiaction.eval import evaluate_policy
from getiaction.gyms import PushTGym
from getiaction.policies.lerobot import ACT

# Load policy
policy = ACT.from_pretrained("path/to/checkpoint")

# Create environment
env = PushTGym()

# Evaluate
results = evaluate_policy(
    env=env,
    policy=policy,
    n_episodes=50,
    start_seed=1000
)

print(f"Success rate: {results['aggregated']['pc_success']:.1f}%")
print(f"Avg reward: {results['aggregated']['avg_sum_reward']:.2f}")
```

## Integration with DataModule

The DataModule automatically handles gym environment setup:

1. **Gym wrapping**: Wraps gym environments as `GymDataset` (each `__getitem__` returns the env after reset)
2. **Time limits**: Applies `TimeLimit` wrapper if `max_episode_steps` specified
3. **Collation**: Custom collate function (`_collate_env`) packages gym for validation_step
4. **Batch structure**: Returns `{'env': Gym}` for gym validation batches
5. **Automatic seeding**: `GymDataset.__getitem__` resets env with index as seed

**Key insight**: The `GymDataset` already handles environment reset, so the rollout function receives
a pre-reset environment. The rollout then resets again with the specified seed for reproducibility.

## Comparison with LeRobot

### Similarities

- Core rollout logic structure
- Metric computation approach
- Support for seeded evaluation
- Return dictionary structure

### Differences

- **No vectorized envs**: getiaction uses single gym environments (simpler, sufficient for most cases)
- **Lightning integration**: Uses callbacks instead of standalone eval scripts
- **Observation format**: Adapted for getiaction's Observation dataclass
- **Type safety**: Stronger typing and better IDE support
- **Simplicity**: Fewer abstractions, more straightforward for users

## File Structure

```
library/src/getiaction/
├── eval/
│   ├── __init__.py          # Exports rollout, evaluate_policy
│   └── rollout.py           # Core evaluation functions
├── train/
│   └── callbacks/
│       ├── __init__.py      # Exports GymEvaluation
│       ├── evaluation.py    # Gym evaluation callback
│       └── policy_dataset_interaction.py  # Existing callback
└── policies/
    ├── base/
    │   └── policy.py        # Added reset(), validation_step()
    └── lerobot/
        ├── act.py           # reset() implemented
        ├── universal.py     # reset() added
        └── diffusion.py     # reset() already present
```

## Future Enhancements

Potential improvements for future work:

1. **Video rendering**: Add support for recording evaluation videos
2. **Multiple gyms**: Better support for evaluating across multiple environments simultaneously
3. **Advanced metrics**: Add more sophisticated metrics (action smoothness, diversity, etc.)
4. **Vectorized envs**: Optional support for gym.vector.VectorEnv for faster evaluation
5. **Async evaluation**: Run evaluation in background thread to avoid blocking training
6. **Custom callbacks**: Easy hooks for user-defined evaluation metrics
7. **Benchmark suite**: Predefined evaluation configurations for common benchmarks

## Code Quality

The implementation follows getiaction's high standards:

- **Type hints**: Comprehensive type annotations throughout
- **Documentation**: Detailed docstrings with examples and notes
- **Error handling**: Proper validation and error messages
- **Clean abstractions**: Clear separation of concerns
- **Extensibility**: Easy to extend with custom metrics or behaviors
- **Testing**: Ready for unit and integration tests

## Testing Recommendations

Suggested tests to add:

1. **Unit tests for rollout()**: Test with mock gym and policy
2. **Unit tests for evaluate_policy()**: Test metric aggregation
3. **Integration test**: Full training run with gym evaluation
4. **Test callback registration**: Ensure callback is properly triggered
5. **Test reset() implementations**: Verify state clearing
6. **Test format conversions**: Observation → LeRobot format

## Conclusion

This evaluation implementation provides an elegant, Lightning-integrated solution for testing policies in gym environments. It maintains compatibility with LeRobot while leveraging getiaction's architecture for cleaner, more maintainable code.

The design is flexible enough to support various use cases while remaining simple and intuitive for users. The callback-based approach ensures seamless integration with Lightning's training loop without requiring custom logic in user code.
