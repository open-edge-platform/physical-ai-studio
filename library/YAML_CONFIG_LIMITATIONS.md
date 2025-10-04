# YAML Configuration Testing for LeRobot Policies

## Summary

❌ **LeRobot policy configs (ACT, Diffusion, VQ-BeT) cannot be fully configured via YAML alone.**

The YAML configs in `configs/lerobot_*.yaml` are **incomplete examples** that won't work without modifications.

## The Problem

LeRobot policies require `input_features` and `output_features` to be provided during initialization. These features depend on the dataset structure and cannot be known ahead of time in a static YAML file.

### Example Error

```bash
$ getiaction fit --config configs/lerobot_act_simple.yaml

# Results in:
TypeError: ACTPolicy.__init__() missing required arguments: 'input_features', 'output_features'
```

## Why This Happens

LeRobot policies need to know:
1. **Input features**: What observations the policy receives (images, states, etc.)
2. **Output features**: What actions the policy predicts
3. **Dataset statistics**: For normalization (mean, std, min, max)

These all come from `dataset.meta.features` and `dataset.meta.stats`, which are only available **after loading the dataset**.

## Current Workarounds

### Option 1: Use Python Code (Recommended)

```python
from getiaction.policies.lerobot import ACT
from getiaction.data.lerobot import LeRobotDataModule
from lerobot.datasets.utils import dataset_to_policy_features
import lightning as L

# Load dataset to get features
datamodule = LeRobotDataModule(repo_id="lerobot/pusht", train_batch_size=8)
datamodule.setup("fit")

# Extract features from dataset
features = dataset_to_policy_features(datamodule.dataset_train.meta.features)
stats = datamodule.dataset_train.meta.stats

# Create policy with features
policy = ACT(
    input_features=features,
    output_features=features,
    dim_model=512,
    chunk_size=100,
    stats=stats,
)

# Train
trainer = L.Trainer(max_epochs=100)
trainer.fit(policy, datamodule)
```

### Option 2: Use Universal Wrapper

The `UniversalLeRobotPolicy` can auto-configure features, but still has limitations:

```python
from getiaction.policies.lerobot import UniversalLeRobotPolicy

# This might work better - auto-configures from dataset
policy = UniversalLeRobotPolicy(
    policy_name="act",
    config={"dim_model": 512, "chunk_size": 100}
)
```

### Option 3: Two-Stage Configuration (Future Enhancement)

A potential solution would be to support two-stage initialization:

```python
# Stage 1: Create policy skeleton from YAML
policy = ACT.from_yaml("configs/lerobot_act.yaml", lazy_init=True)

# Stage 2: Finalize with dataset features
policy.configure_from_dataset(dataset)
```

**This is not currently implemented.**

## Working YAML Configs

Only **Dummy policy** configs work fully with YAML because they don't require dataset-specific features:

✅ `configs/train_working.yaml` - Dummy policy (works)
✅ `configs/train_dummy_*.yaml` - Dummy policy variants (work)
❌ `configs/lerobot_act*.yaml` - ACT (incomplete)
❌ `configs/lerobot_diffusion*.yaml` - Diffusion (incomplete)
❌ `configs/lerobot_vqbet*.yaml` - VQ-BeT (incomplete)

## Recommendations

1. **For experimentation**: Use Python scripts with explicit feature configuration
2. **For production**: Create a training script that loads dataset → extracts features → creates policy
3. **For documentation**: Mark LeRobot YAML configs as "templates" or "examples" that need modification

## Test Results

```bash
# This works (Dummy policy):
getiaction fit --config configs/train_working.yaml --trainer.fast_dev_run=1

# This fails (ACT policy - missing features):
getiaction fit --config configs/lerobot_act_simple.yaml --trainer.fast_dev_run=1
```

## Proposed Solution

Add a `setup_hook` method that Lightning DataModule can call to configure the policy:

```python
class ACT(Policy):
    def setup(self, stage: str, datamodule: DataModule) -> None:
        """Called by Lightning after DataModule setup."""
        if not hasattr(self, 'lerobot_policy'):
            # Lazy initialization with dataset features
            features = dataset_to_policy_features(datamodule.dataset_train.meta.features)
            stats = datamodule.dataset_train.meta.stats
            self._initialize_lerobot_policy(features, stats)
```

This would allow YAML configs to work, but requires implementation changes.

## Conclusion

**YAML configs for LeRobot policies are currently non-functional** without manual intervention. Users must:
1. Load the dataset first
2. Extract features and stats
3. Create policy with these parameters in Python

The provided YAML files serve as **documentation/templates** but cannot be used directly with `getiaction fit`.
