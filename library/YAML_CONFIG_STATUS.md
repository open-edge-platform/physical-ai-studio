# YAML Configuration Status Report

## Executive Summary

**❌ YAML-based training configs are currently NON-FUNCTIONAL** for LeRobot policies (ACT, Diffusion, VQ-BeT).

The provided YAML files in `configs/` are **documentation templates only** and cannot be used directly with `getiaction fit`.

## Test Results

### Test 1: ACT Policy from YAML

```bash
getiaction fit --config configs/lerobot_act_simple.yaml --trainer.fast_dev_run=1
```

**Result**: ❌ FAILS

```
ValueError: You must provide at least one image or the environment state among the inputs.
```

### Test 2: Dummy Policy from YAML

```bash
getiaction fit --config configs/train_working.yaml --trainer.fast_dev_run=1
```

**Result**: ❌ FAILS

```
Parser error: Not a valid subclass of Size. Got value: [7]
```

### Test 3: ACT from Python (No Features)

```python
from getiaction.policies.lerobot import ACT
policy = ACT(dim_model=512, chunk_size=100)
```

**Result**: ❌ FAILS

```
ValueError: You must provide at least one image or the environment state among the inputs.
```

### Test 4: Integration Tests (Python with Features)

```python
features = dataset_to_policy_features(dataset.meta.features)
policy = ACT(input_features=features, output_features=features, stats=dataset.meta.stats, ...)
trainer.fit(policy, dataloader)
```

**Result**: ✅ **WORKS** (see `test_lerobot_integration.py` - 3/3 passing)

## Root Causes

### Cause 1: Dataset-Dependent Features Required

LeRobot policies **cannot be instantiated** without dataset-specific information:

```python
# What LeRobot policies need:
ACT(
    input_features={                    # ← From dataset.meta.features
        "observation.image": PolicyFeature(type="VISUAL", shape=(3, 96, 96)),
        "observation.state": PolicyFeature(type="STATE", shape=(2,)),
        "action": PolicyFeature(type="ACTION", shape=(2,))
    },
    output_features={...},              # ← Same as above
    stats={                             # ← From dataset.meta.stats
        "observation.state": {"mean": [...], "std": [...]},
        "action": {"mean": [...], "std": [...]}
    },
    dim_model=512,                      # ← Can be in YAML
    chunk_size=100,                     # ← Can be in YAML
)
```

**Problem**: Features and stats can only be obtained **after** loading the dataset, but policies need them **during** initialization.

### Cause 2: LightningCLI Instantiation Order

LightningCLI instantiates components in this order:

1. Parse YAML config
2. **Instantiate Model** (Policy)
3. Instantiate DataModule
4. Setup DataModule (loads dataset)

By the time the dataset is loaded (step 4), the policy has already been created (step 2) **without** the required features.

### Cause 3: jsonargparse Type System Limitations

Even simple configs fail due to jsonargparse not understanding certain Python types:

- List types like `action_shape: [7]` fail validation
- Nested class instantiation has parsing issues

## What Currently Works

### ✅ Python Scripts with Explicit Configuration

```python
from getiaction.policies.lerobot import ACT
from getiaction.data.lerobot import LeRobotDataModule
from lerobot.datasets.utils import dataset_to_policy_features
import lightning as L

# Step 1: Load dataset
datamodule = LeRobotDataModule(repo_id="lerobot/pusht", train_batch_size=8)
datamodule.setup("fit")

# Step 2: Extract features from dataset
features = dataset_to_policy_features(datamodule.dataset_train.meta.features)
stats = datamodule.dataset_train.meta.stats

# Step 3: Create policy with features
policy = ACT(
    input_features=features,
    output_features=features,
    stats=stats,
    dim_model=512,
    chunk_size=100,
    n_action_steps=100,
)

# Step 4: Train
trainer = L.Trainer(max_epochs=100)
trainer.fit(policy, datamodule)
```

### ✅ Integration Tests (Python)

All integration tests in `tests/test_lerobot_integration.py` **pass**:

- `test_training_step_with_temporal_data` ✅
- `test_validation_step_with_temporal_data` ✅
- `test_full_training_loop` ✅

These prove the **underlying functionality works** - only YAML configuration is broken.

## What Doesn't Work

### ❌ YAML Configs with LightningCLI

**All** `configs/lerobot_*.yaml` files are non-functional:

- `configs/lerobot_act.yaml` - ACT policy
- `configs/lerobot_act_simple.yaml` - ACT policy (minimal)
- `configs/lerobot_diffusion_*.yaml` - Diffusion policy
- `configs/lerobot_vqbet_*.yaml` - VQ-BeT policy

Even `configs/train_working.yaml` (Dummy policy) fails due to parser issues.

## Possible Solutions

### Solution 1: Two-Stage Initialization (Recommended)

Modify policies to support lazy initialization:

```python
class ACT(Policy):
    def __init__(self, dim_model=512, chunk_size=100, **config_kwargs):
        super().__init__()
        self._config_kwargs = config_kwargs
        self._lazy_init = True  # Flag for lazy initialization
        self.save_hyperparameters()

    def setup(self, stage: str) -> None:
        """Called by Lightning after DataModule setup."""
        if self._lazy_init and self.trainer.datamodule:
            # Now we can access dataset features
            dm = self.trainer.datamodule
            features = dataset_to_policy_features(dm.dataset_train.meta.features)
            stats = dm.dataset_train.meta.stats

            # Initialize LeRobot policy now
            self._initialize_lerobot_policy(features, stats, **self._config_kwargs)
            self._lazy_init = False
```

**Pros**: YAML configs would work
**Cons**: Requires refactoring all LeRobot wrappers

### Solution 2: Custom LightningCLI Subclass

Create a custom CLI that instantiates in the correct order:

```python
class GetiActionCLI(LightningCLI):
    def instantiate_classes(self):
        # 1. Instantiate DataModule first
        self.datamodule = self.instantiate_datamodule()
        self.datamodule.setup("fit")

        # 2. Inject dataset features into model config
        features = dataset_to_policy_features(self.datamodule.dataset_train.meta.features)
        self.config.model.init_args.input_features = features
        self.config.model.init_args.output_features = features
        self.config.model.init_args.stats = self.datamodule.dataset_train.meta.stats

        # 3. Now instantiate model with features
        self.model = self.instantiate_model()
```

**Pros**: No policy refactoring needed
**Cons**: Complex, fragile, might break Lightning updates

### Solution 3: Document Limitations (Current Approach)

Update configs to be clear they're **templates**, not working configs:

```yaml
# configs/lerobot_act_template.yaml
# ⚠️ WARNING: This is a TEMPLATE, not a working configuration
# ⚠️ You must use Python code to configure features from dataset
# ⚠️ See examples/train_act.py for working training script

model:
  class_path: getiaction.policies.lerobot.ACT
  init_args:
    # ⚠️ REQUIRED (cannot be in YAML): input_features, output_features, stats
    # These must be extracted from dataset in Python
    dim_model: 512
    chunk_size: 100
```

**Pros**: Honest about limitations, no code changes needed
**Cons**: YAML configs remain non-functional

## Recommendations

### Immediate Actions

1. **Rename YAML files** to indicate they're templates:
   - `lerobot_act.yaml` → `lerobot_act_template.yaml`
   - Add prominent warnings in each file

2. **Create working example scripts**:
   - `examples/train_act.py` - Working ACT training script
   - `examples/train_diffusion.py` - Working Diffusion training script
   - Document the 4-step process (load dataset → extract features → create policy → train)

3. **Update documentation**:
   - Mark YAML approach as "Not Supported for LeRobot Policies"
   - Point users to Python scripts for LeRobot training
   - Keep YAML working for Dummy policies

### Long-term Actions

1. **Implement Solution 1 (Two-Stage Initialization)**:
   - Modify ACT, Diffusion, VQ-BeT wrappers
   - Add `setup()` method for lazy initialization
   - Enable YAML configs to work properly

2. **Add CLI tests**:
   - Test that YAML configs actually work end-to-end
   - Catch regressions early

3. **Consider universal wrapper enhancement**:
   - Make `UniversalLeRobotPolicy` handle auto-configuration better
   - It might already support lazy init better than direct wrappers

## Status of Config Files

| Config File | Status | Notes |
|------------|--------|-------|
| `train_working.yaml` | ❌ Broken | Parser errors |
| `train_dummy_*.yaml` | ❌ Broken | Parser errors |
| `lerobot_act.yaml` | ❌ Broken | Missing features |
| `lerobot_act_simple.yaml` | ❌ Broken | Missing features |
| `lerobot_diffusion_*.yaml` | ❌ Broken | Missing features |
| `lerobot_vqbet_*.yaml` | ❌ Broken | Missing features |

**0/9 configs work** via `getiaction fit` command.

## Conclusion

**YAML configuration is NOT a viable training method** for LeRobot policies currently.

Users **must** use Python scripts that follow this pattern:

1. Load dataset → 2. Extract features → 3. Create policy → 4. Train

The integration tests prove the **core functionality works perfectly** - this is purely a configuration/CLI limitation.

**Action Required**: Either implement two-stage initialization (Solution 1) or clearly document that YAML configs don't work and provide working Python examples instead.
