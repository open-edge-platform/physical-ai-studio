# Configuration Examples

This directory contains example configurations for training GetiAction policies.

## Quick Start

```bash
# Train with any config
python -m getiaction fit --config configs/train_dummy_class_path.yaml
```

## Available Configs

### `train_dummy_dataclass.yaml`

Basic training config using dataclass pattern. Good starting point for simple experiments.

**Features:**

- Dataclass-based configuration
- Basic callbacks (ModelCheckpoint)
- Simple optimizer setup

**Use when:**

- Learning the system
- Quick experiments
- Type-safe configs preferred

### `train_dummy_class_path.yaml`

Training config using jsonargparse `class_path` pattern.
Recommended for most use cases.

**Features:**

- Dynamic class instantiation
- More callbacks (EarlyStopping, ModelCheckpoint, LearningRateMonitor)
- Flexible optimizer configuration

**Use when:**

- Need to swap components easily
- Configuration-driven experiments
- Maximum flexibility required

### `train_advanced.yaml`

Advanced training config with all features enabled.

**Features:**

- Mixed precision training
- Multiple loggers (TensorBoard, CSV)
- Comprehensive callbacks
- Gradient clipping
- Learning rate scheduling

**Use when:**

- Production training
- Long experiments
- Need full monitoring
- Distributed training

## Configuration Patterns

### Pattern 1: jsonargparse class_path (Recommended)

```yaml
model:
  class_path: getiaction.policies.dummy.policy.Dummy
  init_args:
    model:
      class_path: getiaction.policies.dummy.model.Dummy
      init_args:
        action_shape: [7]
        n_action_steps: 4
```

**Pros:**

- ✅ Dynamic class loading
- ✅ No code changes to swap components
- ✅ Works with any class
- ✅ Standard in Lightning ecosystem

**Cons:**

- ❌ No IDE autocomplete
- ❌ Type errors caught at runtime

### Pattern 2: Dataclass/Pydantic

```yaml
model_config:
  action_shape: [7]
  n_action_steps: 4
  learning_rate: 0.001
```

**Pros:**

- ✅ Type-safe at definition time
- ✅ IDE autocomplete
- ✅ Clear structure

**Cons:**

- ❌ Less flexible
- ❌ Requires code changes for new components

### Pattern 3: Mixed (Best of Both)

```yaml
# Type-safe config definition
model_params:
  action_shape: [7]
  n_action_steps: 4

# Dynamic class instantiation
model:
  class_path: getiaction.policies.dummy.policy.Dummy
  init_args:
    model:
      class_path: getiaction.policies.dummy.model.Dummy
      init_args: ${model_params} # Reference typed config
```

**Pros:**

- ✅ Type safety where needed
- ✅ Flexibility where needed
- ✅ Best of both worlds

## Customization

### Change Model

```yaml
model:
  class_path: getiaction.policies.YOUR_POLICY.YourPolicy
  init_args:
    # Your policy-specific args
```

### Change Dataset

```yaml
data:
  class_path: getiaction.data.YOUR_DATA.YourDataModule
  init_args:
    # Your data-specific args
```

### Change Optimizer

```yaml
model:
  init_args:
    optimizer:
      class_path: torch.optim.SGD # Or any PyTorch optimizer
      init_args:
        lr: 0.01
        momentum: 0.9
```

### Add Callbacks

```yaml
trainer:
  callbacks:
    - class_path: lightning.pytorch.callbacks.YourCallback
      init_args:
        your_param: value
```

## Tips

1. **Start Simple**: Use `train_dummy_class_path.yaml` as template
2. **Print Config**: Use `--print_config` to see all options
3. **Override**: Use CLI args to override specific values
4. **Validate**: Use `fast_dev_run=true` to catch errors quickly
5. **Version Control**: Keep configs in git for reproducibility

## Common Overrides

```bash
# Change epochs
--trainer.max_epochs 200

# Change batch size
--data.init_args.train_batch_size 64

# Change learning rate
--model.init_args.optimizer.init_args.lr 0.0001

# Enable GPU
--trainer.accelerator gpu --trainer.devices 1

# Mixed precision
--trainer.precision 16-mixed
```

## Examples

### Quick Experiment

```bash
python -m getiaction fit \
    --config configs/train_dummy_class_path.yaml \
    --trainer.fast_dev_run true
```

### Full Training Run

```bash
python -m getiaction fit \
    --config configs/train_advanced.yaml \
    --trainer.max_epochs 200
```

### Hyperparameter Sweep

```bash
for lr in 0.001 0.0001 0.00001; do
    python -m getiaction fit \
        --config configs/train_dummy_class_path.yaml \
        --model.init_args.optimizer.init_args.lr $lr \
        --trainer.logger.init_args.name "lr_$lr"
done
```

## Troubleshooting

### Config Validation Error

```bash
# Check config syntax
python -m getiaction fit --config CONFIG --print_config
```

### Import Error

Make sure all classes are importable:

```python
python -c "from getiaction.policies.dummy.policy import Dummy"
```

### Type Error

jsonargparse validates types. Check your config matches class signatures.

## More Information

- Full documentation: `../library/docs/cli_usage.md`
- Quick reference: `../README_CLI.md`
