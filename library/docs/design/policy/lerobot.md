# LeRobot Policy Integration

## Overview

GetiAction provides seamless integration with LeRobot policies through:

1. **Explicit Wrappers** - Full parameter definitions with IDE support
   (Recommended)
2. **Universal Wrapper** - Flexible runtime policy selection (Advanced)

Both approaches provide:

- ✅ **Verified output equivalence** with native LeRobot
- ✅ Full Lightning integration
- ✅ Training, validation, and inference support
- ✅ Seamless PyTorch Lightning Trainer compatibility
- ✅ Automatic data format handling (see [Data Integration](../data/lerobot.md))

## Design Pattern

### Lightning-First with Third-Party Framework Support

```text
┌───────────────────────────────────────────────┐
│            GetiAction (Lightning)             │
│   ┌───────────────────────────────────────┐   │
│   │  GetiAction Policy (LightningModule)  │   │
│   │  ┌─────────────────────────────────┐  │   │
│   │  │   LeRobot Native Policy         │  │   │
│   │  │   (Thin Delegation)             │  │   │
│   │  └─────────────────────────────────┘  │   │
│   └───────────────────────────────────────┘   │
└───────────────────────────────────────────────┘
```

**Key Principles**:

1. **No Reimplementation** - Delegate to native LeRobot
2. **Thin Wrapper** - Only Lightning interface code
3. **Transparent** - All LeRobot features preserved
4. **Verified Equivalence** - Outputs match native LeRobot

## Architecture

### File Structure

```text
library/src/getiaction/
└── policies/lerobot/
    ├── __init__.py              # Conditional imports, availability checks
    ├── act.py                   # Explicit ACT wrapper
    ├── diffusion.py             # Explicit diffusion wrapper
    ├── universal.py             # Universal wrapper
    └── README.md                # Module documentation
```

**Note**: For data module architecture and format conversion details, see
[LeRobot Data Integration](../data/lerobot.md).

### Implementation Components

#### 1. Explicit Wrapper (ACT)

```python
class ACT(LightningModule):
    """Explicit wrapper for LeRobot ACT policy.

    Features:
    - Full parameter definitions with type hints
    - IDE autocomplete support
    - Compile-time type checking
    - Direct YAML configuration
    - Automatic data format handling
    """

    def __init__(
        self,
        input_features: dict,
        output_features: dict,
        dim_model: int = 512,
        chunk_size: int = 100,
        # ... 16 total parameters with full typing
    ):
        super().__init__()
        # Delegate to LeRobot
        config = ACTConfig(...)
        self.lerobot_policy = ACTPolicy(config, dataset_stats=stats)

    def forward(self, batch: dict) -> dict:
        """Delegate to LeRobot policy with automatic format conversion."""
        batch = FormatConverter.to_lerobot_dict(batch)
        return self.lerobot_policy.forward(batch)

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        """Lightning training interface with format conversion."""
        batch = FormatConverter.to_lerobot_dict(batch)
        output = self.lerobot_policy.forward(batch)
        loss = output["loss"]
        self.log("train/loss", loss)
        return loss
```

**Key Features**:

- Thin delegation to native LeRobot policies
- Automatic data format conversion (see [Data Integration](../data/lerobot.md))
- All methods support both GetiAction and LeRobot data formats

#### 2. Universal Wrapper

```python
class LeRobotPolicy(LightningModule):
    """Universal wrapper supporting all LeRobot policies.

    Supported Policies:
    - act, diffusion, tdmpc, vqbet, sac, ppo, ddpg, dqn, ibc
    """

    def __init__(
        self,
        policy_name: str,
        input_features: dict,
        output_features: dict,
        stats: dict | None = None,
        **policy_kwargs,
    ):
        super().__init__()
        # Dynamic policy creation
        policy_cls = get_policy_class(policy_name)
        config = get_policy_config_class(policy_name)(**policy_kwargs)
        self.lerobot_policy = policy_cls(config, dataset_stats=stats)
```

#### 3. Convenience Aliases

```python
# Create policy-specific classes dynamically
VQBeT = lambda **kwargs: LeRobotPolicy(policy_name="vqbet", **kwargs)
TDMPC = lambda **kwargs: LeRobotPolicy(policy_name="tdmpc", **kwargs)
```

## Usage

### Approach 1: Explicit Wrapper (Recommended)

#### CLI Interface

```bash
# Train with config
getiaction fit --config configs/lerobot_act.yaml

# Override parameters
getiaction fit \
  --config configs/lerobot_act.yaml \
  --model.dim_model 1024 \
  --trainer.max_epochs 200
```

#### Python Interface

```python
from getiaction.policies.lerobot import ACT
# Create policy (full IDE support!)
policy = ACT(
    dim_model=512,              # ← Autocomplete works!
    chunk_size=100,
    n_action_steps=100,
)

# Train with datamodule
from getiaction.data.lerobot import LeRobotDataModule
datamodule = LeRobotDataModule(
    repo_id="lerobot/pusht",
    train_batch_size=8,
)

trainer = L.Trainer(max_epochs=100)
trainer.fit(policy, datamodule)
```

### Approach 2: Universal Wrapper

#### LightningCLI

```yaml
# configs/lerobot_diffusion.yaml
model:
  class_path: getiaction.policies.lerobot.LeRobotPolicy
  init_args:
    policy_name: diffusion
    config_kwargs:
      input_features: ...
      output_features: ...
      # Policy-specific kwargs
      down_dims: [512, 1024, 2048]
      n_action_steps: 100
```

#### Python API

```python
from getiaction.policies.lerobot import LeRobotPolicy, Diffusion

# Method 1: Explicit policy_name
policy = LeRobotPolicy(
    policy_name="diffusion",
    config_kwargs={
        "horizon": 16,
        "n_action_steps": 8,
        "down_dims": [512, 1024, 2048],
    },
    learning_rate=1e-4,
)

# Method 2: Convenience alias (same as above)
policy = Diffusion(
    config_kwargs={
        "horizon": 16,
        "n_action_steps": 8,
        "down_dims": [512, 1024, 2048],
    },
    learning_rate=1e-4,
)
```

## Best Practices

### When to Use Explicit Wrappers

✅ **Use explicit wrappers when**:

- You need IDE autocomplete and type hints
- Working in a team (better code readability)
- Building production systems
- You primarily use 1-2 policies
- You want compile-time type checking

**Available**: ACT, Diffusion (more coming soon)

### When to Use Universal Wrapper

✅ **Use universal wrapper when**:

- You need flexibility to switch policies
- Experimenting with multiple policies
- Building dynamic policy selection systems
- You're comfortable with LeRobot documentation
- You need all 9 policies immediately

**Available**: All LeRobot policies

### Configuration Tips

1. **Start with simple configs**: Use `lerobot_act.yaml` for quick testing
2. **Choose data format wisely**: See [Data Integration](../data/lerobot.md)
   for format details
3. **Copy from LeRobot examples**: Most configs can be adapted directly
4. **Validate output equivalence**: Use test suite for new policies

### Data Format Considerations

For detailed information about data formats and conversion, see the dedicated
[LeRobot Data Integration](../data/lerobot.md) documentation. The key points:

- Policies automatically handle both GetiAction and LeRobot data formats
- Format conversion is transparent and zero-overhead in production
- No manual conversion needed - the wrappers handle this automatically

## Implementation Details

### Why This Works

1. **Thin Delegation Pattern**:
   - Wrapper only adds Lightning interface
   - All computation delegated to LeRobot
   - Zero computational overhead

2. **Weight Preservation**:
   - Direct attribute access to `lerobot_policy`
   - State dict operations pass through
   - Checkpointing works seamlessly

3. **Feature Preservation**:
   - All LeRobot methods accessible via `lerobot_policy`
   - Environment reset, action selection preserved
   - Statistics, normalization handled by LeRobot

## References

- [LeRobot GitHub](https://github.com/huggingface/lerobot)
- [LeRobot Documentation](https://huggingface.co/lerobot)
- [GetiAction Best Practices](../../BEST_PRACTICES_FRAMEWORK_INTEGRATION.md)
- [LeRobot Data Module Documentation](../data/lerobot.md) - For data format details
- Module: `library/src/getiaction/policies/lerobot/`
- Data Module: `library/src/getiaction/data/lerobot/`
- Tests: `library/tests/test_lerobot_*.py`
