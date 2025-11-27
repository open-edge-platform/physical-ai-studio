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
    ├── mixin.py                 # LeRobotFromConfig mixin (configuration methods)
    ├── act.py                   # Explicit ACT wrapper
    ├── diffusion.py             # Explicit diffusion wrapper
    ├── universal.py             # Universal wrapper
    └── README.md                # Module documentation
```

**Note**: For data module architecture and format conversion details, see
[LeRobot Data Integration](../data/lerobot.md).

### Implementation Components

#### 1. LeRobotFromConfig Mixin

All LeRobot policies inherit from the `LeRobotFromConfig` mixin, which provides:

```python
class ACT(Policy, LeRobotFromConfig):
    """ACT with full configuration flexibility."""
    pass

class Diffusion(Policy, LeRobotFromConfig):
    """Diffusion with full configuration flexibility."""
    pass
```

**Provided Methods**:

- `from_config(config)` - Auto-detect and load any config format
- `from_dict(config_dict)` - Load from dictionary
- `from_yaml(yaml_path)` - Load from YAML file
- `from_pydantic(model)` - Load from Pydantic model
- `from_dataclass(config)` - Load from dataclass
- `from_lerobot_config(config)` - Load from LeRobot's native config

**Key Features**:

- Type-safe configuration
- Automatic format detection
- LeRobot config compatibility
- Extensible for new formats

#### 2. Explicit Wrapper (ACT)

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

### Configuration Methods

All LeRobot policies support multiple configuration
approaches through the `LeRobotFromConfig` mixin:

#### 1. Direct Instantiation (Recommended for Python)

```python
from getiaction.policies.lerobot import ACT

# Full IDE support with autocomplete
policy = ACT(
    dim_model=512,
    chunk_size=100,
    n_action_steps=100,
    use_vae=True,
    learning_rate=1e-5,
)
```

#### 2. From Dictionary

```python
config_dict = {
    "dim_model": 512,
    "chunk_size": 100,
    "n_action_steps": 100,
    "use_vae": True,
    "learning_rate": 1e-5,
}
policy = ACT.from_dict(config_dict)
```

#### 3. From YAML File

```python
# configs/act_config.yaml
# dim_model: 512
# chunk_size: 100
# ...

policy = ACT.from_config("configs/act_config.yaml")
```

#### 4. From LeRobot Config (Advanced)

```python
from lerobot.policies.act.configuration_act import ACTConfig as LeRobotACTConfig

# Create LeRobot's native config
lerobot_config = LeRobotACTConfig(
    dim_model=512,
    chunk_size=100,
    use_vae=True,
)

# Use it directly with our wrapper
policy = ACT.from_lerobot_config(lerobot_config, learning_rate=1e-5)
# or auto-detect:
policy = ACT.from_config(lerobot_config, learning_rate=1e-5)
```

#### 5. From Pydantic Model

```python
from pydantic import BaseModel

class ACTConfigModel(BaseModel):
    dim_model: int = 512
    chunk_size: int = 100
    learning_rate: float = 1e-5

config_model = ACTConfigModel()
policy = ACT.from_pydantic(config_model)
```

#### 6. Universal Wrapper with Config

```python
from getiaction.policies.lerobot import LeRobotPolicy

# Method 1: With config_kwargs
policy = LeRobotPolicy(
    policy_name="diffusion",
    config_kwargs={
        "horizon": 16,
        "n_action_steps": 8,
    },
    learning_rate=1e-4,
)

# Method 2: Direct kwargs (Python usage)
policy = LeRobotPolicy(
    policy_name="diffusion",
    horizon=16,  # Passed as kwargs
    n_action_steps=8,
    learning_rate=1e-4,
)
```

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
from getiaction.train import Trainer

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

trainer = Trainer(max_epochs=100)
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

### Configuration Flexibility

GetiAction policies support **all LeRobot parameters** through two mechanisms:

#### 1. Explicit Parameters

All commonly used parameters are explicitly defined in the `__init__` signature:

```python
policy = ACT(
    dim_model=512,           # Explicit parameter
    chunk_size=100,          # Explicit parameter
    n_encoder_layers=4,      # Explicit parameter
)
```

**Benefits**:

- Full IDE autocomplete
- Type hints and validation
- Easy to discover parameters

#### 2. Additional Parameters via kwargs

Any LeRobot parameter not explicitly listed can be passed via `**kwargs`:

```python
policy = ACT(
    dim_model=512,
    chunk_size=100,
    # These are not in the explicit parameter list but work via kwargs:
    feedforward_activation="gelu",
    pre_norm=True,
    attention_dropout=0.2,
)
```

**Benefits**:

- Access to ALL LeRobot parameters
- No need to update wrapper for new LeRobot features
- Forward compatibility

#### 3. Mixing Configuration Methods

You can combine different approaches:

```python
# Start with YAML
policy = ACT.from_config("base_config.yaml")

# Or start with LeRobot config and override
from lerobot.policies.act.configuration_act import ACTConfig
lerobot_config = ACTConfig.from_pretrained("lerobot/act_default")
policy = ACT.from_lerobot_config(
    lerobot_config,
    learning_rate=1e-4,  # Override/add parameters
)
```

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
