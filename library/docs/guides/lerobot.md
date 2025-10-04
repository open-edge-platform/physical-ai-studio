# LeRobot Integration - Usage Guide

This guide provides practical examples for using LeRobot policies with GetiAction.

## Quick Start

### Installation

```bash
# Install with LeRobot support
pip install getiaction[lerobot]

# Or install separately
pip install getiaction
pip install lerobot
```

### Check Installation

```python
from getiaction.policies import lerobot

print(f"LeRobot available: {lerobot.is_available()}")
print(f"Available policies: {lerobot.list_available_policies()}")
```

## Training Workflows

### Option 1: LightningCLI (Recommended)

#### Simple Configuration

Create `configs/my_act_config.yaml`:

```yaml
model:
  class_path: getiaction.policies.lerobot.ACT
  init_args:
    dim_model: 512
    chunk_size: 100
    n_action_steps: 100

data:
  class_path: getiaction.data.lerobot.LeRobotDataModule
  init_args:
    repo_id: "lerobot/pusht"
    batch_size: 32

trainer:
  max_epochs: 100
  accelerator: auto
  devices: 1
```

Train:

```bash
getiaction fit --config configs/my_act_config.yaml
```

#### Override from Command Line

```bash
# Change hyperparameters
getiaction fit \
  --config configs/my_act_config.yaml \
  --model.init_args.dim_model 1024 \
  --model.init_args.learning_rate 5e-5 \
  --trainer.max_epochs 200

# Use different dataset
getiaction fit \
  --config configs/my_act_config.yaml \
  --data.init_args.repo_id "lerobot/aloha_sim_insertion_human"
```

### Option 2: Python API

#### Basic Training

```python
import lightning as L
from getiaction.policies.lerobot import ACT
from getiaction.data.lerobot import LeRobotDataModule

# Setup data
datamodule = LeRobotDataModule(
    repo_id="lerobot/pusht",
    batch_size=32,
    num_workers=4,
)

# Setup policy
policy = ACT(
    dim_model=512,
    chunk_size=100,
    n_action_steps=100,
    learning_rate=1e-4,
)

# Train
trainer = L.Trainer(
    max_epochs=100,
    accelerator="auto",
    devices=1,
    log_every_n_steps=10,
)

trainer.fit(policy, datamodule)
```

#### Advanced Training with Callbacks

```python
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

# Setup callbacks
checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",
    filename="act-{epoch:02d}-{val_loss:.2f}",
    save_top_k=3,
    monitor="val/loss",
    mode="min",
)

early_stop_callback = EarlyStopping(
    monitor="val/loss",
    patience=10,
    mode="min",
)

# Train with callbacks
trainer = L.Trainer(
    max_epochs=200,
    callbacks=[checkpoint_callback, early_stop_callback],
    accelerator="auto",
    devices=1,
)

trainer.fit(policy, datamodule)
```

### Option 3: Using Native LeRobot Data

For full compatibility with LeRobot datasets:

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import dataset_to_policy_features
from torch.utils.data import DataLoader

# Load dataset
dataset = LeRobotDataset("lerobot/pusht")
features = dataset_to_policy_features(dataset.meta.features)
stats = dataset.meta.stats

# Create dataloaders
train_loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,
    shuffle=True,
)

# Create policy with proper features and stats
policy = ACT(
    input_features=features,
    output_features=features,
    stats=stats,
    dim_model=512,
    chunk_size=100,
    n_action_steps=100,
)

# Train
trainer.fit(policy, train_dataloaders=train_loader)
```

## Using Different Policies

### ACT (Explicit Wrapper)

```python
from getiaction.policies.lerobot import ACT

policy = ACT(
    dim_model=512,
    chunk_size=100,
    n_action_steps=100,
    n_encoder_layers=4,
    n_decoder_layers=1,
    vision_backbone="resnet18",
    use_vae=True,
    latent_dim=32,
)
```

### Diffusion (Universal Wrapper)

```python
from getiaction.policies.lerobot import Diffusion

policy = Diffusion(
    down_dims=[512, 1024, 2048],
    kernel_size=5,
    n_action_steps=100,
    num_inference_steps=10,
)
```

### VQBeT (Universal Wrapper)

```python
from getiaction.policies.lerobot import VQBeT

policy = VQBeT(
    n_vqvae_training_steps=10000,
    vqvae_n_embed=16,
    dim_model=512,
    n_action_steps=100,
)
```

### Dynamic Policy Selection

```python
from getiaction.policies.lerobot import LeRobotPolicy

# Select policy at runtime
policy_name = "diffusion"  # Could come from config/CLI

policy = LeRobotPolicy(
    policy_name=policy_name,
    down_dims=[512, 1024, 2048],
    n_action_steps=100,
)
```

## Inference

### Load Trained Model

```python
# From checkpoint
policy = ACT.load_from_checkpoint("checkpoints/best.ckpt")
policy.eval()

# From pretrained (if saved)
policy = ACT.load_from_checkpoint("path/to/model.ckpt")
```

### Make Predictions

```python
import torch

# Prepare observation
observation = {
    "observation.image": torch.randn(1, 3, 96, 96),
    "observation.state": torch.randn(1, 2),
}

# Get action
with torch.no_grad():
    action = policy.select_action(observation)

print(f"Predicted action shape: {action.shape}")
```

### Use in Environment

```python
import gymnasium as gym

env = gym.make("PushT-v0")
policy.eval()

observation, info = env.reset()

for step in range(1000):
    # Format observation for policy
    obs_dict = {
        "observation.image": torch.from_numpy(observation["image"]).unsqueeze(0),
        "observation.state": torch.from_numpy(observation["state"]).unsqueeze(0),
    }

    # Get action
    with torch.no_grad():
        action = policy.select_action(obs_dict)

    # Execute in environment
    observation, reward, terminated, truncated, info = env.step(action[0].numpy())

    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

## Configuration Examples

### Minimal ACT Config

```yaml
# configs/act_minimal.yaml
model:
  class_path: getiaction.policies.lerobot.ACT
  init_args:
    dim_model: 256
    chunk_size: 10
    n_action_steps: 10

data:
  class_path: getiaction.data.lerobot.LeRobotDataModule
  init_args:
    repo_id: "lerobot/pusht"
    batch_size: 16

trainer:
  max_epochs: 50
```

### Full ACT Config with Callbacks

```yaml
# configs/act_full.yaml
model:
  class_path: getiaction.policies.lerobot.ACT
  init_args:
    dim_model: 512
    chunk_size: 100
    n_action_steps: 100
    n_encoder_layers: 4
    n_decoder_layers: 1
    n_heads: 8
    dim_feedforward: 3200
    dropout: 0.1
    kl_weight: 10.0
    vision_backbone: "resnet18"
    use_vae: true
    latent_dim: 32
    learning_rate: 0.0001

data:
  class_path: getiaction.data.lerobot.LeRobotDataModule
  init_args:
    repo_id: "lerobot/pusht"
    batch_size: 32
    num_workers: 4

trainer:
  max_epochs: 200
  accelerator: auto
  devices: 1
  log_every_n_steps: 10
  precision: 16-mixed

  callbacks:
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args:
        dirpath: checkpoints
        filename: "act-{epoch:02d}-{val_loss:.2f}"
        save_top_k: 3
        monitor: "val/loss"
        mode: "min"

    - class_path: lightning.pytorch.callbacks.EarlyStopping
      init_args:
        monitor: "val/loss"
        patience: 20
        mode: "min"

  logger:
    class_path: lightning.pytorch.loggers.WandbLogger
    init_args:
      project: "getiaction-lerobot"
      name: "act-pusht"
```

### Diffusion Config

```yaml
# configs/diffusion.yaml
model:
  class_path: getiaction.policies.lerobot.Diffusion
  init_args:
    down_dims: [512, 1024, 2048]
    kernel_size: 5
    n_action_steps: 100
    num_inference_steps: 10
    diffusion_step_embed_dim: 128

data:
  class_path: getiaction.data.lerobot.LeRobotDataModule
  init_args:
    repo_id: "lerobot/pusht"
    batch_size: 64

trainer:
  max_epochs: 100
  accelerator: auto
  devices: 1
```

## Tips & Tricks

### Performance Optimization

```python
# Use mixed precision
trainer = L.Trainer(
    precision="16-mixed",  # Faster training on modern GPUs
    accelerator="gpu",
    devices=1,
)

# Increase batch size if GPU memory allows
datamodule = LeRobotDataModule(
    repo_id="lerobot/pusht",
    batch_size=64,  # Larger batches = faster training
    num_workers=8,   # More workers = faster data loading
)

# Use compiled models (PyTorch 2.0+)
policy = ACT(...)
policy = torch.compile(policy)  # Faster forward pass
```

### Debugging

```python
# Test with small data
trainer = L.Trainer(
    fast_dev_run=True,  # Run 1 batch to test
    limit_train_batches=10,
    limit_val_batches=5,
)

# Check gradients
trainer = L.Trainer(
    track_grad_norm=2,  # Log gradient norms
    detect_anomaly=True,  # Detect NaN/Inf
)

# Verify output equivalence
from getiaction.policies.lerobot import ACT
from lerobot.policies.act import ACTPolicy
import torch

wrapped = ACT(...)
native = ACTPolicy(...)

# Copy weights
native.load_state_dict(wrapped.lerobot_policy.state_dict())

# Compare
wrapped.eval()
native.eval()
with torch.no_grad():
    wrapped_out = wrapped.select_action(batch)
    native_out = native.select_action(batch)

torch.testing.assert_close(wrapped_out, native_out)
print("✅ Outputs match!")
```

### Experiment Tracking

```python
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger

# Weights & Biases
wandb_logger = WandbLogger(
    project="getiaction-lerobot",
    name="act-experiment-1",
    log_model=True,
)

# TensorBoard
tb_logger = TensorBoardLogger(
    save_dir="logs",
    name="act",
)

trainer = L.Trainer(
    logger=[wandb_logger, tb_logger],  # Use multiple loggers
    log_every_n_steps=10,
)
```

### Hyperparameter Tuning

```python
# Use Optuna with Lightning
import optuna
from optuna.integration import PyTorchLightningPruningCallback

def objective(trial):
    # Suggest hyperparameters
    dim_model = trial.suggest_categorical("dim_model", [256, 512, 1024])
    learning_rate = trial.suggest_loguniform("lr", 1e-5, 1e-3)

    # Create policy
    policy = ACT(
        dim_model=dim_model,
        learning_rate=learning_rate,
    )

    # Train
    trainer = L.Trainer(
        max_epochs=50,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val/loss")],
    )
    trainer.fit(policy, datamodule)

    return trainer.callback_metrics["val/loss"].item()

# Run study
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

print(f"Best params: {study.best_params}")
```

## Troubleshooting

### Import Errors

```python
# Check if LeRobot is installed
from getiaction.policies import lerobot
print(lerobot.is_available())  # Should print True

# If False, install LeRobot
# pip install lerobot
```

### Shape Mismatches

```python
# Make sure to use LeRobot dataset format
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import dataset_to_policy_features

dataset = LeRobotDataset("lerobot/pusht")
features = dataset_to_policy_features(dataset.meta.features)

# Pass features to policy
policy = ACT(
    input_features=features,
    output_features=features,
)
```

### Loss Not Decreasing

```python
# Check learning rate
policy = ACT(
    learning_rate=1e-4,  # Try different values: 1e-3, 1e-5
)

# Check if data is normalized
datamodule = LeRobotDataModule(
    repo_id="lerobot/pusht",
    # Make sure dataset has proper stats
)

# Verify forward pass works
batch = next(iter(datamodule.train_dataloader()))
output = policy(batch)
print(f"Loss: {output['loss']}")
```

## Next Steps

- See [Design Documentation](../design/policy/lerobot.md) for architecture details
- Check [Module README](../../../src/getiaction/policies/lerobot/README.md) for API reference
- Browse example configs in `library/configs/lerobot_*.yaml`
- Run tests to verify installation: `pytest tests/test_lerobot_*.py`

## References

- [LeRobot Documentation](https://github.com/huggingface/lerobot)
- [PyTorch Lightning Documentation](https://lightning.ai/docs/pytorch/stable/)
- [GetiAction Documentation](../../README.md)
