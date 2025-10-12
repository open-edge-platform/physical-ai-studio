# LeRobot vs getiaction: Pipeline Comparison

## Current Status: ✅ Feature Parity with Lightning-Friendly Design

### LeRobot's Approach (Manual Training Loop)

```python
# LeRobot: Manual training with eval_policy() function
def train_lerobot():
    # 1. Setup
    policy = DiffusionPolicy(config)
    dataset = LeRobotDataset("lerobot/pusht")
    optimizer = torch.optim.Adam(policy.parameters())
    dataloader = DataLoader(dataset, batch_size=64)

    # 2. Manual training loop
    for batch in dataloader:
        loss, _ = policy.forward(batch)
        loss.backward()
        optimizer.step()

    # 3. Save checkpoint
    policy.save_pretrained("outputs/")

    # 4. Separate evaluation script
    from lerobot.rl.eval_policy import eval_policy
    env = make_robot_env(config)
    eval_policy(env, policy, n_episodes=10)
```

**Characteristics:**

- ❌ Manual training loop
- ❌ No automatic validation during training
- ❌ Separate eval script
- ❌ No integration with experiment tracking
- ✅ Simple and straightforward
- ✅ Full control

---

### getiaction's Approach (Lightning-Integrated)

```python
# getiaction: Lightning-integrated with automatic validation
from getiaction.data import DataModule
from getiaction.data.lerobot import LeRobotDataModule
from getiaction.gyms import PushTGym
from getiaction.policies.lerobot import LeRobotPolicy
from getiaction.train import Trainer
from getiaction.train.callbacks import GymEvaluation

def train_getiaction():
    # 1. Setup data
    datamodule = LeRobotDataModule(
        repo_id="lerobot/pusht",
        train_batch_size=64,
    )

    # 2. Add validation gym
    eval_gym = PushTGym()
    full_datamodule = DataModule(
        train_dataset=datamodule.train_dataset,
        train_batch_size=64,
        val_gyms=eval_gym,              # Validation during training
        num_rollouts_val=10,            # 10 episodes per validation
        test_gyms=eval_gym,             # Post-training evaluation
        num_rollouts_test=50,           # 50 episodes for final test
    )

    # 3. Setup policy
    policy = LeRobotPolicy(
        policy_name="diffusion",
        learning_rate=1e-4,
    )

    # 4. Setup trainer with callbacks
    trainer = Trainer(
        max_epochs=100,
        callbacks=[GymEvaluation()],
        val_check_interval=0.5,        # Validate twice per epoch
        logger=True,                    # TensorBoard/WandB logging
    )

    # 5. Train with automatic validation
    trainer.fit(policy, full_datamodule)
    # Validation runs automatically during training!

    # 6. Post-training evaluation
    trainer.test(policy, full_datamodule)
    # Or standalone validation:
    # trainer.validate(policy, full_datamodule)
```

**Characteristics:**

- ✅ Lightning training loop (DDP, gradient accumulation, mixed precision)
- ✅ Automatic validation during training
- ✅ Integrated experiment tracking (TensorBoard, WandB)
- ✅ Callbacks for extensibility
- ✅ Separate validation and test phases
- ✅ Same results as LeRobot
- ✅ Lightning-friendly

---

## Feature Comparison

| Feature               | LeRobot            | getiaction              | Notes                                          |
| --------------------- | ------------------ | ----------------------- | ---------------------------------------------- |
| **Training**          |
| Training loop         | Manual             | Lightning               | getiaction: automatic DDP, mixed precision     |
| Loss computation      | ✅                 | ✅                      | Same `policy.forward(batch)`                   |
| Optimizer             | Manual setup       | Lightning managed       | getiaction: `configure_optimizers()`           |
| Gradient accumulation | Manual             | Lightning               | getiaction: `accumulate_grad_batches`          |
| Mixed precision       | Manual             | Lightning               | getiaction: `precision="16-mixed"`             |
| **Validation**        |
| During training       | ❌                 | ✅                      | getiaction: automatic via `val_check_interval` |
| Gym rollouts          | Separate script    | Integrated              | getiaction: `GymEvaluation` callback           |
| Metrics logging       | Manual             | Automatic               | getiaction: Lightning logger                   |
| Dataset validation    | ❌                 | ✅                      | getiaction: can validate on held-out dataset   |
| **Evaluation**        |
| Post-training eval    | ✅ `eval_policy()` | ✅ `trainer.test()`     | Both work                                      |
| Standalone eval       | ✅                 | ✅ `trainer.validate()` | Both work                                      |
| Success rate          | ✅                 | ✅                      | Same metrics                                   |
| Episode rewards       | ✅                 | ✅                      | Same metrics                                   |
| **Infrastructure**    |
| Distributed training  | Manual             | Lightning DDP           | getiaction: `trainer = Trainer(devices=4)`     |
| Checkpointing         | Manual             | Lightning               | getiaction: `ModelCheckpoint` callback         |
| Early stopping        | Manual             | Lightning               | getiaction: `EarlyStopping` callback           |
| Experiment tracking   | Manual             | Integrated              | getiaction: TensorBoard/WandB/MLflow           |
| **Results**           |
| Final model quality   | ✅                 | ✅                      | **Same results**                               |
| Training loss         | ✅                 | ✅                      | Same computation                               |
| Validation metrics    | ✅                 | ✅                      | Same rollout logic                             |

---

## Can getiaction Reproduce LeRobot Results?

### ✅ YES - With These Equivalences

#### 1. **Same Policy Training**

```python
# LeRobot
loss, _ = policy.forward(batch)

# getiaction (in training_step)
loss = self.lerobot_policy.forward(batch)
```

→ **Same computation, same loss**

#### 2. **Same Evaluation Logic**

```python
# LeRobot's eval_policy()
for _ in range(n_episodes):
    obs, _ = env.reset()
    while True:
        action = policy.select_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)

# getiaction's rollout()
for step in range(max_steps):
    action = policy.select_action(observation)
    observation, reward, terminated, truncated, info = env.step(action)
```

→ **Same rollout logic, same metrics**

#### 3. **Same Environment**

```python
# Both use Gym environments
env = PushTGym()  # or make_robot_env(config)
```

→ **Same environment, same task**

---

## What getiaction Adds (Lightning-Friendly)

### 1. **Automatic Validation During Training**

```python
trainer = Trainer(
    val_check_interval=0.5,  # Validate at 50% and 100% of each epoch
    callbacks=[GymEvaluation()],
)
trainer.fit(policy, datamodule)
```

**Benefits:**

- Monitor real gym performance during training
- Early detection of overfitting
- Track success rate over time
- No need for separate eval scripts

### 2. **Integrated Experiment Tracking**

```python
trainer = Trainer(
    logger=WandbLogger(project="pusht"),
    callbacks=[GymEvaluation()],
)
```

**Logs automatically:**

- `train/loss` - Training loss
- `val/gym/success` - Validation success rate
- `val/gym/reward_mean` - Average reward
- `val/gym/episode_length` - Episode length

### 3. **Production-Ready Features**

```python
trainer = Trainer(
    max_epochs=100,
    devices=4,  # Multi-GPU training
    precision="16-mixed",  # Mixed precision
    callbacks=[
        GymEvaluation(),
        ModelCheckpoint(monitor="val/gym/success"),
        EarlyStopping(monitor="val/gym/success"),
    ],
)
```

### 4. **Flexible Validation**

```python
# Option 1: During training
trainer.fit(policy, datamodule)

# Option 2: Standalone after training
trainer.validate(policy, datamodule)

# Option 3: Final test
trainer.test(policy, datamodule)

# Option 4: Custom evaluation (like LeRobot)
from getiaction.eval import evaluate_policy
stats = evaluate_policy(env, policy, n_episodes=50)
```

---

## Migration Path: LeRobot → getiaction

### Step 1: Keep LeRobot Code As-Is

```python
# Your existing LeRobot training works!
policy = DiffusionPolicy(config)
# ... manual training loop
```

### Step 2: Wrap in Lightning (Optional)

```python
# Wrap for Lightning benefits
from getiaction.policies.lerobot import LeRobotPolicy

policy = LeRobotPolicy(policy_name="diffusion")
# Get automatic DDP, checkpointing, logging, etc.
```

### Step 3: Add Gym Evaluation (Optional)

```python
# Add automatic gym validation during training
datamodule = DataModule(
    train_dataset=...,
    val_gyms=PushTGym(),  # Just add this!
    num_rollouts_val=10,
)

trainer = Trainer(
    callbacks=[GymEvaluation()],  # And this!
)
trainer.fit(policy, datamodule)
```

---

## Summary: Is It Ready?

### ✅ **YES - Feature Complete and Lightning-Friendly**

**What Works:**

1. ✅ Train policies (same loss computation as LeRobot)
2. ✅ Validate during training (gym rollouts)
3. ✅ Test after training (same as LeRobot's eval_policy)
4. ✅ Standalone evaluation (trainer.validate)
5. ✅ Custom evaluation (evaluate_policy function)
6. ✅ Same results as LeRobot
7. ✅ Lightning-integrated (DDP, checkpointing, logging)
8. ✅ Production-ready (callbacks, experiment tracking)

**Key Design Decisions:**

- `val_gyms` → `val_dataloader()` → Lightning validation loop
- `test_gyms` → `test_dataloader()` → Lightning test loop
- `GymEvaluation` callback → runs rollouts automatically
- `_val_dataset`, `_test_dataset` → internal wrappers (GymDataset)

**Not Breaking Changes:**

- LeRobot policies work as-is
- LeRobot datasets work as-is
- Can still use manual training loops
- Can still use eval_policy() function

**The Big Win:**
You get **LeRobot's flexibility + Lightning's infrastructure** without choosing one over the other! 🎯
