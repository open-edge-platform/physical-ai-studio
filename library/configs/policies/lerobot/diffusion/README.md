# Diffusion Policy Configuration Files

This directory contains YAML configuration files for training Diffusion Policy models using the GetiAction framework.

## Available Configurations

### 1. `pusht_basic.yaml` - Basic Configuration

**Use case:** Standard training setup with sensible defaults
**Dataset:** PushT
**Training time:** ~6-8 hours on single GPU
**Performance:** Good baseline performance

Key features:

- ResNet18 vision backbone
- Standard U-Net dimensions [256, 512, 1024]
- 100 diffusion training steps
- Batch size: 64

```bash
getiaction fit --config configs/policies/lerobot/diffusion/pusht_basic.yaml
```

### 2. `pusht_advanced.yaml` - Advanced Configuration

**Use case:** Full documentation of all available parameters
**Dataset:** PushT

This configuration demonstrates every configurable parameter with detailed comments. Use this as a reference when customizing your training setup.

```bash
getiaction fit --config configs/policies/lerobot/diffusion/pusht_advanced.yaml
```

### 3. `pusht_fast.yaml` - Fast Training

**Use case:** Quick prototyping and testing
**Dataset:** PushT
**Training time:** ~2-3 hours on single GPU
**Performance:** Reduced performance, but fast iteration

Key optimizations:

- Smaller model (down_dims: [128, 256, 512])
- Fewer diffusion steps (50 training, 5 inference)
- DDIM scheduler for faster sampling
- Larger batch size (128)
- No checkpointing

```bash
getiaction fit --config configs/policies/lerobot/diffusion/pusht_fast.yaml
```

### 4. `pusht_cpu.yaml` - CPU Training

**Use case:** Development without GPU, debugging
**Dataset:** PushT (subset: 5 episodes)
**Training time:** Very slow, not recommended for full training

Features:

- Small batch size (16)
- Limited data subset
- Optimized for CPU execution
- No mixed precision

```bash
getiaction fit --config configs/policies/lerobot/diffusion/pusht_cpu.yaml
```

### 5. `pusht_multigpu.yaml` - Multi-GPU Training

**Use case:** Large-scale training with multiple GPUs
**Dataset:** PushT
**Training time:** ~2-3 hours on 4 GPUs
**Performance:** Best performance with largest model

Features:

- ResNet50 backbone
- Large U-Net [512, 1024, 2048]
- Distributed Data Parallel (DDP) strategy
- 4 GPU configuration
- Scaled learning rate
- Synchronized batch normalization
- Effective batch size: 512 (64 *4 GPUs* 2 accumulation)

```bash
getiaction fit --config configs/policies/lerobot/diffusion/pusht_multigpu.yaml
```

### 6. `aloha.yaml` - ALOHA Bimanual Manipulation

**Use case:** Bimanual robot manipulation tasks
**Dataset:** ALOHA sim transfer cube
**Training time:** ~12-16 hours on single GPU

Features:

- ResNet34 backbone for complex scenes
- Separate RGB encoders per camera
- Higher resolution images (224x224)
- More spatial keypoints (64)
- Optimized for multi-camera setups

```bash
getiaction fit --config configs/policies/lerobot/diffusion/aloha.yaml
```

## Configuration Structure

All configurations follow this structure:

```yaml
model:
  class_path: getiaction.policies.lerobot.diffusion.Diffusion
  init_args:
    # Temporal parameters
    n_obs_steps: 2
    horizon: 16
    n_action_steps: 8

    # Vision backbone
    vision_backbone: "resnet18"
    # ... more parameters

    # Diffusion process
    noise_scheduler_type: "DDPM"
    # ... more parameters

    # Optimization
    learning_rate: 1.0e-4
    # ... more parameters

data:
  class_path: getiaction.data.lerobot.LeRobotDataModule
  init_args:
    repo_id: "lerobot/pusht"
    train_batch_size: 64
    delta_timestamps:
      action: [0.0, 0.1, ..., 1.5]  # Must match horizon length

trainer:
  max_epochs: 3000
  accelerator: gpu
  devices: 1
  # ... more trainer config
```

## Key Parameters Explained

### Temporal Configuration

- **n_obs_steps**: Number of observation frames to stack (typically 1-2)
- **horizon**: Prediction horizon, must match `len(delta_timestamps["action"])`
- **n_action_steps**: Number of action steps to execute from the predicted trajectory

### Vision Backbone

- **vision_backbone**: ResNet variant (resnet18, resnet34, resnet50)
- **crop_shape**: Input image size after cropping
- **pretrained_backbone_weights**: Use ImageNet pretrained weights for faster convergence

### U-Net Architecture

- **down_dims**: Channel dimensions for U-Net blocks (affects model capacity)
- **kernel_size**: Convolution kernel size (3 or 5)
- **diffusion_step_embed_dim**: Timestep embedding dimension

### Diffusion Process

- **noise_scheduler_type**: DDPM (standard) or DDIM (faster inference)
- **num_train_timesteps**: Diffusion steps during training (50-100)
- **num_inference_steps**: Steps during inference (fewer = faster)
- **beta_schedule**: Noise schedule (linear, squaredcos_cap_v2)

### Optimization

- **learning_rate**: Base learning rate (1e-4 is a good default)
- **scheduler_name**: LR scheduler (cosine, constant, linear)
- **optimizer_weight_decay**: L2 regularization strength

## Customization Tips

### Override Parameters

You can override any parameter from the command line:

```bash
getiaction fit --config configs/policies/lerobot/diffusion/pusht_basic.yaml \
  --model.init_args.learning_rate 5e-4 \
  --model.init_args.down_dims [512,1024,2048] \
  --data.init_args.train_batch_size 32
```

### Match Horizon and Delta Timestamps

**Important:** The `horizon` parameter must equal the length of `delta_timestamps["action"]`:

```yaml
model:
  init_args:
    horizon: 16  # Must be 16

data:
  init_args:
    delta_timestamps:
      action: [0.0, 0.1, ..., 1.5]  # 16 timesteps
```

### Adjust for Your Dataset

When using a different dataset:

1. Change `repo_id` to your dataset
2. Adjust `horizon` and `delta_timestamps` for your control frequency
3. Modify `crop_shape` based on your camera resolution
4. Set `use_separate_rgb_encoder_per_camera: true` if using multiple cameras

### Performance vs Speed Trade-offs

- **Larger model → Better performance, slower training**
  - Increase `down_dims`, use resnet50, more keypoints
- **Smaller model → Faster training, reduced performance**
  - Decrease `down_dims`, use resnet18, fewer diffusion steps
- **Mixed precision → 2x faster training, minimal performance impact**
  - Set `trainer.precision: 16-mixed`

## Monitoring Training

Track training progress with:

- Loss curves (should decrease steadily)
- Validation metrics (check every N epochs)
- Generated trajectories (visualize predicted actions)

## Troubleshooting

### Out of Memory (OOM)

1. Reduce `train_batch_size`
2. Reduce `down_dims` (smaller U-Net)
3. Reduce `crop_shape` (smaller images)
4. Use `precision: 16-mixed`

### Training Instability

1. Lower `learning_rate`
2. Add `gradient_clip_val: 10.0`
3. Increase `scheduler_warmup_steps`
4. Check `delta_timestamps` alignment

### Slow Training

1. Use `precision: 16-mixed`
2. Increase `train_batch_size` (if memory allows)
3. Use fewer `num_train_timesteps`
4. Consider multi-GPU training

## References

- [Diffusion Policy Paper](https://diffusion-policy.cs.columbia.edu/)
- [LeRobot Documentation](https://github.com/huggingface/lerobot)
- [GetiAction Documentation](../../README.md)
