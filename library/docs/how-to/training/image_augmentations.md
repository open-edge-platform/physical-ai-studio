# Image Augmentations

## Why Use Image Augmentations?

When fine-tuning a policy on a small number of demonstrations, the model can overfit to the exact visual conditions of your training data. Image augmentations randomly perturb the training images so the policy learns to be robust to variations it will encounter in the real world, such as:

- Different lighting conditions (brightness, shadows, color temperature)
- Camera placement differences between training and deployment
- Visual distractions or background changes

This is especially useful when your training data was recorded under controlled conditions but the robot will operate in varying environments.

## Available Transforms

PhysicalAI provides two custom transforms alongside standard torchvision transforms:

| Transform                                | Source      | Description                                                |
| ---------------------------------------- | ----------- | ---------------------------------------------------------- |
| `physicalai.transforms.RandomChoice`     | PhysicalAI  | Randomly selects N transforms from a pool per forward pass |
| `physicalai.transforms.RandomSharpness`  | PhysicalAI  | Samples a continuous random sharpness factor               |
| `torchvision.transforms.v2.ColorJitter`  | torchvision | Adjusts brightness, contrast, saturation, or hue           |
| `torchvision.transforms.v2.RandomAffine` | torchvision | Applies random rotation and translation                    |

`RandomChoice` wraps a pool of transforms and samples a subset on each forward pass using weighted multinomial sampling without replacement. This produces more diverse augmentations than applying all transforms every time.

## Configuration

Image transforms are configured under `data.init_args.image_transforms` in the training YAML config using the standard `class_path` / `init_args` pattern. They apply to all image observations during training only (not during validation or inference).

### ACT Example

The following configuration randomly applies 3 out of 6 available transforms to each training image. Default values match [LeRobot's augmentation pipeline](https://github.com/huggingface/lerobot).

```yaml
data:
  class_path: physicalai.data.lerobot.LeRobotDataModule
  init_args:
    repo_id: "lerobot/pusht"
    train_batch_size: 64
    data_format: "physicalai"
    image_transforms:
      class_path: physicalai.transforms.RandomChoice
      init_args:
        n_subset: 3
        random_order: false
        transforms:
          - class_path: torchvision.transforms.v2.ColorJitter
            init_args:
              brightness:
                - 0.8
                - 1.2
          - class_path: torchvision.transforms.v2.ColorJitter
            init_args:
              contrast:
                - 0.8
                - 1.2
          - class_path: torchvision.transforms.v2.ColorJitter
            init_args:
              saturation:
                - 0.5
                - 1.5
          - class_path: torchvision.transforms.v2.ColorJitter
            init_args:
              hue:
                - -0.05
                - 0.05
          - class_path: physicalai.transforms.RandomSharpness
            init_args:
              sharpness:
                - 0.5
                - 1.5
          - class_path: torchvision.transforms.v2.RandomAffine
            init_args:
              degrees:
                - -5.0
                - 5.0
              translate:
                - 0.05
                - 0.05
```

The same configuration works for all supported policies (ACT, SmolVLA, Pi0, Pi0.5, GR00T).

### Default Values Reference

| Transform  | Parameter    | Range         | Effect                                    |
| ---------- | ------------ | ------------- | ----------------------------------------- |
| Brightness | `brightness` | [0.8, 1.2]    | Scales pixel intensity by 0.8x-1.2x       |
| Contrast   | `contrast`   | [0.8, 1.2]    | Adjusts contrast by 0.8x-1.2x             |
| Saturation | `saturation` | [0.5, 1.5]    | Adjusts color saturation by 0.5x-1.5x     |
| Hue        | `hue`        | [-0.05, 0.05] | Shifts hue by up to 5% of the color wheel |
| Sharpness  | `sharpness`  | [0.5, 1.5]    | 0=blurred, 1=original, 2=double sharpness |
| Affine     | `degrees`    | [-5.0, 5.0]   | Random rotation up to 5 degrees           |
| Affine     | `translate`  | [0.05, 0.05]  | Random translation up to 5% of image size |

### Enabling via CLI Override

You can also enable augmentations from the command line without modifying the config file:

```bash
physicalai fit \
    --config configs/physicalai/act.yaml \
    --data.image_transforms.class_path physicalai.transforms.RandomChoice \
    --data.image_transforms.init_args.n_subset 3
```
