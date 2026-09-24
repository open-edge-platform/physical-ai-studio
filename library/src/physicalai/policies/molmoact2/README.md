# MolmoAct2

MolmoAct2 is a vision-language-action policy from the Allen Institute for AI.
It consumes camera images, robot state, and a task string, then predicts a chunk
of future actions. This package provides a first-party PhysicalAI implementation
for training, Lightning checkpoints, benchmarking, and export.

- [Blog post](https://allenai.org/blog/molmoact2)
- [Paper](https://arxiv.org/abs/2605.02881)

## Installation

```bash
uv sync --extra molmoact2
```

## Loading Policies

MolmoAct2 has two distinct loading paths:

- `MolmoAct2(pretrained_name_or_path=..., norm_tag=...)` initializes from a
  released Hugging Face checkpoint.
- `MolmoAct2.load_from_checkpoint(...)` restores a policy previously trained
  and saved by Lightning.

### Released Hugging Face Checkpoint

```python
import torch

from physicalai.devices.utils import get_device
from physicalai.policies import MolmoAct2

DEVICE = get_device()

policy = MolmoAct2(
    pretrained_name_or_path="allenai/MolmoAct2-LIBERO",
    norm_tag="libero",
    n_action_steps=10,
    use_random_input_noise=True,
)
policy = policy.to(device=DEVICE, dtype=torch.bfloat16).eval()
```

The normalization tag resolves the checkpoint's camera, state, action,
normalization, action-horizon, setup, and control metadata from
`norm_stats.json`.

### Lightning Checkpoint

After fitting a policy, save it through the attached Lightning trainer:

```python
trainer.fit(policy, datamodule=datamodule)
trainer.save_checkpoint("checkpoints/molmoact2.ckpt", weights_only=True)
```

Restore it directly for inference:

```python
from physicalai.policies import MolmoAct2

policy = MolmoAct2.load_from_checkpoint(
    "checkpoints/molmoact2.ckpt",
    map_location="cpu",
    weights_only=True,
).eval()
```

The Lightning checkpoint contains the resolved `MolmoAct2Config`. Loading it
rebuilds the architecture and processors, then applies the trained state dict.
It does not download or reload the original pretrained model weights.

Tokenizer files are not embedded in the Lightning checkpoint. The local path
saved in `config.tokenizer_name_or_path` must still be available when the
checkpoint is restored.

## Training

### CLI

The checked-in configuration contains the complete model, dataset, optimizer,
and trainer setup:

```bash
physicalai fit --config configs/physicalai/molmoact2.yaml
```

See [`configs/physicalai/molmoact2.yaml`](../../../../configs/physicalai/molmoact2.yaml)
for the available overrides.

### Python API

```python
import multiprocessing

from physicalai.data import LeRobotDataModule
from physicalai.policies import MolmoAct2
from physicalai.train import Trainer

multiprocessing.set_start_method("spawn", force=True)

if __name__ == "__main__":
    policy = MolmoAct2(
        use_random_input_noise=True,
      lora_enabled=True,
        gradient_checkpointing=True,
    )

    datamodule = LeRobotDataModule(
        repo_id="lerobot/pusht",
        train_batch_size=8,
        data_format="physicalai",
    )

    trainer = Trainer(max_epochs=30, precision="bf16-mixed")
    trainer.fit(policy, datamodule=datamodule)
```

When the policy is constructed lazily, the training dataset supplies its input
and output feature contract during `setup("fit")`.

By default, `scheduler_decay_steps=None` derives the cosine schedule from
Lightning's estimated optimizer-step budget. After `scheduler_warmup_steps`,
the learning rate decays across all remaining training steps and reaches
`scheduler_decay_lr` at the end of training. Set `scheduler_decay_steps` to an
integer to use a manual decay horizon instead. If that horizon is longer than
the run, MolmoAct2 scales it and the warmup proportionally to fit the run.

When fine-tuning a compatible pretrained policy, set
`preserve_pretrained_normalization_in_training=True` to adopt the dataset's feature and
camera contract while retaining the initialized policy's state and action
normalization statistics. The default is `False`, which uses the training
dataset's statistics. This option is used only by `setup("fit")`; explicit
`set_features()` calls use the supplied feature statistics unless their
corresponding `copy_state_normalization` or `copy_action_normalization` flag is
set.

### Dataset Quantile Statistics

MolmoAct2 defaults to quantile normalization for state and action features. If
your dataset has not been converted with quantile statistics, you can add them
with:

```bash
python -m lerobot.scripts.augment_dataset_quantile_stats \
  --repo-id=your_dataset
```

### SO-101 Fine-Tuning Frames

SO-101 datasets are expected to contain samples and normalization statistics in
the current robot frame. There are two supported fine-tuning modes:

- `adapt_to_so101=True` converts both dataset samples and their state/action
  statistics to the released checkpoint's legacy frame. Use this when
  fine-tuning `allenai/MolmoAct2-SO100_101` to preserve its learned joint
  representation and generally converge faster.
- `adapt_to_so101=False` keeps samples and statistics in the current SO-101
  frame. This is internally consistent but requires the model to adapt its
  learned joint representation during fine-tuning.

With `norm_tag="so100_so101_molmoact2"`, omitting `adapt_to_so101` selects the
legacy-frame mode automatically. Pass `adapt_to_so101=False` explicitly to
select native-frame training. The resolved mode and frame-aligned statistics
are saved in Lightning checkpoints and OpenVINO manifests.

With `adapt_to_so101=True`, the same joint transform as in
[Zero-Shot SO-101](#zero-shot-so-101) is applied to dataset states, action
targets and their statistics. Passing the recording arm's `calibration` puts
them in the checkpoint's true degree units:

- It is **required** with `preserve_pretrained_normalization_in_training=True`,
  because the checkpoint's statistics are in degrees.
- It is optional otherwise. Dataset statistics absorb the unit scale, and only the
  signs and offsets matter.

LeRobot's `MolmoAct2StateFrameTransformStep` transforms only the observation
state. PhysicalAI transforms states, action targets and statistics together,
so the model trains and predicts in one consistent frame.

Checkpoints trained with older versions that transformed samples without also
transforming dataset statistics should be retrained. Their normalized action
targets may have been clamped, so changing only the exported manifest cannot
recover the lost training signal.

With `lora_enabled=True`, MolmoAct2 adds adapters to the VLM and action expert
while freezing their base weights. This matches the recommended small-dataset
fine-tuning recipe. Use `lora_target_modules` to override the default adapter
selection, `lora_adapter_dtype="auto"` to inherit base layer precision, or
`lora_use_dora=True` to use DoRA. Export merges adapters into a disposable model
copy and leaves the live training model unchanged.

For full fine-tuning, leave both `lora_enabled` and `train_action_head_only` false.
For action-head-only training, use `train_action_head_only=True`. LoRA and
action-head-only training are mutually exclusive.

## Benchmarking LIBERO

```python
import random

import numpy as np
import torch

from physicalai.benchmark.gyms import LiberoBenchmark
from physicalai.policies import MolmoAct2

DEVICE = "cuda"

SEED = 0

TASK_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def set_seed(seed: int) -> None:
    """Seed all global RNGs for reproducible benchmark + model sampling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    set_seed(SEED)

    policy = MolmoAct2(
        pretrained_name_or_path="allenai/MolmoAct2-LIBERO",
        norm_tag="libero",
        n_action_steps=10,
        use_random_input_noise=True,
        compile_model=True,
    ).eval()

    policy.rename_features({"wrist_image": "image2"})
    policy = policy.to(device=DEVICE, dtype=torch.bfloat16).eval()

    for task_suite in TASK_SUITES:
        benchmark = LiberoBenchmark(
            task_suite=task_suite,
            num_episodes=1,
            seed=SEED,
            observation_height=378,
            observation_width=378,
            record_mode="all",
            video_dir="./videos/",
        )

        try:
            results = benchmark.evaluate(policy)
            summary = results.summary()

            block_header = f"\n{'=' * 28} {task_suite} {'=' * 28}\n"
            print(block_header, end="")
            print(summary)
        finally:
            for gym in benchmark.gyms:
                gym.close()
```

`rename_features` maps the checkpoint's resolved input feature names to the
names emitted by the environment. It preserves feature order, shapes, and
normalization statistics.

Results on NVIDIA A100:

| Suite          | Tasks  | Avg. success rate (%) | Avg. reward | Avg. episode length | Avg. FPS  |
| -------------- | ------ | --------------------- | ----------- | ------------------- | --------- |
| libero_spatial | 10     | 100.0                 | 1.00        | 107.4               | 14.60     |
| libero_object  | 10     | 100.0                 | 1.00        | 131.7               | 19.00     |
| libero_goal    | 10     | 100.0                 | 1.00        | 108.3               | 18.10     |
| libero_10      | 10     | 100.0                 | 1.00        | 242.8               | 19.50     |
| **Average**    | **40** | **100.0**             | **1.00**    | **147.6**           | **17.80** |

## Export

MolmoAct2 supports Torch and OpenVINO export.

Load a trained Lightning checkpoint before exporting it:

```python
from physicalai.policies import MolmoAct2

policy = MolmoAct2.load_from_checkpoint(
    "checkpoints/molmoact2.ckpt",
    map_location="cpu",
    weights_only=True,
).eval()

policy.export("exports/molmoact2-torch", backend="torch")
policy.export("exports/molmoact2-openvino", backend="openvino")
```

OpenVINO export also exports the tokenizer and requires the tokenizer assets
referenced by the restored config.

## Zero-Shot SO-101

The released `allenai/MolmoAct2-SO100_101` checkpoint was trained on an older
LeRobot joint convention. Its body joints are in **degrees**, `shoulder_lift` is
flipped, and `shoulder_lift`/`elbow_flex` are offset by 90°. The PhysicalAI SO-101
driver instead reports each body joint in **normalized units** `[-100, 100]` over
its calibrated tick range. The gripper is `[0, 100]` in both conventions.

With `adapt_to_so101=True` and the arm's `calibration`, MolmoAct2 maps runtime
joints into the checkpoint frame before normalization, and maps actions back
after denormalization. The checkpoint's normalization statistics are used
unchanged:

```text
Runtime state (normalized units)
    │  checkpoint = sign * scale * runtime + offset
    ▼
Checkpoint frame (legacy degrees)
    │  normalize with the checkpoint's statistics
    ▼
MolmoAct2 pretrained weights
    │  denormalize with the same statistics
    ▼
Checkpoint frame (legacy degrees)
    │  runtime = sign * (checkpoint - offset) / scale
    ▼
Runtime action (normalized units)
```

- `sign = [1, -1, 1, 1, 1, 1]`, `offset = [0, 90, 90, 0, 0, 0]`: the LeRobot
  backward-compatibility transform.
- `scale` is the number of degrees in one runtime unit, computed per body joint
  from the calibration as `(range_max - range_min) / 200 * 360 / 4095`. The
  gripper scale is 1.

LeRobot degrees are `(ticks - mid) * 360 / 4095` and PhysicalAI units are
`(ticks - mid) * 200 / (range_max - range_min)`, so `scale * runtime` equals
LeRobot degrees exactly. Zero-shot inference therefore matches LeRobot's
`MolmoAct2StateFrameTransformStep`/`MolmoAct2ActionFrameTransformStep` pipeline
for the same arm, from servo ticks in to servo ticks out.

The scales come from one arm's calibration and are embedded in the Torch
processors and the OpenVINO manifest (`joint_frame_preprocess` /
`joint_frame_postprocess`). Re-export for an arm with different ranges. Without
a calibration the scale is 1, and runtime units are treated as degrees.

```python
import torch

from physicalai.policies import MolmoAct2

calibration_so101 = "SO101 Follower-calibration.json"

policy = MolmoAct2(
    pretrained_name_or_path="allenai/MolmoAct2-SO100_101",
    norm_tag="so100_so101_molmoact2",
    adapt_to_so101=True,
    calibration=calibration_so101,
)

policy.set_features(
  input_features=input_features,
  output_features=output_features,
  copy_state_normalization=True,
  copy_action_normalization=True,
)

policy = policy.to(dtype=torch.bfloat16).eval()
policy.export("exports/molmoact2-so101-torch", backend="torch")
```

`set_features` replaces the checkpoint feature definitions without reloading
its weights. Visual features are always taken directly from `input_features`,
so the replacement can add, remove, or rename cameras. The two normalization
flags independently copy the checkpoint's state and action normalization onto
replacement features with matching shapes. Copied normalization overwrites any
normalization already present on that replacement feature.

Start the SO-101 from an extended pose near the task workspace. Starting from a
rest pose can cause the policy to remain there.

`calibration` accepts a dictionary or a path to the arm's calibration JSON, for
example:

```json
{
  "shoulder_pan": {
    "id": 1,
    "drive_mode": 0,
    "homing_offset": -457,
    "range_min": 701,
    "range_max": 3381
  },
  "shoulder_lift": {
    "id": 2,
    "drive_mode": 0,
    "homing_offset": 260,
    "range_min": 882,
    "range_max": 3244
  },
  "elbow_flex": {
    "id": 3,
    "drive_mode": 0,
    "homing_offset": 642,
    "range_min": 888,
    "range_max": 3088
  },
  "wrist_flex": {
    "id": 4,
    "drive_mode": 0,
    "homing_offset": 1056,
    "range_min": 778,
    "range_max": 3092
  },
  "wrist_roll": {
    "id": 5,
    "drive_mode": 0,
    "homing_offset": -56,
    "range_min": 28,
    "range_max": 4065
  },
  "gripper": {
    "id": 6,
    "drive_mode": 0,
    "homing_offset": -807,
    "range_min": 2031,
    "range_max": 3538
  }
}
```

## Repository and Normalization Tags

`pretrained_name_or_path` identifies the Hugging Face repository or local
snapshot. `norm_tag` selects one entry from `metadata_by_tag` in that
snapshot's `norm_stats.json`.

```json
{
  "format": "molmoact2_norm_stats.v1",
  "norm_mode": "q01_q99",
  "metadata_by_tag": {
    "so100_so101_molmoact2": {
      "action_key": "action",
      "state_key": "observation.state",
      "camera_keys": [],
      "normalize_gripper": true,
      "action_horizon": 30
    }
  }
}
```

The repository and normalization tag must describe the same embodiment. For
example:

```python
policy = MolmoAct2(
    pretrained_name_or_path="allenai/MolmoAct2-SO100_101",
    norm_tag="so100_so101_molmoact2",
)
```

For a custom dataset, omit `norm_tag` and construct the policy lazily. The
attached PhysicalAI dataset supplies features and normalization statistics at
training setup.

Constructor-provided features override features resolved from `norm_tag`
without inheriting their normalization. For zero-shot use, initialize from the
tag first and call `set_features` with the specific normalization maps that
should be copied.

## Notes

- `n_action_steps` must be between 1 and `chunk_size`.
- `use_random_input_noise=True` starts flow matching from sampled Gaussian
  noise; otherwise inference uses deterministic zero noise.
- Lightning checkpoints restore model weights and resolved configuration, but
  do not package tokenizer files.
- Supported export backends are Torch and OpenVINO.
- Generic video augmentation from the upstream implementation is not included.
