# Patch Policy

Patch Policy uses image patches from a pretrained vision encoder to predict a short sequence of robot actions. It supports
VQ-BeT and diffusion action heads; the default is DINOv2 with VQ-BeT. See the [paper](https://arxiv.org/abs/2607.18236) and
[upstream code](https://github.com/gaoyuezhou/patch_policy).

## Forward Pass

These features match the SO101 example use case:
two cameras and six action values. Predictions use only images; joint state is ignored. No dataset normalization statistics are supplied.

```python
import torch

from physicalai.data import Feature, FeatureType, Observation
from physicalai.policies import PatchPolicy

input_features = [
    Feature(name="overview", ftype=FeatureType.VISUAL, shape=(3, 480, 640)),
    Feature(name="wrist", ftype=FeatureType.VISUAL, shape=(3, 480, 640)),
]
output_features = [Feature(name="action", ftype=FeatureType.ACTION, shape=(6,))]

policy = PatchPolicy(
    input_features=input_features,
    output_features=output_features,
    compile_model=False,
).eval()

# Images: [batch, history, channels, height, width], with pixels in [0, 1].
observation = Observation(
    images={
        "overview": torch.rand(1, 2, 3, 480, 640),
        "wrist": torch.rand(1, 2, 3, 480, 640),
    },
)
with torch.inference_mode():
    actions = policy(observation)
print(actions.shape)  # torch.Size([1, 1, 6])
```

This shows the interface, not a trained controller. The policy resizes and preprocesses images internally. Supply the two history frames oldest first.

## Train PushT

From `library/`, use the [DINOv2 VQ-BeT YAML](../../../../configs/physicalai/patch_policy/pusht/dinov2-vqbet.yaml):

```bash
physicalai fit --config configs/physicalai/patch_policy/pusht/dinov2-vqbet.yaml
```

The recipe trains for 400 epochs with five history frames. It logs metrics to W&B and records local rollout videos in `output/rollouts`.

## Train SO101

From `library/`, use the [SO101 DINOv2 VQ-BeT YAML](../../../../configs/physicalai/patch_policy/so101/dinov2-vqbet.yaml):

```bash
physicalai fit --config configs/physicalai/patch_policy/so101/dinov2-vqbet.yaml
```

The recipe trains on `alfieroddan/pick-n-place-so101` for 60 epochs with two history frames, five-action chunks,
and one executed action per prediction. It uses FP32 and an effective batch size of 128 (batch 32, accumulation 4,
one GPU). Validation uses a 10% held-out episode split with split seed 42, not simulated robot rollouts.

This is a starting recipe, not a replicated SO101 performance result. Predictions use only images, not joint state.
VQ-BeT uses raw action units with the default `vq_action_scale=500`; verify held-out action errors and supervised robot
performance before deployment.

## Training Notes

Fresh VQ-BeT tokenizer fitting supports only LeRobot datasets and single-device training. These restrictions are checked when fitting with the lightning trainer.

`PatchPolicy.from_config(config)` initializes pretrained encoder weights by default. Pass `encoder_pretrained=False` to skip weight initialization; checkpoint loading always restores saved weights without downloading encoder artifacts.

Diffusion with the default `clip_sample=True` requires action min/max statistics before training, prediction, or export. Dataset setup supplies these statistics during training. Construction without statistics is allowed, and constant action dimensions are supported. Deliberate raw-action diffusion without statistics requires `clip_sample=False` in the config.

To resume with the complete saved model config and optimizer state:

```python
from physicalai.policies import PatchPolicy
from physicalai.train import Trainer

checkpoint = "path/to/model.ckpt"
policy = PatchPolicy.load_from_checkpoint(checkpoint)
Trainer(max_epochs=400).fit(policy, datamodule=data, ckpt_path=checkpoint)
```

## Export

Load a trained Lightning checkpoint, then export to Torch or OpenVINO:

```python
from physicalai.policies import PatchPolicy

policy = PatchPolicy.load_from_checkpoint("path/to/model.ckpt", map_location="cpu").eval()
policy.export("exports/patchpolicy-torch", backend="torch")
policy.export("exports/patchpolicy-openvino", backend="openvino")
```

For deterministic export checks, pass `use_random_noise=False`.
