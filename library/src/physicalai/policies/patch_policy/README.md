# Patch Policy

Patch Policy uses image patches from a pretrained vision encoder to predict a short sequence of robot actions. It supports
VQ-BeT and diffusion action heads; the default is DINOv2 with VQ-BeT. See the [paper](https://arxiv.org/abs/2607.18236) and
[upstream code](https://github.com/gaoyuezhou/patch_policy).

## Forward Pass

These features match the SO101 example use case:
two cameras, six joint-state values, and six action values. No dataset normalization statistics are supplied.

```python
import torch

from physicalai.data import Feature, FeatureType, Observation
from physicalai.policies import PatchPolicy

input_features = [
    Feature(name="overview", ftype=FeatureType.VISUAL, shape=(3, 480, 640)),
    Feature(name="wrist", ftype=FeatureType.VISUAL, shape=(3, 480, 640)),
    Feature(name="state", ftype=FeatureType.STATE, shape=(6,)),
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
    state=torch.zeros(1, 6),
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
