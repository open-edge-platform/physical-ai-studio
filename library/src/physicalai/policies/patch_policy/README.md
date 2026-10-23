# Patch Policy

Patch Policy is a modular vision-to-action policy. It encodes one or more
camera views with a `timm` visual backbone and passes the resulting tokens to a
selectable action head that predicts a chunk of future robot actions.

Install its optional visual-backbone dependency with:

```bash
uv sync --extra patch_policy
```

## Model structure

```text
Camera observations [B, T, V, C, H, W]
    │
    ├─ order and resize camera views
    ├─ normalize each camera with dataset statistics
    ▼
TimmFeatureEncoder
    │
    ├─ DINOv2
    ├─ WebSSL
    └─ ResNet-18
    ▼
Visual tokens [B, T × V × P, E]
    │
    ├─ PatchDiffusionHead
    │    Iteratively denoises an action chunk while cross-attending to the
    │    visual tokens.
    │
    └─ PatchBehaviorTransformerHead
         Predicts residual-VQ action codes and code-conditioned continuous
         offsets with a patch-aware causal transformer.
    ▼
Normalized action chunk [B, chunk_size, action_dim]
    │
    ├─ denormalize with dataset statistics
    └─ retain n_action_steps
    ▼
Robot actions [B, n_action_steps, action_dim]
```

`n_obs_steps` is the number of temporal frames supplied for every camera. A
single-frame policy accepts `[B, C, H, W]`; a multi-frame policy requires
`[B, n_obs_steps, C, H, W]`. Training datasets provide these frames through the
policy's observation delta indices. At inference time, callers are responsible
for supplying the temporal window.

The feature definitions also carry dataset normalization statistics. Image and
action values are normalized with the shared `FeatureNormalizeTransform`, and
predicted actions are transformed back to the original action space before
they are returned.

## Basic forward pass

The following example constructs a small deterministic diffusion policy. It
uses randomly initialized weights and is intended to demonstrate the input and
output contract rather than produce useful robot actions.

```python
import torch

from physicalai.data import Feature, FeatureType, Observation
from physicalai.policies import PatchPolicy

policy = PatchPolicy(
    input_features=[
        Feature(name="front", ftype=FeatureType.VISUAL, shape=(3, 28, 28)),
        Feature(name="wrist", ftype=FeatureType.VISUAL, shape=(3, 28, 28)),
    ],
    output_features=[
        Feature(name="action", ftype=FeatureType.ACTION, shape=(6,)),
    ],
    encoder_name="dinov2",
    encoder_output="patches",
    encoder_pretrained=False,
    image_size=(28, 28),
    action_head="diffusion",
    n_obs_steps=2,
    chunk_size=8,
    n_action_steps=4,
    num_inference_steps=2,
    eta=0.0,
    use_random_input_noise=False,
).eval()

batch_size = 1
observation = Observation(
    images={
        "front": torch.rand(batch_size, 2, 3, 28, 28),
        "wrist": torch.rand(batch_size, 2, 3, 28, 28),
    },
)

with torch.no_grad():
    actions = policy(observation)

print(actions.shape)  # torch.Size([1, 4, 6])
```

For training, include an action tensor shaped
`[B, chunk_size, action_dim]`, put the policy in training mode, and call the
same forward method:

```python
observation.action = torch.rand(batch_size, 8, 6)
policy.train()
loss, metrics = policy(observation)
```

## Main policy options

| Option                                             | Purpose                                                            |
| -------------------------------------------------- | ------------------------------------------------------------------ |
| `input_features`                                   | Ordered camera feature definitions and normalization statistics.   |
| `output_features`                                  | Ordered action features concatenated into the model action vector. |
| `encoder_name`                                     | Visual backbone: `dinov2`, `webssl`, or `resnet18`.                |
| `encoder_output`                                   | Backbone representation: `patches`, `cls`, or `average`.           |
| `encoder_pretrained`                               | Load the backbone's pretrained weights.                            |
| `freeze_encoder`                                   | Keep the visual backbone frozen during training.                   |
| `action_head`                                      | `diffusion` or `behavior_transformer`.                             |
| `n_obs_steps`                                      | Number of temporal frames per camera.                              |
| `chunk_size`                                       | Number of future actions predicted by the model.                   |
| `n_action_steps`                                   | Number of predicted actions returned for execution.                |
| `hidden_dim`, `num_layers`, `num_heads`, `dropout` | Action-head transformer architecture.                              |
| `num_train_timesteps`, `num_inference_steps`       | Diffusion schedule lengths.                                        |
| `eta`                                              | DDIM stochasticity; `0.0` gives deterministic update steps.        |
| `use_random_input_noise`                           | Start diffusion sampling from random noise instead of zeros.       |

Behavior Transformer-specific quantizer and loss settings are available on
`PatchPolicyConfig`, including `vq_latent_dim`, `vq_num_embeddings`,
`vq_num_groups`, and the reconstruction, commitment, offset, secondary-code,
and focal-loss weights.

## Export

Patch Policy supports Torch and OpenVINO export through the standard
`policy.export()` API. The initial export contract supports the DINOv2 backbone
with patch-token output:

```text
encoder_name="dinov2"
encoder_output="patches"
```

Load a trained Lightning checkpoint and export it:

```python
from physicalai.policies import PatchPolicy

policy = PatchPolicy.load_from_checkpoint(
    "checkpoints/patch_policy.ckpt",
    map_location="cpu",
    weights_only=True,
).eval()

policy.export("exports/patch-policy-torch", backend="torch")
policy.export("exports/patch-policy-openvino", backend="openvino")
```

Torch export stores the complete policy checkpoint. Runtime reloads that policy,
so its PyTorch preprocessor and postprocessor continue to handle camera packing,
normalization, action denormalization, and execution-horizon trimming.

OpenVINO export stores the processed model graph. Its manifest configures the
Runtime NumPy pipeline to:

1. order, resize, and stack camera frames;
2. normalize images with the exported dataset statistics;
3. run the OpenVINO graph;
4. denormalize the action vector; and
5. trim the chunk to `n_action_steps`.

For deterministic diffusion parity, configure both:

```python
eta = 0.0
use_random_input_noise = False
```

An export preserves `n_obs_steps`. For example, an exported policy configured
with `n_obs_steps=3` requires three frames per camera at both PyTorch and
OpenVINO Runtime inference.
