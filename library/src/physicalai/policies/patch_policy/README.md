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
Camera observations dict[str, B × T × C × H × W]
    │
    ├─ order and resize camera views
    ├─ normalize each camera with encoder statistics
    ├─ stack views internally as [B, T, V, C, H, W]
    ▼
TimmFeatureEncoder
    │
    ├─ DINOv2
    ├─ WebSSL
    └─ ResNet-18
    ▼
Visual tokens [B, T × V × P, E]
    │
    ├─ DiffusionHead
    │    Iteratively denoises an action chunk while cross-attending to the
    │    visual tokens.
    │
    └─ BehaviourTransformerHead
         Predicts residual-VQ action codes and code-conditioned continuous
         offsets with a patch-aware causal transformer.
    ▼
Action chunk [B, chunk_size, action_dim]
    │
    ├─ denormalize with dataset statistics
    └─ retain n_action_steps
    ▼
Robot actions [B, n_action_steps, action_dim]
```

`n_obs_steps` is the number of temporal frames used for every camera. A
single-frame input `[B, C, H, W]` is accepted only when `n_obs_steps=1` and is
expanded to `[B, 1, C, H, W]`. Otherwise, each camera must provide the complete
window `[B, n_obs_steps, C, H, W]` for both training and inference. Training
datasets provide this window through the policy's observation delta indices.
Inference callers own observation history and any episode-start padding; the
model does not retain or repeat observations. `policy.reset()` clears queued
actions.

The feature definitions also carry dataset normalization statistics. Images
use the encoder statistics. Diffusion actions use min/max normalization to
`[-1, 1]`, matching the original policy, while Behavior Transformer actions
remain in dataset units and use `vq_action_scale` inside the quantizer.

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

For training, include the overlapping action target shaped
`[B, n_obs_steps + chunk_size - 1, action_dim]`, put the policy in training
mode, and call the same forward method:

```python
observation.action = torch.rand(batch_size, 9, 6)
policy.train()
loss, metrics = policy(observation)
```

## Main policy options

Defaults follow upstream PushT VQ-BeT: DINOv2 patch features, five observation
and action steps, a 512-wide transformer with eight layers/eight heads, zero
dropout, and VQ action scale 500. Both heads use the normal action queue:
`select_action` executes `n_action_steps` actions before replanning. With
`n_action_steps=1`, it returns the newest chunk's first action without averaging
older predictions. Temporal ensembling is a Runtime execution concern and is
not implemented here. `predict_action_chunk` returns the execution horizon
without updating the queue.

Behavior Transformer optimization also matches upstream: AdamW with betas
`(0.9, 0.999)`, configured decay on GPT linear weights and offset-head parameters,
zero decay on GPT biases/normalization/embeddings, and decay `0.01` on the code
head (the upstream optimizer's inherited default). Use training batch size 8
without gradient accumulation to match the single-GPU PushT run.

| Option                                             | Purpose                                                            |
| -------------------------------------------------- | ------------------------------------------------------------------ |
| `input_features`                                   | Ordered camera feature definitions and normalization statistics.   |
| `output_features`                                  | Ordered action features concatenated into the model action vector. |
| `encoder_name`                                     | Visual backbone: `dinov2`, `webssl`, or `resnet18`.                |
| `encoder_output`                                   | Backbone representation: `patches`, `cls`, or `average`.           |
| `encoder_pretrained`                               | Load the backbone's pretrained weights.                            |
| `image_size`                                       | Encoder input `(height, width)`, defaulting to `(224, 224)`.       |
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
`vq_num_groups`, `vq_action_scale`, and the offset, secondary-code, and
focal-loss weights. Inputs with missing or mismatched observation timesteps are
rejected.

## Training Preparation

Patch Policy prepares training through its own Lightning hooks. No preparation
callback is required:

```python
from physicalai.train import Trainer

trainer = Trainer()
trainer.fit(policy, datamodule)
```

`setup("fit")` resolves dataset features, and Lightning calls `configure_model()`
in its strategy-aware construction context. Studio's `PolicyDatasetInteraction`
callback configures dataset windows before ordinary user callbacks can construct
training loaders. The policy's
`on_fit_start()` fits a fresh VQ tokenizer after model
restoration, device placement, and optimizer creation, but before sanity
validation and training. Diffusion and already-fitted tokenizers skip fitting.
Optimizer construction has no preparation side effects. Plain Lightning Trainer
users can place the existing `PolicyDatasetInteraction` callback before callbacks
that inspect loaders. No Patch-specific callback is needed.

Training, validation, testing, prediction, and gym evaluation receive image
windows and run the encoder and head. The encoder remains frozen and in
evaluation mode, with no gradients. Visual features are computed per batch;
there is no feature cache, dataset replacement, or temporary storage lifecycle.

Set `PatchPolicy(compile_model=True)` to compile the feature encoder in place
for training and inference. Compilation is lazy, so initial calls may be slower;
steady-state speed depends on the device and workload. This execution option is
not saved in hyperparameters or the policy configuration and must be requested
again when loading a checkpoint.

For Behavior Transformer, a fresh action tokenizer is fitted before head
training. Preparation fits unique overlapping training
action windows, including boundary padding, using Adam (`lr=1e-3`,
`weight_decay=1e-4`), reconstruction L1 and residual commitment losses, EMA
codebooks, and dead-code replacement. `vqvae_iters` defaults to 1000 full
passes and `vqvae_batch_size` to 1024. The fitted tokenizer stays frozen and its
checkpointed completion flag prevents refitting on resume. Interrupted tokenizer
preparation restarts; it does not support mid-preparation resume.

Tokenizer fitting currently supports a single device with FP32 parameters (including
mixed-precision head training), not true-half parameters, distributed strategies,
or batch-size/learning-rate tuning. It preserves upstream tokenizer equations,
but deliberately does not reproduce upstream's incidental codebook updates and
zero-loss policy steps during its initial collection epoch. Full training-score
parity still requires matched experiments.
