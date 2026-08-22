# XVLA

XVLA ("cross-embodiment vision-language-action") pairs a **Florence-2** encoder with a
**soft-prompted, domain-aware transformer** that denoises a whole action chunk by flow
matching. What sets it apart from the other Studio VLAs is that one checkpoint is meant to
drive *several robots*: a per-sample `domain_id` selects the domain's own projections and
prompt tokens, and an action space describes how the model's fixed-width action vector maps
onto the embodiment at hand.

Studio ships a first-party port under `physicalai.policies.xvla`; it depends only on
`transformers` (no LeRobot policy code at runtime).

## Architecture

| Component                  | Class                     | Role                                                                    |
| -------------------------- | ------------------------- | ----------------------------------------------------------------------- |
| Vision tower               | Florence-2 DaViT          | Pools each camera into a short token sequence (~50 tokens at 224×224).   |
| Text encoder               | Florence-2 BART encoder   | Encodes the prompt jointly with the primary camera's tokens.             |
| Action transformer         | `SoftPromptedTransformer` | Denoises the action chunk, conditioned on both visual streams.           |
| Action space               | `BaseActionSpace`         | Maps the fixed-width action vector onto a robot; owns the loss.          |

```text
xvla/
├── config.py             # XVLAConfig
├── model.py              # XVLAModel  (Florence-2 + action transformer)
├── policy.py             # XVLA       (Lightning wrapper)
├── action_hub.py         # action-space registry (auto, ee6d, joint, ...)
├── soft_transformer.py   # the domain-aware, soft-prompted transformer
├── preprocessor.py       # cameras, prompt, state, domain id, (de)normalization
└── pretrained_utils.py   # LeRobot-layout checkpoint loading
```

Only the encoder side of Florence-2 is used; the text decoder is deleted at construction, so
it is neither trained nor checkpointed.

## How a chunk is predicted

1. Every camera the mask marks valid is pooled by the vision tower. Masked slots keep zero
   features, so a checkpoint trained with more cameras than the dataset carries still lines up.
2. The **primary** view's tokens are prepended to the tokenized prompt and run through the
   BART encoder, giving `vlm_features`.
3. The remaining views are flattened into a separate `aux_visual_inputs` stream.
4. One token per action step is built from the noised action, the proprioceptive state and
   the flow-matching timestep, through the domain-aware action encoder.
5. The sequence `[action tokens, vlm_features, aux views]` gets a learned positional
   embedding, this domain's soft prompts are appended, and the stack runs.
6. Only the action tokens are decoded back to actions.

Unlike a velocity-field flow model, the transformer predicts the **clean** action chunk at
every step. Inference therefore starts from pure noise and re-noises its own estimate toward
`t = 0` over `num_denoising_steps` iterations, rather than integrating a velocity.

### Sequence-length budget

`max_len_seq` (512 by default) has to cover
`chunk_size + image_tokens + tokenizer_max_length + (num_image_views - 1) * image_tokens`.
The vision tower downsamples by 32×, so a 224×224 camera contributes 50 tokens and a
256×256 one contributes 65. Two 256×256 cameras with a 32-step chunk and a 64-token prompt
come to 226 tokens — comfortably inside the budget, but a third camera at a higher
resolution is worth checking. Overflowing raises rather than silently truncating.

## Cross-embodiment: domains and action spaces

`domain_id` picks the row of every domain-aware layer (`action_encoder`, `action_decoder`,
the soft-prompt table and, when `use_hetero_proj` is on, the visual projections). It comes
from the batch — `domain_id`, `extra.domain_id`, or whatever `domain_feature_key` names —
and falls back to the configured `domain_id` when the batch carries none. A single-robot
dataset can simply leave it at 0.

`action_mode` selects the action space, which fixes the width the transformer predicts, the
loss over it, and the width the policy emits:

| `action_mode`    | Model width  | Emitted width | Loss                                                  |
| ---------------- | ------------ | ------------- | ----------------------------------------------------- |
| `auto` (default) | `max_action_dim` | dataset's | MSE over the dataset's real channels only             |
| `ee6d`           | 20           | 20            | Scaled MSE on xyz / 6D rotation, BCE on the grippers  |
| `agibot_ee6d`    | 20           | 20            | As `ee6d`, but the grippers are regressed too         |
| `joint`          | 14           | 14            | MSE on the joints, BCE on the grippers                |
| `franka_joint7`  | 20           | 7             | MSE on the seven Franka joints                        |
| `so101_bimanual` | 20           | 12            | Per-arm MSE plus a gripper term                       |

`auto` is the Studio default: it keeps the pretrained action width — so published weights
still load — but reads the supervised and emitted slice from the training dataset's
statistics, which makes it correct for any embodiment. The released XVLA checkpoints were
trained with `ee6d`; set `action_mode: ee6d` when finetuning one, and remap the dataset into
that channel layout.

Only `auto` can be re-fitted to a new dataset. Finetuning a fixed layout on a dataset of a
different width logs a warning and keeps the published width, because that width is part of
the checkpoint's contract.

### Normalization

State and action normalization defaults to `IDENTITY`. XVLA's action spaces carry their own
per-channel loss scaling (`ee6d` weights position 500× and rotation 10×, for instance) and
the published checkpoints are trained on raw units, so normalizing by default would put the
model in a space its weights were never fit to. Switch to `MEAN_STD` or `QUANTILES` when
training from scratch on a dataset whose units are far from those checkpoints; the
preprocessor and postprocessor stay exact inverses either way.

## Cameras and prompts

Cameras arrive as Studio's flattened `images.*` keys, are ImageNet-normalized, and are
stacked into one `[B, V, C, H, W]` tensor with a `[B, V]` validity mask. `uint8` cameras are
divided by 255; floating-point cameras are assumed to already be in `[0, 1]`, matching the
LeRobot datamodule. A temporal clip collapses to its most recent frame. Set
`resize_imgs_with_padding` to resize without distortion (padding left and top, the XVLA
convention); leave it unset to keep the dataset's resolution.

The prompt is tokenized with Florence-2's BART tokenizer to a fixed
`tokenizer_max_length`, so the sequence the transformer sees never changes length.

## Usage

Training from the CLI:

```bash
physicalai fit --config configs/physicalai/xvla.yaml
```

From the API:

```python
from physicalai.policies import XVLA
from physicalai.data.lerobot import LeRobotDataModule
from physicalai.train import Trainer

policy = XVLA(action_mode="auto", chunk_size=32)
datamodule = LeRobotDataModule(repo_id="lerobot/libero", train_batch_size=4)
Trainer(max_epochs=10).fit(policy, datamodule)
```

Finetuning a published checkpoint — its architecture and action space come with it, and only
the arguments you change override them:

```python
policy = XVLA(pretrained_name_or_path="lerobot/xvla_libero", freeze_vision_encoder=True)
action = policy.select_action(observation)
```

## Training

`configure_optimizers` reproduces XVLA's differential learning rates, which is what upstream
reports as necessary for stable finetuning:

| Parameter group   | Learning rate                            | Weight decay                             |
| ----------------- | ---------------------------------------- | ---------------------------------------- |
| Florence-2 (`vlm`) | `optimizer_lr * optimizer_vlm_lr_scale` (a tenth) | scaled the same way             |
| Soft prompts      | `optimizer_lr * optimizer_soft_prompt_lr_scale`   | `optimizer_weight_decay`        |
| Everything else   | `optimizer_lr`                           | `optimizer_weight_decay`                 |

One cosine-decay-with-warmup schedule multiplies all three groups, so their relative rates
hold throughout the run.

Four flags control what trains: `freeze_vision_encoder`, `freeze_language_encoder`,
`train_policy_transformer` and `train_soft_prompts`. Freezing both encoders and leaving the
transformer and prompts trainable is the cheap adaptation path; frozen parameters never
reach the optimizer.

Validation reports action-prediction MSE — the full denoising loop compared against the
ground truth — rather than the training loss, whose scale depends on the sampled timesteps.

## Export

Export is **not** supported for this family, so `ExportablePolicyMixin` is not mixed in. The
Florence-2 encoder runs only the camera views the mask marks valid, which makes the traced
shapes depend on the data rather than on the config. Deploy through the training-side policy
API until that path is made static.

## Checkpoints

Published XVLA checkpoints follow the LeRobot layout: `config.json`, `model.safetensors`,
and optionally `policy_preprocessor.json` / `policy_postprocessor.json` with their state
files. Loading reconciles three layout differences:

- LeRobot nests the network one level deeper than Studio, so the `model.` key prefix is
  stripped.
- Checkpoints saved against the old vendored (Microsoft remote-code) Florence-2 module tree
  are remapped onto the native `transformers.models.florence2` layout — the DaViT stem, the
  flattened PreNorm/Mlp wrappers, the multimodal projector, and the transposed
  `image_projection` parameter.
- safetensors deduplicates tied tensors on save, so whichever alias of the shared token
  embedding is missing gets restored.

`config.json` is read with the LeRobot-only keys dropped, and `num_image_views` is derived
from the visual `input_features` when the checkpoint does not pin it, so a checkpoint keeps
the camera count it was trained with.

## Differences from the LeRobot implementation

- **`action_mode` defaults to `auto`** instead of `ee6d`, because Studio builds the model
  from the training dataset's statistics and can size the action space from them. The model
  width is unchanged either way, so weights stay compatible.
- **Prompt padding is fixed** to right-padding at `tokenizer_max_length`; upstream's
  `tokenizer_padding_side` / `pad_language_to` knobs are dropped rather than carried as
  settings that would change the sequence the positional embedding was trained on.
- **`optimizer_soft_prompt_warmup_lr_scale` is not ported.** Upstream sets a lower initial
  soft-prompt rate and relies on the shared schedule to raise it, which never reaches
  `soft_prompt_lr_scale`; Studio exposes the scale alone.
- **Validation reports action-prediction MSE**, matching the other Studio policies, rather
  than reusing the training loss.

## References

- Upstream policy: `lerobot/policies/xvla`
- [Base Policy](base.md) — the `Policy` / `Model` contract
- Training config: `configs/physicalai/xvla.yaml`
