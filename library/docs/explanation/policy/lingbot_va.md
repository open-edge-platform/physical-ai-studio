# LingBot-VA

LingBot-VA is an **autoregressive video-action world model** built on the Wan2.2
video-diffusion stack. A single dual-stream transformer interleaves, in one autoregressive
sequence, the prediction of future **video latents** and robot **actions** — hence "VA".

Studio ships a first-party port under `physicalai.policies.lingbot_va`; it depends only on
`diffusers` and `transformers` (no LeRobot policy code at runtime).

## Architecture

The backbone is a "mixture of transformers": a video-latent stream and an action stream
share all 30 transformer blocks and the same text conditioning, but keep their own input
embedders, timestep embedders and output heads.

| Component                | Class                   | Role                                                                 |
| ------------------------ | ----------------------- | -------------------------------------------------------------------- |
| DiT backbone (trainable) | `WanTransformer3DModel` | ~5B-parameter dual-stream transformer; the only checkpointed module. |
| VAE (frozen)             | `AutoencoderKLWan`      | Wan2.2 VAE, `z_dim=48`, 16x spatial / 4x temporal downsample.        |
| Text encoder (frozen)    | `UMT5EncoderModel`      | UMT5-XXL, `d_model=4096`; runs once per episode, on CPU by default.  |

The frozen VAE + text encoder + tokenizer (~20 GB) are **not** part of the Studio
checkpoint. They live outside the `nn.Module` registry and are pulled lazily from
`config.wan_pretrained_path` the first time the model runs, which keeps checkpoints small
and keeps `.to(device)` from dragging 20 GB onto the GPU.

```text
lingbot_va/
├── config.py            # LingBotVAConfig
├── model.py             # LingBotVAModel  (transformer + frozen stack + streaming state)
├── policy.py            # LingBotVA       (Lightning wrapper)
├── preprocessor.py      # camera-key resolution + action (de)normalization
├── pretrained_utils.py  # LeRobot-layout checkpoint loading
└── components/          # ported Wan2.2 code (transformer, attention, scheduler, VAE, text)
```

## Autoregressive inference

Inference is **stateful across environment steps**, which is what makes this family
different from the other Studio policies:

1. The episode's first observation conditions the first chunk. Its cameras are VAE-encoded
   into the initial latent and the KV cache is allocated.
2. One chunk is denoised: first the video-latent stream (~20 steps, with classifier-free
   guidance), then the action stream (~50 steps), each with its own flow-matching scheduler.
3. The chunk's `frame_chunk_size * action_per_frame` actions are queued and executed.
4. Every executed step's observation is buffered as a **real keyframe**. When the queue
   empties, those keyframes plus the executed actions are written back into the KV cache
   before the next chunk is predicted — so the world model stays anchored to what actually
   happened rather than to what it imagined.

Because step 4 has to see _every_ observation, `LingBotVA.select_action` overrides the base
action-queue behaviour, which only inspects the batch when the queue runs dry. The
streaming path is written for single-environment rollouts (batch size 1).

## Action space

LingBot-VA is an **end-effector (Cartesian) pose** policy: it predicts EEF poses and gripper
commands, not joint positions. Actions live in a fixed multi-embodiment 30-dim layout, and
`used_action_channel_ids` selects the channels a given checkpoint actually drives (this also
fixes the policy's output action width).

| channels | meaning                                               |
| -------- | ----------------------------------------------------- |
| 0-6      | Left-arm end-effector pose                            |
| 7-13     | Right-arm end-effector pose                           |
| 14-20    | Left-arm joints (unused by the released checkpoints)  |
| 21-27    | Right-arm joints (unused by the released checkpoints) |
| 28       | Left gripper                                          |
| 29       | Right gripper                                         |

LIBERO uses channels `0-6` (6-DoF EEF delta + gripper). Joint-space datasets must be
remapped into this schema before fine-tuning a released checkpoint.

Action normalization is symmetric in Studio: the preprocessor maps ground-truth actions into
the model's `[-1, 1]` space with the checkpoint's per-channel q01/q99, and the postprocessor
maps predictions back to physical units with the same statistics. (Upstream LeRobot
normalizes only on the way out.)

## Cameras

**Camera order is fixed and order-sensitive** — per-camera latents are concatenated
spatially in `obs_cam_keys` order, so the physical camera to slot mapping must match
training. The first camera is the exterior/head view; the rest are wrist views.

| benchmark | `obs_cam_keys` (in order)                    | `camera_layout`                                         |
| --------- | -------------------------------------------- | ------------------------------------------------------- |
| LIBERO    | `image` (agentview), `image2` (eye-in-hand)  | `width_concat` (latents concatenated on width)          |
| RoboTwin  | `head_camera`, `left_camera`, `right_camera` | `robotwin_tshape` (full-res head below half-res wrists) |

Both LeRobot spellings (`observation.images.image`) and Studio spellings (`images.image`,
`image`) resolve against a batch, so a checkpoint's config works unchanged against a Studio
datamodule or gym.

## Usage

```python
from physicalai.policies import LingBotVA

policy = LingBotVA(pretrained_name_or_path="lerobot/lingbot_va_libero_long")
policy.eval()

action = policy.select_action(observation)  # [B, 7], physical units
policy.reset()                              # between episodes
```

Benchmarking on LIBERO:

```python
from physicalai.benchmark.gyms import LiberoBenchmark

results = LiberoBenchmark(task_suite="libero_10", num_episodes=20).evaluate(policy)
```

## Training

`compute_loss` implements the dual-stream flow-matching loss (`latent_loss + action_loss`,
timestep-weighted and action-masked): camera clips are VAE-encoded into video latents, the
task is UMT5-encoded, both streams are noised independently, and the transformer runs its
block-causal training pass.

Requirements:

- **`attn_mode="flex"`.** The block-causal / window / noise-vs-clean masks use PyTorch
  flex-attention. The default `torch` SDPA backend is inference-only and
  `compute_loss` fails fast with a clear error.
- **Memory.** The full 5B DiT does not fit a 24-32 GB GPU under AdamW; fine-tune with LoRA
  and/or optimizer offload, at batch size 1. `get_optim_params` returns only the trainable
  parameters, so the frozen stack never reaches the optimizer.
- **Data.** The dataset must supply a temporal camera clip per camera and
  `frame_chunk_size * action_per_frame` action steps per item. The policy's
  `observation_delta_indices` / `action_delta_indices` configure that automatically.

```bash
physicalai fit --config configs/physicalai/lingbot_va.yaml --trainer.fast_dev_run=true
```

## Export

Only the **`torch`** backend is supported. The autoregressive KV cache, the lazily loaded
20 GB frozen stack and the two nested denoising loops have no meaningful static graph, so
the tracing backends (ONNX, OpenVINO, ExecuTorch) are deliberately not offered.
`to_torch()` needs none of them: it serializes the trainable transformer plus the
hyperparameters that rebuild the policy, and Runtime restores the live Python object from
them. The frozen VAE/UMT5 stack stays out of the checkpoint and is pulled from
`wan_pretrained_path` on first use, exactly as during training.

```bash
physicalai export \
  --policy physicalai.policies.lingbot_va.LingBotVA \
  --ckpt_path checkpoints/last.ckpt \
  --backend torch \
  --output_dir ./export
```

```python
from physicalai.export import ExportBackend

policy.export("./export", backend=ExportBackend.TORCH)
```

The export directory holds `lingbotva.pt` and a `manifest.json` naming one visual feature
per configured camera (`images.<name>`, in `obs_cam_keys` order) plus the `task` prompt, and
a single `action` output of shape `(chunk_size, output_action_dim)`. Action (de)normalization
travels inside the restored policy, so the only runtime preprocessor is `to_float_tensor`.

!!! note "Chunk-at-a-time vs. streaming"

    Runtime's `SinglePass` runner calls the policy's `forward()`, which returns a whole
    action chunk — the same contract as every other exported Studio policy. That chunk is
    executed open-loop. To get the closed-loop behaviour described above, where each
    observed frame is replayed into the KV cache, drive the restored policy with
    `select_action()` once per environment step instead.

## Checkpoints

| Variant                | Checkpoint                       |
| ---------------------- | -------------------------------- |
| LIBERO-Long post-train | `lerobot/lingbot_va_libero_long` |
| RoboTwin post-train    | `lerobot/lingbot_va_robotwin`    |
| Pretrained base        | `lerobot/lingbot_va_base`        |

Install with `pip install 'physicalai-train[lingbot_va]'`. Inference needs roughly 18-24 GB
of VRAM.

## References

- [Upstream repository](https://github.com/Robbyant/lingbot-va) (Apache-2.0)
