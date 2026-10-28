# VLA-JEPA

VLA-JEPA is a first-party PhysicalAI policy with a Qwen3-VL backbone and a
flow-matching DiT action head. It loads released pretrained weights, fine-tunes
on robot demonstrations, and predicts action chunks. It does not wrap or import
LeRobot policy code.

Fine-tuning defaults to the joint JEPA/action objective. Action-only fine-tuning
is available when training memory or video decoding is a constraint. Both modes
support LoRA/DoRA fine-tuning and Torch or OpenVINO export for Runtime inference.
Human-video-only pretraining and ONNX/ExecuTorch export are not implemented.

## Install

From `library/`:

```bash
uv sync --frozen --extra cu128 --extra vla_jepa
```

## Fine-tune

```bash
uv run --no-sync physicalai fit \
  --config configs/physicalai/vla_jepa/so101.yaml \
  --data.repo_id your-org/your-robot-dataset
```

Or construct the native policy in Python:

```python
from physicalai.policies import VLAJEPA

policy = VLAJEPA(
    pretrained_name_or_path="lerobot/VLA-JEPA-Pretrain",
    gradient_checkpointing=True,
)
trainer.fit(policy, datamodule=datamodule)
```

Set `enable_world_model=False` for action-only fine-tuning:

```python
policy = VLAJEPA(enable_world_model=False)
```

Or add `--model.enable_world_model=false` to the CLI command. Both modes train
Qwen and the action head. Full mode also trains the native world predictor, while
the pretrained V-JEPA2 target encoder stays frozen:

```text
loss = action_flow_loss + world_model_loss_weight * world_model_loss
```

`world_model_loss_weight` defaults to `0.1`. The world predictor uses the backbone
learning rate. In Studio, new VLA-JEPA runs have **JEPA-assisted fine-tuning**
checked by default; uncheck it for action-only training. Local and remote jobs
carry the same setting. Continuing an existing Studio model preserves the mode
saved in its checkpoint.

The constructor is lazy without explicit features. During training, the dataset
supplies ordered camera/state/action features and normalization statistics.
`configure_model()` builds the resolved architecture once. The model itself
receives network arguments, not dataset features or a policy config.

By default, backbone and action-head learning rates are `1e-5` and `1e-4`.
`train_action_head_only=True` freezes Qwen, including the learned readout tokens;
this saves training memory but limits adaptation to a different robot. It also
disables JEPA supervision, matching the upstream head-only behavior.

`reinitialize_projections=True` permits replacing only incompatible robot
input/output projections when the dataset's state or action width differs from
the checkpoint. The loader logs which tensors it reinitializes and rejects
unrelated missing, unexpected, or mismatched weights. Fine-tune these new
projections before inference. The checkpoint's original robot uses a different
action space from SO-101; matching vector width alone does not imply matching
control semantics.

Actions use min/max normalization and states use mean/std normalization by
default. Training uses the dataset's statistics and preserves its gripper units;
it does not automatically apply the released checkpoint's LIBERO gripper
snapping or sign convention. `clip_normalized_actions` clips only min/max or
quantile-normalized action dimensions to `[-1, 1]`; masked dimensions and
mean/std or identity modes stay unbounded.
Images must be batched RGB tensors, channels-first or channels-last, either uint8 or
floating point in `[0, 1]`. Supply one task string per observation. Padded action
steps at episode boundaries do not supervise training.

Full training requests eight consecutive frames per camera by default. Only the
current image and state (offset zero) condition the policy. The frozen encoder
provides video-feature targets, and the predictor uses frame-causal attention.
Padded future tubelets at episode boundaries do not contribute to the JEPA loss.
With the released two-view predictor, a single camera is duplicated only for the
world model; extra cameras still reach Qwen, but only the first two train the
world model. Camera order follows the feature contract.

## LoRA and DoRA

Enable adapters in Python or pass `--model.lora_enabled=true` to the training CLI:

```python
policy = VLAJEPA(lora_enabled=True, lora_rank=32)
```

Set `lora_use_dora=True` for DoRA. Studio offers the same controls for local and
remote VLA-JEPA training. Default targets cover Qwen's language attention and MLP
projections, the DiT action head and robot projections, and the world predictor
when enabled. Base weights, the visual backbone, target encoder, and token
embeddings stay frozen. Only adapter parameters train.

Use `lora_target_modules` to override the target regex or supply a tuple of module
suffixes. Do not combine adapters with `train_action_head_only=True`; target the
action head explicitly instead. Adapters default to float32 even with a bfloat16
backbone; `lora_adapter_dtype="auto"` inherits each base layer's precision.

`lora_lr_scale` defaults to `10.0` and multiplies both optimizer learning rates
and the scheduler's decay rate. Set it to `1.0` if your learning rates already
account for adapter training. Lightning checkpoints retain adapters and their
configuration for offline resume. These paths have smoke-test coverage, not a
validated robot-task LoRA recipe or a measured memory-reduction claim.

## Inference and checkpoints

For a released checkpoint with its original feature contract:

```python
from physicalai.policies import VLAJEPA

policy = VLAJEPA.from_pretrained("lerobot/VLA-JEPA-LIBERO").to("cuda").eval()
actions = policy.predict_action_chunk(observation)
```

`from_pretrained()` rejects incompatible projections by default. Match the
checkpoint's camera names/order, normalization, state representation, and action
semantics before running it on hardware. LIBERO is a simulation checkpoint, not
an SO-101 policy.

After training:

```python
trainer.save_checkpoint("vla-jepa.ckpt", weights_only=True)
policy = VLAJEPA.load_from_checkpoint(
    "vla-jepa.ckpt", weights_only=True, enable_world_model=False
).eval()
```

Lightning checkpoints include the resolved architecture, feature statistics, and
tokenizer/image-processor assets. Restore does not download the original weights,
Qwen configuration, or tokenizer. `VLAJEPA.from_config(config)` also builds without
network access and does not load pretrained weights. Full-mode checkpoints also
include the frozen encoder and trained world predictor. Loading without an
`enable_world_model` override preserves their training mode for resume. Explicit
`enable_world_model=False` drops these training-only components when loading for
inference or action-only continuation. Older action-only checkpoints stay
action-only; initialize from released pretrained weights to start a new full-mode
run rather than silently adding random world-model weights.

`from_pretrained()` defaults to action-only loading for inference. Set
`enable_world_model=True` there only when intentionally loading the full training
architecture. Neither mode runs the world model during action prediction.

The model returns a full `(batch, chunk_size, action_dim)` chunk. `select_action`
uses the shared policy queue and executes `n_action_steps` before replanning.
Call `reset()` at episode boundaries.

## Export

Export a trained policy with the Python API:

```python
policy.export("./exports/vla-jepa", backend="torch")
```

Or use the CLI:

```bash
uv run --no-sync physicalai export \
  --policy physicalai.policies.VLAJEPA \
  --ckpt_path vla-jepa.ckpt \
  --backend torch \
  --output_dir ./exports/vla-jepa
```

The directory contains `vlajepa.pt` and `manifest.json`. Torch exports include
feature statistics and processor assets, so they reload offline with Studio's
Torch adapter installed:

```python
from physicalai.inference import InferenceModel

model = InferenceModel("./exports/vla-jepa", backend="torch", device="cuda")
outputs = model({"images.front": image_array, "state": state_array, "task": ["pick up the object"]})
```

Use the camera names and state feature from your trained policy, batched CHW
images, and one task string per batch item. The policy handles normalization
and action denormalization. Runtime trims the output chunk to `n_action_steps`,
matching the policy's execution prefix.

Export folds LoRA/DoRA adapters into base weights and records an adapter-free
configuration. It restores the live model's adapters and weights after export,
including on failure. Merging float32 adapters into bfloat16 layers introduces
rounding; compare exported actions before hardware use. To omit the training-only
world modules from a Torch artifact, load the checkpoint with
`enable_world_model=False` before export.

### OpenVINO

```python
policy.export("./exports/vla-jepa-ov", backend="openvino")
model = InferenceModel("./exports/vla-jepa-ov", backend="openvino", device="CPU")
```

The CLI accepts `--backend openvino` with the same checkpoint and policy arguments.
The package contains `vlajepa.xml`, `vlajepa.bin`, `processor.json`, and
`manifest.json`. The graph runs the vision encoder, language decoder, and all
flow-matching steps, then returns denormalized actions. It excludes the world
encoder and predictor even when exporting a joint-training checkpoint.

OpenVINO inference supports batch size one and the exported camera geometry.
Task length can vary between calls. The Runtime preprocessor reuses the saved
Qwen processor and tokenizer offline, so the inference environment still needs
`physicalai-train[vla_jepa]` and its PyTorch/Transformers dependencies. This is
not a torch-free deployment package.

Each call samples fresh Gaussian flow noise outside the graph. For numerical
comparisons, pass `noise` with shape `(1, chunk_size, action_dim)` in the Runtime
input dictionary and use the same noise in the Torch model. Set
`INFERENCE_PRECISION_HINT="f32"` when constructing `InferenceModel` for CPU
parity checks; the device's default precision can introduce additional rounding.

## Local smoke measurements

On an NVIDIA H100 PCIe, the pretrained base checkpoint produced a seven-action
chunk in about 55 ms after warmup. Peak PyTorch-allocated GPU memory was 4.6 GiB
for inference and 9.2 GiB through an action-only training backward pass. Full-mode
backward used about 11 GiB with eight video frames and the pretrained encoder and
predictor. The tests used batch size one, two 224-by-224 camera inputs,
four denoising steps, bfloat16 Qwen
weights, a float32 action head, and gradient checkpointing.

These are synthetic smoke measurements, not minimum requirements. The backward
measurement excludes AdamW optimizer state, and no robot task-success evaluation
was run. Camera/token count, batch size, precision, and optimizer state change
memory usage. Preprocessing matched the upstream reference exactly; the loaded
DiT's action predictions differed by less than `4e-7` with matched noise.
The CLI also completed one fast-dev-run optimizer step in each training mode on
a single PushT episode. This checks dataset and training integration, not task
performance.

## Released weights and limits

The loader supports the converted safetensors checkpoints
`lerobot/VLA-JEPA-Pretrain`, `lerobot/VLA-JEPA-LIBERO`, and
`lerobot/VLA-JEPA-SimplerEnv`, pinned to reviewed revisions. Using these artifacts
does not require LeRobot's policy implementation. Custom Hub checkpoints require
a concrete `pretrained_revision` commit SHA. Original author `.pt` files are not
loaded.

Full mode loads the frozen V-JEPA2 encoder and native world predictor from the
same converted checkpoint. Action-only loading excludes their tensors and does
not download their configuration or construct them. Both modes preserve the
pretrained latent-action prompt and final decoder pre-normalization readout.

Studio lists this policy with Torch and OpenVINO export and no minimum-VRAM claim. A
successful synthetic or pretrained smoke test does not establish task success or
a safe hardware deployment. The first fine-tuning run still needs evaluation on
the intended robot and tasks.

- [Paper](https://arxiv.org/abs/2602.10098)
- [Author implementation](https://github.com/ginwind/VLA-JEPA)
- [Converted pretrained weights](https://huggingface.co/lerobot/VLA-JEPA-Pretrain)
