# NVIDIA Cosmos3 Policy

## Overview

PhysicalAI integrates the NVIDIA Cosmos3 World Action Model policy ([NVIDIA et al. 2026](https://arxiv.org/abs/2606.02800)) for physical robot control. Cosmos3 uses rectified flow matching
over continuous action spaces conditioned on visual observations (single-view or
multi-view composites) and proprioceptive states.

All implementations provide:

- ✅ Full PyTorch Lightning integration for training and validation
- ✅ PEFT LoRA/DoRA fine-tuning and full generation-tower training modes
- ✅ Embodiment-aware action spaces (`identity` with per-dataset normalization, `joint_pos` with flipped gripper)
- ✅ Multi-view camera composition (T-shape view for DROID, horizontal concat)
- ✅ Closed-loop action chunk inference (`chunk_size: 32`) with automatic action queueing
- ✅ Multi-hardware support across Intel XPU, NVIDIA CUDA, and CPU

## Architecture

Cosmos3 is split into a self-contained module under `library/src/physicalai/policies/cosmos3/`:

```text
library/src/physicalai/policies/cosmos3/
├── config.py           # Cosmos3Config (typed dataclass extending Config)
├── flow_matching.py    # Rectified flow matching objective & loss
├── model.py            # Cosmos3Model (Cosmos3 DiT + VAE + Action Heads)
├── normalization.py    # Embodiment normalization (none, minmax, quantile)
├── pipeline.py         # PolicyPipelineWithState & XPU driver checks
├── policy.py           # Cosmos3 LightningModule wrapper
├── preprocessor.py     # Multi-camera view composition (T-shape & horizontal)
├── representation.py   # Embodiment action-space mapping & gripper flipping
└── surgery.py          # Model surgery (domain action heads & LoRA setup)
```

## Embodiments & Action Spaces

The policy is configured using an `embodiment` identifier:

| Embodiment | Action Space | Action Dim | Normalization | Gripper Convention | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `pusht` | `identity` | 2 | `minmax` | Standard | Push-T 2D planar position control |
| `droid_lerobot` | `joint_pos` | 8 | `none` | Inverted (`1 - g`) | DROID 7 arm joints + 1 gripper position |
| `aloha` | `identity` | 14 | `minmax` | Standard | Dual-arm Aloha joint positions |

### Dataset Columns

Cosmos3 consumes the canonical combined `observation.state` and `action` columns,
like the other policies in this repo. When a dataset stores an action's components
in separate columns (e.g. DROID's 7 joints and 1 gripper), the datamodule combines
them into a single column (8D `[joint(7), gripper(1)]`) *before* the policy sees it —
concatenation happens at the datamodule level, not inside the policy.

### Camera Composition & Viewpoints

Unlike VLM policies that accept arbitrary tokenized camera streams via dataset feature contracts, Cosmos3 is a single-canvas video diffusion model. Multi-camera observations are stitched into specific geometric mosaics (e.g., T-shape for DROID) and paired with discrete viewpoint prompt tags (`view_point`, with compatibility alias `viewpoint`) expected by pretrained embodiment checkpoints.

### Prompt Conditioning

Cosmos3 conditions on a **per-task** instruction: each sample's `task` string (provided by the datamodule) is turned into the conditioning prompt. There is no global/static prompt — a sample with no task text conditions on an empty string.

The `prompt_format` config knob controls how that per-task text is turned into the caption the transformer sees:

| `prompt_format` | Caption sent to the model |
| :--- | :--- |
| `task_description` (default) | The raw per-task text, verbatim. |
| `augmented_text` | The task text plus the flat duration/FPS and resolution template sentences. |
| `augmented_json` | The structured JSON caption (viewpoint framing + duration + fps + resolution + aspect_ratio) that the released NVIDIA Cosmos policy checkpoints were trained on. |

All augmentation logic lives in the `diffusers` `Cosmos3OmniPipeline`; Studio only selects the format. `task_description` keeps the prompt minimal and portable, which is the sensible default for training your own heads on this repo's datasets.

#### Matching the released DROID policy checkpoints

Both `Cosmos3-Edge-Policy-DROID` and `Cosmos3-Nano-Policy-DROID` share the same DROID data/prompt recipe; they differ only in the base backbone (`nvidia/Cosmos3-Edge` vs `nvidia/Cosmos3-Nano`, the latter with `max_action_dim=64`), **not** in the prompt. To reproduce their conditioning set:

| Setting | Value for parity | Studio `droid/default.yaml` default |
| :--- | :--- | :--- |
| `prompt_format` | `augmented_json` | `task_description` |
| `resolution_tier` | `480` | `256` |
| `fps` | `15` | `15` ✅ |
| `chunk_size` | `32` (→ 33 frames incl. state token) | `32` ✅ |
| viewpoint | `concat_view` | auto for `droid_lerobot` ✅ |

The Studio DROID default deliberately uses `task_description` + `resolution_tier: 256` for lighter fine-tuning; switch both to the parity values above only when you specifically need byte-comparable prompts against the released checkpoints.

## Adding a new embodiment

`embodiment` is intentionally typed as `str` (not a `Literal`) so new embodiments
register through the mapping tables below rather than a closed enum. The built-ins
are `pusht`, `droid_lerobot`, and `aloha`; an unregistered embodiment raises at
model init listing the registered ones.

To add one:

1. Register a numeric domain id in `_EMBODIMENT_TO_DOMAIN_ID` and the raw action
   width in `_EMBODIMENT_TO_RAW_ACTION_DIM` (`model.py`).
2. Map an action space in `EMBODIMENT_ACTION_SPACE` (`representation.py`); unlisted
   embodiments default to `identity`.
3. Optionally set the action-normalization method in `EMBODIMENT_NORMALIZATION`
   (default `minmax`) and gripper inversion in `EMBODIMENT_GRIPPER_FLIPPED`
   (default off).
4. Optionally set a default viewpoint tag in `DEFAULT_EMBODIMENT_VIEWPOINTS` and, for
   multi-camera mosaics, add the embodiment to `T_SHAPE_EMBODIMENTS` or
   `HORIZONTAL_EMBODIMENTS` (`preprocessor.py`).

These cannot be inferred from dataset metadata: the domain id, action-space
semantics, gripper convention, and viewpoint framing are contracts of the
pretrained checkpoint, not properties of the recorded dataset.

## Quickstart

### Python API

```python
from physicalai.data.lerobot import LeRobotDataModule
from physicalai.policies.cosmos3 import Cosmos3
from physicalai.train import Trainer

# 1. Initialize dataset
datamodule = LeRobotDataModule(
    repo_id="lerobot/pusht",
    train_batch_size=1,
)

# 2. Instantiate policy
policy = Cosmos3(
    embodiment="pusht",
    pretrained_model_name_or_path="nvidia/Cosmos3-Edge",
    mode="peft",
    lora_rank=32,
    dtype="bfloat16",
)

# 3. Train
trainer = Trainer(max_epochs=10, accelerator="xpu")
trainer.fit(model=policy, datamodule=datamodule)
```

### CLI

Configs are available under `library/configs/physicalai/cosmos3/`:

```bash
# Push-T (planar 2D)
physicalai fit --config configs/physicalai/cosmos3/pusht/default.yaml

# DROID (8D joint pos with split-column combining)
physicalai fit --config configs/physicalai/cosmos3/droid/default.yaml
```

## Advanced

Expert/research knobs; keep the defaults unless you have a specific reason.

### Paradigm

Cosmos3 treats action learning as three tasks over a single video world model,
selected by `paradigm`:

- `policy` (default) — from a first frame (plus state), jointly rolls out the
  future video and the action chunk. This is the mode used to drive a robot.
- `fd` (forward dynamics) — from a first frame and a given action sequence, rolls
  out the resulting future video (a predictive world model).
- `id` (inverse dynamics) — infers the actions that connect the observed video
  frames.
- `joint` — trains on a random mix of the three.

The three tasks share one backbone, so training video prediction (`fd`) and action
inference (`id`) alongside `policy` grounds the action head in the same learned
world dynamics it generates against. Use `policy` for control.

| Knob | Default | When to change |
| :--- | :--- | :--- |
| `prompt_format` | `task_description` | Set `augmented_text` / `augmented_json` for prompt parity with released checkpoints (see [Prompt Conditioning](#prompt-conditioning)). |
| `normalizer_stats_path` | `None` | Point at a cosmos-format action-normalizer stats JSON to run a pre-trained per-embodiment head that expects quantile-normalized actions. |
| `action_space` | `None` (auto) | Override the embodiment's resolved action space (`identity` / `joint_pos`). |
| `view_point` | `None` (auto) | Override the viewpoint prompt tag inferred from the embodiment and composition. |
| `resolution_tier` / `fps` | `256` / `10` | Raise for checkpoint parity (e.g. `480` / `15` for DROID) at higher compute cost. |

## Note on Export

Export via ONNX, OpenVINO, or ExecuTorch is currently out of scope for the
multimodal diffusion backbone. Deployment and benchmarking are performed via
native PyTorch inference using `Cosmos3` and `InferenceModel`.
