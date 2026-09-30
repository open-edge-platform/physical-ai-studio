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

## Note on Export

Export via ONNX, OpenVINO, or ExecuTorch is currently out of scope for the
multimodal diffusion backbone. Deployment and benchmarking are performed via
native PyTorch inference using `Cosmos3` and `InferenceModel`.
