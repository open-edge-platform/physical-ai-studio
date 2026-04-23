# LeRobot Policy Integration

## Overview

PhysicalAI integrates LeRobot policies through a single base class
(`LeRobotPolicy`) and one thin alias per first-class supported policy
(`act`, `diffusion`, `groot`, `pi0`, `pi05`, `pi0_fast`, `smolvla`,
`xvla`), each binding a fixed `policy_name`. Configuration flows through
LeRobot's own `PreTrainedConfig` dataclasses (`ACTConfig`,
`DiffusionConfig`, …), so the upstream contract is the single source of
truth for policy parameters. Other LeRobot-registered policies remain
reachable through `LeRobotPolicy` directly as a best-effort escape hatch.

All wrappers provide:

- ✅ Verified output equivalence with native LeRobot for the validated subset
  (see [Validation](#validation) for which policies are covered)
- ✅ Full Lightning integration (training, validation, inference)
- ✅ Lazy or eager initialization (features extracted from a `DataModule`
  or supplied explicitly)
- ✅ Automatic data format handling (see [Data Integration](../data/lerobot.md))

## Design Pattern

### Lightning-First with Third-Party Framework Support

```text
┌───────────────────────────────────────────────┐
│            PhysicalAI (Lightning)             │
│   ┌───────────────────────────────────────┐   │
│   │  LeRobotPolicy (LightningModule)      │   │
│   │  ┌─────────────────────────────────┐  │   │
│   │  │   LeRobot Native Policy         │  │   │
│   │  │   (Thin Delegation)             │  │   │
│   │  └─────────────────────────────────┘  │   │
│   └───────────────────────────────────────┘   │
└───────────────────────────────────────────────┘
```

**Key Principles**:

1. **No Reimplementation** — delegate to native LeRobot.
2. **Thin Wrapper** — Lightning interface only.
3. **Config as Source of Truth** — LeRobot's `PreTrainedConfig` carries every
   tunable parameter; the wrapper does not re-declare them.
4. **Verified Equivalence** — outputs match native LeRobot for the validated
   subset (`act`, `diffusion`, `smolvla`, `pi0`, `pi05`, `pi0_fast`); see
   [Validation](#validation).

## Architecture

### File Structure

```text
library/src/physicalai/
└── policies/lerobot/
    ├── __init__.py              # Public API, availability checks
    ├── mixin.py                 # LeRobotFromConfig mixin (PreTrainedConfig flow)
    ├── policy.py                # LeRobotPolicy + NamedLeRobotPolicy (bases)
    ├── aliases.py               # Named aliases (one per registered policy)
    └── README.md                # Module documentation
```

Every named alias lives in `aliases.py` as a thin subclass of
`NamedLeRobotPolicy` — three lines binding `POLICY_NAME`. The set of
aliases is a curated subset of LeRobot's `PreTrainedConfig.get_known_choices()`
registry; other registry entries remain reachable through
`LeRobotPolicy(policy_name=...)` directly as a best-effort escape hatch.
Adding a new named alias is a one-class addition. Keeping `LeRobotPolicy`
and `NamedLeRobotPolicy` together in `policy.py` reflects that the latter
is a tightly-coupled specialization of the former.

For data module architecture and format conversion details, see
[LeRobot Data Integration](../data/lerobot.md).

### Implementation Components

#### 1. `LeRobotPolicy` — the dynamic base

```python
class LeRobotPolicy(Policy, LeRobotFromConfig):
    """Single LightningModule that wraps any registered LeRobot policy."""

    def __init__(
        self,
        policy_name: str,
        *,
        config: PreTrainedConfig | None = None,
        input_features: dict[str, PolicyFeature] | None = None,
        output_features: dict[str, PolicyFeature] | None = None,
        dataset_stats: dict | None = None,
        policy_config: dict[str, Any] | None = None,
        **overrides: Any,
    ) -> None: ...
```

`policy_name` selects the LeRobot policy class. `config` accepts a fully
built `PreTrainedConfig` (`ACTConfig`, `DiffusionConfig`, …). When
`input_features` / `output_features` are omitted, the wrapper extracts them
lazily from the attached `DataModule` during `setup()`.

#### 2. `NamedLeRobotPolicy` — base for thin aliases

```python
class ACT(NamedLeRobotPolicy):
    POLICY_NAME = "act"
```

Named aliases inherit the entire `LeRobotPolicy` lifecycle but pre-bind
`policy_name` from the class-level `POLICY_NAME`. The base also enforces
two invariants the bare `LeRobotPolicy` cannot:

1. `Subclass.from_dataset(...)` returns an instance of `Subclass`, not a
   bare `LeRobotPolicy`.
2. `Subclass(policy_name=other)` is rejected, so a class named `ACT` can
   never silently build a different policy.

#### 3. `LeRobotFromConfig` mixin

Provides the configuration entry points shared by every wrapper:

- `from_config(config)` — auto-detects whether `config` is a
  `PreTrainedConfig` instance, a dict, a YAML path, or a dataclass.
- `from_pretrained(pretrained_path)` — loads weights and config from a
  LeRobot-compatible saved directory or HuggingFace Hub repo. For
  Lightning `.ckpt` files, use `LeRobotPolicy.load_from_checkpoint(...)`.

LeRobot's own `PreTrainedConfig` dataclasses already enumerate every
tunable parameter with defaults, validation, and serialization, so the
wrapper does not re-declare them.

## Usage

### Direct instantiation

```python
from physicalai.policies.lerobot import ACT
from lerobot.policies.act.configuration_act import ACTConfig

# Eager: provide features explicitly
policy = ACT(
    config=ACTConfig(dim_model=512, chunk_size=100, optimizer_lr=1e-5),
    input_features={...},
    output_features={...},
)

# Lazy: features extracted from the DataModule in setup()
policy = ACT(config=ACTConfig(dim_model=512))
```

### From a dataset

```python
policy = ACT.from_dataset("lerobot/pusht", dim_model=512)
```

`ACT.from_dataset(...)` returns an `ACT` instance (not a bare
`LeRobotPolicy`), preserving `isinstance` discrimination across the family.

### From a pretrained directory or Hub repo

```python
policy = ACT.from_pretrained("path/to/saved_dir")
# or, when the policy class is unknown ahead of time:
policy = LeRobotPolicy.from_pretrained("path/to/saved_dir")
```

Lightning `.ckpt` files are loaded via the standard Lightning entry point:

```python
policy = ACT.load_from_checkpoint("path/to/model.ckpt")
```

### LightningCLI / YAML

```yaml
# configs/lerobot/act.yaml
model:
  class_path: physicalai.policies.lerobot.ACT
  init_args:
    policy_config:
      dim_model: 512
      chunk_size: 100
      optimizer_lr: 1.0e-5
```

```bash
physicalai fit --config configs/lerobot/act.yaml
physicalai fit --config configs/lerobot/act.yaml \
  --model.policy_config.dim_model 1024
```

### Dynamic policy selection

When the policy name is itself driven by configuration, instantiate the
bare `LeRobotPolicy` and supply `policy_name`:

```yaml
model:
  class_path: physicalai.policies.lerobot.LeRobotPolicy
  init_args:
    policy_name: diffusion
    policy_config:
      horizon: 16
      n_action_steps: 8
```

## When to Use Which Class

| Use case                                                     | Class                               |
| ------------------------------------------------------------ | ----------------------------------- |
| Production code, fixed policy choice                         | Named alias (`ACT`, `Diffusion`, …) |
| YAML configs targeting a stable class path                   | Named alias                         |
| `isinstance(policy, ACT)` checks                             | Named alias                         |
| Policy name driven by a CLI flag or runtime config           | `LeRobotPolicy`                     |
| Loading a saved-directory or Hub repo of unknown policy type | `LeRobotPolicy.from_pretrained`     |
| Loading a Lightning `.ckpt`                                  | `<Class>.load_from_checkpoint`      |

Native physicalai policies (`physicalai.policies.act` etc.) keep their
explicit signatures because physicalai owns those contracts. The
config-first design described here applies only to LeRobot wrappers.

## Implementation Details

### Why This Works

1. **Thin delegation** — the wrapper only adds Lightning hooks; all
   computation runs in the underlying `lerobot_policy`. Zero overhead.
2. **Weight preservation** — direct attribute access, state-dict
   pass-through, transparent checkpointing.
3. **Feature preservation** — all LeRobot methods (`reset`, `select_action`,
   `forward`) remain accessible via `lerobot_policy`.

### Where research code belongs

Policy-specific research code (custom forward passes, explainability
hooks, exportable variants, etc.) belongs in a first-party physicalai
package such as `physicalai.policies.smolvla`, **not** in this LeRobot
adapter layer. The adapter layer is reserved for thin aliases over
upstream LeRobot policies.

## Validation

Two registries describe the wrapper's coverage:

- `SUPPORTED_POLICIES` — first-class LeRobot policies exposed as named
  `NamedLeRobotPolicy` subclasses (`act`, `diffusion`, `groot`, `pi0`,
  `pi05`, `pi0_fast`, `smolvla`, `xvla`).
- `VALIDATED_EQUIVALENCE_POLICIES` — subset with measured wrapper-vs-native
  numerical equivalence under tier-appropriate tolerances: `rtol=atol=1e-6`
  at the unit tier (CPU, fp32) and `rtol=1e-5` (loss) / `rtol=5e-5` (weights)
  at the integration tier (Lightning Trainer, CUDA/XPU + bf16-mixed for VLAs).

| Policy                        | Unit (CPU) | Integration (CUDA/XPU) | Notes                                   |
| ----------------------------- | ---------- | ---------------------- | --------------------------------------- |
| `act`, `diffusion`, `smolvla` | ✅         | ✅                     | Validated CPU + accelerator tiers       |
| `pi0`, `pi05`, `pi0_fast`     | —          | ✅                     | VLA, accelerator + bf16-mixed only      |
| `groot`                       | —          | xfail                  | Upstream hardcodes `flash_attention_2`  |
| `xvla`                        | —          | xfail                  | Requires explicit `vision_config` kwarg |

Tracked limitations register as pytest `xfail(reason=...)` rather than silent
skips, so any future fix surfaces as `XPASS` and prompts moving the policy
into `VALIDATED_EQUIVALENCE_POLICIES`.

Other LeRobot-registered policies (`vqbet`, `tdmpc`, `sac`, `multi_task_dit`,
`reward_classifier`, `sarm`, `wall_x`) remain reachable through
`LeRobotPolicy(policy_name=...)` directly as a best-effort escape hatch — a
one-time `UserWarning` is emitted, no equivalence is asserted.

- [LeRobot GitHub](https://github.com/huggingface/lerobot)
- [LeRobot Documentation](https://huggingface.co/lerobot)
- [LeRobot Data Module Documentation](../data/lerobot.md)
- Module: `library/src/physicalai/policies/lerobot/`
- Data Module: `library/src/physicalai/data/lerobot/`
