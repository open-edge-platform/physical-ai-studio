# RoboCasa Kitchen Gym

Wrapper for the [RoboCasa Kitchen](https://github.com/robocasa/robocasa) benchmark environment.
Ported from `lerobot/src/lerobot/envs/robocasa.py` with PAS-native types and the lerobot
`AsyncVectorEnv` plumbing removed.

## Scope

Current implementation is **RoboCasa Kitchen-only** (robocasa v1.0 task groups and named kitchen tasks).
It does **not** implement the full RoboCasa365 benchmark surface.

This is intentional: RoboCasa365 asset footprint and setup/runtime cost are much larger than the
Kitchen subset, so we keep CI and local development on the smaller Kitchen path first.

## Why a separate venv

RoboCasa is **not** a `pyproject.toml` extra. The unified uv resolve is unsolvable:

- `robocasa`'s [`setup.py`](https://github.com/robocasa/robocasa/blob/main/setup.py#L34) pins `lerobot==0.3.3`, colliding with our `lerobot>=0.5.1`.
- `robocasa` requires `robosuite` master (≥1.5dev) for `HybridMobileBase`; the
  `[libero]` extra needs PyPI `robosuite==1.4.0`. They cannot share a venv.
- `robocasa`'s dead `tianshou==0.4.10` pin transitively requires `protobuf<3.20`,
  colliding with `onnx`'s `protobuf>=3.20`.

Install into a dedicated `.venv-robocasa`:

```bash
uv venv .venv-robocasa
source .venv-robocasa/bin/activate
uv sync --active --extra cu128          # or cpu / xpu — do NOT use --extra all or --extra libero

bash library/scripts/benchmark/install_robocasa.sh

# Download kitchen assets (~4.4 GB on disk; prompts interactively)
yes y | python -m robocasa.scripts.download_kitchen_assets \
    --type tex tex_generative fixtures_lw objs_lw

# Headless servers
export MUJOCO_GL=egl
```

SHAs in `install_robocasa.sh` match the lerobot Dockerfile pins:
`robocasa@56e355c` and `robosuite@aaa8b9b`. Bump them together.

## Quick start

```python
from physicalai.gyms import RoboCasaGym

# Single named task
gym = RoboCasaGym(task="CloseFridge")
obs, info = gym.reset(seed=0)
action = gym.sample_action()
next_obs, reward, terminated, truncated, info = gym.step(action)
gym.close()

# Task-group keyword (18 atomic tasks in robocasa v1.0)
from physicalai.gyms import create_robocasa_gyms

gyms = create_robocasa_gyms(tasks="atomic_seen")
```

## API

### `RoboCasaGym`

```python
RoboCasaGym(
   task: str,                          # single task name (use create_robocasa_gyms for group keywords)
   camera_names: Sequence[str] | None, # default: DEFAULT_CAMERAS
   obs_type: str,                      # "pixels" | "pixels_agent_pos" (default)
   render_mode: str,                   # "rgb_array" (default)
   observation_height: int,            # default 256
   observation_width: int,             # default 256
   split: str | None,                  # overrides auto-resolved split
   episode_length: int | None,         # MuJoCo horizon
   obj_registries: Sequence[str],      # default ("lightwheel",)
)
```

**`task` values for `RoboCasaGym`:**

| Value                     | Resolves to                                   |
| ------------------------- | --------------------------------------------- |
| `"TaskName"` or `"T1,T2"` | explicit names, split `None` (auto = `"all"`) |

Task-group keywords are expanded by `create_robocasa_gyms(...)`, not by `RoboCasaGym(...)`.

**task-group values for `create_robocasa_gyms(tasks=...)`:**

| Value                            | Resolves to                            |
| -------------------------------- | -------------------------------------- |
| `"atomic_seen"`                  | 18 v1.0 atomic tasks, split `"target"` |
| `"composite_seen"`               | composite tasks, split `"target"`      |
| `"composite_unseen"`             | composite tasks, split `"target"`      |
| `"pretrain50"` … `"pretrain300"` | pretrain partition, split `"pretrain"` |

**Observation:**

`reset()` and `step()` return `physicalai.data.observation.Observation`:

| Field                              | Shape                | dtype     | Notes                                                            |
| ---------------------------------- | -------------------- | --------- | ---------------------------------------------------------------- |
| `images["robot0_agentview_left"]`  | `(1, 3, H, W)`       | `float32` | normalized to `[0, 1]`                                           |
| `images["robot0_agentview_right"]` | `(1, 3, H, W)`       | `float32` |                                                                  |
| `images["robot0_eye_in_hand"]`     | `(1, 3, H, W)`       | `float32` |                                                                  |
| `state`                            | `(1, 16)`            | `float32` | base_pos(3)+base_quat(4)+ee_pos_rel(3)+ee_quat_rel(4)+gripper(2) |
| `task`                             | `list[str]` length 1 |           | language description                                             |

Camera names are raw RoboCasa v1.0 names. Per-policy renames go through a policy-side adapter, not here.

**Action:**

Flat `(12,)` `torch.Tensor`: `base_motion(4) + control_mode(1) + ee_pos(3) + ee_rot(3) + gripper(1)`.
`step()` splits it internally via `convert_action()`.

### `create_robocasa_gyms`

```python
create_robocasa_gyms(
    tasks: str | list[str],   # group keyword or list of task names
    **gym_kwargs,             # forwarded to RoboCasaGym
) -> list[RoboCasaGym]
```

Returns one `RoboCasaGym` per task name.

### Module-level constants

| Name                     | Value                                                                       | Notes                          |
| ------------------------ | --------------------------------------------------------------------------- | ------------------------------ |
| `OBS_STATE_DIM`          | `16`                                                                        | proprioceptive state dimension |
| `ACTION_DIM`             | `12`                                                                        | flat action dimension          |
| `DEFAULT_CAMERAS`        | `("robot0_agentview_left", "robot0_eye_in_hand", "robot0_agentview_right")` |                                |
| `DEFAULT_OBJ_REGISTRIES` | `("lightwheel",)`                                                           | avoids objaverse NaN crash     |

## Known upstream gotchas

Three bugs from the lerobot port are already encoded as workarounds:

1. **`split="test"` default** — `RoboCasaGymEnv` defaults to `split="test"` which
   `create_env` rejects. The wrapper always passes `split="all"` when no split is set.

2. **objaverse NaN crash** — sampling from a registry with zero objects causes
   `Probabilities contain NaN`. Fixed by defaulting to `obj_registries=("lightwheel",)`.
   If you see this error at `reset()`, re-run the asset download with `--type objs_lw`.

3. **`atomic_seen` → `split="target"`** — robocasa's own group maps to `split="target"`,
   not `"all"`. `_TASK_GROUP_SPLITS` encodes this mapping for all group keywords.

## Paper-24 task mapping (robocasa v1.0)

The paper benchmark used 24 tasks; robocasa v1.0 renamed/merged some of them.

| Status     | Count | Meaning                                                        |
| ---------- | ----- | -------------------------------------------------------------- |
| ✅ direct  | 19    | Exact v1.0 class name; compare 1:1 to paper SR                 |
| 🟡 proxy   | 4     | Door tasks merged into `OpenCabinet` / `CloseCabinet` families |
| ❌ dropped | 1     | `CoffeePressButton` — no v1.0 equivalent                       |

Use `atomic_seen` for smoke-testing; use the 19 ✅ tasks for paper parity runs.

## Future: RoboCasa365 support plan

To extend from Kitchen-only to RoboCasa365, keep the current API stable and add capability in layers:

1. **Asset/profile layering**

   - Add explicit install profiles in `install_robocasa.sh` (for example: `kitchen`, `robocasa365`).
   - Keep `kitchen` as the default profile.
   - Add profile-specific download commands and disk-size estimates in this doc.

2. **Task catalog separation**

   - Add a versioned task catalog module for RoboCasa365 task groups.
   - Keep Kitchen mappings in `_TASK_GROUP_SPLITS` and add a parallel mapping for 365 groups.
   - Expose a strict validation error when a 365 group is requested but 365 assets are not installed.

3. **Config surface extension (backward compatible)**

   - Add an optional benchmark selector (for example `benchmark="kitchen" | "robocasa365"`).
   - Default remains `kitchen` so existing users are unaffected.
   - Keep `RoboCasaGym(task=...)` behavior unchanged for Kitchen users.

4. **Resource-aware CI strategy**

   - Keep existing `integration-tests-robocasa` as Kitchen-only.
   - Add a separate, non-required RoboCasa365 nightly workflow (or manually triggered workflow) to avoid
     blocking regular PR latency with large downloads.
   - Cache heavy assets/artifacts aggressively and pin SHAs for reproducibility.

5. **Validation matrix**

   - Add unit tests for 365 task resolution and split mapping.
   - Add one smoke E2E task for RoboCasa365 first, then expand coverage gradually.
   - Publish a parity table similar to the Paper-24 section once a stable 365 subset is defined.

6. **Docs and migration notes**
   - Document asset requirements, expected setup time, and minimum hardware.
   - Add clear guidance on choosing Kitchen vs RoboCasa365 depending on use case (CI, local dev, full benchmark).

## CI

The RoboCasa integration tests run in a separate job (`integration-tests-robocasa`)
that creates its own `.venv-robocasa` and downloads kitchen assets. They are **not**
included in the regular `integration-tests` job because LIBERO and RoboCasa cannot
share a venv (conflicting `robosuite` versions).

Tests live in `library/tests/integration/gyms/test_robocasa_e2e.py` (marked
`integration` + `slow`). Unit tests that do not spin up MuJoCo are in
`library/tests/unit/gyms/test_robocasa.py`.
