# physicalai: LeRobot Integration Design

**Status**: Proposal
**Author**: [Your Name]
**Date**: 2026-01-13
**Relates to**: [LeRobot Policy Export Design](./policy_export_design.md)

> **Important: LeRobot export is our proposal, not an agreed standard.**
> The PolicyPackage format (`manifest.json`) described in this document is a design we have proposed to the LeRobot team. It has **not yet been reviewed or accepted** upstream. If the LeRobot team adopts a different export format or modifies the proposed schema, this integration design will need to adapt accordingly. The architectural approach (unified manifest format, no lerobot dependency at runtime) remains valid regardless of the final format — only the loader implementation would change.

---

## Executive Summary

This document describes how **physicalai** integrates with LeRobot's proposed PolicyPackage format. The integration is seamless because both physicalai-train and LeRobot use the **same unified `manifest.json` format**. The runtime reads `manifest.json` (pure JSON, no lerobot import) and maps `policy.kind` to built‑in runners. No LeRobot dependency is needed at deployment time.

**Key principle:** All packages (physicalai-train, LeRobot, custom) export models using the same `manifest.json` format. physicalai consumes them identically. No special-casing, no separate format loaders, no circular dependencies.

**Note on status**: The PolicyPackage export format is our proposal to the LeRobot team (see [LeRobot Export Suggestions](../internal/lerobot-export-suggestions.md)). The format details below reflect our proposed design. The integration approach is sound regardless of the final format the LeRobot team adopts.

---

## 1. Architecture Overview

```text
┌────────────────────────────────────────────────────────────────┐
│                         physicalai                              │
│                                                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   Adapters   │  │  Built‑in    │  │     Callbacks        │  │
│  │  (backends)  │  │   Runners    │  │  (instrumentation)   │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                 Unified Manifest Loader                   │  │
│  │                                                          │  │
│  │  manifest.json (same format for all model sources)       │  │
│  │  physicalai-train, LeRobot, custom — all use the same schema   │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
                              │
                              │ reads (pure file I/O)
                              ▼
               ┌──────────────────────────┐
               │     Exported Model       │
               │   (any source)           │
               │                          │
               │   manifest.json          │
               │   model artifacts        │
               └──────────────────────────┘
```

---

## 2. Unified Manifest Format

All packages (physicalai-train, LeRobot, custom) use the same `manifest.json` schema. This section describes the fields relevant to LeRobot policies specifically.

### Package Detection

A directory is an exported model package if it contains `manifest.json` with `"format": "policy_package"`:

```python
def is_policy_package(path: Path) -> bool:
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text())
    return manifest.get("format") == "policy_package"
```

### Manifest Fields Used

| Field           | physicalai Usage                                                                       |
| --------------- | -------------------------------------------------------------------------------------- |
| `format`        | Package type detection                                                                 |
| `version`       | Schema compatibility check                                                             |
| `policy.kind`   | Runner selection (`single_shot` → `SinglePassRunner`, `iterative` → `IterativeRunner`) |
| `artifacts`     | Backend artifact paths                                                                 |
| `io`            | Input/output validation                                                                |
| `action`        | Action semantics (chunk_size, n_action_steps)                                          |
| `iterative`     | Loop parameters (num_steps, scheduler)                                                 |
| `normalization` | Normalizer configuration                                                               |
| `x-physical-ai` | Extension fields (callbacks, adapter options)                                          |

---

## 3. Manifest Loader Implementation

### How It Works

The manifest loader is unified — there is no separate "LeRobot loader" vs "physicalai-train loader". The same code parses `manifest.json` for all model sources. The `policy.kind` field determines which built‑in runner to use.

```python
# physicalai/manifest_loader.py

class ManifestLoader:
    """Unified manifest loader for all model sources."""

    @staticmethod
    def detect(path: Path) -> bool:
        """Check if path contains a valid manifest."""
        manifest_path = path / "manifest.json"
        if not manifest_path.exists():
            return False
        try:
            manifest = json.loads(manifest_path.read_text())
            return manifest.get("format") == "policy_package"
        except (json.JSONDecodeError, KeyError):
            return False

    @staticmethod
    def load(
        path: Path,
        backend: str | None = None,
        device: str = "cpu",
        **kwargs
    ) -> "InferenceModel":
        """Load a model package into an InferenceModel."""
        manifest = json.loads((path / "manifest.json").read_text())

        # Validate schema version
        version = manifest.get("version", "1.0")
        if not version.startswith("1."):
            raise ValueError(f"Unsupported manifest version: {version}")

        # Select backend
        backend = backend or _select_default_backend(manifest)
        artifact_path = path / manifest["artifacts"][backend]

        # Create adapter (via inference core)
        adapter = get_adapter(backend)(artifact_path, device=device)

        # Select runner based on policy kind
        kind = manifest["policy"]["kind"]
        runner = _create_runner(kind, manifest, **kwargs)

        # Create normalizer (if specified)
        normalizer = _create_normalizer(path, manifest)

        # Load callbacks from extensions
        callbacks = _load_callbacks(manifest)

        return InferenceModel(
            adapter=adapter,
            runner=runner,
            normalizer=normalizer,
            callbacks=callbacks,
            metadata=manifest,
        )


def _create_runner(kind: str, manifest: dict, **kwargs) -> InferenceRunner:
    """Map policy.kind to a built‑in runner."""
    if kind == "single_pass":
        return SinglePassRunner()

    elif kind == "iterative":
        iter_config = manifest.get("inference", {})
        return IterativeRunner(
            num_steps=kwargs.get("num_steps", iter_config.get("num_steps", 10)),
            scheduler=kwargs.get("scheduler", iter_config.get("scheduler", "euler")),
            timestep_spacing=iter_config.get("timestep_spacing", "linear"),
        )

    elif kind == "two_phase":
        iter_config = manifest.get("inference", {})
        return TwoPhaseRunner(
            num_steps=kwargs.get("num_steps", iter_config.get("num_steps", 10)),
            scheduler=kwargs.get("scheduler", iter_config.get("scheduler", "euler")),
        )

    elif kind == "custom":
        # Custom runner specified via class_path
        runner_config = manifest.get("runner", {})
        return instantiate(runner_config)

    else:
        raise ValueError(f"Unknown policy kind: {kind}")
```

### Installation

The manifest loader is **built‑in** — it ships with physicalai. No extra install needed.

```bash
# This is all you need to run any exported model (physicalai-train, LeRobot, custom)
pip install physicalai
```

The loader reads `manifest.json` (pure JSON parsing) and maps `policy.kind` to built‑in runners. No `lerobot` import. No `physicalai-train` import. No `physicalai[lerobot]` extra.

---

## 4. Usage Examples

### Basic Usage (Unified API)

```python
from physicalai import InferenceModel

# Load LeRobot package (auto-detected via plugin)
model = InferenceModel("./pi0_exported")

# Run inference (raw outputs)
observation = {
    "observation.images.top": image_array,
    "observation.state": state_array,
}
outputs = model(observation)
action_chunk = outputs["action"]
```

### With Callbacks

```python
from physicalai import InferenceModel
from physicalai.callbacks import TimingCallback, LoggingCallback

model = InferenceModel(
    "./pi0_exported",
    callbacks=[
        TimingCallback(),
        LoggingCallback(log_inputs=False, log_outputs=True),
    ],
)

# Callbacks fire automatically on predict
action = model(observation)
# -> logs timing and output summary
```

### Override Runner Parameters

```python
# Override num_steps at load time (no re-export needed)
model = InferenceModel(
    "./pi0_exported",
    num_steps=20,  # Override manifest default of 10
    scheduler="ddim",
)
```

### Real-Time Control (Policy API)

```python
from physicalai import InferenceModel

policy = InferenceModel("./pi0_exported")
policy.reset()

while not done:
    action = policy.select_action(observation)
    observation, reward, done, info = env.step(action)
```

### Explicit Backend Selection

```python
# Use specific backend
model = InferenceModel(
    "./pi0_exported",
    backend="onnx",
    device="cuda:0",
)

# Or with adapter options
model = InferenceModel(
    "./pi0_exported",
    backend="onnx",
    adapter_options={
        "providers": ["TensorrtExecutionProvider", "CUDAExecutionProvider"],
    },
)
```

---

## 5. Extension Fields

physicalai-specific configuration can be embedded in the manifest under `x-physical-ai`. These fields are ignored by LeRobot's own runtime:

```json
{
  "format": "policy_package",
  "version": "1.0",

  "policy": { ... },
  "artifacts": { ... },

"x-physical-ai": {
    "callbacks": [
      "timing",
      {"class_path": "myproject.callbacks.SafetyCallback", "init_args": {"max_velocity": 1.0}}
    ],
    "adapter": {
      "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
      "graph_optimization_level": "all"
    },
    "preprocessors": [
{"class_path": "physicalai.preprocessors.ImageNormalize", "init_args": {"mean": [0.485, 0.456, 0.406]}}
    ]
  }
}
```

### Extension Schema

| Field            | Type                        | Description             |                     |
| ---------------- | --------------------------- | ----------------------- | ------------------- |
| `callbacks`      | `list[str \                 | CallbackConfig]`        | Callbacks to attach |
| `adapter`        | `dict`                      | Adapter/backend options |                     |
| `preprocessors`  | `list[PreprocessorConfig]`  | Input preprocessors     |                     |
| `postprocessors` | `list[PostprocessorConfig]` | Output postprocessors   |                     |

**Note**: LeRobot ignores `x-physical-ai` fields entirely. They are only read by physicalai.

---

## 6. Runner Mapping

### `policy.kind` → Built‑in Runner

| `policy.kind` | Runner             | Notes                            |
| ------------- | ------------------ | -------------------------------- |
| `single_pass` | `SinglePassRunner` | Direct forward pass              |
| `iterative`   | `IterativeRunner`  | Configurable loop with scheduler |
| `two_phase`   | `TwoPhaseRunner`   | Encode once + denoise loop       |
| `custom`      | via `class_path`   | User-provided runner class       |

### IterativeRunner Configuration

```python
class IterativeRunner(InferenceRunner):
    """Runner for iterative/flow-matching policies."""

    def __init__(
        self,
        num_steps: int = 10,
        scheduler: str = "euler",
        timestep_spacing: str = "linear",
        timestep_range: tuple[float, float] = (1.0, 0.0),
    ):
        self.num_steps = num_steps
        self.scheduler = scheduler
        self.timestep_spacing = timestep_spacing
        self.timestep_range = timestep_range

    def run(self, adapter: RuntimeAdapter, inputs: dict) -> dict:
        # Initialize from noise
        action_shape = self._infer_action_shape(inputs)
        x_t = np.random.randn(*action_shape).astype(np.float32)

        # Generate timesteps
        timesteps = self._generate_timesteps()
        dt = -1.0 / self.num_steps

        # Iterative denoising
        for t in timesteps:
            step_inputs = {
                **inputs,
                "x_t": x_t,
                "timestep": np.array([t], dtype=np.float32),
            }
            v_t = adapter.predict(step_inputs)["v_t"]
            x_t = self._step(x_t, v_t, dt)

        return {"action": x_t}

    def _step(self, x: np.ndarray, v: np.ndarray, dt: float) -> np.ndarray:
        if self.scheduler == "euler":
            return x + dt * v
        elif self.scheduler == "ddim":
            # DDIM update rule
            ...
        else:
            raise ValueError(f"Unknown scheduler: {self.scheduler}")
```

---

## 7. Callbacks for Robotics

physicalai provides callbacks useful for robotics applications:

### ActionSafetyCallback

```python
# physicalai/callbacks/safety.py

class ActionSafetyCallback(Callback):
    """Clamp actions to safe ranges."""

    def __init__(
        self,
        action_min: np.ndarray | float = -1.0,
        action_max: np.ndarray | float = 1.0,
        velocity_limit: float | None = None,
    ):
        self.action_min = action_min
        self.action_max = action_max
        self.velocity_limit = velocity_limit
        self._last_action = None

    def on_predict_end(self, outputs: dict) -> dict:
        action = outputs["action"]

        # Clamp to range
        action = np.clip(action, self.action_min, self.action_max)

        # Velocity limiting
        if self.velocity_limit and self._last_action is not None:
            delta = action - self._last_action
            delta = np.clip(delta, -self.velocity_limit, self.velocity_limit)
            action = self._last_action + delta

        self._last_action = action.copy()
        outputs["action"] = action
        return outputs

    def on_reset(self):
        self._last_action = None
```

### EpisodeLoggingCallback

```python
# physicalai/callbacks/logging.py

class EpisodeLoggingCallback(Callback):
    """Log episode data for replay/debugging."""

    def __init__(self, log_dir: Path, log_observations: bool = True):
        self.log_dir = Path(log_dir)
        self.log_observations = log_observations
        self._episode_data = []
        self._episode_count = 0

    def on_predict_end(self, outputs: dict, inputs: dict | None = None) -> dict:
        step_data = {"action": outputs["action"].tolist()}
        if self.log_observations and inputs:
            step_data["observation"] = {k: v.tolist() for k, v in inputs.items()}
        self._episode_data.append(step_data)
        return outputs

    def on_reset(self):
        if self._episode_data:
            self._save_episode()
        self._episode_data = []
        self._episode_count += 1

    def _save_episode(self):
        path = self.log_dir / f"episode_{self._episode_count:04d}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._episode_data))
```

---

## 8. Unified Manifest Format

All exported models — regardless of source framework — use the same `manifest.json` format.

### Why One Format

Previous designs had two formats: `metadata.yaml` for physicalai-train and `manifest.json` for LeRobot. This created unnecessary divergence:

- Two parsers to maintain
- Two sets of schema conventions
- Confusion about which format to use

The unified `manifest.json` format eliminates this. Benefits:

- **One parser** — simpler codebase, fewer bugs
- **One schema** — consistent across all model sources
- **No special-casing** — the loader doesn't need to know where a model came from
- **JSON for data, not code** — `policy.kind` maps to built‑in runners; `class_path` is only for exotic patterns

### Unified Loading

```python
# Works with LeRobot packages
model = InferenceModel("./lerobot_package")

# Works with physicalai-train packages
model = InferenceModel("./physicalai_train_package")

# Works with custom packages
model = InferenceModel("./custom_package")

# All read manifest.json — same code path
```

---

## 9. Testing Compatibility

### Conformance Test Suite

```python
# tests/format_loaders/test_lerobot_loader.py

class TestLeRobotFormatLoaderConformance:
"""Verify physicalai correctly loads LeRobot packages."""

    def test_detect_lerobot_package(self, lerobot_package_path):
        """Format loader detects LeRobot packages."""
        assert LeRobotFormatLoader.detect(lerobot_package_path)

    def test_load_single_shot(self, act_package_path):
        """Load single_shot policy."""
        model = InferenceModel(act_package_path)
        assert isinstance(model.runner, SinglePassRunner)

    def test_load_iterative(self, pi0_package_path):
        """Load iterative policy."""
        model = InferenceModel(pi0_package_path)
        assert isinstance(model.runner, IterativeRunner)
        assert model.runner.num_steps == 10  # from manifest

    def test_override_num_steps(self, pi0_package_path):
        """Override iterative params at load time."""
        model = InferenceModel(pi0_package_path, num_steps=20)
        assert model.runner.num_steps == 20

    def test_parity_with_lerobot_runtime(self, pi0_package_path):
        """Output matches LeRobot's own runtime."""
# Load with physicalai
ik_model = InferenceModel(pi0_package_path)

        # Load with LeRobot
        from lerobot.export import load as lerobot_load
        lr_runtime = lerobot_load(pi0_package_path)

        # Compare outputs
        obs = generate_test_observation()
        np.random.seed(42)
        ik_output = ik_model(obs)
        np.random.seed(42)
        lr_output = lr_runtime.predict_action_chunk(obs)

        np.testing.assert_allclose(ik_output["action"], lr_output, rtol=1e-5)
```

---

## 10. Summary

### What physicalai Adds Over LeRobot Runtime

| Feature                             | LeRobot Runtime | physicalai              |
| ----------------------------------- | --------------- | ----------------------- |
| Load PolicyPackage                  | ✓               | ✓                       |
| Single-pass inference               | ✓               | ✓                       |
| Iterative inference                 | ✓               | ✓                       |
| Two-phase inference                 | ✓               | ✓                       |
| Action queue wrapper                | ✓               | ✓                       |
| Callbacks (timing, logging, safety) | ✗               | ✓                       |
| Multi-backend with fallback         | ✗               | ✓                       |
| Preprocessor/postprocessor chains   | ✗               | ✓                       |
| Unified manifest format             | ✗               | ✓ (same format for all) |

### Dependency Direction

```text
LeRobot ──────────────────────────────────────────────────┐
    │                                                      │
    │ defines (proposed)                                   │
    ▼                                                      │
manifest.json (unified format)                             │
    │                                                      │
    │ consumed by                                          │
    ▼                                                      │
physicalai (unified manifest loader) ◄─────────┘
                               no dependency on LeRobot code
```

**LeRobot does not depend on physicalai.**
**physicalai can load LeRobot packages without importing LeRobot.**
**physicalai-train exports the same manifest.json format — no special handling needed.**

> **Reminder:** This integration depends on LeRobot adopting the proposed PolicyPackage export format. If LeRobot adopts a different format, the manifest loader implementation changes but the architecture (unified loader, no runtime dependency) remains the same.

---

## Related Documents

- **[Strategy](../../architecture/strategy.md)** - Big-picture architecture
- **[Inference Core Design](./inferencekit.md)** - Domain-agnostic inference layer
- **[LeRobot Export Suggestions](../internal/lerobot-export-suggestions.md)** - Our proposed improvements to LeRobot's export API

---

_Document version: 3.0_
_Last updated: 2026-02-16_
