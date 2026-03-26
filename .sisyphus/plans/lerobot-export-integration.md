# LeRobot Export Integration Plan

> **Status**: Draft v2 — rewritten to enforce one-way dependency
> **Date**: 2026-03-26
> **Scope**: Standalone LeRobot export + physicalai adapter-based consumption
> **Architecture**: LeRobot is fully standalone. PhysicalAI adapts to LeRobot's format at load time.

---

## 1. Context & Goal

### Architectural Constraint (Non-Negotiable)

**LeRobot MUST NOT depend on physicalai.** Not as a runtime dep, not as a type import, not even as a string reference in manifests. LeRobot's export system must work perfectly for users who have never heard of physicalai.

**PhysicalAI SHOULD support LeRobot exports out of the box.** `InferenceModel.load("./lerobot_export")` must just work — detect the format, adapt it internally, and run inference through our pipeline.

The dependency is strictly one-way:

```text
LeRobot (standalone)                    PhysicalAI
────────────────────                    ──────────────────────────
policy.export("./out") ──produces──►    InferenceModel.load("./out")
                                            │
  lerobot_exported_policy                   ├─ detects "policy_package" → use directly
  own manifest format                       └─ detects "lerobot_exported_policy"
  own runners                                     → LeRobotManifestAdapter
  own normalizer                                       translates → internal Manifest
  zero physicalai deps                                 then normal pipeline
```

### What we want

1. **LeRobot policies become exportable** — `policy.export("./out", backend="onnx")` produces a self-describing standalone package. No physicalai references.
2. **PhysicalAI consumes those packages** — `InferenceModel.load("./out")` works out of the box via an adapter layer.
3. **Two manifest formats, one adapter** — LeRobot uses `lerobot_exported_policy`, physicalai uses `policy_package`. PhysicalAI's `LeRobotManifestAdapter` bridges the gap.

### Current state

| Component       | physicalai-studio                      | LeRobot (upstream)                                      | Your fork PR #2                                         |
| --------------- | -------------------------------------- | ------------------------------------------------------- | ------------------------------------------------------- |
| Export API      | `ExportablePolicyMixin.to_onnx()` etc. | None                                                    | `policy.export(path, backend)`                          |
| Manifest format | `policy_package` v1.0                  | None                                                    | `lerobot_exported_policy` v1.0                          |
| Runners         | `SinglePass`, `ActionChunking`         | N/A                                                     | `SinglePassRunner`, `IterativeRunner`, `TwoPhaseRunner` |
| Adapters        | ONNX, OpenVINO, Torch, TorchExportIR  | N/A                                                     | `ONNXRuntimeAdapter`, `OpenVINORuntimeAdapter`          |
| Normalization   | `ActionNormalizer` postprocessor       | `NormalizerProcessorStep` / `UnnormalizerProcessorStep` | Standalone `Normalizer` class with stats.safetensors    |

### The gap

LeRobot PR #2 uses `lerobot_exported_policy` format with its own schema. PhysicalAI expects `policy_package` with `ComponentSpec`-based runner/preprocessor definitions. PhysicalAI needs an **adapter** to translate between the two at load time.

---

## 2. LeRobot's Standalone Format (Owned by LeRobot)

### LeRobot manifest: `lerobot_exported_policy`

This is LeRobot's own format. PhysicalAI does NOT control this schema — we adapt to it.

```jsonc
{
  "format": "lerobot_exported_policy",
  "version": "1.0",
  "policy": {
    "name": "act",
    "kind": "single_pass",        // "single_pass" | "iterative" | "two_phase"
    "class_name": "ACT",
    "repo_id": "lerobot/act_aloha_sim_transfer_cube_human"
  },
  "artifacts": {
    "onnx": "artifacts/model.onnx",
    "openvino": "artifacts/model.xml"
  },
  "io": {
    "inputs": [
      {"name": "observation.state", "dtype": "float32", "shape": [-1, 14]},
      {"name": "observation.images.top", "dtype": "float32", "shape": [-1, 3, 480, 640]}
    ],
    "outputs": [
      {"name": "action", "dtype": "float32", "shape": [-1, 100, 14]}
    ]
  },
  "action": {
    "dim": 14,
    "chunk_size": 100,
    "n_action_steps": 100
  },
  "normalization": {
    "type": "min_max",
    "artifact": "artifacts/stats.safetensors",
    "input_features": ["observation.state"],
    "output_features": ["action"]
  },
  "inference": null   // null | IterativeConfig | TwoPhaseConfig
}
```

### LeRobot's own runtime (standalone)

LeRobot PR #2 ships its own runners for users who don't use physicalai:
- `SinglePassRunner` — direct forward pass
- `IterativeRunner` — multi-step denoising loop with built-in scheduler
- `TwoPhaseRunner` — encode + iterative denoise

These are LeRobot's responsibility. They work without physicalai installed.

---

## 3. PhysicalAI's Format (Owned by PhysicalAI)

### PhysicalAI manifest: `policy_package` v1.1

Our own format, extended with optional fields for richer metadata when available.

```jsonc
{
  "format": "policy_package",
  "version": "1.1",
  "policy": {
    "name": "my_policy",
    "kind": "single_pass",
    "source": {                  // NEW v1.1: provenance info (optional)
      "framework": "physicalai-train",
      "version": "1.0.0"
    }
  },
  "artifacts": { "onnx": "artifacts/model.onnx" },
  "runner": { "class_path": "physicalai.inference.runners.SinglePass", "init_args": {} },
  "robots": [],
  "cameras": [],
  "preprocessors": [],
  "postprocessors": [],

  // --- Optional v1.1 fields ---
  "io": { ... },               // Same structure as LeRobot's — universal I/O contract
  "action": { ... },           // Action semantics
  "normalization": { ... },    // Stats-based normalization config
  "inference": null             // null | IterativeInferenceConfig | TwoPhaseInferenceConfig
}
```

### Backward compatibility

- v1.0 manifests (no `io`, `action`, `normalization`, `inference`) continue to work — all new fields are optional with `None` defaults.
- Legacy `metadata.yaml` fallback path is unchanged.
- `InferenceModel._load_metadata()` reads `version` to determine available fields.

---

## 4. The Adapter: `LeRobotManifestAdapter`

### Purpose

Translates `lerobot_exported_policy` manifests into our internal `Manifest` Pydantic model so the rest of the pipeline (runner selection, normalizer creation, adapter loading) works unchanged.

### Location

`physicalai.inference.adapters.lerobot_manifest` (or `physicalai.inference.manifest_adapters.lerobot`)

### Design

```python
class LeRobotManifestAdapter:
    """Translate LeRobot's lerobot_exported_policy format to internal Manifest."""

    @staticmethod
    def can_handle(raw: dict) -> bool:
        """Check if this raw JSON is a LeRobot manifest."""
        return raw.get("format") == "lerobot_exported_policy"

    @staticmethod
    def adapt(raw: dict, package_path: Path) -> Manifest:
        """Convert LeRobot manifest dict to our internal Manifest model.

        Mapping:
        - policy.kind → runner selection (single_pass/iterative/two_phase)
        - policy.repo_id → policy.source.repo_id
        - artifacts → same key structure (onnx, openvino)
        - io → IOSpec
        - action → ActionSpec
        - normalization → NormalizationSpec + auto-create preprocessor/postprocessor ComponentSpecs
        - inference → IterativeInferenceConfig / TwoPhaseInferenceConfig
        """
        ...
```

### Integration point

In `InferenceModel.load()` (or the manifest loading path):

```python
raw = json.loads((path / "manifest.json").read_text())

if raw.get("format") == "policy_package":
    manifest = Manifest.model_validate(raw)
elif LeRobotManifestAdapter.can_handle(raw):
    manifest = LeRobotManifestAdapter.adapt(raw, path)
else:
    msg = f"Unknown manifest format: {raw.get('format')}"
    raise ValueError(msg)
```

### What the adapter translates

| LeRobot field | → PhysicalAI internal field | Notes |
|---|---|---|
| `policy.kind` | Runner type selection | `single_pass` → `SinglePass`, etc. |
| `policy.repo_id` | `policy.source.repo_id` | Provenance |
| `policy.class_name` | `policy.source.class_name` | Original class info |
| `artifacts` | `artifacts` | Direct mapping (same key convention) |
| `io` | `IOSpec` | Direct mapping |
| `action` | `ActionSpec` | Direct mapping |
| `action.chunk_size` | `ActionChunking` runner wrapper | If chunk_size > 1, wrap with ActionChunking |
| `normalization` | `preprocessors` + `postprocessors` | Auto-generates `StatsNormalizer` + `StatsDenormalizer` ComponentSpecs |
| `inference` (null) | No extra config | SinglePass runner |
| `inference` (iterative) | `IterativeInferenceConfig` | num_steps, scheduler, timestep config |
| `inference` (two_phase) | `TwoPhaseInferenceConfig` | encoder/denoise artifact keys, num_steps |

---

## 5. Runner Design

### PhysicalAI runners (our codebase)

| Runner | Location | Status |
|---|---|---|
| `SinglePass` | `physicalai.inference.runners` | ✅ Done |
| `ActionChunking` | `physicalai.inference.runners` | ✅ Done (decorator) |
| `IterativeRunner` | `physicalai.inference.runners` | ❌ Phase 5B — NEW |
| `TwoPhaseRunner` | `physicalai.inference.runners` | ❌ Phase 6 — NEW |

### LeRobot runners (their codebase, standalone)

LeRobot PR #2 ships its own runners. These are LeRobot's responsibility and have no dependency on physicalai. They exist so LeRobot users can run exported models without installing physicalai.

**We do NOT reference LeRobot's runner class paths in our manifests, and they do NOT reference ours.**

### IterativeRunner design (Phase 5B)

Lives in `physicalai.inference.runners.iterative`.

```python
class IterativeRunner(InferenceRunner):
    """Multi-step denoising / flow-matching runner.

    Calls adapter.predict() N times with different timestep inputs,
    accumulating the result through a noise scheduler.
    """

    def __init__(
        self,
        runner: InferenceRunner,          # inner runner (SinglePass)
        num_steps: int = 10,
        scheduler: str = "ddim",          # "ddpm" | "ddim" | "euler" | "flow_matching"
        timestep_spacing: str = "leading",
        timestep_range: tuple[int, int] = (999, 0),
        noise_key: str = "x_t",
        timestep_key: str = "timestep",
        output_key: str = "action",
    ) -> None: ...

    def run(self, adapter, inputs):
        x_t = self._init_noise(inputs)
        timesteps = self._generate_timesteps()

        for t in timesteps:
            step_inputs = {**inputs, self.noise_key: x_t, self.timestep_key: np.array([t])}
            prediction = self.runner.run(adapter, step_inputs)
            x_t = self._scheduler_step(prediction, x_t, t)

        return {self.output_key: x_t}

    def reset(self) -> None:
        self.runner.reset()
```

### TwoPhaseRunner design (Phase 6)

Lives in `physicalai.inference.runners.two_phase`.

```python
class TwoPhaseRunner(InferenceRunner):
    """Two-phase runner: encode once, then denoise iteratively.

    Phase 1 (encode): Runs encoder adapter once to produce embeddings/KV cache.
    Phase 2 (denoise): Runs denoise adapter N times via IterativeRunner.
    """

    def __init__(
        self,
        encoder_adapter: RuntimeAdapter,
        denoise_runner: IterativeRunner,
    ) -> None: ...

    def run(self, adapter, inputs):
        # Phase 1: Encode (uses self.encoder_adapter, not the passed adapter)
        embeddings = self.encoder_adapter.predict(inputs)

        # Phase 2: Denoise with cached embeddings
        denoise_inputs = {**inputs, **embeddings}
        return self.denoise_runner.run(adapter, denoise_inputs)

    def reset(self) -> None:
        self.denoise_runner.reset()
```

**Multi-adapter pattern**: TwoPhaseRunner owns its encoder adapter directly. The denoise adapter is passed via `run(adapter, ...)` as normal. This keeps `InferenceModel` simple — the factory creates both adapters and wires them into the runner.

---

## 6. Normalization Handling

### Problem

LeRobot policies operate on **normalized** inputs and produce **normalized** outputs. Stats are saved in `stats.safetensors`. When loading a LeRobot export, PhysicalAI needs to:
1. **Pre-normalize** observations before feeding to the model.
2. **Post-denormalize** actions coming out of the model.

### Design

Two new pipeline components:

**`StatsNormalizer` preprocessor** (`physicalai.inference.preprocessors.stats_normalizer`):
```python
class StatsNormalizer(Preprocessor):
    def __init__(
        self,
        stats_path: str,
        input_features: list[str],
        normalization_type: str = "min_max",
    ) -> None: ...

    def __call__(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        for feature in self.input_features:
            if feature in inputs:
                inputs[feature] = self._normalize(inputs[feature], feature)
        return inputs
```

**`StatsDenormalizer` postprocessor** (`physicalai.inference.postprocessors.stats_denormalizer`):
```python
class StatsDenormalizer(Postprocessor):
    def __init__(
        self,
        stats_path: str,
        output_features: list[str],
        normalization_type: str = "min_max",
    ) -> None: ...

    def __call__(self, outputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        for feature in self.output_features:
            if feature in outputs:
                outputs[feature] = self._denormalize(outputs[feature], feature)
        return outputs
```

### How the adapter wires normalization

The `LeRobotManifestAdapter` reads the `normalization` field from the LeRobot manifest and auto-generates `ComponentSpec` entries for the preprocessors/postprocessors arrays:

```python
# Inside LeRobotManifestAdapter.adapt():
if raw.get("normalization"):
    norm = raw["normalization"]
    preprocessors.append(ComponentSpec(
        class_path="physicalai.inference.preprocessors.StatsNormalizer",
        init_args={
            "stats_path": str(package_path / norm["artifact"]),
            "input_features": norm["input_features"],
            "normalization_type": norm["type"],
        },
    ))
    postprocessors.append(ComponentSpec(
        class_path="physicalai.inference.postprocessors.StatsDenormalizer",
        init_args={
            "stats_path": str(package_path / norm["artifact"]),
            "output_features": norm["output_features"],
            "normalization_type": norm["type"],
        },
    ))
```

---

## 7. Execution Plan — Phased Delivery

### Phase 5A: Manifest v1.1 + LeRobotManifestAdapter + Normalization (physicalai-studio)

**Branch**: `feature/manifest-v1.1`

**Commit 1**: Add v1.1 Pydantic models + backward-compatible manifest loading

| Step | File | Change |
|------|------|--------|
| 1 | `inference/manifest.py` | Add `IOSpec`, `ActionSpec`, `NormalizationSpec`, `IterativeInferenceConfig`, `TwoPhaseInferenceConfig`, `PolicySource` Pydantic models. Add as optional fields on `Manifest` (default `None`). |
| 2 | `inference/manifest.py` | Ensure v1.0 manifests still parse (all new fields optional). Accept `MANIFEST_VERSION` of both `"1.0"` and `"1.1"`. |
| 3 | `tests/unit/inference/test_manifest.py` | Add tests: parse v1.1 manifest with all new fields populated; parse v1.0 manifest and verify new fields are `None`; roundtrip save/load. |

> **QA**: `cd library && .venv/bin/python -m pytest tests/unit/inference/test_manifest.py -v` — all pass. `lsp_diagnostics` clean on `manifest.py`.

**Commit 2**: Implement `LeRobotManifestAdapter`

| Step | File | Change |
|------|------|--------|
| 4 | `inference/manifest_adapters/__init__.py` | Create module. |
| 5 | `inference/manifest_adapters/lerobot.py` | Implement `LeRobotManifestAdapter` with `can_handle(raw)` and `adapt(raw, package_path) -> Manifest`. Maps LeRobot fields to internal Manifest. Auto-generates normalizer ComponentSpecs from `normalization` field. Maps `policy.kind` + `inference` to correct runner config. |
| 6 | `inference/model.py` | Update manifest loading path: detect format, route to adapter if `lerobot_exported_policy`, else parse as `policy_package`. |
| 7 | `tests/unit/inference/test_lerobot_manifest_adapter.py` | Test: single_pass LeRobot manifest → correct Manifest with SinglePass runner + ActionChunking wrapper. Test: iterative manifest → IterativeRunner config. Test: two_phase manifest → TwoPhaseRunner config. Test: normalization field → auto-generated pre/postprocessor ComponentSpecs. Test: unknown format → ValueError. |

> **QA**: `cd library && .venv/bin/python -m pytest tests/unit/inference/test_lerobot_manifest_adapter.py -v` — all pass. `lsp_diagnostics` clean.

**Commit 3**: Implement StatsNormalizer preprocessor

| Step | File | Change |
|------|------|--------|
| 8 | `inference/preprocessors/stats_normalizer.py` | Implement `StatsNormalizer(Preprocessor)`. Loads stats from safetensors (or `.npz` fallback). `__call__` normalizes specified features in-place. |
| 9 | `tests/unit/inference/test_stats_normalizer.py` | Test min_max normalization (`(x - min) / (max - min) * 2 - 1`), mean_std (`(x - mean) / std`), identity passthrough, missing feature is no-op, stats loading from fixture. |

> **QA**: `cd library && .venv/bin/python -m pytest tests/unit/inference/test_stats_normalizer.py -v` — all pass.

**Commit 4**: Implement StatsDenormalizer postprocessor

| Step | File | Change |
|------|------|--------|
| 10 | `inference/postprocessors/stats_denormalizer.py` | Implement `StatsDenormalizer(Postprocessor)`. Inverse of StatsNormalizer. |
| 11 | `tests/unit/inference/test_stats_denormalizer.py` | Test inverse min_max, inverse mean_std, identity, roundtrip (normalize → denormalize recovers original within float tolerance). |

> **QA**: `cd library && .venv/bin/python -m pytest tests/unit/inference/test_stats_denormalizer.py -v` — all pass.

**Commit 5**: Register components + wire into export

| Step | File | Change |
|------|------|--------|
| 12 | `inference/component_factory.py` | Register `"stats_normalizer"` and `"stats_denormalizer"` short names. |
| 13 | `export/mixin_policy.py` | Update `_build_manifest()` to populate `io`, `action`, `normalization` fields when available. |
| 14 | `tests/unit/inference/test_component_factory.py` | Test that short names resolve correctly. |

> **QA**: `cd library && .venv/bin/prek run -a` — all 8 checks pass.

### Phase 5B: IterativeRunner (physicalai-studio)

**Branch**: `feature/iterative-runner` (based on `feature/manifest-v1.1`)

**Commit 1**: Pure-numpy scheduler implementations

| Step | File | Change |
|------|------|--------|
| 1 | `inference/runners/schedulers.py` | Implement `DDIMScheduler`, `DDPMScheduler`, `EulerScheduler`, `FlowMatchingScheduler`. Each has `set_timesteps(num_steps)`, `step(model_output, timestep, sample) -> sample`, `get_timesteps() -> np.ndarray`. All pure numpy. |
| 2 | `tests/unit/inference/test_schedulers.py` | Test each scheduler: timestep generation, single step shape, DDIM 1-step approximation, flow matching Euler step `x_t + dt * v_t`. |

> **QA**: `cd library && .venv/bin/python -m pytest tests/unit/inference/test_schedulers.py -v` — all pass. No torch imports in `schedulers.py`.

**Commit 2**: IterativeRunner implementation

| Step | File | Change |
|------|------|--------|
| 3 | `inference/runners/iterative.py` | Implement `IterativeRunner(InferenceRunner)`. Decorator pattern wrapping inner runner. |
| 4 | `inference/runners/__init__.py` | Export `IterativeRunner`. |
| 5 | `inference/component_factory.py` | Register `"iterative"` short name. |
| 6 | `inference/runners/factory.py` | Handle `inference.type == "iterative"` → construct `IterativeRunner`. |
| 7 | `tests/unit/inference/test_iterative_runner.py` | Test with mock adapter: predict called `num_steps` times, output shape correct, reset clears inner runner, factory construction from manifest. |

> **QA**: `cd library && .venv/bin/prek run -a` — all 8 checks pass.

### Phase 6: TwoPhaseRunner + Multi-Adapter (physicalai-studio)

**Branch**: `feature/two-phase-runner` (based on `feature/iterative-runner`)

**Commit 1**: TwoPhaseRunner implementation

| Step | File | Change |
|------|------|--------|
| 1 | `inference/runners/two_phase.py` | Implement `TwoPhaseRunner(InferenceRunner)`. Owns encoder adapter. Delegates denoise to IterativeRunner. |
| 2 | `tests/unit/inference/test_two_phase_runner.py` | Test: encoder called once, denoise called `num_steps` times, output shape correct, reset delegates. |

> **QA**: `cd library && .venv/bin/python -m pytest tests/unit/inference/test_two_phase_runner.py -v` — all pass.

**Commit 2**: Multi-adapter loading in InferenceModel

| Step | File | Change |
|------|------|--------|
| 3 | `inference/model.py` | When manifest has `inference.type == "two_phase"`, load encoder + denoise adapters, wire into TwoPhaseRunner. |
| 4 | `inference/runners/__init__.py` | Export `TwoPhaseRunner`. |
| 5 | `inference/component_factory.py` | Register `"two_phase"` short name. |
| 6 | `inference/runners/factory.py` | Handle `inference.type == "two_phase"` → construct `TwoPhaseRunner` with both adapters. |
| 7 | `tests/unit/inference/test_model.py` | Test `InferenceModel.load()` with two_phase manifest fixture: both adapters loaded, runner correct, `select_action()` returns correct shape. |

> **QA**: `cd library && .venv/bin/prek run -a` — all 8 checks pass.

### Phase L1: LeRobot Export PR Update (samet-akcay/lerobot)

**Your existing PR #2, updated to be fully standalone:**

**Key principle**: Remove ANY reference to physicalai from the LeRobot PR. The export system must work for all LeRobot users regardless of their inference runtime.

| Step | Change |
|------|--------|
| 1 | Ensure manifest uses `"format": "lerobot_exported_policy"` (NOT `policy_package`). |
| 2 | Ensure runner class paths point to `lerobot.export.runners.*` (NOT `physicalai.*`). |
| 3 | Keep LeRobot's standalone runners (`SinglePassRunner`, `IterativeRunner`, `TwoPhaseRunner`) as the default runtime. |
| 4 | Ensure the export produces a complete standalone package loadable by LeRobot's own runtime. |
| 5 | Document: "Compatible with physicalai InferenceModel for production deployment." |
| 6 | Update tests to verify standalone loading without physicalai. |

> **QA**: `cd lerobot && python -m pytest tests/export/ -v` — all pass. Verify no `physicalai` string appears anywhere in the lerobot codebase.

### Phase L2: End-to-End Integration Test

| Step | Description | QA |
|------|-------------|-----|
| 1 | Train ACT policy with LeRobot (smallest dataset). | Training completes. |
| 2 | Export with `policy.export("./out", backend="onnx")`. | `manifest.json` has `format == "lerobot_exported_policy"`. |
| 3 | Load with `InferenceModel.load("./out")`. | No errors. `LeRobotManifestAdapter` is invoked. Runner is `ActionChunking(SinglePass())`. |
| 4 | Run `select_action(obs)` with dummy observation. | Returns numpy array with correct shape. Values are finite. |
| 5 | Repeat for Diffusion (IterativeRunner). | Adapter translates `inference.type == "iterative"`. Runner is correct. |
| 6 | (Optional) Repeat for Pi0 (TwoPhaseRunner). | Adapter handles two_phase. |
| 7 | **Parity test**: Compare physicalai output vs LeRobot's own runtime. | `np.allclose(ik_output, lr_output, rtol=1e-5)` — outputs match. |

---

## 8. Dependency Graph

```
Phase 5A (manifest v1.1 + LeRobotManifestAdapter + normalization)
    ├── Phase 5B (IterativeRunner) — depends on 5A for manifest
    │     └── Phase 6 (TwoPhaseRunner) — depends on 5B for IterativeRunner
    │
    └── Phase L1 (LeRobot PR update) — independent, but informs adapter mapping
          └── Phase L2 (E2E test) — depends on 5A + L1 minimum
                                    depends on 5B for iterative E2E
                                    depends on 6 for two_phase E2E
```

**Critical path**: 5A → 5B → L2 (for iterative E2E)

Phase L1 is independent and can proceed in parallel — it's changes to LeRobot, not physicalai.

Phase 6 (TwoPhaseRunner) is not on critical path for ACT and Diffusion.

---

## 9. What Changes in lerobot.md Design Doc

The design doc at `physicalai/docs/design/integrations/lerobot.md` should be updated to reflect:

| Section | Change |
|---------|--------|
| Architecture diagram | Add `LeRobotManifestAdapter` as explicit component between LeRobot manifests and unified loader |
| Manifest format | Document that LeRobot uses `lerobot_exported_policy` format, NOT `policy_package` |
| Manifest Loader | Add format detection + adapter routing logic |
| Section 9 (Testing) | Update conformance tests to go through `LeRobotManifestAdapter` |
| Runner mapping | Clarify: our runners are independent implementations, not shared with LeRobot |
| Normalization | Add: adapter auto-generates StatsNormalizer/StatsDenormalizer from LeRobot's normalization field |
| New section | Add: "Format Adapter Registry" — extensible pattern for supporting additional export formats in the future |

---

## 10. Open Questions

1. **LeRobot PR acceptance path** — Strategy for getting export merged upstream? Do we need HuggingFace buy-in first, or ship as fork initially?

2. **Scheduler implementations** — All schedulers (DDPM, DDIM, Euler, FlowMatching) in Phase 5B, or start with DDIM only?

3. **Stats format** — Commit to `safetensors` for normalization stats, or also support `.npz` / `.json`?

4. **Image preprocessing** — LeRobot's processor pipeline handles image resizing/normalization. Create image-specific preprocessors, or expect pre-processed images?

5. **Phase 4 PR #384** — Merge callback system before starting Phase 5A, or develop in parallel?

6. **Format adapter registry** — Should the adapter pattern be extensible (registry-based) to support future formats from other frameworks, or is a simple if/elif sufficient for now?

7. **LeRobot manifest versioning** — If LeRobot changes their manifest format in future versions, how do we handle multiple LeRobot manifest versions? Version-specific adapters?
