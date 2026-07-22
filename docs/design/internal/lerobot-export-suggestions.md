# LeRobot Export API - Suggestions and Improvements

Internal document capturing suggestions for improving the LeRobot export API.

**Context:** This document contains notes on the LeRobot PolicyPackage export implementation (currently in a fork, pending upstream proposal). These suggestions aim to improve clarity, extensibility, and compatibility.

**Status:** Draft - for internal reference

---

## Convergence Status

As part of the unified design across getiaction, LeRobot, and physical‑ai‑framework, the following suggestions have been adopted or deferred:

| Suggestion                             | Status       | Notes                                                                                                                                       |
| -------------------------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Add explicit `kind` field              | **Adopted**  | `policy.kind` is now a required field in the unified `manifest.json` format. Used by the framework to select built‑in runners.              |
| Clarify action queuing                 | **Adopted**  | Action handling is explicit in the manifest via `action.chunk_size` and `action.n_action_steps`. ActionChunkingRunner is a built‑in runner. |
| Document single-pass + action chunking | **Adopted**  | The runner mapping table in the design docs clarifies that `single_pass` may still use ActionChunkingRunner.                                |
| Normalization timing                   | **Adopted**  | Normalization is handled via built‑in Preprocessor (normalize inputs) and Postprocessor (denormalize outputs).                              |
| TwoPhase naming                        | **Deferred** | Keeping `two_phase` as a distinct `policy.kind` value. May revisit naming later.                                                            |
| Version strategy                       | **Adopted**  | `version: "1.0"` with `1.x` backward-compatible additions, `2.x` breaking changes.                                                          |

---

## Current Implementation Summary

The LeRobot export implementation uses:

1. **Protocol-based export** - Policies implement `ExportableSinglePhase`, `ExportableIterative`, or `ExportableTwoPhase`
2. **Manifest as pure data** - JSON manifest with no code references
3. **Structural detection** - Inference type determined by manifest structure, not explicit field

### Inference Pattern Detection (Current)

```python
def inference_config_from_dict(data: dict[str, Any]) -> InferenceConfig:
    if "encoder_artifact" in data:
        return TwoPhaseConfig.from_dict(data)
    elif "scheduler" in data:
        return IterativeConfig.from_dict(data)
```

### Manifest Structure (Current)

```json
{
  "format": "policy_package",
  "version": "1.0",
  "policy": {
    "name": "act_policy",
    "source": { "repo_id": "...", "revision": "..." }
  },
  "artifacts": {
    "onnx": "model.onnx",
    "openvino": "model.xml"
  },
  "io": {
    "inputs": [...],
    "outputs": [...]
  },
  "action": {
    "dim": 6,
    "chunk_size": 100,
    "n_action_steps": 100
  },
  "inference": null | IterativeConfig | TwoPhaseConfig,
  "normalization": { ... }
}
```

---

## Suggestions for Improvement

### 1. Add Explicit `kind` Field to Inference Config

**Problem:** Structural detection is implicit and fragile. What if a future config has both `encoder_artifact` AND `scheduler`?

**Suggestion:** Add explicit `kind` field with structural detection as fallback:

```python
def inference_config_from_dict(data: dict[str, Any]) -> InferenceConfig:
    # Explicit kind (preferred, clear intent)
    kind = data.get("kind")
    if kind == "two_phase":
        return TwoPhaseConfig.from_dict(data)
    elif kind == "iterative":
        return IterativeConfig.from_dict(data)
    elif kind == "single_pass":
        return None  # Or a SinglePassConfig if needed

    # Structural fallback (backward compatibility)
    if "encoder_artifact" in data:
        return TwoPhaseConfig.from_dict(data)
    elif "scheduler" in data:
        return IterativeConfig.from_dict(data)

    raise ValueError("Cannot determine inference config type")
```

**Benefits:**

- Explicit intent is clearer for humans and tooling
- Backward compatible with structural detection
- Easier to extend with new inference types
- Less error-prone than implicit detection

**Manifest with explicit kind:**

```json
{
  "inference": {
    "kind": "iterative",
    "num_steps": 100,
    "scheduler": "ddpm"
  }
}
```

---

### 2. Clarify Action Queuing in Manifest

**Problem:** Action queuing is implicit. The runtime must infer from `chunk_size` vs `n_action_steps`:

- If `n_action_steps < chunk_size` → queue actions
- If `n_action_steps == chunk_size` → no queue? or still queue?

**Suggestion:** Make action handling explicit in `ActionSpec`:

```python
@dataclass
class ActionSpec:
    dim: int
    chunk_size: int
    n_action_steps: int
    representation: str = "absolute"

    # NEW: Explicit action handling
    queue_strategy: str = "fifo"  # "fifo", "none", "temporal_ensemble"
```

**Alternative:** Document the implicit behavior clearly:

- `n_action_steps < chunk_size` → FIFO queue, return one action at a time
- `n_action_steps == chunk_size` → Return full chunk (no queuing)
- `n_action_steps == 1` → Single action, no queue

---

### 3. Document Single-Pass + Action Chunking Relationship

**Problem:** "Single-pass" (`inference: null`) doesn't mean "no action queue". ACT is single-pass but needs action chunking.

**Current terminology confusion:**

- `is_single_pass` → means "no iterative denoising"
- But ACT still needs `ActionChunkingRunner` for action queue management

**Suggestion:** Clarify in documentation and/or rename:

| Inference Pattern               | Model Behavior             | Action Handling             |
| ------------------------------- | -------------------------- | --------------------------- |
| `single_pass` (inference: null) | One forward pass           | May still need action queue |
| `iterative`                     | Multiple denoising steps   | Outputs full chunk          |
| `two_phase`                     | Encode once + denoise loop | Outputs full chunk          |

**Note:** The inference pattern (how model runs) is separate from action handling (how outputs are used).

---

### 4. Consider Normalization Timing

**Current:** `NormalizationConfig` specifies stats location and features, but doesn't explicitly state WHEN normalization happens.

**Suggestion:** Add clarity:

```python
@dataclass
class NormalizationConfig:
    type: NormalizationType
    artifact: str
    input_features: list[str]   # Normalized BEFORE inference
    output_features: list[str]  # Denormalized AFTER inference

    # NEW: Make timing explicit
    normalize_inputs: bool = True
    denormalize_outputs: bool = True
```

Or document clearly:

- `input_features` are normalized before model inference
- `output_features` (actions) are denormalized after model inference

---

### 5. TwoPhase Naming Consideration

**Observation:** `TwoPhaseConfig` is architecture-specific (VLA with encoder + denoise). It's essentially "iterative with encoder caching".

**Question:** Should this be a variant of `IterativeConfig` instead?

```python
@dataclass
class IterativeConfig:
    num_steps: int = 10
    scheduler: str = "euler"

    # Optional encoder caching (for VLA models)
    encoder_caching: bool = False
    encoder_artifact: str | None = None
    denoise_artifact: str | None = None
```

**Counter-argument:** Keeping them separate is clearer - TwoPhase has different I/O patterns (KV cache) that justify a distinct config.

**Recommendation:** Keep separate, but consider if naming could be improved:

- `TwoPhaseConfig` → `VLAConfig` or `CachedIterativeConfig`?

---

### 6. Version Strategy

**Current:** `version: "1.0"` with validation `version.startswith("1.")`

**Suggestion:** Document versioning strategy:

- `1.x` → backward compatible additions
- `2.x` → breaking changes

And consider adding:

```python
@dataclass
class Manifest:
    format: str = "policy_package"
    version: str = "1.0"
    min_runtime_version: str | None = None  # NEW: minimum runtime version required
```

---

## Compatibility Matrix

For runtimes consuming PolicyPackage:

| Field           | Required | Notes                           |
| --------------- | -------- | ------------------------------- |
| `format`        | Yes      | Must be "policy_package"        |
| `version`       | Yes      | Currently "1.0"                 |
| `policy.name`   | Yes      | Policy identifier               |
| `artifacts`     | Yes      | At least one backend            |
| `io`            | Yes      | Input/output tensor specs       |
| `action`        | Yes      | Action dimensions and chunking  |
| `inference`     | No       | null = single-pass              |
| `normalization` | No       | Stats for normalize/denormalize |

---

## Summary of Recommendations

| Priority   | Suggestion                       | Effort | Impact                 |
| ---------- | -------------------------------- | ------ | ---------------------- |
| **High**   | Add explicit `kind` field        | Low    | Clarity, extensibility |
| **High**   | Document action queuing behavior | Low    | Reduces confusion      |
| **Medium** | Clarify normalization timing     | Low    | Correctness            |
| **Low**    | Consider TwoPhase naming         | Low    | Clarity                |
| **Low**    | Add min_runtime_version          | Low    | Forward compatibility  |

---

## References

- LeRobot export implementation: `/Users/sakcay/Projects/lerobot/src/lerobot/export/`
- Key files: `manifest.py`, `protocols.py`, `runtime.py`

---

_Last Updated: 2026-02-16_
