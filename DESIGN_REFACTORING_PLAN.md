# Design Refactoring Plan

**Date**: November 12, 2025
**Branch**: `feat/add-lerobot-export-inference`
**Status**: 🚧 In Progress

## Overview

This document outlines identified design issues in the current LeRobot integration and export/inference pipeline, along with recommended fixes.

---

## 🔥 High Priority Issues (Fix First)

### 1. InferenceModel `_prepare_inputs` Complexity

**Problem**:
- Too much policy-specific logic in `InferenceModel._prepare_inputs()`
- Manual handling of LeRobot naming conventions (`observation.*`)
- Camera stacking logic embedded in inference runtime
- ~60 lines of transformation code

**Current Code**:
```python
# library/src/getiaction/inference/model.py:200-260
def _prepare_inputs(self, observation: Observation) -> dict[str, np.ndarray]:
    # Handles: observation.state vs state
    # Handles: camera stacking for observation.images
    # Handles: tensor → numpy conversion
    # ... 60+ lines of logic
```

**Root Cause**:
- Export wrapper should already have correct input format baked in
- Runtime shouldn't need to know about policy-specific conventions

**Solution**:
- **Option A**: Remove `_prepare_inputs` entirely
  - Export wrappers already traced with correct format
  - Adapters just pass through inputs directly

- **Option B**: Move to adapters
  - Each adapter (ONNX, OpenVINO) handles its own prep
  - Keep InferenceModel policy-agnostic

**Recommendation**: **Option A** - Simplest, leverages export tracing

**Files to Change**:
- `library/src/getiaction/inference/model.py`
- `library/src/getiaction/inference/adapters/*.py`

---

### 2. Input Naming Convention Mismatch

**Problem**:
- Different policies use different input naming conventions
- ACT: `state`, `images` (simple names)
- LeRobot: `observation.state`, `observation.images` (dot notation)
- ONNX export: Converts to `observation_state`, `observation_images` (underscores)

**Root Cause**:
- **Python identifiers cannot contain dots (`.`)**
- Dots are attribute access operators, not valid in variable names
- `observation.state` as a keyword argument is invalid Python syntax
- ONNX/PyTorch must convert dots → underscores during export

**Example**:
```python
# ❌ Invalid Python
def forward(observation.state, observation.images):
    pass

# ✅ Valid Python  
def forward(observation_state, observation_images):
    pass
```

**Current Status**:
- LeRobot policies internally use dot notation (their convention)
- Export process converts to underscores (Python requirement)
- InferenceModel._prepare_inputs() handles mapping between conventions

**Solution** ✅ **IMPLEMENTED**:
1. **Keep LeRobot's internal dot notation** (respect their design)
2. **Standardize export to underscores**:
   ```python
   # library/src/getiaction/policies/lerobot/mixin.py
   input_names_normalized = [name.replace(".", "_") for name in input_sample]
   torch.onnx.export(..., input_names=input_names_normalized)
   ```
3. **InferenceModel maps between conventions** based on model type

**Benefits**:
- ✅ LeRobot policies work correctly with their expected format
- ✅ All exported models have valid Python identifier names
- ✅ Consistent ONNX/OpenVINO naming across all exports
- ✅ Clear separation: internal format vs export format

**Files Changed**:
- ✅ `library/src/getiaction/policies/lerobot/mixin.py` (export standardization)

---

### 3. Data Pipeline Format Conversion

**Problem**:
- `Observation` uses `state`, `images`
- LeRobot expects `observation.state`, `observation.images`
- Lots of manual conversion in `training_step`, `_prepare_inputs`, etc.

**Current Workaround**:
```python
# library/src/getiaction/policies/lerobot/diffusion.py:800-870
def training_step(self, batch: Observation, batch_idx: int) -> torch.Tensor:
    batch_dict = batch.to_dict()

    # Manual name mapping
    if "state" in batch_dict and "observation.state" not in batch_dict:
        batch_dict["observation.state"] = batch_dict.pop("state")

    if "images" in batch_dict and "observation.image" not in batch_dict:
        batch_dict["observation.image"] = batch_dict.pop("images")

    # Temporal dimension handling
    if state.ndim == 2:  # (B, state_dim)
        state = state.unsqueeze(1)  # Add n_obs_steps dim

    # ... 80+ lines total
```

**Root Cause**:
- `FormatConverter.to_lerobot_dict()` exists but isn't used consistently
- Or it doesn't handle all edge cases properly

**Solution**:
1. **Fix `FormatConverter.to_lerobot_dict()`** to handle ALL cases:
   - ✅ Name mapping (state → observation.state)
   - ✅ Temporal dimensions (add n_obs_steps if needed)
   - ✅ Camera formats (dict vs single tensor)
   - ✅ Action horizon (match policy config)

2. **Use it everywhere**:
   ```python
   def training_step(self, batch: Observation, batch_idx: int):
       # ONE LINE instead of 80
       lerobot_batch = FormatConverter.to_lerobot_dict(batch)
       loss, _ = self.lerobot_policy(lerobot_batch)
       return loss
   ```

**Files to Change**:
- `library/src/getiaction/data/lerobot/converters.py` (enhance FormatConverter)
- `library/src/getiaction/policies/lerobot/diffusion.py` (simplify training_step)
- `library/src/getiaction/policies/lerobot/act.py` (if exists)

**Tests Needed**:
- Add comprehensive tests for `FormatConverter.to_lerobot_dict()`
- Test all edge cases: single/multi camera, temporal dims, horizons

---

### 3. Action Horizon Adjustment Workarounds

**Problem**:
```python
# library/src/getiaction/policies/lerobot/diffusion.py:850-870
if action_horizon != expected_horizon:
    action = self._adjust_action_horizon(action, expected_horizon)
    action_is_pad = self._adjust_action_is_pad(...)
```

**Why do we need this?**
- Dataset returns wrong horizon?
- Policy config mismatch?
- Should never happen if pipeline is correct

**Investigation Needed**:
1. Where does the horizon mismatch occur?
   - Dataset loading?
   - Collate function?
   - Policy initialization?

2. Why doesn't LeRobot handle this?
   - Does their code assume matching horizons?
   - Do they pad/trim in their dataset?

**Solutions** (in order of preference):

**Option A: Fix at dataset level** ⭐ BEST
```python
class LeRobotDataset:
    def __getitem__(self, idx):
        # Return action with correct horizon from start
        action = self._get_action_chunk(idx, horizon=self.policy_config.horizon)
        return {"action": action, ...}
```

**Option B: Fix in FormatConverter**
```python
def to_lerobot_dict(batch, policy_config):
    # Adjust action horizon based on policy config
    if "action" in batch:
        batch["action"] = adjust_horizon(batch["action"], policy_config.horizon)
```

**Option C: Keep workaround but document WHY**
- Add clear comment explaining the mismatch source
- Link to LeRobot issue if it's their limitation

**Recommendation**: **Option A** - Fix at source

**Files to Investigate**:
- `library/src/getiaction/data/lerobot/dataset.py`
- `library/src/getiaction/data/lerobot/datamodule.py`
- Check LeRobot's own dataset implementations

---

## 🎯 Medium Priority Issues

### 4. Export Wrapper Naming

**Current**: `DiffusionInferenceWrapper`

**Problem**: Name is ambiguous
- Used only for export, not general inference
- Might be confused with inference runtime

**Solution**: Rename to `DiffusionExportWrapper`
```python
class DiffusionExportWrapper(nn.Module):
    """Self-contained export model with embedded denoising loop.

    This wrapper is used ONLY during export to embed the complete
    100-step diffusion inference pipeline into the exported graph.
    """
```

**Files to Change**:
- `library/src/getiaction/policies/lerobot/diffusion.py`

---

### 5. Sample Input Generation

**Current Approach**: Manual tensor creation based on config
```python
@property
def sample_input(self) -> dict[str, torch.Tensor]:
    state_dim = config.input_features["observation.state"].shape[0]
    sample["observation.state"] = torch.randn(batch_size, state_dim)
    # ... manual shape calculations
```

**Problem**:
- Error-prone manual shape calculations
- Doesn't match actual data format edge cases
- Requires config parsing

**Alternative: Dry-run approach**
```python
@property
def sample_input(self) -> dict[str, torch.Tensor]:
    """Get sample input by extracting from actual dataset batch."""
    if not hasattr(self, '_sample_input_cache'):
        # Get one batch from training dataset
        batch = self._get_sample_batch_from_dataset()
        # Extract just observation inputs (first sample only)
        self._sample_input_cache = {
            k: v[:1].clone()
            for k, v in batch.items()
            if k.startswith('observation.')
        }
    return self._sample_input_cache
```

**Pros**:
- ✅ Always matches actual data format
- ✅ Handles temporal dims automatically
- ✅ No manual calculations

**Cons**:
- ⚠️ Requires dataset access during export
- ⚠️ Slight overhead (one-time, cached)

**Recommendation**:
- Keep current approach if it works reliably
- Switch to dry-run if sample input bugs occur

---

## 🤷 Low Priority (Nice to Have)

### 6. ACT Model/Policy Separation

**LeRobot Pattern**:
- `ACTPolicy` = training logic + loss computation
- `ACTModel` = pure inference (no training)

**Current Approach**:
- Single `Diffusion` class handles both
- `DiffusionInferenceWrapper` for export

**Should we refactor?**

**Pros of separation**:
- ✅ Matches LeRobot's architecture
- ✅ Cleaner separation of concerns
- ✅ Could reuse `ACTModel` pattern for other policies

**Cons**:
- ⚠️ Large refactor for unclear benefit
- ⚠️ LeRobot's Diffusion doesn't separate either
- ⚠️ Current approach works fine

**Recommendation**: **Don't refactor** unless:
- You're adding many more policies
- LeRobot refactors their Diffusion similarly
- Current approach causes maintenance issues

---

## Implementation Priority

### Phase 1: Critical Fixes ✅ COMPLETED
1. ✅ **Enhanced `FormatConverter.to_lerobot_dict()`**
   - Added `policy_config` parameter
   - Handles temporal dimension (n_obs_steps)
   - Handles action horizon adjustment
   - Handles action_is_pad adjustment
   - All logic centralized in one place

2. ✅ **Simplified `training_step()`**
   - Reduced from ~45 lines to ~17 lines
   - Single FormatConverter call replaces manual conversions
   - Removed 60+ lines of helper methods (`_adjust_action_horizon`, `_adjust_action_is_pad`)

3. ⏳ **Investigate action horizon mismatch** (NEEDS INVESTIGATION)
   - Helper functions moved to FormatConverter
   - Still need to find root cause in dataset
   - May be able to remove adjustments entirely

### Phase 2: Code Cleanup (Next) 🎯
4. **Simplify or remove `InferenceModel._prepare_inputs()`**
   - Test if export wrappers already handle format
   - Remove if redundant

5. **Rename wrappers for clarity**
   - `DiffusionInferenceWrapper` → `DiffusionExportWrapper`

### Phase 3: Nice to Have (Later) 🤷
6. **Consider dry-run sample input** (only if needed)
7. **Consider model/policy separation** (only if scaling issues)

---

## Success Criteria

After implementing Phase 1 fixes:
- ✅ `training_step()` is < 20 lines
- ✅ No manual name mapping (`state` → `observation.state`)
- ✅ No action horizon adjustment workarounds
- ✅ `FormatConverter` handles all edge cases
- ✅ All E2E tests pass (target: 13/13)

---

## Related Files

### Core Files to Modify:
- `library/src/getiaction/data/lerobot/converters.py`
- `library/src/getiaction/policies/lerobot/diffusion.py`
- `library/src/getiaction/policies/lerobot/act.py` (if exists)
- `library/src/getiaction/inference/model.py`

### Files to Review:
- `library/src/getiaction/data/lerobot/dataset.py`
- `library/src/getiaction/data/lerobot/datamodule.py`
- `library/tests/unit/test_format_converter.py` (create if missing)

---

## Notes & Open Questions

1. **Why does action horizon mismatch?**
   - Dataset returns horizon=8 but policy expects horizon=16?
   - Need to trace through data loading pipeline

2. **Does LeRobot handle variable horizons?**
   - Check their dataset implementations
   - Might be an upstream issue

3. **Is FormatConverter tested comprehensively?**
   - Need unit tests for all edge cases
   - Integration tests with real batches

4. **Can we remove _prepare_inputs entirely?**
   - Need to verify export wrappers handle format
   - Test with ONNX/OpenVINO/TorchExportIR

---

## References

- Main discussion: [See conversation on Nov 12, 2025]
- Current implementation: `feat/add-lerobot-export-inference` branch
- Related: `DiffusionInferenceWrapper` design (lines 47-200 in diffusion.py)
