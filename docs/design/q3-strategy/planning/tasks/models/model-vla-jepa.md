---
type: task
priority: p1
area: model
milestone: Q3 Strategy
owner: E4
parent: ../epics/001-q3-model-coverage.md
---

# Implement VLA-JEPA

## Goal

Add VLA-JEPA for robustness and latent-world-model coverage.

## Acceptance Criteria

- [ ] Config/model/policy classes
- [ ] Inference path scoped with WM dropped where applicable
- [ ] Training path
- [ ] Benchmark entry
- [ ] Export path: ONNX, OpenVINO, Torch
- [ ] Export-equivalence pass
- [ ] `InferenceModel.load(...)` pass
