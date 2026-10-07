---
type: task
priority: p1
area: model
milestone: Q3 Strategy
owner: E3
parent: ../epics/001-q3-model-coverage.md
---

# Implement DreamZero

## Goal

Add DreamZero as the first WAM policy.

## Acceptance Criteria

- [ ] Config/model/policy classes
- [ ] WAM data needs mapped to generic `Observation`
- [ ] Training path
- [ ] Benchmark entry
- [ ] Export path: Torch, ONNX; OpenVINO stretch
- [ ] Export-equivalence pass
- [ ] `InferenceModel.load(...)` pass
