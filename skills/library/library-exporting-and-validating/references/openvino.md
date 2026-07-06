# OpenVINO Export Notes

- Use OpenVINO for Intel CPU/GPU/NPU deployment targets.
- Ensure dynamic shapes, tokenizers, and preprocessing are compatible with OpenVINO conversion.
- Validate first-inference warmup behavior separately from steady-state latency.
- Runtime core ships an OpenVINO adapter, so OpenVINO artifacts should be loadable by `physicalai` when dependencies are installed.
