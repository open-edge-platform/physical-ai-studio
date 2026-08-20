# ONNX Export Notes

- Use ONNX for cross-platform deployment and ONNX Runtime execution.
- Keep input and output names stable; Runtime uses exported metadata and backend introspection to prepare inputs.
- Watch for model code paths that are not ONNX-traceable, especially Python control flow and non-tensor outputs.
- Runtime core ships an ONNX adapter, so ONNX artifacts should be loadable by `physicalai` when dependencies are installed.
