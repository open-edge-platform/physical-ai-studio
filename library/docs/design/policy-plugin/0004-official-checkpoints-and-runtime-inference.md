---
status: accepted
---

# Keep official checkpoints and serve upstream inference through Runtime adapters

Training through an upstream plugin returns the checkpoint in its official format, metrics under their upstream names, and a record of what produced it: plugin, upstream revision, downstream patches, Python environment, and dataset conversion. Inference for upstream-trained policies uses a plugin-provided Runtime adapter that forwards requests to the upstream inference code running in its Python environment, so benchmarks and robot control use Runtime's existing `InferenceModel` and `PolicySource`. Converting every upstream checkpoint to a Studio format, or reimplementing upstream inference, would contradict upstream-first integration and multiply maintenance.

## Consequences

- Studio never renames an upstream checkpoint to `model.ckpt` or presents it as a Lightning checkpoint. Model storage must stop assuming every trained model contains `model.ckpt` and `exports/`.
- Studio's model list shows the plugin and each validated capability for a model.
- Export to OpenVINO or ONNX is a separate capability with its own validation, not a prerequisite for inference.
- The Runtime maintainers must agree to adapters that forward inference to another process. Exactly one component owns normalization and action queues, and timeouts or stale responses must fail safe.

See the [experience proposal](policy-integration-experience.md).
