---
name: pai-export-policy
description: Exports and validates a trained Physical AI Studio policy for Runtime deployment. Use when calling policy.export or physicalai export with an existing checkpoint, choosing ONNX/OpenVINO/Torch/ExecuTorch, checking numerical parity and export metadata, or preparing an artifact for InferenceModel. DO NOT USE FOR implementing a new export backend in Studio source.
license: Apache-2.0
---

# Exporting and Validating Studio Policies

The `physicalai-train` package exposes `ExportBackend` (`onnx`, `openvino`, `torch`, `executorch`) and `policy.export(output_dir, backend=...)`; `physicalai export` offers the CLI equivalent. Studio owns export; Runtime owns loading. See the [export how-to](https://github.com/open-edge-platform/physical-ai-studio/blob/main/library/docs/how-to/export/export_inference.md) and Runtime's [manifest schema](https://github.com/openvinotoolkit/physicalai/blob/main/docs/reference/manifest-schema.md).

## Workflow

1. **Identify the inputs**: installed policy class (e.g. `physicalai.policies.ACT`), `.ckpt` path, target backend, and the Runtime loader behavior expected for that backend.
   - Done when: all four are pinned before exporting.
2. **Pick the route** and keep both consistent — they must produce the same artifact:
   - Python: `policy.export(output_dir, backend=ExportBackend.ONNX)`.
   - CLI: `physicalai export --policy physicalai.policies.ACT --ckpt_path model.ckpt --backend onnx --output_dir ./export`.
3. **Read backend constraints before exporting.** See the backend reference for the target (`references/<backend>.md`). If the selected policy/backend combination is unsupported, explain the constraint instead of modifying Studio source as part of a customer export.
4. **Export, then validate numerical parity** against the Torch policy path on representative inputs. Parity proves correctness.
   - Done when: max abs/rel diff on sample inputs is within the family's tolerance, or the divergence is understood and documented.
5. **Validate artifact structure and metadata** against Runtime's [manifest schema](https://github.com/openvinotoolkit/physicalai/blob/main/docs/reference/manifest-schema.md).
   - Done when: the expected model file and metadata files exist, and input/output/feature names match Runtime preprocessing.
6. **Confirm the Runtime path.** For deployment-bound artifacts, verify Runtime can auto-detect (by extension) or explicitly load the backend via `InferenceModel(...)`.

## Validation loop

Run export → inspect artifact and manifest → check numerical parity → fix the project's inputs/config → repeat:

```bash
# from your project directory
physicalai export --policy <ClassPath> --ckpt_path <model.ckpt> --backend <backend> --output_dir ./export
```

For an API integration, load the same checkpoint, call `policy.export("./export-api", backend=ExportBackend.<BACKEND>)`, and compare artifact metadata with the CLI output. If export fails because of the policy implementation, report the failure to its maintainer rather than changing Studio internals here.

Treat **parity** (correctness) and **latency/warmup** (deployment viability) as separate checks; passing one does not imply the other.

## Backend notes

- **onnx** / **openvino** — deployment-oriented; Runtime core ships adapters, so artifacts load when deps are installed.
- **torch** — development/debugging; only claim deployment support when a matching Runtime adapter is installed and documented.
- **executorch** — optional, dependency-sensitive, edge/mobile; Runtime core ships no adapter in this package. Treat as available only with a documented companion distribution.

## Required checks

- Export directory contains the expected backend model file **and** metadata files.
- Metadata names inputs/outputs/features consistently with Runtime preprocessing and action-chunk semantics.
- Python API export and CLI export produce equivalent artifact structure and metadata.
- The selected backend dependency is installed; report missing optional packages with their install guidance.
- Only describe deployment as supported when Runtime has a matching adapter or documented companion.

## References

- [Runtime manifest schema](https://github.com/openvinotoolkit/physicalai/blob/main/docs/reference/manifest-schema.md) — canonical artifact structure.
- `references/onnx.md`, `references/openvino.md`, `references/torch.md`, `references/executorch.md` — per-backend constraints.
