---
name: physicalai-train-exporting-and-validating
description: Exports and validates Physical AI Studio policies for Runtime deployment. Use when working on policy.export(...), the physicalai export CLI, the ONNX/OpenVINO/Torch/ExecuTorch backends, export metadata, numerical parity checks, or the Studio side of the export/load contract that Runtime consumes with InferenceModel.load(...).
license: Apache-2.0
---

# Exporting and Validating Studio Policies

Export lives in `library/src/physicalai/export/`: `backends.py` (the `ExportBackend` enum — `onnx`, `openvino`, `torch`, `executorch` — plus per-backend parameter classes) and `mixin_policy.py` (`ExportablePolicyMixin`, which gives policies `export(output_dir, backend=...)`). The Python API is primary library behavior; the CLI entry `library/src/physicalai/cli/export.py` must preserve the same artifact contract. Studio owns export; Runtime owns loading.

## Workflow

1. **Identify the inputs**: source policy class (e.g. `physicalai.policies.ACT`), `.ckpt` path, target backend, and the Runtime loader behavior expected for that backend.
   - Done when: all four are pinned before touching code.
2. **Pick the route** and keep both consistent — they must produce the same artifact:
   - Python: `policy.export(output_dir, backend=ExportBackend.ONNX)`.
   - CLI: `physicalai export --policy physicalai.policies.ACT --ckpt_path model.ckpt --backend onnx --output_dir ./export`.
3. **Read backend constraints before editing generic code.** See the backend reference for the target (`references/<backend>.md`). Do not generalize a fix across backends without checking each.
4. **Export, then validate numerical parity** against the Torch policy path on representative inputs. Parity proves correctness.
   - Done when: max abs/rel diff on sample inputs is within the family's tolerance, or the divergence is understood and documented.
5. **Validate artifact structure and metadata** against `references/export-contract.md`.
   - Done when: the expected model file and metadata files exist, and input/output/feature names match Runtime preprocessing.
6. **Confirm the Runtime path.** For deployment-bound artifacts, verify Runtime can auto-detect (by extension) or explicitly load the backend via `InferenceModel.load(...)`.

## Validation loop

Run export → validate → fix → repeat until both parity and structure pass:

```bash
# from library/
physicalai export --policy <ClassPath> --ckpt_path <model.ckpt> --backend <backend> --output_dir ./export
uv run pytest tests/unit/export -k <backend>
```

For API-facing changes, add or run an equivalent Python script/test that loads the checkpoint, calls `policy.export("./export-api", backend=ExportBackend.<BACKEND>)`, and compares artifact metadata with the CLI output.

Treat **parity** (correctness) and **latency/warmup** (deployment viability) as separate checks; passing one does not imply the other.

## Backend notes

- **onnx** / **openvino** — deployment-oriented; Runtime core ships adapters, so artifacts load when deps are installed.
- **torch** — development/debugging; only claim deployment support when a matching Runtime adapter is installed and documented.
- **executorch** — optional, dependency-sensitive, edge/mobile; Runtime core ships no adapter in this package. Treat as available only with a documented companion distribution.

## Required checks

- Export directory contains the expected backend model file **and** metadata files.
- Metadata names inputs/outputs/features consistently with Runtime preprocessing and action-chunk semantics.
- Python API export and CLI export produce equivalent artifact structure and metadata.
- Backend-specific dependencies are imported lazily or guarded with clear install guidance.
- Do not add a backend to user-facing docs unless Runtime can load it in-package or via a documented companion.
- CLI docs (`library/docs/how-to/export/`) and Python API examples stay consistent.

## References

- `references/export-contract.md` — artifact requirements shared with Runtime (keep synchronized; CI should fail on divergence).
- `references/onnx.md`, `references/openvino.md`, `references/torch.md`, `references/executorch.md` — per-backend constraints.
