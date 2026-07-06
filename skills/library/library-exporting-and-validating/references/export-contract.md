# Export/Load Contract

Studio produces deployment artifacts; Runtime loads them with `InferenceModel.load(...)`.

## Required Artifact Properties

- A backend-identifying model file exists in the export directory.
- Metadata describes the policy, backend, expected inputs, expected outputs, preprocessing/postprocessing requirements, and action chunk semantics.
- Runtime can either auto-detect the backend from file extensions or load it when the backend is explicitly provided.
- Optional backend dependencies fail with clear installation guidance.

## Backend Ownership

- Studio owns export implementation and export metadata generation.
- Runtime owns adapter discovery, backend loading, preprocessing, inference execution, and action selection from exported artifacts.
- Studio and Runtime must keep this contract synchronized. If this file is mirrored into Runtime, CI should fail when copies diverge.

## Compatibility Rules

- Do not change metadata field names or semantics without coordinating Runtime changes.
- Do not add a backend to user-facing instructions unless Runtime can load it in the current package or via a documented companion distribution.
- Treat numerical parity and latency validation as separate checks: parity proves correctness; latency proves deployment viability.
