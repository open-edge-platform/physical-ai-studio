# Model exports and variants

Model exports are delivered in three sequential changes:

1. Re-export a stored checkpoint to a missing, supported backend.
2. Remove a backend export without deleting its checkpoint or training history.
3. Create OpenVINO INT8 variants, select them for inference/download, and remove individual variants.

RTC and additional SnapFlow variants are follow-up work, not part of this implementation.

## Identity and storage

A training model retains one model ID, checkpoint, dataset provenance, and training history.
Exporting or compressing it does not create another training-model database row.

An exported package is identified by **model ID, backend, and variant ID**:

```text
<model>/model.ckpt
<model>/exports/<backend>/manifest.json             # original package
<model>/exports/<backend>/...                       # original artifacts and dependencies
<model>/exports/<backend>/variants/int8/manifest.json
<model>/exports/<backend>/variants/int8/...
```

Existing packages without Studio metadata remain original exports. Re-export jobs initially
recorded publication metadata under `studio.export`; variant-aware readers also accept that
metadata. New packages use versioned `studio.variant` metadata, including the variant ID,
compression method, creation time, and producing job ID. Runtime's artifact manifest remains
the source of truth for artifact paths and IO specifications.

`ExportPackage` in `application/backend/src/schemas/model_export.py` centralizes package paths,
manifest validation, and dependency enumeration. Original-package downloads and compression
exclude the `variants/` subtree. Full-model downloads include every published variant.
Variant-only backends remain discoverable after their original package is removed.

## Export jobs and removal

`ModelExportJob` uses the `model_export` job type. Its operation distinguishes checkpoint
export (`export`) from OpenVINO symmetric weight compression (`int8_sym`). The latter compresses
weights, not activations, and never modifies the original package.

Both operations share the existing job scheduler's local execution slot, process isolation,
logs, cancellation, and recovery. Each job builds in `<models_dir>/export_jobs/<job_id>/`,
outside the model's downloadable directory. Publication commits the job's completion decision
before renaming the complete package into place. If the process exits between those actions,
the scheduler finishes publication using the recorded job ID. A persisted cancellation request
competes with completion on the same job row, so losing a process signal cannot publish canceled work.

Submission and backend/variant removal serialize their filesystem decisions using the model's
database row. Removal rejects pending/running jobs and committed-but-unpublished output.
Inference holds an export lease from loading through teardown; backend deletion excludes every
variant's lease, while variant deletion only excludes that package. Checkpoints, training history,
and unrelated packages are preserved.

The public routes keep format and variant separate:

- `POST /api/models/{model_id}/exports/{backend}`: export a missing backend from its checkpoint.
- `DELETE /api/models/{model_id}/exports/{backend}`: remove the backend and all its variants.
- `POST /api/models/{model_id}/optimize`: create the OpenVINO INT8 package.
- `DELETE /api/models/{model_id}/exports/{backend}/variants/{variant}`: remove one package.
- Backend downloads and runtime-session model loading accept a variant, defaulting to `original`.

Model details advertise supported `available_actions`; the UI does not infer export capability
from a missing directory alone. Checkpoint exports require the policy and backend dependencies
to be installed. Existing nonempty or empty backend directories are not overwritten by checkpoint
export; remove the backend first to retry it. A backend with retained variants is likewise not
offered for original re-export until that backend is removed.

## RTC and SnapFlow follow-ups

Keep the export backend as the serialization/execution format (`openvino`, `onnx`, `torch`, or
`executorch`). RTC and SnapFlow must not become fake backend formats or replace the training
model's identity. Decide whether each feature needs a distinct exported graph, a distinct source
checkpoint, or only runtime configuration before adding an operation.

For features that produce another package:

- Extend the validated variant-ID and operation allowlists; do not accept arbitrary filesystem paths.
- Keep package identity separate from transformation metadata. `int8` is a package ID;
  `int8_sym` describes its weight-compression recipe. Additional recipe dimensions can describe RTC,
  SnapFlow, and supported combinations without treating every variant as an INT8 model.
- Replace the current two-operation target selection with an explicit recipe-to-target mapping when
  adding a third operation. Define source package/checkpoint selection and collision rules explicitly.
- Add capabilities only for policy/backend combinations the library actually supports. Reuse the
  export worker's staging, publication, cancellation, recovery, and inference-lease lifecycle.
- Preserve existing `original` and `int8` package paths and default loading behavior. Extend UI labels,
  selection, downloads, import validation, and removal tests alongside new manifest metadata.

SnapFlow already appears in training-model properties. Do not label an ordinary checkpoint export
as newly distilled or add post-training distillation unless the source checkpoint and library
workflow support it. Runtime-only RTC settings should remain runtime configuration rather than
creating duplicate artifact packages.

## Validation

From `application/backend/`:

```bash
uv run pytest tests/integration/test_model_export.py tests/integration/test_model_optimization.py \
  tests/models/test_model_variants.py tests/models/test_int8_compression.py
uv run pytest tests/runtime_session tests/services/test_job_handlers.py tests/unit/core/test_scheduler.py
```

The compression smoke test needs the `nncf` extra and checks that a real compressed OpenVINO
artifact still compiles on CPU. Workflow tests mock conversion and use isolated SQLite databases.

Regenerate the gitignored UI API files for each branch before type-checking or testing:

```bash
# From application/backend/
uv run physicalai-studio gen-api --target-path ../ui/src/api/openapi-spec.json
# From application/ui/
npm run build:api
npm run type-check
npm run test:unit
npm run test:component -- tests/int8-model-variants.spec.ts
```
