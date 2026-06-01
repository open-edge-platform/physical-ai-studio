# Dataset Import

This folder contains the backend implementation of dataset import jobs.

## What this feature does

Dataset import is implemented as an **asynchronous, multi-step job flow**:

1. API creates an import job (`awaiting_archive_upload`)
2. API receives and stores uploaded ZIP, then marks job `pending` (`queued_for_detection`)
3. Worker picks up the job, detects adapter compatibility, and builds a draft manifest
4. Worker runs draft assessment (post-parse checks), then pauses for user review (`awaiting_user_review`)
5. API finalizes with required input (environment), marks job `pending` (`queued_for_import`)
6. Worker runs **pre-commit validation** and, if valid, commits import and creates the dataset

Core files:

- `service.py`: Job lifecycle operations for dataset import
- `adapters/base.py`: Adapter contract
- `adapters/lerobot_v3.py`: Current adapter implementation
- `workers/dataset_import_worker.py` (outside this folder): worker orchestration loop
- `api/dataset_import.py` (outside this folder): HTTP endpoints

---

## UI → API integration flow

The UI should use this sequence for a successful import:

### 1) Prepare job

`POST /api/projects/{project_id}/imports/datasets:prepare`

Form fields:

- `format_hint`: usually `auto` (recommended), or `lerobot_v3`

Response:

- Returns `Job` with `payload.step = awaiting_archive_upload`
- Save `job_id`

### 2) Upload archive

`PUT /api/projects/{project_id}/imports/datasets/{job_id}:upload`

Multipart fields:

- `archive`: ZIP file only

Response:

- Job is updated to `status=pending`, `payload.step=queued_for_detection`
- Worker will pick it up

### 3) Poll job state

Use existing job endpoints/streams to monitor job updates.

After detection+validation, worker moves the job to:

- `status=pending`
- `payload.step=awaiting_user_review`
- `payload.validation_report`
- `payload.dataset_manifest_draft`

The `dataset_manifest_draft.dataset_schema` field is populated with a **recording schema** inferred
from the source dataset metadata (e.g. LeRobot `meta/info.json`).  It describes:

- **cameras** – one entry per video stream found under `observation.images.*`,
  with name, width, height, and fps where available.
- **robots** – one entry per robot with its joint names (derived from
  `action` / `observation.state` feature `names` by stripping per-joint
  suffixes such as `.pos`).  The robot type comes from the top-level
  `robot_type` field.

Both lists default to empty when the source metadata is absent or incomplete,
so existing imports are unaffected.

UI should then present finalization inputs.

### 4) Finalize

`POST /api/projects/{project_id}/imports/datasets/{job_id}:finalize`

JSON body (`DatasetImportFinalizeInput`):

- `environment_id` (required)
- `default_task` (optional)

Note: `dataset_name` is provided during prepare (`POST .../datasets:prepare`).

Response:

- Job becomes `status=pending`, `payload.step=queued_for_import`
- Worker resumes, runs pre-commit validation, then performs commit

### 5) Completion

Worker eventually marks job `completed` and sets:

- `payload.result_dataset_id`

UI can navigate to/open the resulting dataset.

### Cancel behavior

`POST /api/projects/{project_id}/imports/datasets/{job_id}:cancel`

Cancellation is currently allowed only while the job is in `awaiting_user_review`.

---

## Adapter design

Dataset import uses a pluggable adapter model.

Contract (`adapters/base.py`):

- `detect(archive: SafeZipArchive) -> tuple[bool, ImportValidationReport]`
- `build_draft(archive: SafeZipArchive, payload: DatasetImportJobPayload) -> tuple[DatasetManifest, ImportValidationReport]`
- `validate_pre_commit(payload: DatasetImportJobPayload) -> ImportValidationReport`
- `commit(payload: DatasetImportJobPayload, project_id: UUID, archive: SafeZipArchive) -> Dataset`

### Stage semantics (important)

`build_draft` and `validate_pre_commit` happen at different points for different purposes:

| Method | When it runs | Purpose |
|---|---|---|
| `build_draft` | After adapter detection, before user finalization input | Parse canonical draft manifest and return user-facing assessment messages |
| `validate_pre_commit` | Immediately before `commit(...)` | Final gate: block execution if required data is missing or invalid |

### Suggested naming (conceptual)

Current names are kept for code stability, but conceptually:

- `build_draft` ≈ **`parse_and_assess_draft`**
- `validate_pre_commit` ≈ **`validate_ready_to_commit`**

These names better communicate intent than "pre_*" alone.

### "Parse, don't validate" in current design

We follow this by combining draft parsing and user-facing draft assessment into one operation:

- `build_draft(...)` returns `(manifest, report)`
- hard parse failures still raise exceptions
- soft issues are returned as validation messages for UX

`validate_pre_commit(...)` remains separate as the strict execution gate immediately before import.

Worker behavior:

- If `format_hint != auto`, worker tries matching adapter first
- If `format_hint == auto`, worker scans adapters and uses first `detect(...) == True`
- If no adapter matches, job fails

Currently enabled adapters in worker:

- `LeRobotV2Adapter`
- `LeRobotV3Adapter`

---

## LeRobotV3 adapter compatibility criteria

`LeRobotV3Adapter.detect(...)` opens the uploaded ZIP and checks that it looks like a supported **v3** dataset archive.

The ZIP is considered compatible when **all** are true:

1. It contains metadata info file:
   - `meta/info.json` (directly or via normalized path)
2. It contains v3 tasks index:
   - `meta/tasks.parquet`
3. It contains v3 data evidence:
   - at least one parquet with `.../file-*.parquet` under `data/...`

It is rejected as v3 when v2-only marker files are present:

- `meta/tasks.jsonl`
- `meta/episodes.jsonl`

If ZIP cannot be opened (`BadZipFile`), detection returns `False`.

Additional adapter behavior:

- Validates finalize input presence before commit
- Extracts ZIP safely (rejects unsafe paths like absolute paths or `..` traversal)
- Creates a new dataset record

---

## Practical UI notes

- Prefer `format_hint=auto` unless user explicitly chooses `lerobot_v3`
- Show progress and status messages from job updates
- Wait for `awaiting_user_review` before enabling final submit
- Surface worker validation messages (`payload.validation_report.messages`) clearly to users
  - treat `severity=error` as blocking
  - treat `severity=warning` as non-blocking

---

## Operations

### Environment variables

All knobs are read from environment (or `.env` file) at startup via `Settings` in `settings.py`.

| Variable | Default | Description |
|---|---|---|
| `DATA_IMPORT_MAX_UPLOAD_BYTES` | 100 GiB | Maximum raw archive size accepted by data-import upload endpoints. Checked by `upload_size_guard_middleware` before the body is read, and again by `check_disk_headroom` before file write. |
| `DATA_IMPORT_MIN_FREE_BYTES` | 1 GiB | Minimum free headroom that must remain on the target filesystem *after* a write (applies to both cache dir staging and extraction destination). |
| `DATA_IMPORT_MAX_UNCOMPRESSED_BYTES` | 200 GiB | Maximum total uncompressed size permitted across all ZIP entries. |
| `STORAGE_DIR` | Linux: `${XDG_DATA_HOME:-~/.local/share}/physicalai`; macOS: `~/Library/Application Support/physicalai`; Docker: `/app/storage` | Root for `datasets/` (extraction destination) and `cache/imports/datasets/` (staging area). Both must live on a filesystem with adequate free space. Existing users of the old default `~/.cache/physicalai` are prompted to migrate on interactive startup; non-interactive startup fails unless `AUTO_MIGRATE_STORAGE_DIR=true` is set. |

### Recommended values for large imports (≥ 50 GB)

```bash
# Raise the raw upload ceiling to match your largest expected archive
DATA_IMPORT_MAX_UPLOAD_BYTES=107374182400   # 100 GiB (default is already fine)

# Raise the uncompressed cap if archives expand beyond the 200 GiB default
DATA_IMPORT_MAX_UNCOMPRESSED_BYTES=322122547200   # 300 GiB example

# Keep at least 10 GiB of headroom so the OS is not starved during extraction
DATA_IMPORT_MIN_FREE_BYTES=10737418240

# Point STORAGE_DIR at a volume with enough room for both staging and extraction
# (the upload is cached under $STORAGE_DIR/cache/imports/datasets/ and then
#  extracted into $STORAGE_DIR/datasets/<dataset-id>/)
STORAGE_DIR=/mnt/large-volume/physicalai
```

Disk is touched **twice** per import: once when the raw archive lands in the cache dir, and once when it is extracted to the datasets dir. Ensure both paths are on a filesystem that can absorb the uncompressed payload plus the configured headroom.

---

## Security

### HTTP upload size guard (`middleware/upload_size_guard.py`)

`upload_size_guard_middleware` intercepts data-import upload requests. If the `Content-Length` header is present and exceeds `DATA_IMPORT_MAX_UPLOAD_BYTES`, the middleware returns a `413` response **before FastAPI reads a single byte of the body**. Requests without a `Content-Length` header pass through (disk headroom guards still apply).

### ZIP safety checks (`services/archive_safety.py`)

Validation runs in the upload endpoint immediately after the archive is persisted to disk via `SafeZipArchive.validate()`:

| Check | Raises | Configured by |
|---|---|---|
| **Nested ZIP** – any entry whose extension is `.zip` (case-insensitive) | `ZipBombDetectedError` | Internal policy (not configurable) |
| **File count** – number of ZIP entries exceeds the limit | `ZipBombDetectedError` | Internal policy (`200000`) |
| **Uncompressed bytes** – sum of all `file_size` fields in the central directory exceeds the limit | `ZipBombDetectedError` | `DATA_IMPORT_MAX_UNCOMPRESSED_BYTES` |

### Path traversal and symlink protection (`archive_safety.validate_zip_entries`)

For every entry in the archive (checked at both upload time and during extraction in `SafeZipArchive.extract_to`):

- **Absolute paths** are rejected.
- **Directory traversal** (`..`) entries are rejected.
- **Symlinks** (Unix mode `0o120000`) are rejected.
- **Resolved-path escape** is blocked during extraction by validating target paths stay inside destination root.

All violations raise `ZipBombDetectedError`, which causes the endpoint to delete the staged archive and return an error response.

### Disk headroom checks (`archive_safety.check_disk_headroom`)

`check_disk_headroom` is called at two points:

1. **Upload endpoint** (`api/dataset_import.py`) – checks the **cache dir** (`$STORAGE_DIR/cache/imports/datasets/`) before writing the incoming archive.
2. **Adapter commit phase** (`services/dataset_import/adapters/lerobot_v3.py`) – checks the **datasets dir** (`$STORAGE_DIR/datasets/`) before extraction.

Both calls enforce `free_bytes ≥ required_bytes + DATA_IMPORT_MIN_FREE_BYTES`. Violation raises `InsufficientDiskSpaceError`.

### Staged archive cleanup behavior

The staged archive is removed in these scenarios:

- **Upload-time validation failure** – endpoint deletes it before returning error.
- **Successful commit** – worker deletes it after commit finalization.
- **Cancel / worker failure** – best-effort cleanup in service/worker paths.

**Caveat**: cleanup is best-effort in error paths. If the process is killed mid-import (OOM/SIGKILL/host restart), orphaned staged archives or partially extracted directories can remain and should be monitored/cleaned operationally.

---

## Planned: resumable uploads (TUS)

Chunked/resumable upload support via TUS is planned but **not yet implemented**. Current `PUT …:upload` is single-shot multipart upload; interrupted uploads cannot be resumed.
