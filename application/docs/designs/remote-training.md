# Design: Remote Training

## Summary

Remote training moves GPU-heavy policy training from a recording station to a dedicated GPU server (the **trainer service**). That keeps recording machines lightweight. The Studio backend uploads a dataset snapshot to an ephemeral private Hugging Face dataset repo, submits a job over HTTP, mirrors remote progress into its own job record, downloads the trained model archive, and imports it. In the UI, the Models screen behaves the same as local training.

## Goals

- Run training on a separate GPU machine without changing the user workflow.
- Keep recording-only installs free of `torch`/`physicalai`.
- Serve multiple recording stations from one GPU server with a persistent, restart-safe queue.
- Mirror remote progress and support cancellation.

## Non-goals

- **Authenticated or multi-tenant trainer access.** The trainer trusts callers on its network. It has no per-station auth, authorization, or request signing. Deploy it on a private network; do not expose it publicly.
- **Resumable transfers or jobs.** If snapshot upload, job submission, or artifact download fails, the job fails and the user retrains. There is no checkpoint-level resume across the HTTP boundary.
- **Dataset transfer without Hugging Face Hub.** The snapshot always moves through an ephemeral HF dataset repo. Direct backend-to-trainer upload is not supported.

## Architecture

Two services deploy independently:

- **Studio backend** (`application/backend/`) - orchestrates the job and owns the user-facing job record.
- **Trainer service** (`application/trainer/`) - standalone FastAPI app that queues and runs training with `physicalai` + Lightning. Only deployed in remote training mode.

The dataset moves through Hugging Face Hub. Everything else moves over HTTP.

```mermaid
flowchart LR
    subgraph Studio["Studio backend (recording station)"]
        W[Training worker] --> RB[RemoteTrainingBackend]
    end
    subgraph HF["Hugging Face Hub"]
        DS[(Ephemeral private<br/>dataset repo)]
    end
    subgraph Trainer["Trainer service (GPU server)"]
        API[FastAPI /jobs] --> QM[QueueManager]
        QM --> RN[TrainerRunner]
        ST[(SQLite job store)]
        QM <--> ST
    end

    RB -- "1. push snapshot (pinned SHA)" --> DS
    RB -- "2. POST /jobs" --> API
    RB -- "3. stream GET /jobs/{id}/events (SSE)" --> API
    RN -- "pull snapshot (pinned SHA)" --> DS
    RB -- "4. GET /jobs/{id}/artifact" --> API
    RB -- "5. delete repo" --> DS
```

## Backend integration

The training worker is backend-agnostic. `get_training_backend()` returns `LocalTrainingBackend` or `RemoteTrainingBackend` based on `settings.training_mode`. Both implement the `TrainingBackend` protocol:

```python
async def train(self, context: TrainingContext) -> None: ...
```

`TrainingContext` includes the `Job`, `Model`, `Snapshot`, `TrainJobPayload`, optional `base_model`, an `output_dir`, a `progress` reporter, and a `should_stop` callback. Backends must leave a fully populated model directory at `output_dir` (checkpoint, logger output, `exports/`), report progress, and stop quickly when `should_stop()` returns `True`. This contract is the interchange point between local and remote.

Heavy imports are deferred so `TRAINING_MODE=remote` never imports `torch`. `RemoteTrainingBackend` lazily imports `huggingface_hub` and never imports `physicalai`.

## Local training (default)

Local training remains the default and is unchanged. With `TRAINING_MODE=local`, `get_training_backend()` returns `LocalTrainingBackend`, which trains in the worker process with `torch`/Lightning. This requires the `[train]` extra on the recording station. No trainer service, Hugging Face transfer, or `TRAINER_URL` is involved.

Because both backends satisfy the same protocol and produce the same `output_dir` layout, the training worker, job record, API, and Models screen are agnostic to backend choice. In local mode, progress uses the full 0-100 range, cancellation is checked directly inside the Lightning callback instead of over HTTP, and the device comes from the payload.

```mermaid
flowchart LR
    W[Training worker] --> GB{TRAINING_MODE}
    GB -- local --> LB[LocalTrainingBackend<br/>torch/Lightning in-process]
    GB -- remote --> RB[RemoteTrainingBackend<br/>offload over HTTP]
    LB --> OUT[output_dir:<br/>checkpoint + logs + exports/]
    RB --> OUT
```

## End-to-end flow (remote)

`RemoteTrainingBackend.train()` runs three steps in one local progress bar (0-100%):

1. **Push snapshot (0-10%)** - `_push_snapshot()` creates a private dataset repo named `pais-snapshot-<uuid>` under `TRAINER_HF_NAMESPACE`, uploads the snapshot folder with an allowlist (`*.safetensors`, `*.json`, `*.txt`, `*.md`, `*.parquet`, `*.mp4`, `*.png`, `*.jpg`), and captures the concrete commit SHA. Missing SHA fails the job.
2. **Submit and wait (10-95%)** - `_submit_job()` POSTs `{payload, repo_id, revision, policy}` to `/jobs` and receives `remote_job_id`. `_wait_for_completion()` consumes trainer SSE from `GET /jobs/{id}/events`, maps trainer raw 0-100 into local 10-95, and mirrors `message` and `extra_info` (for example, step loss) into the local job. The trainer emits a `state` event on each change and closes the stream on terminal state. If the stream drops before terminal state (idle timeout or network blip), the backend reconnects with backoff, and the trainer re-emits current state on the new connection. If reconnects continue without receiving any event, the backend aborts the job instead of looping forever.
3. **Download and import (95-100%)** - `_download_and_extract()` streams `GET /jobs/{id}/artifact` to a temp zip, verifies byte count against `Content-Length`, validates the archive with `SafeZipArchive` (zip-bomb and path-traversal guards), and extracts into `output_dir`.

A `finally` block deletes the ephemeral repo regardless of outcome (best effort).

```mermaid
sequenceDiagram
    participant W as Training worker
    participant RB as RemoteTrainingBackend
    participant HF as HF Hub
    participant T as Trainer /jobs
    participant R as TrainerRunner

    W->>RB: train(context)
    RB->>HF: create_repo + upload_folder (allowlist)
    HF-->>RB: repo_id, commit SHA
    RB->>T: POST /jobs {payload, repo_id, revision, policy}
    T-->>RB: 202 {remote_job_id, queued}
    RB->>T: GET /jobs/{id}/events (SSE)
    loop state event per change until terminal
        T-->>RB: state {status, progress, message, extra_info}
        RB->>W: progress(10..95)
    end
    Note over T,R: queue dispatch -> runner.run()
    R->>HF: snapshot_download (pinned SHA, allowlist)
    R->>R: train -> export -> zip
    RB->>T: GET /jobs/{id}/artifact
    T-->>RB: model.zip (streamed)
    RB->>RB: verify + SafeZipArchive extract -> output_dir
    RB->>HF: delete_repo (finally)
```

## Trainer service

### HTTP API

| Method | Path                  | Purpose                                                                                    |
|--------|-----------------------|--------------------------------------------------------------------------------------------|
| `POST` | `/jobs`               | Enqueue a job; returns `remote_job_id` + `queued`.                              |
| `GET` | `/jobs/{id}`          | Current `JobState` (one-off query).                                                        |
| `GET` | `/jobs/{id}/events`   | SSE stream of state changes until terminal. Primary progress channel the backend consumes. |
| `GET` | `/jobs/{id}/artifact` | Download the model zip.                                                                    |
| `POST` | `/jobs/{id}/cancel`   | Cancel the job.                                                                            |
| `GET` | `/health`             | Liveness probe.                                                                            |
| `GET` | `/info`               | Trainer information for the UI.                                                            |

### Schemas

`SubmitJobRequest` validates untrusted input at the edge: `repo_id` against a conservative regex, `revision` as a 40-char hex SHA, and `policy` against an allowlist (`act`, `pi0`, `pi05`, `smolvla`). `TrainerJobStatus` is `queued | running | completed | failed | canceled`.

### Queue and dispatch

`QueueManager` owns `JobStore` and one asyncio `_dispatch_loop`. The loop takes the oldest queued job, marks it `running`, and runs it in a worker thread (training is blocking). Cancellation is cooperative through an in-memory `_cancel_requested` set checked by runner `should_stop`. On startup, `reset_orphans()` marks any job left `running` by a crashed process as `failed`.

### Persistence

One SQLite `jobs` table (`id`, `status`, `progress`, `message`, `extra_info`, `request`, `artifact`, `created_at`) makes the queue restart-safe. Access is serialized with a lock.

### Execution

`TrainerRunner.run()`:

1. `_pull_snapshot()` - `snapshot_download()` with pinned revision and the same format allowlist.
2. `_train()` - builds `LeRobotDataModule` from the snapshot and a `physicalai` Lightning `Trainer`, instantiates policy via `_setup_policy()`, and trains. A `ModelCheckpoint` keeps best `val/loss`; a progress callback reports `global_step / max_steps` and step loss, and sets `trainer.should_stop` on cancellation. `_resolve_device()` selects `xpu`, `cuda`, or `cpu`.
3. `_export_policy()` - exports to each backend the policy supports; missing optional dependencies or one failing backend are logged and are not fatal.
4. `_archive_model()` - zips the model directory (checkpoint, logs, `exports/`) for download.

## Progress mapping

The trainer reports raw 0-100 during training. The backend reserves both ends of its bar for transfer work and maps the remote value once:

```text
local 0–10   snapshot upload
local 10–95  remote training   = min(95, 10 + round(remote * 0.85))
local 95–100 model download + import
```

## Cancellation

Cancellation is cooperative across both services:

- The backend event loop checks `context.should_stop()` between SSE events (on each idle timeout and reconnect). On stop, it POSTs `/jobs/{id}/cancel` and raises `TrainingCanceledError`.
- The trainer adds the id to `_cancel_requested`. A queued job flips directly to `canceled`; a running job stops at the next safe point (Lightning callback sets `trainer.should_stop`, and the runner checks again after `fit`).
- Canceled jobs are logged at info level and do not dump error tracebacks, which keeps cancellation distinct from true failures.

## Configuration

| Studio backend (`application/backend/.env`) | Trainer service (`application/trainer/.env`) |
|---------------------------------------------|----------------------------------------------|
| `TRAINING_MODE=remote` | - |
| `TRAINER_URL` -> trainer address | `HOST` / `PORT` (default `0.0.0.0:8001`) |
| `TRAINER_HF_NAMESPACE` (snapshot target) | `HF_TOKEN` with **read** access |
| `HF_TOKEN` with **write** access | `STORAGE_DIR`, `TRAINER_MAX_CONCURRENT_JOBS`, `TRAINER_DEVICE` |

`Settings.validate_remote_training_config` fails fast at startup: `TRAINING_MODE=remote` without `TRAINER_URL` raises.

## Related docs

- [Remote Training Server](../08-remote-training-server.md) - operator setup guide.
- [Training Policies](../06-training-policies.md) - enable remote mode on the backend.
- [Hugging Face Integration](../../backend/docs/huggingface_integration.md) - required token permissions.
