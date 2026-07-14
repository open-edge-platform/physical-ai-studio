# Remote Training Server

The remote training server (the **trainer service**) runs the heavy training stack on a GPU machine so your recording stations stay lightweight. This guide sets up the service, connects it to Physical AI Studio, and covers its configuration, queueing behavior, and API.

Read [Training Policies](./06-training-policies.md) first to enable remote mode on the Studio backend. This guide covers the server side.

## When to use it

Run the trainer service when you record datasets on a machine without a capable GPU, or when you want one GPU server to handle training for several recording stations. The service trains a model policy, exports it to every supported backend, and returns the finished model to Studio over HTTP.

The service queues jobs and runs them one at a time by default, so several recording stations can submit jobs to a single GPU server without conflict.

## How it works

1. You start training from the Studio Models screen as usual.
2. The Studio backend submits a job and streams the dataset snapshot to the trainer service over HTTP (the default transfer mode).
3. The trainer service queues the job, trains the model policy, and exports it.
4. The Studio backend downloads the finished model and imports it as a model.

Set `TRAINER_DATASET_TRANSFER=hf` on the Studio backend only when you need Hugging Face Hub transfer. In that mode, the backend pushes the snapshot to a temporary private dataset repository, pins its commit, the trainer pulls that commit, and the backend deletes the repository after import.

Studio mirrors the trainer's progress into the same training job, so the Models screen looks identical to local training.

## Prerequisites

- A GPU machine (NVIDIA CUDA or Intel XPU) reachable from the Studio backend at the URL you set as `TRAINER_URL`. CPU works for testing but is impractical for real training.
- For the optional `TRAINER_DATASET_TRANSFER=hf` mode, a Hugging Face token (`HF_TOKEN`) with **read** access to pull dataset snapshots. The Studio backend needs a token with **write** access (create, upload, delete) to manage the temporary snapshot repos — see [Hugging Face Integration](../backend/docs/huggingface_integration.md#required-token-permissions). The default HTTP transfer does not need a token for the dataset transfer.
- [uv](https://docs.astral.sh/uv/) installed on the GPU machine.

## Install

Install the trainer service from `application/trainer/`. Choose the extra that matches your hardware:

```bash
cd application/trainer
uv sync --extra cuda   # NVIDIA GPU
# uv sync --extra xpu  # Intel GPU
# uv sync --extra cpu  # CPU (testing only)
```

## Configure

Set these environment variables on the trainer service, for example in `application/trainer/.env`:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_TOKEN` | only for `hf` transfer or Hub-backed policy assets | — | Hugging Face token with **read** access to dataset snapshots for `hf` transfer. See [token permissions](../backend/docs/huggingface_integration.md#required-token-permissions). |
| `STORAGE_DIR` | no | platform default | Working directory for snapshots, checkpoints, and model archives. |
| `TRAINER_MAX_CONCURRENT_JOBS` | no | `1` | Number of jobs to run at once. `1` keeps a single GPU job at a time. |
| `HOST` | no | `0.0.0.0` | Bind address. |
| `PORT` | no | `8001` | Listen port. |

Example `.env`:

```env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TRAINER_MAX_CONCURRENT_JOBS=1
```

## Run

Start the service from `application/trainer/`:

```bash
uv run --extra <xpu|cuda> physicalai-trainer
```

The service listens on `PORT` (default `8001`). Confirm it is up:

```bash
curl http://localhost:8001/health
# {"status":"healthy"}
```

Set the Studio backend's `TRAINER_URL` to this service's reachable address, for example `http://trainer.internal:8001`. Use HTTPS only when a reverse proxy or TLS terminator serves the trainer.

## Configuration contract

Point the Studio backend at the trainer service, and give both sides a Hugging Face token.

| Studio backend (`application/backend/.env`) | Trainer service (`application/trainer/.env`) |
|---------------------------------------------|----------------------------------------------|
| `TRAINING_MODE=remote` | — |
| `TRAINER_URL` → trainer service address | `HOST` / `PORT` |
| `TRAINER_DATASET_TRANSFER=http` (default) | — |
| `TRAINER_DATASET_TRANSFER=hf`, `TRAINER_HF_NAMESPACE` (where snapshots are pushed) | `HF_TOKEN` (reads those snapshots) |
| `HF_TOKEN` with **write** access for `hf` transfer | `HF_TOKEN` with **read** access for `hf` transfer |

## Queueing and concurrency

The trainer service keeps a persistent job queue backed by SQLite, so jobs survive a restart. Jobs run in submission order. `TRAINER_MAX_CONCURRENT_JOBS` caps how many run at once (default `1`). Increase it only if the GPU has enough memory for parallel jobs.

You can cancel a queued or running job. Cancellation is cooperative: a running job stops at the next safe point.

## API

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/jobs` | Enqueue a training job. |
| `PUT` | `/jobs/{id}/dataset` | Upload a dataset ZIP for the default HTTP transfer. |
| `HEAD` | `/jobs/{id}/dataset` | Read the resumable HTTP-upload offset. |
| `GET` | `/jobs/{id}` | Read a job's current state. |
| `GET` | `/jobs/{id}/events` | Stream job progress as server-sent events. |
| `GET` | `/jobs/{id}/artifact` | Download the trained model archive. |
| `POST` | `/jobs/{id}/cancel` | Cancel a queued or running job. |
| `GET` | `/devices` | Report the trainer's training devices (CPU, Intel XPU, NVIDIA CUDA). |
| `GET` | `/health` | Liveness probe. |

The Studio backend drives these routes for you. Call them directly only for diagnostics.

## Security

- HTTP-uploaded snapshots are validated before extraction for ZIP safety, disk space, file count, and path traversal. The trainer removes the uploaded copy when the job finishes.
- With `hf` transfer, it pulls each dataset snapshot from a pinned commit, accepts only an allowlist of safe file formats, and validates the dataset repository id and commit before any Hub call.
- It reads `HF_TOKEN` from the environment and never logs it.
- The trainer has no built-in authentication. Restrict its port to the Studio backend on a private network; do not expose it publicly.

Never commit `HF_TOKEN`. Store it in local `.env` files or your secret manager, and rotate it immediately if exposed.

## Troubleshooting

- **Backend fails to start in remote mode**: `TRAINING_MODE=remote` requires `TRAINER_URL`. Set it.
- **HTTP dataset upload fails**: confirm the trainer is reachable from the backend and that `STORAGE_DIR` has enough free space for the uploaded ZIP and its extracted contents.
- **HF snapshot upload fails**: ensure `TRAINER_DATASET_TRANSFER=hf`, the Studio backend's `HF_TOKEN` has **write** access, and `TRAINER_HF_NAMESPACE` is writable by that token.
- **Trainer cannot pull an HF snapshot**: the trainer's `HF_TOKEN` lacks **read** access to the pushed repository.
- **Job stays queued**: another job is running and `TRAINER_MAX_CONCURRENT_JOBS` is `1`. Wait for it to finish or raise the limit if the GPU allows.
- **Slow start on large datasets**: the dataset upload runs before training and scales with dataset size. Watch the early progress step on the Models screen; the job continues once the upload completes.

## Next

- Configure training and tokens: [Training Policies](./06-training-policies.md).
- Run/deploy in UI: [Deploying Model Policies](./07-deploying-model-policies.md).
