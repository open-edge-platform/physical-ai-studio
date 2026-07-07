# Physical AI Trainer

Standalone remote training service for Physical AI Studio. Runs the heavy
torch/`physicalai` training stack on a GPU server so recording nodes stay
lightweight.

## How it fits together

The studio backend (`TRAINING_MODE=remote`) delivers the dataset snapshot to
this service one of two ways, set by `TRAINER_DATASET_TRANSFER`:

- **`http` (default)** — the backend zips the snapshot and streams it straight
  to `PUT /jobs/{id}/dataset`. No external services or `HF_TOKEN` required.
- **`hf`** — the backend pushes the snapshot to an ephemeral private
  HuggingFace dataset repo; this service pulls it at a pinned commit SHA.

Then, regardless of transfer:

1. The service queues the job and trains, exports, and zips the model.
2. The backend polls progress, downloads the archive, and imports it as a model.
3. For `hf` transfer the backend deletes the ephemeral repo; for `http` the
   service deletes the uploaded dataset once the job finishes.

## Install

```bash
cd application/trainer
uv sync --extra cuda   # or --extra cpu / --extra xpu
```

The `cpu` and `cuda` extras include `executorch`, enabling the ExecuTorch
export backend. The `xpu` extra omits it: executorch conflicts with the xpu
torch build, so ExecuTorch export is skipped on xpu installs.

## Configure

Set environment variables (or an `.env` file):

> [!IMPORTANT]
> The trainer service has no built-in authentication. Any machine that can reach its port can submit jobs, cancel jobs, and download trained model artifacts.
> Deploy it on a private network reachable only by the Physical AI Studio backend IP address – never expose this port to the internet.

> [!WARNING]
> The Physical AI Studio backend inherits proxy settings (`HTTP_PROXY` / `HTTPS_PROXY`) from its host environment. When those variables are set, all trainer communication (including model artifact download) routes through the configured proxy, so whoever controls those variables controls where model artifacts flow. Only deploy the backend in a fully trusted environment where proxy variables cannot be set by other users. Do not run it on a shared or multi-tenant host.

| Variable                     | Required | Description                                  |
| ---------------------------- | -------- | -------------------------------------------- |
| `HF_TOKEN`                   | hf transfer only | **Read** access to the snapshot repos. The Studio backend that pushes them needs **write** access. See [token permissions](../backend/docs/huggingface_integration.md#required-token-permissions). Unused for the default `http` transfer. |
| `STORAGE_DIR`                | no       | Working directory for jobs and artifacts.    |
| `TRAINER_MAX_CONCURRENT_JOBS`| no       | Queue concurrency (default 1).               |
| `TRAINER_DEVICE`             | no       | Force `cuda`/`xpu`/`cpu` (auto if unset).    |
| `TRAINER_MAX_UNCOMPRESSED_BYTES` | no   | Cap on an uploaded dataset's uncompressed size (http transfer). |
| `TRAINER_MIN_FREE_BYTES`     | no       | Disk headroom kept free after extraction (http transfer). |
| `PORT`                       | no       | Listen port (default 8001).                  |

Never commit `HF_TOKEN`. Store it in a secret manager or local `.env`.

## Run

```bash
uv run --no-sync physicalai-trainer   # loads .env, runs `uv sync`, starts the service
```

`physicalai-trainer` loads the trainer `.env`, syncs dependencies for the
selected hardware, and starts the service. Control the hardware extra and
dependency sync via flags or environment variables:

```bash
DEVICE=cuda physicalai-trainer          # or: physicalai-trainer --device cuda
SYNC=false physicalai-trainer           # or: physicalai-trainer --no-sync (skip `uv sync`)
```

To skip the launcher and start the ASGI app directly (assumes deps are synced):

```bash
uv run python -m trainer.main
```

## API

| Method | Path                   | Purpose                          |
| ------ | ---------------------- | -------------------------------- |
| POST   | `/jobs`                | Enqueue a training job.          |
| PUT    | `/jobs/{id}/dataset`   | Upload the dataset ZIP (http transfer). |
| GET    | `/jobs/{id}`           | Current job state.               |
| GET    | `/jobs/{id}/events`    | SSE stream of state changes.     |
| GET    | `/jobs/{id}/artifact`  | Download the model archive.      |
| POST   | `/jobs/{id}/cancel`    | Cancel a queued or running job.  |
| GET    | `/health`              | Liveness probe.                  |

## Security

- HTTP-uploaded datasets are validated before extraction: ZIP-only, size and
  file-count caps, disk-headroom check, and per-entry path containment (no
  traversal, symlinks, or nested archives).
- HF snapshots are pulled at a pinned commit SHA with a format allowlist
  (`*.safetensors`, `*.json`, `*.txt`, `*.md`, `*.parquet`, `*.mp4`, `*.png`, `*.jpg`).
- `repo_id` and `revision` are strictly validated before any Hub call.
- `HF_TOKEN` is read from the environment and never logged.
