# Physical AI Trainer

Standalone remote training service for Physical AI Studio. Runs the heavy
torch/`physicalai` training stack on a GPU server so recording nodes stay
lightweight.

## How it fits together

1. The studio backend (`TRAINING_MODE=remote`) pushes the dataset snapshot to an
   ephemeral private HuggingFace dataset repo and submits a job here.
2. This service queues the job, pulls the snapshot at a pinned commit SHA,
   trains, exports, and zips the model.
3. The backend polls progress, downloads the archive, imports it as a model,
   then deletes the ephemeral repo.

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

| Variable                     | Required | Description                                  |
| ---------------------------- | -------- | -------------------------------------------- |
| `HF_TOKEN`                   | yes      | **Read** access to the snapshot repos. The Studio backend that pushes them needs **write** access. See [token permissions](../backend/docs/huggingface_integration.md#required-token-permissions). |
| `STORAGE_DIR`                | no       | Working directory for jobs and artifacts.    |
| `TRAINER_MAX_CONCURRENT_JOBS`| no       | Queue concurrency (default 1).               |
| `TRAINER_DEVICE`             | no       | Force `cuda`/`xpu`/`cpu` (auto if unset).    |
| `PORT`                       | no       | Listen port (default 8001).                  |

Never commit `HF_TOKEN`. Store it in a secret manager or local `.env`.

## Run

```bash
uv run python -m trainer.main
```

## API

| Method | Path                   | Purpose                          |
| ------ | ---------------------- | -------------------------------- |
| POST   | `/jobs`                | Enqueue a training job.          |
| GET    | `/jobs/{id}`           | Current job state.               |
| GET    | `/jobs/{id}/events`    | SSE stream of state changes.     |
| GET    | `/jobs/{id}/artifact`  | Download the model archive.      |
| POST   | `/jobs/{id}/cancel`    | Cancel a queued or running job.  |
| GET    | `/health`              | Liveness probe.                  |

## Security

- Snapshots are pulled at a pinned commit SHA with a format allowlist
  (`*.safetensors`, `*.json`, `*.txt`, `*.md`, `*.parquet`, `*.mp4`, `*.png`, `*.jpg`).
- `repo_id` and `revision` are strictly validated before any Hub call.
- `HF_TOKEN` is read from the environment and never logged.
