# Physical AI Trainer

Standalone remote training service for Physical AI Studio. Runs the heavy
torch/`physicalai` training stack on a GPU server so recording nodes stay
lightweight.

## How it fits together

The studio backend (`TRAINING_MODE=remote`) delivers the dataset snapshot to
this service by zipping it and streaming it straight to `PUT /jobs/{id}/dataset`
over HTTP.

Then:

1. The service queues the job and trains, exports, and zips the model.
2. The backend polls progress, downloads the archive, and imports it as a model.
3. The service deletes the uploaded dataset once the job finishes.

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

> [!WARNING]
> The trainer has no built-in authentication. Anyone who can reach its port can submit or cancel jobs and download model artifacts. Keep it on a private network that only the Physical AI Studio backend IP address can reach—never expose it to the internet.

> The backend honors `HTTP_PROXY` and `HTTPS_PROXY`. A configured proxy receives all trainer traffic, including model artifact downloads; anyone who controls these variables controls where artifacts go. Run the backend only on a trusted, non-shared, non-multi-tenant host where other users cannot set them.

| Variable                     | Required | Description                                  |
| ---------------------------- | -------- | -------------------------------------------- |
| `STORAGE_DIR`                | no       | Working directory for jobs and artifacts.    |
| `TRAINER_MAX_CONCURRENT_JOBS`| no       | Queue concurrency (default 1).               |
| `TRAINER_MAX_UNCOMPRESSED_BYTES` | no   | Cap on an uploaded dataset's uncompressed size. |
| `TRAINER_MIN_FREE_BYTES`     | no       | Disk headroom kept free after extraction.    |
| `PORT`                       | no       | Listen port (default 8001).                  |


## Run

```bash
uv run --no-sync physicalai-trainer   # loads .env, starts the service
```

`physicalai-trainer` loads the trainer `.env` and starts the service. It does
not install dependencies itself, so run `uv sync --extra <cpu|cuda|xpu>` first
(see [Install](#install)) to pull in the matching torch build.

Use `--no-sync` so the run reuses that install. A plain `uv run` triggers an
implicit sync that ignores the hardware extra and can re-resolve `torch` from
the default index, clobbering your `cuda`/`xpu` build. If you prefer not to pass
the flag every time, either export `UV_NO_SYNC=1`, or repeat the extra on the
run command so the resolution matches:

```bash
uv run --extra cuda physicalai-trainer   # or --extra xpu / --extra cpu
```

Override the bind address with flags:

```bash
uv run --no-sync physicalai-trainer --host 0.0.0.0 --port 8001
```

To run the ASGI app module directly:

```bash
uv run --no-sync python -m trainer.main
```

## API

| Method | Path                   | Purpose                          |
| ------ | ---------------------- | -------------------------------- |
| POST   | `/jobs`                | Enqueue a training job.          |
| PUT    | `/jobs/{id}/dataset`   | Upload the dataset ZIP.          |
| GET    | `/jobs/{id}`           | Current job state.               |
| GET    | `/jobs/{id}/events`    | SSE stream of state changes.     |
| GET    | `/jobs/{id}/artifact`  | Download the model archive.      |
| POST   | `/jobs/{id}/cancel`    | Cancel a queued or running job.  |
| GET    | `/health`              | Liveness probe.                  |

## Security

- HTTP-uploaded datasets are validated before extraction: ZIP-only, size and
  file-count caps, disk-headroom check, and per-entry path containment (no
  traversal, symlinks, or nested archives).
