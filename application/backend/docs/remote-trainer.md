# Physical AI Trainer

Remote training service for Physical AI Studio. Runs the heavy
torch/`physicalai` training stack on a GPU server so recording nodes stay
lightweight.

It lives in this project (`src/trainer/`) but is a separate entry point:
`physicalai-trainer` serves training jobs over HTTP, while `physicalai-studio`
serves the studio itself. Both call the same training code
(`training.run_training_job`), so a policy trains identically whether it
runs locally or here. Studio's own local training does not start this service —
it trains in-process.

## How it fits together

The studio backend delivers the dataset snapshot to this service by zipping it
and streaming it straight to `PUT /jobs/{id}/dataset` over HTTP, once a job is
submitted against a remote trainer registered in the Studio UI/API.

Then:

1. The service queues the job and trains, exports, and zips the model.
2. The backend polls progress, downloads the archive, and imports it as a model.
3. The service deletes the uploaded dataset once the job finishes.

## Install

```bash
cd application/backend
uv sync --extra cuda   # or --extra cpu / --extra xpu
```

The `cpu` and `cuda` extras include `executorch`, enabling the ExecuTorch
export backend. The `xpu` extra omits it: executorch conflicts with the xpu
torch build, so ExecuTorch export is skipped on xpu installs.

## Configure

Set environment variables, or copy `.env.example` to `.env` in
`application/backend/` and fill it in — the same file the studio backend
reads. The trainer's own settings use `TRAINER_`-prefixed names
(`TRAINER_HOST`, `TRAINER_PORT`, `TRAINER_STORAGE_DIR`) rather than the
studio's `HOST`/`PORT`/`STORAGE_DIR`, so one file configures both without the
trainer binding the studio's port or writing into the studio's storage
directory.

> [!WARNING]
> The trainer has no built-in authentication. Anyone who can reach its port can submit or cancel jobs and download model artifacts. Keep it on a private network that only the Physical AI Studio backend IP address can reach—never expose it to the internet.

> The backend honors `HTTP_PROXY` and `HTTPS_PROXY`. A configured proxy receives all trainer traffic, including model artifact downloads; anyone who controls these variables controls where artifacts go. Run the backend only on a trusted, non-shared, non-multi-tenant host where other users cannot set them.

| Variable                     | Required | Description                                  |
| ---------------------------- | -------- | -------------------------------------------- |
| `HF_TOKEN`                   | yes, if training a policy that downloads gated/private model weights | **Read** access to any gated/private model weights selected for training. |
| `TRAINER_STORAGE_DIR`        | no       | Working directory for jobs and artifacts.    |
| `TRAINER_MAX_CONCURRENT_JOBS`| no       | Queue concurrency (default 1).               |
| `TRAINER_MAX_UNCOMPRESSED_BYTES` | no   | Cap on an uploaded dataset's uncompressed size. |
| `TRAINER_MIN_FREE_BYTES`     | no       | Disk headroom kept free after extraction.    |
| `TRAINER_PORT`               | no       | Listen port (default 8001).                  |


## Run

```bash
uv run --no-sync physicalai-trainer   # loads .env, starts the service
```

`physicalai-trainer` loads `.env` and starts the service. It does
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

## Container images

Remote SSH provisioning uses the dedicated, non-root trainer images instead of
the Studio application images:

- `ghcr.io/open-edge-platform/physicalai-trainer-cuda:<git-sha>`
- `ghcr.io/open-edge-platform/physicalai-trainer-xpu:<git-sha>`

Each image contains only the `trainer` package, `physicalai-train`, and the
device-specific runtime dependencies. Although the trainer is built from this
project, the image copies `src/trainer/` alone: no Studio API, robot, database,
or UI code, and no datasets, model artifacts, SSH credentials, or Docker socket.
The entrypoint is `physicalai-trainer`; run it with a loopback-only port
publishing rule when it is provisioned remotely.

The image is built from
[`application/docker/Dockerfile.trainer`](../../docker/Dockerfile.trainer),
separately from the Studio application images in
`application/docker/Dockerfile`.

CI publishes a full Git-SHA tag, then attaches an SBOM and provenance
attestations, scans the immutable digest, signs it with keyless Sigstore, and
only then advances the moving `latest` tag. Use a SHA tag or resolved digest for
reproducible deployments; `latest` is a compatibility fallback only.

`GET /health` returns the image attributes that the Studio backend verifies
before work is accepted:

```json
{
  "status": "healthy",
  "protocol_version": 1,
  "device_type": "cuda",
  "build_revision": "<git-sha>",
  "build_date": "<RFC 3339 timestamp>",
  "application_version": "<version>"
}
```

### Run a container manually

Use manual startup only for administrator validation or the existing static
remote-trainer workflow. SSH-provisioned jobs will create a job-scoped
container, loopback port, and SSH tunnel automatically when that feature is
enabled. Do not expose the trainer port publicly: the service has no built-in
authentication.

[`application/docker/docker-compose.trainer.yaml`](../../docker/docker-compose.trainer.yaml)
wraps the `docker run` invocations below in a Docker Compose file with `cuda`
and `xpu` profiles, for administrators who prefer Compose over raw `docker
run`. It is standalone: it does not start the Studio backend/UI images from
`application/docker/docker-compose.yaml`.

```bash
cd application/docker
cp .env.trainer.example .env.trainer   # then edit REGISTRY/TRAINER_IMAGE_TAG
docker compose -f docker-compose.trainer.yaml --profile cuda up -d   # or --profile xpu
curl --fail --silent http://127.0.0.1:8001/health
docker compose -f docker-compose.trainer.yaml --profile cuda down   # add -v to also drop the data volume
```

Set `TRAINER_IMAGE_TAG` in `.env.trainer` to an immutable Git-SHA tag or
resolved digest before using this in anything but a throwaway environment;
`latest` is a compatibility fallback only. The XPU profile also needs
`RENDER_NODE` and `RENDER_GID` set to the host's Intel GPU render node (see
`.env.trainer.example`).

The examples below use a Docker-managed volume so the image's non-root
`trainer` user can persist its queue, uploaded datasets, and artifacts
without host ownership changes. Remove the volume after you no longer need
its contents.

```bash
docker volume create physicalai-trainer-data
```

Replace `<image-reference>` with an immutable Git-SHA tag or a resolved digest.
Use `latest` only when explicitly accepting the compatibility fallback.

PyTorch data loaders can exhaust Docker's default 64 MB `/dev/shm` allocation
during larger training jobs. On a trusted single-tenant host, prefer the host's
shared-memory pool with `--ipc=host` (or `ipc: host` in Docker Compose). If you
need an isolated limit instead, set an explicit shared-memory size such as
`--shm-size=16g` (or `shm_size: 16g` in Docker Compose).

#### CUDA

The Docker host needs an NVIDIA driver and NVIDIA Container Toolkit. Verify the
host can grant GPU access with a disposable CUDA container before starting the
trainer.

```bash
docker run --rm --gpus all \
  nvidia/cuda:12.8.0-base-ubuntu24.04 \
  nvidia-smi
```

Start the trainer with GPU access and a loopback-only port binding:

```bash
docker run --rm \
  --name physicalai-trainer \
  --gpus all \
  --ipc=host \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=1g \
  -p 127.0.0.1:8001:8001 \
  -v physicalai-trainer-data:/var/lib/physicalai-trainer \
  ghcr.io/open-edge-platform/physicalai-trainer-cuda:<image-reference>
```

#### XPU

The Docker host needs a supported Intel GPU driver and an accessible
`/dev/dri/renderD*` device. Set `RENDER_NODE` to the render node on your host;
the numeric group is passed through so the non-root trainer user can access it.

```bash
export RENDER_NODE=/dev/dri/renderD128
test -c "$RENDER_NODE"
export RENDER_GID="$(stat -c '%g' "$RENDER_NODE")"

docker run --rm \
  --name physicalai-trainer \
  --device /dev/dri:/dev/dri \
  --group-add "$RENDER_GID" \
  --ipc=host \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=1g \
  -p 127.0.0.1:8001:8001 \
  -v physicalai-trainer-data:/var/lib/physicalai-trainer \
  ghcr.io/open-edge-platform/physicalai-trainer-xpu:<image-reference>
```

Do not use `--privileged` or mount the Docker socket for either image.

#### Optional Hugging Face token

Set `HF_TOKEN` only when you train a policy that downloads gated/private
model weights. Place a read-only token in a host file with restrictive
permissions and pass it as an environment file:

```bash
printf 'HF_TOKEN=<read-only-token>\n' > trainer.env
chmod 600 trainer.env
```

Add this option to the appropriate `docker run` command, before the image
reference:

```bash
--env-file ./trainer.env
```

Never commit `trainer.env` or put a token directly in shell history.

#### Verify and stop

Verify the running image before pointing Studio at it. The reported device,
protocol, and build revision must match the image you selected.

```bash
curl --fail --silent http://127.0.0.1:8001/health
```

Stop a manually managed trainer and remove its retained data when it is no
longer needed:

```bash
docker stop physicalai-trainer
docker volume rm physicalai-trainer-data
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
| GET    | `/health`              | Liveness and image/protocol metadata. |

## Security

- HTTP-uploaded datasets are validated before extraction: ZIP-only, size and
  file-count caps, disk-headroom check, and per-entry path containment (no
  traversal, symlinks, or nested archives).
