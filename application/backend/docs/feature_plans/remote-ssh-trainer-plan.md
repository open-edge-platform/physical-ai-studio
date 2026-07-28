# Remote Server SSH Container Provisioning for Training

## Overview

Add a managed "remote server" concept to Physical AI Studio. Users register a
remote GPU/XPU server with SSH credentials and a target device type through the
web UI. When a training job is started, the backend SSHes into the selected
server, resolves the device-specific trainer image from the local Git SHA (or
`latest` when the SHA-tagged image cannot be resolved), starts one isolated
trainer container for the job, runs the job through the existing HTTP
`RemoteTrainingBackend`, and removes the container when the job completes.

This extends today's per-job remote trainer selection into per-job,
dynamically provisioned trainers.

### Relationship to the existing remote trainer registry

Backend selection today is entirely per-job: `TrainJobPayload.training_target`
is `local` or `remote`; for `remote` jobs, `remote_trainer_id` refers to a
`RemoteTrainer` record (name + URL) registered through the Studio UI/API, and
`JobService.submit_train_job` resolves it to a pinned `remote_trainer_url` on
the payload. `get_training_backend(payload)` reads only that payload.
SSH provisioning is a **third kind of per-job target**, not a new global mode:

- **Selector precedence:** an SSH job is discriminated by
  `training_target is TrainingTarget.SSH` (a new enum member — see Step 5) with
  `remote_server_id` set; otherwise `training_target is REMOTE` uses the
  existing direct-URL registry and its pinned `remote_trainer_url`; otherwise
  training runs locally. This precedence lives in `get_training_backend(...)`
  (see Step 5) and is documented next to the schema fields. Do not discriminate
  purely on "`remote_server_id` is not None" — the enum must stay the single
  source of truth so the payload cannot express two targets at once.
- **No new required settings:** SSH provisioning must not require any global
  setting. It is configured entirely through the `RemoteServerDB` record
  referenced by `remote_server_id`, mirroring how `remote_trainer_id` already
  works for the direct-URL registry.
- **Device listing:** `SystemService.get_available_training_devices` already
  always reports the Studio host's local devices (`mode="local"`); per-trainer
  device listing for the direct-URL registry goes through
  `RemoteTrainerService`'s health check, not this endpoint. The SSH path
  follows the same pattern: it serves the server's **configured** `device_type`
  from the DB record (the trainer is not running at dialog time; see Step 6)
  through the remote-server status endpoint (Step 1), not a live `/devices`
  probe.

## Confirmed Decisions

| Topic                           | Decision                                                                                                                                                                                                                                                                                                                                               |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Secret storage                  | Fernet symmetric encryption at rest; key from an env var.                                                                                                                                                                                                                                                                                              |
| Encrypted (confidential) fields | Fernet-encrypt only the SSH auth material: `ssh_secret_encrypted` (private key for `auth_type=key`, or password for `auth_type=password`) and `ssh_key_passphrase_encrypted` (nullable, key auth only).                                                                                                                                                |
| Non-encrypted, never-serialized | `host_key` is integrity data (the server's public host key for TOFU), not a secret: stored in plaintext, but never returned in API responses.                                                                                                                                                                                                          |
| Secret key lifecycle            | Fernet key lives in an env var (e.g. `REMOTE_SERVER_SECRET_KEY`), never in the DB. Losing/rotating the key makes stored secrets undecryptable, so document that rotating the key requires re-entering each server's secret.                                                                                                                            |
| Auth material                   | `auth_type=key` supports passphrase-protected private keys via an optional encrypted `ssh_key_passphrase`; `auth_type=password` stores the encrypted password.                                                                                                                                                                                         |
| Host key verification           | Pin the server host key on first successful preflight (TOFU), store it with the record, and verify it (fail-closed) on every later connect. A mismatch blocks the job/preflight with a clear error.                                                                                                                                                    |
| Command safety                  | All remote commands run as argument arrays (no shell string interpolation). Image names, container names, labels, and device arguments come from trusted application constants or validated identifiers, not arbitrary user input.                                                                                                                     |
| Trainer distribution            | Publish dedicated `physicalai-trainer-cuda` and `physicalai-trainer-xpu` OCI images. Do not reuse the full Studio application images as trainer images.                                                                                                                                                                                                |
| Trainer launch                  | Prefer the device-specific image tagged with the local Git SHA; use `latest` only when that SHA-tagged image cannot be resolved. Resolve and record the selected image's immutable digest, then run `physicalai-trainer` in a job-scoped Docker container bound only to remote loopback.                                                               |
| Trainer lifecycle               | One container per job. Persist the container id/name, image digest, remote published port, and local tunnel port so orphans can be swept after a crash.                                                                                                                                                                                                |
| Concurrency                     | Reuse the existing **per-execution-target** serialization in `TrainingWorker.run_loop`: one job at a time per target, jobs on distinct targets run concurrently. SSH jobs need their own target key (next row). Throttle status/preflight SSH connections per server with short timeouts so UI polling cannot disrupt a running job.                   |
| Execution target key            | An SSH job's target key must be `ssh:<remote_server_id>`. Reusing the existing `remote:<remote_trainer_id>` branch is **not** acceptable: SSH jobs carry no `remote_trainer_id`, so every SSH job on every server would collapse onto the single key `remote:None`. `TrainingWorker._target_key` must be extended explicitly.                          |
| Job target discriminator        | Add a third `TrainingTarget.SSH` member rather than overloading `REMOTE` with an optional `remote_server_id`. Overloading `REMOTE` breaks `get_training_backend` (raises when `remote_trainer_url is None`) and the worker's `reattaching` check.                                                                                                      |
| Backend restart behavior        | **Reattach.** On startup, for each non-terminal job with a `JobProvisioningDB` row, re-open the SSH tunnel to the persisted `remote_port`/`container_id`, re-verify `/health` and image digest, and resume streaming. Only genuinely orphaned containers are swept. See [Restart and Reattach](#restart-and-reattach).                                 |
| Tunnel drop mid-training        | **Reconnect and resume.** A dropped tunnel does not fail the job. Use SSH keepalives, re-open the forward against the still-running container, and resume streaming. Consistent with the reattach decision above.                                                                                                                                      |
| Bastion / `ProxyJump`           | **Out of scope.** All target servers are directly reachable from where Studio runs. Document the limitation; do not add a hop config field.                                                                                                                                                                                                            |
| Build revision source           | Read the Studio build revision from a baked-in OCI label / build arg / `../../../VERSION`, **not** `git rev-parse HEAD`. The backend ships inside a container with no `.git`, so a git-only lookup would silently make `latest` the permanent production path. Git is a developer-mode fallback only.                                               |
| Dataset transfer                | Keep the HTTP `http` transfer, streamed through an SSH tunnel.                                                                                                                                                                                                                                                                                         |
| Device type                     | User provides GPU/XPU device type when configuring the server.                                                                                                                                                                                                                                                                                         |
| Image selection                 | Resolve the build revision, then resolve the corresponding device-specific image tag. Fall back to `latest` only if the revision or its image cannot be resolved. Persist the selected ref, fallback reason, and immutable digest with the job.                                                                                                        |
| Trainer protocol compatibility  | **Grandfather the direct-URL registry, strict for SSH.** A direct-URL trainer reporting no protocol version is allowed (a human registered and owns it). An SSH-provisioned image must report compatible metadata or the job fails before dataset upload (Studio selected that image itself).                                                          |
| Registry access                 | Pull public trainer images from GHCR. Registry credentials and private registries are outside the initial scope.                                                                                                                                                                                                                                       |
| First-job cost                  | The first image pull can be large; later jobs reuse Docker's cached layers.                                                                                                                                                                                                                                                                            |
| Driver check                    | CUDA: `nvidia-smi` plus an in-container `torch.cuda.is_available()` probe. XPU: host render-node/driver probes plus an in-container `torch.xpu.is_available()` probe.                                                                                                                                                                                  |
| Preflight                       | Two tiers. Tier 1 (cheap) gates save: reachability/auth, host key, Docker access, disk, driver. Tier 2 (expensive) runs as an explicit async action: registry/image pull, signature policy, in-container device probe, protocol compatibility. See Step 2.                                                                                             |
| Supply chain                    | Publish an SBOM, scan and sign trainer images, and verify the expected image identity before launch. Always launch the selected image by its resolved digest.                                                                                                                                                                                          |
| Management UI                   | **One global "Training targets" screen**, outside the project subtree, listing local, direct-URL trainers, and SSH servers with a type badge and status. The train dialog uses a single unified target picker backed by the same concept — not two parallel dropdowns. See Steps 6 and 7.                                                              |
| User-facing naming              | **Unify.** Users pick a "training target"; the direct-URL vs SSH distinction is a type badge, not a separate product concept. Internally the models stay `RemoteTrainerDB` (direct URL) and `RemoteServerDB` (SSH).                                                                                                                                    |
| GPU availability                | Before launching training, check the server GPU is free. CUDA: reliable via `nvidia-smi` compute-apps + memory. XPU: best-effort via `xpu-smi stats` / memory heuristic. If occupied, the job **stays pending with backoff** in a visible "waiting for GPU" state and a give-up timeout — it is not failed immediately. See Step 4.                    |
| Backend authorization           | **None today — single trusted local user.** Anyone who can reach the API can register a server and submit a job, which grants root-equivalent execution on that host. Therefore remote SSH training ships **feature-flagged off by default**, and the trust assumption is stated plainly in the docs. See Step 8.                                      |
| Progress reporting              | Phase-windowed 0–100 bar plus a structured `phase` descriptor in `extra_info` so the UI shows a stepper (connect → image pull → verify → start → upload → train → download). Indeterminate setup phases stream sanitized command output + heartbeats. **One phase table for all targets** — the ~1–2% shift for local and direct-URL jobs is accepted. |
| Fernet key fingerprint          | **Store a key fingerprint/version alongside each ciphertext** so records encrypted under a lost or rotated key are identifiable in the UI before provisioning fails. Additive follow-up migration on `remote_servers`.                                                                                                                                 |

## Architecture Context

- Training runs through the `TrainingBackend` abstraction
  (`services/training_backends/`). `LocalTrainingBackend` trains in-process;
  `RemoteTrainingBackend` offloads to a trainer service over HTTP at a URL
  pinned per job (`TrainJobPayload.remote_trainer_url`), resolved from the
  `RemoteTrainer` registry at submission time.
- The backend `TrainingWorker.run_loop` reserves one job per **execution
  target** (`local`, or `remote:<remote_trainer_id>`) and runs jobs on distinct
  targets concurrently as asyncio tasks; only jobs competing for the same
  target are serialized. Each running job gets its own per-job interrupt flag
  (keyed by job id in a shared dict) so cancelling one job cannot affect
  another running concurrently on a different target.
- The trainer service (`../../../trainer`) is a FastAPI app exposing
  `/jobs`, `/jobs/{id}/dataset`, `/jobs/{id}/events` (SSE), `/jobs/{id}/artifact`,
  `/jobs/{id}/cancel`, `/devices`, and `/health`. It has no built-in auth and is
  intended for a trusted private network.
- The existing `http` dataset transfer streams a validated ZIP via
  `PUT /jobs/{id}/dataset` with progress mirroring and archive-safety checks.
- Persistence uses SQLAlchemy models in `db/schema.py`, repositories under
  `repositories/`, Pydantic schemas under `schemas/`, and Alembic migrations.
- The existing `../../../docker/Dockerfile` produces full Studio CPU/XPU/CUDA
  images whose runtime installs the backend environment and starts `run.sh`.
  Add minimal trainer-specific image targets instead of using those images as-is.

## Implementation Steps

### 1. Persist servers with encrypted secrets — **Done**

Implemented on `albert/ssh-server-persistence`. Recorded here as the shipped
shape, not as remaining work.

- `RemoteServerDB` in `../../src/db/schema.py` with `id`, `name`,
  `host`, `port`, `username`, `auth_type` (`SSHAuthType`), `device_type`
  (`DeviceType`), `created_at`, `updated_at`, plus:
  - **Fernet-encrypted (confidential):** `ssh_secret_encrypted` (private key or
    password), `ssh_key_passphrase_encrypted` (nullable, key auth only).
  - **Plaintext but never serialized:** `host_key` (pinned public host key for
    TOFU verification — integrity data, not a secret, so no encryption needed).
  - **Last-check summary:** `last_check_status`, `last_check_at`,
    `last_check_latency_ms`, `last_check_reason_code`. These exist so a
    transient preflight failure updates status instead of destroying the record.
  - A `uq_remote_servers_host_port_username` unique constraint prevents
    duplicate registrations of the same endpoint.
- `JobProvisioningDB` (separate table keyed by `job_id`, **not** the job payload
  JSON) holds per-job provisioning state so a crashed backend can sweep or
  reclaim an orphaned container from durable, queryable columns: `image_ref`,
  `image_fallback_reason`, `image_digest`, `container_id`, `container_name`,
  `remote_port`, `local_tunnel_port`, `trainer_build_version`,
  `trainer_protocol_version`.
- Alembic migration `d4f8a1c9b3e6_add_remote_servers.py`.
- `repositories/remote_server_repo.py`, `repositories/job_provisioning_repo.py`,
  and mappers under `repositories/mappers/`.
- `schemas/remote_server.py`, `schemas/job_provisioning.py` — the encrypted
  fields and `host_key` are never serialized; asserted by
  `tests/schemas/test_remote_server.py`.
- `services/remote_server_service.py` and the CRUD router `api/remote_servers.py`.
- `REMOTE_SERVER_SECRET_KEY` in `settings.py` (`remote_server_secret_key`,
  default `None`) with lazy, fail-closed cipher construction in
  `core/secret_encryption.py` (`RemoteServerSecretKeyMissingError`).

**Remaining follow-ups on this step:**

- **Store a Fernet key fingerprint/version alongside each ciphertext** (agreed).
  Add an additive migration on `remote_servers` carrying, for example, a short
  non-reversible digest of the active key. On read, compare it to the currently
  configured key so records encrypted under a lost or rotated key are
  _identifiable up front_ and can be flagged as "secret needs re-entry" in the
  list UI, instead of failing opaquely at provisioning time. The fingerprint must
  not weaken the key: store a truncated hash of the key, never the key itself.
- Define the "key not configured" UX: `RemoteServerSecretKeyMissingError` must
  surface as an actionable "set `REMOTE_SERVER_SECRET_KEY`" message on the
  training-targets screen, not a 500.
- The status endpoint (e.g. `POST /api/remote-servers/{id}:check` and/or
  `GET /api/remote-servers/{id}/status`) still needs to run the Step 2 preflight
  and return a structured result the UI can render (reachable, authenticated,
  Docker usable, registry reachable, driver present/version, container device
  probe result, image/protocol version, last-checked timestamp, and whether the
  server is currently in use by a running job or waiting on a busy GPU).

### 2. Verify-on-save SSH preflight

Split preflight into two tiers. A multi-GB registry pull plus a one-shot GPU
container must not run inside a create/update request handler.

**Tier 1 — cheap checks, gate the save (seconds, bounded timeout):**

- reachability and authentication,
- **host key**: on the first successful preflight, pin (store) the presented
  host key on the record (TOFU); on later connects, verify it and fail-closed
  on mismatch,
- `docker version` succeeds for the configured SSH user without privilege
  escalation,
- enough free disk space for the image and a nominal job (per-job dataset size
  is re-checked at provisioning time — see Step 4),
- device driver matching the configured type:
  - **CUDA** — `nvidia-smi`.
  - **XPU** — try `xpu-smi` first. It is not always installed, so fall back
    to lightweight, no-install probes when it is missing: check for Intel
    render nodes and vendor id (`/dev/dri/renderD*` plus
    `/sys/class/drm/*/device/vendor` == `0x8086`), or `lspci`/`clinfo`/
    `sycl-ls` if available. These confirm an Intel GPU is present and the
    kernel driver is bound.

**Tier 2 — expensive verification, explicit async action with progress:**

Triggered by a "Verify" / "Test connection" action (not by save), reported
through the same phase/progress channel as a job so the UI can show activity:

- the public GHCR trainer image can be resolved and pulled,
- image identity/signature policy passes,
- definitive device access inside the selected trainer image:
  - **CUDA** — Docker has NVIDIA Container Toolkit integration and a one-shot
    container reports `torch.cuda.is_available()`.
  - **XPU** — the container receives the required `/dev/dri` render nodes and
    groups, and a one-shot container reports `torch.xpu.is_available()`.
- the image's trainer API protocol version is compatible with this backend.

**Cross-cutting requirements:**

- Report which detection method succeeded so the UI/status can show it.
- Return a clear pass/fail result to the UI and block save on Tier 1 failure.
- Record outcomes in the existing `last_check_*` columns. A **transient** Tier 1
  failure (server rebooting, network blip) must mark the record unhealthy, never
  delete or invalidate it, and must never re-pin a changed host key.
- Every preflight must have an overall timeout budget and be cancellable.

### 3. Build and resolve trainer images

- Add minimal, non-root `physicalai-trainer-cuda` and
  `physicalai-trainer-xpu` image targets built from `../../../trainer` and
  the local `../../../../library` package. Bake the matching PyTorch extra and required
  user-space GPU libraries into each image.
- Set `physicalai-trainer` as the image entry point. Do not include the backend,
  UI, SSH credentials, datasets, model caches, or generated artifacts.
- Publish the images to GHCR with an immutable Git SHA tag and a moving `latest`
  tag. Include OCI labels for source repository, Git revision, application
  version, trainer API protocol version, and build date.
- Publish an SBOM, scan the image, and sign it in CI. Provisioning verifies the
  expected repository/signature identity before launch.
- Resolve the Studio build revision from a **baked-in** source: an OCI label on
  the running Studio image, a build arg exposed as a setting, or
  `../../../VERSION`. **Do not rely on `git rev-parse HEAD`** — the backend
  ships inside `physical-ai-studio-{cpu,xpu,cuda}` with no `.git` directory, so
  a git-only lookup always fails in production and silently makes `latest` the
  permanent path, defeating SHA pinning entirely. Git may be used only as a
  developer-checkout fallback. Add a test asserting revision resolution succeeds
  when `.git` is absent.
- If the revision resolves, resolve the corresponding device-specific
  `<git-sha>` image tag through the registry. If the revision is unavailable or
  that tag does not exist, resolve the device-specific `latest` tag instead and
  record the fallback reason. Emit a warning-level log on every fallback so a
  misconfigured build does not silently degrade.
- Resolve the selected tag to an immutable repo digest before provisioning,
  persist the selected ref, fallback reason, and digest (`JobProvisioningDB`),
  and pull the image by digest on the remote server. If neither the SHA-tagged
  image nor `latest` can be resolved, fail the job clearly; do not clone or
  install trainer source on the remote server.
- Extend `/health` with trainer build and protocol metadata, and reject
  incompatible images before uploading a dataset. **This is a trainer API
  change that also affects the existing direct-URL registry:** today
  `../../../trainer/src/trainer/main.py` returns only
  `{"status": "healthy"}`. **Decision: grandfather the direct-URL registry,
  strict for SSH.** A direct-URL trainer that reports no protocol version is
  accepted (a human registered it and owns that deployment); log at info level
  and show "protocol unknown" in its status. An SSH-provisioned image must report
  compatible metadata or the job fails before dataset upload, because Studio
  selected that image itself and can guarantee it is current. Add tests for both
  branches so the grandfather path cannot silently widen to SSH.

### 4. SSH container provisioning service

Per job, on the selected server:

- **Safety invariants for every remote command** — run commands as argument
  arrays (never interpolate config fields into a shell string), verify the
  pinned host key on connect, derive image names from the configured device
  type, validate job ids used in names/labels, and never pass user-controlled
  strings as Docker options.
- **Pre-launch GPU availability check** — before pulling/launching, verify the
  server's GPU is not already occupied by another task (foreign training,
  inference server, notebook, etc.):
  - **CUDA** — `nvidia-smi --query-compute-apps=pid,used_memory,process_name`
    lists all processes holding the GPU; combined with
    `--query-gpu=utilization.gpu,memory.used,memory.total` for memory/util
    thresholds. Prefer allocated-memory + process presence over instantaneous
    util%.
  - **XPU** — best-effort via `xpu-smi stats` (utilization/memory); when
    `xpu-smi` is absent, fall back to a memory-usage heuristic. Per-process
    attribution is limited on XPU.
  - If the GPU is occupied, **leave the job pending with backoff** rather than
    failing it or launching into an OOM. This requires: a visible
    `waiting_for_gpu` job state with a user-facing message, exponential backoff
    on the re-check (`run_loop` polls every 0.5 s, so a naive implementation
    would SSH-probe a busy server twice a second), a per-server throttle shared
    with the status endpoint, and a give-up timeout after which the job fails
    with a clear message. Surface this state in the server status/selector so
    users see it upfront.
- **Re-check free disk at provisioning time** against the actual snapshot size
  for this job plus expected artifact size. The Step 2 save-time check used a
  nominal size and cannot know the dataset.
- Resolve the SHA-tagged image first and use `latest` only when that image cannot
  be resolved. Pull the selected image by its immutable repo digest. Stream
  sanitized pull output and emit heartbeats while it runs.
- Verify the image identity/signature and trainer protocol metadata before
  launch, then launch the container by the resolved digest rather than the
  mutable tag.
- Launch a deterministically named container such as
  `physicalai-trainer-<job-id>` with `--restart=no` and management/job/server
  labels. Run as a non-root user, drop unnecessary capabilities, avoid
  `--privileged`, use a read-only root filesystem where practical, and mount
  only bounded writable job/cache directories.
- Include a **`backend_instance_id` label** (a per-process/per-deployment id)
  alongside the management/job/server labels. Remote servers are global and two
  Studio instances can legitimately target the same host; ownership must be
  provable, not assumed (see the orphan-sweep bullet below).
- Set a container-side watchdog (and a bounded `--stop-timeout`) so a crashed or
  disconnected backend cannot leave a container holding the GPU indefinitely
  while no one is reading its output.
- For CUDA, request the GPU through NVIDIA Container Toolkit. For XPU, pass only
  the required `/dev/dri` devices and render/video group ids. Re-run the
  definitive PyTorch device check in the job container before accepting work.
- Publish the trainer port to an ephemeral **remote loopback-only** host port
  (`127.0.0.1`), inspect the assigned port, and never expose it on all interfaces.
- Open an SSH local-forward tunnel from an ephemeral local port (bind to port 0
  and read back the assigned port) to the remote loopback port.
- Enable SSH keepalives and detect tunnel drops. **A dropped tunnel does not fail
  the job.** Re-open the local forward against the still-running container and
  resume streaming, sharing one code path with
  [Restart and Reattach](#restart-and-reattach). Apply a bounded retry budget and
  fail only once it is exhausted or the container is confirmed gone. The existing
  `trainer_stream_reconnect_*` settings cover HTTP-level retries only and do not
  help once the forward itself is dead.
- **Bastion / `ProxyJump` is out of scope.** All target servers are assumed
  directly reachable from where Studio runs. Document the limitation; do not add
  a hop configuration field.
- Persist the container id/name, resolved image digest, remote port, and local
  tunnel port for the job (Step 1, `JobProvisioningDB`) so startup reattach and
  the orphan sweep can find it.
- Poll `/health` with a bounded timeout and verify the reported image/protocol
  metadata matches the inspected image before dataset upload.
- Report progress for each stage via the phase-windowed model (see
  [Progress Reporting](#progress-reporting-for-ssh-train-jobs)), streaming
  sanitized `docker pull`/container output as live messages.
- On startup, **run reattach first** (see
  [Restart and Reattach](#restart-and-reattach)), then sweep. A container may be
  removed **only** when it carries all expected management labels, its
  `backend_instance_id` belongs to this deployment, and it was **not** claimed by
  an active job during reattach. Sweeping on management labels alone would
  destroy a concurrent Studio instance's running job on a shared server; sweeping
  before reattach would destroy this instance's own recoverable jobs. If trusted
  identity cannot be established, log and leave the container alone.
- Stop/remove the container and close the tunnel in a `finally` block.

### 5. Server-aware remote backend

- **Add `TrainingTarget.SSH`.** `schemas/job.py` currently defines only
  `LOCAL` and `REMOTE`. Do **not** express an SSH job as
  `training_target=REMOTE` + `remote_server_id`, because existing code breaks
  immediately:

  - `get_training_backend` raises
    `ValueError("Remote training job is missing its pinned trainer URL")` when
    `payload.remote_trainer_url is None` — an SSH job has no URL at submit time;
  - `TrainingWorker._run_training_job` computes
    `reattaching = training_target is REMOTE and bool(remote_job_id)`, which
    would misfire;
  - `TrainJobPayload.validate_training_target` rejects local jobs carrying
    remote fields and must be extended to validate the SSH combination
    (`remote_server_id` set, `remote_trainer_id`/`remote_trainer_url` unset).

  Audit and update **every** `is TrainingTarget.REMOTE` / `is TrainingTarget.LOCAL`
  branch in the backend as part of this step.

- **Fix `TrainingWorker._target_key`.** It currently returns
  `f"{TrainingTarget.REMOTE.value}:{payload.remote_trainer_id}"` for anything
  non-local. Add an explicit SSH branch returning `f"ssh:{payload.remote_server_id}"`.
  Without it, every SSH job across every server collapses onto the key
  `remote:None`, needlessly serializing unrelated servers and colliding with
  malformed remote jobs. Add tests asserting two different servers yield
  distinct keys and that `None` can never appear in a target key.
- **Backend selection plumbing** — today `get_training_backend(payload)` already
  reads the persisted `TrainJobPayload` (`training_target`, `remote_trainer_url`)
  with no other arguments. Extend the payload/factory so it can also select SSH
  provisioning:
  - `TrainingWorker._run_training_job` resolves `payload.remote_server_id` (new
    field, see Step 6) when present.
  - `get_training_backend(payload)` applies the precedence in
    [Relationship to the existing remote trainer registry](#relationship-to-the-existing-remote-trainer-registry):
    `training_target is SSH` → SSH-provisioned backend; else
    `training_target is REMOTE` (existing direct-URL registry) →
    `RemoteTrainingBackend(payload.remote_trainer_url)`; else `LocalTrainingBackend`.
- Refactor `../../src/services/training_backends/remote.py` so the
  SSH path can inject the tunnel URL and chosen device the same way the
  existing direct-URL path already injects `payload.remote_trainer_url`; keep
  the direct-URL path working unchanged.
- Wrap `train()` with provision-before / teardown-after (in `finally`).
- Generalize the existing progress window constants into the ordered phase table
  - `report_phase` helper (see
    [Progress Reporting](#progress-reporting-for-ssh-train-jobs)).
- Keep the `http` dataset transfer streaming through the tunnel; avoid the `hf`
  transfer for this flow.
- Implement the [Restart and Reattach](#restart-and-reattach) decision.
- On cancel, trigger provisioning teardown (stop/remove container + close tunnel)
  in addition to the existing remote `/jobs/{id}/cancel` + the job's per-job
  interrupt flag (`TrainingWorker.job_interrupt_flags`).

### Restart and Reattach

**Decision: reattach.** A backend restart must not destroy in-flight remote
training.

The worker already supports resuming an in-flight direct-URL remote job via
`payload.remote_job_id` plus the pinned `payload.remote_trainer_url`. That works
because the URL is stable across a backend restart.

For an SSH job it is not: the trainer is reachable only through an **ephemeral
local tunnel port owned by the backend process**. If the backend restarts, the
tunnel is gone, but the remote container is still training. Simply sweeping it
would discard hours of completed GPU work on every restart or redeploy.

**Startup recovery procedure.** For each `JobProvisioningDB` row whose job is in
a non-terminal state:

1. Connect to the server and verify the pinned host key (fail-closed as usual).
2. Inspect the persisted `container_id` / `container_name`. Confirm it exists, is
   running, and carries this deployment's management labels including
   `backend_instance_id`.
3. Re-open the SSH local-forward to the persisted `remote_port`, binding a fresh
   ephemeral local port, and update `local_tunnel_port`.
4. Poll `/health` and confirm the reported image digest still matches the
   persisted `image_digest`.
5. Resume streaming from the trainer's `/jobs/{remote_job_id}/events`, reusing
   the existing SSE reconnect path.

**Failure branches — each must be explicit:**

- Container is gone → the job failed while Studio was down. Mark the job failed
  with a clear message; do not silently restart training.
- Container exists but `/health` never becomes ready within the bounded timeout
  → tear down and fail the job.
- Digest mismatch → treat as untrusted; tear down and fail.
- Host key mismatch → fail-closed, and do **not** tear down (we cannot prove the
  host is the one we provisioned on).
- The remote port is no longer bound / reachable → tear down and fail.

**Interaction with the orphan sweep (Step 4).** Reattach runs _before_ the sweep.
The sweep may only remove containers that are both owned by this deployment and
**not** claimed by an active job during reattach. This ordering is what makes the
sweep safe; without it, the sweep would kill exactly the containers reattach
wants.

Because tunnel drops are also reconnected rather than fatal (see the decisions
table), reattach and mid-run reconnect should share one "re-establish the
forward and resume streaming" code path.

### 6. Thread `remote_server_id` through jobs + train dialog

- Add `remote_server_id` to `TrainJobPayload` in
  `../../src/schemas/job.py`, alongside the existing
  `remote_trainer_id`/`remote_trainer_url` fields for the direct-URL registry,
  and add the `TrainingTarget.SSH` member plus validator rules from Step 5.
- Validate it in `JobService.submit_train_job` (reject if the server is unknown
  or its last preflight failed).
- Resolve the server in `../../src/workers/training_worker.py` and
  pass it into the backend factory (Step 5). Update `_target_key` (Step 5).
- Serve the server's **configured** `device_type` from the DB record via the
  remote-server status endpoint (Step 1) — `SystemService.get_available_training_devices`
  already only reports the Studio host's local devices (`mode="local"`) and is
  not extended for this path; the trainer is not running at dialog time, so the
  SSH path never probes a live `/devices` endpoint.
- **Unify the training-target selector** in
  `../../../ui/src/routes/models/train-model-dialog.tsx`. That dialog
  **already has a remote-trainer picker** (it queries `/api/remote-trainers`,
  health-checks the choice, and sets `training_target` + `remote_trainer_id`).
  Adding a second, separate "remote server" dropdown would leave users with two
  similarly-named pickers and no indication of which wins. Instead, present one
  **"training target"** control listing local, registered direct-URL trainers,
  and registered SSH servers as entries of a single list, each with a type badge
  and status, and derive `training_target` from the selected entry. Show status
  inline and disable submit when the selection is unhealthy. A target whose GPU
  is busy stays selectable — the job simply waits (see Step 4) — but the wait
  must be stated in the dialog before submit.
- Regenerate OpenAPI types
  (`npm run build:api:download && npm run build:api`).

### 7. Training targets management screen (UI) — **Partial**

One **global** screen for managing everywhere training can run — local,
direct-URL trainers, and SSH servers — presented as a single "training targets"
list with a type badge per entry. It mirrors the existing list/detail pattern
used by robots and cameras (`routes/robots/layout.tsx` + `robot.tsx`).

**Already present** on `albert/ssh-server-persistence`:

- A `remoteTrainers` build-time feature flag in
  `application/ui/src/config/feature-flags.ts` (default off,
  `PUBLIC_ENABLE_REMOTE_TRAINERS`, with a `localStorage` dev override).
- A flag-gated route in `../../../ui/src/router.tsx` and
  `application/ui/src/routes/remote-servers/index.tsx`, a thin wrapper that
  renders `features/remote-trainers/remote-trainers-page.tsx`.

**Decided corrections:**

- **Promote the route to global.** `router.tsx` currently defines
  `const remoteServers = project.path('/remote-servers')` under
  `paths.project.*`, yielding `/projects/:project_id/remote-servers`. Neither
  `RemoteServerDB` nor `RemoteTrainerDB` has a project FK, so move the route out
  of the `project` subtree to a top-level path (e.g. `/training-targets`) with
  its own primary-navigation entry, outside `ProjectLayout`.
- **Unify the naming.** Users see one concept, "training target"; the direct-URL
  vs SSH distinction is a type badge, not a separate product noun. Rename the
  route folder and component accordingly (the current
  `routes/remote-servers/` → `features/remote-trainers/RemoteTrainersPage`
  split uses both nouns for the same screen), and pick one of `routes/` or
  `features/` per `../../../ui/AGENTS.md`. Internal model names
  (`RemoteTrainerDB`, `RemoteServerDB`) stay as they are.

**Remaining work:**

- Build out the list/detail split: list of targets (name, type badge, host,
  device type, status badge) with a "New" action; create/edit form for SSH
  entries covering name, host, port, username, auth type (SSH key or password)
  with a secret field, and device type (CUDA/XPU); and a detail/status view.
- **Status view** — driven by the status endpoint (Step 1): reachable,
  authenticated, Docker usable, registry reachable, driver present + version,
  container device probe, compatible image version (or "protocol unknown" for a
  grandfathered direct-URL trainer), last-checked time, and an "in use by job" /
  "waiting for GPU" indicator. Add a "Test connection" button that runs the
  Tier 2 verification (Step 2) with progress, and reflect live state with a
  status badge (Healthy / Unreachable / Misconfigured / Busy / Checking).
- Surface a distinct, actionable state when `REMOTE_SERVER_SECRET_KEY` is not
  configured, or when a record's stored key fingerprint does not match the
  active key (secret needs re-entry after rotation).
- **Data layer** — use the generated `$api` hooks (`$api.useQuery` /
  `$api.useMutation`) against the new endpoints, following existing route
  patterns; never render secret material returned from the API (it never is).
- **Empty/error states** — reuse the shared `EmptySelection` / illustrated
  message pattern from `router.tsx` for "no target selected" and connection
  errors.

### 8. Threat model, docs, security review, tests

#### Threat model (do this first, not last)

Granting the configured SSH account access to the Docker daemon is
**effectively root on that host**. Combined with per-job container launch, this
means: _anyone who can register a remote server and submit a training job gains
root-equivalent code execution on that machine._

**The Studio backend has no authentication or user model today** — it assumes a
single trusted local user. That is the decisive fact for this feature, and it
forces the following:

- **Remote SSH training ships feature-flagged off by default** and stays off
  until an authorization story exists. The existing `remoteTrainers` UI flag is
  not sufficient on its own — it only hides the screen. Add a **backend-side**
  enablement switch so a disabled deployment _rejects SSH job submission and
  remote-server writes_, rather than merely hiding the UI from a browser that
  could still call the API directly.
- **Document the trust assumption plainly** in the user-facing docs: enabling
  this feature means anyone who can reach the Studio API can execute code as
  root on every registered server. Do not enable it on a network-exposed Studio
  instance.
- **Blast radius of a compromised backend:** it holds Fernet-decryptable SSH
  credentials for every registered server. A single compromise is a compromise
  of the whole GPU fleet. Note this explicitly; it is the strongest argument for
  a dedicated, unprivileged, per-purpose SSH account.

Record in the threat model at minimum:

- who is authorized to create/edit `RemoteServerDB` records (today: anyone who
  can reach the API) and what the backend enforces;
- what a job submitter effectively gains on the remote host;
- the mitigations: a dedicated unprivileged SSH account used only for this,
  rootless Docker where possible, never mounting the Docker socket into the
  trainer container, and non-root + dropped-capability containers;
- the re-review trigger: this model must be revisited before the feature is
  enabled by default or before Studio is exposed beyond localhost.

#### Docs

- Extend `../../../trainer/README.md` and backend docs with the secret/trust
  and provisioning model, Fernet key setup, key rotation/loss remediation, host
  setup for CUDA/XPU, XPU limitations, image verification, and cleanup/recovery
  procedures.

#### Tests

- Add unit/integration tests mirroring
  `../../tests/services/test_remote_training_backend.py`, covering
  provisioning, preflight, teardown-on-failure, and cached-image reuse. Add tests
  for:
  - host-key mismatch fails-closed,
  - config fields with shell metacharacters are rejected / cannot inject
    commands,
  - secret material (`ssh_secret_encrypted`, `ssh_key_passphrase_encrypted`,
    `host_key`) is never serialized in API responses,
  - registry/image pull failures return a clear error,
  - an incompatible or incorrectly signed image is rejected before upload,
  - a resolvable SHA-tagged image is selected instead of `latest`,
  - `latest` is selected only when the build revision or SHA-tagged image cannot
    be resolved, with the fallback reason persisted,
  - build-revision resolution succeeds when `.git` is absent (containerized
    backend),
  - the selected ref and resolved digest are persisted and the job container
    launches by that digest rather than a mutable tag,
  - Docker publishes the trainer only on remote loopback,
  - CUDA/XPU container device-probe failures block save and provisioning,
  - orphan-sweep reclaims a persisted labeled container/port after a simulated
    crash, leaves unrelated containers untouched, **and leaves containers owned
    by a different `backend_instance_id` untouched**,
  - `_target_key` returns distinct keys for two different SSH servers and never
    embeds `None`,
  - two SSH jobs on two different servers run concurrently; two on the same
    server serialize,
  - **reattach**: after a simulated backend restart, a still-running container is
    reclaimed, the tunnel is re-opened on a fresh local port, and streaming
    resumes; and each failure branch (container gone, `/health` never ready,
    digest mismatch, host-key mismatch, port unreachable) behaves as specified,
  - **sweep ordering**: reattach runs before the sweep, so a recoverable
    container is never removed,
  - a dropped SSH tunnel mid-training reconnects and resumes rather than failing
    the job, and fails only once the retry budget is exhausted,
  - a busy GPU leaves the job pending with backoff (not failed), the wait is
    visible, and the give-up timeout eventually fails it,
  - a direct-URL trainer reporting no protocol version is accepted
    (grandfathered) while an SSH image reporting no protocol version is rejected
    before dataset upload,
  - a record whose stored Fernet key fingerprint does not match the active key is
    flagged as needing re-entry rather than failing at provisioning time,
  - SSH job submission is rejected when the backend-side feature switch is off,
  - streamed remote command output is sanitized/capped before becoming a
    `message`.
- Add at least one **integration test against a containerized `sshd`**. Every
  test listed above is otherwise mock-level, which cannot catch host-key,
  tunnel, or auth-negotiation regressions.
- Add UI tests for the remote server management screen (list/detail render,
  create/edit form validation, status badge states, and the "Test connection"
  flow) following existing route test patterns.

## Progress Reporting for SSH Train Jobs

A remote SSH job has more phases than a local run (connect, image pull,
verification, trainer start) on top of the existing upload → train → download.
The existing pipeline already carries everything needed: `ProgressReporter` accepts
`(progress: int 0-100, message, extra_info: dict)`, and
`TrainingTrackingDispatcher.report` forwards each update to the job store and a
`JOB_UPDATE` event. The design extends that model rather than changing the
contract.

### Model: phase-windowed bar + structured phase descriptor

- **Ordered phases**, each owning a slice of the 0–100 bar (single place to
  retune, mirroring today's `SNAPSHOT_UPLOAD_PROGRESS` / `TRAINING_PROGRESS_END`
  constants). Suggested windows:

  | Phase               | Key             | Window | Notes                                                                                        |
  | ------------------- | --------------- | ------ | -------------------------------------------------------------------------------------------- |
  | Connect & preflight | `connect`       | 0–2    | SSH, Docker, driver, disk, registry, and GPU-free checks.                                    |
  | Image pull          | `image_pull`    | 2–5    | Resolve SHA tag or fallback `latest`, then pull by digest; cached layers can make this fast. |
  | Image verification  | `image_verify`  | 5–7    | Resolve digest, verify identity/signature and protocol metadata.                             |
  | Trainer start       | `trainer_start` | 7–9    | Launch container, inspect port, open tunnel, poll `/health`.                                 |
  | Dataset upload      | `upload`        | 9–17   | Existing snapshot ZIP stream (real byte %).                                                  |
  | Training            | `train`         | 17–96  | Existing remote training progress (dominant slice).                                          |
  | Model download      | `download`      | 96–100 | Existing artifact stream (real byte %).                                                      |

- **Overall percent** is the phase's byte/step progress mapped into its window
  (reuse the existing `_upload_progress` / `_download_progress` / `_to_local_progress`
  helpers, generalized to `(window_start, window_end)`).
- **One phase table for all targets** (decided). Retuning
  `SNAPSHOT_UPLOAD_PROGRESS = 10` / `TRAINING_PROGRESS_END = 95` in
  `services/training_backends/remote.py` to the windows above also shifts the
  progress curve for local and direct-URL remote jobs by roughly 1–2%. That
  shift is accepted in exchange for a single, non-branching implementation.
  Update any existing progress assertions accordingly rather than special-casing
  non-SSH targets.
- **Structured phase descriptor** in `extra_info["phase"]` so the UI can render a
  stepper: `{ key, label, index, total, state: "active"|"done"|"skipped", indeterminate: bool }`.
  This is additive; the plain `progress`/`message` still drive the basic bar.
- **Phase keys must be a shared, versioned constant** consumed by both the
  backend and the UI, so the two cannot drift silently. Note that `phase` shares
  the existing 16 KB `extra_info` cap with remote telemetry — budget for it.

### Indeterminate setup phase (`image_pull`)

Registry and Docker pull output does not provide one stable cross-runtime
percentage, so:

- Set the bar to the phase's window start, mark `indeterminate: true`, and let
  the UI show a spinner within that step instead of a misleading exact %.
- **Stream command stdout/stderr line-by-line over the SSH channel** and forward
  meaningful lines as `message` updates (e.g. "Pulling CUDA trainer image" or
  "Layer already exists") so the user sees live activity.
- Emit a periodic **heartbeat** message while a long command runs so the UI never
  looks frozen.
- On command exit, advance the bar to the window end and mark the phase `done`.

### Skipped / fast phases

- If all image layers are already cached or image verification finishes
  immediately, advance to that window's end and mark the step `done` so the jump
  is explained rather than looking stuck. Pull the resolved digest for every job;
  query the moving `latest` tag only when SHA resolution falls back to it.

### Trust boundary

- `connect`/`image_pull`/`image_verify`/`trainer_start`/`upload`/`download` are
  _driven_ by the **studio backend's own provisioning code**, so their `phase`
  descriptor and progress values are trusted.
- However, the **stdout/stderr streamed from remote Docker commands** is
  environment-influenced content, not trusted text. Before forwarding it as a
  `message`, strip control characters and cap line/message length (reuse the
  sanitize helper). Do **not** treat streamed command output as trusted just
  because the phase is studio-driven.
- Only the `train` phase's `extra_info` originates from the remote trainer; keep
  the existing sanitize + 16 KB cap for that untrusted telemetry.

### Backend changes

- Generalize the window constants in
  `services/training_backends/remote.py` into an ordered phase table and a
  `report_phase(...)` helper that maps sub-progress into the active window and
  attaches the `phase` descriptor.
- The provisioning service (Step 4) calls `report_phase` at each stage and pumps
  streamed command output through as `message` updates, **sanitized and length-
  capped** (strip control chars) since remote command output is not trusted text.
- `ProgressReporter`, the dispatcher, and the job schema are unchanged (`phase`
  rides inside the existing `extra_info` dict).

### UI changes

- Extend the training progress view (`train-model-dialog.tsx` / the model
  training status component) to render a **phase stepper** from
  `extra_info.phase` (pending/active/done/skipped, spinner for indeterminate),
  above the existing overall bar + message line.
- During `train`, keep the existing step-loss telemetry rendering.
- Degrade gracefully when `phase` is absent (local jobs): show only the bar +
  message, exactly as today.

### Error attribution

- On failure, the current phase + message pinpoints where it broke ("Failed
  during image verification"), persisted in `job.message` / `extra_info` and
  shown in the stepper by marking the active step as failed.

## Data Transfer: SSH vs HTTP

Keep the existing HTTP `http` transfer, streamed through the SSH tunnel:

- Publish the container's trainer port to an ephemeral `127.0.0.1` port on the
  remote host and local-forward that port; the existing chunked
  `PUT /jobs/{id}/dataset` flows encrypted through the tunnel.
- Reuses all existing streaming, progress, and archive-safety code — no new
  transfer path.
- Raw SFTP/scp would add code with no progress reporting or ZIP validation.
- Throughput is network-bound either way; SSH encryption overhead is negligible
  with AES-NI, and it is a one-time per-job transfer.
- Binding to localhost + tunnel also resolves the trainer's "no auth / do not
  expose the port" concern.

## Concurrency

- The backend `TrainingWorker` already serializes at most one job per
  **execution target** but runs jobs on distinct targets concurrently. An
  SSH-provisioned job must use its own target key `ssh:<remote_server_id>`
  (see Step 5) so at most one trainer is ever provisioned per server at once
  without blocking jobs on other servers or the direct-URL registry. Reusing
  the existing `remote:<remote_trainer_id>` branch is a bug, not a shortcut:
  SSH jobs have no `remote_trainer_id`, so all of them would share the key
  `remote:None`. This target-per-job model also means per-job cancellation must
  key off the job id (via `TrainingWorker.job_interrupt_flags`), never a single
  shared interrupt signal, or cancelling one job could incorrectly interrupt
  another job running concurrently.
- **GPU-busy jobs stay pending with backoff** (decided). Because `run_loop` polls
  every 0.5 s and reserves the target before running, a naive implementation
  would SSH-probe a busy server twice a second. Required: a distinct
  `waiting_for_gpu` job state surfaced in the UI, exponential backoff on the
  re-check, a per-server connection throttle shared with the status endpoint, and
  a give-up timeout after which the job fails with a clear message.
- **Status/preflight SSH is out-of-band:** the status endpoint and device
  resolution open SSH connections from API request handlers, which can run while
  a job trains on the same server. Throttle these per server (short timeouts,
  limited concurrency) so UI polling cannot disrupt provisioning or pile up
  connections, and give the UI explicit "Checking" loading states.
- Optional future safety net: a per-server "busy" flag so a job whose selected
  server is occupied is left pending rather than double-provisioned.
- Per-server parallelism (two different servers at once) already falls out of
  the target-per-job model above; no extra work is required for it, only for
  the busy-flag safety net.

## Open Risks / Follow-ups

All ten previously-open decisions are now resolved in
[Confirmed Decisions](#confirmed-decisions). What remains are execution risks,
not open questions.

1. **Reattach correctness** — reattach is now the required behaviour, which makes
   startup recovery a correctness-critical path rather than a cleanup path. Every
   failure branch in [Restart and Reattach](#restart-and-reattach) must be
   implemented and tested; a bug here either kills recoverable jobs or resurrects
   dead ones.
2. **Sweep ordering** — the orphan sweep must run _after_ reattach and must skip
   claimed containers. Getting this backwards destroys exactly the work reattach
   exists to save.
3. **Shared-server ownership** — remote servers are global and two Studio
   instances can target the same host. Any cleanup keyed only on management
   labels will destroy another instance's running job; `backend_instance_id` is
   the mitigation.
4. **Teardown robustness** — reliably stop/remove the trainer container on job
   cancel and on unrecoverable failure (persisted identity + labels + ownership +
   container-side watchdog) so a stale container does not hold the GPU.
5. **Pending-job starvation** — with GPU-busy jobs waiting rather than failing, a
   permanently occupied server accumulates waiting jobs. The give-up timeout and
   a visible queue state are what keep this from looking like a hang.
6. **Fallback-tag reproducibility** — SHA-tagged images are preferred, but
   `latest` intentionally advances when used as the fallback. Persist the
   selected ref, fallback reason, resolved digest, and trainer build metadata
   with each job, and watch for the containerized-backend revision-resolution
   trap in Step 3.
7. **Grandfathered trainers** — allowing direct-URL trainers without protocol
   metadata means a genuinely incompatible old trainer can still be selected.
   Bound this: log it, show "protocol unknown" in status, and revisit once the
   metadata has propagated.
8. **Host/runtime compatibility** — image contents do not replace host GPU
   drivers, NVIDIA Container Toolkit, XPU device permissions, or enough local
   disk. Keep host and in-container checks in preflight and provisioning, and
   re-check disk against the actual dataset size at provisioning time.
9. **Image supply chain** — pulling a public image adds registry availability
   and artifact-trust dependencies. Sign, scan, verify, and pin the resolved
   digest for the running container rather than launching an unresolved tag.
10. **GPU-busy detection limits** — the pre-launch check has an inherent race
    (a foreign task can claim the GPU between check and launch on a shared
    server) and XPU per-process attribution is weaker than CUDA. Treat the check
    as a best-effort guard, prefer allocated-memory + process presence over
    spiky utilization%, and keep OOM-at-startup handling as the final backstop.
11. **No backend authorization** — the feature grants root-equivalent execution
    on registered hosts to anyone who can reach the API. It must stay
    feature-flagged off by default, with a backend-side switch, until Studio has
    an auth model. Re-review before default-on or any network exposure.
12. **No bastion support** — declared out of scope. If a future target sits
    behind a jump host, the SSH transport will need a hop configuration; keep the
    transport abstraction from hard-coding a single-hop assumption where cheap.
