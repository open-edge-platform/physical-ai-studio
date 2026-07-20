# Remote Server SSH Container Provisioning for Training

## Overview

Add a managed "remote server" concept to Physical AI Studio. Users register a
remote GPU/XPU server with SSH credentials and a target device type through the
web UI. When a training job is started, the backend SSHes into the selected
server, resolves the device-specific trainer image from the local Git SHA (or
`latest` when the SHA-tagged image cannot be resolved), starts one isolated
trainer container for the job, runs the job through the existing HTTP
`RemoteTrainingBackend`, and removes the container when the job completes.

This extends today's single static `TRAINER_URL` model into per-job,
dynamically provisioned trainers.

### Relationship to the existing `training_mode` / `TRAINER_URL`

Backend selection today is global: `get_training_backend()` reads
`settings.training_mode` (`local` in-process vs `remote` → static
`TRAINER_URL`), and `Settings.validate_remote_training_config` requires
`TRAINER_URL` whenever `training_mode="remote"`. SSH provisioning is a **third
path selected per job**, not a new global mode:

- **Selector precedence:** if a job carries a `remote_server_id`, it is
  provisioned over SSH regardless of `training_mode`; the static `TRAINER_URL`
  path is used only when `training_mode="remote"` **and** no `remote_server_id`
  is present; otherwise training runs locally. This precedence lives in
  `get_training_backend(...)` (see Step 5) and is documented next to the
  setting.
- **No new required settings:** SSH provisioning must not require
  `TRAINER_URL`. The `validate_remote_training_config` validator is unchanged
  and only governs the static-`TRAINER_URL` path.
- **Device listing:** `SystemService.get_available_training_devices` keeps its
  current behavior for `local` and static-`remote`; the SSH path serves the
  server's **configured** `device_type` from the DB record (the trainer is not
  running at dialog time; see Step 6), replacing — not supplementing — the live
  `/devices` probe for that path.

## Confirmed Decisions

| Topic                           | Decision                                                                                                                                                                                                                                                                                 |
| ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Secret storage                  | Fernet symmetric encryption at rest; key from an env var.                                                                                                                                                                                                                                |
| Encrypted (confidential) fields | Fernet-encrypt only the SSH auth material: `ssh_secret` (private key for `auth_type=key`, or password for `auth_type=password`) and `ssh_key_passphrase` (nullable, key auth only).                                                                                                      |
| Non-encrypted, never-serialized | `host_key` is integrity data (the server's public host key for TOFU), not a secret: stored in plaintext, but never returned in API responses.                                                                                                                                            |
| Secret key lifecycle            | Fernet key lives in an env var (e.g. `REMOTE_SERVER_SECRET_KEY`), never in the DB. Losing/rotating the key makes stored secrets undecryptable, so document that rotating the key requires re-entering each server's secret.                                                              |
| Auth material                   | `auth_type=key` supports passphrase-protected private keys via an optional encrypted `ssh_key_passphrase`; `auth_type=password` stores the encrypted password.                                                                                                                           |
| Host key verification           | Pin the server host key on first successful preflight (TOFU), store it with the record, and verify it (fail-closed) on every later connect. A mismatch blocks the job/preflight with a clear error.                                                                                      |
| Command safety                  | All remote commands run as argument arrays (no shell string interpolation). Image names, container names, labels, and device arguments come from trusted application constants or validated identifiers, not arbitrary user input.                                                       |
| Trainer distribution            | Publish dedicated `physicalai-trainer-cuda` and `physicalai-trainer-xpu` OCI images. Do not reuse the full Studio application images as trainer images.                                                                                                                                  |
| Trainer launch                  | Prefer the device-specific image tagged with the local Git SHA; use `latest` only when that SHA-tagged image cannot be resolved. Resolve and record the selected image's immutable digest, then run `physicalai-trainer` in a job-scoped Docker container bound only to remote loopback. |
| Trainer lifecycle               | One container per job. Persist the container id/name, image digest, remote published port, and local tunnel port so orphans can be swept after a crash.                                                                                                                                  |
| Concurrency                     | Rely on the existing single-worker serialization (one training at a time). Throttle status/preflight SSH connections per server with short timeouts so UI polling cannot disrupt a running job.                                                                                          |
| Dataset transfer                | Keep the HTTP `http` transfer, streamed through an SSH tunnel.                                                                                                                                                                                                                           |
| Device type                     | User provides GPU/XPU device type when configuring the server.                                                                                                                                                                                                                           |
| Image selection                 | Resolve the local Git SHA, then resolve the corresponding device-specific image tag. Fall back to `latest` only if the local SHA or its image cannot be resolved. Persist the selected ref, fallback reason, and immutable digest with the job.                                          |
| Registry access                 | Pull public trainer images from GHCR. Registry credentials and private registries are outside the initial scope.                                                                                                                                                                         |
| First-job cost                  | The first image pull can be large; later jobs reuse Docker's cached layers.                                                                                                                                                                                                              |
| Driver check                    | CUDA: `nvidia-smi` plus an in-container `torch.cuda.is_available()` probe. XPU: host render-node/driver probes plus an in-container `torch.xpu.is_available()` probe.                                                                                                                    |
| Preflight                       | On SSH config save: verify reachability/auth, Docker access, disk space, registry access, matching drivers/device passthrough, and trainer image compatibility.                                                                                                                          |
| Supply chain                    | Publish an SBOM, scan and sign trainer images, and verify the expected image identity before launch. Always launch the selected image by its resolved digest.                                                                                                                            |
| Management UI                   | Dedicated global screen to manage SSH connections and view their status (health + in-use), plus a server selector in the train dialog.                                                                                                                                                   |
| GPU availability                | Before launching training, check the server GPU is free. CUDA: reliable via `nvidia-smi` compute-apps + memory. XPU: best-effort via `xpu-smi stats` / memory heuristic. Block or warn if occupied.                                                                                      |
| Progress reporting              | Phase-windowed 0–100 bar plus a structured `phase` descriptor in `extra_info` so the UI shows a stepper (connect → image pull → verify → start → upload → train → download). Indeterminate setup phases stream sanitized command output + heartbeats.                                    |

## Architecture Context

- Training runs through the `TrainingBackend` abstraction
  (`services/training_backends/`). `LocalTrainingBackend` trains in-process;
  `RemoteTrainingBackend` offloads to a trainer service over HTTP at
  `TRAINER_URL`.
- The backend `TrainingWorker.run_loop` fetches and runs **one** pending job at
  a time (awaits `_train_model` to completion before looping), so training is
  already serialized on the Studio side.
- The trainer service (`application/trainer/`) is a FastAPI app exposing
  `/jobs`, `/jobs/{id}/dataset`, `/jobs/{id}/events` (SSE), `/jobs/{id}/artifact`,
  `/jobs/{id}/cancel`, `/devices`, and `/health`. It has no built-in auth and is
  intended for a trusted private network.
- The existing `http` dataset transfer streams a validated ZIP via
  `PUT /jobs/{id}/dataset` with progress mirroring and archive-safety checks.
- Persistence uses SQLAlchemy models in `db/schema.py`, repositories under
  `repositories/`, Pydantic schemas under `schemas/`, and Alembic migrations.
- The existing `application/docker/Dockerfile` produces full Studio CPU/XPU/CUDA
  images whose runtime installs the backend environment and starts `run.sh`.
  Add minimal trainer-specific image targets instead of using those images as-is.

## Implementation Steps

### 1. Persist servers with encrypted secrets

- Add `RemoteServerDB` to `application/backend/src/db/schema.py` with columns:
  `id`, `name`, `host`, `port`, `username`, `auth_type` (key/password),
  `device_type` (cuda/xpu), `created_at`, `updated_at`, plus:
  - **Fernet-encrypted (confidential):** `ssh_secret` (private key or password),
    `ssh_key_passphrase` (nullable, key auth only).
  - **Plaintext but never serialized:** `host_key` (pinned public host key for
    TOFU verification — integrity data, not a secret, so no encryption needed).
- Persist per-job provisioning state so orphaned trainers can be reclaimed after
  a backend crash: store the container id/name, resolved image digest, remote
  published port, and local tunnel port (e.g. on the job record or a small
  provisioning table keyed by job id).
- Add an Alembic migration for the new table and per-job container/tunnel state.
- Add a repository under `application/backend/src/repositories/`.
- Add Pydantic schemas under `application/backend/src/schemas/` (never expose
  `ssh_secret`, `ssh_key_passphrase`, or `host_key` in responses — the first two
  are confidential, `host_key` is internal integrity data; add a test asserting
  none are serialized).
- Add a `RemoteServerService` and a CRUD API router registered like
  `application/backend/src/api/settings.py`.
- Add a Fernet key env var (e.g. `REMOTE_SERVER_SECRET_KEY`) to
  `application/backend/src/settings.py`, kept out of the DB; encrypt the
  confidential fields (`ssh_secret`, `ssh_key_passphrase`) on write and decrypt
  only when provisioning. Document that rotating/losing the key requires
  re-entering each server's secret.
- Expose a status endpoint (e.g. `POST /api/remote-servers/{id}:check` and/or
  `GET /api/remote-servers/{id}/status`) that runs the SSH preflight (Step 2)
  on demand and returns a structured result the UI can render (reachable,
  authenticated, Docker usable, registry reachable, driver present/version,
  container device probe result, image/protocol version, last-checked timestamp,
  and whether the server is currently in use by a running job).

### 2. Verify-on-save SSH preflight

- In the create/update flow, connect over SSH (e.g. `asyncssh`) and verify:
  - reachability and authentication,
  - **host key**: on the first successful preflight, pin (store) the presented
    host key on the record (TOFU); on later connects, verify it and fail-closed
    on mismatch,
  - `docker version` succeeds for the configured SSH user without privilege
    escalation,
  - enough free disk space is available for the image, dataset, and artifact,
  - the public GHCR trainer image can be resolved and pulled,
  - device driver matching the configured type:
    - **CUDA** — `nvidia-smi`.
    - **XPU** — try `xpu-smi` first. It is not always installed, so fall back
      to lightweight, no-install probes when it is missing: check for Intel
      render nodes and vendor id (`/dev/dri/renderD*` plus
      `/sys/class/drm/*/device/vendor` == `0x8086`), or `lspci`/`clinfo`/
      `sycl-ls` if available. These confirm an Intel GPU is present and the
      kernel driver is bound.
  - definitive device access inside the selected trainer image:
    - **CUDA** — Docker has NVIDIA Container Toolkit integration and a one-shot
      container reports `torch.cuda.is_available()`.
    - **XPU** — the container receives the required `/dev/dri` render nodes and
      groups, and a one-shot container reports `torch.xpu.is_available()`.
  - the image's trainer API protocol version is compatible with this backend.
- Report which detection method succeeded so the UI/status can show it.
- Return a clear pass/fail result to the UI and block save on failure.

### 3. Build and resolve trainer images

- Add minimal, non-root `physicalai-trainer-cuda` and
  `physicalai-trainer-xpu` image targets built from `application/trainer/` and
  the local `library/` package. Bake the matching PyTorch extra and required
  user-space GPU libraries into each image.
- Set `physicalai-trainer` as the image entry point. Do not include the backend,
  UI, SSH credentials, datasets, model caches, or generated artifacts.
- Publish the images to GHCR with an immutable Git SHA tag and a moving `latest`
  tag. Include OCI labels for source repository, Git revision, application
  version, trainer API protocol version, and build date.
- Publish an SBOM, scan the image, and sign it in CI. Provisioning verifies the
  expected repository/signature identity before launch.
- Resolve the local commit SHA with `git rev-parse HEAD`. If it succeeds, resolve
  the corresponding device-specific `<git-sha>` image tag through the registry.
  If the local SHA is unavailable or that tag does not exist, resolve the
  device-specific `latest` tag instead and record the fallback reason.
- Resolve the selected tag to an immutable repo digest before provisioning,
  persist the selected ref, fallback reason, and digest, and pull the image by
  digest on the remote server. If neither the SHA-tagged image nor `latest` can
  be resolved, fail the job clearly; do not clone or install trainer source on
  the remote server.
- Expose trainer build and protocol metadata from `/health`; reject incompatible
  images before uploading a dataset.

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
  - If the GPU is occupied, block the job with a clear "server GPU busy"
    message (or leave it pending) rather than launching into an OOM. Surface
    this state in the server status/selector so users see it upfront.
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
- For CUDA, request the GPU through NVIDIA Container Toolkit. For XPU, pass only
  the required `/dev/dri` devices and render/video group ids. Re-run the
  definitive PyTorch device check in the job container before accepting work.
- Publish the trainer port to an ephemeral **remote loopback-only** host port
  (`127.0.0.1`), inspect the assigned port, and never expose it on all interfaces.
- Open an SSH local-forward tunnel from an ephemeral local port (bind to port 0
  and read back the assigned port) to the remote loopback port.
- Persist the container id/name, resolved image digest, remote port, and local
  tunnel port for the job (Step 1) so a crash recovery sweep can reclaim it.
- Poll `/health` with a bounded timeout and verify the reported image/protocol
  metadata matches the inspected image before dataset upload.
- Report progress for each stage via the phase-windowed model (see
  [Progress Reporting](#progress-reporting-for-ssh-train-jobs)), streaming
  sanitized `docker pull`/container output as live messages.
- On startup, sweep orphaned managed containers by persisted id/name and trusted
  management labels. Verify all labels before stopping anything so unrelated
  containers cannot be removed.
- Stop/remove the container and close the tunnel in a `finally` block.

### 5. Server-aware remote backend

- **Backend selection plumbing** — today `get_training_backend()` takes no
  arguments and the worker calls it blind. Extend the factory (and
  `TrainingContext`) so the worker can pass the resolved `RemoteServer`:
  - `training_worker._train_model` resolves `payload.remote_server_id` and
    passes the server to `get_training_backend(remote_server=...)`.
  - `get_training_backend(...)` applies the precedence in
    [Relationship to `training_mode`](#relationship-to-the-existing-training_mode--trainer_url):
    `remote_server` present → SSH-provisioned backend; else
    `training_mode="remote"` → static-`TRAINER_URL` backend; else local.
- Refactor `application/backend/src/services/training_backends/remote.py` so the
  backend accepts the tunnel URL and chosen device (constructor args) instead of
  reading only `settings.trainer_url` in `__init__`; keep the static-URL path
  working for the existing `training_mode="remote"` flow.
- Wrap `train()` with provision-before / teardown-after (in `finally`).
- Generalize the existing progress window constants into the ordered phase table
  - `report_phase` helper (see
    [Progress Reporting](#progress-reporting-for-ssh-train-jobs)).
- Keep the `http` dataset transfer streaming through the tunnel; avoid the `hf`
  transfer for this flow.
- On cancel, trigger provisioning teardown (stop/remove container + close tunnel)
  in addition to the existing remote `/jobs/{id}/cancel` + `interrupt_event`.

### 6. Thread `remote_server_id` through jobs + train dialog

- Add `remote_server_id` to `TrainJobPayload` in
  `application/backend/src/schemas/job.py`.
- Validate it in `JobService.submit_train_job` (reject if the server is unknown
  or its last preflight failed).
- Resolve the server in `application/backend/src/workers/training_worker.py` and
  pass it into the backend factory (Step 5).
- Serve the server's **configured** `device_type` from the DB record via
  `SystemService.get_available_training_devices` — the trainer is not running at
  dialog time, so this replaces (not supplements) the live `/devices` probe for
  the SSH path.
- Add a server selector to the training flow in
  `application/ui/src/routes/models/train-model-dialog.tsx` (choose which
  registered remote server runs the job; show its status inline and disable
  submit when no healthy server is selected).
- Regenerate OpenAPI types
  (`npm run build:api:download && npm run build:api`).

### 7. Remote server management screen (UI)

A dedicated, global (non-project-scoped) screen to manage SSH connections and
view their status. Mirrors the existing list/detail pattern used by robots and
cameras (`routes/robots/layout.tsx` + `robot.tsx`).

- **Routing** — add a top-level route under `application/ui/src/router.tsx`
  (e.g. `paths.remoteServers` at `/remote-servers` or nested under a new
  `/settings` area), outside the `project` subtree since servers are shared
  across projects. Add an entry point in the app's primary navigation.
- **Route folder** — new `application/ui/src/routes/remote-servers/` with:
  - `layout.tsx` — list of registered servers (name, host, device type, and a
    status badge) with a "New" action; left-list / right-detail split.
  - `new.tsx` / `edit.tsx` — form to create/edit a connection: name, host,
    port, username, auth type (SSH key or password) with a secret field,
    and device type (CUDA/XPU). On submit, call the CRUD API; run the
    verify-on-save preflight (Step 2) and surface pass/fail before the record is
    accepted.
  - `remote-server.tsx` — detail/status view for the selected server.
- **Status view** — for the selected server show a health panel driven by the
  status endpoint (Step 1): reachable, authenticated, Docker usable, registry
  reachable, driver present + version, container device probe, compatible image
  version, last-checked time, and an "in use by job" indicator. Add a "Test
  connection" button that re-runs the preflight on demand, and reflect live
  state with a status badge (e.g. Healthy / Unreachable / Misconfigured / Busy /
  Checking).
- **Data layer** — use the generated `$api` hooks (`$api.useQuery` /
  `$api.useMutation`) against the new endpoints, following existing route
  patterns; never render secret material returned from the API (it never is).
- **Empty/error states** — reuse the shared `EmptySelection` / illustrated
  message pattern from `router.tsx` for "no server selected" and connection
  errors.

### 8. Docs, security review, tests

- Extend `application/trainer/README.md` and backend docs with the secret/trust
  and provisioning model. Document that granting the SSH account access to the
  Docker daemon is effectively host-level privilege; use a dedicated account
  and do not mount the Docker socket inside the trainer container.
- Add unit/integration tests mirroring
  `application/backend/tests/services/test_remote_training_backend.py`, covering
  provisioning, preflight, teardown-on-failure, and cached-image reuse. Add tests
  for:
  - host-key mismatch fails-closed,
  - config fields with shell metacharacters are rejected / cannot inject
    commands,
  - secret material (`ssh_secret`, `ssh_key_passphrase`, `host_key`) is never
    serialized in API responses,
  - registry/image pull failures return a clear error,
  - an incompatible or incorrectly signed image is rejected before upload,
  - a resolvable SHA-tagged image is selected instead of `latest`,
  - `latest` is selected only when the local SHA or SHA-tagged image cannot be
    resolved, with the fallback reason persisted,
  - the selected ref and resolved digest are persisted and the job container
    launches by that digest rather than a mutable tag,
  - Docker publishes the trainer only on remote loopback,
  - CUDA/XPU container device-probe failures block save and provisioning,
  - orphan-sweep reclaims a persisted labeled container/port after a simulated
    crash and leaves unrelated containers untouched,
  - streamed remote command output is sanitized/capped before becoming a
    `message`.
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
- **Structured phase descriptor** in `extra_info["phase"]` so the UI can render a
  stepper: `{ key, label, index, total, state: "active"|"done"|"skipped", indeterminate: bool }`.
  This is additive; the plain `progress`/`message` still drive the basic bar.

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

- The backend `TrainingWorker` already serializes to one job at a time, so only
  one trainer is ever provisioned at once. No extra locking is required now.
- **Status/preflight SSH is out-of-band:** the status endpoint and device
  resolution open SSH connections from API request handlers, which can run while
  a job trains on the same server. Throttle these per server (short timeouts,
  limited concurrency) so UI polling cannot disrupt provisioning or pile up
  connections, and give the UI explicit "Checking" loading states.
- Optional future safety net: a per-server "busy" flag so a job whose selected
  server is occupied is left pending rather than double-provisioned.
- Per-server parallelism (two different servers at once) can be added later by
  loosening the single-worker constraint and gating on a per-server semaphore.

## Open Risks / Follow-ups

1. **Teardown robustness** — reliably stop/remove the trainer container on job
   cancel, backend crash, or tunnel drop (persisted container identity + trusted
   labels + startup orphan sweep) so a stale container does not hold the GPU.
2. **Fallback-tag reproducibility** — SHA-tagged images are preferred, but
   `latest` intentionally advances when used as the fallback. Persist the
   selected ref, fallback reason, resolved digest, and trainer build metadata
   with each job so results remain attributable, and retain published digests
   according to an explicit registry policy.
3. **Host/runtime compatibility** — image contents do not replace host GPU
   drivers, NVIDIA Container Toolkit, XPU device permissions, or enough local
   disk. Keep host and in-container checks in preflight and provisioning.
4. **Image supply chain** — pulling a public image adds registry availability
   and artifact-trust dependencies. Sign, scan, verify, and pin the resolved
   digest for the running container rather than launching an unresolved tag.
5. **GPU-busy detection limits** — the pre-launch check has an inherent race
   (a foreign task can claim the GPU between check and launch on a shared
   server) and XPU per-process attribution is weaker than CUDA. Treat the check
   as a best-effort guard, prefer allocated-memory + process presence over
   spiky utilization%, and keep OOM-at-startup handling as the final backstop.
