# Remote Server SSH Container Provisioning for Training

**Scope of this document: backend only** — SSH transport, preflight, container
provisioning, and worker integration. The management screen, the unified target
selector, and the progress stepper live in
[`remote-ssh-trainer-ui-plan.md`](remote-ssh-trainer-ui-plan.md). PR sequencing
across both lives in
[`remote-ssh-trainer-pr-plan.md`](remote-ssh-trainer-pr-plan.md).

## Overview

Add a managed "remote server" concept to Physical AI Studio. A user registers a
remote GPU/XPU server by picking one of the SSH hosts they **already have
configured in `~/.ssh/config`**, names it, and declares its device type. When a
training job is started, the backend SSHes into that server, resolves the
device-specific trainer image from Studio's own compiled-in trainer protocol
version (no `latest` fallback), starts one isolated trainer
container for the job, runs the job through the existing HTTP
`RemoteTrainingBackend`, and removes the container when the job completes.

This extends today's per-job remote trainer selection into per-job, dynamically
provisioned trainers.

### Deployment model: the user's own workstation

**Studio runs on the user's machine.** `Settings.storage_dir` defaults to a
per-user directory, and the API has no authentication because
it assumes a single trusted local user. That assumption is what makes this
feature's credential design simple:

- There is **no deployment operator** to inject secrets. The person configuring
  a remote server is the person who owns the SSH key.
- The user almost certainly **already has a working `Host` entry** for their GPU
  box — that is how they log into it today.
- Therefore Studio does not need to receive, store, encrypt, or transport SSH
  credentials at all. It needs to name one.

The containerized `physical-ai-studio-{cpu,xpu,cuda}` images are a **secondary**
target and inherit the identical mechanism: the operator mounts the user's
`~/.ssh` (or exposes an agent socket) into the container. No separate code path.

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
- **Credential configuration:** `RemoteServerDB` stores a single non-secret
  `ssh_host_alias` — the name of a `Host` stanza in the user's SSH config. It
  stores no key, password, passphrase, or username. `asyncssh` resolves the
  alias and authenticates; Studio never reads key material.
- **Device listing:** `SystemService.get_available_training_devices` already
  always reports the Studio host's local devices (`mode="local"`); per-trainer
  device listing for the direct-URL registry goes through
  `RemoteTrainerService`'s health check, not this endpoint. The SSH path
  follows the same pattern: it serves the server's **configured** `device_type`
  from the DB record (the trainer is not running at dialog time) through the
  remote-server status endpoint (Step 1), not a live `/devices` probe.

## Confirmed Decisions

| Topic                          | Decision                                                                                                                                                                                                                                                                                                                                                      |
|--------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Deployment model               | **Studio runs on the user's own workstation.** The container images are a secondary target that mounts the same `~/.ssh`. There is no deployment operator and no secret-injection contract. See [Deployment model](#deployment-model-the-users-own-workstation).                                                                                              |
| SSH login credentials          | **The user's existing `~/.ssh/config`.** `asyncssh` is given the user's SSH config and resolves the selected `Host` alias for hostname, port, user, and identity. Studio never receives, reads, stores, encrypts, or transports a private key, password, or passphrase.                                                                                       |
| Credential persistence         | `RemoteServerDB` stores one non-secret `ssh_host_alias` string. No secret material exists anywhere in Studio's database, payload JSON, logs, API responses, UI state, or trainer images — because Studio never has it in the first place.                                                                                                                     |
| Host key verification          | **`~/.ssh/known_hosts`**, via `asyncssh`'s default verification. An unknown host fails closed with an actionable "run `ssh <alias>` once to accept its fingerprint" message; a changed host key surfaces as `asyncssh`'s verification failure.                                                                                                                |
| Missing/invalid alias          | A `ssh_host_alias` that is absent from the SSH config, or matches only a wildcard stanza, is a distinct actionable state ("SSH host alias not found in your SSH config"), never a 500. Validated on save and re-checked on every preflight.                                                                                                                   |
| SSH library                    | **`asyncssh`.** Native asyncio, native `~/.ssh/config` parsing, native `known_hosts` verification, and a local-forward API that supports the reconnect-and-resume requirement without a thread bridge.                                                                                                                                                        |
| Command safety                 | All remote commands run as argument arrays (no shell string interpolation). Image names, container names, labels, and device arguments come from trusted application constants or validated identifiers, not arbitrary user input.                                                                                                                            |
| Trainer distribution           | **Already shipped.** `physicalai-trainer-cuda` / `physicalai-trainer-xpu` build targets exist in `../../docker/Dockerfile.trainer`; `../../../.github/workflows/trainer-images.yml` publishes them to GHCR with SBOM, provenance, and cosign signing. Vulnerability scanning is **not** in this workflow — it runs in `security-scan.yml`'s `trivy-trainer-image-scan` job (see the Supply chain row and risk 23).                                                                                                                |
| Trainer launch                 | Require the device-specific image tagged with Studio's own compiled-in trainer protocol version (`physicalai-trainer-<device>:protocol-<N>`); there is no fallback tag. Resolve and record the selected image's immutable digest, then run `physicalai-trainer` in a job-scoped Docker container bound only to remote loopback.                                                                           |
| Trainer lifecycle              | One container per job. Persist the container id/name, image digest, remote published port, local tunnel port, and the non-secret `ssh_host_alias` so orphans can be swept and recoverable jobs reattached after a crash.                                                                                                                                      |
| Concurrency                    | Reuse the existing **per-execution-target** serialization in `TrainingWorker.run_loop`: one job at a time per target, jobs on distinct targets run concurrently. SSH jobs need their own target key (next row). Throttle status/preflight SSH connections per server with short timeouts so UI polling cannot disrupt a running job.                          |
| Execution target key           | An SSH job's target key must be `ssh:<remote_server_id>`. Reusing the existing `remote:<remote_trainer_id>` branch is **not** acceptable: SSH jobs carry no `remote_trainer_id`, so every SSH job on every server would collapse onto the single key `remote:None`. `TrainingWorker._target_key` must be extended explicitly.                                 |
| Job target discriminator       | Add a third `TrainingTarget.SSH` member rather than overloading `REMOTE` with an optional `remote_server_id`. Overloading `REMOTE` breaks `get_training_backend` (raises when `remote_trainer_url is None`) and the worker's `reattaching` check.                                                                                                             |
| Backend restart behavior       | **Reattach.** On startup, for each non-terminal job with a `JobProvisioningDB` row, re-open the SSH tunnel to the persisted `remote_port`/`container_id`, re-verify `/health` and image digest, and resume streaming. Only genuinely orphaned containers are swept. See [Restart and Reattach](#restart-and-reattach).                                        |
| Tunnel drop mid-training       | **Reconnect and resume.** A dropped tunnel does not fail the job. Use SSH keepalives, re-open the forward against the still-running container, and resume streaming. Consistent with the reattach decision above.                                                                                                                                             |
| Dataset transfer               | Keep the HTTP `http` transfer, streamed through an SSH tunnel.                                                                                                                                                                                                                                                                                                |
| Device type                    | User provides GPU/XPU device type when configuring the server.                                                                                                                                                                                                                                                                                                |
| Image selection                | Resolve `physicalai-trainer-<device>:protocol-<N>`, where `N` is Studio's own compiled-in `TRAINER_API_PROTOCOL_VERSION` constant, not the Studio build revision — the trainer image only rebuilds on trainer-relevant path changes (see `trainer-images.yml`'s `paths:` filter), so most Studio commits never produce a matching revision-tagged image at all. **No fallback tag**: `trainer-images.yml` publishes no `latest` at all, and the only other tags are the immutable `<version>-dev-<short-sha>` and the moving `main`, which tracks the newest build regardless of protocol and is therefore exactly the stale, wire-incompatible image this scheme exists to avoid picking. The strict SSH `/health` check would reject it anyway, just later and after a wasted pull/launch. Fail the job immediately with an actionable message ("no trainer image published yet for protocol `<N>`") and persist the selected ref and immutable digest with the job.                                                                               |
| Trainer protocol compatibility | **Grandfather the direct-URL registry, strict for SSH.** A direct-URL trainer reporting no protocol version is allowed (a human registered and owns it). An SSH-provisioned image must report compatible metadata or the job fails before dataset upload (Studio selected that image itself).                                                                 |
| Library/training-logic compatibility | **Protocol version does not cover this — it is a separate, range-based check.** `protocol-<N>` only guarantees wire-schema compatibility; a `physicalai-train`-only change republishes under the same protocol number with no version bump (`library/**` is in `trainer-images.yml`'s `paths:` filter). Publish the library version as the OCI label `org.open-edge-platform.physicalai.trainer.library-version` (from `importlib.metadata`, not the hand-maintained `0.1.0` semver) and read it off the registry manifest **at digest-resolution time, before pulling**: trainer older than Studio's → non-fatal warning; a `policy` with a documented minimum → hard fail, scoped to that policy alone. `/health` re-reports it as defense-in-depth. **Not encoded in the tag** (`protocol-<N>-lib-<X>`): tags match by equality but library compatibility is a range, and a composite tag makes the namespace combinatorial, multiplying both no-fallback outage windows and retention-protection surface. |
| Registry access                | Pull public trainer images from GHCR. Registry credentials and private registries are outside the initial scope.                                                                                                                                                                                                                                              |
| First-job cost                 | The first image pull can be large; later jobs reuse Docker's cached layers.                                                                                                                                                                                                                                                                                   |
| Driver check                   | CUDA: `nvidia-smi` plus an in-container `torch.cuda.is_available()` probe. XPU: `xpu-smi` or an Intel render-node check on the host, plus an in-container `torch.xpu.is_available()` probe.                                                                                                                                                                   |
| Preflight                      | Two tiers. Tier 1 (cheap) gates save: alias resolution, reachability/auth, host key, Docker access, disk, driver, registry reachability. Tier 2 (expensive) runs as an explicit async action: image resolve/pull, signature policy, in-container device probe, protocol compatibility. GPU occupancy is reported by Tier 1 but never blocks save. See Step 2. |
| Supply chain                   | Per trainer image: SBOM, provenance, and a cosign signature, with moving tags (`main`, `protocol-<N>`) promoted onto the digest **only after** signing so they never point at an unsigned build. **Vulnerability scanning is no longer a publish gate** — `trivy-trainer-image-scan` in `security-scan.yml` runs nightly (or on dispatch) against `:main`, *after* images are published and tagged, so `protocol-<N>` can point at a not-yet-scanned image. Provisioning verifies the expected image identity before launch and always launches by resolved digest. See risk 23.                                                                                                                                                                          |
| GPU availability               | Before launching training, check the server GPU is free. CUDA: reliable via `nvidia-smi` compute-apps + memory. XPU: best-effort via `xpu-smi stats` / memory heuristic. If occupied, the job **stays `pending`** with backoff and a visible waiting phase, plus a give-up timeout — it is not failed immediately. See Step 4.                                |
| GPU-busy job state             | **No new `JobStatus` member.** `JobStatus` is `pending\|running\|completed\|failed\|canceled`, generated into `openapi-spec.d.ts` and consumed across the UI; adding a member forces every consumer to change. The wait is expressed in the existing `extra_info["phase"]` channel the UI already needs. See the UI plan.                                     |
| Progress phase table           | **Per-target phase table**, selected once at backend construction. Do **not** retune the shared `SNAPSHOT_UPLOAD_PROGRESS` / `TRAINING_PROGRESS_END` constants: shifting local and direct-URL progress curves (and rewriting their assertions) buys nothing, and a per-target table is equally non-branching at the call sites.                               |
| Backend authorization          | **None today — single trusted local user.** Anyone who can reach the API can register a server and submit a job, which grants root-equivalent execution on that host.                                                                                                                                                                                         |

## Architecture Context

- Training runs through the `TrainingBackend` abstraction
  (`application/backend/src/services/training_backends/`).
  `LocalTrainingBackend` trains in-process; `RemoteTrainingBackend` offloads
  to a trainer service over HTTP at a URL pinned per job
  (`TrainJobPayload.remote_trainer_url`), resolved from the `RemoteTrainer`
  registry at submission time. Backend selection is
  `get_training_backend(payload)` in
  `services/training_backends/__init__.py`, which today branches only on
  `payload.training_target is TrainingTarget.REMOTE` vs. else-local.
- Both backends build the same `training.TrainingJobSpec` (via the shared
  `build_spec(context)` helper in `services/training_backends/local.py`) and,
  for in-process runs, hand it to the shared
  `application/backend/src/training/job.py:run_training_job(...)` — the single
  place Lightning fit/checkpoint/export logic lives, used identically by
  `LocalTrainingBackend.train()` and by the trainer service's
  `trainer/runner.py:TrainerRunner._train()`. This plan's SSH work sits
  entirely above this layer: it never touches `training/`, it only decides
  *where* a `RemoteTrainingBackend` instance points its `base_url`.
- The remote backend's dataset-transfer strategy is pluggable
  (`services/training_backends/_training_methods.py`: an abstract
  `TrainingMethod` with one concrete `HttpTrainingMethod`), and
  `RemoteTrainingBackend._training_method()` always returns
  `HttpTrainingMethod(self)` — there is no `hf` transfer to accidentally pick
  up, which is why the [Data Transfer](#data-transfer-ssh-vs-http) decision
  below reuses that same path rather than adding a new one.
- The backend `TrainingWorker.run_loop` reserves one job per **execution
  target** (`local`, or `remote:<remote_trainer_id>`) and runs jobs on distinct
  targets concurrently as asyncio tasks; only jobs competing for the same
  target are serialized. Each running job gets its own per-job interrupt flag
  (keyed by job id in a shared dict) so cancelling one job cannot affect
  another running concurrently on a different target.
- The trainer service (`application/backend/src/trainer/`) is a FastAPI app
  exposing `/jobs`, `/jobs/{id}/dataset`, `/jobs/{id}/events` (SSE),
  `/jobs/{id}/artifact`, `/jobs/{id}/cancel`, `/devices`, `/storage`, and
  `/health`. It has no built-in auth and is intended for a trusted private
  network.
- `/health` **already returns** `status`, `protocol_version`, `device_type`,
  `build_revision`, `build_date`, and `application_version`
  (`application/backend/src/trainer/schemas.py`, `HealthInfo`), each sourced
  from `TRAINER_API_PROTOCOL_VERSION`/`TRAINER_DEVICE_TYPE`/
  `TRAINER_BUILD_REVISION`/`TRAINER_BUILD_DATE`/`TRAINER_APPLICATION_VERSION`
  env vars baked into the published images at build time. `protocol_version`
  doubles as the axis CI publishes the moving `protocol-<N>` tag on (see
  [Resolve trainer images](#3-resolve-trainer-images)), so the same constant
  drives both image selection and the post-pull compatibility check — the
  Studio backend's own side (reading its compiled-in protocol version so it can
  pick a matching trainer tag) is what Step 3 below still has to build.
- The existing `http` dataset transfer streams a validated ZIP via
  `PUT /jobs/{id}/dataset` with progress mirroring and archive-safety checks.
- Persistence uses SQLAlchemy models in `db/schema.py`, repositories under
  `repositories/`, Pydantic schemas under `schemas/`, and Alembic migrations.

## Implementation Steps

### 1. Persist servers by SSH host alias

Add a new migration for `RemoteServerDB` and `JobProvisioningDB`.

- `RemoteServerDB` in `../../backend/src/db/schema.py` with:
  - `id`, `name`, `device_type` (`DeviceType`, restricted to CUDA/XPU),
    `created_at`, `updated_at`;
  - `ssh_host_alias` — the `Host` stanza name from the user's SSH config.
    Non-secret. **Unique**; a second record for the same alias is meaningless;
  - **Last-check summary:** `last_check_status`, `last_check_at`,
    `last_check_latency_ms`, `last_check_reason_code`. These exist so a
    transient preflight failure updates status instead of destroying the record.
- **Host and port for display** are resolved from the SSH config at read time
  and returned as derived, non-persisted fields. If the alias has disappeared,
  the record renders in the "alias not found" state instead of showing stale
  values.
- `JobProvisioningDB` (separate table keyed by `job_id`, **not** the job payload
  JSON) holds per-job provisioning state so a crashed backend can sweep or
  reclaim an orphaned container from durable, queryable columns: `image_ref`,
  `image_digest`, `container_id`, `container_name`,
  `remote_port`, `local_tunnel_port`, `ssh_host_alias`,
  `trainer_build_version`, `trainer_protocol_version`.
- `repositories/remote_server_repo.py`, `repositories/job_provisioning_repo.py`,
  and mappers under `repositories/mappers/`.
- `schemas/remote_server.py`, `schemas/job_provisioning.py`. There is no
  internal-vs-public schema split to maintain and no `to_public()` sanitizer,
  because no field is secret. `tests/schemas/test_remote_server.py` asserts the
  schema has no key/password/passphrase field at all.
- `services/remote_server_service.py` and the CRUD router `api/remote_servers.py`
  — **note this router is not yet wired**; `main.py` and `api/dependencies.py`
  register only `remote_trainers_router` today.
- An **SSH config reader** that lists selectable aliases (parsing `Host` stanzas,
  excluding wildcard patterns) for the create/edit form, and resolves one alias
  to its effective hostname/port/user for display. Read-only; never returns
  `IdentityFile` contents.

**Status endpoint** (e.g. `POST /api/remote-servers/{id}:check` and/or
`GET /api/remote-servers/{id}/status`) runs the Step 2 preflight and returns a
structured result the UI can render: alias resolvable, reachable, authenticated,
host key verified, Docker usable, registry reachable, driver present/version,
container device probe result, image/protocol version, last-checked timestamp,
and whether the server is currently in use by a running job or waiting on a
busy GPU.

### 2. Verify-on-save SSH preflight

Split preflight into two tiers. A multi-GB registry pull plus a one-shot GPU
container must not run inside a create/update request handler.

**Tier 1 — cheap checks, gate the save (seconds, bounded timeout):**

- resolve `ssh_host_alias` in the user's SSH config; reject an alias that is
  absent or wildcard-only with an actionable message,
- reachability and authentication using the resolved config (agent or
  `IdentityFile` — `asyncssh` decides, Studio does not),
- **host key**: verified against `~/.ssh/known_hosts` by `asyncssh`. An unknown
  host fails closed with "run `ssh <alias>` once to accept its fingerprint"; a
  mismatch surfaces as a verification failure. Studio neither pins nor writes
  host keys,
- `docker version` succeeds for the SSH user without privilege escalation,
- enough free disk space for the image and a nominal job (per-job dataset size
  is re-checked at provisioning time — see Step 4),
- device driver matching the configured type:
  - **CUDA** — `nvidia-smi`.
  - **XPU** — `xpu-smi` if installed; otherwise check for Intel render nodes
    (`/dev/dri/renderD*` plus `/sys/class/drm/*/device/vendor` == `0x8086`).
    Tier 2's in-container `torch.xpu.is_available()` is the authoritative check;
    Tier 1 only needs enough signal to reject an obviously wrong host.
- **registry reachable** — an unauthenticated manifest `HEAD` against the public
  GHCR trainer repository. This is Tier 1 because it is one sub-second request
  and it separates "this host cannot reach the registry at all" (a real
  misconfiguration, worth catching at save time) from "the pull failed midway"
  (Tier 2's job). It does **not** resolve or pull the image.
- **GPU free** — reuse the `nvidia-smi` / `xpu-smi` invocation already made for
  the driver check and report occupancy in the same pass.

**GPU-busy must not gate save.** It is the one Tier 1 result that is reported but
not blocking: a busy GPU is a transient state, not a misconfiguration, and a user
must be able to register or edit a target while a job is running on it. Blocking
here would also contradict the decision that a busy target stays selectable in
the train dialog. Distinguish *reported* Tier 1 results from *blocking* ones.

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
- Return a clear pass/fail result to the UI and block save on a **blocking**
  Tier 1 failure. GPU occupancy is reported, never blocking (see above).
- Record outcomes in the existing `last_check_*` columns. A **transient** Tier 1
  failure (server rebooting, network blip) must mark the record unhealthy and
  never delete or invalidate it.
- Every preflight must have an overall timeout budget and be cancellable.
- Per-server throttling shared with the status endpoint, so UI polling cannot
  disrupt a running job.

### 3. Resolve trainer images

The images and their supply chain already exist. What remains is backend-side
resolution.

**Already shipped — do not rebuild:**

- Non-root `physicalai-trainer-cuda` / `physicalai-trainer-xpu` build targets in
  `../../docker/Dockerfile.trainer`, entry point `physicalai-trainer`, no
  backend or UI content baked in (the image build asserts neither
  `/app/application/backend` nor `/app/application/ui` exist, and no Docker
  socket).
- `../../../.github/workflows/trainer-images.yml` publishes an immutable
  `<version>-dev-<short-sha>` tag at build time, then promotes the moving
  `main` and `protocol-<N>` tags onto that verified, signed digest **after**
  the signing step, with `sbom: true`, `provenance: mode=max`, cosign signing,
  and a metadata-verification step. **No `latest` tag is published** (there is
  no release trigger yet, so every published trainer image is a development
  build). Vulnerability scanning is not in this workflow — see the Supply
  chain decision row and risk 23. Labels include
  `org.opencontainers.image.source`, `.revision`, `.version`, `.created`, and
  `org.open-edge-platform.physicalai.trainer.api-protocol`.
- `/health` returns `protocol_version`, `device_type`, `build_revision`,
  `build_date`, and `application_version`.
- **The `protocol-<N>` moving tag now exists**, promoted post-signing
  (`albert/fix-trainer-images-trivy-egress`, not yet on `main`). This was the
  main blocking prerequisite for Studio-side resolution.

**Remaining work:**

- Confirm `protocol-*` survives `cleanup-old-app-images.yml`'s retention rules
  and extend the allowlist if not (risk 20). This is now more urgent than when
  it was first recorded: with `latest` gone, **no** `physicalai-trainer-*` tag
  matches the release/RC/`latest` allowlist, so every trainer tag — including
  the one Studio auto-resolves — falls inside the `min_versions_to_keep`
  window.
- **Add an `org.open-edge-platform.physicalai.trainer.library-version` OCI
  label** to both trainer targets in `Dockerfile.trainer`, alongside the
  existing `api-protocol` label, populated from
  `importlib.metadata.version("physicalai-train")` resolved at build time (not
  the hand-maintained `library/pyproject.toml` semver). Extend the workflow's
  existing metadata-verification step to assert it is present and non-`unknown`,
  and the smoke test to assert `/health` reports the same value. This label is
  what makes the pre-pull library check possible (see below).
- Resolve Studio's own compiled-in trainer protocol version — the same
  `TRAINER_API_PROTOCOL_VERSION`-shaped constant the trainer bakes in, shared
  as a single source of truth (e.g. exported from `trainer/schemas.py` or a
  small shared constants module) so CI, the trainer image, and the Studio
  backend can never drift independently. **Do not use the Studio build
  revision (git SHA) as the resolution key**: `trainer-images.yml` only
  rebuilds the trainer image on commits touching trainer-relevant paths (see
  its `paths:` filter), so most Studio commits produce no matching
  revision-tagged trainer image at all — resolving by exact SHA would
  guarantee an almost-permanent silent fallback to `latest`, the SHA-analog of
  the exact trap the original `../../VERSION` rejection existed to avoid (a
  0.1.0-shaped value can never match a SHA tag either, for the same reason —
  neither the Studio semver nor the Studio git SHA is the axis that actually
  determines wire compatibility). Add a test asserting protocol-version
  resolution succeeds however Studio is deployed (container or dev checkout),
  since it is a plain compiled-in constant with no `.git`/environment
  dependency.
- **`TRAINER_API_PROTOCOL_VERSION` is a wire-compatibility version. Increment
  it only in the PR that changes the HTTP wire contract itself: the `trainer/`
  FastAPI request/response schemas, `/health`'s shape, or `TrainingJobSpec`
  (`application/backend/src/training/job.py`). Bumping on unrelated merges
  reintroduces the exact failure mode `protocol-<N>` exists to avoid — a
  version bump on a commit that doesn't touch trainer-relevant paths produces
  a tag CI never publishes, and since there is no `latest` fallback, every SSH
  job fails until a trainer-relevant commit happens to follow and rebuild. It
  also makes `protocol_version` meaningless as a compatibility signal: if it
  advances on every PR, two builds one merge apart are flagged incompatible
  even when the wire contract didn't change.
- **Protocol version alone does not guarantee training-logic parity.**
  `protocol-<N>` and `main` are both moving tags repointed to the newest
  digest on every trainer-relevant rebuild — and `library/**` is in
  `trainer-images.yml`'s `paths:` filter, so a `physicalai-train`-only change
  (a new policy, a training-loop fix, a changed default) republishes under the
  **same** protocol number with no version bump. That keeps the wire contract
  correctly unaffected, but it also means two images tagged `protocol-<N>` can
  contain materially different training logic, and Studio's own currently
  running build can end up older or newer than whatever `protocol-<N>`
  currently resolves to.
- **Expose the library version as an OCI label, and check it before pulling —
  not as part of the tag.** Add
  `org.open-edge-platform.physicalai.trainer.library-version` to both trainer
  targets in `Dockerfile.trainer`, populated from
  `importlib.metadata.version("physicalai-train")` at build time (a
  code-derived value that changes when library content changes, unlike the
  hand-maintained `library/pyproject.toml` semver, which is `0.1.0` and would
  be nearly static). Also report the same value at `/health` alongside
  `build_revision`. **Do not encode the library version in the image tag**
  (e.g. `protocol-<N>-lib-<X>`): a tag can only be matched for equality, but
  library compatibility is a **range** ("trainer library >= what this Studio
  build needs" — newer is normally fine, older is the risk), so a composite
  tag forces equality semantics onto a range-shaped question. It would also
  make the tag namespace combinatorial (protocol × library), multiplying both
  the unpublished-tag windows that the no-fallback decision turns into hard
  failures and the number of moving tags needing retention protection
  (see risk 20).
- Resolve the device-specific `physicalai-trainer-<device>:protocol-<N>` image
  tag through the registry. **There is no fallback tag.** If that tag does
  not exist (e.g. a protocol bump landed in Studio before CI has published a
  trainer image advertising it), fail the job immediately with an actionable
  message rather than substituting another tag. `trainer-images.yml` publishes
  no `latest`; the only alternatives are the moving `main`, which tracks the
  newest build irrespective of protocol and so is precisely the
  wire-incompatible image to avoid, and immutable `<version>-dev-<short-sha>`
  tags that Studio has no reliable way to select (see risk 21). The strict SSH
  `/health` check below would reject a mismatched image anyway, just after
  wasting a pull, container launch, and tunnel setup. Emit a warning-level log
  when resolution fails so a misconfigured or delayed build is visible
  immediately.
- Resolve the selected tag to an immutable repo digest before provisioning,
  persist the selected ref and digest (`JobProvisioningDB`), and pull the
  image by digest on the remote server. If the protocol-tagged image cannot be
  resolved, fail the job clearly; do not clone or install trainer source on
  the remote server.
- **Check the library version at digest-resolution time, before the pull.**
  The `library-version` label is readable straight off the registry manifest
  (`docker buildx imagetools inspect`, the same call that resolves the digest),
  so this costs one metadata read and requires no image pull, container launch,
  or tunnel. Apply **range** logic against Studio's own installed
  `physicalai-train` version:
  - trainer **older** than Studio's → non-fatal warning surfaced in job status
    ("trainer image reports physicalai-train `<X>`, older than this Studio
    build's `<Y>` — recent training-logic changes may not be present"). It is
    not a failure: the protocol check already proves the wire contract holds,
    and most library differences (unrelated bugfixes, policies this job doesn't
    use) don't affect this job.
  - trainer **newer than or equal to** Studio's → proceed silently.
  - job's `policy`/feature has a **documented minimum** library version that
    the image doesn't meet → fail before pulling, naming the policy and the
    required version. This is the only hard gate on this axis, and it is scoped
    to the feature that needs it so unrelated library bumps never force a
    failure.
- Enforce protocol compatibility before uploading a dataset, with the
  **grandfather rule**: a direct-URL trainer reporting no protocol version is
  accepted (log at info, show "protocol unknown" in status); an SSH-provisioned
  image reporting no or incompatible metadata fails the job before dataset
  upload. This `/health` check remains a defense-in-depth safety net for the
  direct-URL grandfather path and for any drift between the resolved digest and
  what actually runs. Add tests for both branches so the grandfather path
  cannot silently widen to SSH.
- Re-confirm the library version from `/health` once the container is up, as
  defense-in-depth against the manifest label disagreeing with what actually
  runs. The pre-pull label check is the one that gates; this one only has to
  catch the mismatch case, and a disagreement between label and `/health` is
  itself a reason to fail (the image is not what it advertised).

### 4. SSH container provisioning service

Per job, on the selected server:

- **Safety invariants for every remote command** — run commands as argument
  arrays (never interpolate config fields into a shell string), let `asyncssh`
  verify the host key against `known_hosts`, derive image names from the
  configured device type, validate job ids used in names/labels, and never pass
  user-controlled strings as Docker options. Studio holds no credential to leak
  into commands, progress, errors, or telemetry.
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
  - If the GPU is occupied, **leave the job `pending` with backoff** rather than
    failing it or launching into an OOM. This requires: a visible waiting state
    expressed through `extra_info["phase"]` with a user-facing message (**not** a
    new `JobStatus` member), exponential backoff on the re-check (`run_loop`
    polls every 0.5 s, so a naive implementation would SSH-probe a busy server
    twice a second), a per-server throttle shared with the status endpoint, and
    a give-up timeout after which the job fails with a clear message.
- **Re-check free disk at provisioning time** against the actual snapshot size
  for this job plus expected artifact size. The Step 2 save-time check used a
  nominal size and cannot know the dataset.
- Resolve the protocol-tagged image (`protocol-<N>`); there is no `latest`
  fallback — fail the job immediately if it cannot be resolved. Pull the
  selected image by its immutable repo digest. Stream sanitized pull output
  and emit heartbeats while it runs.
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
- Persist the container id/name, resolved image digest, remote port, local
  tunnel port, and `ssh_host_alias` for the job (Step 1, `JobProvisioningDB`)
  so startup reattach and the orphan sweep can find it.
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
- **Teardown is conditional, not an unconditional `finally`.** Stop/remove the
  container and close the tunnel on normal completion, cancellation, and
  unrecoverable failure — but **not** when `train()` exits via
  `TrainingSuspendedError` (the existing backend-shutdown/restart path in
  `TrainingWorker._train_model`). An unconditional `finally` would tear the
  container down on every graceful-restart suspend, which destroys the exact
  container the reattach procedure exists to recover; the job would then have
  nothing left to reattach to on the next startup. See
  [Restart and Reattach](#restart-and-reattach) for the corresponding startup
  side of this.

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
- Refactor `../../backend/src/services/training_backends/remote.py` so the
  SSH path can inject the tunnel URL and chosen device the same way the
  existing direct-URL path already injects `payload.remote_trainer_url`; keep
  the direct-URL path working unchanged.
- Wrap `train()` with provision-before / teardown-after (in `finally`), **but
  make teardown conditional on outcome**: skip it when `train()` exits via
  `TrainingSuspendedError` so a graceful restart leaves the container running
  for reattach, matching the existing `suspended` branch in
  `TrainingWorker._train_model`. Only tear down on completion, cancellation, or
  an unrecoverable error.
- Introduce the ordered phase table + `report_phase` helper as a **per-target**
  table (see [Progress Reporting](#progress-reporting-for-ssh-train-jobs)),
  leaving the local and direct-URL curves untouched.
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

**Prerequisite: `abort_orphan_jobs` must not fail the job before reattach runs.**
`TrainingWorker.setup()` calls `TrainingService.abort_orphan_jobs` **before**
`run_loop` starts, i.e. before any container-level reattach can happen.
Today `TrainingService._reattachable_remote_job_id` only recognizes
`TrainingTarget.REMOTE` with a `remote_job_id`; every other RUNNING job —
including every SSH job — falls through to `FAILED`. Left as-is, this would
fail every in-flight SSH job on startup, before the startup recovery procedure
below ever gets a chance to run, making the reattach design unreachable for SSH.
`_reattachable_remote_job_id` must gain an SSH branch: a RUNNING job with
`training_target is TrainingTarget.SSH` and a `JobProvisioningDB` row is
requeued to `PENDING` (like REMOTE today), not failed. The actual container
inspection, tunnel re-open, and health/digest checks happen afterward, when
`run_loop` picks the requeued job back up and the SSH backend's `train()` runs
the steps below — `abort_orphan_jobs` only decides "is this worth trying to
reattach," it does not itself talk to the remote host.

**Startup recovery procedure.** For each `JobProvisioningDB` row whose job is in
a non-terminal state:

1. Resolve the persisted `ssh_host_alias` in the current SSH config, then
   connect, letting `asyncssh` verify the host key against `known_hosts`
   (fail-closed as usual).
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
- Host key verification fails → fail-closed, and do **not** tear down (we cannot
  prove the host is the one we provisioned on).
- `ssh_host_alias` no longer present in the SSH config → mark the job failed with
  an actionable "restore this Host entry to recover" message. Do not attempt an
  unauthenticated fallback, do not substitute a different alias, and do not tear
  down when host identity and container ownership cannot be established.
- The remote port is no longer bound / reachable → tear down and fail.

**Interaction with the orphan sweep (Step 4).** Reattach runs _before_ the sweep.
The sweep may only remove containers that are both owned by this deployment and
**not** claimed by an active job during reattach. This ordering is what makes the
sweep safe; without it, the sweep would kill exactly the containers reattach
wants.

Because tunnel drops are also reconnected rather than fatal (see the decisions
table), reattach and mid-run reconnect should share one "re-establish the
forward and resume streaming" code path.

### 6. Thread `remote_server_id` through jobs

- Add `remote_server_id` to `TrainJobPayload` in
  `../../backend/src/schemas/job.py`, alongside the existing
  `remote_trainer_id`/`remote_trainer_url` fields for the direct-URL registry,
  and add the `TrainingTarget.SSH` member plus validator rules from Step 5.
- Validate it in `JobService.submit_train_job` (reject if the server is unknown,
  its `ssh_host_alias` no longer resolves, or its last preflight failed).
- Resolve the server in `../../backend/src/workers/training_worker.py` and
  pass it into the backend factory (Step 5). Update `_target_key` (Step 5).
- Serve the server's **configured** `device_type` from the DB record via the
  remote-server status endpoint (Step 1) — `SystemService.get_available_training_devices`
  already only reports the Studio host's local devices (`mode="local"`) and is
  not extended for this path; the trainer is not running at dialog time, so the
  SSH path never probes a live `/devices` endpoint.
- Regenerate OpenAPI types
  (`npm run build:api:download && npm run build:api`).

The train-dialog target selector is UI work — see
[`remote-ssh-trainer-ui-plan.md`](remote-ssh-trainer-ui-plan.md).

### 7. UI

Moved to [`remote-ssh-trainer-ui-plan.md`](remote-ssh-trainer-ui-plan.md):
the global training-targets management screen, the unified target selector in
the train dialog, and the progress stepper.

### 8. Docs and tests

#### Docs

- Extend `../../backend/docs/remote-trainer.md` and backend docs with: the
  `~/.ssh/config` contract (what a usable `Host` entry looks like), the agent
  requirement for passphrase-protected keys, accepting a host fingerprint
  before first use, recovery when an alias is removed or renamed, the fact
  that Studio stores no SSH credentials whatsoever, host setup for CUDA/XPU,
  XPU limitations, image verification, cleanup/recovery procedures, and the
  direct-reachability assumption.
- Now that the `protocol-<N>` tag exists in `trainer-images.yml` (see
  [Resolve trainer images](#3-resolve-trainer-images)), update the "Container
  images" section of `../../backend/docs/remote-trainer.md` to document the
  full three-tag scheme (immutable `<version>-dev-<short-sha>`, moving `main`,
  moving `protocol-<N>`), that no `latest` is published, which tags are
  CI-published vs. resolved automatically by Studio, and why exact-SHA matching
  was rejected (risk 21).
- For the containerized deployment, document mounting `~/.ssh` and/or exposing
  `SSH_AUTH_SOCK`, and that every instance eligible to reattach a job needs the
  same alias resolvable.

#### Tests

Add unit/integration tests mirroring
`../../backend/tests/services/test_remote_training_backend.py`, covering provisioning,
preflight, teardown-on-failure, and cached-image reuse. Add tests for:

- an unknown or wildcard-only `ssh_host_alias` fails preflight, provisioning, and
  reattach closed with the actionable message,
- host-key verification failure fails closed, and reattach does not tear down,
- config fields with shell metacharacters are rejected / cannot inject commands,
- no server or provisioning schema has any private-key, password, or passphrase
  field, and no API response contains one,
- registry/image pull failures return a clear error,
- an incompatible or incorrectly signed image is rejected before upload,
- a resolvable `protocol-<N>` image is selected; there is no `latest`
  fallback, and an unresolvable `protocol-<N>` tag fails the job immediately
  with an actionable message,
- protocol-version resolution succeeds regardless of deployment shape
  (container or dev checkout), since it is a plain compiled-in constant with
  no `.git` dependency,
- the `library-version` label is read from the registry manifest **before any
  pull**, and: an older trainer library produces a warning without failing the
  job, an equal/newer one proceeds silently, and a policy with a documented
  minimum fails before the pull naming that policy and the required version,
- a `library-version` label that disagrees with what `/health` later reports
  fails the job (the image did not match what it advertised),
- the selected ref and resolved digest are persisted and the job container
  launches by that digest rather than a mutable tag,
- Docker publishes the trainer only on remote loopback,
- CUDA/XPU container device-probe failures block **provisioning** and fail the
  Tier 2 verification action — but do **not** block save, since the probe is
  Tier 2 and save runs Tier 1 only,
- a save succeeds while the target's GPU is busy (occupancy is reported, not
  blocking) and still fails on a blocking Tier 1 error such as a missing alias
  or an unverified host key,
- an unreachable registry fails Tier 1 without pulling or resolving an image,
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
  digest mismatch, host-key failure, alias missing, port unreachable) behaves as
  specified,
- **sweep ordering**: reattach runs before the sweep, so a recoverable
  container is never removed,
- a dropped SSH tunnel mid-training reconnects and resumes rather than failing
  the job, and fails only once the retry budget is exhausted,
- **teardown is conditional**: a job ending via `TrainingSuspendedError` leaves
  the container and tunnel intact (no teardown call), while completion,
  cancellation, and unrecoverable failure all trigger stop/remove,
- `TrainingService.abort_orphan_jobs` requeues a RUNNING SSH job with a
  `JobProvisioningDB` row to `PENDING` instead of marking it `FAILED`, and this
  runs (in `TrainingWorker.setup()`) before the container-level reattach
  procedure gets a chance to run on the next pickup,
- a busy GPU leaves the job `pending` with backoff (not failed), the wait is
  visible in `extra_info["phase"]`, and the give-up timeout eventually fails it,
- **`JobStatus` gains no new member** — assert the enum's members explicitly so a
  future change cannot silently break generated UI consumers,
- **local and direct-URL progress curves are unchanged** by the introduction of
  the per-target phase table,
- a direct-URL trainer reporting no protocol version is accepted
  (grandfathered) while an SSH image reporting no protocol version is rejected
  before dataset upload,
- streamed remote command output is sanitized/capped before becoming a
  `message`,
- SSH job submission is rejected when the backend-side switch is off.

Add at least one **integration test against a containerized `sshd`**, with a
generated key and a purpose-built SSH config pointed at it via the config-path
setting. Every test listed above is otherwise mock-level, which cannot catch
host-key, tunnel, or auth-negotiation regressions.

## Progress Reporting for SSH Train Jobs

A remote SSH job has more phases than a local run (connect, image pull,
verification, trainer start) on top of the existing upload → train → download.
The existing pipeline already carries everything needed: `ProgressReporter` accepts
`(progress: int 0-100, message, extra_info: dict)`, and
`TrainingTrackingDispatcher.report` forwards each update to the job store and a
`JOB_UPDATE` event. The design extends that model rather than changing the
contract.

### Model: phase-windowed bar + structured phase descriptor

- **Ordered phases**, each owning a slice of the 0–100 bar. The SSH table:

  | Phase               | Key             | Window | Notes                                                                                        |
  | ------------------- | --------------- | ------ | -------------------------------------------------------------------------------------------- |
  | Connect & preflight | `connect`       | 0–2    | SSH, Docker, driver, disk, registry, and GPU-free checks.                                    |
  | Image pull          | `image_pull`    | 2–5    | Resolve the `protocol-<N>` tag (no `latest` fallback), read the `library-version` label off the manifest and range-check it, then pull by digest; cached layers can be fast. |
  | Image verification  | `image_verify`  | 5–7    | Resolve digest, verify identity/signature and protocol metadata.                             |
  | Trainer start       | `trainer_start` | 7–9    | Launch container, inspect port, open tunnel, poll `/health`.                                 |
  | Dataset upload      | `upload`        | 9–17   | Existing snapshot ZIP stream (real byte %).                                                  |
  | Training            | `train`         | 17–96  | Existing remote training progress (dominant slice).                                          |
  | Model download      | `download`      | 96–100 | Existing artifact stream (real byte %).                                                      |

- **Per-target table, selected once at backend construction.** Local and
  direct-URL backends keep their existing windows
  (`SNAPSHOT_UPLOAD_PROGRESS = 10`, `TRAINING_PROGRESS_END = 95`) and their
  existing progress assertions. Retuning the shared constants would shift those
  curves by ~1–2% and force assertion rewrites for no gain; a table chosen at
  construction is equally non-branching at every call site.
- **Overall percent** is the phase's byte/step progress mapped into its window
  (reuse the existing `_upload_progress` / `_download_progress` / `_to_local_progress`
  helpers, generalized to `(window_start, window_end)`).
- **Structured phase descriptor** in `extra_info["phase"]` so the UI can render a
  stepper: `{ key, label, index, total, state: "active"|"done"|"skipped"|"waiting"|"failed", indeterminate: bool }`.
  This is additive; the plain `progress`/`message` still drive the basic bar.
- **The GPU-busy wait is a phase state, not a job status.** A job waiting on a
  busy GPU stays `pending` and reports `connect` with `state: "waiting"` plus a
  message and the give-up deadline. This keeps `JobStatus` at its existing five
  members and avoids a breaking change to every generated UI consumer.
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
  is explained rather than looking stuck. Pull the resolved digest for every job.

### Trust boundary

- `connect`/`image_pull`/`image_verify`/`trainer_start`/`upload`/`download` are
  _driven_ by the **studio backend's own provisioning code**, so their `phase`
  descriptor and progress values are trusted.
- However, the **stdout/stderr streamed from remote Docker commands** is
  environment-influenced content, not trusted text. Before forwarding it as a
  `message`, strip control characters and cap line/message length (reuse the
  sanitize helper). Do **not** treat streamed command output as trusted just
  because the phase is studio-driven. "Control characters" for this sanitize
  step means, precisely:
  - all Unicode `Cc` category characters (`U+0000`–`U+001F`, `U+007F`–`U+009F`)
    except `\n`, which is preserved as a line separator and re-split before the
    per-line cap is applied;
  - all Unicode `Cf` category bidi-override characters (`U+200F`,
    `U+202A`–`U+202E`, `U+2066`–`U+2069`, `U+FEFF`), which can otherwise reorder
    or hide rendered text in the UI;
  - every ESC-introduced sequence, from `\x1b` through its terminator — this
    covers ANSI SGR (color/style), OSC hyperlinks, OSC clipboard writes, and CSI
    cursor-movement/screen-clear sequences, all of which a hostile or noisy
    remote process could otherwise use to manipulate what the operator sees.
- Only the `train` phase's `extra_info` originates from the remote trainer; keep
  the existing sanitize + 16 KB cap for that untrusted telemetry.

### Backend changes

- Add the per-target phase table and a `report_phase(...)` helper in
  `services/training_backends/remote.py` that maps sub-progress into the active
  window and attaches the `phase` descriptor.
- The provisioning service (Step 4) calls `report_phase` at each stage and pumps
  streamed command output through as `message` updates, **sanitized and length-
  capped** (strip control chars) since remote command output is not trusted text.
- `ProgressReporter`, the dispatcher, and the job schema are unchanged (`phase`
  rides inside the existing `extra_info` dict).
- **`JobStatus` is unchanged.**

The UI side of the stepper is in
[`remote-ssh-trainer-ui-plan.md`](remote-ssh-trainer-ui-plan.md).

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
- **GPU-busy jobs stay `pending` with backoff** (decided). Because `run_loop`
  polls every 0.5 s and reserves the target before running, a naive
  implementation would SSH-probe a busy server twice a second. Required: the
  `waiting` phase state surfaced in the UI, exponential backoff on the re-check,
  a per-server connection throttle shared with the status endpoint, and a
  give-up timeout after which the job fails with a clear message.
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

1. **Reattach correctness** — reattach is required behaviour, which makes
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
   a visible waiting phase are what keep this from looking like a hang.
6. **Unresolvable protocol tag** — `protocol-<N>` is the only trainer image tag
   Studio will use for SSH provisioning; there is no fallback, because no
   `latest` is published and the only moving alternative (`main`) would be a
   stale, pre-bump image in the exact window `protocol-<N>` is missing. Fail
   the job with an actionable message
   ("no trainer image published yet for protocol `<N>`") and persist the
   resolved ref, digest, and trainer build metadata with each job. Keep the
   resolution key on Studio's compiled-in protocol-version constant rather than
   the Studio build revision (git SHA) or `VERSION` semver — neither axis
   reliably matches a trainer tag, since the trainer image only rebuilds on
   trainer-relevant path changes.
7. **Grandfathered trainers** — allowing direct-URL trainers without protocol
   metadata means a genuinely incompatible old trainer can still be selected.
   Bound this: log it, show "protocol unknown" in status, and revisit once the
   metadata has propagated.
8. **Protocol version is not a training-logic guarantee** — `protocol-<N>`
   only proves wire-schema compatibility. Because `library/**` changes
   republish under the same protocol number (no version bump required), two
   images tagged `protocol-<N>` can run materially different training logic,
   and Studio's own build can drift older or newer than whatever currently
   resolves. Bound this with the separate library-version freshness check
   (non-fatal warning on an older trainer, hard gate only for a
   `policy`/feature with a documented minimum) rather than folding it into the
   protocol check.
9. **Host/runtime compatibility** — image contents do not replace host GPU
   drivers, NVIDIA Container Toolkit, XPU device permissions, or enough local
   disk. Keep host and in-container checks in preflight and provisioning, and
   re-check disk against the actual dataset size at provisioning time.
10. **Image supply chain** — pulling a public image adds registry availability
   and artifact-trust dependencies. Sign, scan, verify, and pin the resolved
   digest for the running container rather than launching an unresolved tag.
11. **GPU-busy detection limits** — the pre-launch check has an inherent race
    (a foreign task can claim the GPU between check and launch on a shared
    server) and XPU per-process attribution is weaker than CUDA. Treat the check
    as a best-effort guard, prefer allocated-memory + process presence over
    spiky utilization%, and keep OOM-at-startup handling as the final backstop.
12. **No backend authorization** — the feature grants root-equivalent execution
    on registered hosts to anyone who can reach the API. Acceptable on a
    localhost-only workstation deployment; the backend switch (Step 8) plus a
    re-review are required before any network exposure.
13. **SSH agent reach exceeds the registered servers** — reusing the user's SSH
    setup means a compromised Studio process can reach every identity the agent
    holds, not just the GPU boxes. This is the one security cost of the
    ssh_config design versus a narrowly-scoped injected credential. Mitigate by
    documenting a dedicated per-host `IdentityFile`.
14. **SSH config drift** — a renamed or deleted `Host` entry breaks a saved
    server and can block reattach of a running job. Surface the missing-alias
    state prominently, fail closed, and never substitute a different alias.
15. **`abort_orphan_jobs` runs before container-level reattach** —
    `TrainingWorker.setup()` calls `TrainingService.abort_orphan_jobs` ahead of
    `run_loop`. Its `_reattachable_remote_job_id` check must be extended to
    treat a RUNNING SSH job backed by a `JobProvisioningDB` row as reattachable
    (requeue to `PENDING`), or every in-flight SSH job is marked `FAILED` on
    startup before the startup recovery procedure ever runs, silently defeating
    the reattach feature this plan is built around.
16. **Teardown must not fire on graceful suspend** — the provisioning
    teardown described in Step 4/5 is wrapped in `finally`, but must skip
    stop/remove when `train()` exits via `TrainingSuspendedError`; otherwise a
    normal backend restart tears down the very container reattach is meant to
    recover.
17. **CI concurrency can drop a `protocol-<N>` promotion under
    frequent trainer rebuilds** — `trainer-images.yml`'s concurrency group is
    `trainer-images-${{ github.workflow }}-${{ github.event.pull_request.number || github.ref }}`
    with `cancel-in-progress: true`. For a `push` to `main`,
    `github.event.pull_request.number` is empty, so the group falls back to
    `github.ref`, which is `refs/heads/main` for **every** push — one shared
    group across all main commits. A fast follow-up merge cancels the
    previous commit's still-running build/scan/sign/publish job outright.
    This fails closed (nothing unsigned or unscanned is ever promoted), but
    under high merge velocity to `library/**`/`application/backend/**`/
    `application/plugin/**`/`Dockerfile.trainer` it can silently skip
    promoting `protocol-<N>` for an intermediate commit, and wastes
    CI minutes re-running the same build repeatedly. Since there is no
    `latest` fallback, a skipped promotion now fails SSH training jobs outright
    (with an actionable message) rather than silently degrading — surfacing
    this gap sooner, which is the point, but also raising the cost of leaving
    it unfixed. **Follow-up:** key the
    `push` case on `github.sha` instead of `github.ref` (each merge gets its
    own group and always runs to completion) and only set
    `cancel-in-progress: true` for the `pull_request` case, where cancelling a
    superseded PR run is actually correct.
18. **Trainer image promotion is not gated on `library.yml`/`backend.yml`
    passing for the same commit** — `build-and-smoke-test` in
    `trainer-images.yml` only asserts image metadata (non-root user,
    entrypoint, the `api-protocol` label, no Studio/UI/Docker-socket content)
    and that `/health` returns the expected shape; it never runs an actual
    training step. `library.yml` and `backend.yml` run the real unit/
    integration suites as **separate workflows** with no `needs:` dependency
    from `trainer-images.yml`. A `library/**` commit that fails
    `library.yml`'s own tests can still promote a fresh `protocol-<N>`
    trainer image — the exact tag every SSH job auto-resolves — with
    no real training validation in between. Higher trainer-image update
    frequency raises the odds of this happening before anyone notices.
    **Follow-up:** gate the `publish` job's tag-promotion step on the
    `library.yml`/`backend.yml` conclusion for the same commit SHA (e.g. a
    `workflow_run` check, or restructuring so `trainer-images.yml` depends on
    their result), not just on its own smoke test.
19. **Layer-cache reuse on remote hosts is weaker than assumed, because of the
    editable path dependency** — `physicalai-train` is
    `{ path = "../../library", editable = true }`
    (`application/backend/pyproject.toml`), so `Dockerfile.trainer` must copy
    the *entire* `library/` source tree into the builder stage
    (`COPY --link --from=libs / /app/library`) **before** `uv sync`. Since
    `library/**` is one of `trainer-images.yml`'s trigger paths, nearly any
    library change invalidates that layer and therefore the multi-GB
    torch/CUDA venv layer built on top of it — "later jobs reuse cached
    layers" (Step 3, Confirmed Decisions "First-job cost") only holds between
    jobs that happen to share a digest. The higher the trainer's update
    frequency, the more often remote hosts re-pull that large layer instead of
    just on a server's first job. This is independent of the tagging scheme
    (any resolution strategy pays this cost once it picks a new digest) but
    is the concrete price of frequent trainer updates and should be measured,
    not assumed away. **Follow-up:** consider a registry-backed BuildKit cache
    (`cache-from`/`cache-to`) for CI build time, and budget/monitor the
    `image_pull` phase window (Step 4/Progress Reporting) against real pull
    times on a representative remote host rather than assuming cache hits.
20. **GHCR retention may garbage-collect every trainer tag, `protocol-<N>`
    included** — `cleanup-old-app-images.yml` runs weekly against
    `physicalai-trainer-cuda` and `physicalai-trainer-xpu` with
    `min-versions-to-keep: 10`, delegating to the shared `geti-ci`
    `cleanup-images` action. Its documented retention allowlist is release
    semver (`X.Y.Z`), release candidates (`X.Y.ZrcN`), and `latest`. Since the
    trainer packages publish **none** of those — only
    `<version>-dev-<short-sha>`, `main`, and `protocol-<N>` — *no* trainer tag
    is allowlisted, so all of them fall inside the `min_versions_to_keep`
    window. Whether a tagged version is actually collectable depends on how
    that shared action treats tagged-but-not-allowlisted versions, which this
    plan must not guess at. If it is collectable, the single tag every SSH job
    resolves can be deleted by a scheduled job, and — with no fallback tag —
    that breaks all SSH training until the next trainer-relevant commit
    republishes it. **Follow-up:** confirm the action's behavior for these tag
    shapes and, if needed, add `protocol-*` (and `main`) to the retention
    allowlist. A note recording this is already in
    `cleanup-old-app-images.yml`.
21. **Exact-SHA image matching was reconsidered and re-rejected** — building
    the trainer on every commit and resolving `:<git-sha>` looks like a
    stronger guarantee, but it fails on four independent axes. (a) Studio
    cannot reliably know its own SHA at runtime: it ships as a PyPI package
    and as containers, neither carrying `.git`, and a dev checkout with
    uncommitted changes maps to no published build. (b) `trainer-images.yml`'s
    publish job is gated `if: github.ref == 'refs/heads/main'`, so no branch or
    PR commit ever has an image — every developer running Studio from a branch
    would be permanently unable to run SSH jobs, with no fallback to soften it.
    (c) Retention (risk 20) keeps only the 10 most recent non-release versions,
    so SHA tags are deleted within days of frequent builds, permanently
    breaking any Studio build older than that window. (d) It requires removing
    the `paths:` filter, so every docs/UI commit rebuilds two multi-GB
    scan-and-sign images and produces a new digest that every remote host must
    re-pull (risk 19). It is also not more correct: per-job reproducibility is
    already guaranteed by resolving and persisting an immutable digest, so SHA
    tagging adds cost without adding a guarantee. **If explicit pinning is
    wanted, add it as an opt-in per-server image ref/digest override**, not as
    the default resolution mechanism.
22. **Composite `protocol-<N>-lib-<X>` tags were considered and rejected** —
    encoding the library version in the tag would make library compatibility a
    resolution key rather than a post-resolution check, but tags are matched by
    **equality** while library compatibility is a **range** (newer is normally
    fine; older is the risk), so the tag cannot express the actual rule. It
    would also be near-useless in practice today, since `library/pyproject.toml`
    is a hand-maintained `0.1.0` — the same shape already rejected for
    `../../VERSION` — and it would make the tag namespace combinatorial,
    multiplying both the unpublished-tag windows that the no-fallback decision
    converts into hard failures and the number of moving tags needing retention
    protection. The OCI-label + registry-inspect approach (Step 3) achieves the
    same pre-pull check with range semantics and no new tags.
23. **Vulnerability scanning is no longer a publish gate for trainer images** —
    the Trivy step was removed from `trainer-images.yml` and now lives in
    `security-scan.yml` as `trivy-trainer-image-scan`, which runs only on
    `schedule` (nightly) or `workflow_dispatch`, against
    `physicalai-trainer-<device>:main`. Publishing and tag promotion therefore
    complete without any vulnerability scan blocking them, so `protocol-<N>` —
    the tag Studio auto-resolves for every SSH job, with no fallback behind it
    — can point at an image that has not been scanned yet, for up to a day.
    Signing and metadata verification *are* still gates (moving tags are
    promoted only after signing), so this is a vulnerability-freshness gap, not
    a provenance gap. The nightly job also scans `:main` rather than
    `protocol-<N>`; these are the same digest today because both are promoted
    from the same build, but that is a coincidence of the current scheme rather
    than something enforced. **Follow-up:** decide whether SSH provisioning
    requires a scanned image, and if so either restore a blocking scan before
    tag promotion or have the nightly job scan `protocol-<N>` and gate
    provisioning on a recorded scan result.
