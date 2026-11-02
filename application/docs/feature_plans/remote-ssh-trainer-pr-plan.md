# Remote Server SSH Trainer Provisioning — PR Plan

Dependency-ordered, independently reviewable pull requests for
[`remote-ssh-trainer-plan.md`](remote-ssh-trainer-plan.md) (backend) and
[`remote-ssh-trainer-ui-plan.md`](remote-ssh-trainer-ui-plan.md) (UI).

## Status

- **PR 1 — landed on `main`.** Trainer image targets, GHCR publishing with SBOM /
  provenance / cosign signing, OCI labels, and `/health` protocol
  metadata all exist (`../../docker/Dockerfile.trainer`,
  `../../../.github/workflows/trainer-images.yml`,
  `../../backend/src/trainer/schemas.py`).
- **PR 5 — partially landed.** A flag-gated route and page exist, but the route is
  project-scoped and reuses the remote-_trainers_ page. See PR 5.
- All other PRs are not started. `asyncssh` is not yet a dependency.

## Dependency overview

```text
PR 0 (decisions/contracts)
 ├─ PR 2 (remote-server persistence by SSH host alias)
 │   └─ PR 3 (asyncssh transport + safe command/preflight primitives)
 │       └─ PR 4 (remote-server CRUD/status API)
 │           └─ PR 5 (training-targets management UI)
 └─ PR 6 (job payload/provisioning persistence + progress contract)
     └─ PR 7 (SSH Docker provisioning service)
         └─ PR 8 (worker/backend integration + unified target selector)
             └─ PR 9 (recovery, cancellation, observability hardening)
                 └─ PR 10 (backend enablement switch, docs, rollout)

PR 1 (trainer images + CI supply chain) — already on main
```

---

## PR 0 — Architecture decisions and compatibility contracts

**Purpose:** Resolve decisions which otherwise cause backend and UI churn.

### Scope

Record as ADRs / an implementation design record, from
[Confirmed Decisions](remote-ssh-trainer-plan.md#confirmed-decisions):

- **Deployment model: Studio on the user's own workstation.** Containers are
  secondary and mount the same `~/.ssh`. This is the decision the credential
  design follows from.
- **Credentials: the user's existing `~/.ssh/config`.** `RemoteServerDB` stores
  one non-secret `ssh_host_alias`. No encryption layer, no secret injection.
  Passphrases go through the SSH agent; host keys through `known_hosts`.
- **SSH library: `asyncssh`** — asyncio-native, parses `~/.ssh/config`, verifies
  `known_hosts`, and its local-forward API supports reconnect-and-resume without
  a thread bridge.
- Backend-selection precedence keyed on a **new `TrainingTarget.SSH` member**
  (not on "`remote_server_id` is not None").
- The **execution target key** `ssh:<remote_server_id>` and the required
  `TrainingWorker._target_key` change.
- **Backend restart: reattach**, before the orphan sweep.
- **Tunnel drop: reconnect and resume**, one code path with reattach, bounded
  retry budget.
- **Trainer image resolution key: Studio's compiled-in trainer protocol
  version, resolved via the `protocol-<N>` moving tag** — not the Studio build
  revision (git SHA) and not `../../VERSION`. `trainer-images.yml` only
  rebuilds the trainer image on commits touching trainer-relevant paths, so
  most Studio commits never produce a matching revision-tagged trainer image;
  resolving by exact SHA (or the `0.1.0`-shaped `VERSION` semver, which can
  never match a SHA tag either) would guarantee an almost-permanent silent
  fallback situation. `protocol-<N>` tracks wire compatibility instead of the
  commit that happened to build it, so it stays resolvable across the commits
  that don't touch the trainer. **There is no fallback tag**:
  `trainer-images.yml` publishes no `latest`, and the only moving alternative
  (`main`) tracks the newest build regardless of protocol, so it is exactly the
  stale, wire-incompatible image to avoid — fail the job immediately instead.
- Trainer protocol contract: **direct-URL trainers grandfathered** when they
  report no version, **SSH-provisioned images held strictly**. This covers
  wire-schema compatibility only, **not** training-logic parity: since
  `library/**` is in `trainer-images.yml`'s `paths:` filter, a library-only
  change republishes under the same `protocol-<N>` with no version bump, so
  Studio's own build can drift older or newer than what currently resolves.
  Publish the library version as an OCI label
  (`org.open-edge-platform.physicalai.trainer.library-version`, from
  `importlib.metadata`) and range-check it off the registry manifest **before
  pulling**: older than Studio's → non-fatal warning; a `policy` with a
  documented minimum → hard fail scoped to that policy. `/health` re-reports it
  as defense-in-depth. **Rejected:** encoding it in the tag
  (`protocol-<N>-lib-<X>`) — tags match by equality, library compatibility is a
  range, and the namespace would become combinatorial. **Also rejected:**
  exact-SHA image matching (Studio can't know its SHA at runtime, branch
  commits never publish, retention deletes SHA tags).
- Managed-container labels, `backend_instance_id` ownership, and orphan-sweep
  criteria (label match alone is insufficient; reattach claims win).
- **GPU busy: stay `pending` with backoff** and a give-up timeout, expressed as a
  `waiting` **phase state** in `extra_info["phase"]` — **not** a new `JobStatus`
  member, which would break every generated UI consumer.
- **Per-target progress phase table**, so local and direct-URL curves and their
  existing assertions stay untouched.
- Bastion / `ProxyJump` is out of scope as a feature; a user's existing config
  may supply it via `asyncssh`.

Also define shared constants/types for device types, ordered progress phases
(versioned, shared with the UI), and image/provisioning result schemas.

Produce the **threat model** for Docker-daemon-as-root, including the point that
reusing the user's SSH agent gives a compromised Studio process reach beyond the
registered servers.

### Exclusions

- No migration, SSH connection, container launch, or UI work.

### Acceptance criteria

- Image resolution and host-key trust behavior have unambiguous failure paths.
- Reattach, tunnel-reconnect, and GPU-busy-pending behaviors are each specified
  with their failure branches and give-up conditions.
- The `JobStatus`-unchanged and per-target-phase-table decisions are recorded with
  their rationale, so neither is "simplified" back later.

---

## PR 1 — Dedicated trainer images and image supply chain — **LANDED ON MAIN**

### Delivered

- Non-root `physicalai-trainer-cuda` / `physicalai-trainer-xpu` build targets in
  `../../docker/Dockerfile.trainer`, entry point `physicalai-trainer`, containing no
  backend or UI code.
- `../../../.github/workflows/trainer-images.yml`: an immutable
  `<version>-dev-<short-sha>` tag pushed at build time, then moving `main` and
  `protocol-<N>` tags promoted onto that verified, signed digest **after**
  signing, plus `sbom: true`, `provenance: mode=max`, a metadata
  verification step, and cosign signing. **No `latest` tag is published.**
- OCI labels `org.opencontainers.image.source`, `.revision`, `.version`,
  `.created`, and `org.open-edge-platform.physicalai.trainer.api-protocol`.
- `/health` returns `status`, `protocol_version`, `device_type`,
  `build_revision`, `build_date`, and `application_version` via `HealthInfo`.

> **Note:** the `protocol-<N>` tag and the post-signing promotion currently sit
> on `albert/fix-trainer-images-trivy-egress`, not yet on `main`. The same
> branch moved trainer Trivy scanning out to `security-scan.yml`'s
> `trivy-trainer-image-scan`, which runs nightly/on-dispatch against `:main`
> rather than gating publish (plan risk 23).

### Remaining

The `protocol-<N>` moving tag is **done** (promoted post-signing, so it can
never point at an unsigned image). Backend-side **resolution** of these images
is PR 7. Two CI/image changes remain, because PR 7 depends on both:

- **Confirm `protocol-*` survives `cleanup-old-app-images.yml`.** Its retention
  allowlist is release semver, RC tags, and `latest`. The trainer packages
  publish none of those, so *no* trainer tag is allowlisted and the weekly job
  keeps only the 10 most recent versions. If the shared `geti-ci`
  `cleanup-images` action can collect tagged versions, add `protocol-*` (and
  `main`) to the allowlist — with no fallback tag, a collected `protocol-<N>`
  breaks every SSH job.
- **Add an `org.open-edge-platform.physicalai.trainer.library-version` OCI
  label** to both trainer targets, from
  `importlib.metadata.version("physicalai-train")` at build time (not the
  hand-maintained `library/pyproject.toml` `0.1.0`). Extend the existing
  metadata-verification step to assert it is present and non-`unknown`, and the
  smoke test to assert `/health` reports the same value. PR 7's pre-pull
  library range-check reads this label off the registry manifest.

---

## PR 2 — Remote-server persistence by SSH host alias

**Purpose:** Establish the durable domain model without making network
connections.

### Scope

- `RemoteServerDB` + a **rewritten** `d4f8a1c9b3e6_add_remote_servers.py`
  (edited in place — it never reached `main`, so there is no released schema and
  no legacy data): `id`, `name`, `ssh_host_alias` (unique, non-secret),
  `device_type`, `last_check_*`, `created_at`, `updated_at`.
- **Deliberately absent:** `username`, `auth_type`, `ssh_secret_encrypted`,
  `ssh_key_passphrase_encrypted`, `host_key`, `host`, `port`, and the
  `uq_remote_servers_host_port_username` constraint. Host/port/user are derived
  from the SSH config at read time so they cannot silently disagree with it.
- `JobProvisioningDB` keyed by `job_id` (not the payload JSON) with
  `remote_server_id`, `image_ref`, `image_digest`,
  `container_id`, `container_name`, `remote_port`, `local_tunnel_port`,
  `ssh_host_alias`, `trainer_build_version`, `trainer_protocol_version`.
- Repositories, mappers, schemas, and `services/remote_server_service.py`.
  **No internal-vs-public schema split and no `to_public()` sanitizer** — no
  field is secret.
- An **SSH config reader**: list selectable `Host` aliases (excluding wildcard
  patterns), and resolve one alias to its effective hostname/port/user for
  display. Read-only; never returns `IdentityFile` contents.

### Exclusions

- No SSH connection. The config reader parses a file; it does not dial.
- The CRUD router is PR 4.

### Security gate

- No schema, model, or response has a private-key, password, or passphrase field.
- The config reader cannot be coerced into returning key file contents.

### Dependencies

- PR 0

### Acceptance criteria

- Tests prove the schema has no secret field and the DB holds only an alias.
- The config reader handles a missing config, an empty config, wildcard-only
  stanzas, and an alias that resolves through `Include`.
- **An explicit test asserts the SSH config reader's response never includes
  `IdentityFile`, `IdentityAgent`, `CertificateFile`, or any `Password` field**,
  using a fixture config where each of these directives is present on the
  resolved `Host` stanza — not just an absence-by-omission check on a config
  that never had them.
- Existing jobs and local/direct-URL behavior are unaffected.

---

## PR 3 — SSH transport, validation, and preflight primitives

**Purpose:** Build the security-critical remote execution boundary before
exposing it through an API or job flow.

### Scope

- Add `asyncssh` and an async SSH transport abstraction over it, configured with
  the user's SSH config path and `known_hosts`.
- Strictly validate server configuration: alias format, device type. (There are
  no credential sizes or auth types to validate — `asyncssh` owns that.)
- **Host-key verification is `asyncssh` against `known_hosts`.** No TOFU
  implementation, no pinning, no host-key column. Map an unknown host and a
  changed host key to two distinct actionable errors.
- Map a passphrase-protected key with no usable agent to its own actionable
  error.
- Run every remote command as an argument array; never interpolate a shell
  string.
- Add bounded timeouts, cancellation, per-server preflight throttling, output
  sanitization, output caps, keepalives, and heartbeats.
- **Tier 1** (cheap, save-gating): alias resolution, reachability/auth, Docker
  access, disk capacity, CUDA/XPU host detection, registry reachability (a
  manifest `HEAD`, no pull). Also **reports** GPU occupancy, which is
  non-blocking — a busy GPU is transient, must not stop a user editing the
  target, and must not contradict the train dialog keeping busy targets
  selectable. Return blocking and non-blocking results distinguishably.
- **Tier 2** (expensive, explicitly invoked): image resolve/pull, signature
  policy, in-container CUDA/XPU device checks, trainer protocol compatibility.
  Separately callable so PR 4 can gate saves on Tier 1 only.
- XPU host detection is **two probes** — `xpu-smi`, else Intel render nodes
  (`/dev/dri/renderD*` + vendor `0x8086`). Tier 2's in-container
  `torch.xpu.is_available()` is authoritative.

### Security gate

- Tests use shell metacharacters in all configurable values and prove no command
  injection.
- Host-key verification failure blocks all operations.
- Remote command output is sanitized before it reaches API responses or job
  messages.

### Dependencies

- PR 2

### Acceptance criteria

- Tests cover a resolvable alias, a missing alias, a wildcard-only alias, an
  unknown host key, a changed host key, and a passphrase-protected key with no
  agent.
- CUDA/XPU probe results identify the successful detection method.
- No API or training path uses the transport yet.

---

## PR 4 — Remote-server CRUD and status API

**Purpose:** Expose remote-server administration and verify-on-save
independently from training execution.

### Scope

- Add global remote-server CRUD routes **and wire the router** — `main.py` and
  `api/dependencies.py` currently register only `remote_trainers_router`.
- Add an endpoint listing selectable SSH host aliases for the create/edit form.
- **Tier 1 preflight gates create/update** with a bounded timeout. **Tier 2
  (registry pull, signature policy, in-container `torch.{cuda,xpu}.is_available()`)
  must be a separate async action**, never inline in the save request — it can
  pull multiple gigabytes.
- Reject saving a server whose **blocking** Tier 1 checks fail. A busy GPU is
  reported but still saves.
- A transient Tier 1 failure updates `last_check_*` and marks the server
  unhealthy; it must never delete the record.
- Add a structured status/check endpoint: alias resolvable, reachable,
  authenticated, host key verified, Docker, registry, driver + detection method,
  container probe, compatible image, last-check time, and busy/in-use status.
  **Tag each result with its tier and its own last-checked time**, so the UI can
  group them and avoid presenting a stale Tier 2 result as current.
- Map each credential-adjacent failure to a **distinct, actionable** error rather
  than a 500: alias not found, host fingerprint not accepted, passphrase needs an
  agent. Update OpenAPI output.

### Security gate

- API responses and errors never include host keys, raw SSH exceptions, remote
  command text, or any path to a key file.

### Dependencies

- PR 3

### Acceptance criteria

- API tests cover healthy create/check/edit/delete paths.
- **An explicit test asserts the SSH alias-listing endpoint's response never
  includes `IdentityFile`, `IdentityAgent`, `CertificateFile`, or any
  `Password` field**, using a fixture config where each directive is present on
  the resolved `Host` stanza.
- A blocking preflight failure cannot create a usable server.
- **A save request never performs an image pull** — assert on the transport, not
  just on timing, so the guarantee cannot regress silently.
- A save succeeds when the only failing Tier 1 result is GPU occupancy.
- Each of the three actionable credential errors is returned with its own code.
- Existing training APIs remain unchanged.

---

## PR 5 — Training-targets management UI — **PARTIALLY LANDED, NEEDS CORRECTION**

**Purpose:** Deliver training-target management independently from training-job
dispatch.

Full detail in [`remote-ssh-trainer-ui-plan.md`](remote-ssh-trainer-ui-plan.md).

### Already present

- The `remoteTrainers` build-time flag in
  `../../ui/src/config/feature-flags.ts`.
- A flag-gated route in `router.tsx` and `routes/remote-servers/index.tsx`
  rendering `features/remote-trainers/remote-trainers-page.tsx`.

### Corrections required

- **Promote the route to global** — `/training-targets` at top level with its own
  primary-nav entry, outside `ProjectLayout`. It is currently
  `project.path('/remote-servers')`, but neither model has a project FK.
- **Resolve the naming split** — `routes/training-targets/` (thin shell) +
  `features/training-targets/` (logic), per `../../ui/AGENTS.md`.

### Remaining scope

- List/detail layout with type badge, device type, health, and in-use /
  waiting-for-GPU status.
- SSH create/edit form with an **alias select** (populated from PR 4's endpoint)
  and resolved host/port/user shown read-only. **No key, password, or passphrase
  field exists.**
- Tier 1 feedback on save and an explicit test-connection action running Tier 2
  with progress.
- Structured preflight status, including "protocol unknown" for grandfathered
  direct-URL trainers and the three actionable credential states with inline
  recovery instructions.
- Generated `$api` hook usage and UI tests.

### Security gate

- No form field accepts secret material.
- Client logs and notifications must not include submitted form values.

### Dependencies

- PR 4

### Acceptance criteria

- Tests cover list/detail, form validation, the alias select (including an empty
  SSH config), and healthy/unreachable/misconfigured/busy/checking states.
- Remote jobs still cannot be launched at this stage.

---

## PR 6 — Training job contract and progress framework

**Purpose:** Prepare job orchestration and UI-compatible telemetry without
provisioning a remote container.

### Scope

- Add `remote_server_id` to `TrainJobPayload`.
- **Add `TrainingTarget.SSH`** (`schemas/job.py` has only `LOCAL`/`REMOTE`) and
  extend `validate_training_target` so an SSH job requires `remote_server_id` and
  forbids `remote_trainer_id`/`remote_trainer_url`. Overloading `REMOTE` is not
  acceptable: `get_training_backend` raises when `remote_trainer_url is None`, and
  the worker's `reattaching` check would misfire.
- **Update `TrainingWorker._target_key`** to return `ssh:<remote_server_id>`. It
  currently returns `remote:<remote_trainer_id>` for anything non-local, mapping
  every SSH job on every server to the single key `remote:None`.
- Audit every existing `is TrainingTarget.REMOTE` / `is TrainingTarget.LOCAL`
  branch for the new third case.
- Validate remote-server existence, alias resolvability, and successful preflight
  at job submission.
- Add the ordered phase table and `report_phase` helper: connect, image pull,
  image verification, trainer start, upload, train, download. Phase keys in a
  versioned constant shared with the UI.
- **Per-target phase table** — local and direct-URL backends keep their existing
  `SNAPSHOT_UPLOAD_PROGRESS = 10` / `TRAINING_PROGRESS_END = 95` windows and
  their existing assertions. Do not retune shared constants.
- **No new `JobStatus` member.** The GPU-busy wait is a `waiting` phase state in
  `extra_info["phase"]`; the job stays `pending`.
- Attach additive `extra_info["phase"]` descriptors, budgeting for the existing
  16 KB cap.
- Refactor `RemoteTrainingBackend` to accept an injected endpoint/device while
  preserving existing direct-URL behavior.

### Exclusions

- No SSH transport, tunnels, or remote Docker operations.

### Dependencies

- PR 0, PR 2

### Acceptance criteria

- Local and direct-URL jobs retain their existing behavior **and their exact
  existing progress values**.
- Local jobs remain valid without a phase descriptor.
- A test asserts `JobStatus` members explicitly, so a future change cannot
  silently break generated UI consumers.
- Target-key tests prove distinct keys per server and no `None` in any key.
- A remote-server ID is storable but does not yet change backend selection.

---

## PR 7 — SSH Docker provisioning service

**Purpose:** Implement secure lifecycle management for a per-job trainer
container behind a dedicated service boundary.

### Scope

- Resolve Studio's own compiled-in trainer protocol version, require the
  device-specific `protocol-<N>` trainer image tag (**no fallback tag**),
  and fail the job with an actionable message + log if that tag cannot be
  resolved. Not the Studio build revision/git SHA (the trainer only rebuilds on
  trainer-relevant path changes, so it rarely matches) and not `VERSION`
  (`0.1.0` can never match a SHA or protocol tag either).
- Resolve an immutable digest and verify image identity/signature policy before
  use for the resolved `protocol-<N>` tag. Verify with
  `cosign verify`, pinning the certificate identity to the
  Studio release workflow and the Sigstore OIDC issuer, e.g.:

  ```sh
  cosign verify open-edge-platform/physicalai-trainer-<device>:<version> \
    --certificate-identity-regexp 'https://github\.com/open-edge-platform/physical-ai-studio/\.github/workflows/.+' \
    --certificate-oidc-issuer "https://token.actions.githubusercontent.com"
  ```

  Fail closed (do not launch the container) if verification fails or cosign is
  unavailable.
- Check GPU availability before launch; when busy, leave the job `pending` with
  exponential backoff and a `waiting` phase state, a per-server probe throttle,
  and a give-up timeout.
- Re-check remote free disk against this job's actual snapshot size.
- Pull and launch **by digest only**.
- Validated deterministic container name, management/job/server labels, a
  **`backend_instance_id` ownership label**, `--restart=no`, bounded
  `--stop-timeout`, a container-side watchdog, and an ephemeral loopback-only
  host port.
- Least privilege: non-root, dropped capabilities, no `--privileged`, minimal
  device passthrough, bounded writable storage.
- Create and close the SSH local-forward tunnel with keepalives. A dropped tunnel
  **reconnects and resumes** against the still-running container within a bounded
  retry budget; it does not fail the job.
- Verify container readiness and `/health` metadata before upload, rejecting an
  SSH image that reports no protocol version.
- **Before pulling**, read the `library-version` label off the registry
  manifest (the same `imagetools inspect` call that resolves the digest) and
  range-check it against Studio's own installed `physicalai-train` version:
  older → non-fatal warning in job status; equal/newer → proceed; a `policy`
  with a documented minimum it doesn't meet → fail before the pull, naming the
  policy and required version. Re-confirm from `/health` after launch as
  defense-in-depth; a label/`/health` disagreement fails the job.
- Persist container/tunnel state before accepting work.
- Tear down containers and tunnels in `finally` on unrecoverable failure and
  cancellation.
- Orphan sweeping requiring all management labels **and** a matching
  `backend_instance_id` **and** no reattach claim, plus sanitized/capped
  provisioning output.

### Security gate

- The container launches by resolved digest, never a mutable tag.
- The trainer is bound only to remote `127.0.0.1`.
- Orphan cleanup only touches containers this deployment provably owns and that
  no active job has claimed.
- CUDA/XPU in-container device check failure blocks the job before dataset
  upload.

### Dependencies

- PR 3, PR 6, and PR 1's remaining CI work (the `protocol-<N>` tag, its
  retention protection, and the `library-version` label — PR 7 resolves the
  first and reads the last).

### Acceptance criteria

- Tests cover `protocol-<N>` tag resolution with no fallback tag,
  protocol-version resolution with no `.git` present, unresolvable-tag
  failure, pull
  failures, digest persistence, loopback binding, protocol mismatch (strict for
  SSH), library-version range-check read from the manifest **before any pull**
  (older → warning not failure; equal/newer → silent; policy minimum unmet →
  pre-pull failure; label disagreeing with `/health` → failure), cleanup
  at every failure point, cached images, output handling,
  tunnel-drop reconnect, GPU-busy pending/backoff/give-up, and non-destructive
  orphan sweeping including a foreign `backend_instance_id`.
- At least one integration test runs against a containerized `sshd` with a
  generated key and a purpose-built SSH config, rather than mocks.

---

## PR 8 — Enable server-aware training execution

**Purpose:** Connect provisioning to the worker and expose target selection in
the train dialog.

### Scope

- Resolve the selected server in `TrainingWorker`.
- `get_training_backend(payload)` precedence: `SSH` → SSH provisioning;
  `REMOTE` → pinned-URL registry backend; otherwise local.
- Wrap remote training with provision → HTTP dataset transfer through tunnel →
  training/download → teardown.
- Extend cancellation to cancel both the remote trainer job and provisioning
  resources, using the job's own entry in `TrainingWorker.job_interrupt_flags`
  (never a shared interrupt signal).
- Use the selected server's configured device type instead of a live `/devices`
  probe.
- **Unify the training-target selector** in `train-model-dialog.tsx` rather than
  adding a second dropdown — the dialog already queries `/api/remote-trainers`
  and sets `training_target` + `remote_trainer_id`. One control listing local,
  direct-URL, and SSH targets with type badges and status. Submit is gated on
  unreachable/misconfigured targets only; a **busy target stays selectable** with
  a pre-submit notice that the job will wait, because the job queues rather than
  failing. Remove the local-vs-remote mode toggle — local becomes an entry in the
  same list, which is what makes the derived `training_target` unambiguous.
- Render the progress stepper from `extra_info.phase`, degrading to the current
  bar-only view when absent.

### Security gate

- A `remote_server_id` must never use the direct-URL `remote_trainer_url` path.
- SSH-provisioned jobs force the existing streamed HTTP transfer through the
  tunnel.
- Cancellation and failures always trigger teardown.

### Dependencies

- PR 5, PR 6, PR 7

### Acceptance criteria

- Integration tests prove SSH selection takes precedence over the direct-URL
  path.
- The trainer is contacted only through the local tunnel.
- Containers are removed after success, failure, and cancellation.
- Direct-URL and local paths remain functional, including their progress curves.
- The dialog exposes **one** target control — assert there is no second
  remote-server dropdown and no local/remote mode toggle, since reintroducing
  either is the most likely regression here.
- Selecting each target type derives the correct `training_target` and id field.
- A busy target is selectable and submits; an unreachable one does not.
- The stepper renders each phase state including `waiting`, and a job with no
  `phase` renders the current bar-only view unchanged.

---

## PR 9 — Recovery, operational hardening, and observability

**Purpose:** Close lifecycle gaps that cannot be addressed in isolated
provisioning tests.

### Scope

- Implement **startup reattach**: for each `JobProvisioningDB` row on a
  non-terminal job, resolve the persisted alias, verify the host key, confirm the
  container exists/runs/is owned, re-open the tunnel on a fresh local port,
  re-verify `/health` and digest, and resume streaming. Implement every failure
  branch explicitly (container gone, health never ready, digest mismatch,
  host-key failure, **alias missing from the SSH config**, port unreachable).
- Run reattach **before** the orphan sweep, and exclude reattach-claimed
  containers from sweeping.
- Add the startup/recovery orphan sweep from persisted job state, trusted labels,
  and `backend_instance_id` ownership.
- Reconcile stale tunnel/container records.
- Implement the GPU-busy backoff, per-server probe throttle, and give-up timeout
  (the 0.5 s `run_loop` poll must not become a 2 Hz SSH probe).
- Bounded retry and timeout policies for SSH, Docker, tunnel readiness, tunnel
  keepalive/reconnect, and trainer health.
- Structured operational logs and metrics for preflight, image
  resolution, provisioning duration, cleanup, reattach outcomes, orphan
  recovery, and GPU-busy waits.
- Improve error attribution in job messages and phase state.

### Security gate

- Review logs, metrics, traces, and job messages for host-key and
  untrusted-output leakage.
- If trusted SSH identity or container ownership cannot be established, recovery
  must not perform destructive cleanup.
- A host-key verification failure during reattach fails closed and does **not**
  tear down the container.

### Dependencies

- PR 8

### Acceptance criteria

- A simulated backend restart reattaches a still-running container, re-opens the
  tunnel, and resumes streaming without losing training progress.
- Each reattach failure branch produces the specified outcome, including a
  renamed or deleted SSH host alias.
- Unrelated containers and containers owned by another Studio instance remain
  untouched.
- A job waiting on a busy GPU is visibly `pending`, backs off, and eventually
  times out.

---

## PR 10 — Backend enablement switch, documentation, and staged rollout

**Purpose:** Make the feature supportable and bound its blast radius.

### Scope

- Document the `~/.ssh/config` contract: what a usable `Host` entry looks like,
  the SSH agent requirement for passphrase-protected keys, accepting a host
  fingerprint before first use, recovery when an alias is removed or renamed, and
  that **Studio stores no SSH credentials whatsoever**.
- Document host setup for CUDA/XPU, XPU limitations, image verification,
  cleanup/recovery procedures, and the direct-reachability (no bastion)
  assumption.
- For the containerized deployment, document mounting `~/.ssh` and/or exposing
  `SSH_AUTH_SOCK`, and that every instance eligible to reattach a job needs the
  same alias resolvable.
- **Document the trust assumption plainly:** Studio has no auth model, so an
  exposed instance means anyone who can reach the API can execute code as root on
  every registered server. Also document that a compromised Studio process can
  reach every identity in the user's SSH agent, not just the registered servers —
  recommend a dedicated per-host `IdentityFile`.
- **Enforce loopback-only binding at startup when the SSH feature is enabled.**
  At backend startup, when the SSH remote-trainer feature flag is on, inspect the
  configured bind host/interface (not just the documented default) and verify it
  resolves only to a loopback address (`127.0.0.1` / `::1`). If a non-loopback
  bind is detected: log a clear warning (and surface it in the UI, e.g. a banner
  on the training-targets page) stating that the instance is network-exposed
  with no auth model and every registered server is at risk; **refuse to start
  the SSH feature** (fail closed — disable SSH routes/registration, or exit,
  whichever is technically feasible given the ASGI server's binding model)
  rather than merely warning, if the check can be performed reliably. Treat this
  as defense-in-depth on top of, not a replacement for, the documentation and the
  "not enabled without an auth model" policy.
- Add an integration-test matrix and hardware-validation checklist for CUDA, XPU,
  host-key failure, missing alias, cache reuse, cancellation, tunnel
  drop/reconnect, GPU-busy waiting, and backend restart/reattach.

### Security gate

- Complete the PR 0 threat model review before any network exposure. The feature
  is safe-by-default on a localhost workstation; it must not be enabled on a
  network-exposed instance without an auth model.
- **The loopback-binding check runs whenever the SSH feature flag is enabled**
  and cannot be bypassed by a UI-only toggle — it inspects the actual bind
  configuration at process startup, not a config value that merely claims
  loopback.
- The warning/refusal path itself must not leak SSH host aliases, container
  names, or other registered-server details into logs reachable pre-auth.

### Dependencies

- PR 9

### Acceptance criteria

- Users can configure, test, diagnose, and decommission servers using the
  documentation.
- The feature can be disabled without impacting local or direct-URL training.
- **A test binds the backend to a non-loopback interface with the SSH feature
  enabled and asserts** the warning is logged, the UI is informed (or the
  startup refuses/disables the SSH feature, per whichever behavior is
  implemented), and the same startup bound to `127.0.0.1`/`::1` produces neither
  the warning nor the refusal.

---

## Recommended merge and release order

1. Merge PRs 0, 2–6 with remote execution not yet reachable. (PR 1 is on `main`.)
2. Merge PR 7 and validate against disposable CUDA and XPU hosts.
3. Merge PR 8 with the UI selector feature-flagged.
4. Merge PRs 9–10.
5. Do not enable this feature for any deployment that is not a single-user
   localhost workstation, pending an auth model.

This order ensures the SSH trust boundary, image identity, and Docker lifecycle
boundaries are reviewed and tested before the product can launch any remote
training container.
