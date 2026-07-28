# Remote Server SSH Trainer Provisioning — PR Plan

This document divides [`remote-ssh-trainer-plan.md`](remote-ssh-trainer-plan.md) into dependency-ordered, independently reviewable pull requests.

## Status

- **PR 2 — merged** on `albert/ssh-server-persistence` (`RemoteServerDB`, `JobProvisioningDB`, `core/secret_encryption.py`, migration `d4f8a1c9b3e6`, repos/mappers/schemas/service/API, tests).
- **PR 5 — partially landed ahead of its dependencies.** A feature-flagged route and page already exist, but the route is project-scoped and reuses the remote-_trainers_ page. See PR 5 for the corrections required.
- All other PRs are not started. `asyncssh` is not yet a dependency and no trainer image target exists.

## Goals

- Establish trust, persistence, and image supply-chain boundaries before a job can launch a remote container.
- Keep existing local and direct-URL remote-trainer training (`remote_trainer_id` / `remote_trainer_url`) functional throughout implementation.
- Make the highest-risk areas—SSH identity, secret storage, image identity, and Docker lifecycle—small enough for focused review and testing.
- Gate real remote-job execution behind explicit security and operational validation.

## Dependency overview

```text
PR 0 (decisions/contracts)
 ├─ PR 1 (trainer images + CI supply chain)
 ├─ PR 2 (remote-server persistence + secret encryption)
 │   └─ PR 3 (SSH client + safe command/preflight primitives)
 │       └─ PR 4 (remote-server CRUD/status API)
 │           └─ PR 5 (remote-server management UI)
 └─ PR 6 (job payload/provisioning persistence + progress contract)
     └─ PR 7 (SSH Docker provisioning service)
         └─ PR 8 (worker/backend integration + server selector)
             └─ PR 9 (recovery, cancellation, observability hardening)
                 └─ PR 10 (end-to-end validation, documentation, rollout)
```

---

## PR 0 — Architecture decisions and compatibility contracts

**Purpose:** Resolve decisions which otherwise cause backend, trainer-image, and UI churn.

### Scope

- Add ADRs or an implementation design record for the decisions now confirmed in [`remote-ssh-trainer-plan.md`](remote-ssh-trainer-plan.md#confirmed-decisions):
  - backend-selection precedence keyed on a **new `TrainingTarget.SSH` enum member** (not on "`remote_server_id` is not None"): `SSH` selects SSH provisioning; `REMOTE` selects the existing direct-URL registry (`remote_trainer_id`/`remote_trainer_url`); otherwise training is local;
  - the **execution target key** for SSH jobs (`ssh:<remote_server_id>`) and the required `TrainingWorker._target_key` change;
  - **backend restart behavior: reattach** — re-open the tunnel to the persisted `container_id`/`remote_port`, re-verify `/health` and digest, resume streaming; reattach runs before the orphan sweep;
  - **tunnel drop: reconnect and resume**, sharing one code path with reattach, with a bounded retry budget;
  - **build revision source**: a baked-in OCI label / build arg / `../../../VERSION`, since the backend ships containerized with no `.git`;
  - trainer API protocol version and `/health` contract, with **direct-URL trainers grandfathered when they report no protocol version, and SSH-provisioned images held strictly**;
  - OCI image naming, labels, revision-first resolution, and `latest` fallback;
  - Fernet key lifecycle, **including a stored key fingerprint** so records under a lost/rotated key are identifiable before provisioning;
  - TOFU behavior and host-key mismatch handling;
  - managed-container labels, `backend_instance_id` ownership, and orphan-sweep criteria (label match alone is insufficient; reattach claims take precedence);
  - **GPU busy: stay pending with backoff** in a visible `waiting_for_gpu` state with a give-up timeout — not immediate failure;
  - **bastion / `ProxyJump` is out of scope**; all servers are directly reachable.
- Define shared constants/types for device types, SSH auth types, ordered progress phases (versioned, shared with the UI, **one table for all targets**), and image/provisioning result schemas.
- Produce the **threat model** for Docker-daemon-as-root. The Studio backend has **no auth model today** (single trusted local user), so the feature must ship flag-off with a backend-side enablement switch.

### Exclusions

- No database migration, SSH connection, container launch, or UI work.

### Security gate

- A pinned host-key mismatch never falls back to a new key.
- A mutable image tag is never launchable; only a resolved digest is launchable.
- The threat model is reviewed before any PR that can launch a remote container merges.

### Acceptance criteria

- The trainer `/health` metadata and backend compatibility check are specified, including the grandfather rule for direct-URL trainers.
- The image resolution and host-key trust behavior have unambiguous failure paths.
- Reattach, tunnel-reconnect, and GPU-busy-pending behaviors are each specified with their failure branches and give-up conditions.

---

## PR 1 — Dedicated trainer images and image supply chain

**Purpose:** Create the minimal remote trainer artifacts which provisioning will launch.

### Scope

- Add dedicated non-root `physicalai-trainer-cuda` and `physicalai-trainer-xpu` image targets.
- Include only trainer and required library/runtime dependencies, not the Studio backend or UI.
- Set `physicalai-trainer` as the entrypoint.
- Add OCI labels for source repository, Git revision, application/build version, trainer API protocol version, and build date.
- Publish immutable Git-SHA tags and moving `latest` tags in CI.
- Publish SBOMs and add image scanning and signing/attestation.
- Extend trainer `/health` with image/build/protocol metadata.
- Add image smoke tests, plus explicit CUDA/XPU hardware integration jobs where required.

### Security gate

- Images run as non-root.
- Images do not contain SSH credentials, Docker sockets, datasets, model artifacts, or backend/UI code.
- CI fails when required metadata, SBOM, scan, or signing steps are absent.

### Dependencies

- PR 0

### Acceptance criteria

- A SHA-tagged image can be resolved and run.
- `/health` exposes a protocol version usable by the backend.
- Published metadata supports later identity verification.

---

## PR 2 — Remote-server persistence, encryption, and job state — **MERGED**

**Purpose:** Establish the durable domain model without making network connections.

### Delivered

- `RemoteServerDB` + migration `d4f8a1c9b3e6_add_remote_servers.py` with configuration, device type, timestamps, `last_check_*` summary columns, a `uq_remote_servers_host_port_username` constraint, plaintext-internal `host_key`, and encrypted `ssh_secret_encrypted` / `ssh_key_passphrase_encrypted`.
- `JobProvisioningDB` (a dedicated table keyed by `job_id`, not the job payload JSON) with `remote_server_id`, `image_ref`, `image_fallback_reason`, `image_digest`, `container_id`, `container_name`, `remote_port`, `local_tunnel_port`, `trainer_build_version`, `trainer_protocol_version`.
- `repositories/remote_server_repo.py`, `repositories/job_provisioning_repo.py`, mappers, `services/remote_server_service.py`, `api/remote_servers.py`.
- `REMOTE_SERVER_SECRET_KEY` settings validation plus lazy fail-closed Fernet wrappers in `core/secret_encryption.py`.
- Pydantic schemas which cannot serialize `ssh_secret_encrypted`, `ssh_key_passphrase_encrypted`, or `host_key`, with migration, encryption, and serialization tests.

### Follow-ups deferred from this PR

- `remote_server_id` on `TrainJobPayload` and the `TrainingTarget.SSH` member moved to PR 6 (they were listed here but are job-contract concerns).
- **Add a Fernet key fingerprint/version column** (additive migration on `remote_servers`) storing a truncated, non-reversible digest of the active key, so records encrypted under a rotated/lost key are identifiable before provisioning. Confirmed decision; schedule immediately after PR 2.
- Map `RemoteServerSecretKeyMissingError` and key-fingerprint mismatch to actionable API errors rather than a 500.

### Dependencies

- PR 0

### Acceptance criteria

- Tests prove encrypted-at-rest storage.
- Existing jobs and local/static remote behavior are unaffected.

---

## PR 3 — SSH transport, validation, and preflight primitives

**Purpose:** Build the security-critical remote execution boundary before exposing it through an API or job flow.

### Scope

- Introduce an async SSH transport abstraction (`asyncssh` is not yet a dependency and must be added here).
- Strictly validate server configuration: host, port, username, credential sizes, device type, and auth type.
- Implement TOFU host-key behavior and fail-closed verification on subsequent connections.
- Run every remote command as an argument array; do not interpolate a shell command string.
- Add bounded timeouts, cancellation, per-server preflight throttling, output sanitization, output caps, keepalives, and heartbeats.
- Implement **Tier 1** (cheap, save-gating) probes: reachability/authentication, Docker access, disk capacity, CUDA/XPU host detection, GPU availability.
- Implement **Tier 2** (expensive, explicitly invoked) probes: registry/image access, in-container CUDA/XPU device checks, and trainer protocol compatibility. Keep the two tiers separately callable so PR 4 can gate saves on Tier 1 only.
- **Bastion / `ProxyJump` is out of scope** — assume direct reachability. Avoid hard-coding a single-hop assumption into the transport abstraction where that is cheap to avoid.

### Security gate

- Tests use shell metacharacters in all configurable values and prove no command injection.
- Host-key mismatch blocks all operations.
- Remote command output is sanitized before it reaches API responses or job messages.

### Dependencies

- PR 1
- PR 2

### Acceptance criteria

- Tests cover password auth, key auth, passphrase-protected keys, TOFU, and mismatch handling.
- CUDA/XPU probe results identify the successful detection method.
- No API or training path uses the transport yet.

---

## PR 4 — Remote-server CRUD and status API

**Purpose:** Expose remote-server administration and verify-on-save independently from training execution.

### Scope

- Add global remote-server CRUD routes.
- **Tier 1 preflight gates create/update** (reachability, auth, host key, `docker version`, driver, nominal disk) with a bounded timeout. **Tier 2 verification (registry pull, signature policy, in-container `torch.{cuda,xpu}.is_available()`) must be a separate async action**, never inline in the save request — it can pull multiple gigabytes.
- Persist the initial TOFU key only after successful first preflight; use the existing pinned key on updates.
- Reject saving an enabled server when required Tier 1 checks fail.
- A transient Tier 1 failure updates `last_check_*` and marks the server unhealthy; it must never delete the record or re-pin a changed host key.
- Add a structured status/check endpoint for reachability, auth, Docker, registry, driver, container probe, compatible image, last-check time, and busy/in-use status.
- Add safe, stable error mapping (including a distinct, actionable error when `REMOTE_SERVER_SECRET_KEY` is unset or a secret is undecryptable) and update OpenAPI output.

### Security gate

- API responses and errors never include secret material, passphrases, host keys, raw SSH exceptions, or remote command text.

### Dependencies

- PR 2
- PR 3

### Acceptance criteria

- API tests cover healthy create/check/edit/delete paths.
- A failed preflight cannot create an enabled usable server.
- A save request never performs an image pull.
- Existing training APIs remain unchanged.

---

## PR 5 — Training targets management UI — **PARTIALLY LANDED, NEEDS CORRECTION**

**Purpose:** Deliver training-target management independently from training-job dispatch.

### Already present

- `remoteTrainers` build-time feature flag in `application/ui/src/config/feature-flags.ts` (default off, `PUBLIC_ENABLE_REMOTE_TRAINERS`, with a `localStorage` dev override).
- A flag-gated route in `../../../ui/src/router.tsx` and `application/ui/src/routes/remote-servers/index.tsx`, a thin wrapper rendering `features/remote-trainers/remote-trainers-page.tsx`.

### Corrections required (decided)

- **Promote the route to global.** `router.tsx` defines `const remoteServers = project.path('/remote-servers')` under `paths.project.*`, yielding `/projects/:project_id/remote-servers`. Neither `RemoteServerDB` nor `RemoteTrainerDB` has a project FK, so move it to a top-level path (e.g. `/training-targets`) with its own primary-nav entry, outside `ProjectLayout`.
- **Unify to one "training target" concept.** Users see a single list of places training can run — local, direct-URL trainers, SSH servers — distinguished by a type badge, not by separate product nouns. Rename the route folder and component to match (currently the same screen is `routes/remote-servers/` rendering `RemoteTrainersPage` from `features/remote-trainers/`), and pick `routes/` or `features/` per `../../../ui/AGENTS.md`. Internal model names stay `RemoteTrainerDB` / `RemoteServerDB`.

### Remaining scope

- Implement list/detail layout with target identity, type badge, configured device type, health, and in-use / waiting-for-GPU status.
- Implement create/edit forms with Tier 1 verification feedback and an explicit test-connection action that runs Tier 2 with progress.
- Render structured preflight status and last-check data, including "protocol unknown" for grandfathered direct-URL trainers and an actionable state for a missing `REMOTE_SERVER_SECRET_KEY` or a key-fingerprint mismatch.
- Add generated OpenAPI type usage and UI tests.

### Security gate

- The UI never receives or renders stored secret fields.
- Client logs and notifications must not include submitted secrets.

### Dependencies

- PR 4

### Acceptance criteria

- Tests cover list/detail, form validation, and healthy/unreachable/misconfigured/busy/checking states.
- Remote jobs still cannot be launched at this stage.

---

## PR 6 — Training job contract and progress framework

**Purpose:** Prepare job orchestration and UI-compatible telemetry without provisioning a remote container.

### Scope

- Add `remote_server_id` to `TrainJobPayload`.
- **Add a `TrainingTarget.SSH` enum member** (`schemas/job.py` currently has only `LOCAL`/`REMOTE`) and extend `validate_training_target` so an SSH job requires `remote_server_id` and forbids `remote_trainer_id`/`remote_trainer_url`. Overloading `REMOTE` is not acceptable: `get_training_backend` raises when `remote_trainer_url is None`, and the worker's `reattaching` check would misfire.
- **Update `TrainingWorker._target_key`** to return `ssh:<remote_server_id>` for SSH jobs. It currently returns `remote:<remote_trainer_id>` for anything non-local, which would map every SSH job on every server to the single key `remote:None`.
- Audit every existing `is TrainingTarget.REMOTE` / `is TrainingTarget.LOCAL` branch in the backend for the new third case.
- Persist provisioning/image-attribution fields introduced in PR 2.
- Validate remote-server existence and successful preflight at job submission.
- Generalize existing remote progress windows into an ordered phase table and helper: connect, image pull, image verification, trainer start, upload, train, and download. Phase keys live in a versioned constant shared with the UI.
- Apply **one phase table to all targets** (decided). Retuning `SNAPSHOT_UPLOAD_PROGRESS = 10` / `TRAINING_PROGRESS_END = 95` shifts local and direct-URL progress by ~1–2%; update existing progress assertions rather than branching per target.
- Add a `waiting_for_gpu` job state so a job pending on a busy server is visibly waiting rather than appearing stuck.
- Attach additive `extra_info["phase"]` descriptors, budgeting for the existing 16 KB `extra_info` cap.
- Refactor `RemoteTrainingBackend` to accept an injected endpoint/device while preserving the existing direct-URL (`remote_trainer_id`/`remote_trainer_url`) behavior.
- Test phase mapping, job validation, serialization, target-key uniqueness, and legacy progress compatibility.

### Exclusions

- No SSH transport, tunnels, or remote Docker operations.

### Dependencies

- PR 0
- PR 2

### Acceptance criteria

- Local and static-remote jobs retain their existing behavior.
- Local jobs remain valid without a phase descriptor.
- A remote-server ID is storable but does not yet change backend selection.

---

## PR 7 — SSH Docker provisioning service

**Purpose:** Implement secure lifecycle management for a per-job trainer container behind a dedicated service boundary.

### Scope

- Resolve the Studio build revision from a **baked-in OCI label / build arg / `../../../VERSION`** (not `git rev-parse HEAD`, which always fails in the containerized backend), prefer its trainer image tag, and fall back to `latest` only when required; persist and log the fallback reason.
- Resolve an immutable digest and verify required image identity/signature/SBOM policy before use.
- Check GPU availability before launch; when busy, leave the job pending with exponential backoff in the `waiting_for_gpu` state rather than failing it, with a per-server probe throttle and a give-up timeout.
- Re-check remote free disk against this job's actual snapshot size.
- Pull and launch by digest only.
- Use a validated deterministic container name, management/job/server labels, a **`backend_instance_id` ownership label**, `--restart=no`, a bounded `--stop-timeout`, a container-side watchdog, and an ephemeral loopback-only host port.
- Apply least privilege: non-root execution, dropped capabilities, no `--privileged`, minimal device passthrough, and bounded writable storage.
- Create and close the SSH local-forward tunnel, with keepalives. A dropped tunnel **reconnects and resumes** against the still-running container within a bounded retry budget; it does not fail the job.
- Verify container readiness and `/health` image/protocol metadata before upload, rejecting an SSH image that reports no protocol version.
- Persist container/tunnel state before accepting work.
- Tear down containers and tunnels in `finally` on unrecoverable failure and cancellation.
- Implement orphan sweeping that requires all management labels **and** a matching `backend_instance_id` **and** no reattach claim, plus sanitized/capped provisioning progress output.

### Security gate

- The container launches by resolved digest, never a mutable tag.
- The trainer is bound only to remote `127.0.0.1`.
- Orphan cleanup only touches containers this deployment provably owns and that no active job has claimed; label match alone is insufficient on a shared server.
- CUDA/XPU in-container device check failure blocks the job before dataset upload.

### Dependencies

- PR 1
- PR 3
- PR 6

### Acceptance criteria

- Tests cover SHA preference, revision resolution without `.git`, fallback persistence, pull failures, digest persistence, loopback binding, protocol mismatch (strict for SSH), cleanup at every failure point, cached images, output handling, tunnel-drop reconnect, GPU-busy pending/backoff/give-up, and non-destructive orphan sweeping (including containers owned by another `backend_instance_id`).
- At least one integration test runs against a containerized `sshd` rather than mocks.

---

## PR 8 — Enable server-aware training execution

**Purpose:** Connect provisioning to the worker and expose selected-server training in the train dialog.

### Scope

- Resolve the selected server in `TrainingWorker`.
- Update `get_training_backend(payload)` precedence:
  1. `training_target is TrainingTarget.SSH` → SSH provisioning;
  2. `training_target is TrainingTarget.REMOTE` → pinned-URL direct-URL registry backend;
  3. otherwise use local training.
- Wrap remote training with provision → HTTP dataset transfer through tunnel → training/download → teardown.
- Extend cancellation to cancel both the remote trainer job and provisioning resources, using the job's own entry in `TrainingWorker.job_interrupt_flags` (never a single shared interrupt signal, so concurrent jobs on other targets are unaffected).
- Use the selected server’s configured device type instead of a live trainer `/devices` probe in the dialog path.
- **Unify the training-target selector** in `../../../ui/src/routes/models/train-model-dialog.tsx` rather than adding a second dropdown. The dialog already queries `/api/remote-trainers`, health-checks the selection, and sets `training_target` + `remote_trainer_id`. Present one **"training target"** control listing local, direct-URL trainers, and SSH servers as entries in a single list with a type badge and status, deriving `training_target` from the choice. Add health indication, submit gating on unhealthy targets, a pre-submit notice when the chosen target's GPU is busy (the job will wait), and progress stepper rendering.

### Security gate

- A `remote_server_id` must never use the direct-URL `remote_trainer_url` path.
- SSH-provisioned jobs force the existing streamed HTTP transfer through the tunnel.
- Cancellation and failures always trigger teardown.

### Dependencies

- PR 4
- PR 5
- PR 6
- PR 7

### Acceptance criteria

- Integration tests prove SSH selection takes precedence over static remote mode.
- The trainer is contacted only through the local tunnel.
- Containers are removed after success, failure, and cancellation.
- Static remote and local paths remain functional.

---

## PR 9 — Recovery, operational hardening, and observability

**Purpose:** Close lifecycle gaps that cannot be fully addressed in isolated provisioning tests.

### Scope

- Implement **startup reattach**: for each `JobProvisioningDB` row on a non-terminal job, verify the pinned host key, confirm the container exists/runs/is owned, re-open the tunnel on a fresh local port, re-verify `/health` and image digest, and resume streaming. Implement every failure branch explicitly (container gone, health never ready, digest mismatch, host-key mismatch, port unreachable).
- Run reattach **before** the orphan sweep, and exclude reattach-claimed containers from sweeping.
- Add the startup/recovery orphan sweep from persisted job state, trusted labels, and `backend_instance_id` ownership.
- Reconcile stale tunnel/container records.
- Implement `waiting_for_gpu` backoff, the per-server probe throttle, and the give-up timeout (the 0.5 s `run_loop` poll must not become a 2 Hz SSH probe).
- Establish bounded retry and timeout policies for SSH, Docker, tunnel readiness, tunnel keepalive/reconnect, and trainer health.
- Add structured operational logs and metrics for preflight, image resolution/fallback, provisioning duration, cleanup, reattach outcomes, orphan recovery, and GPU-busy waits.
- Improve error attribution in job messages and phase state.

### Security gate

- Review logs, metrics, traces, and job messages for credential, host-key, and untrusted-output leakage.
- If trusted SSH identity or container ownership cannot be established, recovery must not perform destructive cleanup.
- A host-key mismatch during reattach fails closed and does **not** tear down the container.

### Dependencies

- PR 8

### Acceptance criteria

- A simulated backend restart reattaches a still-running container, re-opens the tunnel, and resumes streaming without losing training progress.
- Each reattach failure branch produces the specified outcome.
- Unrelated containers and containers owned by another Studio instance remain untouched.
- A job waiting on a busy GPU is visibly pending, backs off, and eventually times out.

---

## PR 10 — Documentation, deployment enablement, and staged rollout

**Purpose:** Make the feature supportable and release it with a controlled blast radius.

### Scope

- Document Fernet key setup, key rotation/loss recovery, host setup for CUDA/XPU, Docker-daemon privilege risks, XPU limitations, image verification, cleanup/recovery procedures, and the **direct-reachability (no bastion) limitation**.
- **Document the trust assumption plainly:** the Studio backend has no auth model, so enabling this feature means anyone who can reach the Studio API can execute code as root on every registered server. Do not enable on a network-exposed instance.
- Add deployment environment documentation without real secrets.
- Extend the existing `remoteTrainers` feature flag (already in `application/ui/src/config/feature-flags.ts`, default off via `PUBLIC_ENABLE_REMOTE_TRAINERS`) with a matching **backend-side** enablement switch, so a disabled deployment rejects SSH job submission and remote-server writes rather than only hiding the UI from a browser that could still call the API directly.
- Add an integration-test matrix and hardware-validation checklist for CUDA, XPU, both SSH auth types, host-key mismatch, cache reuse, cancellation, tunnel drop/reconnect, GPU-busy waiting, and backend restart/reattach.
- Roll out first to an allowlisted set of controlled servers.

### Security gate

- Complete the PR 0 threat model review before default-on enablement. Because there is no backend auth model, remote SSH training **stays flag-off by default indefinitely** until one exists; re-review before default-on or any network exposure.
- Production secrets must be injected through approved secret management and must not be committed or embedded in images.

### Dependencies

- PR 9

### Acceptance criteria

- Operators can configure, test, diagnose, rotate, and decommission servers using the documentation.
- The feature can be disabled without impacting local or direct-URL remote-trainer training.

---

## Recommended merge and release order

1. Merge PRs 0–6 with remote execution disabled.
2. Merge PR 7 behind a backend-only feature flag and validate it against disposable CUDA and XPU hosts.
3. Merge PR 8 with the UI selector feature-flagged.
4. Merge PRs 9–10.
5. Enable the feature for an allowlisted set of remote servers only after real-host cancellation, recovery, and image-verification validation succeeds.

This order ensures the trust, encryption, image identity, and Docker lifecycle boundaries are reviewed and tested before the product can launch any remote training container.
