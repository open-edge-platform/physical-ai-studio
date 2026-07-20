# Remote Server SSH Trainer Provisioning — PR Plan

This document divides [`remote-ssh-trainer-plan.md`](./remote-ssh-trainer-plan.md) into dependency-ordered, independently reviewable pull requests.

## Goals

- Establish trust, persistence, and image supply-chain boundaries before a job can launch a remote container.
- Keep existing local and static-`TRAINER_URL` training functional throughout implementation.
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

- Add ADRs or an implementation design record for:
  - backend-selection precedence: `remote_server_id` selects SSH provisioning; otherwise `training_mode="remote"` selects static `TRAINER_URL`; otherwise training is local;
  - trainer API protocol version and `/health` contract;
  - OCI image naming, labels, Git-SHA-first resolution, and `latest` fallback;
  - Fernet key lifecycle and remediation after key loss or rotation;
  - TOFU behavior and host-key mismatch handling;
  - managed-container labels and orphan-sweep ownership criteria.
- Define shared constants/types for device types, SSH auth types, ordered progress phases, and image/provisioning result schemas.

### Exclusions

- No database migration, SSH connection, container launch, or UI work.

### Security gate

- A pinned host-key mismatch never falls back to a new key.
- A mutable image tag is never launchable; only a resolved digest is launchable.

### Acceptance criteria

- The trainer `/health` metadata and backend compatibility check are specified.
- The image resolution and host-key trust behavior have unambiguous failure paths.

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

## PR 2 — Remote-server persistence, encryption, and job state

**Purpose:** Establish the durable domain model without making network connections.

### Scope

- Add `RemoteServerDB` and an Alembic migration for configuration, device type, timestamps, last-check summary, plaintext internal `host_key`, encrypted `ssh_secret`, and encrypted optional `ssh_key_passphrase`.
- Add per-job provisioning fields: `remote_server_id`, image reference, fallback reason, resolved digest, container ID/name, remote port, local tunnel port, and build/protocol attribution.
- Add repository and service interfaces.
- Add `REMOTE_SERVER_SECRET_KEY` settings validation plus Fernet encryption/decryption wrappers.
- Add Pydantic request/response schemas which cannot serialize `ssh_secret`, `ssh_key_passphrase`, or `host_key`.
- Add migration, encryption, and serialization tests.

### Security gate

- Validate required Fernet configuration when remote-server functionality is enabled.
- Encrypt secrets before persistence and decrypt only inside the provisioning boundary.
- Test every API response and error path to prove confidential/internal fields are absent.

### Dependencies

- PR 0

### Acceptance criteria

- Tests prove encrypted-at-rest storage.
- Existing jobs and local/static remote behavior are unaffected.

---

## PR 3 — SSH transport, validation, and preflight primitives

**Purpose:** Build the security-critical remote execution boundary before exposing it through an API or job flow.

### Scope

- Introduce an async SSH transport abstraction.
- Strictly validate server configuration: host, port, username, credential sizes, device type, and auth type.
- Implement TOFU host-key behavior and fail-closed verification on subsequent connections.
- Run every remote command as an argument array; do not interpolate a shell command string.
- Add bounded timeouts, cancellation, per-server preflight throttling, output sanitization, output caps, and heartbeats.
- Implement preflight probes for reachability/authentication, Docker access, disk capacity, registry/image access, CUDA/XPU host detection, GPU availability, in-container device checks, and trainer protocol compatibility.

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
- Preflight candidate configuration on create/update.
- Persist the initial TOFU key only after successful first preflight; use the existing pinned key on updates.
- Reject saving an enabled server when required preflight checks fail.
- Add a structured status/check endpoint for reachability, auth, Docker, registry, driver, container probe, compatible image, last-check time, and busy/in-use status.
- Add safe, stable error mapping and update OpenAPI output.

### Security gate

- API responses and errors never include secret material, passphrases, host keys, raw SSH exceptions, or remote command text.

### Dependencies

- PR 2
- PR 3

### Acceptance criteria

- API tests cover healthy create/check/edit/delete paths.
- A failed preflight cannot create an enabled usable server.
- Existing training APIs remain unchanged.

---

## PR 5 — Remote-server management UI

**Purpose:** Deliver remote-server management independently from training-job dispatch.

### Scope

- Add a global `/remote-servers` route and navigation item.
- Implement list/detail layout with server identity, configured device type, health, and busy/in-use status.
- Implement create/edit forms with verification feedback and an explicit test-connection action.
- Render structured preflight status and last-check data.
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
- Persist provisioning/image-attribution fields introduced in PR 2.
- Validate remote-server existence and successful preflight at job submission.
- Generalize existing remote progress windows into an ordered phase table and helper: connect, image pull, image verification, trainer start, upload, train, and download.
- Attach additive `extra_info["phase"]` descriptors.
- Refactor `RemoteTrainingBackend` to accept an injected endpoint/device while preserving static `TRAINER_URL` behavior.
- Test phase mapping, job validation, serialization, and legacy progress compatibility.

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

- Resolve the local Git SHA, prefer its trainer image tag, and fall back to `latest` only when required; persist the fallback reason.
- Resolve an immutable digest and verify required image identity/signature/SBOM policy before use.
- Check GPU availability before launch.
- Pull and launch by digest only.
- Use a validated deterministic container name, management/job/server labels, `--restart=no`, and an ephemeral loopback-only host port.
- Apply least privilege: non-root execution, dropped capabilities, no `--privileged`, minimal device passthrough, and bounded writable storage.
- Create and close the SSH local-forward tunnel.
- Verify container readiness and `/health` image/protocol metadata before upload.
- Persist container/tunnel state before accepting work.
- Tear down containers and tunnels in `finally`.
- Implement trusted-label orphan sweeping and sanitized/capped provisioning progress output.

### Security gate

- The container launches by resolved digest, never a mutable tag.
- The trainer is bound only to remote `127.0.0.1`.
- Orphan cleanup only touches containers having all expected trusted labels.
- CUDA/XPU in-container device check failure blocks the job before dataset upload.

### Dependencies

- PR 1
- PR 3
- PR 6

### Acceptance criteria

- Tests cover SHA preference, fallback persistence, pull failures, digest persistence, loopback binding, protocol mismatch, cleanup at every failure point, cached images, output handling, and non-destructive orphan sweeping.

---

## PR 8 — Enable server-aware training execution

**Purpose:** Connect provisioning to the worker and expose selected-server training in the train dialog.

### Scope

- Resolve the selected server in `TrainingWorker`.
- Update `get_training_backend(remote_server=...)` precedence:
  1. selected remote server uses SSH provisioning;
  2. otherwise static remote mode uses `TRAINER_URL`;
  3. otherwise use local training.
- Wrap remote training with provision → HTTP dataset transfer through tunnel → training/download → teardown.
- Extend cancellation to cancel both the remote trainer job and provisioning resources.
- Use the selected server’s configured device type instead of a live trainer `/devices` probe in the dialog path.
- Add train-dialog server selection, health indication, submit gating, and progress stepper rendering.

### Security gate

- A `remote_server_id` must never use static `TRAINER_URL`.
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

- Add startup/recovery orphan sweep from persisted job state and trusted labels.
- Reconcile stale tunnel/container records.
- Refine per-server busy/in-use state.
- Establish bounded retry and timeout policies for SSH, Docker, tunnel readiness, and trainer health.
- Add structured operational logs and metrics for preflight, image resolution/fallback, provisioning duration, cleanup, orphan recovery, and GPU-busy blocks.
- Improve error attribution in job messages and phase state.

### Security gate

- Review logs, metrics, traces, and job messages for credential, host-key, and untrusted-output leakage.
- If trusted SSH identity cannot be established, recovery must not perform destructive cleanup.

### Dependencies

- PR 8

### Acceptance criteria

- A simulated backend crash reclaims only its own persisted, labeled container.
- Unrelated containers remain untouched.
- Busy state is visible and blocks conflicting launch.

---

## PR 10 — Documentation, deployment enablement, and staged rollout

**Purpose:** Make the feature supportable and release it with a controlled blast radius.

### Scope

- Document Fernet key setup, key rotation/loss recovery, host setup for CUDA/XPU, Docker-daemon privilege risks, XPU limitations, image verification, and cleanup/recovery procedures.
- Add deployment environment documentation without real secrets.
- Add explicit feature enablement/flagging for remote SSH training.
- Add an integration-test matrix and hardware-validation checklist for CUDA, XPU, both SSH auth types, host-key mismatch, cache reuse, cancellation, and backend restart.
- Roll out first to an allowlisted set of controlled servers.

### Security gate

- Complete a security review/threat model before default-on enablement.
- Production secrets must be injected through approved secret management and must not be committed or embedded in images.

### Dependencies

- PR 9

### Acceptance criteria

- Operators can configure, test, diagnose, rotate, and decommission servers using the documentation.
- The feature can be disabled without impacting local or static `TRAINER_URL` training.

---

## Recommended merge and release order

1. Merge PRs 0–6 with remote execution disabled.
2. Merge PR 7 behind a backend-only feature flag and validate it against disposable CUDA and XPU hosts.
3. Merge PR 8 with the UI selector feature-flagged.
4. Merge PRs 9–10.
5. Enable the feature for an allowlisted set of remote servers only after real-host cancellation, recovery, and image-verification validation succeeds.

This order ensures the trust, encryption, image identity, and Docker lifecycle boundaries are reviewed and tested before the product can launch any remote training container.
