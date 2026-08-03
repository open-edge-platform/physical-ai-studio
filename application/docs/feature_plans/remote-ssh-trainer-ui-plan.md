# Remote SSH Trainer — UI Plan

The UI half of [`remote-ssh-trainer-plan.md`](remote-ssh-trainer-plan.md), split
out because every item here depends on backend endpoints that do not exist yet.
PR sequencing across both documents lives in
[`remote-ssh-trainer-pr-plan.md`](remote-ssh-trainer-pr-plan.md).

Three deliverables:

1. A global **training targets** management screen.
2. A **unified target selector** in the train dialog.
3. A **phase stepper** in the training progress view.

Interactive mockup: `../../ui/mockups/remote-server-ui.html`.

## Naming and concepts

**Users see one concept: a "training target"** — somewhere training can run. The
local machine, a direct-URL trainer, and an SSH-provisioned server are three
_types_ of target, distinguished by a badge, not by separate product nouns.
Internal model names (`RemoteTrainerDB`, `RemoteServerDB`) stay as they are.

**Credentials are never entered in Studio.** An SSH target is configured by
picking a `Host` alias the user already has in `~/.ssh/config`. There is no key,
password, or passphrase field anywhere in the UI, and no "encrypted at rest"
messaging — Studio never receives secret material. See the deployment-model
section of the backend plan for why.

## 1. Training targets management screen

One **global** screen, mirroring the existing list/detail pattern used by robots
and cameras (`routes/robots/layout.tsx` + `robot.tsx`).

### Already present

- A `remoteTrainers` build-time feature flag in
  `../../ui/src/config/feature-flags.ts` (default off,
  `PUBLIC_ENABLE_REMOTE_TRAINERS`, with a `localStorage` dev override).
- A flag-gated route in `../../ui/src/router.tsx` and
  `../../ui/src/routes/remote-servers/index.tsx`, a thin wrapper that
  renders `features/remote-trainers/remote-trainers-page.tsx`.

### Corrections required

- **Promote the route to global.** `router.tsx` currently has
  `const remoteServers = project.path('/remote-servers')` under `paths.project.*`,
  yielding `/projects/:project_id/remote-servers`. Neither `RemoteServerDB` nor
  `RemoteTrainerDB` has a project FK, so move it to a top-level `/training-targets`
  with its own primary-navigation entry, outside `ProjectLayout`.
- **Resolve the naming split.** The same screen is currently
  `routes/remote-servers/` rendering `RemoteTrainersPage` from
  `features/remote-trainers/` — two nouns, neither of them "training target".
  Per `../../ui/AGENTS.md`, `routes/` holds thin shells and `features/` owns
  the logic, so: `routes/training-targets/` (shell) +
  `features/training-targets/` (implementation).

### Remaining work

- **List pane** — one list of all targets: name, type badge (Local / Direct URL /
  SSH), host, device type, status badge. A "New" action.
- **Create/edit form for SSH targets** — name, **SSH host alias** (a select
  populated from the backend's SSH-config reader, not a free-text field), and
  device type (CUDA/XPU). Show the alias's resolved hostname/port/user read-only
  beneath the select so the user can confirm they picked the right host. No
  key/password/passphrase fields exist.
- **Create/edit form for direct-URL targets** — unchanged from today.
- **Status view** — driven by the status endpoint, and **grouped by tier**, since
  the two have very different costs and the user needs to know which results are
  current:
  - _Tier 1 (cheap, refreshed on save and on status poll)_ — alias resolvable,
    reachable, authenticated, host key verified, Docker usable, registry
    reachable, disk free, driver present + version and which detection method
    succeeded, GPU occupancy, and an "in use by job" / "waiting for GPU"
    indicator.
  - _Tier 2 (expensive, only as fresh as the last explicit run)_ — image resolved
    and pulled with its digest, in-container device probe, and compatible
    protocol version (or "protocol unknown" for a grandfathered direct-URL
    trainer). Show when it last ran; do not present a stale Tier 2 result as a
    current one.
- A **"Test connection"** button runs Tier 2 with progress. It must be explicit —
  never triggered by opening the screen or by list polling — because it pulls a
  multi-gigabyte image.
- **Busy is not an error.** GPU occupancy is reported by Tier 1 but never blocks
  saving, and a busy target stays selectable in the train dialog. Style it as a
  neutral/notice state, not a failure.
- **Status badges** — Healthy / Unreachable / Misconfigured / Busy / Checking.
- **Distinct actionable states**, each with the recovery action inline, not a
  generic error:
  - _SSH host alias not found_ — the saved alias is gone from `~/.ssh/config`.
    Offer re-selection.
  - _Host fingerprint not accepted_ — tell the user to run `ssh <alias>` once.
  - _Key requires a passphrase and no SSH agent is available_ — tell the user to
    `ssh-add` the key.
- **Data layer** — generated `$api` hooks (`$api.useQuery` / `$api.useMutation`)
  against the new endpoints, following existing route patterns.
- **Empty/error states** — reuse the shared `EmptySelection` / illustrated
  message pattern from `router.tsx` for "no target selected" and connection
  errors.

### Tests

List/detail render, create/edit form validation, the alias select populating and
handling an empty SSH config, each status badge state, each of the three
actionable credential states, and the "Test connection" flow — following existing
route test patterns. Also assert that **opening the detail screen does not fire
the Tier 2 request**, since that regression is invisible in review and expensive
in practice.

## 2. Unified target selector in the train dialog

`../../ui/src/routes/models/train-model-dialog.tsx` **already has a
remote-trainer picker**: it queries `/api/remote-trainers`, health-checks the
choice, and sets `training_target` + `remote_trainer_id`.

**Do not add a second "remote server" dropdown.** Two similarly-named pickers
with no indication of which wins is a worse outcome than either alone. Instead:

- One **"training target"** control listing local, registered direct-URL
  trainers, and registered SSH servers as entries in a single list, each with a
  type badge and status.
- Derive `training_target` (`LOCAL` / `REMOTE` / `SSH`) and the corresponding id
  field from the selected entry.
- Show status inline; disable submit when the selection is unhealthy.
- A target whose GPU is busy **stays selectable** — the job waits rather than
  failing — but say so in the dialog before submit, so the wait is expected
  rather than looking like a hang.
- Use the SSH target's **configured** `device_type` from its record. The trainer
  is not running at dialog time, so there is no live `/devices` probe on this
  path.

Regenerate OpenAPI types after the backend lands:
`npm run build:api:download && npm run build:api`.

## 3. Progress phase stepper

The backend attaches a `phase` descriptor to `extra_info`:
`{ key, label, index, total, state, indeterminate }`, where `state` is one of
`active | done | skipped | waiting | failed`. Phase keys come from a **versioned
constant shared with the backend** so the two cannot drift.

- Render a stepper from `extra_info.phase` above the existing overall bar and
  message line: connect → image pull → verify → start → upload → train →
  download.
- `indeterminate: true` → spinner in that step instead of a misleading exact %.
- `state: "waiting"` → the GPU-busy case. The job is still `pending`; show the
  wait and its give-up deadline. **This is a phase state, not a job status** —
  `JobStatus` remains `pending | running | completed | failed | canceled`, so no
  generated consumer changes.
- `state: "failed"` → mark the active step failed and surface the phase message,
  which is how a failure gets attributed ("Failed during image verification").
- During `train`, keep the existing step-loss telemetry rendering.
- **Degrade gracefully when `phase` is absent** — local and direct-URL jobs show
  only the bar + message, exactly as today. They keep their existing progress
  windows; the backend uses a per-target phase table specifically so those curves
  do not shift.

### Tests

Stepper renders each state; a job with no `phase` renders the current view
unchanged; a `waiting` phase shows the wait without implying failure.

## Security notes for the UI

- No form field anywhere accepts SSH key material, a password, or a passphrase.
- No API response contains secret material, so there is nothing to accidentally
  render — but do not echo submitted form values into client logs or
  notifications regardless.
- The `remoteTrainers` feature flag only hides the screen. It is **not** a
  security control.
