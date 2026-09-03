# SSH remote-trainer feature

The SSH remote-trainer feature lets the Studio backend provision a training
job on a GPU server it can reach over SSH: it copies/starts a job-scoped
trainer container on that server, tunnels to it over the SSH connection, and
runs the job exactly like a locally-registered remote trainer (see
[`../remote-trainer.md`](../remote-trainer.md) for what runs on the server side).

## Always active

There is no settings-page or environment switch to turn the SSH
remote-trainer feature off - it is always active, subject only to the
fail-closed network-exposure check below. The timeouts and limits
(`ssh.connect_timeout_s`, `ssh.command_timeout_s`, `ssh.preflight_timeout_s`,
`ssh.image_pull_timeout_s`, `ssh.readiness_timeout_s`, `ssh.gpu_wait_giveup_s`,
`ssh.min_free_disk_bytes`) remain configurable from the Studio settings page
(Settings > General > SSH-provisioned training):

```http
PATCH /api/settings
{"ssh": {"connect_timeout_s": 15}}
```

The value is persisted to the settings JSON file (see
`settings.get_settings_file_path`), and a value in the process environment or
a `.env` file for any of them is silently ignored (see
`settings._EnvExclusionSource`).

The SSH config path, `known_hosts` path, trainer image registry, and cosign
signature policy are not part of this group and remain environment-only:
they configure *how* Studio trusts a host or an image, which the
(unauthenticated) settings API must never be able to move.

> [!NOTE]
> Signature verification runs on this backend's own host, not on a registered
> remote trainer server: the image reference it checks is a fully qualified
> registry digest, so verification needs only this backend's own network
> egress to the registry and to Sigstore. It uses the `sigstore` PyPI
> package, so verification needs no other tooling installed anywhere. See
> `services.ssh.sigstore_verify`, `services.ssh.oci_registry`, and
> `services.ssh.docker_ops.verify_image_signature`.

> [!WARNING]
> This feature has no authentication model of its own. Anyone who can reach
> the backend's API can ask it to run arbitrary training jobs — and therefore
> arbitrary code as root inside a container — on every server registered in
> Studio. A compromised backend process can also reach every identity loaded
> in the operating user's SSH agent, not just the registered servers. Only
> run Studio on a single-user localhost workstation that nobody else can
> reach. This risk is repeated as an explicit warning in the UI at the point
> a user registers an SSH target.

## Fail-closed loopback enforcement

Being always active is not sufficient by itself to make the feature safe on
every deployment. At startup (and whenever `cli.serve.start_server`
reconciles the actual `--host`/`--port` bind arguments), the backend
re-evaluates whether it is safe to serve the feature:

- If the backend is bound to a loopback address (`127.0.0.1`, `::1`, or
  `localhost` — the packaged default), the feature is active.
- If it is bound to anything else (including `HOST=0.0.0.0`, as set by the
  packaged Docker Compose file), the feature fails closed: it behaves as if
  it were disabled, and the backend logs a `critical` startup message
  explaining why.

This check lives in `core.security.ssh_network_exposure` and is applied in
four independent places so a change in configuration is never enough on its
own to expose the feature, and a job that made it past submission is never
enough on its own to let it run:

- `api.dependencies.require_ssh_feature_active` — gates
  `GET /api/remote-servers` (and any future SSH-provisioning route) with a
  `503` when the feature is inactive.
- `services.job_service.JobService.submit_train_job` — refuses to accept a
  new SSH-target job when the feature is inactive at submission time.
- `services.training_backends.get_training_backend` — refuses to dispatch an
  already-queued SSH-target job when the feature is inactive at pickup time,
  so a job that was accepted while the feature was active does not silently
  run after a restart changed the bind address.
- `services.ssh.recovery.recover_ssh_jobs` — skips startup reattachment
  entirely when the feature is inactive, so a `JobProvisioningDB` row left
  over from an earlier, active run never causes a studio restart to dial SSH
  into a registered server or sweep/stop its containers while the feature is
  failing closed. The generic orphan-job abort that runs afterward
  (`TrainingService.abort_orphan_jobs`) still reconciles those jobs from the
  database alone — no SSH connection made — and the normal pickup gate above
  fails any that get requeued.

## Checking the feature's status

`GET /api/remote-servers/feature-status` is unauthenticated by design (an
authenticated status check would be circular — the UI needs the status *to
explain* why the gated endpoints are unavailable) and returns:

```json
{
  "network_exposed": false,
  "reason": null
}
```

`reason`, when present, never names a host alias, container, or other
registered-server detail: it only ever explains the network-exposure
condition, since this endpoint (and the equivalent startup log line) can be
reached without authentication.

## Staged rollout guidance

- Confirm `HOST` is a loopback address (`127.0.0.1`/`::1`, the packaged
  default) — running under the Docker Compose deployment, which sets
  `HOST=0.0.0.0`, will cause the feature to fail closed, since there is no
  settings-page switch to turn it off instead.
- Treat a `critical`-level `"SSH remote-trainer feature disabled at
  startup"` log line as an operator action item: narrow `HOST` to a loopback
  address.
- Review [`../remote-trainer.md`](../remote-trainer.md) for the server-side
  container image, network, and authentication expectations before
  registering any remote server. The Studio UI shows the same risk warning
  when a user registers an SSH target.
