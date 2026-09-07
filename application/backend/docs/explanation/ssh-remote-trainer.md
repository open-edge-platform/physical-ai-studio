# SSH remote-trainer feature

The SSH remote-trainer feature lets the Studio backend provision a training
job on a GPU server it can reach over SSH: it copies/starts a job-scoped
trainer container on that server, tunnels to it over the SSH connection, and
runs the job exactly like a locally-registered remote trainer (see
[`../remote-trainer.md`](../remote-trainer.md) for what runs on the server side).

## Master switch

The feature is **off by default** and must be explicitly enabled:

```dotenv
SSH_REMOTE_TRAINER_ENABLED=true
```

> [!WARNING]
> This feature has no authentication model of its own. Anyone who can reach
> the backend's API can ask it to run arbitrary training jobs — and therefore
> arbitrary code as root inside a container — on every server registered in
> Studio. A compromised backend process can also reach every identity loaded
> in the operating user's SSH agent, not just the registered servers. Only
> enable it on a single-user localhost workstation that nobody else can reach.

## Fail-closed loopback enforcement

Turning the setting on is not sufficient by itself. At startup (and whenever
`cli.serve.start_server` reconciles the actual `--host`/`--port` bind
arguments), the backend re-evaluates whether it is safe to serve the feature:

- If `SSH_REMOTE_TRAINER_ENABLED` is `false` (the default), the feature is
  simply off.
- If it is `true` **and** the backend is bound to a loopback address
  (`127.0.0.1`, `::1`, or `localhost` — the packaged default), the feature is
  active.
- If it is `true` **and** the backend is bound to anything else (including
  `HOST=0.0.0.0`, as set by the packaged Docker Compose file), the feature
  fails closed: it behaves
  as if it were disabled, and the backend logs a `critical` startup message
  explaining why.

This check lives in `core.security.ssh_network_exposure` and is applied in
three independent places so a change in configuration is never enough on its
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
  run after a restart changed the configuration or the bind address.

## Checking the feature's status

`GET /api/remote-servers/feature-status` is unauthenticated by design (an
authenticated status check would be circular — the UI needs the status *to
explain* why the gated endpoints are unavailable) and returns:

```json
{
  "enabled": false,
  "network_exposed": false,
  "reason": null
}
```

`reason`, when present, never names a host alias, container, or other
registered-server detail: it only ever explains the network-exposure
condition, since this endpoint (and the equivalent startup log line) can be
reached without authentication.

## Staged rollout guidance

- Keep `SSH_REMOTE_TRAINER_ENABLED=false` (the default) for any deployment
  that is not a single operator's own workstation.
- When enabling it, also confirm `HOST` is a loopback address
  (`127.0.0.1`/`::1`, the packaged default) — running under the Docker Compose
  deployment, which sets `HOST=0.0.0.0`, will cause the
  feature to fail closed even with the switch on.
- Treat a `critical`-level `"SSH remote-trainer feature disabled at
  startup"` log line as an operator action item: either narrow `HOST` to a
  loopback address, or turn the feature off.
- Review [`../remote-trainer.md`](../remote-trainer.md) for the server-side
  container image, network, and authentication expectations before
  registering any remote server.
