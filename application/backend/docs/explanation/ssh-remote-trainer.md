# SSH training targets

Studio can register a GPU server as an SSH training target instead of requiring
an independently deployed trainer URL. For each registered SSH target, Studio
starts one persistent trainer container on the server and forwards its fixed
loopback port through a standing SSH tunnel. The container and its data volume
are shared across jobs on that target; they are not tied to a single job or SSH
session. See [Physical AI Trainer](../remote-trainer.md) for the server-side
service and image requirements.

## Availability and security

SSH targets are available when the Studio backend binds only to localhost
(`127.0.0.1`, `::1`, or `localhost`). There is no master enable switch. When
Studio binds to a non-loopback address, including the Docker Compose default
`0.0.0.0`, the SSH feature is unavailable; direct-URL trainers remain usable.
Check `GET /api/remote-servers/feature-status` for `network_exposed` and a
safe-to-display `reason`. The SSH host-alias endpoints and SSH trainer
registration are gated; Studio does not open managed tunnels when the feature
is unavailable. Avoid changing the backend bind address while SSH jobs are
active: they need their tunnel to finish reattachment and artifact download.

> [!WARNING]
> Studio has no authentication for SSH target administration. Anyone who can
> reach its API can request work on registered GPU servers. Studio's SSH
> account must have Docker access, which is privileged on the remote host.
> Run Studio on a trusted, single-user localhost workstation. The managed
> trainer container itself runs as a non-root user with dropped capabilities;
> this does not remove the risk of granting the SSH account Docker access.

## Register a target

Use **Training targets** in the Studio UI to add an SSH target. Pick a `Host`
entry in `~/.ssh/config`, or enter a new SSH host there. The API also supports
manual SSH connection fields (`ssh_connection`) without an alias. Studio stores
the alias or connection details (including an optional private-key *path*),
not private-key contents. Set a stable, unused local loopback port and a port
on the SSH server's loopback interface (both default to 8001); Studio derives
the trainer URL from the local port. Each SSH target needs its own local port,
and targets on the same server need distinct remote ports.

A host not yet in `known_hosts` requires **explicit fingerprint confirmation**
in the UI. The first request returns the presented fingerprint; confirm it
before retrying. Studio then records the accepted key and rejects changed or
revoked keys. It never silently accepts an unknown key.

Saving the target opens the tunnel before reporting success and starts the
managed container in the background. A first image pull can take time; check
the target's health in the UI before starting a job. Image signatures are
verified by Studio before launch. Restarting Studio restores standing tunnels
for configured targets; an SSH host that is temporarily offline at startup is
retried in the background. A container that stopped independently may still
need to be started again by saving the target.

Settings > General > Managed SSH Training exposes four timeouts
(`connect_timeout_s`, `command_timeout_s`, `preflight_timeout_s`, and
`image_pull_timeout_s`) and the trainer's shared-memory size
(`trainer_shm_size_gb`, default 32 GiB). They can also be changed with
`PATCH /api/settings` under `ssh`, for example
`{"ssh": {"trainer_shm_size_gb": 48}}`. These settings are stored in Studio's
settings file; environment variables do not override them. Changing shared
memory affects only newly created containers; stop an existing trainer and
save its target to recreate it with the same data volume. SSH config and
`known_hosts` paths, the image registry, and signature policy are
environment-only settings.

## Connection loss and cleanup

The remote trainer keeps running when Studio loses its VPN or SSH connection.
Studio retries the tunnel on its **configured local port**, reconnects the
job's event stream with `GET /jobs/{id}` as a fallback, and downloads a
completed model when the connection returns. The defaults
`SSH_TUNNEL_RECONNECT_BUDGET_S=300` and
`trainer.stream_reconnect_max_s=900` are **warning thresholds**, not failure
or retry limits. While disconnected, a submitted job remains running and
shows that Studio is waiting to reconnect. Studio reattaches after a restart
using the persisted remote job ID; it does not resubmit the training job.
Jobs already marked failed before this behavior was introduced are not
reattached automatically.

If the server will not become reachable, choose **Stop** for the running job.
The job remains running locally until the training worker acknowledges the
stop; only then can you **Delete** the canceled job from Studio. Canceling on
the remote trainer is best-effort while offline: Stop and Delete remove local
tracking, **not a guaranteed remote process or its artifacts**. If the server
later returns, clean up any remaining remote data separately. Completed
artifacts are deleted from the trainer only after Studio successfully
retrieves and extracts them.

Deleting an SSH training target removes its managed container and volume.
Studio rejects target deletion or connection changes while it has queued or
running jobs. A server that cannot be reached cannot have its container and
volume cleaned up remotely; restore access before removing the target.
