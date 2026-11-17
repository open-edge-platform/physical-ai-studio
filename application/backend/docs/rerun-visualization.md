# Rerun Visualization

## Runtime Recordings

Pressing **Play** in runtime inference mode starts a fresh Rerun recording and
starts (or reuses) a background Rerun server on
`127.0.0.1:9876`, without opening a window. Connect a native Rerun viewer manually
with `rerun rerun+http://127.0.0.1:9876/proxy` while the session is running.
Opening the runtime page alone does not start Rerun. Pressing **Stop**, leaving
the page, or unloading or failing the model/control stops the recording.
Studio stops its background server when the last recording using it stops;
servers started outside Studio are left running. Studio hosts the server in-process
through Rerun's SDK, without launching a separate executable. The server buffers
up to 256 MiB.

Each model run has a separate recording of measured joints,
target actions, latest camera frames, model predictions, queue depth, inference
latency, and host/Studio CPU and memory usage. Visualization failures do not stop
the session. Actions are targets rather than the robot's interpolated commands;
camera frames are not guaranteed to be the inputs to a particular inference.
Visualization uses Runtime's `RerunCallback`, including its joint colors,
prediction markers, action/queue timeline windows, and CPU/GPU monitoring sidebar.
Studio dispatches control events on a bounded observer thread and caps latest
camera frames at 10 Hz. Each run has a thread-local recording so simultaneous
sessions cannot send visualization data into one another's streams.

## Settings

Under **Settings → General → Rerun visualization**, enable or disable logging,
opt into opening the native viewer, change the local port, or select connect mode
with a `host:port` or `rerun+http(s)://host:port/proxy` address. Settings are persisted
in the user settings JSON and apply to new model runs, not ones already running.
The viewer and connection address are relative to the backend host, not the browser.

## Docker Deployment

Docker Compose binds the headless Rerun server to the container's interfaces
(`RERUN_BIND_HOST=0.0.0.0`) and publishes port 9876 **only on Docker host loopback**.
Connect a viewer on the Docker host to `rerun+http://127.0.0.1:9876/proxy`.
Keep the native-viewer option off in Docker. If you change the local Rerun port in
Studio, set `RERUN_PORT` to the same value in Docker's `.env` and recreate the
container. Native deployments default to `RERUN_BIND_HOST=127.0.0.1`.
Rerun has no authentication; do not publish this port on a public interface.
