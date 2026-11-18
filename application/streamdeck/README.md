# Stream Deck recording and inference controls

Use a USB Stream Deck Mini or Neo with the session already open in Studio.
The bridge runs on the computer with the Stream Deck attached: macOS, Windows, or Ubuntu.
Studio can run there or on another computer. No Elgato plugin, browser focus, or backend
USB passthrough is needed.

## Recording layout

```text
┌───────────────┬───────────────┬───────────────┐
│ Start episode │ Accept / save │ Discard       │
│               │               │ press twice   │
├───────────────┼───────────────┼───────────────┤
│ Recording     │ Saved in this │ Connection /  │
│ status        │ session       │ command error │
└───────────────┴───────────────┴───────────────┘
```

On Neo, the fourth column is unused. Larger LCD decks with at least two rows and
three columns also fit this layout; dials, touch strips, and Pedal are not supported.
Exactly one LCD deck must be connected. Quit Elgato Stream Deck, StreamController,
and other applications using it before connecting from Studio.

## Install prerequisites

Install [uv](https://docs.astral.sh/uv/getting-started/installation/). It downloads
Python 3.12+ and the bridge's Python dependencies when needed; no backend Python
environment or separate `pip install` is required. Install the native HIDAPI library too:

### macOS

```sh
brew install hidapi
```

### Windows

Download the HIDAPI release from the [upstream HIDAPI releases](https://github.com/libusb/hidapi/releases).
Extract `hidapi.dll` matching the architecture of the Python interpreter (normally x64).
Place it in a directory on `PATH`. Alternatively, put it beside `streamdeck_bridge.py`:
`application/streamdeck/` for native launch, or the installed launcher data directory
(printed by the installer) for host launch.
Do not replace the Stream Deck's driver with Zadig; the bridge uses the normal HID driver.

### Ubuntu

```sh
sudo apt install libhidapi-libusb0
```

Grant the logged-in desktop user access to Mini/Neo USB devices with a udev rule:

```sh
sudo tee /etc/udev/rules.d/70-physicalai-streamdeck.rules <<'EOF'
SUBSYSTEM=="usb", ATTRS{idVendor}=="0fd9", ATTRS{idProduct}=="0063", TAG+="uaccess"
SUBSYSTEM=="usb", ATTRS{idVendor}=="0fd9", ATTRS{idProduct}=="00b3", TAG+="uaccess"
SUBSYSTEM=="usb", ATTRS{idVendor}=="0fd9", ATTRS{idProduct}=="0090", TAG+="uaccess"
SUBSYSTEM=="usb", ATTRS{idVendor}=="0fd9", ATTRS{idProduct}=="00b8", TAG+="uaccess"
SUBSYSTEM=="usb", ATTRS{idVendor}=="0fd9", ATTRS{idProduct}=="009a", TAG+="uaccess"
EOF
sudo udevadm control --reload-rules
sudo udevadm trigger
```

Unplug and reconnect the deck. Other models need their own product ID in the rule.
`uaccess` assumes a local desktop login; headless hosts need an administrator-managed
group/ACL instead. Do not run the bridge as root or grant world-writable USB access.

## Connect from Studio (native or Docker)

Use a Studio backend and UI containing this integration. After installing the prerequisites,
open the Record or Inference sidebar and click **Start Stream Deck helper**.

### Native Studio on the USB host

For a local, loopback-bound backend running from this checkout, Studio launches the helper
with `uv` directly. No launcher installation or recurring terminal command is needed. The
first launch may download Python/dependencies; wait for **Helper ready**. Studio stops
its own helper process tree when the backend shuts down. `uv` must be on the backend's `PATH`.

Native launch is deliberately unavailable in containers, for remote/network-bound backends,
or when the bridge script is not installed with the backend. Those deployments use the host
launcher below. Native launch always runs on the **backend computer**, not a remote browser's
computer; use host launch for a remote USB host.

### Docker or a separate USB host: one-time launcher installation

Run this once from the checkout **on the USB host, outside Docker**:

```sh
uv run application/streamdeck/host_launcher.py --install
```

The default backend URL is `http://localhost:7860`. To use another published port or a remote
Studio backend, set it at installation:

```sh
uv run application/streamdeck/host_launcher.py --install --url http://STUDIO_HOST:7860
```

Use the backend URL, not a frontend development server. Only use a backend you trust.
The installer copies the helper scripts into per-user application data and registers
`physicalai-streamdeck://start` using a macOS application, Windows user registry entries,
or a Linux desktop entry (`xdg-mime` required). It needs no administrator privileges.
Moving the checkout afterward does not break the launcher; rerun installation to update
its copied scripts or backend URL. Stop the old helper before updating.

Now **Start Stream Deck helper** opens that installed launcher. Approve the browser's
external-application prompt. No terminal command is needed for subsequent launches.
Repeated clicks do not create duplicate host helpers. Windows may show a launcher console;
keep it open while using the helper. There is no tray application or login startup service.

The browser link accepts only the fixed start action: it cannot supply a backend URL,
script path, or command. The helper connects outward to the backend saved at installation;
it exposes no local HTTP listener, needs no Docker USB passthrough, and does not claim USB
until you explicitly connect a workspace.

After stopping the host helper, remove the URL handler if desired:

```sh
uv run application/streamdeck/host_launcher.py --uninstall
```

Uninstall removes the registration and copied scripts/configuration, retaining diagnostic
logs. It does not stop a running host helper. Stop it through the OS process manager (or
its console on Windows) first. Native Studio never terminates an independently started helper.

### Attach a recording workspace

1. Open a Studio workspace, select an environment and recording dataset, and enter a task.
   Enable teleoperation in Studio if desired. **Keep the workspace tab open**: it owns the session.
2. In the Record sidebar, click **Connect Stream Deck**. Studio selects this workspace's
   session and task automatically; no session IDs or task arguments are needed.
3. Wait for the connected device status, then press **Start**, demonstrate, and **Accept / save**.
   To throw away an episode, press **Discard** twice within two seconds. Disabled keys do nothing.
4. Task edits synchronize automatically after a short debounce. Wait for synchronization before
   starting the next episode. An empty task disables the deck's Start key. If synchronization
   fails, disconnect before recording; do not assume the deck has the edited task.
5. Click **Disconnect Stream Deck** to release USB. This does **not** save/discard an episode
   or stop teleoperation. The helper stays ready for another explicit connection.

Closing/reloading the owning workspace releases its attachment. Connect again in the new
workspace; neither the helper nor another workspace silently takes over. Only one helper
per backend and one attached workspace are supported. After stopping/restarting the helper,
allow up to 15 seconds for its old lease to expire, then click Connect again.

To run a managed helper manually instead of either launch button:

```sh
uv run application/streamdeck/streamdeck_bridge.py
```

Add `--url` for a non-default backend; Ctrl+C stops a manually started helper.

The saved count is the session's pending saved episodes, **not** the dataset's total. Studio
copies them into the dataset when it unloads the recording session, using its normal lifecycle.

## Inference page and local microphone dictation

In the Inference sidebar, select an environment and model, then click **Connect Stream Deck**.
If this workspace already has the recording layout attached, click **Use inference page**.
The Record sidebar offers **Use recording page** to switch back. Opening a sidebar alone
never changes the deck's page. The current workspace's task, Hz and sync/async execution
settings synchronize to the inference page; wait for synchronization before pressing Start.
Keep the owning browser workspace open.

```text
┌───────────────┬───────────────┬───────────────┐
│ Start         │ Stop / Hold   │ Mic / Finish  │
│ inference     │               │ dictation     │
├───────────────┼───────────────┼───────────────┤
│ Control       │ Microphone /  │ Inference page│
│ status        │ speech status │ / error       │
└───────────────┴───────────────┴───────────────┘
```

1. Stop inference before dictating. Mic is disabled while **any** control is active,
   including teleoperation. The inference Stop button only stops model control; it does
   not take over a teleoperation session.
2. Press **Mic** once and wait for **LISTENING**, then speak. Press it again to finish,
   or capture ends after 60 seconds. Preparation/transcription key presses are not queued.
3. Wait for local transcription. The text fills the Inference task field as a **draft**.
   Review or edit it in Studio. Dictation never starts inference or changes a running task.
4. Press **Start inference** separately. It uses the synchronized workspace task and
   inference settings. Press **Stop / Hold** to send Studio's normal stop control command.
   This holds the robot; it is **not an emergency stop**.

An initially empty inference draft is allowed so Mic can provide the first task. Start is
blocked during capture/transcription. Disconnecting, switching pages, losing the connection,
or starting control elsewhere cancels capture and drops any late transcript. A model download
or transcription already in progress may finish locally, but its cancelled result is never
published. Only one local speech worker is used, including across page switches.

**Switching pages or disconnecting does not stop active inference.** Stop it explicitly first.
No Start, Stop, or transcript delivery is automatically retried after a network failure.

### Microphone prerequisites and privacy

- Select the attached microphone as the **OS default input** on the USB host. Capture uses
  that device's native sample rate; Whisper resamples the in-memory audio as needed. There
  is no browser microphone permission or audio streaming to Studio.
- The helper installs `sounddevice`, `numpy`, and `faster-whisper` automatically. On Ubuntu,
  also install PortAudio: `sudo apt install libportaudio2`. macOS/Windows Python wheels
  normally include it ([sounddevice installation](https://python-sounddevice.readthedocs.io/en/latest/installation.html)).
  Windows also needs the [Visual C++ runtime](https://opennmt.net/CTranslate2/installation.html)
  for the speech engine; no CUDA/GPU setup is required.
- Grant microphone access to the launcher/Python host when the OS asks. On macOS check
  **Privacy & Security → Microphone**; for native terminal-started Studio this may be the
  terminal application. Windows must allow microphone access for desktop applications.
- First use downloads the English-only Whisper **small.en** model from Hugging Face and can
  take longer. Speech is assumed to be English; language detection is disabled. It runs
  locally on CPU with int8 computation and caches its weights. Compared with the previous
  `tiny` model, expect a larger download and slower transcription in exchange for better
  recognition. Internet is needed for initial dependencies/model downloads, not for
  sending speech to a cloud API.
- Audio stays in RAM on the USB host and is not saved or sent to the backend. Only recognized
  task text is sent to Studio. Check the draft: small local models can mishear robot tasks.
- After updating Studio, stop an old installed helper and rerun `host_launcher.py --install`
  (with the same `--url` if customized) to copy the new speech module and microphone permission
  metadata. Native helpers need a restart too. Old recording-only helpers cannot show this page.

A microphone/permission error or no detected speech leaves the task unchanged. See Studio's
Stream Deck error and the helper log, fix the default input/permission, then press Mic again.

If you see `open() got an unexpected keyword argument 'metadata_errors'`, an old helper
resolved incompatible PyAV 19 with faster-whisper 1.2. The helper now constrains PyAV to
`>=11,<19`. Stop and restart it from the updated checkout; for an installed host launcher,
rerun its installer first. No model re-download is needed.

### Optional manual recording mode

The original terminal-driven mode remains available, independently of the UI connection flow:

```sh
uv run application/streamdeck/streamdeck_bridge.py --list-sessions
uv run application/streamdeck/streamdeck_bridge.py --session SESSION_ID --task "Pick up the block"
```

In manual mode, `--task` is a fixed snapshot independent of the browser field. Restart the
bridge to change it or select another session. Add `--url` to both commands for a non-default
backend. Do not run manual mode and a managed helper against the same physical deck.

## Safety and failure behavior

- This is **not an emergency stop**. Recording keys do not enable/disable teleoperation or
  stop robot motion. Inference Start can move the robot; Stop uses Studio's normal hold
  behavior, not a safety-rated stop. Keep the normal robot safety controls available.
- Status is polled about four times per second, not real-time. Coordinate with anyone
  using browser controls; the server's existing recording validation remains authoritative.
- No automatic session selection, fallback, or switching. In managed mode, closing the
  workspace clears its attachment; losing the helper for 15 seconds clears it too. In
  manual mode, a closed session becomes offline, even if a different workspace opens.
- Offline and in-flight key presses are dropped, not replayed. A command waits up to
  ten seconds for its HTTP response. On timeout, the command **may still finish** in
  Studio; it is never automatically retried. Check Studio before trying again.
- `ERROR / SEE LOG` means a command failed or its outcome is unknown. Details go to
  the terminal for manual mode, `streamdeck-helper.log` in Studio's log directory for native
  launch, or `helper.log`/`launcher.log` in the installed launcher's data directory for host
  launch. Refreshed session status remains authoritative.
- Managed mode reports USB failures in Studio and retries while the explicitly requested
  attachment remains active. Disconnect in Studio to stop retrying. Manual mode exits on
  USB disconnect and must be restarted.
- A stale managed attachment's commands are rejected by the backend, including after
  disconnect/reconnect to the same workspace. Commands already accepted may still finish.
- Use only a trusted local network or a protected tunnel. These endpoints use Studio's
  existing network security model; this bridge adds no authentication. Do not expose
  the backend directly to the internet.

If USB probing fails, check HIDAPI, its architecture, and OS permissions. If no device
is found or opening fails, fully quit competing deck applications. If status stays
offline, verify the backend URL, that it contains this integration, and that the
selected browser workspace is still open.

## Development checks

Hardware-free bridge tests, from the repository root:

```sh
uv run --no-project --with "streamdeck>=0.10,<0.11" --with "pillow>=11,<13" \
  --with "httpx>=0.28,<0.29" --with "numpy>=2,<3" \
  --with "faster-whisper>=1.2,<2" --with "av>=11,<19" \
  python -m unittest discover -s application/streamdeck -v
```

Backend attachment and websocket regression tests, from `application/backend/`:

```sh
uv run pytest tests/api/test_runtime_ws.py tests/api/test_streamdeck_inference.py tests/services/test_streamdeck_service.py tests/services/test_streamdeck_launch.py tests/runtime_session
```

UI component tests, from `application/ui/`:

```sh
npm run test:unit -- src/features/runtime-session/workspace/streamdeck-controls.test.tsx
```

The bridge reads `GET /api/runtime/sessions/{id}/state` and uses recording-only
`POST /api/runtime/sessions/{id}/recording`. Managed mode coordinates through
`GET /api/streamdeck`, `POST /api/streamdeck/heartbeat`, and
`PUT`/`DELETE /api/streamdeck/sessions/{id}`. Manual session discovery uses
`GET /api/runtime/sessions`. It never opens a project runtime websocket: that would
create a separate session rather than attach. Native launch uses `POST /api/streamdeck/start`
with the required `X-StreamDeck-Launch: 1` header and loopback client/Host/Origin checks;
it never accepts executable paths, backend URLs, or extra arguments from a request.
Inference uses recording-independent `POST /api/streamdeck/sessions/{id}/inference` (only
`start`/`stop`) and `/dictation` (text draft only). Both require the current
`X-StreamDeck-Connection` token and inference page. Switching pages replaces that token,
fencing keys and late transcripts from the previous page.
