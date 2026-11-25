# Simulation application E2E testing

`application/backend/tests/integration/test_simulation_e2e.py` exercises the supported application path:

**Import recording → train with UI defaults → export that exact model → download its deployment bundle → run the unchanged Runtime recipe in MuJoCo → validate actions, timing, task score, and video.**

The test uses the real FastAPI lifecycle, isolated SQLite database, scheduler, and background workers. Only unrelated physical USB discovery is disabled. Training, export, camera capture, and action execution are not mocked. This covers application APIs and deployment recipes, not browser interactions or remote SSH trainers.

## Daily workflow

`.github/workflows/simulation-application-e2e.yml` runs daily at 19:00 UTC or by manual dispatch on a self-hosted GPU. Only ACT is enabled. Scheduled workflows run from the default branch; use manual dispatch for a test branch.

1. **Controls:** compare scripted expert (`drive`) and no actions (`off`) on five identical seeds at a 0.04 m/s belt speed. The expert must beat no-action before the training job starts. Also check decoded MJPEG image freshness through the application's camera builder.
2. **Training:** import the full pinned recording through the API. Submit a normal training job using the application's payload defaults: five epochs, batch size eight, automatic data-loader workers, a 10% validation split, and BF16 mixed precision on CUDA. Automatic batch scaling, compilation, LoRA, SnapFlow, and augmentation remain off; all supported export formats are requested. Require all five epochs, finite published losses, and the OpenVINO export used for deployment.
3. **Deployment and inference:** download the returned model's OpenVINO CPU bundle using its model ID, environment, and recorded task. Execute its unchanged `runtime.yaml` for five completed episodes on the control seeds. There is no separate fixed checkpoint or model-import path. Other export formats are not independently validated by this E2E.

A completed training job or moving arm is not a task-quality pass. The fresh model must score strictly above the no-action control. This is a minimum regression signal, not production qualification.

Controls have a 30-minute CI deadline; training-to-inference has six hours. The job allows 7.5 hours in total, reserving an additional hour for setup, simulator cleanup, and artifact uploads. The self-hosted runner must support GitHub's self-repository action syntax (`$/...`, runner 2.336.0 or newer). API training polling defaults to five hours (`SIM_E2E_JOB_TIMEOUT_SECONDS`). Timeouts fail rather than reducing epochs, cropping the dataset, or reporting incomplete work as a pass.

## Rollout contracts

The recording contains overview/wrist cameras and six-dimensional state/actions at 30 FPS. Each emitted action, including warmup, must contain six finite joint values; invalid actions explicitly fail even if Runtime logs callback errors or the simulator discards them. Require real actions and robot movement.

Trials pause during setup, seed/reset, verify queued commands have taken effect, warm up Runtime, then reseed/reset before scoring. Episode auto-reset must stay enabled while scoring: disabling it pauses the conveyor. Stop on the first completed episode and account for all ten items. Score is `correct / (correct + wrong + missed)`.

Control frequency and every expected camera's observed distinct-frame frequency must reach 80% of target (24 Hz). Camera update gaps must not exceed 0.25 seconds, from startup through observed episode completion, not just the last Runtime tick. Every simulator timing sample must report every expected camera (using `SIM_E2E_CAMERA_MAP` when configured) and meet camera-rate and real-time-factor (0.9–1.1) limits; a healthy final sample cannot hide an earlier slowdown. HTTP health polling stays off the control thread.

Use the application's `SharedCamera` recipe, not direct buffered `IPCamera.read_latest()` calls: a newly assigned timestamp can hide an old image. The camera-freshness integration test encodes a frame counter into a local MJPEG stream and checks actual image age after a consumer pause. For this moving conveyor fixture, every expected view must also change from its initial scored pixels at least once. This rejects an entirely frozen view even when capture assigns new timestamps. The comparison owns its initial pixels because camera buffers may be reused. Pixel changes are not treated as frame-rate measurements: legitimate stationary streaks are allowed, and this whole-trial content check does not rule out every intermittent freeze or replay. Separate regressions cover identical pixels with increasing timestamps, frozen/stopped cameras, a stall after the last tick, and an ended video feed.

Record a side-by-side video across all rollout episodes, including failing-score runs. Encoding must finish cleanly and full FFmpeg decoding must succeed. Either input ending must stop the composite rather than repeat a frozen view. Decoded duration and frame count must cover the rollout at its target FPS, with one second of timestamp/flush tolerance.

The harness observes Runtime's callback bus but does not replace camera queues, synchronous refill, smoothing, inference, or action execution. Each `runtime.run()` reconnects/resets its source. Fix production configuration when sorting regresses rather than adding a test-only controller. ACT export preprocessing converts uint8 inputs to float before resizing, matching native training; re-export old checkpoints to receive that metadata fix.

## Recording and dependencies

The pinned dataset is [physicalai-studio/conveyor-sort-sim-demos-100-episodes](https://huggingface.co/datasets/physicalai-studio/conveyor-sort-sim-demos-100-episodes), revision `17db8f758b32141d795a71a064cfa9f5c6341b5b`: 100 episodes, 132,547 frames, and a 0.04 m/s belt. The pin lives in `application/backend/tests/integration/simulation_e2e.py`. Downloads/training check episode count; training also checks scene, belt speed, FPS, and state/action shapes. Read the exact single task from `meta/tasks.parquet`. Original recording seeds are unknown, so rollout seeds cannot be certified as unseen.

The MuJoCo plugin is pinned to PyPI release `0.5.0`, with the application's Studio SDK on the 0.5 series. The backend's `sim-e2e` extra supplies the plugin. The suite uses the installed Runtime and MuJoCo plugin without enforcing their revisions or versions, and records their installed versions in the E2E reports. Use EGL rendering; software OSMesa was observed near 2 FPS. Set repository variable `SIM_E2E_CUDA_VISIBLE_DEVICES` to select the GPU (`0` by default), and provide `HF_TOKEN` for gated assets.

CI reuses Hugging Face's standard Hub cache outside checkout cleanup (`HF_HOME`/`HF_HUB_CACHE` may override it), exposing the returned snapshot as `SIM_E2E_DATASET_DIR`. Keep that cache outside the workspace. CI isolates `STORAGE_DIR` and `SETTINGS_FILE`; never point tests at a running Studio installation.

## Outputs

GitHub Actions artifacts are uploaded on failure as well as success:

- **`sim-full-act`**, retained 14 days: JSON provenance/results, JUnit, logs, performance video, and deployment recipe. Job/model IDs link training to inference.
- **`training-outputs-act`**, retained two days: published model directories, exports, metrics/config, backend logs, the downloaded `runtime.zip`, and any partial checkpoints/Lightning logs already written before failure.

The recording, snapshots, database, and settings are not uploaded. GitHub's built-in expiry handles retention; there is no cross-run deletion script or Actions write permission. Two daily training artifacts may overlap, so allow storage headroom. Local workspace files may be removed by the next checkout, but uploaded artifacts survive independently until expiry. Files never written before a failure cannot be recovered.

## Local run

From `application/backend/`, install the normal hardware and test extras. Select an unused GPU; CI concurrency does not protect independently launched local processes. Allow disk/RAM for import copies, snapshots, exports, and buffered HTTP bodies.

```bash
uv sync --frozen --extra tests --extra sim-e2e --extra cuda
work=$(mktemp -d)
export STORAGE_DIR="$work/studio" SETTINGS_FILE="$work/studio/settings.json"
export SIM_E2E_OUTPUT_DIR="$work/evidence"
export SIM_E2E_DATASET_DIR=/absolute/path/to/the/pinned/recording
export SIM_E2E_CONTROLS_FILE="$SIM_E2E_OUTPUT_DIR/controls.json"
export SIM_E2E_HTTP_URL=http://127.0.0.1:8080
export SIM_E2E_OWNER=your-separately-started-conveyor_sort-owner
export SIM_E2E_RECORD_VIDEO=true
uv run --no-sync pytest tests/integration/test_simulation_e2e.py -k test_simulator_controls -v
uv run --no-sync pytest tests/integration/test_simulation_e2e.py -k test_train_export_and_run_in_simulation -v
```

Start the pinned simulator owner separately. Video recording requires FFmpeg. For a CPU run, install `--extra cpu` instead of `cuda` and set `SIM_E2E_ACCELERATOR=cpu`; the same full recording, five epochs, and rollout gates apply. There is no train-only or synthetic-smoke mode.
