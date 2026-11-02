# Simulation application E2E testing

The canonical simulation workflow lives in `application/backend/tests/integration/test_simulation_e2e.py`. It exercises Studio rather than maintaining a parallel library training/deployment pipeline. **Training regression**, **deployment wiring**, and **task quality** are separate verdicts; a completed job, finite loss, or moving arm is not a sorting success.

## Application ownership

The test runs the real FastAPI lifecycle, isolated SQLite database, scheduler, and on-demand worker processes. Discovery of unrelated physical USB robots is disabled; training, export, and camera capture are not mocked.

- Create project/environment profiles and import datasets through normal APIs.
- Submit `POST /api/jobs:train` and poll the normal job/model endpoints. Production workers own snapshots, policy construction, normalization, validation, checkpointing, and export.
- Download the model's Runtime bundle through the normal export endpoint with `environment_id`, `device`, and the recorded task. Execute its **unchanged `runtime.yaml`**. A telemetry observer is attached to Runtime's callback bus because the SDK config loader has no live-callback argument; it does not replace cameras, queues, smoothing, inference, or action execution.

The harness owns fixture preparation, simulator lifecycle/seeds, scoring, and evidence collection only. This is API/job and exported-Runtime integration testing, not browser automation or end-to-end testing of remote SSH trainers. The application's GUI Runtime session and downloaded recipe share `runtime/config_builder.py` configuration.

## Daily checks

`.github/workflows/simulation-application-e2e.yml` runs daily at 19:00 UTC or manually on one self-hosted GPU. This testing branch enables only the connected ACT training-to-inference run while the workflow is verified; other policies remain disabled in the E2E matrix. GitHub only triggers scheduled workflows on the default branch; use manual dispatch to test a non-default branch.

1. **Simulator controls:** compare scripted expert (`drive`) and no actions (`off`) on five identical seeds at 0.04 m/s. Controls must pass before the connected training-to-inference job starts. Also check actual MJPEG image freshness through the application's camera builder.
2. **Training regression:** import the full pinned 100-episode recording through the API and train with the same defaults users see in the UI: five epochs, batch size eight, automatic data-loader workers, a 10% validation split, and BF16 mixed precision on CUDA (32-bit on CPU). Automatic batch scaling, compilation, LoRA, SnapFlow, and image augmentation remain disabled; all supported export formats are requested. The request uses the application's payload defaults rather than private E2E hyperparameters. Verify completion of all five epochs, the selected checkpoint's optimizer/counter state, application checkpoint reload, finite published validation/training metrics when logged, and a Torch export that Runtime's `InferenceModel` can load on CPU. Training completion alone is not a task-performance pass. The training gate verifies Torch export/reload and requires the OpenVINO export used in the next step; requesting all supported formats does not prove that every other format succeeded.
3. **Inference with that exact model:** download the just-trained model's application-generated OpenVINO CPU bundle and execute its unchanged Runtime recipe for five completed conveyor episodes, on the same seeds as the controls. Gate real actions, timing, and a sorting score strictly above the no-action control. Training and inference share the same job/model provenance; there is no separate checkpoint download, model import, qualification manifest, or fixed-checkpoint evaluation. Every emitted action, including warmup, must contain six finite joint values; invalid actions explicitly stop and fail the run even if the simulator discards them. The 180-second episode deadline is a failure safeguard, not a scoring window.

The connected training-to-inference job has a six-hour CI deadline; controls retain a 30-minute deadline, for a maximum configured serial budget of 6.5 hours. `SIM_E2E_JOB_TIMEOUT_SECONDS` separately bounds API training-job polling (five hours by default). These are failure safeguards, not training limits: a timeout fails rather than reducing epochs, cropping the dataset, or reporting incomplete work as a pass. Full-recording training and all supported exports cost substantially more time and disk than a smoke test; the deadlines are not measured runtime estimates.

Reports separate training/reload, export/actions, timing, and quality for the same fresh model. A failed or incomplete inference stage cannot be replaced by a training pass. Policies above Studio's advertised device-memory minimum are **unsupported hardware**, not training passes. CI records `inference-act-conveyor_sort.mp4` across all rollout episodes, including failing-score runs. Video recording must finish cleanly, and FFmpeg must decode the entire saved video without errors. Input errors fail encoding, and the side-by-side video stops when either camera feed ends, so a surviving feed cannot hide a frozen view. Decoded duration and frame count must cover the recorded rollout at its target FPS, allowing one second of timestamp/flush tolerance. Reports and video retain a 14-day history, including on failure. Training/deployment outputs are uploaded on failure too, but only the latest upload per workflow, branch, and policy is kept, with a 14-day fallback expiry. Runner-local files remain ephemeral and may be cleaned by the next CI checkout; uploaded artifacts are independent of that workspace.

### Inspecting a run

Open the GitHub Actions run's **Artifacts** section:

- **`sim-full-act`** contains JSON reports/provenance, JUnit results, logs, the inference video, and the deployment recipe.
- **`training-outputs-act`** contains the complete published model directory (checkpoint, training metrics/configuration, and every produced export), the exact downloaded `runtime.zip` deployment bundle, backend logs, and JSON job/model/request provenance. If training fails before publishing, any checkpoints and Lightning logs already written to the training cache are retained instead. Missing outputs cannot be recovered if a job dies before writing them.

The large training artifact is separate from the report artifact so the summary job does not download model weights. The imported recording, snapshots, database, and settings are not uploaded. After a training artifact uploads successfully, the policy job verifies its artifact ID through GitHub's API, then deletes older same-name training artifacts from this workflow and branch. Other policies, branches, workflows, reports, and videos are untouched. Failed or missing uploads leave the previous artifact in place; partial outputs from failed training can replace older outputs if their upload succeeds. Re-running an older workflow run cannot delete artifacts from newer runs. Cleanup requires `actions: write` only on the policy job. Uploading before deleting temporarily requires room for both versions; quota or upload failures do not trigger deletion. If cleanup fails, the job reports the error and the 14-day expiry limits accumulation; if runs stop, the latest artifact also expires after 14 days. Local runs retain their files under the selected Studio `STORAGE_DIR`; copy those elsewhere before reusing or cleaning that storage.

### Hardware and dependency contracts

Use normal Studio training settings, not the old harness's private MolmoAct2/XR0 constructor overrides. On a 40 GiB GPU, Studio's advertised 80 GiB MolmoAct2 and 60 GiB XR0 defaults are reported unsupported; testing those defaults requires larger hardware. This is a coverage limit, not evidence that those policies pass. Batch size and device-dependent precision follow the UI. UI-default parity is verified for the enabled ACT matrix; policies requiring camera-slot mappings (such as SmolVLA) need that UI mapping reproduced before being enabled.

Rendering uses EGL rather than software OSMesa, previously observed near 2 FPS. Set repository variable `SIM_E2E_CUDA_VISIBLE_DEVICES` to the physical GPU (`0` by default); memory is not combined across cards. Provide `HF_TOKEN` with access to gated model/tokenizer assets.

The application backend's `sim-e2e` extra supplies the MuJoCo plugin. Runtime and the plugin are pinned together at `1cc910b451508ce40cd7319ece2a29283fadfcfb`, following current main. Both installed VCS revisions and both project lockfiles are checked. CI isolates `STORAGE_DIR` and `SETTINGS_FILE`; it does not reuse a developer's Studio database or settings.

## Reproducible episodes

The E2E workflow uses only the **100-episode** recording, including scheduled runs. The `50` and `120` options have been removed. The repository-reviewed ID and immutable revision live in `application/backend/tests/integration/simulation_e2e.py`.

| Choice | Hugging Face dataset                                                                                  | Episodes / frames | Pinned revision                            |
| ------ | ----------------------------------------------------------------------------------------------------- | ----------------- | ------------------------------------------ |
| `100`  | [Expanded recording](https://huggingface.co/datasets/Albertvh97/conveyor-sort-sim-demos-100-episodes) | 100 / 132,547     | `17db8f758b32141d795a71a064cfa9f5c6341b5b` |

The recording contains overview/wrist cameras and six-dimensional state/actions at 30 FPS with a 0.04 m/s belt. It preserves the original 50 episodes and adds 50 stock-expert episodes with seeds 40000–40049: 500 correct items, none wrong or missed, and all 24 shape/colour/damage combinations. It preserves the natural spawn distribution rather than equal bin counts. New videos use H.264 CRF 18; original videos are copied without re-encoding, so encoding differs between sources. The original recording seeds were not captured, so evaluation seeds cannot be certified as unseen across the dataset.

Simulator controls and evaluation stay at 0.04 m/s. Rollouts pass the recording's exact task instruction from `meta/tasks.parquet` to Runtime; missing, blank, or multiple instructions fail before export. Every connected run trains a new model from this recording; it does not change evaluation seeds or the five-episode scoring protocol. Provenance records the repository and revision. Downloads and training reject a recording with the wrong episode count.

CI downloads the pinned recording into Hugging Face's standard Hub cache (normally `~/.cache/huggingface/hub`, configurable with `HF_HOME` or `HF_HUB_CACHE`) and exports the returned snapshot path as `SIM_E2E_DATASET_DIR` for later steps. The cache sits outside the checkout workspace, so subsequent jobs and runs on the same self-hosted runner reuse it instead of downloading the recording again. Keep the configured cache outside the workspace and preserve it when cleaning the runner; a new runner or cleared cache needs a cold download.

Each trial pauses the conveyor during setup, fixes a seed, resets, verifies queued commands have taken effect, connects the policy runtime, then resumes the belt. Do not disable episode auto-reset during scoring: in this simulator that also pauses the belt. Scoring stops on the first completed episode and validates all 10 items as correct/wrong/missed. The score is `correct / (correct + wrong + missed)`.

Simulation cameras use Runtime's existing `SharedCamera.from_config(...)` with a nested `physicalai.capture.IPCamera` recipe, matching Studio's production Runtime configuration. The publisher continuously drains the MJPEG stream; inference reads its latest shared frame even while setup, video initialization, or model computation is paused. Direct `IPCamera.read_latest()` calls on this pinned backend consume the next buffered OpenCV frame and timestamp it at read time. That can leave an image seconds old despite apparently fresh timestamps and healthy camera/control rates. Unique timestamps and simulator camera FPS alone do not prove observation freshness.

The controls job also runs `test_simulation_camera_freshness.py`: a local 30-FPS MJPEG stream encodes an actual frame counter into each image. After a 1.5-second consumer pause, the rollout camera must return an image no more than 0.25 seconds old. This checks decoded image content, not the timestamp assigned by the reader, and prevents a regression to direct buffered capture.

HTTP health polling runs on a separate thread, not inside the control tick callback. Results record wall-clock control frequency, inference p95 latency, unique camera timestamps, simulator speed/camera rates, seeds, checkpoint identity, and code/dependency provenance. The timing gate requires at least 80% of the recorded control and camera frequency (24 Hz), with simulator real-time factor between 0.9 and 1.1. Every camera required by the recording must also deliver distinct frames to Runtime at at least 24 Hz, with no observed update gap longer than 0.25 seconds, including startup and the end of the trial. A cached simulator FPS cannot substitute for actual frame progress. It checks every simulator timing sample collected during the scored episode, not just the final sliding-window measurement. It includes startup-to-first-action delay after a paused warmup and reseed/reset. Investigate camera blocking, inference cost, action units/order, and preprocessing parity when timing or task scores regress; do not slow the belt to make a policy pass.

Execution settings now come exclusively from the application's downloaded recipe. The current shared builder uses synchronous half-chunk refill and the default five-frame `LerpSmoother`, including ACT; the harness no longer substitutes full-chunk/no-crossfade ACT execution. If that production configuration fails sorting, fix it once in the application/shared Runtime path rather than adding a test-only controller. Each `runtime.run()` reconnects and resets its source automatically; calling `source.reset()` between runs is invalid because it is disconnected.

### ACT export parity diagnostic

A bounded CPU-only ACT probe compared native predictions with OpenVINO predictions on five recorded conveyor observations, using Runtime's real camera-input conversion. It exposed integer interpolation rounding when uint8 camera bytes were resized before float conversion. ACT export recipes now use the existing `to_float_tensor` processor before `resize`, matching training's float resize order. Re-export existing checkpoints to receive this metadata fix; model weights and state-dict keys are unchanged.

## Fresh-model verdicts

The connected job uses only the model returned by its normal training job. It records that model ID and training job ID in `training.json` and `result.json`, and uses the model's normal download endpoint for deployment. No Hugging Face checkpoint registry or pre-trained evaluation baseline is used.

The controls artifact must contain successful expert/no-action trials on the exact rollout seeds. The harness computes control scores from their completed episodes rather than trusting summary scores. The newly trained policy must score strictly above the no-action control; zero sorting success therefore cannot pass. This is a minimum regression signal, not deployment qualification or a claim of production-level accuracy. Training variability can make this gate fail, and failures must be investigated rather than hidden behind a fixed checkpoint or reduced training settings.

## Local commands

From `application/backend/`, install the `tests`, `sim-e2e`, and appropriate hardware extras (`cuda` or `cpu`). Select an unused GPU before any CUDA run; CI concurrency does not protect independently launched training processes.

```bash
uv sync --frozen --extra tests --extra sim-e2e --extra cuda

# Always use a disposable application database/settings, not your running Studio store.
work=$(mktemp -d)
export STORAGE_DIR="$work/studio"
export SETTINGS_FILE="$work/studio/settings.json"
export SIM_E2E_OUTPUT_DIR="$work/evidence"
export SIM_E2E_POLICY=act
export SIM_E2E_DATASET=100
export SIM_E2E_DATASET_DIR=/absolute/path/to/the/pinned/recording

# Full recording, UI-default training and exports; no simulator.
SIM_E2E_MODE=train uv run --no-sync pytest tests/integration/test_simulation_e2e.py -k test_train_export_and_run_in_simulation -v
```

For ACT-only full-recording CPU training, install `--extra cpu` instead of `cuda` and set `SIM_E2E_ACCELERATOR=cpu`; training still uses the default five epochs. The separate `test_application_training_smoke` CPU integration test uses a small synthetic 60-frame fixture with the same training defaults; CI's simulation E2E never substitutes that fixture for the recording. `SIM_E2E_EPOCHS` and `SIM_E2E_FULL_DATASET` no longer change training. Full training and import create copies/snapshots and large checkpoints; allow sufficient disk and RAM. The in-process HTTP test client can buffer large upload/download bodies.

For the connected run, separately start the pinned `physicalai-mujoco` owner in `conveyor_sort` and set `SIM_E2E_HTTP_URL` and `SIM_E2E_OWNER` to that owner. Run controls first and then train and deploy a fresh model:

```bash
export SIM_E2E_CONTROLS_FILE="$SIM_E2E_OUTPUT_DIR/controls.json"
uv run --no-sync pytest tests/integration/test_simulation_e2e.py -k test_simulator_controls -v
SIM_E2E_MODE=full SIM_E2E_RECORD_VIDEO=true uv run --no-sync pytest tests/integration/test_simulation_e2e.py -k test_train_export_and_run_in_simulation -v
```

Video recording requires FFmpeg. `SIM_E2E_MODE=evaluate`, `SIM_E2E_CHECKPOINT_MANIFEST`, and `SIM_E2E_BASE_MODEL_DIR` are no longer supported; the E2E always starts a fresh training job rather than importing a fixed or resumed model.
