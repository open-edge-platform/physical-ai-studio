# Simulation application E2E testing

The canonical simulation workflow lives in `application/backend/tests/integration/test_simulation_e2e.py`. It exercises Studio rather than maintaining a parallel library training/deployment pipeline. **Training regression**, **deployment wiring**, and **task quality** are separate verdicts; a completed job, finite loss, or moving arm is not a sorting success.

## Application ownership

The test runs the real FastAPI lifecycle, isolated SQLite database, scheduler, and on-demand worker processes. Discovery of unrelated physical USB robots is disabled; training, export, and camera capture are not mocked.

- Create project/environment profiles and import datasets through normal APIs.
- Submit `POST /api/jobs:train` and poll the normal job/model endpoints. Production workers own snapshots, policy construction, normalization, validation, checkpointing, and export.
- Import existing models through the supported `physicalai-studio models import-dir --copy` command. Qualified standalone checkpoints are first staged using the application's shared checkpoint loader/exporter, not a second policy/export implementation.
- Download the model's Runtime bundle through the normal export endpoint with `environment_id`, `device`, and the recorded task. Execute its **unchanged `runtime.yaml`**. A telemetry observer is attached to Runtime's callback bus because the SDK config loader has no live-callback argument; it does not replace cameras, queues, smoothing, inference, or action execution.

The harness owns fixture preparation, simulator lifecycle/seeds, scoring, and evidence collection only. This is API/job and exported-Runtime integration testing, not browser automation or end-to-end testing of remote SSH trainers. The application's GUI Runtime session and downloaded recipe share `runtime/config_builder.py` configuration.

## Daily checks

`.github/workflows/simulation-application-e2e.yml` runs daily at 19:00 UTC or manually on one self-hosted GPU. This testing branch enables only ACT training and evaluation while the workflow is verified; other policies remain disabled in the E2E matrix. GitHub only triggers scheduled workflows on the default branch; use manual dispatch to test a non-default branch.

1. **Simulator controls:** compare scripted expert (`drive`) and no actions (`off`) on five identical seeds at 0.04 m/s. Expert failure blocks quality evaluation, not training regression. Also check actual MJPEG image freshness through the application's camera builder.
2. **Training regression:** derive a clearly labelled fixture with 20 contiguous frames from each of three source episodes, preserving FPS and decoded pixels. Import through the API and request a normal one-epoch job at batch size two and `val_split=0.2`: two training episodes produce **20 optimizer updates**. There is no hidden Trainer step override. Verify the published checkpoint's optimizer/counter state, application checkpoint reload, finite published validation/training metrics when logged, and the requested Torch export. This fixture is not suitable for behavioral qualification.
3. **Qualified export/evaluation:** verify a reviewed checkpoint's SHA-256, stage it with the application's exporter, import the model normally, and execute the application-generated OpenVINO CPU recipe for five completed conveyor episodes. The 180-second episode deadline is a failure safeguard, not a scoring window.

Each job is limited to 30 minutes; the serial two-cell ACT matrix plus controls has a maximum configured budget of 1.5 hours. Cold caches or slow hardware can time out; incomplete work is not a pass. `SIM_E2E_JOB_TIMEOUT_SECONDS` separately bounds API training-job polling (20 minutes by default).

Reports separate training/reload, export/actions, timing, and quality. Missing qualified checkpoints are **baseline unavailable**. Policies above Studio's advertised device-memory minimum are **unsupported hardware**, not training passes. Report-only artifacts (JSON, JUnit, logs, recipes, video) last 14 days. Daily model files are not uploaded or explicitly deleted by the test; their workspace is ephemeral and may be cleaned by the next CI checkout. Retain model artifacts separately when training locally or through Studio.

### Hardware and dependency contracts

Use normal Studio training settings, not the old harness's private MolmoAct2/XR0 constructor overrides. On a 40 GiB GPU, Studio's advertised 80 GiB MolmoAct2 and 60 GiB XR0 defaults are reported unsupported; testing those defaults requires larger hardware. This is a coverage limit, not evidence that those policies pass. Batch size is two; ACT uses `32-true`, other policies `bf16-true`.

Rendering uses EGL rather than software OSMesa, previously observed near 2 FPS. Set repository variable `SIM_E2E_CUDA_VISIBLE_DEVICES` to the physical GPU (`0` by default); memory is not combined across cards. Provide `HF_TOKEN` with access to gated model/tokenizer assets.

The application backend's `sim-e2e` extra supplies the MuJoCo plugin. Runtime and the plugin are pinned together at `1cc910b451508ce40cd7319ece2a29283fadfcfb`, following current main. Both installed VCS revisions and both project lockfiles are checked. CI isolates `STORAGE_DIR` and `SETTINGS_FILE`; it does not reuse a developer's Studio database or settings.

## Reproducible episodes

The E2E workflow uses only the **100-episode** recording, including scheduled runs. The `50` and `120` options have been removed. The repository-reviewed ID and immutable revision live in `application/backend/tests/integration/simulation_e2e.py`.

| Choice | Hugging Face dataset                                                                                  | Episodes / frames | Pinned revision                            |
| ------ | ----------------------------------------------------------------------------------------------------- | ----------------- | ------------------------------------------ |
| `100`  | [Expanded recording](https://huggingface.co/datasets/Albertvh97/conveyor-sort-sim-demos-100-episodes) | 100 / 132,547     | `17db8f758b32141d795a71a064cfa9f5c6341b5b` |

The recording contains overview/wrist cameras and six-dimensional state/actions at 30 FPS with a 0.04 m/s belt. It preserves the original 50 episodes and adds 50 stock-expert episodes with seeds 40000–40049: 500 correct items, none wrong or missed, and all 24 shape/colour/damage combinations. It preserves the natural spawn distribution rather than equal bin counts. New videos use H.264 CRF 18; original videos are copied without re-encoding, so encoding differs between sources. The original recording seeds were not captured, so evaluation seeds cannot be certified as unseen across the dataset.

Simulator controls and evaluation stay at 0.04 m/s. Rollouts pass the recording's exact task instruction from `meta/tasks.parquet` to Runtime; missing, blank, or multiple instructions fail before export. The recording does not retrain or replace a qualified checkpoint, change evaluation seeds, or change the five-episode scoring protocol. Provenance records the repository and revision. Downloads and training reject a recording with the wrong episode count.

Each trial pauses the conveyor during setup, fixes a seed, resets, verifies queued commands have taken effect, connects the policy runtime, then resumes the belt. Do not disable episode auto-reset during scoring: in this simulator that also pauses the belt. Scoring stops on the first completed episode and validates all 10 items as correct/wrong/missed. The score is `correct / (correct + wrong + missed)`.

Simulation cameras use Runtime's existing `SharedCamera.from_config(...)` with a nested `physicalai.capture.IPCamera` recipe, matching Studio's production Runtime configuration. The publisher continuously drains the MJPEG stream; inference reads its latest shared frame even while setup, video initialization, or model computation is paused. Direct `IPCamera.read_latest()` calls on this pinned backend consume the next buffered OpenCV frame and timestamp it at read time. That can leave an image seconds old despite apparently fresh timestamps and healthy camera/control rates. Unique timestamps and simulator camera FPS alone do not prove observation freshness.

The controls job also runs `test_simulation_camera_freshness.py`: a local 30-FPS MJPEG stream encodes an actual frame counter into each image. After a 1.5-second consumer pause, the rollout camera must return an image no more than 0.25 seconds old. This checks decoded image content, not the timestamp assigned by the reader, and prevents a regression to direct buffered capture.

HTTP health polling runs on a separate thread, not inside the control tick callback. Results record wall-clock control frequency, inference p95 latency, unique camera timestamps, simulator speed/camera rates, seeds, checkpoint identity, and code/dependency provenance. The timing gate requires at least 80% of the recorded control and camera frequency (24 Hz), with simulator real-time factor between 0.9 and 1.1. It includes startup-to-first-action delay after a paused warmup and reseed/reset. Investigate camera blocking, inference cost, action units/order, and preprocessing parity when timing or task scores regress; do not slow the belt to make a policy pass.

Execution settings now come exclusively from the application's downloaded recipe. The current shared builder uses synchronous half-chunk refill and the default five-frame `LerpSmoother`, including ACT; the harness no longer substitutes full-chunk/no-crossfade ACT execution. If that production configuration fails sorting, fix it once in the application/shared Runtime path rather than adding a test-only controller. Each `runtime.run()` reconnects and resets its source automatically; calling `source.reset()` between runs is invalid because it is disconnected.

### ACT export parity diagnostic

A bounded CPU-only ACT probe compared native predictions with OpenVINO predictions on five recorded conveyor observations, using Runtime's real camera-input conversion. It exposed integer interpolation rounding when uint8 camera bytes were resized before float conversion. ACT export recipes now use the existing `to_float_tensor` processor before `resize`, matching training's float resize order. Re-export existing checkpoints to receive this metadata fix; model weights and state-dict keys are unchanged.

## Qualifying a checkpoint

`application/backend/configs/simulation-e2e/checkpoints.json` deliberately starts with `null` for each policy. No usable task-performance baseline has been demonstrated yet. To qualify one:

1. Train a candidate through Studio or the local full-training command below and retain its request, job/model IDs, dependency pins, metrics, and checkpoints. The harness uses the application's normal validation objective and normalization.
2. Confirm simulator controls and compare native versus exported predictions on representative recorded observations. Check camera views/order, normalization, action units, and timing.
3. Evaluate the candidate repeatedly on the fixed protocol, inspect videos, and measure score variability.
4. Publish the trusted Studio-generated `.ckpt` to a versioned model repository. Record its immutable revision, SHA-256, and a positive minimum score justified by measured results in the manifest, reviewed as code.

Each entry has exactly these fields (placeholders below must be replaced):

```json
{
  "repo_id": "organization/qualified-conveyor-policy",
  "revision": "<40-character commit SHA>",
  "filename": "model.ckpt",
  "sha256": "<64-character SHA-256>",
  "min_score": 0.5
}
```

`0.5` is an example, not a measured threshold. Use a threshold supported by qualification trials and their variability. Only trusted, checksum-verified training artifacts belong in this manifest. Loading and adapter merging use the application's existing checkpoint/export path; only trusted, reviewed artifacts should be supplied. Checkpoint reload, export, timing, and score must all pass; a 0% sorting score cannot pass qualification.

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

# One normal application epoch on the 60-frame wiring fixture; no simulator.
SIM_E2E_MODE=train uv run --no-sync pytest tests/integration/test_simulation_e2e.py -k test_train_export_and_run_in_simulation -v

# Three full application epochs; optional SIM_E2E_BASE_MODEL_DIR is a complete Studio model.
SIM_E2E_MODE=train SIM_E2E_FULL_DATASET=true SIM_E2E_EPOCHS=3 uv run --no-sync pytest tests/integration/test_simulation_e2e.py -k test_train_export_and_run_in_simulation -v
```

For an ACT-only CPU smoke, install `--extra cpu` instead of `cuda` and set `SIM_E2E_ACCELERATOR=cpu`. Use the pinned 100-episode recording directory. Full training and import create copies/snapshots and large checkpoints; allow sufficient disk and RAM. The in-process HTTP test client can buffer large upload/download bodies.

For evaluation/controls, separately start the pinned `physicalai-mujoco` owner in `conveyor_sort`, set `SIM_E2E_HTTP_URL` and `SIM_E2E_OWNER`, and select `-k test_simulator_controls` or use `SIM_E2E_MODE=evaluate`. `SIM_E2E_RECORD_VIDEO=true` retains video (FFmpeg required). `SIM_E2E_MODE=full` performs full application train → export → Runtime rollout, but an unqualified candidate is never a quality pass.
