# [EXPERIMENTAL] VLA Evaluation Harness

This folder provides model servers that let you benchmark Physical AI Studio policies with AllenAI's [vla-evaluation-harness](https://github.com/allenai/vla-evaluation-harness).

## Installation

### vla-evaluation-harness

The upstream harness is checked out as a git submodule at
`library/benchmarks/vla-evaluation-harness/vla-evaluation-harness/` (see
`.gitmodules`). Initialize it on a fresh clone:

```bash
git submodule update --init --recursive
```

The harness orchestrator and benchmarks may still run in Docker, but the model
servers below are launched directly in your active Python environment.

### Physical AI Studio / PhysicalAI

The model servers in this folder require `vla-eval` plus the Physical AI Studio
`physicalai` library (or the open-source
[PhysicalAI](https://github.com/openvinotoolkit/physicalai) inference framework)
and the policy checkpoint dependencies to be installed in the **active** Python
environment.

There is no automatic dependency installation: these scripts are launched with
plain `python`, not `uv run` or `vla-eval serve`. Install everything beforehand
(e.g. `uv sync` from `library/`, or `pip install -e .` for your local setup).

## Examples

Run these from within `library/benchmarks/vla-evaluation-harness`.

### Mode 1 — Exported checkpoint via `InferenceModel`

Run after exporting a policy with `physicalai export`:

```bash
python model_servers/physicalai_harness.py \
  --config configs/pi05_libero_inference_model.yaml
```

### Mode 2 — Direct policy via `jsonargparse` config

Load a `physicalai.policies.Policy` subclass directly from the same `class_path` / `init_args` YAML used for training:

```bash
python model_servers/physicalai_harness.py \
  --config configs/pi05_libero_policy.yaml
```

### Mode 3 — Hardcoded subclass

Use the dedicated `Pi05LiberoServer` with no separate policy YAML:

```bash
python model_servers/pi05_libero.py --config configs/pi05_libero_direct.yaml
```

Or with no YAML at all:

```bash
python model_servers/pi05_libero.py --port 8000

# or with explicit checkpoint / device
python model_servers/pi05_libero.py \
  --port 8000 \
  --pretrained_name_or_path lerobot/pi05_libero_finetuned_v044 \
  --device cuda
```

### CLI overrides

`--port` and `--host` are handled at the server level, while other fields are passed as `--args.<key>`:

```bash
python model_servers/physicalai_harness.py \
  --config configs/pi05_libero_policy.yaml \
  --port 8001 \
  --args.device=cpu
```

### Notes

- These configs and entry points are designed for direct `python model_servers/<file>.py --config <yaml>` usage. They no longer contain the `script:` key that `vla-eval serve` / `vla-eval test --server` expect.
- If you still need the upstream `vla-eval serve` / `vla-eval test --server` path, restore the `script:` field pointing at the server file. That path is not covered by this README.
- **Image layout contract**: vla-eval delivers images as `(H, W, C)` uint8. The bridge converts them differently depending on the policy type:
  - `physicalai.policies.Policy` subclasses (Training checkpoint path): images are converted to `(B, C, H, W)` float32 in `[0, 1]` — the layout the training-time preprocessors expect.
  - `physicalai.inference.InferenceModel` (Exported artifact path): images are kept as `(B, H, W, C)` uint8 — the exported preprocessors handle layout detection and normalisation internally.

## Running a full LIBERO evaluation (two terminals)

The model server (this folder) and the `vla-eval` orchestrator (which runs the
LIBERO benchmark in Docker) are two separate processes. This walks through
running the full loop with `Pi05LiberoServer` end to end.

### 0. One-time environment setup

Model-server environment (this repo's own venv — needs `physicalai-train` and
`vla-eval` together). Pick the PyTorch backend extra that matches your GPU
(`--extra cu128` for NVIDIA CUDA, `--extra xpu` for Intel, `--extra cpu` for
no GPU):

```bash
cd library
uv sync --extra cu128 --extra pi05          # physicalai-train + pi05/pi0 extras + CUDA torch
uv pip install -e benchmarks/vla-evaluation-harness/vla-evaluation-harness   # adds the vla_eval package into the same venv
```

`vla-eval` CLI / orchestrator environment (the submodule's own venv — runs the
benchmark inside Docker, so it doesn't need `physicalai-train` at all):

```bash
cd library/benchmarks/vla-evaluation-harness/vla-evaluation-harness
uv sync --python 3.11 --all-extras --dev
```

Docker must be installed and running — the first `vla-eval run` invocation
pulls `ghcr.io/allenai/vla-evaluation-harness/libero:latest` if it isn't
already cached, which can take a while.

### 1. Terminal 1 — start the model server

```bash
cd library/benchmarks/vla-evaluation-harness
python model_servers/pi05_libero.py \
  --config configs/pi05_libero_direct.yaml
# Use --args.device=cpu to force CPU; omit (or --args.device=cuda) for NVIDIA GPU.
```

Wait for the `Starting server on ws://0.0.0.0:8000` log line (the first run
also downloads the `lerobot/pi05_libero_finetuned_v044` weights from the HF
Hub). In a separate throwaway shell, confirm readiness:

```bash
curl -fsS http://localhost:8000/health
# {"status": "ok"}
```

Leave this terminal running — the same server instance can handle multiple
evaluation runs below.

### 2. Terminal 2 — smoke test first

Before committing to a full suite, validate the whole pipeline (image/state
mapping, action spec, chunking) with a 1-task, 1-episode run:

```bash
cd library/benchmarks/vla-evaluation-harness/vla-evaluation-harness
uv run vla-eval run --config configs/benchmarks/libero/smoke_test.yaml
```

This should complete quickly with no connection/protocol errors and print a
1/1 episode summary.

### 3. Terminal 2 — full evaluation

Once the smoke test passes, run a full suite, e.g. LIBERO-Spatial
(10 tasks × 50 episodes = 500 episodes):

```bash
uv run vla-eval run --config configs/benchmarks/libero/spatial.yaml
```

CPU inference of a VLA policy at this scale can take a long time — consider
running this inside `tmux`/`screen`/`nohup` if it needs to survive a
disconnect.

### Recording episodes (video + step data)

By default, evaluation records step metrics (`record_step: true`) and skips
video (`record_video: false`). You can record MP4 videos of each episode using
either approach:

#### Option A — CLI flag (no config edits required)

Pass `--record-video` to enable video recording across all tasks at 20 FPS:

```bash
uv run vla-eval run --config configs/benchmarks/libero/10.yaml --record-video
```

To skip all recording entirely (no SQLite, no MP4s/JSONL, in-memory summary
only), pass `--no-save` (mutually exclusive with `--record-video`):

```bash
uv run vla-eval run --config configs/benchmarks/libero/10.yaml --no-save
```

#### Option B — YAML `recording:` block (custom FPS / settings)

Add a `recording:` block as a sibling of `params:` inside the benchmark entry in
`configs/benchmarks/libero/10.yaml`:

```yaml
benchmarks:
  - benchmark: "vla_eval.benchmarks.libero.benchmark:LIBEROBenchmark"
    subname: libero_10
    episodes_per_task: 50
    params:
      suite: libero_10
      seed: 7
      num_steps_wait: 10
    recording:
      record_video: true
      video_fps: 10
```

Videos land in `results/episodes/libero_10/task{NNNN}_ep{NNNN}_{status}.mp4` alongside matching `.jsonl` step files. MP4 creation and EGL headless rendering are handled automatically.

### 4. Inspect results

Single-shard runs auto-merge on completion. Results land under
`benchmarks/vla-evaluation-harness/vla-evaluation-harness/results/` (per
`output_dir: "./results"` in the benchmark config):

```bash
cat results/libero_spatial_aggregate.json   # mean_success + per-task breakdown
cat results/libero_spatial_episodes.jsonl   # one line per episode
```

Also watch Terminal 1's logs during the run — repeated `ERROR` messages
indicate an observation/action mapping problem even if the evaluation
"completes" (episode failures don't abort the whole run).
