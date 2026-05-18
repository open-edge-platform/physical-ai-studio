# Backend Accuracy Comparison: OpenVINO vs PyTorch

Compare exported policy accuracy between OpenVINO and PyTorch backends using two complementary approaches:

1. **Closed-Loop Simulation** (`closed_loop_benchmark.py`) - Full rollout with error accumulation
2. **Numerical Comparison** (`numerical_comparison.py`) - Single-step predictions without simulator

## Project Structure

```
library/scripts/backend_accuracy_comparison/
├── README.md                      # This file
├── closed_loop_benchmark.py       # Main script: Simulator-based benchmark
├── numerical_comparison.py        # Quick script: Numerical validation
├── batch_comparison.py            # Helper: Compare multiple models
├── check_setup.py                 # Environment verification
├── requirements.txt               # Dependencies
├── Makefile                       # Convenience commands
└── examples.sh                    # Usage examples
```

**Demo Notebook:** See [`library/notebooks/benchmark/backend_accuracy_comparison_demo.ipynb`](../../notebooks/benchmark/backend_accuracy_comparison_demo.ipynb) for interactive walkthrough.

## Quick Start

```bash
cd library/scripts/backend_accuracy_comparison

# Setup (development environment)
# Note: Assumes you've already installed the library from library/ root
make check  # Verify environment

# If starting fresh, install from library root first:
# cd ../../ && uv sync --all-extras && cd scripts/backend_accuracy_comparison

# Run quick tests
make test-numerical      # ~2 minutes
make test-closed-loop    # ~10 minutes
```

## Two Complementary Approaches

### 1. Closed-Loop Benchmark (Recommended - Real Accuracy)

**Full simulation with LiberoGym + MuJoCo:**

```bash
python closed_loop_benchmark.py \
    --checkpoint path/to/model.ckpt \
    --policy-class physicalai.policies.Pi05 \
    --task-suite libero_10 \
    --task-ids 0 1 2 \
    --num-episodes 20
```

**What it does:**
- Exports policy to both PyTorch and OpenVINO
- Runs complete episodes in MuJoCo simulator (up to 520 steps each)
- Each action affects next observation (closed-loop)
- **Errors accumulate** over entire episode
- Measures **success rate** - did the robot complete the task?

**Why this matters:**
- Shows **real production accuracy**
- Accounts for error accumulation
- Physics simulator renders new images after each action
- Final metric: Can the robot actually complete tasks?

**Expected output:**
```
PyTorch  Success Rate: 75.0% (45/60 episodes)
OpenVINO Success Rate: 74.5% (44.7/60 episodes)
Delta: 0.5% (within measurement noise)
✅ Backends are EQUIVALENT
```

### 2. Numerical Comparison (Quick Sanity Check)

**Single-step predictions on dataset:**

```bash
python numerical_comparison.py \
    --checkpoint path/to/model.ckpt \
    --policy-class physicalai.policies.ACT \
    --dataset lerobot/pusht \
    --num-samples 100
```

**What it does:**
- Loads samples from HDF5 dataset
- For each observation: predicts action in both backends
- Compares numerical differences (max diff, mean diff)
- **Does NOT simulate** action effects

**What it validates:**
- ✅ Export is numerically lossless
- ✅ Fast (minutes vs hours)
- ❌ Does NOT show real accuracy (no closed-loop)
- ❌ Does NOT account for error accumulation

**Expected output:**
```
Max absolute difference: 0.000342
Mean absolute difference: 0.000019
P99 difference: 0.000123
✅ Numerically equivalent (within bfloat16 precision)
```

## Key Differences

| Aspect | Numerical | Closed-Loop |
|--------|-----------|-------------|
| **Input Source** | Dataset (HDF5) | Simulator (MuJoCo) |
| **Predictions** | Single-step | Multi-step (up to 520) |
| **Error Accumulation** | ❌ No | ✅ Yes |
| **Metric** | Numerical diff | Success rate |
| **Speed** | Fast (~2 min) | Slow (~30 min) |
| **Use Case** | Export validation | Production accuracy |
| **When to Use** | Debugging exports | Before deployment |

## Supported Models

### PhysicalAI Policies (Fully Supported)

All export to OpenVINO out-of-the-box:

- **ACT** - Action Chunking Transformer
- **Pi0** - Physical Intelligence 0
- **Pi0.5** - Physical Intelligence 0.5 (recommended for LIBERO)
- **Groot** - Hierarchical transformer
- **SmolVLA** - Small Vision-Language-Action

### LeRobot Policies (Via Wrapper)

- **Diffusion** - Denoising Diffusion Policy
- **PI0/PI05** - From HuggingFace Hub
- **VQ-BeT** - Best-effort support

### Pretrained Models

Available on HuggingFace Hub (most are gated; accept the license on each model
page while logged in):
- `lerobot/pi05_libero_finetuned_v044` (recommended, gated)
- `lerobot/act_pusht`
- `lerobot/groot_libero_finetuned_v044`

### Datasets

Used for `numerical_comparison.py --dataset`:
- `lerobot/libero_10_image` (recommended, matches the Pi0.5 LIBERO checkpoint)
- `lerobot/pusht` (for ACT/Diffusion)

## Installation

### Prerequisites

1. **HuggingFace Authentication** (required for downloading gated models and datasets):
   ```bash
   # Get token from https://huggingface.co/settings/tokens
   export HF_TOKEN="your_token_here"

   # Or login with CLI
   huggingface-cli login
   ```

   Both scripts fail fast with **exit code 2** if no HuggingFace token is found.
   Gated models such as `lerobot/pi05_libero_finetuned_v044` and
   `google/paligemma-3b-pt-224` require accepting their license on the Hub
   while logged in with the same account.

2. **System Dependencies** (for LIBERO closed-loop benchmarks):
   ```bash
   sudo apt-get install cmake libclang-dev libegl1-mesa-dev \
       libgl1-mesa-dev libgles2-mesa-dev libglew-dev libglfw3-dev \
       libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev
   ```

### For Development (From Source)

If you're working with the Physical AI Studio repository:

```bash
# 1. Clone repository (if not already done)
git clone https://github.com/open-edge-platform/physical-ai-studio.git
cd physical-ai-studio/library

# 2. Create virtual environment
uv venv
source .venv/bin/activate

# 3. Install library with all dependencies
uv sync --all-extras

# 4. Install LIBERO (for closed-loop benchmarks)
# See troubleshooting section if build fails
uv pip install hf-libero

# 5. Navigate to scripts and verify
cd scripts/backend_accuracy_comparison
make check
```

### For Production Use

If `physicalai-train` is published on PyPI:

```bash
# Minimal (Numerical Comparison Only)
uv pip install "physicalai-train[export]"

# Full (Closed-Loop Benchmark)
uv pip install "physicalai-train[export]"
uv pip install hf-libero  # See troubleshooting for build issues
```

**Note:** The `[export]` extra is currently not defined in the package. For development, use the source installation method above.

## Usage Examples

### Example 1: Quick Test with Pretrained Model

```bash
# Download pretrained model
python << EOF
from physicalai.policies.lerobot import PI05
policy = PI05.from_pretrained('lerobot/pi05_libero_finetuned_v044')
policy.save('test_model.ckpt')
EOF

# Run both benchmarks
python numerical_comparison.py \
    --checkpoint test_model.ckpt \
    --policy-class physicalai.policies.pi05.Pi05 \
    --dataset lerobot/libero_10_image \
    --num-samples 50

python closed_loop_benchmark.py \
    --checkpoint test_model.ckpt \
    --policy-class physicalai.policies.pi05.Pi05 \
    --task-suite libero_10 \
    --task-ids 0 1 \
    --num-episodes 10
```

### Example 2: Your Own Trained Model

```bash
# After training with physicalai
CKPT=experiments/lightning_logs/version_0/checkpoints/last.ckpt

python closed_loop_benchmark.py \
    --checkpoint $CKPT \
    --policy-class physicalai.policies.ACT \
    --task-suite libero_10 \
    --num-episodes 20
```

### Example 3: Batch Compare Multiple Models

```bash
python batch_comparison.py \
    --models \
        "act:checkpoints/act.ckpt:physicalai.policies.ACT" \
        "pi05:checkpoints/pi05.ckpt:physicalai.policies.Pi05" \
        "groot:checkpoints/groot.ckpt:physicalai.policies.Groot" \
    --task-suite libero_10 \
    --task-ids 0 1 2 \
    --num-episodes 10
```

See `examples.sh` for more.

## Interpreting Results

### When Results Match (Success!)

**Numerical Comparison:**
```json
{"analysis": {"max_diff": 0.0003, "conclusion": "✅ Numerically EQUIVALENT"}}
```
→ Export is lossless ✅

**Closed-Loop Benchmark:**
```json
{"comparison": {"overall_success_rate_delta": 0.5, "conclusion": "✅ Backends are EQUIVALENT"}}
```
→ OpenVINO has same accuracy as PyTorch ✅

### When Success Rates Differ (Investigation Needed)

**Delta < 5%:** Likely measurement noise
- **Action:** Increase `--num-episodes` to 50+
- **Why:** With 20 episodes, 1 episode = 5% difference

**Delta > 5% but numerical_diff < 0.001:** Measurement noise
- **Action:** Run with more episodes and same seed
- **Why:** Simulator has inherent randomness

**Delta > 5% AND numerical_diff > 0.01:** Export problem
- **Action:** Check export implementation
- **Why:** Numerical differences affecting rollouts

## Output Files

Results saved to `results/` directory:

```
results/
├── closed_loop_results.json      # Full benchmark results
│   ├── pytorch: {success_rate, per_task_results, ...}
│   ├── openvino: {success_rate, per_task_results, ...}
│   └── comparison: {delta, is_equivalent, conclusion}
│
└── numerical_results.json        # Numerical comparison
    ├── comparison: {max_diff, mean_diff, percentiles}
    └── analysis: {level, conclusion, is_equivalent}
```

Exported models saved to `exports/`:

```
exports/
├── torch/
│   ├── model.pt
│   ├── metadata.json
│   └── manifest.json   # cache sentinel
└── openvino/
    ├── model.xml
    ├── model.bin
    ├── metadata.json
    └── manifest.json   # cache sentinel
```

### Export Caching

Exports are cached between runs. If `exports/{torch,openvino}/manifest.json`
exists, the scripts skip re-exporting and reuse the artifacts on disk. Pass
`--force-export` to either script to rebuild the exports from the checkpoint.

### Quantile Stats Caching

Pi0/Pi0.5 require q01/q99 quantile statistics that are computed from the
dataset on first use and persisted to `<dataset_root>/meta/stats.json`. The
first run on a new dataset is slow (a few minutes for LIBERO-sized datasets);
subsequent runs reuse the cached stats and start almost instantly.

## Troubleshooting

### "LIBERO not available" or CMake build errors

**Problem:** Installation of `hf-libero` fails with CMake error:

```
CMake Error: Compatibility with CMake < 3.5 has been removed
```

**Solution:** The `egl-probe` package needs patching. Follow these steps:

```bash
# 1. Install system dependencies
sudo apt-get install -y cmake libclang-dev libegl1-mesa-dev \
  libgl1-mesa-dev libgles2-mesa-dev libglew-dev libglfw3-dev \
  libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev

# 2. Download and patch egl-probe
cd /tmp
pip download --no-binary :all: egl-probe
tar xzf egl_probe-*.tar.gz
cd egl_probe-*/
sed -i 's/cmake_minimum_required(VERSION 2\.8\.12)/cmake_minimum_required(VERSION 3.5)/g' \
  egl_probe/CMakeLists.txt

# 3. Install patched version
uv pip install .  # or: pip install .
cd / && rm -rf /tmp/egl_probe*

# 4. Now install hf-libero
uv pip install hf-libero
```

This is a known issue with modern CMake (3.22+) refusing to build projects requesting ancient versions.

See [Installation Guide](../../docs/getting-started/installation.md#libero-installation-fails-cmake-error) for details.

### "LIBERO not available" (simple case)

If you just need to install hf-libero without build issues:

```bash
uv pip install hf-libero
```

### "MuJoCo initialization error"

```bash
# Linux
sudo apt-get install libglew-dev libglfw3-dev

# macOS
brew install glew glfw
```

### "Policy does not support openvino export"

Check supported backends:

```python
from physicalai.policies import ACT
policy = ACT()
print(policy.get_supported_export_backends())
```

### Large success rate differences (>10%)

1. Check numerical comparison - is max_diff < 0.001?
2. Increase episodes: `--num-episodes 50`
3. Verify same seed is used (already implemented)
4. Check if model uses stochastic components

## Project Context

This project addresses a critical question for deploying robot learning models:

**"Does exporting to OpenVINO change the robot's behavior?"**

### Why This Matters

- **Export optimization** (graph fusion, quantization) might introduce errors
- **Small numerical differences** can compound over 500+ timesteps
- **Success/failure** of real robot tasks depends on cumulative accuracy
- **Dataset validation** alone doesn't reveal deployment issues

### What We Measure

1. **Numerical equivalence** - Are single predictions identical?
2. **Behavioral equivalence** - Does the robot complete the same tasks?

Both must pass for production deployment confidence.

## Citation

If you use this benchmark in research:

```bibtex
@software{physical-ai-studio-backend-comparison,
  title = {Backend Accuracy Comparison for Robot Learning Policies},
  author = {Intel Corporation},
  year = {2026},
  url = {https://github.com/open-edge-platform/physical-ai-studio}
}
```

## Related Documentation

- **Demo Notebook:** [`backend_accuracy_comparison_demo.ipynb`](../../notebooks/benchmark/backend_accuracy_comparison_demo.ipynb)
- **Architecture Guide:** See notebook for diagrams and flow charts
- **LIBERO Benchmark:** [`libero.ipynb`](../../notebooks/benchmark/libero.ipynb)
- **Export Guide:** [`docs/how-to/export/export_inference.md`](../../docs/how-to/export/export_inference.md)

## License

Apache 2.0 - See [LICENSE](../../../LICENSE)
