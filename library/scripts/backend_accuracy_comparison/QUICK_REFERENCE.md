# Backend Comparison Quick Reference

## 📁 New Location

All backend comparison scripts have been moved to:
```
library/scripts/backend_accuracy_comparison/
```

## 📊 Project Structure

```
library/
├── scripts/
│   └── backend_accuracy_comparison/          ← Scripts relocated here
│       ├── README.md                # Full documentation (English)
│       ├── closed_loop_benchmark.py # Main script (simulator-based)
│       ├── numerical_comparison.py  # Quick script (dataset-based)
│       ├── batch_comparison.py      # Helper for multiple models
│       ├── check_setup.py           # Environment verification
│       ├── requirements.txt         # Dependencies
│       ├── Makefile                 # Convenience commands
│       ├── examples.sh              # Usage examples
│       ├── .gitignore              # Ignore results/exports
│       └── results/                 # Output directory
│
└── notebooks/
    └── benchmark/
        ├── backend_accuracy_comparison_demo.ipynb  ← New demo notebook
        ├── lerobot_benchmark_comparison.ipynb
        └── libero.ipynb
```

## ⚙️ Prerequisites

**HuggingFace Authentication** (required for downloading gated models and datasets):

```bash
# Get token from https://huggingface.co/settings/tokens
export HF_TOKEN="your_token_here"

# Or login with CLI
huggingface-cli login
```

Both scripts fail fast with **exit code 2** if no token is found. Gated
models (e.g. `lerobot/pi05_libero_finetuned_v044`, `google/paligemma-3b-pt-224`)
require accepting their license on the Hub.

## 💾 Caching

- **Exports**: artifacts under `exports/{torch,openvino}/` are reused when
  `manifest.json` is present. Pass `--force-export` to rebuild.
- **Quantile stats**: Pi0/Pi0.5 require q01/q99 stats. They are computed once
  per dataset (slow first run) and persisted to `<dataset_root>/meta/stats.json`;
  subsequent runs start almost instantly.

## 🚀 Quick Start

### Option 1: Use the Notebook (Recommended for Learning)

```bash
jupyter notebook library/notebooks/benchmark/backend_accuracy_comparison_demo.ipynb
```

The notebook includes:
- Interactive visualizations of error accumulation
- Step-by-step comparison examples
- Result interpretation guides
- Decision matrices

### Option 2: Use Scripts Directly (Production)

```bash
cd library/scripts/backend_accuracy_comparison

# Verify environment (assumes library installed from root)
make check

# Quick test (minutes for small models, longer for VLA-scale policies)
make test-numerical

# Full test (plan as a multi-hour run; launch in background or overnight)
make test-closed-loop
```

**For fresh setup:** Install the library first from `library/` root with `uv sync --all-extras`

## 📖 Documentation

- **Main README:** [library/scripts/backend_accuracy_comparison/README.md](library/scripts/backend_accuracy_comparison/README.md)
- **Demo Notebook:** [library/notebooks/benchmark/backend_accuracy_comparison_demo.ipynb](library/notebooks/benchmark/backend_accuracy_comparison_demo.ipynb)

## 🎯 Use Cases

| Scenario | Command | Time |
|----------|---------|------|
| Quick export validation | `python numerical_comparison.py ...` | Minutes (longer for VLA policies) |
| Production readiness | `python closed_loop_benchmark.py ...` | Hours; plan as background/overnight run |
| Faster closed-loop on Intel GPU | `python closed_loop_benchmark.py --ov-device GPU ...` | Typically much faster than CPU |
| Compare multiple models | `python batch_comparison.py ...` | Variable |
| Environment check | `python check_setup.py` | <1 min |

## ✨ Key Features

1. **Two Complementary Approaches:**
   - Numerical: Fast sanity check on dataset samples
   - Closed-loop: Real accuracy with MuJoCo simulator

2. **Supported Policies:**
   - ACT, Pi0, Pi0.5, Groot, SmolVLA, Diffusion

3. **Clear Success Criteria:**
   - Numerical: max_diff < 0.001
   - Closed-loop: success_rate_delta < 5%

## 🔗 Related Resources

- LIBERO Benchmark: `library/notebooks/benchmark/libero.ipynb`
- Export Guide: `library/docs/how-to/export/export_inference.md`
- PhysicalAI Training: `library/README.md`
