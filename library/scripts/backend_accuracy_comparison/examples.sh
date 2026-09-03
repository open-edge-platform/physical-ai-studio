#!/bin/bash
# Usage examples for the backend accuracy comparison scripts.

set -e

echo "============================================="
echo "Backend Accuracy Comparison - Examples"
echo "============================================="

# Prerequisite: HuggingFace authentication.
# Either run `huggingface-cli login` once, or `export HF_TOKEN=...`.
# Both scripts fail fast (exit code 2) with no token.

# ============================================
# 1. CLOSED-LOOP BENCHMARK (recommended)
# ============================================
echo ""
echo "1. CLOSED-LOOP BENCHMARK"
echo "   - Full simulation in LiberoGym + MuJoCo"
echo "   - Captures error accumulation"
echo "   - Reports success rate"
echo ""

# Example 1a: Pi0.5 on libero_10 (first 3 tasks)
echo "Example 1a: Pi0.5 on LIBERO-10 (tasks 0-2)"
python closed_loop_benchmark.py \
    --checkpoint checkpoints/pi05_libero.ckpt \
    --policy-class physicalai.policies.pi05.Pi05 \
    --task-suite libero_10 \
    --task-ids 0 1 2 \
    --num-episodes 20 \
    --seed 42 \
    --output results/pi05_closed_loop.json

# Example 1b: ACT on libero_spatial (all tasks)
echo ""
echo "Example 1b: ACT on libero_spatial (all tasks)"
python closed_loop_benchmark.py \
    --checkpoint checkpoints/act_spatial.ckpt \
    --policy-class physicalai.policies.ACT \
    --task-suite libero_spatial \
    --num-episodes 50 \
    --seed 42 \
    --output results/act_closed_loop.json

# Example 1c: LeRobot Pi0.5 from HuggingFace
echo ""
echo "Example 1c: LeRobot Pi0.5 pretrained"
# First download the model:
python -c "
from physicalai.policies.lerobot import PI05
policy = PI05.from_pretrained('lerobot/pi05_libero_finetuned_v044')
policy.save('checkpoints/pi05_hf.ckpt')
"

python closed_loop_benchmark.py \
    --checkpoint checkpoints/pi05_hf.ckpt \
    --policy-class physicalai.policies.lerobot.PI05 \
    --task-suite libero_10 \
    --task-ids 0 \
    --num-episodes 10 \
    --seed 42 \
    --output results/pi05_hf_closed_loop.json


# ============================================
# 2. NUMERICAL COMPARISON (quick sanity check)
# ============================================
echo ""
echo ""
echo "2. NUMERICAL COMPARISON"
echo "   - Per-sample predictions on a dataset"
echo "   - No simulator"
echo "   - Fast export sanity check"
echo ""

# Example 2a: ACT on PushT
echo "Example 2a: ACT on PushT dataset"
python numerical_comparison.py \
    --checkpoint checkpoints/act_pusht.ckpt \
    --policy-class physicalai.policies.ACT \
    --dataset lerobot/pusht \
    --num-samples 100 \
    --output results/act_numerical.json

# Example 2b: Pi0.5 on LIBERO-10 image dataset
echo ""
echo "Example 2b: Pi0.5 on LIBERO-10 image dataset"
python numerical_comparison.py \
    --checkpoint checkpoints/pi05_libero.ckpt \
    --policy-class physicalai.policies.pi05.Pi05 \
    --dataset lerobot/libero_10_image \
    --num-samples 200 \
    --output results/pi05_numerical.json


# ============================================
# 3. BATCH COMPARISON (multiple models)
# ============================================
echo ""
echo ""
echo "3. BATCH COMPARISON (multiple models)"
echo ""

# Models to evaluate.
MODELS=(
    "checkpoints/act_libero.ckpt:physicalai.policies.ACT:act"
    "checkpoints/pi05_libero.ckpt:physicalai.policies.pi05.Pi05:pi05"
    "checkpoints/groot_libero.ckpt:physicalai.policies.Groot:groot"
)

for model_spec in "${MODELS[@]}"; do
    IFS=':' read -r checkpoint policy_class name <<< "$model_spec"

    echo "Testing $name..."

    # Closed-loop
    python closed_loop_benchmark.py \
        --checkpoint "$checkpoint" \
        --policy-class "$policy_class" \
        --task-suite libero_10 \
        --task-ids 0 1 \
        --num-episodes 10 \
        --output "results/${name}_closed_loop.json"

    # Numerical
    python numerical_comparison.py \
        --checkpoint "$checkpoint" \
        --policy-class "$policy_class" \
        --dataset lerobot/libero_10_image \
        --num-samples 50 \
        --output "results/${name}_numerical.json"

    echo "✓ $name complete"
    echo ""
done


# ============================================
# 4. RESULTS ANALYSIS
# ============================================
echo ""
echo "4. RESULTS ANALYSIS"
echo ""

# A small helper that aggregates the JSON result files.
cat > analyze_results.py << 'EOF'
import json
import sys
from pathlib import Path

results_dir = Path("results")
closed_loop_files = list(results_dir.glob("*closed_loop.json"))
numerical_files = list(results_dir.glob("*numerical.json"))

print("\n" + "="*80)
print("SUMMARY OF ALL RESULTS")
print("="*80)

if closed_loop_files:
    print("\nClosed-Loop Benchmarks:")
    print(f"{'Model':<20} {'PyTorch SR':>12} {'OpenVINO SR':>12} {'Delta':>10}")
    print("-"*60)

    for f in closed_loop_files:
        with open(f) as fp:
            data = json.load(fp)

        model = f.stem.replace("_closed_loop", "")
        pt_sr = data["pytorch"]["overall_success_rate"]
        ov_sr = data["openvino"]["overall_success_rate"]
        delta = data["comparison"]["overall_success_rate_delta"]

        print(f"{model:<20} {pt_sr:>11.1f}% {ov_sr:>11.1f}% {delta:>9.1f}%")

if numerical_files:
    print("\nNumerical Comparisons:")
    print(f"{'Model':<20} {'Max Diff':>15} {'Mean Diff':>15} {'Status':>15}")
    print("-"*70)

    for f in numerical_files:
        with open(f) as fp:
            data = json.load(fp)

        model = f.stem.replace("_numerical", "")
        max_diff = data["comparison"]["overall_max_diff"]
        mean_diff = data["comparison"]["overall_mean_diff"]
        is_eq = data["analysis"]["is_equivalent"]
        status = "✓ OK" if is_eq else "⚠ CHECK"

        print(f"{model:<20} {max_diff:>15.8f} {mean_diff:>15.8f} {status:>15}")

print("\n" + "="*80)
EOF

python analyze_results.py


# ============================================
# 5. CLEANUP
# ============================================
echo ""
echo "Results saved under results/"
echo "Exported models under exports/"
