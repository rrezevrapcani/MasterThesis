#!/bin/bash
#SBATCH --job-name=synthetic_twins
#SBATCH --output=logs/synthetic_twins_%j.out
#SBATCH --error=logs/synthetic_twins_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=A40medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

set -euo pipefail

echo "========================================"
echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Current directory: $(pwd)"
echo "SLURM job ID: $SLURM_JOB_ID"
echo "========================================"

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_DIR"
mkdir -p logs outputs/generation/synthetic_twins_train

# --- Resolve GPU UUID to numeric index (single-GPU) ---
if [[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* ]]; then
    RESOLVED_IDX=$(nvidia-smi -L | grep "$CUDA_VISIBLE_DEVICES" | grep -oP '^GPU \K[0-9]+')
    if [[ -z "$RESOLVED_IDX" ]]; then
        echo "ERROR: could not resolve $CUDA_VISIBLE_DEVICES to a GPU index" >&2
        exit 1
    fi
    export CUDA_VISIBLE_DEVICES=$RESOLVED_IDX
    echo "Resolved CUDA_VISIBLE_DEVICES to index $CUDA_VISIBLE_DEVICES"
fi

# Avoid FlashInfer sampler JIT failures on clusters without a discoverable CUDA toolkit/nvcc.
# vLLM will fall back to its native sampler.
export VLLM_USE_FLASHINFER_SAMPLER=0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate env

echo "Starting synthetic twin generation..."
python scripts/generation/generate_synthetic_twins.py \
  --config configs/generation/synthetic_twins.yaml

echo "========================================"
echo "Job finished on $(date)"
echo "========================================"