#!/bin/bash
#SBATCH --job-name=easysteer_vec
#SBATCH --output=logs/easysteer_generation_vector_%j.out
#SBATCH --error=logs/easysteer_generation_vector_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=A40medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

set -euo pipefail

echo "========================================"
echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Current directory: $(pwd)"
echo "SLURM job ID: ${SLURM_JOB_ID:-N/A}"
echo "========================================"

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_DIR"
mkdir -p logs outputs/generation/easysteer_vectors

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

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate env

python scripts/generation/compute_qwen_generation_steering_vector.py \
  --config "${CONFIG:-configs/generation/easysteer_qwen_generation_vector.yaml}"
