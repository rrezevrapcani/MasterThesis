#!/bin/bash
#SBATCH --job-name=activation_vec
#SBATCH --output=logs/activation_steering_vector_%j.out
#SBATCH --error=logs/activation_steering_vector_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=A40medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

echo "========================================"
echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Current directory: $(pwd)"
echo "SLURM job ID: ${SLURM_JOB_ID:-N/A}"
echo "========================================"

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_DIR"
mkdir -p logs outputs/analysis/activation_steering_vectors

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-env}"

python scripts/embeddings/compute_activation_steering_vector.py \
  --config "${BINARY_CONFIG:-configs/analysis/activation_steering_vectors_binary.yaml}"

python scripts/embeddings/compute_activation_steering_vector.py \
  --config "${MULTICLASS_CONFIG:-configs/analysis/activation_steering_vectors_multiclass.yaml}"
