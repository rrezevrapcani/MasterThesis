#!/bin/bash
#SBATCH --job-name=steering_vectors
#SBATCH --output=logs/steering_vectors_%j.out
#SBATCH --error=logs/steering_vectors_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=A40short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

echo "========================================"
echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Current directory: $(pwd)"
echo "SLURM job ID: ${SLURM_JOB_ID:-N/A}"
echo "========================================"

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_DIR"
mkdir -p logs outputs/analysis/steering_vectors

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate env

python scripts/embeddings/apply_steering_vectors.py \
  --config configs/analysis/steering_vectors.yaml
