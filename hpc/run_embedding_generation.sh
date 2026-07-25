#!/bin/bash
#SBATCH --job-name=embedding_generation
#SBATCH --output=logs/embedding_generation_%j.out
#SBATCH --error=logs/embedding_generation_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=A40medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

echo "========================================"
echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Current directory: $(pwd)"
echo "SLURM job ID: $SLURM_JOB_ID"
echo "========================================"

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_DIR"
mkdir -p logs outputs/analysis/embeddings

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-env}"

python scripts/embeddings/generate_embeddings.py \
  --config configs/analysis/embedding_generation.yaml
