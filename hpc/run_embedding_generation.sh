#!/bin/bash
#SBATCH --job-name=embedding_generation
#SBATCH --output=logs/embedding_generation_%j.out
#SBATCH --error=logs/embedding_generation_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=A40medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_DIR"
mkdir -p logs outputs/analysis/embeddings

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-env}"

python scripts/embeddings/generate_embeddings.py \
  --config configs/analysis/embedding_generation.yaml
