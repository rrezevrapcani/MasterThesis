#!/bin/bash
#SBATCH --job-name=pca_embeddings
#SBATCH --output=logs/pca_%j.out
#SBATCH --error=logs/pca_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=A40short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_DIR"
mkdir -p logs outputs/analysis/pca

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-env}"

python scripts/embeddings/run_pca.py \
  --config configs/analysis/pca.yaml
