#!/bin/bash
#SBATCH --job-name=synthetic_generation
#SBATCH --output=logs/synthetic_generation_%j.out
#SBATCH --error=logs/synthetic_generation_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=A40medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_DIR"
mkdir -p logs outputs/generation/synthetic

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-env}"

python scripts/generation/generate_synthetic_data.py \
  --config configs/generation/synthetic_generation.yaml
