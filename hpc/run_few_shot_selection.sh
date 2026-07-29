#!/bin/bash
#SBATCH --job-name=few_shot_selection
#SBATCH --output=logs/few_shot_selection_%j.out
#SBATCH --error=logs/few_shot_selection_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_DIR"
mkdir -p logs outputs/generation/few_shot

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-env}"

python scripts/data/select_few_shot_examples.py \
  --config configs/generation/few_shot_selection.yaml
