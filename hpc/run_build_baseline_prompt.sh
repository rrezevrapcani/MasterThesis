#!/bin/bash
#SBATCH --job-name=baseline_prompt
#SBATCH --output=logs/baseline_prompt_%j.out
#SBATCH --error=logs/baseline_prompt_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_DIR"
mkdir -p logs outputs/generation/prompts

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-env}"

python scripts/generation/build_baseline_prompt.py \
  --config configs/generation/synthetic_prompt.yaml
