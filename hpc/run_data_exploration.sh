#!/bin/bash
#SBATCH --job-name=data_exploration
#SBATCH --output=logs/data_exploration_%j.out
#SBATCH --error=logs/data_exploration_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_DIR"
mkdir -p logs outputs/analysis/exploration
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-env}"
python scripts/data/data_exploration.py --config configs/analysis/data_exploration.yaml
