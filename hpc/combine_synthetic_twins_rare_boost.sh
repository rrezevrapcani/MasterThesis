#!/bin/bash
#SBATCH --job-name=combine_synth
#SBATCH --output=logs/combine_synth_%j.out
#SBATCH --error=logs/combine_synth_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=A40medium
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_DIR"
mkdir -p logs outputs/generation/synthetic_twins_train

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-env}"

python scripts/generation/combine_synthetic_datasets.py \
  --base outputs/generation/synthetic_twins_train/synthetic_twin_relations_clean.csv \
  --boost outputs/generation/synthetic_twins_rare_boost/synthetic_twin_relations_clean.csv \
  --output outputs/generation/synthetic_twins_train/synthetic_twin_relations_clean_rare_boosted.csv \
  --summary outputs/generation/synthetic_twins_train/synthetic_twin_relations_clean_rare_boosted_summary.json
