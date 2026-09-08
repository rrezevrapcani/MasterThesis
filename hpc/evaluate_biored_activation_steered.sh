#!/bin/bash
#SBATCH --job-name=biored_steered_eval
#SBATCH --output=logs/biored_steered_eval_%j.out
#SBATCH --error=logs/biored_steered_eval_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=A40short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_DIR"
mkdir -p logs outputs/biored

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-env}"

STEERING_ALPHA="${STEERING_ALPHA:-1.0}"
PRED_TSV="${PRED_TSV:-outputs/biored/predictions_activation_steered_alpha_${STEERING_ALPHA}.tsv}"
OUT_PUBTATOR="${OUT_PUBTATOR:-outputs/biored/biored_pred_activation_steered_alpha_${STEERING_ALPHA}.txt}"

python scripts/eval/run_biored_eval.py --exp_option 'to_pubtator' \
  --in_test_pubtator_file datasets/biored/BioRED/Test.PubTator \
  --in_test_tsv_file datasets/biored/processed/test.tsv \
  --in_pred_rel_tsv_file "$PRED_TSV" \
  --in_pred_novelty_tsv_file outputs/biored/dummy_novelty_test_results.tsv \
  --out_pred_pubtator_file "$OUT_PUBTATOR"

python scripts/eval/run_biored_eval.py --exp_option 'biored_eval' \
  --in_gold_pubtator_file datasets/biored/BioRED/Test.PubTator \
  --in_pred_pubtator_file "$OUT_PUBTATOR"
