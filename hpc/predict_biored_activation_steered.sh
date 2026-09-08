#!/bin/bash
#SBATCH --job-name=biored_steered_pred
#SBATCH --output=logs/biored_steered_pred_%j.out
#SBATCH --error=logs/biored_steered_pred_%j.err
#SBATCH --gres=gpu:1
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --partition=A40medium

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_DIR"
mkdir -p logs outputs/biored

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-env}"

STEERING_ALPHA="${STEERING_ALPHA:-1.0}"
STEERING_LAYER="${STEERING_LAYER:-11}"
STEERING_POSITION="${STEERING_POSITION:-cls}"

BINARY_VECTOR="${BINARY_VECTOR:-outputs/analysis/activation_steering_vectors/qwen4b_initial_length_restricted_binary/activation_steering_vector.npy}"
MULTICLASS_VECTOR="${MULTICLASS_VECTOR:-outputs/analysis/activation_steering_vectors/qwen4b_initial_length_restricted_multiclass/activation_steering_vector.npy}"

python scripts/prediction/predict_biored_logits.py \
  --test_csv datasets/biored/processed/test_for_my_pipeline.csv \
  --binary_model results/pubmedbert_hierarchical_biored/stage1_binary_classifier/final_model \
  --multiclass_model results/pubmedbert_hierarchical_biored/stage2_multiclass_classifier/final_model \
  --output_logits "outputs/biored/predictions_activation_steered_alpha_${STEERING_ALPHA}.tsv" \
  --device cuda \
  --steering_layer "$STEERING_LAYER" \
  --steering_alpha "$STEERING_ALPHA" \
  --steering_position "$STEERING_POSITION" \
  --binary_steering_vector "$BINARY_VECTOR" \
  --multiclass_steering_vector "$MULTICLASS_VECTOR"
