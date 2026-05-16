#!/bin/bash
#SBATCH --job-name=pubmedbert_biored
#SBATCH --output=logs/pubmedbert_biored_%j.out
#SBATCH --error=logs/pubmedbert_biored_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=A40medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -e

echo "========================================"
echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Current directory: $(pwd)"
echo "SLURM job ID: $SLURM_JOB_ID"
echo "========================================"

cd "$SLURM_SUBMIT_DIR"

mkdir -p logs
mkdir -p results/pubmedbert_baseline

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate env

echo "Starting training..."
python -m src.training.train_pubmedbert \
  --config configs/experiments/pubmedbert_baseline.yaml

echo "========================================"
echo "Job finished on $(date)"
echo "========================================"