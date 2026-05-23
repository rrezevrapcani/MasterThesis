#!/bin/bash
#SBATCH --job-name=two-stage-pipeline
#SBATCH --output=logs/two_stage_pipeline_%j.out
#SBATCH --error=logs/two_stage_pipeline_%j.err
#SBATCH --time=24:00:00
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
mkdir -p results

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate env

echo "Starting two-stage pipeline..."
python src/training/train_two_stage_pipeline.py

echo "========================================"
echo "Job finished on $(date)"
echo "========================================"
