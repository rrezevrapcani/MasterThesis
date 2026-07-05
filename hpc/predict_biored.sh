#!/bin/bash
#SBATCH --job-name=my_pipeline_biored_pred
#SBATCH --output=my_pipeline_biored_pred_%j.log
#SBATCH --error=my_pipeline_biored_pred_%j.err
#SBATCH --gres=gpu:1
#SBATCH --gpus=1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --partition=A40medium
#SBATCH --exclude=node-01,node-02,node-05

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate env

cd /home/s93rvrap/projects/MasterThesis

python scripts/prediction/predict_biored_logits.py \
  --test_csv datasets/biored/processed/test_for_my_pipeline.csv \
  --binary_model results/pubmedbert_hierarchical_biored/stage1_binary_classifier/final_model \
  --multiclass_model results/pubmedbert_hierarchical_biored/stage2_multiclass_classifier/final_model \
  --output_logits outputs/biored/predictions.tsv \
  --device cuda