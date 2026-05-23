#!/usr/bin/env python
"""
Two-stage training pipeline for biomedical relation extraction.

Stage 1: Binary classification (Relation vs No-Relation)
Stage 2: Multi-class classification of relation types (only on predicted relations)
"""

import argparse
import subprocess
import sys
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import torch

from sklearn.metrics import accuracy_score, f1_score, classification_report

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


def setup_logging(output_dir):
    log_dir = Path(output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "pipeline.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def prepare_data():
    """Prepare two-stage datasets from sorted data."""
    logging.info("=" * 70)
    logging.info("STAGE 0: DATA PREPARATION")
    logging.info("=" * 70)

    result = subprocess.run(
        ["python", "src/preprocessing/prepare_two_stage_dataset.py"],
        cwd=".",
    )

    if result.returncode != 0:
        logging.error("Data preparation failed!")
        sys.exit(1)

    logging.info("Data preparation completed successfully.")


def train_stage(stage_num, config_path):
    """Train a single stage."""
    logging.info("=" * 70)
    logging.info(f"STAGE {stage_num}: TRAINING")
    logging.info("=" * 70)

    result = subprocess.run(
        ["python", "src/training/train_pubmedbert.py", "--config", config_path],
        cwd=".",
    )

    if result.returncode != 0:
        logging.error(f"Stage {stage_num} training failed!")
        sys.exit(1)

    logging.info(f"Stage {stage_num} training completed successfully.")


def evaluate_two_stage_pipeline():
    """
    Evaluate the two-stage pipeline end-to-end.
    Uses predictions from stage 1 to filter stage 2 inputs.
    """
    logging.info("=" * 70)
    logging.info("PIPELINE EVALUATION: TWO-STAGE INFERENCE")
    logging.info("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load models
    stage1_model_dir = Path("results/stage1_binary_classification/final_model")
    stage2_model_dir = Path("results/stage2_relation_classification/final_model")

    tokenizer1 = AutoTokenizer.from_pretrained(stage1_model_dir)
    model1 = AutoModelForSequenceClassification.from_pretrained(stage1_model_dir).to(device)
    model1.eval()

    tokenizer2 = AutoTokenizer.from_pretrained(stage2_model_dir)
    model2 = AutoModelForSequenceClassification.from_pretrained(stage2_model_dir).to(device)
    model2.eval()

    # Load test data
    test_df = pd.read_csv("data/processed/two_stage/stage1_binary/test.csv")

    logging.info(f"Total test samples: {len(test_df)}")

    # Stage 1: Binary classification
    logging.info("\nRunning Stage 1 (Binary Classification)...")
    predictions_stage1 = []

    with torch.no_grad():
        for text in test_df["passage_text"]:
            inputs = tokenizer1(
                text,
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt",
            ).to(device)
            outputs = model1(**inputs)
            pred = np.argmax(outputs.logits.cpu().numpy(), axis=-1)[0]
            predictions_stage1.append(pred)

    test_df["stage1_pred"] = predictions_stage1

    # Load label mappings
    with open("results/stage1_binary_classification/id2label.json") as f:
        id2label1 = json.load(f)

    test_df["stage1_pred_label"] = test_df["stage1_pred"].map(
        {int(k): v for k, v in id2label1.items()}
    )

    # Calculate Stage 1 metrics
    true_labels_stage1 = test_df["relation_type"].tolist()
    pred_labels_stage1 = test_df["stage1_pred_label"].tolist()

    logging.info(f"\nStage 1 Binary Classification Results:")
    logging.info(f"Accuracy: {accuracy_score(true_labels_stage1, pred_labels_stage1):.4f}")
    logging.info(
        f"Macro F1: {f1_score(true_labels_stage1, pred_labels_stage1, average='macro', zero_division=0):.4f}"
    )
    logging.info(f"\n{classification_report(true_labels_stage1, pred_labels_stage1)}")

    # Stage 2: Only on predicted relations
    logging.info("\nRunning Stage 2 (Relation Classification) on predicted relations...")

    relation_mask = test_df["stage1_pred_label"] == "Relation"
    relation_samples = test_df[relation_mask].copy()
    no_relation_samples = test_df[~relation_mask].copy()

    logging.info(f"Predicted as Relation: {relation_mask.sum()}")
    logging.info(f"Predicted as No-Relation: {(~relation_mask).sum()}")

    # Initialize stage 2 predictions (all "No_Relation" by default)
    stage2_predictions = ["No_Relation"] * len(test_df)

    # Load label mapping for stage 2
    with open("results/stage2_relation_classification/id2label.json") as f:
        id2label2 = json.load(f)

    # Run stage 2 inference only on relation samples
    if len(relation_samples) > 0:
        with torch.no_grad():
            for idx, text in zip(relation_samples.index, relation_samples["passage_text"]):
                inputs = tokenizer2(
                    text,
                    truncation=True,
                    padding="max_length",
                    max_length=512,
                    return_tensors="pt",
                ).to(device)
                outputs = model2(**inputs)
                pred_id = np.argmax(outputs.logits.cpu().numpy(), axis=-1)[0]
                pred_label = id2label2[str(pred_id)]
                stage2_predictions[idx] = pred_label

    test_df["stage2_pred"] = stage2_predictions

    # Calculate Stage 2 metrics (only on relation samples)
    if len(relation_samples) > 0:
        true_labels_stage2 = relation_samples["relation_type"].tolist()
        pred_labels_stage2 = [
            stage2_predictions[idx] for idx in relation_samples.index
        ]

        logging.info(f"\nStage 2 Relation Classification Results (on predicted relations):")
        logging.info(
            f"Accuracy: {accuracy_score(true_labels_stage2, pred_labels_stage2):.4f}"
        )
        logging.info(
            f"Macro F1: {f1_score(true_labels_stage2, pred_labels_stage2, average='macro', zero_division=0):.4f}"
        )
        logging.info(f"\n{classification_report(true_labels_stage2, pred_labels_stage2)}")

    # Save predictions
    output_dir = Path("results/two_stage_pipeline")
    output_dir.mkdir(parents=True, exist_ok=True)

    test_df.to_csv(output_dir / "predictions.csv", index=False)
    logging.info(f"\nPredictions saved to: {output_dir / 'predictions.csv'}")

    return test_df


def main():
    parser = argparse.ArgumentParser(description="Two-stage relation extraction pipeline")
    parser.add_argument(
        "--skip-data-prep",
        action="store_true",
        help="Skip data preparation step",
    )
    parser.add_argument(
        "--skip-stage1",
        action="store_true",
        help="Skip stage 1 training",
    )
    parser.add_argument(
        "--skip-stage2",
        action="store_true",
        help="Skip stage 2 training",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Only run evaluation (skip training)",
    )

    args = parser.parse_args()

    setup_logging("results/pipeline_logs")

    logging.info("=" * 70)
    logging.info("TWO-STAGE BIOMEDICAL RELATION EXTRACTION PIPELINE")
    logging.info("=" * 70)

    # Step 0: Data preparation
    if not args.skip_data_prep and not args.evaluate_only:
        prepare_data()

    # Step 1: Stage 1 training
    if not args.skip_stage1 and not args.evaluate_only:
        train_stage(1, "configs/experiments/bipred_stage1_binary.yaml")

    # Step 2: Stage 2 training
    if not args.skip_stage2 and not args.evaluate_only:
        train_stage(2, "configs/experiments/bipred_stage2_relations.yaml")

    # Step 3: Evaluation
    evaluate_two_stage_pipeline()

    logging.info("\n" + "=" * 70)
    logging.info("PIPELINE COMPLETED SUCCESSFULLY!")
    logging.info("=" * 70)


if __name__ == "__main__":
    main()
