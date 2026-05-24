import argparse
import json
import logging
from pathlib import Path

import yaml
import pandas as pd
import numpy as np

import torch
import torch.nn as nn

from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(output_dir) / "training.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def compute_class_weights(label_counts, no_relation_label="No_Relation"):
    """
    Compute class weights to handle class imbalance.
    Increases weight for minority classes and decreases weight for majority (No_Relation).
    """
    total_samples = sum(label_counts.values())
    weights = {}
    
    for label, count in label_counts.items():
        if label == no_relation_label:
            # Reduce weight for No_Relation label
            weights[label] = total_samples / (2.0 * count)
        else:
            # Increase weight for positive relation labels
            weights[label] = total_samples / (2.0 * count)
    
    return weights


def balance_negative_samples(df, label_column, no_relation_label="No_Relation", max_neg_scale=1.0):
    """
    Balance negative (No_Relation) samples with positive samples.
    Randomly samples negative instances to match the count of positive instances.
    """
    positive_df = df[df[label_column] != no_relation_label]
    negative_df = df[df[label_column] == no_relation_label]
    
    num_positive = len(positive_df)
    num_negative = len(negative_df)
    
    logging.info(f"Original distribution - Positive: {num_positive}, Negative: {num_negative}")
    
    # Calculate how many negative samples to keep
    target_neg_count = int(num_positive * max_neg_scale)
    
    if num_negative > target_neg_count:
        # Randomly sample negative instances
        negative_df = negative_df.sample(n=target_neg_count, random_state=42)
        logging.info(f"Balanced negative samples to: {target_neg_count}")
    
    balanced_df = pd.concat([positive_df, negative_df]).reset_index(drop=True)
    
    logging.info(f"Final balanced distribution:\n{balanced_df[label_column].value_counts()}")
    
    return balanced_df


def load_data(config):
    train_path = config["data"]["train_path"]
    dev_path = config["data"]["dev_path"]
    test_path = config["data"]["test_path"]
    label_column = config["data"]["label_column"]
    
    # Get no_relation label name from config or default
    no_relation_label = config["data"].get("no_relation_label", "No_Relation")

    logging.info("Loading datasets...")
    train_df = pd.read_csv(train_path)
    dev_df = pd.read_csv(dev_path)
    test_df = pd.read_csv(test_path)

    logging.info(f"Original Train shape: {train_df.shape}")
    logging.info(f"Original Dev shape: {dev_df.shape}")
    logging.info(f"Test shape: {test_df.shape}")

    logging.info("\nOriginal Train label distribution:\n%s", train_df[label_column].value_counts())
    logging.info("\nOriginal Dev label distribution:\n%s", dev_df[label_column].value_counts())
    logging.info("\nTest label distribution:\n%s", test_df[label_column].value_counts())

    # Balance negative samples in training data (optional, controlled by config)
    balance_training = config["data"].get("balance_negative_samples", True)
    max_neg_scale = config["data"].get("max_neg_scale", 1.0)
    
    if balance_training:
        logging.info("\nBalancing negative samples in training data...")
        train_df = balance_negative_samples(train_df, label_column, no_relation_label, max_neg_scale)
    
    logging.info("\nFinal Train label distribution:\n%s", train_df[label_column].value_counts())

    return train_df, dev_df, test_df, no_relation_label


class WeightedLossTrainer(Trainer):
    """
    Custom Hugging Face Trainer that overrides compute_loss to inject class weights.
    Balances minority and majority classes dynamically during CrossEntropy loss evaluation.
    """
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            # Sort weights by string name to exactly align with sorted(unique()) label IDs
            sorted_weights = [class_weights[label] for label in sorted(class_weights.keys())]
            self.class_weights = torch.tensor(sorted_weights, dtype=torch.float32)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if self.class_weights is not None:
            weight_tensor = self.class_weights.to(logits.device)
            loss_fct = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            loss_fct = nn.CrossEntropyLoss()
            
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def main():
    args = parse_args()
    config = load_config(args.config)

    output_dir = config["output"]["dir"]
    setup_logging(output_dir)

    seed = config["training"]["seed"]
    set_seed(seed)

    model_name = config["model"]["name"]
    max_length = config["model"]["max_length"]

    text_column = config["data"]["text_column"]
    label_column = config["data"]["label_column"]

    logging.info("Starting PubMedBERT BioRED baseline training")
    logging.info(f"Config file: {args.config}")
    logging.info(f"Model: {model_name}")
    logging.info(f"Output directory: {output_dir}")

    train_df, dev_df, test_df, no_relation_label = load_data(config)

    labels = sorted(train_df[label_column].unique())
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    logging.info(f"Labels: {labels}")
    logging.info(f"Number of labels: {len(labels)}")
    
    # Compute class weights for weighted loss
    label_counts = train_df[label_column].value_counts().to_dict()
    class_weights = compute_class_weights(label_counts, no_relation_label)
    logging.info(f"Class weights: {class_weights}")

    train_df["labels"] = train_df[label_column].map(label2id)
    dev_df["labels"] = dev_df[label_column].map(label2id)
    test_df["labels"] = test_df[label_column].map(label2id)

    assert train_df["labels"].isna().sum() == 0
    assert dev_df["labels"].isna().sum() == 0
    assert test_df["labels"].isna().sum() == 0

    with open(Path(output_dir) / "label2id.json", "w", encoding="utf-8") as f:
        json.dump(label2id, f, indent=2)

    with open(Path(output_dir) / "id2label.json", "w", encoding="utf-8") as f:
        json.dump(id2label, f, indent=2)

    train_dataset = Dataset.from_pandas(train_df)
    dev_dataset = Dataset.from_pandas(dev_df)
    test_dataset = Dataset.from_pandas(test_df)

    logging.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    special_tokens_dict = {'additional_special_tokens': ['[ENTITY1]', '[/ENTITY1]', '[ENTITY2]', '[/ENTITY2]']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    logging.info(f"Added {num_added_toks} new special tokens to the tokenizer.")

    def tokenize(batch):
        return tokenizer(
            batch[text_column],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    logging.info("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize, batched=True)
    dev_dataset = dev_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    keep_columns = ["input_ids", "attention_mask", "labels"]

    train_dataset = train_dataset.remove_columns(
        [col for col in train_dataset.column_names if col not in keep_columns]
    )
    dev_dataset = dev_dataset.remove_columns(
        [col for col in dev_dataset.column_names if col not in keep_columns]
    )
    test_dataset = test_dataset.remove_columns(
        [col for col in test_dataset.column_names if col not in keep_columns]
    )

    train_dataset.set_format("torch")
    dev_dataset.set_format("torch")
    test_dataset.set_format("torch")

    logging.info("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    model.resize_token_embeddings(len(tokenizer))
    logging.info(f"Resized model token embeddings to: {len(tokenizer)}")

    # Get no_relation label ID
    no_relation_id = label2id[no_relation_label]
    
    def compute_metrics(eval_pred):
        """
        Compute metrics excluding the "No_Relation" label.
        This matches the BioRED evaluation approach where F1 is calculated
        only on positive relation types.
        """
        logits, gold_labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        # Create mask for non-negative (positive relation) instances
        positive_mask = gold_labels != no_relation_id
        
        # Filter to positive instances only
        positive_gold = gold_labels[positive_mask]
        positive_pred = predictions[positive_mask]
        
        # Calculate metrics on positive instances only
        if len(positive_gold) > 0:
            positive_f1 = f1_score(positive_gold, positive_pred, average="macro", zero_division=0)
            positive_precision, positive_recall, _, _ = precision_recall_fscore_support(
                positive_gold, positive_pred, average="macro", zero_division=0
            )
        else:
            positive_f1 = 0.0
            positive_precision = 0.0
            positive_recall = 0.0
        
        # Also compute overall metrics for reference
        overall_accuracy = accuracy_score(gold_labels, predictions)
        overall_f1 = f1_score(gold_labels, predictions, average="macro", zero_division=0)

        return {
            "accuracy": overall_accuracy,
            "macro_f1": overall_f1,
            "f1_without_no_relation": positive_f1,
            "precision_without_no_relation": positive_precision,
            "recall_without_no_relation": positive_recall,
        }

    training_args = TrainingArguments(
        output_dir=output_dir,

        eval_strategy=config["training"].get("eval_strategy", "epoch"),  # "steps" or "epoch"
        save_strategy="steps" if config["training"].get("eval_strategy") == "steps" else "epoch",
        logging_strategy="steps",
        logging_steps=config["training"].get("eval_steps", 25),  # Log frequency

        # Evaluation and saving
        eval_steps=config["training"].get("eval_steps", None),  # If eval_strategy="steps"
        save_steps=config["training"].get("save_steps", None),

        learning_rate=float(config["training"].get("learning_rate", 1e-5)),
        per_device_train_batch_size=config["training"].get("batch_size", 16),
        per_device_eval_batch_size=config["training"].get("eval_batch_size", config["training"].get("batch_size", 16)),
        num_train_epochs=config["training"].get("epochs", 10),
        weight_decay=float(config["training"].get("weight_decay", 0.01)),

        load_best_model_at_end=True,
        metric_for_best_model="f1_without_no_relation",
        greater_is_better=True,

        save_total_limit=2,
        report_to="none",

        seed=seed,
        fp16=config["training"].get("fp16", True),
    )

    # Replaced standard Trainer with our WeightedLossTrainer subclass
    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    logging.info("Starting training...")
    trainer.train()

    logging.info("Evaluating best model on test set...")
    test_results = trainer.evaluate(test_dataset)

    logging.info("Final test results:")
    logging.info(test_results)

    with open(Path(output_dir) / "test_results.json", "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2)

    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=-1)
    y_true = predictions.label_ids

    report = classification_report(
        y_true,
        y_pred,
        target_names=labels,
        zero_division=0,
    )
    
    # Also create a report excluding No_Relation
    positive_mask = y_true != no_relation_id
    positive_labels = [label for label in labels if label != no_relation_label]
    positive_ids = [label2id[label] for label in positive_labels] 

    if len(y_true[positive_mask]) > 0:
        report_without_no_relation = classification_report(
            y_true[positive_mask],
            y_pred[positive_mask],
            labels=positive_ids,
            target_names=positive_labels,
            zero_division=0,
        )
    else:
        report_without_no_relation = "No positive relations in test set"

    cm = confusion_matrix(y_true, y_pred)

    logging.info("\nClassification report (all labels):\n%s", report)
    logging.info("\nClassification report (excluding No_Relation):\n%s", report_without_no_relation)
    logging.info("\nConfusion matrix:\n%s", cm)

    with open(Path(output_dir) / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("CLASSIFICATION REPORT (All Labels)\n")
        f.write("=" * 80 + "\n")
        f.write(report)
        f.write("\n\n")
        f.write("=" * 80 + "\n")
        f.write("CLASSIFICATION REPORT (Excluding No_Relation)\n")
        f.write("=" * 80 + "\n")
        f.write(report_without_no_relation)

    pd.DataFrame(cm, index=labels, columns=labels).to_csv(
        Path(output_dir) / "confusion_matrix.csv"
    )

    final_model_dir = Path(output_dir) / "final_model"
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    logging.info(f"Model saved to: {final_model_dir}")
    logging.info("Done.")


if __name__ == "__main__":
    main()