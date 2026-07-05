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


def create_binary_labels(df, label_column, no_relation_label="No_Relation"):
    """
    Create binary labels: 1 for has_relation, 0 for no_relation
    """
    df = df.copy()
    df["binary_label"] = (df[label_column] != no_relation_label).astype(int)
    return df


def get_positive_examples(df, label_column, no_relation_label="No_Relation"):
    """
    Filter to only positive examples (has relation)
    """
    return df[df[label_column] != no_relation_label].copy()


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


def compute_metrics(eval_pred, is_binary=False):
    """
    Compute metrics for binary or multi-class classification.
    """
    logits, gold_labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    if is_binary:
        # Binary classification metrics
        accuracy = accuracy_score(gold_labels, predictions)
        f1 = f1_score(gold_labels, predictions, average="binary", zero_division=0)
        precision, recall, _, _ = precision_recall_fscore_support(
            gold_labels, predictions, average="binary", zero_division=0
        )
        return {
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }
    else:
        # Multi-class metrics (using micro F1 as per BiORED paper)
        overall_accuracy = accuracy_score(gold_labels, predictions)
        micro_f1 = f1_score(gold_labels, predictions, average="micro", zero_division=0)
        
        return {
            "accuracy": overall_accuracy,
            "micro_f1": micro_f1,
        }


def train_binary_stage(config, model_name, text_column, tokenizer, output_dir):
    """
    Stage 1: Train binary classifier for relation detection (relation vs no_relation)
    """
    logging.info("\n" + "=" * 80)
    logging.info("STAGE 1: Training Binary Relation Detector (relation vs no_relation)")
    logging.info("=" * 80 + "\n")
    
    train_df, dev_df, test_df, no_relation_label = load_data(config)
    
    # Create binary labels
    train_df = create_binary_labels(train_df, config["data"]["label_column"], no_relation_label)
    dev_df = create_binary_labels(dev_df, config["data"]["label_column"], no_relation_label)
    test_df = create_binary_labels(test_df, config["data"]["label_column"], no_relation_label)
    
    logging.info("Binary training data distribution:\n%s", train_df["binary_label"].value_counts())
    logging.info("Binary dev data distribution:\n%s", dev_df["binary_label"].value_counts())
    
    # Prepare datasets
    train_dataset = Dataset.from_pandas(train_df)
    dev_dataset = Dataset.from_pandas(dev_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    max_length = config["model"]["max_length"]
    
    def tokenize(batch):
        return tokenizer(
            batch[text_column],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
    
    logging.info("Tokenizing binary datasets...")
    train_dataset = train_dataset.map(tokenize, batched=True)
    dev_dataset = dev_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)
    
    # Rename binary_label to labels for trainer
    train_dataset = train_dataset.rename_column("binary_label", "labels")
    dev_dataset = dev_dataset.rename_column("binary_label", "labels")
    test_dataset = test_dataset.rename_column("binary_label", "labels")
    
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
    
    logging.info("Loading binary model...")
    binary_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "No_Relation", 1: "Has_Relation"},
        label2id={"No_Relation": 0, "Has_Relation": 1},
    )
    binary_model.resize_token_embeddings(len(tokenizer))
    
    binary_output_dir = Path(output_dir) / "stage1_binary_classifier"
    binary_output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(binary_output_dir),
        eval_strategy=config["training"].get("eval_strategy", "epoch"),
        save_strategy="steps" if config["training"].get("eval_strategy") == "steps" else "epoch",
        logging_strategy="steps",
        logging_steps=config["training"].get("eval_steps", 25),
        eval_steps=config["training"].get("eval_steps", None),
        save_steps=config["training"].get("save_steps", None),
        learning_rate=float(config["training"].get("learning_rate", 1e-5)),
        per_device_train_batch_size=config["training"].get("batch_size", 16),
        per_device_eval_batch_size=config["training"].get("eval_batch_size", config["training"].get("batch_size", 16)),
        num_train_epochs=config["training"].get("epochs", 10),
        weight_decay=float(config["training"].get("weight_decay", 0.01)),
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=config["training"]["seed"],
        fp16=config["training"].get("fp16", True),
    )
    
    def binary_metrics(eval_pred):
        return compute_metrics(eval_pred, is_binary=True)
    
    trainer = Trainer(
        model=binary_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        compute_metrics=binary_metrics,
    )
    
    logging.info("Starting binary classifier training...")
    trainer.train()
    
    logging.info("Evaluating binary model on test set...")
    test_results = trainer.evaluate(test_dataset)
    logging.info("Binary classifier test results: %s", test_results)
    
    with open(binary_output_dir / "test_results.json", "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2)
    
    # Save binary model
    final_binary_model_dir = binary_output_dir / "final_model"
    trainer.save_model(final_binary_model_dir)
    tokenizer.save_pretrained(final_binary_model_dir)
    
    logging.info(f"Binary model saved to: {final_binary_model_dir}")
    
    return trainer, binary_model, train_df, dev_df, test_df, no_relation_label


def train_multiclass_stage(config, model_name, text_column, tokenizer, output_dir, no_relation_label):
    """
    Stage 2: Train multi-class classifier for relation type classification
    (only on positive examples - those with actual relations)
    """
    logging.info("\n" + "=" * 80)
    logging.info("STAGE 2: Training Multi-class Relation Type Classifier")
    logging.info("=" * 80 + "\n")
    
    train_df, dev_df, test_df, _ = load_data(config)
    label_column = config["data"]["label_column"]
    
    # Filter to only positive examples
    train_pos = get_positive_examples(train_df, label_column, no_relation_label)
    dev_pos = get_positive_examples(dev_df, label_column, no_relation_label)
    test_pos = get_positive_examples(test_df, label_column, no_relation_label)
    
    logging.info(f"Positive train examples: {len(train_pos)} / {len(train_df)}")
    logging.info(f"Positive dev examples: {len(dev_pos)} / {len(dev_df)}")
    logging.info(f"Positive test examples: {len(test_pos)} / {len(test_df)}")
    
    logging.info("Positive train distribution:\n%s", train_pos[label_column].value_counts())
    logging.info("Positive dev distribution:\n%s", dev_pos[label_column].value_counts())
    
    # Get relation types (excluding no_relation)
    relation_labels = sorted(train_pos[label_column].unique())
    label2id = {label: idx for idx, label in enumerate(relation_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    
    logging.info(f"Relation types: {relation_labels}")
    logging.info(f"Number of relation types: {len(relation_labels)}")
    
    train_pos[label_column + "_id"] = train_pos[label_column].map(label2id)
    dev_pos[label_column + "_id"] = dev_pos[label_column].map(label2id)
    test_pos[label_column + "_id"] = test_pos[label_column].map(label2id)
    
    # Prepare datasets
    train_dataset = Dataset.from_pandas(train_pos)
    dev_dataset = Dataset.from_pandas(dev_pos)
    test_dataset = Dataset.from_pandas(test_pos)
    
    max_length = config["model"]["max_length"]
    
    def tokenize(batch):
        return tokenizer(
            batch[text_column],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
    
    logging.info("Tokenizing multi-class datasets...")
    train_dataset = train_dataset.map(tokenize, batched=True)
    dev_dataset = dev_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)
    
    # Rename to labels for trainer
    train_dataset = train_dataset.rename_column(label_column + "_id", "labels")
    dev_dataset = dev_dataset.rename_column(label_column + "_id", "labels")
    test_dataset = test_dataset.rename_column(label_column + "_id", "labels")
    
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
    
    logging.info("Loading multi-class model...")
    multiclass_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(relation_labels),
        id2label=id2label,
        label2id=label2id,
    )
    multiclass_model.resize_token_embeddings(len(tokenizer))
    
    # Save label mappings
    with open(Path(output_dir) / "label2id.json", "w", encoding="utf-8") as f:
        json.dump(label2id, f, indent=2)
    with open(Path(output_dir) / "id2label.json", "w", encoding="utf-8") as f:
        json.dump(id2label, f, indent=2)
    
    multiclass_output_dir = Path(output_dir) / "stage2_multiclass_classifier"
    multiclass_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute class weights for multi-class stage
    label_counts = train_pos[label_column].value_counts().to_dict()
    class_weights = compute_class_weights(label_counts, no_relation_label="")  # Empty string so no label matches
    logging.info(f"Class weights: {class_weights}")
    
    training_args = TrainingArguments(
        output_dir=str(multiclass_output_dir),
        eval_strategy=config["training"].get("eval_strategy", "epoch"),
        save_strategy="steps" if config["training"].get("eval_strategy") == "steps" else "epoch",
        logging_strategy="steps",
        logging_steps=config["training"].get("eval_steps", 25),
        eval_steps=config["training"].get("eval_steps", None),
        save_steps=config["training"].get("save_steps", None),
        learning_rate=float(config["training"].get("learning_rate", 1e-5)),
        per_device_train_batch_size=config["training"].get("batch_size", 16),
        per_device_eval_batch_size=config["training"].get("eval_batch_size", config["training"].get("batch_size", 16)),
        num_train_epochs=config["training"].get("epochs", 10),
        weight_decay=float(config["training"].get("weight_decay", 0.01)),
        load_best_model_at_end=True,
        metric_for_best_model="micro_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=config["training"]["seed"],
        fp16=config["training"].get("fp16", True),
    )
    
    def multiclass_metrics(eval_pred):
        return compute_metrics(eval_pred, is_binary=False)
    
    trainer = WeightedLossTrainer(
        model=multiclass_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        compute_metrics=multiclass_metrics,
        class_weights=class_weights,
    )
    
    logging.info("Starting multi-class classifier training...")
    trainer.train()
    
    logging.info("Evaluating multi-class model on positive test examples...")
    test_results = trainer.evaluate(test_dataset)
    logging.info("Multi-class classifier test results: %s", test_results)
    
    with open(multiclass_output_dir / "test_results.json", "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2)
    
    # Get predictions for classification report
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=-1)
    y_true = predictions.label_ids
    
    report = classification_report(
        y_true,
        y_pred,
        target_names=relation_labels,
        zero_division=0,
    )
    
    cm = confusion_matrix(y_true, y_pred)
    
    logging.info("\nMulti-class Classification Report:\n%s", report)
    logging.info("\nConfusion Matrix:\n%s", cm)
    
    with open(multiclass_output_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    
    pd.DataFrame(cm, index=relation_labels, columns=relation_labels).to_csv(
        multiclass_output_dir / "confusion_matrix.csv"
    )
    
    # Save multi-class model
    final_multiclass_model_dir = multiclass_output_dir / "final_model"
    trainer.save_model(final_multiclass_model_dir)
    tokenizer.save_pretrained(final_multiclass_model_dir)
    
    logging.info(f"Multi-class model saved to: {final_multiclass_model_dir}")
    
    return trainer, multiclass_model, relation_labels, label2id, id2label


def main():
    args = parse_args()
    config = load_config(args.config)

    output_dir = config["output"]["dir"]
    setup_logging(output_dir)

    seed = config["training"]["seed"]
    set_seed(seed)

    model_name = config["model"]["name"]
    text_column = config["data"]["text_column"]

    logging.info("Starting Hierarchical PubMedBERT Training (BioPred-style)")
    logging.info(f"Config file: {args.config}")
    logging.info(f"Model: {model_name}")
    logging.info(f"Output directory: {output_dir}")
    logging.info("\nApproach: Two-stage hierarchical classification")
    logging.info("  Stage 1: Binary classifier (relation vs no_relation)")
    logging.info("  Stage 2: Multi-class classifier (specific relation types)")

    # Load tokenizer once
    logging.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    special_tokens_dict = {'additional_special_tokens': [
        '[CHEMICAL_SRC]', '[/CHEMICAL_SRC]', '[CHEMICAL_TGT]', '[/CHEMICAL_TGT]',
        '[GENE_SRC]',     '[/GENE_SRC]',     '[GENE_TGT]',     '[/GENE_TGT]',
        '[DISEASE_SRC]',  '[/DISEASE_SRC]',  '[DISEASE_TGT]',  '[/DISEASE_TGT]',
        '[ENTITY_SRC]',   '[/ENTITY_SRC]',   '[ENTITY_TGT]',   '[/ENTITY_TGT]'
    ]}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    logging.info(f"Added {num_added_toks} new special tokens to the tokenizer.")

    # Stage 1: Train binary classifier
    binary_trainer, binary_model, train_df, dev_df, test_df, no_relation_label = train_binary_stage(
        config, model_name, text_column, tokenizer, output_dir
    )

    # Stage 2: Train multi-class classifier on positive examples
    multiclass_trainer, multiclass_model, relation_labels, label2id, id2label = train_multiclass_stage(
        config, model_name, text_column, tokenizer, output_dir, no_relation_label
    )

    logging.info("\n" + "=" * 80)
    logging.info("Training Complete - Both stages trained successfully")
    logging.info("=" * 80)
    logging.info(f"Binary classifier saved to: {Path(output_dir) / 'stage1_binary_classifier/final_model'}")
    logging.info(f"Multi-class classifier saved to: {Path(output_dir) / 'stage2_multiclass_classifier/final_model'}")
    logging.info("\nFor inference, use both models hierarchically:")
    logging.info("  1. Use binary model to detect if relation exists")
    logging.info("  2. If relation detected, use multi-class model to classify relation type")
    logging.info("  3. If no relation detected, output 'No_Relation'")


if __name__ == "__main__":
    main()
