# import pandas as pd
# import numpy as np

# from datasets import Dataset
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, f1_score

# from transformers import (
#     AutoTokenizer,
#     AutoModelForSequenceClassification,
#     TrainingArguments,
#     Trainer
# )

# MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"

# TRAIN_DATA_PATH = "data/processed/train.csv"
# TEST_DATA_PATH = "data/processed/test.csv"


# # -----------------------
# # Load dataset
# # -----------------------

# train_df = pd.read_csv(TRAIN_DATA_PATH)
# test_df = pd.read_csv(TEST_DATA_PATH)

# print("Training set:")
# print(train_df.head())
# print(train_df["label"].value_counts())

# print("\nTest set:")
# print(test_df.head())
# print(test_df["label"].value_counts())

# # -----------------------
# # Encode labels
# # -----------------------

# labels = sorted(train_df["label"].unique())

# label2id = {
#     label: idx
#     for idx, label in enumerate(labels)
# }

# id2label = {
#     idx: label
#     for label, idx in label2id.items()
# }

# train_df["label_id"] = train_df["label"].map(label2id)
# test_df["label_id"] = test_df["label"].map(label2id)

# # -----------------------
# # Prepare datasets
# # -----------------------

# train_dataset = Dataset.from_pandas(train_df)
# test_dataset = Dataset.from_pandas(test_df)

# # -----------------------
# # Tokenizer
# # -----------------------

# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# def tokenize(batch):
#     return tokenizer(
#         batch["text"],
#         truncation=True,
#         padding="max_length",
#         max_length=256
#     )


# train_dataset = train_dataset.map(tokenize, batched=True)
# test_dataset = test_dataset.map(tokenize, batched=True)

# # -----------------------
# # Model
# # -----------------------

# model = AutoModelForSequenceClassification.from_pretrained(
#     MODEL_NAME,
#     num_labels=len(labels),
#     id2label=id2label,
#     label2id=label2id
# )

# # -----------------------
# # Metrics
# # -----------------------

# def compute_metrics(eval_pred):

#     logits, labels = eval_pred

#     predictions = np.argmax(logits, axis=-1)

#     accuracy = accuracy_score(labels, predictions)

#     f1 = f1_score(
#         labels,
#         predictions,
#         average="macro"
#     )

#     return {
#         "accuracy": accuracy,
#         "macro_f1": f1
#     }

# # -----------------------
# # Training args
# # -----------------------

# training_args = TrainingArguments(
#     output_dir="results/pubmedbert_baseline",

#     evaluation_strategy="epoch",
#     save_strategy="epoch",

#     learning_rate=2e-5,

#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,

#     num_train_epochs=5,

#     weight_decay=0.01,

#     logging_dir="results/logs",

#     load_best_model_at_end=True,

#     metric_for_best_model="macro_f1"
# )

# # -----------------------
# # Trainer
# # -----------------------

# trainer = Trainer(
#     model=model,
#     args=training_args,

#     train_dataset=train_dataset,
#     eval_dataset=test_dataset,

#     tokenizer=tokenizer,

#     compute_metrics=compute_metrics
# )

# # -----------------------
# # Train
# # -----------------------

# trainer.train()

# # -----------------------
# # Final evaluation
# # -----------------------

# results = trainer.evaluate()

# print("\nFINAL RESULTS")
# print(results)

# trainer.save_model(
#     "results/pubmedbert_baseline/final_model"
# )

import argparse
import json
import logging
from pathlib import Path

import yaml
import pandas as pd
import numpy as np

from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
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


def load_data(config):
    train_path = config["data"]["train_path"]
    dev_path = config["data"]["dev_path"]
    test_path = config["data"]["test_path"]
    label_column = config["data"]["label_column"]

    logging.info("Loading datasets...")
    train_df = pd.read_csv(train_path)
    dev_df = pd.read_csv(dev_path)
    test_df = pd.read_csv(test_path)

    logging.info(f"Train shape: {train_df.shape}")
    logging.info(f"Dev shape: {dev_df.shape}")
    logging.info(f"Test shape: {test_df.shape}")

    logging.info("\nTrain label distribution:\n%s", train_df[label_column].value_counts())
    logging.info("\nDev label distribution:\n%s", dev_df[label_column].value_counts())
    logging.info("\nTest label distribution:\n%s", test_df[label_column].value_counts())

    return train_df, dev_df, test_df


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

    train_df, dev_df, test_df = load_data(config)

    labels = sorted(train_df[label_column].unique())
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    logging.info(f"Labels: {labels}")
    logging.info(f"Number of labels: {len(labels)}")

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

    def compute_metrics(eval_pred):
        logits, gold_labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        return {
            "accuracy": accuracy_score(gold_labels, predictions),
            "macro_f1": f1_score(gold_labels, predictions, average="macro", zero_division=0),
            "weighted_f1": f1_score(gold_labels, predictions, average="weighted", zero_division=0),
        }

    training_args = TrainingArguments(
        output_dir=output_dir,

        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=25,

        learning_rate=float(config["training"]["learning_rate"]),
        per_device_train_batch_size=config["training"]["batch_size"],
        per_device_eval_batch_size=config["training"]["batch_size"],
        num_train_epochs=config["training"]["epochs"],
        weight_decay=float(config["training"]["weight_decay"]),

        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,

        save_total_limit=2,
        report_to="none",

        seed=seed,
        fp16=config["training"]["fp16"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
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

    cm = confusion_matrix(y_true, y_pred)

    logging.info("\nClassification report:\n%s", report)
    logging.info("\nConfusion matrix:\n%s", cm)

    with open(Path(output_dir) / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

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