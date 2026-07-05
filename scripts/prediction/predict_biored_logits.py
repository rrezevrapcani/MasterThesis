import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


BIORED_LABELS = [
    "None",
    "Association",
    "Bind",
    "Comparison",
    "Conversion",
    "Cotreatment",
    "Drug_Interaction",
    "Negative_Correlation",
    "Positive_Correlation",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--binary_model", required=True)
    parser.add_argument("--multiclass_model", required=True)
    parser.add_argument("--output_logits", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_id2label(model_dir):
    config_path = Path(model_dir) / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    return {int(k): v for k, v in config["id2label"].items()}


def main():
    args = parse_args()

    df = pd.read_csv(args.test_csv)
    texts = df["marked_text"].astype(str).tolist()

    device = torch.device(args.device)

    binary_tokenizer = AutoTokenizer.from_pretrained(args.binary_model)
    binary_model = AutoModelForSequenceClassification.from_pretrained(args.binary_model).to(device)
    binary_model.eval()

    multi_tokenizer = AutoTokenizer.from_pretrained(args.multiclass_model)
    multi_model = AutoModelForSequenceClassification.from_pretrained(args.multiclass_model).to(device)
    multi_model.eval()

    binary_id2label = load_id2label(args.binary_model)
    multi_id2label = load_id2label(args.multiclass_model)

    final_labels = []

    with torch.no_grad():
        for start in range(0, len(texts), args.batch_size):
            batch_texts = texts[start:start + args.batch_size]

            enc = binary_tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt"
            ).to(device)

            binary_logits = binary_model(**enc).logits
            binary_preds = torch.argmax(binary_logits, dim=1).cpu().numpy()

            for i, binary_pred in enumerate(binary_preds):
                binary_label = binary_id2label[int(binary_pred)]

                if binary_label in ["No_Relation", "None"]:
                    final_labels.append("None")
                else:
                    text = batch_texts[i]

                    multi_enc = multi_tokenizer(
                        [text],
                        padding=True,
                        truncation=True,
                        max_length=args.max_length,
                        return_tensors="pt"
                    ).to(device)

                    multi_logits = multi_model(**multi_enc).logits
                    multi_pred = torch.argmax(multi_logits, dim=1).item()
                    rel_label = multi_id2label[int(multi_pred)]

                    if rel_label == "No_Relation":
                        rel_label = "None"

                    final_labels.append(rel_label)

    label2id = {label: i for i, label in enumerate(BIORED_LABELS)}

    logits = []
    for label in final_labels:
        row = [-10.0] * len(BIORED_LABELS)
        row[label2id[label]] = 10.0
        logits.append(row)

    pd.DataFrame(logits).to_csv(args.output_logits, sep="\t", header=False, index=False)
    pd.DataFrame({"predicted_label": final_labels}).to_csv(
        args.output_logits.replace(".tsv", "_labels.csv"),
        index=False
    )
    
    print(f"Saved BioRED-style logits to {args.output_logits}")
    print(pd.Series(final_labels).value_counts())


if __name__ == "__main__":
    main()