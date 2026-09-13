#Adds the steering direction during prediction
# import argparse
# import json
# from pathlib import Path

# import numpy as np
# import pandas as pd
# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification


# BIORED_LABELS = [
#     "None",
#     "Association",
#     "Bind",
#     "Comparison",
#     "Conversion",
#     "Cotreatment",
#     "Drug_Interaction",
#     "Negative_Correlation",
#     "Positive_Correlation",
# ]


# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--test_csv", required=True)
#     parser.add_argument("--binary_model", required=True)
#     parser.add_argument("--multiclass_model", required=True)
#     parser.add_argument("--output_logits", required=True)
#     parser.add_argument("--batch_size", type=int, default=32)
#     parser.add_argument("--max_length", type=int, default=512)
#     parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
#     parser.add_argument(
#         "--steering_layer",
#         type=int,
#         default=None,
#         help="Encoder layer index where the steering vector should be injected. Disabled if omitted.",
#     )
#     parser.add_argument(
#         "--steering_alpha",
#         type=float,
#         default=0.0,
#         help="Steering strength. 0.0 means no steering.",
#     )
#     parser.add_argument(
#         "--steering_position",
#         choices=["cls", "all_tokens"],
#         default="cls",
#         help="Apply steering only to CLS or to all token activations.",
#     )
#     parser.add_argument(
#         "--steering_vector",
#         default=None,
#         help="Optional shared .npy steering vector for both binary and multiclass models.",
#     )
#     parser.add_argument(
#         "--binary_steering_vector",
#         default=None,
#         help="Optional .npy steering vector for the binary model. Overrides --steering_vector.",
#     )
#     parser.add_argument(
#         "--multiclass_steering_vector",
#         default=None,
#         help="Optional .npy steering vector for the multiclass model. Overrides --steering_vector.",
#     )
#     return parser.parse_args()


# def load_id2label(model_dir):
#     config_path = Path(model_dir) / "config.json"
#     with open(config_path) as f:
#         config = json.load(f)

#     return {int(k): v for k, v in config["id2label"].items()}


# def get_bert_encoder_layer(model, layer_index):
#     base_model = getattr(model, "bert", None)
#     if base_model is None:
#         base_model = getattr(model, model.base_model_prefix, None)
#     if base_model is None or not hasattr(base_model, "encoder"):
#         raise ValueError("Could not find a BERT-style encoder on the model for activation steering.")

#     layers = base_model.encoder.layer
#     if layer_index < 0:
#         layer_index = len(layers) + layer_index
#     if layer_index < 0 or layer_index >= len(layers):
#         raise ValueError(f"Invalid steering_layer={layer_index}; model has {len(layers)} encoder layers.")
#     return layers[layer_index].output


# def register_activation_steering_hook(model, vector_path, alpha, layer_index, position, device):
#     if not vector_path or alpha == 0.0 or layer_index is None:
#         return None

#     vector = np.load(vector_path)
#     if vector.ndim != 1:
#         raise ValueError(f"Steering vector must be 1D, got shape {vector.shape}: {vector_path}")
#     hidden_size = int(model.config.hidden_size)
#     if vector.shape[0] != hidden_size:
#         raise ValueError(
#             f"Steering vector dimension {vector.shape[0]} does not match model hidden size {hidden_size}: {vector_path}"
#         )
#     vector_tensor = torch.tensor(vector, dtype=torch.float32, device=device)

#     def hook(_module, _inputs, output):
#         shift = (alpha * vector_tensor).to(dtype=output.dtype, device=output.device)
#         steered = output.clone()
#         if position == "cls":
#             steered[:, 0, :] = steered[:, 0, :] + shift
#         else:
#             steered = steered + shift.view(1, 1, -1)
#         return steered

#     target_module = get_bert_encoder_layer(model, layer_index)
#     return target_module.register_forward_hook(hook)


# def main():
#     args = parse_args()

#     df = pd.read_csv(args.test_csv)
#     texts = df["marked_text"].astype(str).tolist()

#     device = torch.device(args.device)

#     binary_tokenizer = AutoTokenizer.from_pretrained(args.binary_model)
#     binary_model = AutoModelForSequenceClassification.from_pretrained(args.binary_model).to(device)
#     binary_model.eval()

#     multi_tokenizer = AutoTokenizer.from_pretrained(args.multiclass_model)
#     multi_model = AutoModelForSequenceClassification.from_pretrained(args.multiclass_model).to(device)
#     multi_model.eval()

#     binary_vector = args.binary_steering_vector or args.steering_vector
#     multiclass_vector = args.multiclass_steering_vector or args.steering_vector
#     binary_hook = register_activation_steering_hook(
#         binary_model,
#         binary_vector,
#         args.steering_alpha,
#         args.steering_layer,
#         args.steering_position,
#         device,
#     )
#     multiclass_hook = register_activation_steering_hook(
#         multi_model,
#         multiclass_vector,
#         args.steering_alpha,
#         args.steering_layer,
#         args.steering_position,
#         device,
#     )
#     if binary_hook or multiclass_hook:
#         print(
#             "Activation steering enabled: "
#             f"layer={args.steering_layer}, alpha={args.steering_alpha}, position={args.steering_position}"
#         )
#         print(f"Binary vector: {binary_vector}")
#         print(f"Multiclass vector: {multiclass_vector}")

#     binary_id2label = load_id2label(args.binary_model)
#     multi_id2label = load_id2label(args.multiclass_model)

#     final_labels = []

#     with torch.no_grad():
#         for start in range(0, len(texts), args.batch_size):
#             batch_texts = texts[start:start + args.batch_size]

#             enc = binary_tokenizer(
#                 batch_texts,
#                 padding=True,
#                 truncation=True,
#                 max_length=args.max_length,
#                 return_tensors="pt"
#             ).to(device)

#             binary_logits = binary_model(**enc).logits
#             binary_preds = torch.argmax(binary_logits, dim=1).cpu().numpy()

#             for i, binary_pred in enumerate(binary_preds):
#                 binary_label = binary_id2label[int(binary_pred)]

#                 if binary_label in ["No_Relation", "None"]:
#                     final_labels.append("None")
#                 else:
#                     text = batch_texts[i]

#                     multi_enc = multi_tokenizer(
#                         [text],
#                         padding=True,
#                         truncation=True,
#                         max_length=args.max_length,
#                         return_tensors="pt"
#                     ).to(device)

#                     multi_logits = multi_model(**multi_enc).logits
#                     multi_pred = torch.argmax(multi_logits, dim=1).item()
#                     rel_label = multi_id2label[int(multi_pred)]

#                     if rel_label == "No_Relation":
#                         rel_label = "None"

#                     final_labels.append(rel_label)

#     label2id = {label: i for i, label in enumerate(BIORED_LABELS)}

#     logits = []
#     for label in final_labels:
#         row = [-10.0] * len(BIORED_LABELS)
#         row[label2id[label]] = 10.0
#         logits.append(row)

#     pd.DataFrame(logits).to_csv(args.output_logits, sep="\t", header=False, index=False)
#     pd.DataFrame({"predicted_label": final_labels}).to_csv(
#         args.output_logits.replace(".tsv", "_labels.csv"),
#         index=False
#     )
#     if binary_hook:
#         binary_hook.remove()
#     if multiclass_hook:
#         multiclass_hook.remove()
    
#     print(f"Saved BioRED-style logits to {args.output_logits}")
#     print(pd.Series(final_labels).value_counts())


# if __name__ == "__main__":
#     main()


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