#!/usr/bin/env python3
"""Generate reusable transformer embeddings and aligned row metadata."""
from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import AutoModel, AutoTokenizer

REQUIRED_CONFIG_KEYS = (
    "input_file",
    "output_dir",
    "text_column",
    "model_name",
    "max_length",
    "batch_size",
    "pooling",
)
METADATA_CONFIG = (
    ("doc_id_column", "doc_id"),
    ("label_column", "relation_type"),
    ("entity1_id_column", "entity1_id"),
    ("entity2_id_column", "entity2_id"),
    ("entity1_type_column", "entity1_type"),
    ("entity2_type_column", "entity2_type"),
    ("entity_pair_type_column", "entity_pair_type"),
)


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8-sig") as stream:
        config = yaml.safe_load(stream) or {}
    missing = [key for key in REQUIRED_CONFIG_KEYS if config.get(key) is None]
    if missing:
        raise ValueError(f"Missing required config value(s): {', '.join(missing)}")
    config["pooling"] = str(config["pooling"]).lower()
    if config["pooling"] not in {"cls", "mean"}:
        raise ValueError("pooling must be either 'cls' or 'mean'")
    for key in ("max_length", "batch_size"):
        config[key] = int(config[key])
        if config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    return config


def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_input(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series]:
    input_path = Path(config["input_file"]).expanduser()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path.resolve()}")
    frame = pd.read_csv(input_path)
    text_column = config["text_column"]
    if text_column not in frame.columns:
        raise KeyError(f"Required text column not found: {text_column!r}")
    missing_text = int(frame[text_column].isna().sum())
    if missing_text:
        warnings.warn(f"Replacing {missing_text} missing text value(s) with empty strings")
    texts = frame[text_column].fillna("").astype(str)
    return frame, texts


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    token_counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / token_counts


def pool_hidden_state(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    pooling: str,
) -> torch.Tensor:
    if pooling == "cls":
        return last_hidden_state[:, 0, :]
    if pooling == "mean":
        return mean_pool(last_hidden_state, attention_mask)
    raise ValueError(f"Unsupported pooling strategy: {pooling}")


def generate_embeddings(
    texts: pd.Series,
    tokenizer: Any,
    model: torch.nn.Module,
    device: torch.device,
    max_length: int,
    batch_size: int,
    pooling: str,
    progress_every: int = 10,
) -> tuple[np.ndarray, int]:
    number_of_rows = len(texts)
    number_of_batches = math.ceil(number_of_rows / batch_size) if number_of_rows else 0
    chunks: list[np.ndarray] = []
    model.eval()

    with torch.no_grad():
        for batch_index, start in enumerate(range(0, number_of_rows, batch_size), start=1):
            batch_texts = texts.iloc[start : start + batch_size].tolist()
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
            outputs = model(**encoded)
            pooled = pool_hidden_state(outputs.last_hidden_state, encoded["attention_mask"], pooling)
            chunks.append(pooled.detach().cpu().to(torch.float32).numpy())

            if batch_index == 1 or batch_index % progress_every == 0 or batch_index == number_of_batches:
                completed = min(start + batch_size, number_of_rows)
                print(f"Processed batch {batch_index}/{number_of_batches} ({completed}/{number_of_rows} rows)", flush=True)

    if chunks:
        return np.concatenate(chunks, axis=0), number_of_batches
    hidden_size = int(model.config.hidden_size)
    return np.empty((0, hidden_size), dtype=np.float32), number_of_batches


def build_metadata(frame: pd.DataFrame, texts: pd.Series, config: dict[str, Any]) -> pd.DataFrame:
    metadata = pd.DataFrame({"row_id": np.arange(len(frame), dtype=np.int64)})
    for config_key, output_name in METADATA_CONFIG:
        source_name = config.get(config_key)
        if source_name and source_name in frame.columns:
            metadata[output_name] = frame[source_name].reset_index(drop=True)
        elif source_name:
            warnings.warn(f"Optional metadata column unavailable: {source_name!r}")
    metadata["text_length_words"] = texts.str.split().str.len().astype(np.int64).reset_index(drop=True)
    return metadata


def write_json(data: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)


def run(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(config["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, texts = load_input(config)
    device = select_device()
    print(f"Loading model: {config['model_name']}", flush=True)
    print(f"Device: {device}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    model = AutoModel.from_pretrained(config["model_name"]).to(device)
    embeddings, number_of_batches = generate_embeddings(
        texts=texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_length=config["max_length"],
        batch_size=config["batch_size"],
        pooling=config["pooling"],
        progress_every=int(config.get("progress_every", 10)),
    )
    metadata = build_metadata(frame, texts, config)
    if len(metadata) != embeddings.shape[0]:
        raise RuntimeError("Embedding and metadata row counts are not aligned")

    np.save(output_dir / "embeddings.npy", embeddings, allow_pickle=False)
    metadata.to_csv(output_dir / "metadata.csv", index=False)
    write_json(config, output_dir / "embedding_config.json")
    summary = {
        "number_of_rows": int(embeddings.shape[0]),
        "embedding_dimension": int(embeddings.shape[1]),
        "model_name": config["model_name"],
        "pooling": config["pooling"],
        "max_length": config["max_length"],
        "batch_size": config["batch_size"],
        "device": str(device),
        "number_of_batches": number_of_batches,
    }
    write_json(summary, output_dir / "embedding_summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Path to embedding-generation YAML")
    args = parser.parse_args()
    summary = run(load_config(args.config))
    print(
        f"Embedding generation complete: {summary['number_of_rows']} rows x "
        f"{summary['embedding_dimension']} dimensions using {summary['pooling']} pooling."
    )


if __name__ == "__main__":
    main()
