#!/usr/bin/env python3
"""Compute Qwen real-minus-synthetic activation vectors for EasySteer generation.

The output is a per-layer direction:

    v_l = mean(h_l(real BioRED abstracts)) - mean(h_l(synthetic abstracts))

This is meant for generation-time steering
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


ENTITY_MARKER_RE = re.compile(r"\[/?[A-Z0-9]+_(?:SRC|TGT)\]\s*")
REQUIRED_CONFIG_KEYS = ("real_input_file", "synthetic_input_file", "output_dir", "model_name")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8-sig") as stream:
        config = yaml.safe_load(stream) or {}
    missing = [key for key in REQUIRED_CONFIG_KEYS if config.get(key) is None]
    if missing:
        raise ValueError(f"Missing required config value(s): {', '.join(missing)}")
    config["doc_id_column"] = str(config.get("doc_id_column", "doc_id"))
    config["text_column"] = str(config.get("text_column", "marked_text"))
    config["synthetic_abstract_column"] = str(config.get("synthetic_abstract_column", "abstract"))
    config["max_length"] = int(config.get("max_length", 1024))
    config["batch_size"] = int(config.get("batch_size", 4))
    config["layer_indices"] = [int(layer) for layer in config.get("layer_indices", [-1])]
    config["position"] = str(config.get("position", "last_token")).lower()
    if config["position"] not in {"last_token", "mean"}:
        raise ValueError("position must be 'last_token' or 'mean'")
    config["max_documents_per_domain"] = config.get("max_documents_per_domain")
    if config["max_documents_per_domain"] is not None:
        config["max_documents_per_domain"] = int(config["max_documents_per_domain"])
    config["random_state"] = int(config.get("random_state", 42))
    config["dtype"] = str(config.get("dtype", "auto"))
    config["trust_remote_code"] = bool(config.get("trust_remote_code", True))
    config["progress_every"] = int(config.get("progress_every", 10))
    return config


def strip_entity_markers(text: str) -> str:
    return re.sub(r"\s+", " ", ENTITY_MARKER_RE.sub("", str(text))).strip()


def load_unique_document_texts(path: str, config: dict[str, Any], domain: str) -> list[str]:
    frame = pd.read_csv(Path(path).expanduser())
    doc_col = config["doc_id_column"]
    text_col = config["text_column"]
    synthetic_abstract_col = config["synthetic_abstract_column"]

    if domain == "synthetic" and synthetic_abstract_col in frame.columns:
        source_col = synthetic_abstract_col
        strip_markers = False
    elif text_col in frame.columns:
        source_col = text_col
        strip_markers = True
    else:
        raise KeyError(f"{domain} input must contain either {synthetic_abstract_col!r} or {text_col!r}")

    if doc_col in frame.columns:
        series = frame.groupby(doc_col, sort=False)[source_col].first()
    else:
        series = frame[source_col]

    texts = series.fillna("").astype(str).tolist()
    if strip_markers:
        texts = [strip_entity_markers(text) for text in texts]
    texts = [text for text in texts if text.strip()]

    limit = config.get("max_documents_per_domain")
    if limit is not None and len(texts) > limit:
        rng = np.random.default_rng(config["random_state"])
        indices = rng.choice(np.arange(len(texts)), size=limit, replace=False)
        texts = [texts[int(i)] for i in indices]
    return texts


def torch_dtype_from_config(config: dict[str, Any]) -> Any:
    if config["dtype"] == "auto":
        return "auto"
    if hasattr(torch, config["dtype"]):
        return getattr(torch, config["dtype"])
    return config["dtype"]


def resolve_layer_indices(requested: list[int], n_hidden_states: int) -> dict[int, int]:
    resolved: dict[int, int] = {}
    n_model_layers = n_hidden_states - 1
    for layer in requested:
        model_layer = n_model_layers + layer if layer < 0 else layer
        hidden_index = model_layer + 1
        if model_layer < 0 or hidden_index >= n_hidden_states:
            raise ValueError(
                f"Requested layer {layer} resolves to model layer {model_layer}, "
                f"but model returned {n_model_layers} layers"
            )
        resolved[layer] = model_layer
    return resolved


def pool_hidden(hidden: torch.Tensor, attention_mask: torch.Tensor, position: str) -> torch.Tensor:
    if position == "last_token":
        lengths = attention_mask.sum(dim=1).clamp(min=1) - 1
        batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[batch_idx, lengths, :]
    mask = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


def mean_layer_states(
    texts: list[str],
    tokenizer: Any,
    model: torch.nn.Module,
    config: dict[str, Any],
    domain: str,
) -> tuple[dict[int, np.ndarray], dict[int, int], int]:
    batches = math.ceil(len(texts) / config["batch_size"]) if texts else 0
    sums: dict[int, torch.Tensor] = {}
    resolved_layers: dict[int, int] | None = None
    count = 0

    model.eval()
    with torch.no_grad():
        for batch_index, start in enumerate(range(0, len(texts), config["batch_size"]), start=1):
            batch_texts = texts[start : start + config["batch_size"]]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=config["max_length"],
                return_tensors="pt",
            )
            encoded = {key: value.to(model.device) for key, value in encoded.items()}
            outputs = model(**encoded, output_hidden_states=True, use_cache=False)
            if resolved_layers is None:
                resolved_layers = resolve_layer_indices(config["layer_indices"], len(outputs.hidden_states))
            for _requested_layer, model_layer in resolved_layers.items():
                hidden_index = model_layer + 1
                pooled = pool_hidden(outputs.hidden_states[hidden_index], encoded["attention_mask"], config["position"])
                batch_sum = pooled.detach().sum(dim=0).to(torch.float64)
                sums[model_layer] = batch_sum if model_layer not in sums else sums[model_layer] + batch_sum
            count += len(batch_texts)
            if batch_index == 1 or batch_index % config["progress_every"] == 0 or batch_index == batches:
                print(f"{domain}: processed batch {batch_index}/{batches} ({count}/{len(texts)} docs)", flush=True)

    if not sums or resolved_layers is None:
        raise ValueError(f"No hidden states collected for {domain}")
    means = {layer: (value / count).cpu().numpy().astype(np.float32) for layer, value in sums.items()}
    return means, resolved_layers, count


def write_json(data: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)


def run(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(config["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config["random_state"])

    real_texts = load_unique_document_texts(config["real_input_file"], config, "real")
    synthetic_texts = load_unique_document_texts(config["synthetic_input_file"], config, "synthetic")
    n = min(len(real_texts), len(synthetic_texts))
    if config.get("max_documents_per_domain") is not None:
        n = min(n, config["max_documents_per_domain"])
    real_texts = real_texts[:n]
    synthetic_texts = synthetic_texts[:n]

    print(f"Loading Qwen model: {config['model_name']}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], trust_remote_code=config["trust_remote_code"])
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        torch_dtype=torch_dtype_from_config(config),
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=config["trust_remote_code"],
    )
    if not torch.cuda.is_available():
        model = model.to("cpu")

    real_means, resolved_layers, real_count = mean_layer_states(real_texts, tokenizer, model, config, "real")
    synth_means, _, synth_count = mean_layer_states(synthetic_texts, tokenizer, model, config, "synthetic")

    arrays: dict[str, np.ndarray] = {}
    layer_summaries: dict[str, Any] = {}
    for layer, real_mean in real_means.items():
        synth_mean = synth_means[layer]
        direction = real_mean - synth_mean
        arrays[f"layer_{layer}"] = direction.astype(np.float32)
        layer_summaries[str(layer)] = {
            "direction_norm": float(np.linalg.norm(direction)),
            "real_centroid_norm": float(np.linalg.norm(real_mean)),
            "synthetic_centroid_norm": float(np.linalg.norm(synth_mean)),
            "centroid_cosine_similarity": float(
                np.dot(real_mean, synth_mean) / (np.linalg.norm(real_mean) * np.linalg.norm(synth_mean) + 1e-12)
            ),
        }

    np.savez(output_dir / "qwen_generation_steering_vectors.npz", **arrays)
    summary = {
        "model_name": config["model_name"],
        "real_input_file": config["real_input_file"],
        "synthetic_input_file": config["synthetic_input_file"],
        "real_documents": int(real_count),
        "synthetic_documents": int(synth_count),
        "position": config["position"],
        "max_length": config["max_length"],
        "batch_size": config["batch_size"],
        "requested_layer_indices": config["layer_indices"],
        "resolved_model_layers": sorted(int(key.replace("layer_", "")) for key in arrays),
        "requested_to_resolved_layers": {str(k): int(v) for k, v in resolved_layers.items()},
        "layers": layer_summaries,
        "outputs": {
            "qwen_generation_steering_vectors": "qwen_generation_steering_vectors.npz",
        },
    }
    write_json(summary, output_dir / "qwen_generation_steering_vector_summary.json")
    write_json(config, output_dir / "qwen_generation_steering_vector_config.json")
    return summary


def main() -> None:
    args = parse_args()
    summary = run(load_config(args.config))
    print("Qwen generation steering vector computed.")
    print(f"Resolved layers: {summary['resolved_model_layers']}")
    print(f"Documents per domain: {summary['real_documents']}")
    print(f"Outputs saved under: {Path(summary['outputs']['qwen_generation_steering_vectors']).name}")


if __name__ == "__main__":
    main()
