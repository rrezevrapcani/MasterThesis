#!/usr/bin/env python3
"""Compute real-minus-synthetic hidden-state steering vectors for PubMedBERT.

This script estimates an activation steering direction from hidden states:

    v_l = mean(h_l(real)) - mean(h_l(synthetic))

The vector can then be injected into the same layer during prediction with
scripts/prediction/predict_biored_logits.py.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import AutoModel, AutoTokenizer


REQUIRED_CONFIG_KEYS = (
    "real_input_file",
    "synthetic_input_file",
    "output_dir",
    "model_name_or_path",
    "text_column",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path, help="Path to YAML config.")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8-sig") as stream:
        config = yaml.safe_load(stream) or {}
    missing = [key for key in REQUIRED_CONFIG_KEYS if config.get(key) is None]
    if missing:
        raise ValueError(f"Missing required config value(s): {', '.join(missing)}")
    config["max_length"] = int(config.get("max_length", 512))
    config["batch_size"] = int(config.get("batch_size", 32))
    config["layer_index"] = int(config.get("layer_index", -1))
    config["position"] = str(config.get("position", "cls")).lower()
    if config["position"] not in {"cls", "mean"}:
        raise ValueError("position must be 'cls' or 'mean'")
    config["balance_by_label"] = bool(config.get("balance_by_label", True))
    config["max_rows_per_domain"] = config.get("max_rows_per_domain")
    if config["max_rows_per_domain"] is not None:
        config["max_rows_per_domain"] = int(config["max_rows_per_domain"])
    config["random_state"] = int(config.get("random_state", 42))
    config["progress_every"] = int(config.get("progress_every", 10))
    return config


def load_frame(path: str, text_column: str, domain: str) -> pd.DataFrame:
    frame = pd.read_csv(Path(path).expanduser())
    if text_column not in frame.columns:
        raise KeyError(f"{domain} input is missing required text column: {text_column}")
    frame = frame.copy()
    frame[text_column] = frame[text_column].fillna("").astype(str)
    frame["domain"] = domain
    return frame


def sample_balanced_frames(real: pd.DataFrame, synth: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(config["random_state"])
    label_column = config.get("label_column")

    if config["balance_by_label"] and label_column and label_column in real.columns and label_column in synth.columns:
        real_parts: list[pd.DataFrame] = []
        synth_parts: list[pd.DataFrame] = []
        per_label: dict[str, int] = {}
        common_labels = sorted(set(real[label_column].dropna()) & set(synth[label_column].dropna()))
        for label in common_labels:
            real_label = real[real[label_column] == label]
            synth_label = synth[synth[label_column] == label]
            n = min(len(real_label), len(synth_label))
            if n == 0:
                continue
            real_parts.append(real_label.sample(n=n, random_state=int(rng.integers(0, 2**31 - 1))))
            synth_parts.append(synth_label.sample(n=n, random_state=int(rng.integers(0, 2**31 - 1))))
            per_label[str(label)] = int(n)
        if not real_parts:
            raise ValueError("No shared labels found for balanced sampling")
        return (
            pd.concat(real_parts, ignore_index=True),
            pd.concat(synth_parts, ignore_index=True),
            {"sampling": "balanced_by_label", "per_label_counts_per_domain": per_label},
        )

    n = min(len(real), len(synth))
    if config.get("max_rows_per_domain") is not None:
        n = min(n, config["max_rows_per_domain"])
    return (
        real.sample(n=n, random_state=config["random_state"]).reset_index(drop=True),
        synth.sample(n=n, random_state=config["random_state"]).reset_index(drop=True),
        {"sampling": "balanced_by_domain", "rows_per_domain": int(n)},
    )


def pool_hidden(hidden: torch.Tensor, attention_mask: torch.Tensor, position: str) -> torch.Tensor:
    if position == "cls":
        return hidden[:, 0, :]
    mask = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


def resolve_hidden_state_index(layer_index: int, n_hidden_states: int) -> int:
    # hidden_states[0] is the embedding output; hidden_states[i + 1] is encoder layer i.
    if layer_index < 0:
        return n_hidden_states + layer_index
    return layer_index + 1


def mean_hidden_state(
    frame: pd.DataFrame,
    tokenizer: Any,
    model: torch.nn.Module,
    device: torch.device,
    config: dict[str, Any],
    domain: str,
) -> tuple[np.ndarray, int]:
    texts = frame[config["text_column"]].tolist()
    batches = math.ceil(len(texts) / config["batch_size"]) if texts else 0
    running_sum: torch.Tensor | None = None
    count = 0

    model.eval()
    with torch.no_grad():
        for batch_idx, start in enumerate(range(0, len(texts), config["batch_size"]), start=1):
            batch_texts = texts[start : start + config["batch_size"]]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=config["max_length"],
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**encoded, output_hidden_states=True)
            hidden_index = resolve_hidden_state_index(config["layer_index"], len(outputs.hidden_states))
            if hidden_index < 0 or hidden_index >= len(outputs.hidden_states):
                raise ValueError(
                    f"Requested layer_index={config['layer_index']} maps to hidden_states[{hidden_index}], "
                    f"but model returned {len(outputs.hidden_states)} hidden-state tensors"
                )
            pooled = pool_hidden(outputs.hidden_states[hidden_index], encoded["attention_mask"], config["position"])
            batch_sum = pooled.detach().sum(dim=0).to(torch.float64)
            running_sum = batch_sum if running_sum is None else running_sum + batch_sum
            count += pooled.shape[0]

            if batch_idx == 1 or batch_idx % config["progress_every"] == 0 or batch_idx == batches:
                print(f"{domain}: processed batch {batch_idx}/{batches} ({count}/{len(texts)} rows)", flush=True)

    if running_sum is None or count == 0:
        raise ValueError(f"No rows available for {domain}")
    return (running_sum / count).cpu().numpy().astype(np.float32), count


def write_json(data: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)


def run(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(config["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    real = load_frame(config["real_input_file"], config["text_column"], "real")
    synth = load_frame(config["synthetic_input_file"], config["text_column"], "synthetic")
    real, synth, sampling_summary = sample_balanced_frames(real, synth, config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model for hidden states: {config['model_name_or_path']}", flush=True)
    print(f"Device: {device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])
    model = AutoModel.from_pretrained(config["model_name_or_path"]).to(device)

    real_mean, real_count = mean_hidden_state(real, tokenizer, model, device, config, "real")
    synth_mean, synth_count = mean_hidden_state(synth, tokenizer, model, device, config, "synthetic")
    direction = real_mean - synth_mean

    np.save(output_dir / "real_hidden_centroid.npy", real_mean, allow_pickle=False)
    np.save(output_dir / "synthetic_hidden_centroid.npy", synth_mean, allow_pickle=False)
    np.save(output_dir / "activation_steering_vector.npy", direction.astype(np.float32), allow_pickle=False)

    summary = {
        "model_name_or_path": config["model_name_or_path"],
        "real_input_file": config["real_input_file"],
        "synthetic_input_file": config["synthetic_input_file"],
        "text_column": config["text_column"],
        "label_column": config.get("label_column"),
        "layer_index": config["layer_index"],
        "position": config["position"],
        "max_length": config["max_length"],
        "batch_size": config["batch_size"],
        "device": str(device),
        "real_count": int(real_count),
        "synthetic_count": int(synth_count),
        "direction_dimension": int(direction.shape[0]),
        "direction_norm": float(np.linalg.norm(direction)),
        "centroid_cosine_similarity": float(
            np.dot(real_mean, synth_mean) / (np.linalg.norm(real_mean) * np.linalg.norm(synth_mean) + 1e-12)
        ),
        **sampling_summary,
        "outputs": {
            "real_hidden_centroid": "real_hidden_centroid.npy",
            "synthetic_hidden_centroid": "synthetic_hidden_centroid.npy",
            "activation_steering_vector": "activation_steering_vector.npy",
        },
    }
    write_json(summary, output_dir / "activation_steering_vector_summary.json")
    write_json(config, output_dir / "activation_steering_vector_config.json")
    return summary


def main() -> None:
    args = parse_args()
    summary = run(load_config(args.config))
    print("Activation steering vector computed.")
    print(f"Layer index: {summary['layer_index']}")
    print(f"Position: {summary['position']}")
    print(f"Rows per domain: {summary['real_count']}")
    print(f"Direction norm: {summary['direction_norm']:.4f}")
    print(f"Outputs saved to: {Path(summary['outputs']['activation_steering_vector']).name}")


if __name__ == "__main__":
    main()
