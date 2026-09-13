#!/usr/bin/env python3
"""Analyze whether real and synthetic embeddings are separable.

This is a diagnostic analysis. It compares
PubMedBERT embeddings from real and synthetic relation candidates to estimate
domain shift and to produce a candidate real-vs-synthetic direction vector.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


REQUIRED_CONFIG_KEYS = (
    "real_embeddings_file",
    "real_metadata_file",
    "synthetic_embeddings_file",
    "synthetic_metadata_file",
    "output_dir",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8-sig") as stream:
        config = yaml.safe_load(stream) or {}
    missing = [key for key in REQUIRED_CONFIG_KEYS if config.get(key) is None]
    if missing:
        raise ValueError(f"Missing required config value(s): {', '.join(missing)}")
    config["random_state"] = int(config.get("random_state", 42))
    config["test_size"] = float(config.get("test_size", 0.2))
    config["standardize"] = bool(config.get("standardize", True))
    config["balance_by_label"] = bool(config.get("balance_by_label", True))
    config["max_rows_per_domain"] = config.get("max_rows_per_domain")
    if config["max_rows_per_domain"] is not None:
        config["max_rows_per_domain"] = int(config["max_rows_per_domain"])
    config["pca"] = config.get("pca") or {}
    return config


def load_embedding_set(embeddings_file: str, metadata_file: str, domain: str) -> tuple[np.ndarray, pd.DataFrame]:
    embeddings = np.load(Path(embeddings_file).expanduser())
    metadata = pd.read_csv(Path(metadata_file).expanduser())
    if embeddings.ndim != 2:
        raise ValueError(f"{embeddings_file} must contain a 2D array, got {embeddings.shape}")
    if embeddings.shape[0] != len(metadata):
        raise ValueError(
            f"Row mismatch for {domain}: {embeddings.shape[0]} embeddings vs {len(metadata)} metadata rows"
        )
    if np.isnan(embeddings).any() or np.isinf(embeddings).any():
        raise ValueError(f"{domain} embeddings contain NaN or infinite values")
    metadata = metadata.copy()
    metadata["domain"] = domain
    return embeddings.astype(np.float32, copy=False), metadata


def sample_indices(
    real_meta: pd.DataFrame,
    synth_meta: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(config["random_state"])
    label_column = config.get("label_column", "relation_type")

    if config["balance_by_label"] and label_column in real_meta.columns and label_column in synth_meta.columns:
        real_parts: list[np.ndarray] = []
        synth_parts: list[np.ndarray] = []
        per_label: dict[str, int] = {}
        common_labels = sorted(set(real_meta[label_column].dropna()) & set(synth_meta[label_column].dropna()))
        for label in common_labels:
            real_idx = real_meta.index[real_meta[label_column] == label].to_numpy()
            synth_idx = synth_meta.index[synth_meta[label_column] == label].to_numpy()
            n = min(len(real_idx), len(synth_idx))
            if n == 0:
                continue
            real_parts.append(rng.choice(real_idx, size=n, replace=False))
            synth_parts.append(rng.choice(synth_idx, size=n, replace=False))
            per_label[str(label)] = int(n)
        return (
            np.concatenate(real_parts),
            np.concatenate(synth_parts),
            {"sampling": "balanced_by_label", "per_label_counts_per_domain": per_label},
        )

    max_rows = config.get("max_rows_per_domain")
    n = min(len(real_meta), len(synth_meta))
    if max_rows is not None:
        n = min(n, max_rows)
    return (
        rng.choice(real_meta.index.to_numpy(), size=n, replace=False),
        rng.choice(synth_meta.index.to_numpy(), size=n, replace=False),
        {"sampling": "balanced_by_domain", "rows_per_domain": int(n)},
    )


def build_combined_dataset(config: dict[str, Any]) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    real_x, real_meta = load_embedding_set(config["real_embeddings_file"], config["real_metadata_file"], "real")
    synth_x, synth_meta = load_embedding_set(
        config["synthetic_embeddings_file"], config["synthetic_metadata_file"], "synthetic"
    )
    real_idx, synth_idx, sampling_summary = sample_indices(real_meta, synth_meta, config)
    x = np.vstack([real_x[real_idx], synth_x[synth_idx]])
    meta = pd.concat(
        [real_meta.iloc[real_idx].reset_index(drop=True), synth_meta.iloc[synth_idx].reset_index(drop=True)],
        ignore_index=True,
    )
    y = (meta["domain"] == "synthetic").astype(int).to_numpy()
    meta["domain_label"] = y
    summary = {
        "real_embeddings_shape": list(real_x.shape),
        "synthetic_embeddings_shape": list(synth_x.shape),
        "analysis_rows": int(len(meta)),
        "analysis_rows_real": int((meta["domain"] == "real").sum()),
        "analysis_rows_synthetic": int((meta["domain"] == "synthetic").sum()),
        **sampling_summary,
    }
    return x, meta, summary


def fit_domain_probe(x: np.ndarray, meta: pd.DataFrame, config: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    y = meta["domain_label"].to_numpy()
    x_train, x_test, y_train, y_test, meta_train, meta_test = train_test_split(
        x,
        y,
        meta,
        test_size=config["test_size"],
        random_state=config["random_state"],
        stratify=y,
    )
    scaler = StandardScaler() if config["standardize"] else None
    if scaler is not None:
        x_train_model = scaler.fit_transform(x_train)
        x_test_model = scaler.transform(x_test)
    else:
        x_train_model = x_train
        x_test_model = x_test

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=config["random_state"])
    clf.fit(x_train_model, y_train)
    pred = clf.predict(x_test_model)
    prob = clf.predict_proba(x_test_model)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision_synthetic": float(precision_score(y_test, pred, zero_division=0)),
        "recall_synthetic": float(recall_score(y_test, pred, zero_division=0)),
        "f1_synthetic": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, prob)),
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "standardized": bool(config["standardize"]),
    }

    predictions = meta_test.reset_index(drop=True).copy()
    predictions["domain_true"] = np.where(y_test == 1, "synthetic", "real")
    predictions["domain_pred"] = np.where(pred == 1, "synthetic", "real")
    predictions["synthetic_probability"] = prob
    return metrics, predictions


def compute_centroid_analysis(x: np.ndarray, meta: pd.DataFrame, config: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    real_x = x[meta["domain"].to_numpy() == "real"]
    synth_x = x[meta["domain"].to_numpy() == "synthetic"]
    real_centroid = real_x.mean(axis=0)
    synth_centroid = synth_x.mean(axis=0)
    direction = real_centroid - synth_centroid
    centroid_distance = float(np.linalg.norm(direction))
    cosine_similarity = float(
        np.dot(real_centroid, synth_centroid)
        / (np.linalg.norm(real_centroid) * np.linalg.norm(synth_centroid) + 1e-12)
    )
    summary = {
        "real_centroid_norm": float(np.linalg.norm(real_centroid)),
        "synthetic_centroid_norm": float(np.linalg.norm(synth_centroid)),
        "real_minus_synthetic_direction_norm": float(np.linalg.norm(direction)),
        "centroid_euclidean_distance": centroid_distance,
        "centroid_cosine_similarity": cosine_similarity,
    }

    label_column = config.get("label_column", "relation_type")
    rows: list[dict[str, Any]] = []
    if label_column in meta.columns:
        for label, group in meta.groupby(label_column):
            real_mask = (meta[label_column] == label) & (meta["domain"] == "real")
            synth_mask = (meta[label_column] == label) & (meta["domain"] == "synthetic")
            if real_mask.sum() == 0 or synth_mask.sum() == 0:
                continue
            real_label_centroid = x[real_mask.to_numpy()].mean(axis=0)
            synth_label_centroid = x[synth_mask.to_numpy()].mean(axis=0)
            label_direction = real_label_centroid - synth_label_centroid
            rows.append(
                {
                    label_column: label,
                    "real_count": int(real_mask.sum()),
                    "synthetic_count": int(synth_mask.sum()),
                    "centroid_euclidean_distance": float(np.linalg.norm(label_direction)),
                    "centroid_cosine_similarity": float(
                        np.dot(real_label_centroid, synth_label_centroid)
                        / (np.linalg.norm(real_label_centroid) * np.linalg.norm(synth_label_centroid) + 1e-12)
                    ),
                }
            )
    return summary, pd.DataFrame(rows), direction.astype(np.float32)


def plot_combined_pca(x: np.ndarray, meta: pd.DataFrame, config: dict[str, Any], output_dir: Path) -> pd.DataFrame:
    pca_config = config.get("pca", {})
    rng = np.random.default_rng(config["random_state"])
    max_points = int(pca_config.get("max_points_per_domain", 10000))
    plot_indices: list[np.ndarray] = []
    for domain, group in meta.groupby("domain"):
        idx = group.index.to_numpy()
        if len(idx) > max_points:
            idx = rng.choice(idx, size=max_points, replace=False)
        plot_indices.append(idx)
    selected = np.concatenate(plot_indices)
    x_plot = x[selected]
    meta_plot = meta.iloc[selected].reset_index(drop=True).copy()

    pca = PCA(n_components=int(pca_config.get("n_components", 2)), random_state=config["random_state"])
    pcs = pca.fit_transform(x_plot)
    meta_plot["PC1"] = pcs[:, 0]
    meta_plot["PC2"] = pcs[:, 1]

    alpha = float(pca_config.get("alpha", 0.45))
    point_size = float(pca_config.get("point_size", 8))
    colors = {"real": "#1f77b4", "synthetic": "#ff7f0e"}
    plt.figure(figsize=(9, 6))
    for domain in ["real", "synthetic"]:
        group = meta_plot[meta_plot["domain"] == domain]
        plt.scatter(group["PC1"], group["PC2"], s=point_size, alpha=alpha, label=domain, c=colors[domain])
    plt.title("PCA projection: real vs synthetic embeddings")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_dir / "pca_real_vs_synthetic.png", dpi=200)
    plt.close()

    return meta_plot


def write_json(data: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)


def run(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(config["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    x, meta, data_summary = build_combined_dataset(config)

    probe_metrics, probe_predictions = fit_domain_probe(x, meta, config)
    centroid_summary, centroid_by_label, direction = compute_centroid_analysis(x, meta, config)
    pca_metadata = plot_combined_pca(x, meta, config, output_dir)

    probe_predictions.to_csv(output_dir / "domain_probe_predictions.csv", index=False)
    centroid_by_label.to_csv(output_dir / "centroid_distances_by_label.csv", index=False)
    pca_metadata.to_csv(output_dir / "combined_pca_domain_metadata.csv", index=False)
    np.save(output_dir / "real_minus_synthetic_direction.npy", direction)

    label_column = config.get("label_column", "relation_type")
    per_label_probe: dict[str, Any] = {}
    if label_column in probe_predictions.columns:
        for label, group in probe_predictions.groupby(label_column):
            per_label_probe[str(label)] = {
                "rows": int(len(group)),
                "domain_accuracy": float((group["domain_true"] == group["domain_pred"]).mean()),
                "mean_synthetic_probability": float(group["synthetic_probability"].mean()),
            }

    summary = {
        "real_embeddings_file": config["real_embeddings_file"],
        "synthetic_embeddings_file": config["synthetic_embeddings_file"],
        "label_column": label_column,
        "data": data_summary,
        "domain_probe": probe_metrics,
        "domain_probe_by_label": per_label_probe,
        "centroids": centroid_summary,
        "interpretation_hint": (
            "If domain-probe accuracy/AUC are high, real and synthetic embeddings are systematically separable. "
            "The saved real_minus_synthetic_direction.npy can be treated as an initial domain-shift direction "
            "to motivate steering-vector experiments."
        ),
        "outputs": {
            "domain_probe_predictions": "domain_probe_predictions.csv",
            "centroid_distances_by_label": "centroid_distances_by_label.csv",
            "combined_pca_domain_metadata": "combined_pca_domain_metadata.csv",
            "pca_real_vs_synthetic": "pca_real_vs_synthetic.png",
            "real_minus_synthetic_direction": "real_minus_synthetic_direction.npy",
        },
    }
    write_json(summary, output_dir / "domain_separation_summary.json")
    return summary


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    summary = run(config)
    print("Domain separation analysis complete.")
    print(f"Analysis rows: {summary['data']['analysis_rows']}")
    print(f"Domain probe accuracy: {summary['domain_probe']['accuracy']:.4f}")
    print(f"Domain probe ROC-AUC: {summary['domain_probe']['roc_auc']:.4f}")
    print(f"Centroid distance: {summary['centroids']['centroid_euclidean_distance']:.4f}")
    print(f"Outputs saved to: {config['output_dir']}")


if __name__ == "__main__":
    main()
