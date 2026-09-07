#!/usr/bin/env python3
"""Apply simple real-minus-synthetic steering vectors to synthetic embeddings.

This is an embedding-space diagnostic step. It does not retrain PubMedBERT and
does not change the generated text. For several alpha values, it shifts
synthetic embeddings toward the real embedding centroid and evaluates whether
real-vs-synthetic domain separability decreases.
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
    config["standardize_probe"] = bool(config.get("standardize_probe", True))
    config["balance_by_label"] = bool(config.get("balance_by_label", True))
    config["max_rows_per_domain"] = config.get("max_rows_per_domain")
    if config["max_rows_per_domain"] is not None:
        config["max_rows_per_domain"] = int(config["max_rows_per_domain"])
    config["steering_mode"] = str(config.get("steering_mode", "overall_centroid"))
    if config["steering_mode"] not in {"overall_centroid", "per_label_centroid"}:
        raise ValueError("steering_mode must be 'overall_centroid' or 'per_label_centroid'")
    config["alphas"] = [float(alpha) for alpha in config.get("alphas", [0.0, 0.5, 1.0])]
    config["save_steered_embeddings"] = bool(config.get("save_steered_embeddings", False))
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
        if not real_parts:
            raise ValueError("No shared relation labels found for balanced sampling")
        return (
            np.concatenate(real_parts),
            np.concatenate(synth_parts),
            {"sampling": "balanced_by_label", "per_label_counts_per_domain": per_label},
        )

    n = min(len(real_meta), len(synth_meta))
    if config.get("max_rows_per_domain") is not None:
        n = min(n, config["max_rows_per_domain"])
    return (
        rng.choice(real_meta.index.to_numpy(), size=n, replace=False),
        rng.choice(synth_meta.index.to_numpy(), size=n, replace=False),
        {"sampling": "balanced_by_domain", "rows_per_domain": int(n)},
    )


def fit_domain_probe(
    real_x: np.ndarray,
    synth_x: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    x = np.vstack([real_x, synth_x])
    y = np.array([0] * len(real_x) + [1] * len(synth_x), dtype=np.int64)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=config["test_size"],
        random_state=config["random_state"],
        stratify=y,
    )

    scaler = StandardScaler() if config["standardize_probe"] else None
    if scaler is not None:
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=config["random_state"])
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    prob = clf.predict_proba(x_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision_synthetic": float(precision_score(y_test, pred, zero_division=0)),
        "recall_synthetic": float(recall_score(y_test, pred, zero_division=0)),
        "f1_synthetic": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, prob)),
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
    }


def build_overall_direction(real_x: np.ndarray, synth_x: np.ndarray) -> np.ndarray:
    return (real_x.mean(axis=0) - synth_x.mean(axis=0)).astype(np.float32)


def build_per_label_directions(
    real_x: np.ndarray,
    real_meta: pd.DataFrame,
    synth_x: np.ndarray,
    synth_meta: pd.DataFrame,
    label_column: str,
) -> dict[str, np.ndarray]:
    directions: dict[str, np.ndarray] = {}
    common_labels = sorted(set(real_meta[label_column].dropna()) & set(synth_meta[label_column].dropna()))
    for label in common_labels:
        real_mask = (real_meta[label_column] == label).to_numpy()
        synth_mask = (synth_meta[label_column] == label).to_numpy()
        if real_mask.sum() == 0 or synth_mask.sum() == 0:
            continue
        directions[str(label)] = (real_x[real_mask].mean(axis=0) - synth_x[synth_mask].mean(axis=0)).astype(np.float32)
    return directions


def apply_steering(
    synth_x: np.ndarray,
    synth_meta: pd.DataFrame,
    config: dict[str, Any],
    overall_direction: np.ndarray,
    per_label_directions: dict[str, np.ndarray],
    alpha: float,
) -> np.ndarray:
    if config["steering_mode"] == "overall_centroid":
        return (synth_x + alpha * overall_direction).astype(np.float32)

    label_column = config.get("label_column", "relation_type")
    steered = synth_x.copy()
    for label, direction in per_label_directions.items():
        mask = (synth_meta[label_column].astype(str) == label).to_numpy()
        steered[mask] = steered[mask] + alpha * direction
    return steered.astype(np.float32)


def centroid_metrics(real_x: np.ndarray, synth_x: np.ndarray) -> dict[str, float]:
    real_centroid = real_x.mean(axis=0)
    synth_centroid = synth_x.mean(axis=0)
    direction = real_centroid - synth_centroid
    return {
        "centroid_euclidean_distance": float(np.linalg.norm(direction)),
        "centroid_cosine_similarity": float(
            np.dot(real_centroid, synth_centroid)
            / (np.linalg.norm(real_centroid) * np.linalg.norm(synth_centroid) + 1e-12)
        ),
        "real_centroid_norm": float(np.linalg.norm(real_centroid)),
        "synthetic_centroid_norm": float(np.linalg.norm(synth_centroid)),
    }


def plot_pca(
    real_x: np.ndarray,
    synth_x: np.ndarray,
    real_meta: pd.DataFrame,
    synth_meta: pd.DataFrame,
    config: dict[str, Any],
    output_path: Path,
    alpha_value: float,
) -> None:
    pca_config = config.get("pca") or {}
    max_points = int(pca_config.get("max_points_per_domain", 10000))
    rng = np.random.default_rng(config["random_state"])

    real_idx = np.arange(len(real_x))
    synth_idx = np.arange(len(synth_x))
    if len(real_idx) > max_points:
        real_idx = rng.choice(real_idx, size=max_points, replace=False)
    if len(synth_idx) > max_points:
        synth_idx = rng.choice(synth_idx, size=max_points, replace=False)

    x_plot = np.vstack([real_x[real_idx], synth_x[synth_idx]])
    domain = np.array(["real"] * len(real_idx) + ["synthetic"] * len(synth_idx))
    pcs = PCA(n_components=2, random_state=config["random_state"]).fit_transform(x_plot)

    plot_alpha = float(pca_config.get("alpha", 0.45))
    point_size = float(pca_config.get("point_size", 8))
    colors = {"real": "#1f77b4", "synthetic": "#ff7f0e"}

    fig, ax = plt.subplots(figsize=(9, 6))
    for name in ["real", "synthetic"]:
        mask = domain == name
        ax.scatter(pcs[mask, 0], pcs[mask, 1], s=point_size, alpha=plot_alpha, label=name, c=colors[name])
    ax.set_title(f"Real vs steered synthetic embeddings (alpha={alpha_value:g})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_json(data: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)


def safe_alpha_name(alpha: float) -> str:
    return f"{alpha:.2f}".replace("-", "m").replace(".", "p")


def run(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(config["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    real_all, real_meta_all = load_embedding_set(config["real_embeddings_file"], config["real_metadata_file"], "real")
    synth_all, synth_meta_all = load_embedding_set(
        config["synthetic_embeddings_file"], config["synthetic_metadata_file"], "synthetic"
    )
    real_idx, synth_idx, sampling_summary = sample_indices(real_meta_all, synth_meta_all, config)

    real_x = real_all[real_idx]
    synth_x = synth_all[synth_idx]
    real_meta = real_meta_all.iloc[real_idx].reset_index(drop=True)
    synth_meta = synth_meta_all.iloc[synth_idx].reset_index(drop=True)

    overall_direction = build_overall_direction(real_x, synth_x)
    np.save(output_dir / "overall_real_minus_synthetic_direction.npy", overall_direction, allow_pickle=False)

    label_column = config.get("label_column", "relation_type")
    per_label_directions: dict[str, np.ndarray] = {}
    if label_column in real_meta.columns and label_column in synth_meta.columns:
        per_label_directions = build_per_label_directions(real_x, real_meta, synth_x, synth_meta, label_column)
        per_label_dir_dir = output_dir / "per_label_directions"
        per_label_dir_dir.mkdir(exist_ok=True)
        for label, direction in per_label_directions.items():
            safe_label = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in label)
            np.save(per_label_dir_dir / f"{safe_label}_real_minus_synthetic_direction.npy", direction, allow_pickle=False)

    rows: list[dict[str, Any]] = []
    for alpha in config["alphas"]:
        steered_synth = apply_steering(
            synth_x,
            synth_meta,
            config,
            overall_direction,
            per_label_directions,
            alpha,
        )
        probe = fit_domain_probe(real_x, steered_synth, config)
        centroids = centroid_metrics(real_x, steered_synth)
        alpha_name = safe_alpha_name(alpha)
        plot_pca(
            real_x,
            steered_synth,
            real_meta,
            synth_meta,
            config,
            output_dir / f"pca_real_vs_steered_synthetic_alpha_{alpha_name}.png",
            alpha,
        )
        if config["save_steered_embeddings"]:
            np.save(output_dir / f"synthetic_embeddings_steered_alpha_{alpha_name}.npy", steered_synth, allow_pickle=False)
        rows.append(
            {
                "alpha": alpha,
                **probe,
                **centroids,
            }
        )

    summary_table = pd.DataFrame(rows)
    summary_table.to_csv(output_dir / "steering_summary.csv", index=False)

    summary = {
        "real_embeddings_file": config["real_embeddings_file"],
        "synthetic_embeddings_file": config["synthetic_embeddings_file"],
        "output_dir": config["output_dir"],
        "steering_mode": config["steering_mode"],
        "label_column": label_column,
        "alphas": config["alphas"],
        "data": {
            "real_embeddings_shape": list(real_all.shape),
            "synthetic_embeddings_shape": list(synth_all.shape),
            "analysis_rows_real": int(len(real_x)),
            "analysis_rows_synthetic": int(len(synth_x)),
            **sampling_summary,
        },
        "best_by_centroid_distance": summary_table.loc[
            summary_table["centroid_euclidean_distance"].idxmin()
        ].to_dict(),
        "best_by_domain_auc": summary_table.loc[summary_table["roc_auc"].idxmin()].to_dict(),
        "outputs": {
            "steering_summary": "steering_summary.csv",
            "overall_direction": "overall_real_minus_synthetic_direction.npy",
            "per_label_directions": "per_label_directions/",
            "pca_plots": "pca_real_vs_steered_synthetic_alpha_*.png",
        },
        "interpretation_hint": (
            "Promising alpha values should reduce centroid distance and, ideally, reduce domain-probe "
            "accuracy/ROC-AUC. If ROC-AUC remains near 1.0, a simple global steering direction is not enough."
        ),
    }
    write_json(summary, output_dir / "steering_summary.json")
    return summary


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    summary = run(config)
    best_centroid = summary["best_by_centroid_distance"]
    best_auc = summary["best_by_domain_auc"]
    print("Steering-vector analysis complete.")
    print(f"Mode: {summary['steering_mode']}")
    print(f"Rows per domain: {summary['data']['analysis_rows_real']}")
    print(
        "Best centroid alpha: "
        f"{best_centroid['alpha']:.2f} "
        f"(distance={best_centroid['centroid_euclidean_distance']:.4f}, "
        f"AUC={best_centroid['roc_auc']:.4f})"
    )
    print(
        "Best domain-AUC alpha: "
        f"{best_auc['alpha']:.2f} "
        f"(AUC={best_auc['roc_auc']:.4f}, "
        f"accuracy={best_auc['accuracy']:.4f})"
    )
    print(f"Outputs saved to: {config['output_dir']}")


if __name__ == "__main__":
    main()
