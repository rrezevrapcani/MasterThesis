#!/usr/bin/env python3
"""Run PCA over saved embedding matrices and aligned metadata."""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

REQUIRED_CONFIG_KEYS = (
    "embeddings_file",
    "metadata_file",
    "output_dir",
    "n_components",
)


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8-sig") as stream:
        config = yaml.safe_load(stream) or {}

    missing = [key for key in REQUIRED_CONFIG_KEYS if config.get(key) is None]
    if missing:
        raise ValueError(f"Missing required config value(s): {', '.join(missing)}")

    config["n_components"] = int(config["n_components"])
    if config["n_components"] <= 0:
        raise ValueError("n_components must be a positive integer")

    config["standardize"] = bool(config.get("standardize", False))
    config["random_state"] = int(config.get("random_state", 42))
    config["plots"] = config.get("plots") or {}
    return config


def load_inputs(config: dict[str, Any]) -> tuple[np.ndarray, pd.DataFrame]:
    embeddings_path = Path(config["embeddings_file"]).expanduser()
    metadata_path = Path(config["metadata_file"]).expanduser()

    if not embeddings_path.is_file():
        raise FileNotFoundError(f"Embedding matrix not found: {embeddings_path.resolve()}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_path.resolve()}")

    embeddings = np.load(embeddings_path)
    metadata = pd.read_csv(metadata_path)

    if embeddings.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix, got shape {embeddings.shape}")
    if len(metadata) != embeddings.shape[0]:
        raise ValueError(
            "Embedding and metadata row counts do not match: "
            f"{embeddings.shape[0]} embeddings vs {len(metadata)} metadata rows"
        )
    if np.isnan(embeddings).any():
        raise ValueError("Embedding matrix contains NaN values")
    if np.isinf(embeddings).any():
        raise ValueError("Embedding matrix contains infinite values")

    return embeddings.astype(np.float32, copy=False), metadata


def prepare_matrix(embeddings: np.ndarray, standardize: bool) -> tuple[np.ndarray, dict[str, Any]]:
    if not standardize:
        return embeddings, {"standardized": False}

    scaler = StandardScaler()
    transformed = scaler.fit_transform(embeddings).astype(np.float32, copy=False)
    return transformed, {
        "standardized": True,
        "scaler_mean_shape": list(scaler.mean_.shape),
        "scaler_scale_shape": list(scaler.scale_.shape),
    }


def fit_pca(matrix: np.ndarray, n_components: int, random_state: int) -> tuple[np.ndarray, PCA]:
    max_components = min(matrix.shape)
    if n_components > max_components:
        warnings.warn(
            f"Requested n_components={n_components}, but the maximum possible value is "
            f"{max_components}; using {max_components} instead."
        )
        n_components = max_components

    pca = PCA(n_components=n_components, random_state=random_state)
    transformed = pca.fit_transform(matrix).astype(np.float32, copy=False)
    return transformed, pca


def build_pca_metadata(metadata: pd.DataFrame, transformed: np.ndarray) -> pd.DataFrame:
    pca_columns = {
        f"PC{index + 1}": transformed[:, index]
        for index in range(transformed.shape[1])
    }
    return pd.concat([metadata.reset_index(drop=True), pd.DataFrame(pca_columns)], axis=1)


def build_explained_variance(pca: PCA) -> pd.DataFrame:
    ratios = pca.explained_variance_ratio_
    return pd.DataFrame(
        {
            "component": np.arange(1, len(ratios) + 1, dtype=np.int64),
            "explained_variance": pca.explained_variance_,
            "explained_variance_ratio": ratios,
            "cumulative_explained_variance_ratio": np.cumsum(ratios),
        }
    )


def write_json(data: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)


def save_scree_plot(explained: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(explained["component"], explained["explained_variance_ratio"], color="#4c78a8")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("PCA explained variance by component")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_cumulative_variance_plot(explained: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        explained["component"],
        explained["cumulative_explained_variance_ratio"],
        marker="o",
        linewidth=1.5,
        markersize=3,
        color="#f58518",
    )
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative explained variance ratio")
    ax.set_title("PCA cumulative explained variance")
    ax.set_ylim(0, min(1.0, max(0.05, explained["cumulative_explained_variance_ratio"].max() * 1.05)))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def sample_for_plot(frame: pd.DataFrame, max_points: int, random_state: int) -> pd.DataFrame:
    if len(frame) <= max_points:
        return frame
    return frame.sample(n=max_points, random_state=random_state)


def save_scatter_plot(
    pca_metadata: pd.DataFrame,
    output_path: Path,
    color_column: str | None,
    config: dict[str, Any],
) -> None:
    if "PC1" not in pca_metadata.columns or "PC2" not in pca_metadata.columns:
        warnings.warn("Skipping PC1/PC2 scatter plot because fewer than two PCA components are available")
        return

    plots = config.get("plots") or {}
    max_points = int(plots.get("max_points", 10000))
    max_classes = int(plots.get("max_classes", 12))
    point_size = float(plots.get("point_size", 10))
    alpha = float(plots.get("alpha", 0.65))
    frame = sample_for_plot(pca_metadata, max_points=max_points, random_state=config["random_state"])

    fig, ax = plt.subplots(figsize=(8, 6))
    if color_column and color_column in frame.columns:
        labels = frame[color_column].fillna("Missing").astype(str)
        top_labels = labels.value_counts().head(max_classes).index.tolist()
        other_mask = ~labels.isin(top_labels)
        plot_labels = labels.mask(other_mask, "Other")

        cmap = plt.get_cmap("tab20")
        for index, label in enumerate(plot_labels.value_counts().index):
            mask = plot_labels == label
            ax.scatter(
                frame.loc[mask, "PC1"],
                frame.loc[mask, "PC2"],
                s=point_size,
                alpha=alpha,
                label=label,
                color=cmap(index % cmap.N),
                edgecolors="none",
            )
        ax.legend(markerscale=2, fontsize=8, frameon=False, loc="best")
        title_suffix = f" by {color_column}"
    else:
        ax.scatter(frame["PC1"], frame["PC2"], s=point_size, alpha=alpha, edgecolors="none")
        title_suffix = ""

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"PCA projection{title_suffix}")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def run(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(config["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings, metadata = load_inputs(config)
    matrix, preprocessing_summary = prepare_matrix(embeddings, config["standardize"])
    transformed, pca = fit_pca(matrix, config["n_components"], config["random_state"])

    pca_metadata = build_pca_metadata(metadata, transformed)
    explained = build_explained_variance(pca)

    np.save(output_dir / "pca_embeddings.npy", transformed, allow_pickle=False)
    pca_metadata.to_csv(output_dir / "pca_metadata.csv", index=False)
    explained.to_csv(output_dir / "explained_variance.csv", index=False)
    write_json(config, output_dir / "pca_config.json")

    label_column = config.get("label_column")
    save_scree_plot(explained, output_dir / "explained_variance.png")
    save_cumulative_variance_plot(explained, output_dir / "cumulative_explained_variance.png")
    scatter_plot_name = None
    if transformed.shape[1] >= 2:
        scatter_plot_name = "pca_pc1_pc2_by_relation_type.png"
        save_scatter_plot(
            pca_metadata,
            output_dir / scatter_plot_name,
            label_column if isinstance(label_column, str) else None,
            config,
        )

    summary = {
        "number_of_rows": int(embeddings.shape[0]),
        "original_embedding_dimension": int(embeddings.shape[1]),
        "pca_components": int(transformed.shape[1]),
        "standardize": bool(config["standardize"]),
        "explained_variance_ratio_first_component": float(explained.loc[0, "explained_variance_ratio"]),
        "cumulative_explained_variance_ratio": float(explained["cumulative_explained_variance_ratio"].iloc[-1]),
        "outputs": {
            "pca_embeddings": "pca_embeddings.npy",
            "pca_metadata": "pca_metadata.csv",
            "explained_variance": "explained_variance.csv",
            "explained_variance_plot": "explained_variance.png",
            "cumulative_explained_variance_plot": "cumulative_explained_variance.png",
            "scatter_plot": scatter_plot_name,
        },
        "preprocessing": preprocessing_summary,
    }
    write_json(summary, output_dir / "pca_summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Path to PCA YAML config")
    args = parser.parse_args()

    summary = run(load_config(args.config))
    print(
        "PCA complete: "
        f"{summary['number_of_rows']} rows, "
        f"{summary['original_embedding_dimension']} -> {summary['pca_components']} dimensions, "
        f"cumulative explained variance={summary['cumulative_explained_variance_ratio']:.4f}."
    )


if __name__ == "__main__":
    main()
