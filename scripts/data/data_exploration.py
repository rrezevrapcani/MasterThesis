#!/usr/bin/env python3
"""Reusable, config-driven exploration of relation-extraction CSV data."""
from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

NO_RELATION = "No_Relation"
ENTITY_MARKER_RE = re.compile(r"\[/?[A-Z0-9]+_(?:SRC|TGT)\]")
SRC_MARKER_RE = re.compile(r"\[[A-Z0-9]+_SRC\]")
TGT_MARKER_RE = re.compile(r"\[[A-Z0-9]+_TGT\]")


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8-sig") as stream:
        config = yaml.safe_load(stream) or {}
    missing = [key for key in ("input_file", "output_dir") if not config.get(key)]
    if missing:
        raise ValueError(f"Missing config value(s): {', '.join(missing)}")
    return config


def get_series(df: pd.DataFrame, config: dict[str, Any], key: str) -> pd.Series | None:
    name = config.get(key)
    if not name or name not in df.columns:
        warnings.warn(f"Optional column unavailable for {key}: {name!r}")
        return None
    return df[name]


def clean(series: pd.Series) -> pd.Series:
    values = series.dropna().astype(str).str.strip()
    return values[values.ne("")]


def combine(series: list[pd.Series | None]) -> pd.Series | None:
    available = [clean(item) for item in series if item is not None]
    return pd.concat(available, ignore_index=True) if available else None


def distribution(series: pd.Series | None, category: str) -> pd.DataFrame:
    if series is None:
        return pd.DataFrame(columns=[category, "count", "percentage"])
    counts = clean(series).value_counts()
    table = counts.rename_axis(category).reset_index(name="count")
    table["percentage"] = table["count"].div(counts.sum()).mul(100) if counts.sum() else 0.0
    return table


def save_plot(table: pd.DataFrame, category: str, title: str, path: Path) -> None:
    width = max(8, min(18, 0.55 * max(len(table), 1)))
    fig, ax = plt.subplots(figsize=(width, 6))
    if table.empty:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
    else:
        ax.bar(table[category].astype(str), table["count"], color="#4472C4")
        ax.tick_params(axis="x", rotation=45)
        plt.setp(ax.get_xticklabels(), ha="right")
    ax.set(title=title, ylabel="Count")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def describe(values: pd.Series) -> dict[str, float | int | None]:
    values = values.dropna()
    if values.empty:
        return dict.fromkeys(("min", "max", "mean", "median", "std"))
    return {
        "min": int(np.min(values)),
        "max": int(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
    }


def length_statistics(text: pd.Series) -> dict[str, Any]:
    text = text.fillna("").astype(str)
    return {
        "character_length": describe(text.str.len()),
        "word_length": describe(text.str.split().str.len()),
    }


def valid_id_mask(series: pd.Series) -> pd.Series:
    return series.notna() & series.astype(str).str.strip().ne("")


def dataset_statistics(df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    labels = get_series(df, config, "label_column")
    docs = get_series(df, config, "doc_id_column")
    entity1_ids = get_series(df, config, "entity1_id_column")
    entity2_ids = get_series(df, config, "entity2_id_column")
    label_values = clean(labels) if labels is not None else pd.Series(dtype=str)
    negative = int(label_values.eq(NO_RELATION).sum())
    positive = int(label_values.ne(NO_RELATION).sum())

    summary = {
        "number_of_rows": int(len(df)),
        "number_of_unique_documents": int(docs.nunique(dropna=True)) if docs is not None else None,
        "number_of_relation_types": int(label_values.nunique()) if labels is not None else None,
        "number_of_positive_examples": positive if labels is not None else None,
        "number_of_no_relation_examples": negative if labels is not None else None,
        "positive_to_no_relation_ratio": positive / negative if labels is not None and negative else None,
    }

    if entity1_ids is not None and entity2_ids is not None:
        valid_pairs = valid_id_mask(entity1_ids) & valid_id_mask(entity2_ids)
        pair_frame = pd.DataFrame({
            "entity1_id": entity1_ids[valid_pairs].astype(str).str.strip(),
            "entity2_id": entity2_ids[valid_pairs].astype(str).str.strip(),
        })
        summary["number_of_unique_entity_pairs"] = int(pair_frame.drop_duplicates().shape[0])

        if docs is not None and labels is not None:
            valid_triples = valid_pairs & docs.notna() & labels.notna()
            triple_frame = pd.DataFrame({
                "doc_id": docs[valid_triples].astype(str).str.strip(),
                "entity1_id": entity1_ids[valid_triples].astype(str).str.strip(),
                "entity2_id": entity2_ids[valid_triples].astype(str).str.strip(),
                "relation_type": labels[valid_triples].astype(str).str.strip(),
            })
            nonempty = triple_frame.ne("").all(axis=1)
            summary["number_of_unique_relation_triples"] = int(
                triple_frame[nonempty].drop_duplicates().shape[0]
            )
        else:
            summary["number_of_unique_relation_triples"] = None
    else:
        summary["number_of_unique_entity_pairs"] = None
        summary["number_of_unique_relation_triples"] = None
    return summary


def relation_statistics(df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = get_series(df, config, "label_column")
    binary = clean(labels).map(lambda value: NO_RELATION if value == NO_RELATION else "Positive") if labels is not None else None
    return distribution(labels, "relation_type"), distribution(binary, "relation_group")


def build_entity_frame(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    frames = []
    for side in ("entity1", "entity2"):
        ids = get_series(df, config, f"{side}_id_column")
        if ids is None:
            continue
        types = get_series(df, config, f"{side}_type_column")
        mentions = get_series(df, config, f"{side}_text_column")
        frame = pd.DataFrame({
            "entity_id": ids,
            "entity_type": types if types is not None else pd.Series(pd.NA, index=df.index),
            "entity_mention": mentions if mentions is not None else pd.Series(pd.NA, index=df.index),
        })
        frame = frame[valid_id_mask(frame["entity_id"])].copy()
        frame["entity_id"] = frame["entity_id"].astype(str).str.strip()
        frame["entity_type"] = frame["entity_type"].fillna("").astype(str).str.strip()
        frame["entity_mention"] = frame["entity_mention"].fillna("").astype(str).str.strip()
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["entity_id", "entity_type", "entity_mention"]
    )


def representative_mention(values: pd.Series) -> str | None:
    values = values[values.ne("")]
    if values.empty:
        return None
    modes = values.mode()
    return str(modes.iloc[0])


def entity_statistics(df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    entity_frame = build_entity_frame(df, config)
    if entity_frame.empty:
        empty_types = distribution(None, "entity_type")
        top_entities = pd.DataFrame(columns=["entity_id", "entity_type", "representative_mention", "count"])
        return empty_types, top_entities, {
            "number_of_unique_entity_ids": None,
            "number_of_unique_entity_mentions": None,
        }

    entity_types = entity_frame.loc[entity_frame["entity_type"].ne(""), "entity_type"]
    mentions = entity_frame.loc[entity_frame["entity_mention"].ne(""), "entity_mention"]
    top_entities = (
        entity_frame.groupby(["entity_id", "entity_type"], dropna=False, sort=False)
        .agg(representative_mention=("entity_mention", representative_mention), count=("entity_id", "size"))
        .reset_index()
        .sort_values(["count", "entity_id"], ascending=[False, True])
        .head(50)
        .reset_index(drop=True)
    )
    summary = {
        "number_of_unique_entity_ids": int(entity_frame["entity_id"].nunique()),
        "number_of_unique_entity_mentions": int(mentions.nunique()),
    }
    return distribution(entity_types, "entity_type"), top_entities, summary


def candidate_text_statistics(df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    text = get_series(df, config, "text_column")
    if text is None:
        return length_statistics(pd.Series(dtype=str))
    return length_statistics(text)


def document_text_statistics(df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    text = get_series(df, config, "text_column")
    docs = get_series(df, config, "doc_id_column")
    if text is None or docs is None:
        return length_statistics(pd.Series(dtype=str))
    valid = docs.notna() & docs.astype(str).str.strip().ne("")
    document_frame = pd.DataFrame({"doc_id": docs[valid], "text": text[valid]}).drop_duplicates("doc_id")
    unmarked = document_frame["text"].fillna("").astype(str).str.replace(ENTITY_MARKER_RE, "", regex=True)
    unmarked = unmarked.str.replace(r"\s+", " ", regex=True).str.strip()
    return length_statistics(unmarked)


def marker_coverage(df: pd.DataFrame, config: dict[str, Any]) -> dict[str, float | int | None]:
    text = get_series(df, config, "text_column")
    total = int(len(df))
    if text is None:
        return {
            "total_rows": total,
            "rows_with_src_marker": None,
            "rows_with_tgt_marker": None,
            "rows_with_both_markers": None,
            "rows_missing_src_marker": None,
            "rows_missing_tgt_marker": None,
            "src_marker_coverage_pct": None,
            "tgt_marker_coverage_pct": None,
            "both_marker_coverage_pct": None,
        }
    values = text.fillna("").astype(str)
    has_src = values.str.contains(SRC_MARKER_RE, regex=True)
    has_tgt = values.str.contains(TGT_MARKER_RE, regex=True)
    both = has_src & has_tgt
    percentage = lambda count: float(count / total * 100) if total else None
    return {
        "total_rows": total,
        "rows_with_src_marker": int(has_src.sum()),
        "rows_with_tgt_marker": int(has_tgt.sum()),
        "rows_with_both_markers": int(both.sum()),
        "rows_missing_src_marker": int((~has_src).sum()),
        "rows_missing_tgt_marker": int((~has_tgt).sum()),
        "src_marker_coverage_pct": percentage(int(has_src.sum())),
        "tgt_marker_coverage_pct": percentage(int(has_tgt.sum())),
        "both_marker_coverage_pct": percentage(int(both.sum())),
    }


def write_json(data: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)


def run_exploration(config: dict[str, Any]) -> dict[str, Any]:
    source = Path(config["input_file"]).expanduser()
    output_dir = Path(config["output_dir"]).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Input CSV not found: {source.resolve()}")
    output_dir.mkdir(parents=True, exist_ok=True)
    legacy_text_stats = output_dir / "text_statistics.json"
    if legacy_text_stats.exists():
        legacy_text_stats.unlink()

    df = pd.read_csv(source)
    summary = dataset_statistics(df, config)
    relations, binary = relation_statistics(df, config)
    entity_types, top_entities, entity_summary = entity_statistics(df, config)
    summary.update(entity_summary)
    pair_types = distribution(get_series(df, config, "entity_pair_type_column"), "entity_pair_type")

    write_json(summary, output_dir / "dataset_summary.json")
    write_json(candidate_text_statistics(df, config), output_dir / "candidate_text_statistics.json")
    write_json(document_text_statistics(df, config), output_dir / "document_text_statistics.json")
    write_json(marker_coverage(df, config), output_dir / "marker_coverage.json")
    for table, filename in (
        (relations, "relation_distribution.csv"),
        (binary, "binary_relation_distribution.csv"),
        (entity_types, "entity_type_distribution.csv"),
        (top_entities, "top_entities.csv"),
        (pair_types, "entity_pair_type_distribution.csv"),
    ):
        table.to_csv(output_dir / filename, index=False)
    save_plot(relations, "relation_type", "Relation Type Distribution", output_dir / "relation_distribution.png")
    save_plot(entity_types, "entity_type", "Entity Type Distribution", output_dir / "entity_type_distribution.png")
    save_plot(pair_types, "entity_pair_type", "Entity Pair Type Distribution", output_dir / "entity_pair_type_distribution.png")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    summary = run_exploration(config)
    print(
        f"Exploration complete: {summary['number_of_rows']} rows, "
        f"{summary['number_of_unique_documents']} unique documents, "
        f"{summary['number_of_relation_types']} relation types. "
        f"Outputs: {Path(config['output_dir']).resolve()}"
    )


if __name__ == "__main__":
    main()
