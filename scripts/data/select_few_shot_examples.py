#!/usr/bin/env python3
"""Select a small set of BioRED documents for few-shot synthetic generation prompts."""
from __future__ import annotations

import argparse
import itertools
import json
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

SRC_MARKER_RE = re.compile(r"\[[A-Z0-9]+_SRC\]")
TGT_MARKER_RE = re.compile(r"\[[A-Z0-9]+_TGT\]")

DEFAULT_REQUIRED_LABELS = [
    "No_Relation",
    "Association",
    "Bind",
    "Comparison",
    "Conversion",
    "Cotreatment",
    "Drug_Interaction",
    "Negative_Correlation",
    "Positive_Correlation",
]

METADATA_CONFIG_KEYS = [
    "entity1_text_column",
    "entity2_text_column",
    "entity1_id_column",
    "entity2_id_column",
    "entity1_type_column",
    "entity2_type_column",
    "entity_pair_type_column",
]


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8-sig") as stream:
        config = yaml.safe_load(stream) or {}
    required = ["input_file", "output_dir", "doc_id_column", "text_column", "label_column"]
    missing = [key for key in required if not config.get(key)]
    if missing:
        raise ValueError(f"Missing required config value(s): {', '.join(missing)}")
    config["required_labels"] = list(config.get("required_labels") or DEFAULT_REQUIRED_LABELS)
    config["number_of_documents"] = int(config.get("number_of_documents", 3))
    config["beam_width"] = int(config.get("beam_width", 5000))
    config["ideal_min_words"] = int(config.get("ideal_min_words", 150))
    config["ideal_max_words"] = int(config.get("ideal_max_words", 650))
    if config["number_of_documents"] <= 0:
        raise ValueError("number_of_documents must be positive")
    if config["beam_width"] <= 0:
        raise ValueError("beam_width must be positive")
    return config


def clean_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def load_input(config: dict[str, Any]) -> pd.DataFrame:
    input_path = Path(config["input_file"]).expanduser()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path.resolve()}")
    frame = pd.read_csv(input_path)
    for key in ("doc_id_column", "text_column", "label_column"):
        column = config[key]
        if column not in frame.columns:
            raise KeyError(f"Required column not found: {column!r}")
    return frame


def row_has_src_marker(text: Any) -> bool:
    return bool(SRC_MARKER_RE.search(clean_value(text)))


def row_has_tgt_marker(text: Any) -> bool:
    return bool(TGT_MARKER_RE.search(clean_value(text)))


def text_word_length(text: str) -> int:
    return len(text.split())


def metadata_missing_count(group: pd.DataFrame, config: dict[str, Any]) -> int:
    missing = 0
    for key in METADATA_CONFIG_KEYS:
        column = config.get(key)
        if not column or column not in group.columns:
            warnings.warn(f"Optional metadata column unavailable for scoring: {column!r}")
            continue
        values = group[column]
        missing += int(values.isna().sum())
        missing += int(values.dropna().astype(str).str.strip().eq("").sum())
    return missing


def unique_nonempty(group: pd.DataFrame, column: str | None) -> set[str]:
    if not column or column not in group.columns:
        return set()
    values = group[column].dropna().astype(str).str.strip()
    return set(values[values.ne("")])


def representative_text(group: pd.DataFrame, text_column: str) -> str:
    texts = group[text_column].dropna().astype(str)
    if texts.empty:
        return ""
    return texts.iloc[texts.str.len().argmax()]


def build_document_summaries(frame: pd.DataFrame, config: dict[str, Any]) -> list[dict[str, Any]]:
    doc_id_column = config["doc_id_column"]
    text_column = config["text_column"]
    label_column = config["label_column"]
    pair_type_column = config.get("entity_pair_type_column")
    summaries: list[dict[str, Any]] = []

    for doc_id, group in frame.groupby(doc_id_column, dropna=False, sort=False):
        labels = unique_nonempty(group, label_column)
        pair_types = unique_nonempty(group, pair_type_column)
        text = representative_text(group, text_column)
        word_length = text_word_length(text)
        rows_with_src = group[text_column].map(row_has_src_marker)
        rows_with_tgt = group[text_column].map(row_has_tgt_marker)
        complete_marker_rows = rows_with_src & rows_with_tgt
        missing_metadata = metadata_missing_count(group, config)
        ideal_length = config["ideal_min_words"] <= word_length <= config["ideal_max_words"]
        length_penalty = 0
        if word_length < config["ideal_min_words"]:
            length_penalty = config["ideal_min_words"] - word_length
        elif word_length > config["ideal_max_words"]:
            length_penalty = word_length - config["ideal_max_words"]

        summaries.append(
            {
                "doc_id": clean_value(doc_id),
                "labels": labels,
                "entity_pair_types": pair_types,
                "row_count": int(len(group)),
                "complete_marker_rows": int(complete_marker_rows.sum()),
                "marker_coverage_ratio": float(complete_marker_rows.mean()) if len(group) else 0.0,
                "missing_metadata_values": int(missing_metadata),
                "representative_text_word_length": int(word_length),
                "ideal_length": bool(ideal_length),
                "length_penalty": int(length_penalty),
            }
        )
    return summaries


def score_selection(selection: tuple[dict[str, Any], ...], required_labels: set[str]) -> tuple[Any, ...]:
    covered_labels = set().union(*(doc["labels"] for doc in selection))
    pair_types = set().union(*(doc["entity_pair_types"] for doc in selection))
    marker_coverage = float(np.mean([doc["marker_coverage_ratio"] for doc in selection]))
    missing_metadata = sum(doc["missing_metadata_values"] for doc in selection)
    length_penalty = sum(doc["length_penalty"] for doc in selection)
    rows = sum(doc["row_count"] for doc in selection)
    exact_cover = required_labels.issubset(covered_labels)

    return (
        int(exact_cover),
        len(covered_labels & required_labels),
        marker_coverage,
        -missing_metadata,
        -length_penalty,
        len(pair_types),
        -rows,
    )


def exhaustive_select_documents(
    summaries: list[dict[str, Any]],
    config: dict[str, Any],
    required_labels: set[str],
) -> tuple[tuple[dict[str, Any], ...], tuple[Any, ...]]:
    best_selection: tuple[dict[str, Any], ...] | None = None
    best_score: tuple[Any, ...] | None = None
    for selection in itertools.combinations(summaries, config["number_of_documents"]):
        score = score_selection(selection, required_labels)
        if best_score is None or score > best_score:
            best_score = score
            best_selection = selection

    if best_selection is None or best_score is None:
        raise RuntimeError("No document selection could be computed")
    return best_selection, best_score


def beam_select_documents(
    summaries: list[dict[str, Any]],
    config: dict[str, Any],
    required_labels: set[str],
) -> tuple[tuple[dict[str, Any], ...], tuple[Any, ...]]:
    beam_width = config["beam_width"]
    k = config["number_of_documents"]
    beams: list[tuple[dict[str, Any], ...]] = [()]

    for _ in range(k):
        candidates: list[tuple[dict[str, Any], ...]] = []
        for selection in beams:
            selected_ids = {doc["doc_id"] for doc in selection}
            for doc in summaries:
                if doc["doc_id"] not in selected_ids:
                    expanded = selection + (doc,)
                    candidates.append(tuple(sorted(expanded, key=lambda item: item["doc_id"])))

        unique_candidates = list({tuple(doc["doc_id"] for doc in selection): selection for selection in candidates}.values())
        unique_candidates.sort(key=lambda selection: score_selection(selection, required_labels), reverse=True)
        beams = unique_candidates[:beam_width]

    if not beams:
        raise RuntimeError("No document selection could be computed")

    best_selection = max(beams, key=lambda selection: score_selection(selection, required_labels))
    best_score = score_selection(best_selection, required_labels)
    return best_selection, best_score


def select_documents(summaries: list[dict[str, Any]], config: dict[str, Any]) -> tuple[list[str], bool, str]:
    required_labels = set(config["required_labels"])
    k = config["number_of_documents"]
    if len(summaries) < k:
        raise ValueError(f"Requested {k} documents, but only {len(summaries)} document(s) are available")

    if k <= 3:
        best_selection, best_score = exhaustive_select_documents(summaries, config, required_labels)
        search_strategy = "exhaustive"
    else:
        best_selection, best_score = beam_select_documents(summaries, config, required_labels)
        search_strategy = f"beam_width_{config['beam_width']}"

    selected_ids = [doc["doc_id"] for doc in best_selection]
    exact_cover = bool(best_score[0]) if best_score is not None else False
    return selected_ids, exact_cover, search_strategy


def entity_pair_records(group: pd.DataFrame, config: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    column_map = {
        "entity1_text": config.get("entity1_text_column"),
        "entity2_text": config.get("entity2_text_column"),
        "entity1_id": config.get("entity1_id_column"),
        "entity2_id": config.get("entity2_id_column"),
        "entity1_type": config.get("entity1_type_column"),
        "entity2_type": config.get("entity2_type_column"),
        "entity_pair_type": config.get("entity_pair_type_column"),
        "relation_type": config.get("label_column"),
    }
    for _, row in group.iterrows():
        record = {}
        for output_name, source_column in column_map.items():
            record[output_name] = clean_value(row[source_column]) if source_column in group.columns else ""
        records.append(record)
    return records


def selected_examples_payload(frame: pd.DataFrame, selected_ids: list[str], config: dict[str, Any]) -> dict[str, Any]:
    doc_id_column = config["doc_id_column"]
    text_column = config["text_column"]
    label_column = config["label_column"]
    documents = []
    selected_set = set(selected_ids)
    selected_frame = frame[frame[doc_id_column].astype(str).isin(selected_set)].copy()

    for doc_id in selected_ids:
        group = selected_frame[selected_frame[doc_id_column].astype(str) == str(doc_id)]
        labels = sorted(unique_nonempty(group, label_column))
        documents.append(
            {
                "doc_id": doc_id,
                "labels": labels,
                "representative_marked_text": representative_text(group, text_column),
                "entity_pairs": entity_pair_records(group, config),
            }
        )
    return {"selected_documents": documents}


def write_json(data: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)


def run(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(config["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_input(config)

    summaries = build_document_summaries(frame, config)
    selected_ids, exact_cover, search_strategy = select_documents(summaries, config)
    doc_id_column = config["doc_id_column"]
    label_column = config["label_column"]
    selected_frame = frame[frame[doc_id_column].astype(str).isin(set(selected_ids))].copy()
    selected_frame.to_csv(output_dir / "few_shot_examples.csv", index=False)

    examples = selected_examples_payload(frame, selected_ids, config)
    write_json(examples, output_dir / "few_shot_examples.json")

    selected_doc_summaries = [doc for doc in summaries if doc["doc_id"] in set(selected_ids)]
    labels_by_doc = {doc["doc_id"]: sorted(doc["labels"]) for doc in selected_doc_summaries}
    covered = sorted(set().union(*(doc["labels"] for doc in selected_doc_summaries)))
    required = set(config["required_labels"])
    missing = sorted(required - set(covered))
    coverage = {
        "selection_type": f"exact_{config['number_of_documents']}_document_cover" if exact_cover else "maximum_coverage_fallback",
        "search_strategy": search_strategy,
        "requested_number_of_documents": config["number_of_documents"],
        "selected_document_ids": selected_ids,
        "required_labels": config["required_labels"],
        "labels_in_each_document": labels_by_doc,
        "union_of_covered_labels": covered,
        "missing_labels": missing,
        "covers_all_required_labels": len(missing) == 0,
        "selected_document_summaries": [
            {
                **doc,
                "labels": sorted(doc["labels"]),
                "entity_pair_types": sorted(doc["entity_pair_types"]),
            }
            for doc in selected_doc_summaries
        ],
        "label_counts_in_selected_examples": selected_frame[label_column].value_counts().to_dict(),
        "number_of_selected_rows": int(len(selected_frame)),
    }
    write_json(coverage, output_dir / "few_shot_coverage.json")
    return coverage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Path to few-shot selection YAML")
    args = parser.parse_args()
    coverage = run(load_config(args.config))
    status = "complete" if coverage["covers_all_required_labels"] else "partial"
    print(
        f"Few-shot selection {status}: {len(coverage['union_of_covered_labels'])}/"
        f"{len(coverage['required_labels'])} labels covered by "
        f"{len(coverage['selected_document_ids'])} documents."
    )
    if coverage["missing_labels"]:
        print(f"Missing labels: {', '.join(coverage['missing_labels'])}")
    print(f"Selected document IDs: {', '.join(coverage['selected_document_ids'])}")


if __name__ == "__main__":
    main()
