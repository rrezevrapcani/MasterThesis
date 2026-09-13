#!/usr/bin/env python3
"""Select real BioRED documents as few-shot style examples for twin generation.

They are real abstract/style examples with entity and relation-label summaries. 
The current document being generated still controls its own entities and relation/candidate structure.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


ENTITY_MARKER_RE = re.compile(r"\[/?[A-Z0-9]+_(?:SRC|TGT)\]\s*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input BioRED-style pipeline CSV.")
    parser.add_argument("--output", required=True, help="Output few-shot examples JSON.")
    parser.add_argument("--num-examples", type=int, default=3)
    parser.add_argument("--doc-id-column", default="doc_id")
    parser.add_argument("--text-column", default="marked_text")
    parser.add_argument("--label-column", default="relation_type")
    parser.add_argument("--max-abstract-chars", type=int, default=1200)
    parser.add_argument("--max-entities", type=int, default=10)
    return parser.parse_args()


def clean_cell(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def strip_entity_markers(text: str) -> str:
    return re.sub(r"\s+", " ", ENTITY_MARKER_RE.sub("", str(text))).strip()


def collect_entities(group: pd.DataFrame, max_entities: int) -> list[dict[str, str]]:
    entities: dict[str, dict[str, str]] = {}
    for _, row in group.iterrows():
        for prefix in ("entity1", "entity2"):
            entity_id = clean_cell(row.get(f"{prefix}_id"))
            entity_text = clean_cell(row.get(f"{prefix}_text"))
            entity_type = clean_cell(row.get(f"{prefix}_type"))
            if entity_id and entity_text:
                entities.setdefault(
                    entity_id,
                    {
                        "entity_id": entity_id,
                        "entity_text": entity_text,
                        "entity_type": entity_type,
                    },
                )
    return list(entities.values())[:max_entities]


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    missing = [column for column in (args.doc_id_column, args.text_column, args.label_column) if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")

    scored_docs: list[tuple[tuple[int, int, int], str, pd.DataFrame]] = []
    for doc_id, group in df.groupby(args.doc_id_column, sort=False):
        label_counts = group[args.label_column].value_counts()
        positive_count = int(label_counts.drop(labels=["No_Relation"], errors="ignore").sum())
        num_labels = int((label_counts > 0).sum())
        total_rows = int(len(group))
        if positive_count == 0:
            continue
        # Prefer examples with several labels and positive relations, but avoid
        # very large documents that would make the few-shot prompt too long.
        scored_docs.append(((num_labels, positive_count, -total_rows), str(doc_id), group))

    examples: list[dict[str, Any]] = []
    for _, doc_id, group in sorted(scored_docs, reverse=True)[: args.num_examples]:
        source_abstract = strip_entity_markers(clean_cell(group[args.text_column].iloc[0]))
        examples.append(
            {
                "source_doc_id": doc_id,
                "real_abstract": source_abstract[: args.max_abstract_chars],
                "relation_label_counts": group[args.label_column].value_counts().to_dict(),
                "entities": collect_entities(group, args.max_entities),
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump({"examples": examples}, stream, indent=2, ensure_ascii=False)

    print(f"Saved {len(examples)} real BioRED few-shot example(s) to: {output_path}")
    for example in examples:
        print(example["source_doc_id"], example["relation_label_counts"])


if __name__ == "__main__":
    main()
