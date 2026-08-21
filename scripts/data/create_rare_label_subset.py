#!/usr/bin/env python3
"""Create a document-level subset containing rare relation labels.

The synthetic twin generator works at document level.  If a document contains a
rare label, we keep all rows for that document so the full candidate-relation
structure remains intact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_RARE_LABELS = [
    "Bind",
    "Comparison",
    "Conversion",
    "Cotreatment",
    "Drug_Interaction",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input BioRED-style pipeline CSV.")
    parser.add_argument("--output", required=True, help="Output CSV with all rows from rare-label documents.")
    parser.add_argument("--summary", default=None, help="Optional JSON summary output path.")
    parser.add_argument("--doc-id-column", default="doc_id")
    parser.add_argument("--label-column", default="relation_type")
    parser.add_argument("--rare-labels", nargs="+", default=DEFAULT_RARE_LABELS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary) if args.summary else output_path.with_suffix(".summary.json")

    df = pd.read_csv(input_path)
    missing = [column for column in (args.doc_id_column, args.label_column) if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")

    rare_labels = set(args.rare_labels)
    rare_doc_ids = df.loc[df[args.label_column].isin(rare_labels), args.doc_id_column].dropna().unique()
    subset = df[df[args.doc_id_column].isin(rare_doc_ids)].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(output_path, index=False)

    summary = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "rare_labels": sorted(rare_labels),
        "source_rows": int(len(df)),
        "source_documents": int(df[args.doc_id_column].nunique()),
        "subset_rows": int(len(subset)),
        "subset_documents": int(subset[args.doc_id_column].nunique()),
        "source_label_counts": df[args.label_column].value_counts().to_dict(),
        "subset_label_counts": subset[args.label_column].value_counts().to_dict(),
    }
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)

    print(f"Saved rare-label document subset to: {output_path}")
    print(f"Saved summary to: {summary_path}")
    print(f"Rare-label documents: {summary['subset_documents']}")
    print(f"Subset rows: {summary['subset_rows']}")
    print(subset[args.label_column].value_counts())


if __name__ == "__main__":
    main()
