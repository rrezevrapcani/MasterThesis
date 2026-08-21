#!/usr/bin/env python3
"""Combine base synthetic twins with an additional boosted synthetic dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Base clean synthetic CSV.")
    parser.add_argument("--boost", required=True, help="Boosted clean synthetic CSV.")
    parser.add_argument("--output", required=True, help="Combined output CSV.")
    parser.add_argument("--summary", default=None, help="Optional summary JSON path.")
    parser.add_argument("--label-column", default="relation_type")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_path = Path(args.base)
    boost_path = Path(args.boost)
    output_path = Path(args.output)
    summary_path = Path(args.summary) if args.summary else output_path.with_suffix(".summary.json")

    base = pd.read_csv(base_path)
    boost = pd.read_csv(boost_path)
    if args.label_column not in base.columns or args.label_column not in boost.columns:
        raise ValueError(f"Both input files must contain '{args.label_column}'.")

    base = base.copy()
    boost = boost.copy()
    base["synthetic_dataset_source"] = "base_one_to_one"
    boost["synthetic_dataset_source"] = "rare_label_boost"

    all_columns = list(dict.fromkeys(list(base.columns) + list(boost.columns)))
    combined = pd.concat(
        [base.reindex(columns=all_columns), boost.reindex(columns=all_columns)],
        ignore_index=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    summary = {
        "base_file": str(base_path),
        "boost_file": str(boost_path),
        "output_file": str(output_path),
        "base_rows": int(len(base)),
        "boost_rows": int(len(boost)),
        "combined_rows": int(len(combined)),
        "base_label_counts": base[args.label_column].value_counts().to_dict(),
        "boost_label_counts": boost[args.label_column].value_counts().to_dict(),
        "combined_label_counts": combined[args.label_column].value_counts().to_dict(),
    }
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)

    print(f"Saved combined synthetic dataset to: {output_path}")
    print(f"Saved summary to: {summary_path}")
    print(combined[args.label_column].value_counts())


if __name__ == "__main__":
    main()
