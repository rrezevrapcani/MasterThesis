#!/usr/bin/env python3
"""A reproducible baseline prompt for synthetic biomedical data generation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8-sig") as stream:
        config = yaml.safe_load(stream) or {}
    required = ["few_shot_examples_file", "output_dir", "labels", "requested_number_of_abstracts"]
    missing = [key for key in required if config.get(key) is None]
    if missing:
        raise ValueError(f"Missing required config value(s): {', '.join(missing)}")
    config["requested_number_of_abstracts"] = int(config["requested_number_of_abstracts"])
    return config


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(data: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)


def format_labels(labels: list[str]) -> str:
    return "\n".join(f"- {label}" for label in labels)


def compact_example(document: dict[str, Any]) -> dict[str, Any]:
    return {
        "doc_id": document.get("doc_id", ""),
        "labels": document.get("labels", []),
        "marked_text": document.get("representative_marked_text", ""),
        "entity_pairs": document.get("entity_pairs", []),
    }


def build_prompt_payload(config: dict[str, Any], examples_payload: dict[str, Any], coverage: dict[str, Any] | None) -> dict[str, Any]:
    labels = list(config["labels"])
    entity_types = list(config.get("entity_types") or [])
    examples = [compact_example(doc) for doc in examples_payload.get("selected_documents", [])]

    schema = {
        "synthetic_id": "",
        "title": "",
        "abstract": "",
        "entities": [
            {
                "entity_id": "",
                "entity_text": "",
                "entity_type": "",
            }
        ],
        "relations": [
            {
                "entity1_id": "",
                "entity2_id": "",
                "relation_type": "",
                "marked_text": "",
            }
        ],
    }

    system_instruction = (
        "You are a biomedical relation-extraction data generator. Produce realistic, original "
        "biomedical abstracts and structured candidate-level annotations for relation extraction. "
        "Follow the requested JSON schema exactly and do not copy the few-shot source abstracts."
    )

    task_instruction = (
        f"Generate {config['requested_number_of_abstracts']} synthetic biomedical abstracts. "
        "Each abstract should contain explicit source and target entity mentions, candidate entity pairs, "
        "and one final relation label per candidate pair. Use only the provided nine-label BioRED schema."
    )

    constraints = [
        "Do not copy or lightly paraphrase the source few-shot abstracts.",
        "Preserve valid biomedical terminology and plausible biomedical context.",
        "Use exactly one final relation_type for each candidate relation.",
        "Use only labels from the provided label schema.",
        "Insert valid source and target markers in each relation-level marked_text field.",
        config.get(
            "marker_instruction",
            "Use source markers ending in _SRC and target markers ending in _TGT around the relevant entity mentions.",
        ),
        "Return valid JSON only. Do not include markdown fences or explanatory text.",
        "Every entity_id used in relations must appear in the entities list.",
        "Include No_Relation examples as genuine negative candidate pairs, not as missing annotations.",
    ]

    payload = {
        "system_instruction": system_instruction,
        "task_instruction": task_instruction,
        "label_schema": labels,
        "entity_types": entity_types,
        "few_shot_coverage": coverage or {},
        "few_shot_examples": examples,
        "output_json_schema": {
            "synthetic_abstracts": [schema],
        },
        "generation_constraints": constraints,
    }
    return payload


def render_prompt(payload: dict[str, Any]) -> str:
    sections = [
        ("SYSTEM INSTRUCTION", payload["system_instruction"]),
        ("TASK INSTRUCTION", payload["task_instruction"]),
        ("NINE-LABEL BIORED SCHEMA", format_labels(payload["label_schema"])),
        ("ALLOWED ENTITY TYPES", format_labels(payload["entity_types"])),
        ("FEW-SHOT COVERAGE", json.dumps(payload["few_shot_coverage"], indent=2, ensure_ascii=False)),
        ("FEW-SHOT EXAMPLES", json.dumps(payload["few_shot_examples"], indent=2, ensure_ascii=False)),
        ("OUTPUT JSON SCHEMA", json.dumps(payload["output_json_schema"], indent=2, ensure_ascii=False)),
        ("GENERATION CONSTRAINTS", format_labels(payload["generation_constraints"])),
    ]
    return "\n\n".join(f"## {title}\n\n{content}" for title, content in sections) + "\n"


def run(config: dict[str, Any]) -> dict[str, Any]:
    examples_file = Path(config["few_shot_examples_file"]).expanduser()
    if not examples_file.is_file():
        raise FileNotFoundError(f"Few-shot examples JSON not found: {examples_file.resolve()}")
    examples_payload = load_json(examples_file)

    coverage = None
    coverage_file = config.get("coverage_file")
    if coverage_file and Path(coverage_file).expanduser().is_file():
        coverage = load_json(Path(coverage_file).expanduser())

    output_dir = Path(config["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_name = str(config.get("prompt_name", "baseline_prompt"))

    prompt_payload = build_prompt_payload(config, examples_payload, coverage)
    prompt_text = render_prompt(prompt_payload)
    text_path = output_dir / f"{prompt_name}.txt"
    json_path = output_dir / f"{prompt_name}.json"
    text_path.write_text(prompt_text, encoding="utf-8")
    write_json(prompt_payload, json_path)

    summary = {
        "prompt_text_file": str(text_path),
        "prompt_json_file": str(json_path),
        "number_of_few_shot_documents": len(prompt_payload["few_shot_examples"]),
        "requested_number_of_abstracts": config["requested_number_of_abstracts"],
        "number_of_labels": len(prompt_payload["label_schema"]),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Path to synthetic prompt YAML")
    args = parser.parse_args()
    summary = run(load_yaml(args.config))
    print(
        "Baseline prompt written: "
        f"{summary['prompt_text_file']} and {summary['prompt_json_file']} "
        f"using {summary['number_of_few_shot_documents']} few-shot documents."
    )


if __name__ == "__main__":
    main()
