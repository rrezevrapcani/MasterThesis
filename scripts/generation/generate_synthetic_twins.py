#!/usr/bin/env python3
"""Generate one-to-one synthetic twins for BioRED-style relation extraction documents.

This script preserves the original document-level relation/candidate list and asks an LLM
only to create a new biomedical abstract that mentions the same entities. Candidate rows
are then rebuilt from the original labels and entity pairs so the synthetic dataset is
one-to-one comparable with the real training data.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import pandas as pd
import yaml

ENTITY_TYPE_TO_MARKER = {
    "GeneOrGeneProduct": "GENE",
    "DiseaseOrPhenotypicFeature": "DISEASE",
    "ChemicalEntity": "CHEMICAL",
}
ENTITY_MARKER_RE = re.compile(r"\[/?[A-Z0-9]+_(?:SRC|TGT)\]\s*")
THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)

REQUIRED_CONFIG_KEYS = (
    "input_file",
    "output_dir",
    "model_name",
    "text_column",
    "label_column",
    "doc_id_column",
    "entity1_id_column",
    "entity2_id_column",
    "entity1_text_column",
    "entity2_text_column",
    "entity1_type_column",
    "entity2_type_column",
    "entity_pair_type_column",
    "allowed_labels",
)


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8-sig") as stream:
        config = yaml.safe_load(stream) or {}
    missing = [key for key in REQUIRED_CONFIG_KEYS if config.get(key) is None]
    if missing:
        raise ValueError(f"Missing required config value(s): {', '.join(missing)}")

    config["backend"] = str(config.get("backend", "vllm")).lower()
    config["max_tokens"] = int(config.get("max_tokens", 2048))
    config["temperature"] = float(config.get("temperature", 0.8))
    config["top_p"] = float(config.get("top_p", 0.95))
    config["do_sample"] = bool(config.get("do_sample", True))
    config["seed"] = int(config.get("seed", 42))
    config["dtype"] = str(config.get("dtype", "auto"))
    config["trust_remote_code"] = bool(config.get("trust_remote_code", True))
    config["tensor_parallel_size"] = int(config.get("tensor_parallel_size", 1))
    config["gpu_memory_utilization"] = float(config.get("gpu_memory_utilization", 0.9))
    config["n_generations_per_document"] = int(config.get("n_generations_per_document", 1))
    config["limit_documents"] = config.get("limit_documents")
    if config["limit_documents"] is not None:
        config["limit_documents"] = int(config["limit_documents"])
    config["progress_every"] = int(config.get("progress_every", 25))
    config["max_full_relations_in_prompt"] = config.get("max_full_relations_in_prompt", 120)
    if config["max_full_relations_in_prompt"] is not None:
        config["max_full_relations_in_prompt"] = int(config["max_full_relations_in_prompt"])

    if config["backend"] not in {"vllm", "transformers"}:
        raise ValueError("backend must be either 'vllm' or 'transformers'")
    if config["n_generations_per_document"] <= 0:
        raise ValueError("n_generations_per_document must be positive")
    return config


def strip_entity_markers(text: str) -> str:
    return re.sub(r"\s+", " ", ENTITY_MARKER_RE.sub("", str(text))).strip()


def clean_cell(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def marker_prefix(entity_type: str) -> str:
    return ENTITY_TYPE_TO_MARKER.get(entity_type, "ENTITY")


def build_document_records(df: pd.DataFrame, config: dict[str, Any]) -> list[dict[str, Any]]:
    doc_col = config["doc_id_column"]
    text_col = config["text_column"]
    label_col = config["label_column"]
    e1_id = config["entity1_id_column"]
    e2_id = config["entity2_id_column"]
    e1_text = config["entity1_text_column"]
    e2_text = config["entity2_text_column"]
    e1_type = config["entity1_type_column"]
    e2_type = config["entity2_type_column"]
    pair_type = config["entity_pair_type_column"]

    records: list[dict[str, Any]] = []
    for doc_id, group in df.groupby(doc_col, sort=False):
        first_text = clean_cell(group[text_col].iloc[0]) if text_col in group else ""
        source_abstract = strip_entity_markers(first_text)

        entities: dict[str, dict[str, str]] = {}
        relations: list[dict[str, Any]] = []
        for row_index, (_, row) in enumerate(group.iterrows()):
            ent1 = {
                "entity_id": clean_cell(row.get(e1_id)),
                "entity_text": clean_cell(row.get(e1_text)),
                "entity_type": clean_cell(row.get(e1_type)),
            }
            ent2 = {
                "entity_id": clean_cell(row.get(e2_id)),
                "entity_text": clean_cell(row.get(e2_text)),
                "entity_type": clean_cell(row.get(e2_type)),
            }
            if ent1["entity_id"]:
                entities.setdefault(ent1["entity_id"], ent1)
            if ent2["entity_id"]:
                entities.setdefault(ent2["entity_id"], ent2)
            relations.append(
                {
                    "source_row_index": int(row.name) if isinstance(row.name, int) else row_index,
                    "relation_index": row_index,
                    "entity1_id": ent1["entity_id"],
                    "entity2_id": ent2["entity_id"],
                    "entity1_text": ent1["entity_text"],
                    "entity2_text": ent2["entity_text"],
                    "entity1_type": ent1["entity_type"],
                    "entity2_type": ent2["entity_type"],
                    "entity_pair_type": clean_cell(row.get(pair_type)),
                    "relation_type": clean_cell(row.get(label_col)),
                }
            )
        records.append(
            {
                "doc_id": str(doc_id),
                "source_abstract": source_abstract,
                "entities": list(entities.values()),
                "relations": relations,
            }
        )
    limit = config.get("limit_documents")
    return records[:limit] if limit else records


def compact_relations_for_prompt(relations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "entity1_id": rel["entity1_id"],
            "entity2_id": rel["entity2_id"],
            "entity1_text": rel["entity1_text"],
            "entity2_text": rel["entity2_text"],
            "entity1_type": rel["entity1_type"],
            "entity2_type": rel["entity2_type"],
            "relation_type": rel["relation_type"],
        }
        for rel in relations
    ]


def build_prompt(record: dict[str, Any], config: dict[str, Any], generation_index: int) -> str:
    labels = "\n".join(f"- {label}" for label in config.get("allowed_labels", []))
    entities_json = json.dumps(record["entities"], ensure_ascii=False, indent=2)
    relation_count = len(record["relations"])
    relation_label_counts = pd.Series([rel["relation_type"] for rel in record["relations"]]).value_counts().to_dict()
    relation_summary_json = json.dumps(relation_label_counts, ensure_ascii=False, indent=2)
    max_full = config.get("max_full_relations_in_prompt")
    include_full_relations = max_full is None or relation_count <= int(max_full)
    if include_full_relations:
        relations_for_prompt: Any = compact_relations_for_prompt(record["relations"])
        relation_list_note = "Full original candidate relation list to preserve exactly:"
    else:
        relations_for_prompt = compact_relations_for_prompt(record["relations"][: int(max_full)])
        relation_list_note = (
            f"The document has {relation_count} original candidate rows, too many to show in full. "
            f"Below is a representative prefix of {max_full} rows. The script still preserves all original rows/labels in the output CSV."
        )
    relations_json = json.dumps(relations_for_prompt, ensure_ascii=False, indent=2)
    source_preview = record["source_abstract"][:4000]
    return f"""
You are generating a synthetic twin of one biomedical abstract for relation-extraction experiments.

Goal:
Create one original biomedical abstract that is directly comparable to the provided real document.
Keep the SAME entity list and the SAME relation/candidate list. Do not invent, remove, or relabel relations.
The new abstract should be a plausible biomedical rewrite/twin, not a copy of the source text.

Real document ID: {record['doc_id']}
Generation index for this document: {generation_index}

Allowed relation labels:
{labels}

Entities that must appear verbatim at least once in the synthetic abstract:
{entities_json}

Original relation label counts:
{relation_summary_json}

{relation_list_note}
{relations_json}

Source abstract for context only. Do not copy it verbatim:
{source_preview}

Requirements:
- Return only valid JSON, no markdown and no explanation.
- The first character must be {{ and the last character must be }}.
- Output exactly these keys: synthetic_id, source_doc_id, title, abstract.
- source_doc_id must be "{record['doc_id']}".
- The abstract must mention every entity_text from the entity list exactly as written.
- Do not insert relation-label tags such as [NO_RELATION] or [ASSOCIATION].
- Do not insert entity markers in the abstract; the script will add [GENE_SRC]/[GENE_TGT]-style markers later.
- Keep the biomedical topic and relation evidence compatible with the original relation labels and label-count summary.
- Use varied wording and avoid repeating generic PPARgamma/metformin/type 2 diabetes templates unless they are central to this source document.

JSON schema:
{{
  "synthetic_id": "synthetic_twin_{record['doc_id']}_{generation_index:02d}",
  "source_doc_id": "{record['doc_id']}",
  "title": "",
  "abstract": ""
}}
""".strip()


def build_prompt_records(document_records: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    request_index = 0
    for doc_record in document_records:
        for generation_index in range(1, config["n_generations_per_document"] + 1):
            prompts.append(
                {
                    "request_index": request_index,
                    "source_doc_id": doc_record["doc_id"],
                    "generation_index": generation_index,
                    "prompt": build_prompt(doc_record, config, generation_index),
                }
            )
            request_index += 1
    return prompts


def import_vllm() -> tuple[Any, Any]:
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise ImportError("vLLM is not installed in this environment.") from exc
    return LLM, SamplingParams


def generate_with_vllm(prompt_records: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    LLM, SamplingParams = import_vllm()
    sampling_params = SamplingParams(
        temperature=config["temperature"],
        top_p=config["top_p"],
        max_tokens=config["max_tokens"],
        seed=config["seed"],
    )
    llm = LLM(
        model=config["model_name"],
        tensor_parallel_size=config["tensor_parallel_size"],
        dtype=config["dtype"],
        gpu_memory_utilization=config["gpu_memory_utilization"],
        trust_remote_code=config["trust_remote_code"],
    )
    outputs = llm.generate([record["prompt"] for record in prompt_records], sampling_params)
    raw_records: list[dict[str, Any]] = []
    for i, output in enumerate(outputs):
        prompt_record = prompt_records[i]
        for candidate_index, candidate in enumerate(output.outputs):
            raw_records.append(
                {
                    **{k: prompt_record[k] for k in ("request_index", "source_doc_id", "generation_index")},
                    "candidate_index": candidate_index,
                    "text": candidate.text,
                    "finish_reason": getattr(candidate, "finish_reason", None),
                }
            )
    return raw_records


def generate_with_transformers(prompt_records: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    except ImportError as exc:
        raise ImportError("transformers and torch are required for backend='transformers'.") from exc

    set_seed(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], trust_remote_code=config["trust_remote_code"])
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype: Any = "auto"
    if config["dtype"] != "auto":
        dtype_name = str(config["dtype"])
        torch_dtype = getattr(torch, dtype_name) if hasattr(torch, dtype_name) else dtype_name

    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=config["trust_remote_code"],
    )
    if not torch.cuda.is_available():
        model = model.to("cpu")
    model.eval()

    raw_records: list[dict[str, Any]] = []
    for index, prompt_record in enumerate(prompt_records, start=1):
        prompt = prompt_record["prompt"]
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            kwargs = {"add_generation_prompt": True, "tokenize": False}
            if config.get("disable_thinking"):
                kwargs["enable_thinking"] = False
            try:
                model_input_text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], **kwargs)
            except TypeError:
                kwargs.pop("enable_thinking", None)
                model_input_text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], **kwargs)
        else:
            model_input_text = prompt

        encoded = tokenizer(model_input_text, return_tensors="pt")
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)

        generation_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": config["max_tokens"],
            "do_sample": config["do_sample"],
            "temperature": config["temperature"],
            "top_p": config["top_p"],
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if attention_mask is None:
            generation_kwargs.pop("attention_mask")
        if not config["do_sample"]:
            generation_kwargs.pop("temperature", None)
            generation_kwargs.pop("top_p", None)

        with torch.no_grad():
            output_ids = model.generate(**generation_kwargs)
        generated_ids = output_ids[0, input_ids.shape[-1]:]
        raw_records.append(
            {
                **{k: prompt_record[k] for k in ("request_index", "source_doc_id", "generation_index")},
                "candidate_index": 0,
                "text": tokenizer.decode(generated_ids, skip_special_tokens=True),
                "finish_reason": None,
            }
        )
        if config["progress_every"] and index % config["progress_every"] == 0:
            print(f"Generated {index}/{len(prompt_records)} twin prompt(s)", flush=True)
    return raw_records


def generate_text(prompt_records: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    if config["backend"] == "vllm":
        return generate_with_vllm(prompt_records, config)
    return generate_with_transformers(prompt_records, config)


def extract_first_json_object(text: str) -> Any:
    stripped = THINK_RE.sub("", str(text).strip())
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE).strip()
    stripped = re.sub(r"\s*```$", "", stripped).strip()
    decoder = json.JSONDecoder()
    errors: list[str] = []
    for start in [i for i, char in enumerate(stripped) if char == "{"]:
        try:
            value, _ = decoder.raw_decode(stripped[start:])
            return value
        except json.JSONDecodeError as exc:
            errors.append(str(exc))
    raise ValueError(f"No valid JSON object found. Last parser errors: {errors[-3:]}")


def find_span(text: str, needle: str, start: int = 0) -> tuple[int, int] | None:
    if not needle:
        return None
    pattern = re.compile(re.escape(needle), flags=re.IGNORECASE)
    match = pattern.search(text, pos=start)
    if not match:
        return None
    return match.start(), match.end()


def insert_marker(text: str, span: tuple[int, int], marker: str, role: str) -> tuple[str, tuple[int, int]]:
    start, end = span
    open_marker = f"[{marker}_{role}] "
    close_marker = f" [/{marker}_{role}]"
    marked = text[:start] + open_marker + text[start:end] + close_marker + text[end:]
    shift = len(open_marker) + len(close_marker)
    return marked, (start, end + shift)


def build_marked_text(abstract: str, relation: dict[str, Any]) -> tuple[str, list[str]]:
    text = strip_entity_markers(abstract)
    errors: list[str] = []
    e1 = relation["entity1_text"]
    e2 = relation["entity2_text"]
    span1 = find_span(text, e1)
    span2 = find_span(text, e2)
    if span1 is None:
        errors.append("missing_entity1_text_in_generated_abstract")
    if span2 is None:
        errors.append("missing_entity2_text_in_generated_abstract")
    if errors:
        return text, errors

    marker1 = marker_prefix(relation["entity1_type"])
    marker2 = marker_prefix(relation["entity2_type"])
    assert span1 is not None and span2 is not None
    first_role = (span1, marker1, "SRC")
    second_role = (span2, marker2, "TGT")
    if span2[0] < span1[0]:
        first_role, second_role = second_role, first_role
    marked, updated_first = insert_marker(text, first_role[0], first_role[1], first_role[2])
    second_span = second_role[0]
    if second_span[0] > first_role[0][0]:
        delta = updated_first[1] - first_role[0][1]
        second_span = (second_span[0] + delta, second_span[1] + delta)
    marked, _ = insert_marker(marked, second_span, second_role[1], second_role[2])
    return marked, errors


def normalize_twin(parsed: Any, raw: dict[str, Any]) -> dict[str, str]:
    if not isinstance(parsed, dict):
        raise ValueError("parsed output is not a JSON object")
    synthetic_id = clean_cell(parsed.get("synthetic_id")) or f"synthetic_twin_{raw['source_doc_id']}_{raw['generation_index']:02d}"
    source_doc_id = clean_cell(parsed.get("source_doc_id")) or str(raw["source_doc_id"])
    title = clean_cell(parsed.get("title"))
    abstract = clean_cell(parsed.get("abstract"))
    if not abstract:
        raise ValueError("generated abstract is empty")
    return {
        "synthetic_id": synthetic_id,
        "doc_id": synthetic_id,
        "source_doc_id": source_doc_id,
        "generation_index": str(raw["generation_index"]),
        "title": title,
        "abstract": abstract,
    }


def build_outputs(
    twins: list[dict[str, str]],
    document_lookup: dict[str, dict[str, Any]],
    allowed_labels: set[str],
) -> tuple[pd.DataFrame, list[dict[str, str]], list[str]]:
    rows: list[dict[str, Any]] = []
    validation_errors: list[str] = []
    for twin in twins:
        source_doc = document_lookup.get(str(twin["source_doc_id"]))
        if not source_doc:
            validation_errors.append(f"{twin['synthetic_id']}: source_doc_id not found: {twin['source_doc_id']}")
            continue
        for relation in source_doc["relations"]:
            marked_text, marker_errors = build_marked_text(twin["abstract"], relation)
            row_errors = list(marker_errors)
            if relation["relation_type"] not in allowed_labels:
                row_errors.append("invalid_relation_type")
            rows.append(
                {
                    "synthetic_id": twin["synthetic_id"],
                    "doc_id": twin["doc_id"],
                    "source_doc_id": twin["source_doc_id"],
                    "generation_index": twin["generation_index"],
                    "title": twin["title"],
                    "abstract": twin["abstract"],
                    "source_row_index": relation["source_row_index"],
                    "relation_index": relation["relation_index"],
                    "entity1_id": relation["entity1_id"],
                    "entity2_id": relation["entity2_id"],
                    "entity1_text": relation["entity1_text"],
                    "entity2_text": relation["entity2_text"],
                    "entity1_type": relation["entity1_type"],
                    "entity2_type": relation["entity2_type"],
                    "entity_pair_type": relation["entity_pair_type"],
                    "relation_type": relation["relation_type"],
                    "marked_text": marked_text,
                    "relation_is_valid": len(row_errors) == 0,
                    "relation_validation_errors": ";".join(row_errors),
                }
            )
    return pd.DataFrame(rows), twins, validation_errors


def write_json(data: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def run(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(config["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(config["input_file"])
    document_records = build_document_records(df, config)
    document_lookup = {record["doc_id"]: record for record in document_records}
    prompt_records = build_prompt_records(document_records, config)
    write_jsonl(prompt_records, output_dir / "twin_prompts.jsonl")

    raw_records = generate_text(prompt_records, config)
    write_jsonl(raw_records, output_dir / "raw_generations.jsonl")

    parse_errors: list[str] = []
    twins: list[dict[str, str]] = []
    for raw in raw_records:
        try:
            parsed = extract_first_json_object(raw["text"])
            twins.append(normalize_twin(parsed, raw))
        except Exception as exc:
            parse_errors.append(f"request {raw.get('request_index')} source_doc_id={raw.get('source_doc_id')}: {exc}")

    relations, normalized_twins, validation_errors = build_outputs(twins, document_lookup, set(config["allowed_labels"]))
    clean_relations = relations[relations["relation_is_valid"]].copy() if "relation_is_valid" in relations else relations.copy()

    write_json(normalized_twins, output_dir / "synthetic_twins.json")
    relations.to_csv(output_dir / "synthetic_twin_relations.csv", index=False)
    clean_relations.to_csv(output_dir / "synthetic_twin_relations_clean.csv", index=False)
    write_json(config, output_dir / "synthetic_twin_generation_config.json")

    label_counts = relations["relation_type"].value_counts().to_dict() if "relation_type" in relations else {}
    clean_label_counts = clean_relations["relation_type"].value_counts().to_dict() if "relation_type" in clean_relations else {}
    source_doc_count = df[config["doc_id_column"]].nunique()
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "strategy": "synthetic_twins_one_to_one_relation_preserving",
        "model_name": config["model_name"],
        "backend": config["backend"],
        "input_file": config["input_file"],
        "source_documents": int(source_doc_count),
        "source_relation_rows": int(len(df)),
        "documents_requested": int(len(prompt_records)),
        "n_generations_per_document": config["n_generations_per_document"],
        "raw_generations": int(len(raw_records)),
        "parsed_twins": int(len(normalized_twins)),
        "relation_rows": int(len(relations)),
        "clean_relation_rows": int(len(clean_relations)),
        "dropped_relation_rows": int(len(relations) - len(clean_relations)),
        "unique_source_documents_in_clean": int(clean_relations["source_doc_id"].nunique()) if "source_doc_id" in clean_relations else 0,
        "unique_synthetic_documents_in_clean": int(clean_relations["doc_id"].nunique()) if "doc_id" in clean_relations else 0,
        "label_counts": label_counts,
        "clean_label_counts": clean_label_counts,
        "parse_error_count": len(parse_errors),
        "validation_error_count": len(validation_errors),
        "invalid_relation_row_count": int((~relations["relation_is_valid"]).sum()) if "relation_is_valid" in relations else 0,
        "parse_errors": parse_errors[:50],
        "validation_errors": validation_errors[:50],
        "outputs": {
            "twin_prompts": "twin_prompts.jsonl",
            "raw_generations": "raw_generations.jsonl",
            "synthetic_twins": "synthetic_twins.json",
            "synthetic_twin_relations": "synthetic_twin_relations.csv",
            "synthetic_twin_relations_clean": "synthetic_twin_relations_clean.csv",
            "synthetic_twin_generation_config": "synthetic_twin_generation_config.json",
        },
    }
    write_json(summary, output_dir / "synthetic_twin_generation_summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Path to synthetic twin generation YAML")
    args = parser.parse_args()
    summary = run(load_config(args.config))
    print(
        "Synthetic twin generation complete: "
        f"{summary['parsed_twins']} twins, "
        f"{summary['clean_relation_rows']} clean relation rows, "
        f"{summary['invalid_relation_row_count']} invalid relation row(s)."
    )
    if summary["parse_error_count"]:
        print(f"Parse errors: {summary['parse_error_count']} -- see synthetic_twin_generation_summary.json")


if __name__ == "__main__":
    main()