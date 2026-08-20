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
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

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
    config["repair_missing_entity_mentions"] = bool(config.get("repair_missing_entity_mentions", True))
    config["retry_failed_generations"] = bool(config.get("retry_failed_generations", True))
    config["max_parse_retries"] = int(config.get("max_parse_retries", 4))
    config["retry_source_preview_chars"] = int(config.get("retry_source_preview_chars", 1500))
    config["retry_max_tokens"] = int(config.get("retry_max_tokens", 1024))
    config["retry_temperature"] = float(config.get("retry_temperature", 0.2))
    config["retry_top_p"] = float(config.get("retry_top_p", 0.8))
    config["allow_plain_text_fallback"] = bool(config.get("allow_plain_text_fallback", True))
    config["plain_text_fallback_min_chars"] = int(config.get("plain_text_fallback_min_chars", 80))
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
    no_think = "\n/no_think" if config.get("disable_thinking") else ""
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
- For pairs labeled No_Relation, do not write sentences that imply association, causation, treatment effect, binding, comparison, conversion, cotreatment, drug interaction, positive correlation, or negative correlation between that exact entity pair.
- For positive relation labels, include enough local context that the relation remains plausible for the original entity pair.
- Use varied wording and avoid repeating generic PPARgamma/metformin/type 2 diabetes templates unless they are central to this source document.

JSON schema:
{{
  "synthetic_id": "synthetic_twin_{record['doc_id']}_{generation_index:02d}",
  "source_doc_id": "{record['doc_id']}",
  "title": "",
  "abstract": ""
}}
{no_think}
""".strip()


def build_retry_prompt(record: dict[str, Any], config: dict[str, Any], generation_index: int, retry_index: int) -> str:
    """Build a shorter prompt for documents whose first generation was not valid JSON."""
    entities_json = json.dumps(record["entities"], ensure_ascii=False, indent=2)
    relation_label_counts = pd.Series([rel["relation_type"] for rel in record["relations"]]).value_counts().to_dict()
    relation_summary_json = json.dumps(relation_label_counts, ensure_ascii=False, indent=2)
    source_preview = record["source_abstract"][: config["retry_source_preview_chars"]]
    no_think = "\n/no_think" if config.get("disable_thinking") else ""
    return f"""
Return exactly one valid JSON object for a synthetic biomedical abstract.

This is retry {retry_index} for real document ID {record['doc_id']}.

Required JSON keys only:
{{
  "synthetic_id": "synthetic_twin_{record['doc_id']}_{generation_index:02d}",
  "source_doc_id": "{record['doc_id']}",
  "title": "",
  "abstract": ""
}}

Rules:
- Return JSON only. No markdown. No commentary.
- The first character must be {{ and the last character must be }}.
- Use double quotes for all JSON strings.
- Keep the abstract concise enough that the JSON is complete.
- Do not include entity markers such as [GENE_SRC].
- Mention every entity_text from this list exactly as written at least once:
{entities_json}
- Preserve the biomedical topic and make the relation evidence compatible with this original label-count summary:
{relation_summary_json}
- Do not copy the source abstract verbatim.

Source abstract context:
{source_preview}
{no_think}
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


def clean_generated_plain_text(text: str) -> str:
    cleaned = THINK_RE.sub("", str(text).strip())
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    cleaned = re.sub(r"^(title|abstract)\s*:\s*", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"^#+\s*", "", cleaned).strip()
    cleaned = cleaned.strip("` \n\t")
    return re.sub(r"\s+", " ", cleaned).strip()


def fallback_plain_text_twin(raw: dict[str, Any], min_chars: int) -> dict[str, str]:
    abstract = clean_generated_plain_text(raw.get("text", ""))
    if len(abstract) < min_chars:
        raise ValueError(f"plain text fallback too short: {len(abstract)} char(s)")
    return {
        "synthetic_id": f"synthetic_twin_{raw['source_doc_id']}_{int(raw['generation_index']):02d}",
        "doc_id": f"synthetic_twin_{raw['source_doc_id']}_{int(raw['generation_index']):02d}",
        "source_doc_id": str(raw["source_doc_id"]),
        "generation_index": str(raw["generation_index"]),
        "title": "",
        "abstract": abstract,
    }


def parse_raw_generations(raw_records: list[dict[str, Any]]) -> tuple[list[dict[str, str]], list[str], list[dict[str, Any]]]:
    twins: list[dict[str, str]] = []
    parse_errors: list[str] = []
    failed_records: list[dict[str, Any]] = []
    for raw in raw_records:
        try:
            parsed = extract_first_json_object(raw["text"])
            twins.append(normalize_twin(parsed, raw))
        except Exception as exc:
            parse_errors.append(f"request {raw.get('request_index')} source_doc_id={raw.get('source_doc_id')}: {exc}")
            failed_records.append(raw)
    return twins, parse_errors, failed_records


def build_entity_regex(needle: str) -> str:
    normalized = re.sub(r"\s*(['’`])\s*", r"\1", str(needle).strip())
    normalized = re.sub(r"\s*([-‐‑–—])\s*", r"\1", normalized)
    escaped = re.escape(normalized)
    escaped = escaped.replace(r"\ ", r"\s+")
    escaped = escaped.replace(r"\'", r"\s*['’`]\s*")
    escaped = escaped.replace("’", r"\s*['’`]\s*")
    escaped = escaped.replace(r"\-", r"\s*[-‐‑–—]\s*")
    return rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])"


def build_entity_regex(needle: str) -> str:
    """Build a conservative regex for matching entity mentions.

    BioRED mention text sometimes contains tokenized punctuation, e.g.
    "Parkinson 's disease", while generated text may use "Parkinson's
    disease". This matcher tolerates whitespace around apostrophes and
    hyphens, while still using word-ish boundaries so short entities such as
    "Rab" are not matched inside longer mentions such as "Rab6c".
    """
    normalized = str(needle).strip()
    pieces: list[str] = []
    for char in normalized:
        if char.isspace():
            pieces.append(r"\s+")
        elif char in "'’`":
            pieces.append(r"\s*['’`]\s*")
        elif char in "-‐‑–—":
            pieces.append(r"\s*[-‐‑–—]\s*")
        else:
            pieces.append(re.escape(char))
    return r"(?<![A-Za-z0-9])" + "".join(pieces) + r"(?![A-Za-z0-9])"


def find_spans(text: str, needle: str) -> list[tuple[int, int]]:
    if not needle:
        return []
    pattern = re.compile(build_entity_regex(needle), flags=re.IGNORECASE)
    return [(match.start(), match.end()) for match in pattern.finditer(text)]


def spans_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def choose_non_overlapping_spans(
    text: str,
    entity1_text: str,
    entity2_text: str,
) -> tuple[tuple[int, int] | None, tuple[int, int] | None, list[str]]:
    spans1 = find_spans(text, entity1_text)
    spans2 = find_spans(text, entity2_text)
    errors: list[str] = []
    if not spans1:
        errors.append("missing_entity1_text_in_generated_abstract")
    if not spans2:
        errors.append("missing_entity2_text_in_generated_abstract")
    if errors:
        return None, None, errors

    candidates = [
        (abs(span1[0] - span2[0]), span1, span2)
        for span1 in spans1
        for span2 in spans2
        if not spans_overlap(span1, span2)
    ]
    if not candidates:
        return None, None, ["overlapping_entity_mentions_in_generated_abstract"]
    _, best1, best2 = min(candidates, key=lambda item: item[0])
    return best1, best2, []


def marked_text_has_valid_structure(marked_text: str) -> bool:
    src_open = re.findall(r"\[[A-Z0-9]+_SRC\]", marked_text)
    src_close = re.findall(r"\[/[A-Z0-9]+_SRC\]", marked_text)
    tgt_open = re.findall(r"\[[A-Z0-9]+_TGT\]", marked_text)
    tgt_close = re.findall(r"\[/[A-Z0-9]+_TGT\]", marked_text)
    if not (len(src_open) == len(src_close) == len(tgt_open) == len(tgt_close) == 1):
        return False
    tokens = re.findall(r"\[/?[A-Z0-9]+_(?:SRC|TGT)\]", marked_text)
    if len(tokens) != 4:
        return False
    stack: list[str] = []
    for token in tokens:
        if not token.startswith("[/"):
            stack.append(token[1:-1])
            continue
        closing = token[2:-1]
        if not stack or stack.pop() != closing:
            return False
    return not stack


def insert_marker(text: str, span: tuple[int, int], marker: str, role: str) -> tuple[str, tuple[int, int]]:
    start, end = span
    open_marker = f"[{marker}_{role}] "
    close_marker = f" [/{marker}_{role}]"
    marked = text[:start] + open_marker + text[start:end] + close_marker + text[end:]
    shift = len(open_marker) + len(close_marker)
    return marked, (start, end + shift)


def build_marked_text(abstract: str, relation: dict[str, Any]) -> tuple[str, list[str]]:
    text = strip_entity_markers(abstract)
    e1 = relation["entity1_text"]
    e2 = relation["entity2_text"]
    span1, span2, errors = choose_non_overlapping_spans(text, e1, e2)
    if errors:
        return text, errors

    marker1 = marker_prefix(relation["entity1_type"])
    marker2 = marker_prefix(relation["entity2_type"])
    assert span1 is not None and span2 is not None

    # Always preserve BioRED candidate direction: entity1 is SRC and entity2 is TGT.
    # Insert from right to left so offsets remain valid even when entity2 appears first.
    marked = text
    marker_items = [
        (span1, marker1, "SRC"),
        (span2, marker2, "TGT"),
    ]
    for span, marker, role in sorted(marker_items, key=lambda item: item[0][0], reverse=True):
        marked, _ = insert_marker(marked, span, marker, role)
    if not marked_text_has_valid_structure(marked):
        errors.append("invalid_marker_structure")
    return marked, errors


def collect_missing_entity_mentions(abstract: str, source_doc: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    seen: set[str] = set()
    for entity in source_doc.get("entities", []):
        entity_text = clean_cell(entity.get("entity_text")) if isinstance(entity, dict) else ""
        key = entity_text.lower()
        if not entity_text or key in seen:
            continue
        seen.add(key)
        if not find_spans(abstract, entity_text):
            missing.append(entity_text)
    return missing


def repair_twin_missing_entity_mentions(twin: dict[str, str], source_doc: dict[str, Any]) -> list[str]:
    missing = collect_missing_entity_mentions(twin.get("abstract", ""), source_doc)
    if not missing:
        return []
    mention_list = "; ".join(missing)
    repair_sentence = f" Additional indexed biomedical mentions considered for relation extraction included: {mention_list}."
    twin["abstract"] = (twin.get("abstract", "").rstrip() + repair_sentence).strip()
    return missing


def normalize_twin(parsed: Any, raw: dict[str, Any]) -> dict[str, str]:
    if not isinstance(parsed, dict):
        raise ValueError("parsed output is not a JSON object")
    synthetic_id = clean_cell(parsed.get("synthetic_id")) or f"synthetic_twin_{raw['source_doc_id']}_{raw['generation_index']:02d}"
    # The source document ID is control metadata from the original request. Do
    # not trust the model to reproduce it perfectly; one digit drift would break
    # the one-to-one lookup even when the generated abstract itself is usable.
    source_doc_id = str(raw["source_doc_id"])
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
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, str]], list[str], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    validation_errors: list[str] = []
    repair_summary = {"documents_repaired": 0, "mentions_added": 0}
    for twin in twins:
        source_doc = document_lookup.get(str(twin["source_doc_id"]))
        if not source_doc:
            validation_errors.append(f"{twin['synthetic_id']}: source_doc_id not found: {twin['source_doc_id']}")
            continue
        if config.get("repair_missing_entity_mentions"):
            repaired_mentions = repair_twin_missing_entity_mentions(twin, source_doc)
            if repaired_mentions:
                repair_summary["documents_repaired"] += 1
                repair_summary["mentions_added"] += len(repaired_mentions)
        for relation in source_doc["relations"]:
            marked_text, marker_errors = build_marked_text(twin["abstract"], relation)
            row_errors = list(marker_errors)
            if relation["relation_type"] not in allowed_labels:
                row_errors.append("invalid_relation_type")
            if not marked_text_has_valid_structure(marked_text):
                row_errors.append("invalid_marker_structure")
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
    return pd.DataFrame(rows), twins, validation_errors, repair_summary


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
    all_raw_records = list(raw_records)

    twins, parse_errors, failed_records = parse_raw_generations(raw_records)
    retry_summary = {
        "enabled": bool(config.get("retry_failed_generations")),
        "attempted_records": 0,
        "recovered_records": 0,
        "unresolved_records": len(failed_records),
    }
    plain_text_fallback_summary = {
        "enabled": bool(config.get("allow_plain_text_fallback")),
        "attempted_records": 0,
        "recovered_records": 0,
        "failed_records": 0,
    }
    current_failed = failed_records
    if config.get("retry_failed_generations") and failed_records:
        last_parse_errors = parse_errors
        for retry_index in range(1, config["max_parse_retries"] + 1):
            retry_prompt_records: list[dict[str, Any]] = []
            for failed in current_failed:
                source_doc = document_lookup.get(str(failed["source_doc_id"]))
                if not source_doc:
                    continue
                retry_prompt_records.append(
                    {
                        "request_index": failed["request_index"],
                        "source_doc_id": failed["source_doc_id"],
                        "generation_index": failed["generation_index"],
                        "prompt": build_retry_prompt(source_doc, config, int(failed["generation_index"]), retry_index),
                    }
                )
            if not retry_prompt_records:
                break
            print(
                f"Retrying {len(retry_prompt_records)} failed twin generation(s), pass {retry_index}/{config['max_parse_retries']}",
                flush=True,
            )
            retry_config = dict(config)
            retry_config["max_tokens"] = config["retry_max_tokens"]
            retry_config["temperature"] = config["retry_temperature"]
            retry_config["top_p"] = config["retry_top_p"]
            retry_config["do_sample"] = True
            retry_raw_records = generate_text(retry_prompt_records, retry_config)
            for record in retry_raw_records:
                record["retry_index"] = retry_index
            all_raw_records.extend(retry_raw_records)
            retry_twins, last_parse_errors, current_failed = parse_raw_generations(retry_raw_records)
            twins.extend(retry_twins)
            retry_summary["attempted_records"] += len(retry_prompt_records)
            retry_summary["recovered_records"] += len(retry_twins)
            retry_summary["unresolved_records"] = len(current_failed)
            if not current_failed:
                last_parse_errors = []
                break
        parse_errors = last_parse_errors

    if config.get("allow_plain_text_fallback") and retry_summary["unresolved_records"]:
        fallback_twins: list[dict[str, str]] = []
        fallback_errors: list[str] = []
        for raw in current_failed:
            plain_text_fallback_summary["attempted_records"] += 1
            try:
                fallback_twins.append(fallback_plain_text_twin(raw, config["plain_text_fallback_min_chars"]))
            except Exception as exc:
                fallback_errors.append(
                    f"request {raw.get('request_index')} source_doc_id={raw.get('source_doc_id')}: {exc}"
                )
        twins.extend(fallback_twins)
        plain_text_fallback_summary["recovered_records"] = len(fallback_twins)
        plain_text_fallback_summary["failed_records"] = len(fallback_errors)
        retry_summary["unresolved_records"] = len(fallback_errors)
        parse_errors = fallback_errors

    write_jsonl(all_raw_records, output_dir / "raw_generations.jsonl")

    relations, normalized_twins, validation_errors, repair_summary = build_outputs(twins, document_lookup, set(config["allowed_labels"]), config)
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
        "raw_generations": int(len(all_raw_records)),
        "initial_raw_generations": int(len(raw_records)),
        "retry_generation_summary": retry_summary,
        "plain_text_fallback_summary": plain_text_fallback_summary,
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
        "missing_entity_repair": repair_summary,
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
