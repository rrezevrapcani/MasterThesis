#!/usr/bin/env python3
"""Generate synthetic biomedical relation-extraction data with vLLM."""
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

REQUIRED_CONFIG_KEYS = (
    "prompt_file",
    "output_dir",
    "model_name",
    "requested_number_of_abstracts",
    "allowed_labels",
)

MARKER_RE = re.compile(r"\[[A-Z0-9]+_(?:SRC|TGT)\].+?\[/[A-Z0-9]+_(?:SRC|TGT)\]", re.DOTALL)


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8-sig") as stream:
        config = yaml.safe_load(stream) or {}

    missing = [key for key in REQUIRED_CONFIG_KEYS if config.get(key) is None]
    if missing:
        raise ValueError(f"Missing required config value(s): {', '.join(missing)}")

    config["requested_number_of_abstracts"] = int(config["requested_number_of_abstracts"])
    config["backend"] = str(config.get("backend", "vllm")).lower()
    config["temperature"] = float(config.get("temperature", 0.7))
    config["top_p"] = float(config.get("top_p", 0.9))
    config["max_tokens"] = int(config.get("max_tokens", 4096))
    config["do_sample"] = bool(config.get("do_sample", True))
    config["seed"] = int(config.get("seed", 42))
    config["tensor_parallel_size"] = int(config.get("tensor_parallel_size", 1))
    config["gpu_memory_utilization"] = float(config.get("gpu_memory_utilization", 0.9))
    config["trust_remote_code"] = bool(config.get("trust_remote_code", True))
    config["dtype"] = str(config.get("dtype", "auto"))

    if config["requested_number_of_abstracts"] <= 0:
        raise ValueError("requested_number_of_abstracts must be positive")
    if config["max_tokens"] <= 0:
        raise ValueError("max_tokens must be positive")
    if config["tensor_parallel_size"] <= 0:
        raise ValueError("tensor_parallel_size must be positive")
    if config["backend"] not in {"vllm", "transformers"}:
        raise ValueError("backend must be either 'vllm' or 'transformers'")
    return config


def read_text(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path.resolve()}")
    return path.read_text(encoding="utf-8")


def load_prompt(config: dict[str, Any]) -> str:
    prompt = read_text(Path(config["prompt_file"]).expanduser())
    prompt = prompt.replace(
        "Generate 3 synthetic biomedical abstracts.",
        f"Generate {config['requested_number_of_abstracts']} synthetic biomedical abstracts.",
    )
    return prompt


def import_vllm() -> tuple[Any, Any]:
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise ImportError(
            "vLLM is not installed in this environment. Run this script in the GPU/vLLM conda "
            "environment or install vllm before running synthetic generation."
        ) from exc
    return LLM, SamplingParams


def generate_with_vllm(prompt: str, config: dict[str, Any]) -> list[dict[str, Any]]:
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
    outputs = llm.generate([prompt], sampling_params)
    records: list[dict[str, Any]] = []
    for request_index, output in enumerate(outputs):
        for candidate_index, candidate in enumerate(output.outputs):
            records.append(
                {
                    "request_index": request_index,
                    "candidate_index": candidate_index,
                    "text": candidate.text,
                    "finish_reason": getattr(candidate, "finish_reason", None),
                }
            )
    return records


def generate_with_transformers(prompt: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    except ImportError as exc:
        raise ImportError(
            "transformers and torch are required for backend='transformers'. "
            "Run this in an environment with torch/transformers installed."
        ) from exc

    set_seed(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"],
        trust_remote_code=config["trust_remote_code"],
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype: Any = "auto"
    if config["dtype"] != "auto":
        dtype_name = str(config["dtype"])
        torch_dtype = getattr(torch, dtype_name) if hasattr(torch, dtype_name) else dtype_name

    cuda_available = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        torch_dtype=torch_dtype,
        device_map="auto" if cuda_available else None,
        trust_remote_code=config["trust_remote_code"],
    )
    if not cuda_available:
        model = model.to("cpu")
    model.eval()

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        input_ids = encoded.to(model.device)
        attention_mask = None
    else:
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)

    generation_kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "max_new_tokens": config["max_tokens"],
        "do_sample": config["do_sample"],
        "temperature": config["temperature"],
        "top_p": config["top_p"],
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if attention_mask is not None:
        generation_kwargs["attention_mask"] = attention_mask

    with torch.no_grad():
        output_ids = model.generate(**generation_kwargs)

    generated_ids = output_ids[0, input_ids.shape[-1]:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return [
        {
            "request_index": 0,
            "candidate_index": 0,
            "text": text,
            "finish_reason": None,
        }
    ]


def generate_text(prompt: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    if config["backend"] == "vllm":
        return generate_with_vllm(prompt, config)
    if config["backend"] == "transformers":
        return generate_with_transformers(prompt, config)
    raise ValueError(f"Unsupported backend: {config['backend']}")


def extract_first_json_object(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)

    decoder = json.JSONDecoder()
    starts = [index for index, char in enumerate(stripped) if char in "[{"]
    errors: list[str] = []
    for start in starts:
        try:
            value, _ = decoder.raw_decode(stripped[start:])
            return value
        except json.JSONDecodeError as exc:
            errors.append(str(exc))
    raise ValueError(f"No valid JSON object or array found in model output. Last parser errors: {errors[-3:]}")


def normalize_generation(parsed: Any) -> list[dict[str, Any]]:
    if isinstance(parsed, dict) and isinstance(parsed.get("synthetic_abstracts"), list):
        abstracts = parsed["synthetic_abstracts"]
    elif isinstance(parsed, list):
        abstracts = parsed
    elif isinstance(parsed, dict) and {"synthetic_id", "abstract", "relations"}.issubset(parsed):
        abstracts = [parsed]
    else:
        raise ValueError("Parsed JSON does not contain a synthetic_abstracts list or a single abstract object")

    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(abstracts):
        if not isinstance(item, dict):
            raise ValueError(f"Generated abstract at index {index} is not a JSON object")
        normalized.append(item)
    return normalized


def has_valid_marker_pair(marked_text: str) -> bool:
    if not marked_text:
        return False
    has_src = "_SRC]" in marked_text and "[/" in marked_text
    has_tgt = "_TGT]" in marked_text and "[/" in marked_text
    return has_src and has_tgt and bool(MARKER_RE.search(marked_text))


def validate_abstracts(abstracts: list[dict[str, Any]], config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    allowed_labels = set(config.get("allowed_labels") or [])
    allowed_entity_types = set(config.get("allowed_entity_types") or [])
    validated: list[dict[str, Any]] = []
    errors: list[str] = []

    for abstract_index, abstract in enumerate(abstracts):
        synthetic_id = str(abstract.get("synthetic_id") or f"synthetic_{abstract_index + 1:04d}")
        abstract["synthetic_id"] = synthetic_id

        entities = abstract.get("entities")
        relations = abstract.get("relations")
        if not isinstance(entities, list):
            errors.append(f"{synthetic_id}: entities is missing or not a list")
            entities = []
            abstract["entities"] = entities
        if not isinstance(relations, list):
            errors.append(f"{synthetic_id}: relations is missing or not a list")
            relations = []
            abstract["relations"] = relations

        entity_ids = set()
        for entity_index, entity in enumerate(entities):
            if not isinstance(entity, dict):
                errors.append(f"{synthetic_id}: entity {entity_index} is not an object")
                continue
            entity_id = str(entity.get("entity_id") or "").strip()
            entity_type = str(entity.get("entity_type") or "").strip()
            if entity_id:
                entity_ids.add(entity_id)
            else:
                errors.append(f"{synthetic_id}: entity {entity_index} has empty entity_id")
            if allowed_entity_types and entity_type and entity_type not in allowed_entity_types:
                errors.append(f"{synthetic_id}: entity {entity_id} has unsupported type {entity_type!r}")

        for relation_index, relation in enumerate(relations):
            if not isinstance(relation, dict):
                errors.append(f"{synthetic_id}: relation {relation_index} is not an object")
                continue
            relation_type = str(relation.get("relation_type") or "").strip()
            entity1_id = str(relation.get("entity1_id") or "").strip()
            entity2_id = str(relation.get("entity2_id") or "").strip()
            marked_text = str(relation.get("marked_text") or "").strip()
            if relation_type not in allowed_labels:
                errors.append(f"{synthetic_id}: relation {relation_index} has unsupported label {relation_type!r}")
            if entity1_id and entity1_id not in entity_ids:
                errors.append(f"{synthetic_id}: relation {relation_index} entity1_id {entity1_id!r} is not in entities")
            if entity2_id and entity2_id not in entity_ids:
                errors.append(f"{synthetic_id}: relation {relation_index} entity2_id {entity2_id!r} is not in entities")
            if not entity1_id or not entity2_id:
                errors.append(f"{synthetic_id}: relation {relation_index} has missing entity IDs")
            if not has_valid_marker_pair(marked_text):
                errors.append(f"{synthetic_id}: relation {relation_index} has missing or invalid source/target markers")

        validated.append(abstract)
    return validated, errors


def flatten_relations(abstracts: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for abstract in abstracts:
        entity_lookup = {
            str(entity.get("entity_id", "")): entity
            for entity in abstract.get("entities", [])
            if isinstance(entity, dict)
        }
        for relation_index, relation in enumerate(abstract.get("relations", [])):
            if not isinstance(relation, dict):
                continue
            entity1 = entity_lookup.get(str(relation.get("entity1_id", "")), {})
            entity2 = entity_lookup.get(str(relation.get("entity2_id", "")), {})
            rows.append(
                {
                    "synthetic_id": abstract.get("synthetic_id", ""),
                    "title": abstract.get("title", ""),
                    "abstract": abstract.get("abstract", ""),
                    "relation_index": relation_index,
                    "entity1_id": relation.get("entity1_id", ""),
                    "entity2_id": relation.get("entity2_id", ""),
                    "entity1_text": entity1.get("entity_text", ""),
                    "entity2_text": entity2.get("entity_text", ""),
                    "entity1_type": entity1.get("entity_type", ""),
                    "entity2_type": entity2.get("entity_type", ""),
                    "relation_type": relation.get("relation_type", ""),
                    "marked_text": relation.get("marked_text", ""),
                }
            )
    return pd.DataFrame(rows)


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
    prompt = load_prompt(config)

    raw_records = generate_text(prompt, config)
    write_jsonl(raw_records, output_dir / "raw_generations.jsonl")

    parse_errors: list[str] = []
    all_abstracts: list[dict[str, Any]] = []
    for record in raw_records:
        try:
            parsed = extract_first_json_object(record["text"])
            all_abstracts.extend(normalize_generation(parsed))
        except Exception as exc:
            parse_errors.append(f"candidate {record['candidate_index']}: {exc}")

    validated_abstracts, validation_errors = validate_abstracts(all_abstracts, config)
    relations = flatten_relations(validated_abstracts)

    write_json(validated_abstracts, output_dir / "synthetic_abstracts.json")
    relations.to_csv(output_dir / "synthetic_relations.csv", index=False)
    write_json(config, output_dir / "synthetic_generation_config.json")

    label_counts = relations["relation_type"].value_counts().to_dict() if "relation_type" in relations else {}
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_name": config["model_name"],
        "requested_number_of_abstracts": config["requested_number_of_abstracts"],
        "number_of_raw_generations": len(raw_records),
        "number_of_parsed_abstracts": len(validated_abstracts),
        "number_of_relation_rows": int(len(relations)),
        "label_counts": label_counts,
        "parse_error_count": len(parse_errors),
        "validation_error_count": len(validation_errors),
        "parse_errors": parse_errors,
        "validation_errors": validation_errors,
        "outputs": {
            "raw_generations": "raw_generations.jsonl",
            "synthetic_abstracts": "synthetic_abstracts.json",
            "synthetic_relations": "synthetic_relations.csv",
            "synthetic_generation_config": "synthetic_generation_config.json",
        },
    }
    write_json(summary, output_dir / "generation_summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Path to synthetic generation YAML")
    args = parser.parse_args()
    summary = run(load_config(args.config))
    print(
        "Synthetic generation complete: "
        f"{summary['number_of_parsed_abstracts']} parsed abstracts, "
        f"{summary['number_of_relation_rows']} relation rows, "
        f"{summary['validation_error_count']} validation issue(s)."
    )
    if summary["parse_error_count"]:
        print(f"Parse errors: {summary['parse_error_count']} -- see generation_summary.json")
    if summary["validation_error_count"]:
        print(f"Validation issues: {summary['validation_error_count']} -- see generation_summary.json")


if __name__ == "__main__":
    main()
