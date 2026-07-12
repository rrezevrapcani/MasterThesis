"""Convert BioRED TSV splits to the repository's enriched pipeline CSV format."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

COLUMNS = [
    "doc_id", "entity1_type", "entity2_type", "entity1_id", "entity2_id",
    "same_sentence", "pair_distance", "raw_marked_text", "relation_type", "novelty",
]
TAG_REPLACEMENTS = {
    "@ChemicalEntitySrc$": "[CHEMICAL_SRC]", "@/ChemicalEntitySrc$": "[/CHEMICAL_SRC]",
    "@ChemicalEntityTgt$": "[CHEMICAL_TGT]", "@/ChemicalEntityTgt$": "[/CHEMICAL_TGT]",
    "@GeneOrGeneProductSrc$": "[GENE_SRC]", "@/GeneOrGeneProductSrc$": "[/GENE_SRC]",
    "@GeneOrGeneProductTgt$": "[GENE_TGT]", "@/GeneOrGeneProductTgt$": "[/GENE_TGT]",
    "@DiseaseOrPhenotypicFeatureSrc$": "[DISEASE_SRC]",
    "@/DiseaseOrPhenotypicFeatureSrc$": "[/DISEASE_SRC]",
    "@DiseaseOrPhenotypicFeatureTgt$": "[DISEASE_TGT]",
    "@/DiseaseOrPhenotypicFeatureTgt$": "[/DISEASE_TGT]",
    "@SequenceVariantSrc$": "[ENTITY_SRC]", "@/SequenceVariantSrc$": "[/ENTITY_SRC]",
    "@SequenceVariantTgt$": "[ENTITY_TGT]", "@/SequenceVariantTgt$": "[/ENTITY_TGT]",
}


def convert_tags(text: str) -> str:
    for old, new in TAG_REPLACEMENTS.items():
        text = text.replace(old, new)
    return text


def extract_first_mention(text: str, role: str) -> str | None:
    """Return the first source/target surface form from BioRED inline tags."""
    pattern = rf"@([^/$]+){role}\$\s*(.*?)\s*@/\1{role}\$"
    match = re.search(pattern, text, flags=re.DOTALL)
    return re.sub(r"\s+", " ", match.group(2)).strip() if match else None


def novelty_to_boolean(value: str):
    if value == "Novel":
        return True
    if value == "No":
        return False
    return pd.NA


def convert_split(split: str) -> Path:
    input_path = Path(f"datasets/biored/processed/{split}.tsv")
    output_path = Path(f"datasets/biored/processed/{split}_for_my_pipeline.csv")
    raw = pd.read_csv(input_path, sep="\t", header=None, dtype=str, keep_default_na=False)
    if raw.shape[1] < len(COLUMNS):
        raise ValueError(f"Expected at least {len(COLUMNS)} TSV columns, found {raw.shape[1]}")
    raw = raw.iloc[:, :len(COLUMNS)].copy()
    raw.columns = COLUMNS

    out = pd.DataFrame({
        "doc_id": raw["doc_id"],
        "marked_text": raw["raw_marked_text"].map(convert_tags),
        "relation_type": raw["relation_type"].replace({"None": "No_Relation"}),
        "entity1_text": raw["raw_marked_text"].map(lambda text: extract_first_mention(text, "Src")),
        "entity2_text": raw["raw_marked_text"].map(lambda text: extract_first_mention(text, "Tgt")),
        "entity1_id": raw["entity1_id"],
        "entity2_id": raw["entity2_id"],
        "entity1_type": raw["entity1_type"],
        "entity2_type": raw["entity2_type"],
        "entity_pair_type": raw["entity1_type"] + "__" + raw["entity2_type"],
        "is_novel": raw["novelty"].map(novelty_to_boolean).astype("boolean"),
    })

    # Prove enrichment cannot change inputs or labels used by existing training runs.
    if output_path.exists():
        previous = pd.read_csv(output_path, dtype={"doc_id": str})
        critical = ["doc_id", "marked_text", "relation_type"]
        pd.testing.assert_frame_equal(
            previous[critical].reset_index(drop=True),
            out[critical].reset_index(drop=True),
            check_dtype=False,
            obj=f"training-critical columns in {split}",
        )

    out.to_csv(output_path, index=False)
    print(f"Saved {len(out):,} enriched rows to {output_path}")
    print(out["relation_type"].value_counts().to_string())
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=["train", "dev", "test", "all"])
    args = parser.parse_args()
    for split in (["train", "dev", "test"] if args.split == "all" else [args.split]):
        convert_split(split)


if __name__ == "__main__":
    main()
