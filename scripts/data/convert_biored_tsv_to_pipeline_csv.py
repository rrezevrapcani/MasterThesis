import sys
import pandas as pd

split = sys.argv[1]

IN_FILE = f"datasets/biored/processed/{split}.tsv"
OUT_FILE = f"datasets/biored/processed/{split}_for_my_pipeline.csv"

def convert_tags(text):
    replacements = {
        "@ChemicalEntitySrc$": "[CHEMICAL_SRC]",
        "@/ChemicalEntitySrc$": "[/CHEMICAL_SRC]",
        "@ChemicalEntityTgt$": "[CHEMICAL_TGT]",
        "@/ChemicalEntityTgt$": "[/CHEMICAL_TGT]",
        "@GeneOrGeneProductSrc$": "[GENE_SRC]",
        "@/GeneOrGeneProductSrc$": "[/GENE_SRC]",
        "@GeneOrGeneProductTgt$": "[GENE_TGT]",
        "@/GeneOrGeneProductTgt$": "[/GENE_TGT]",
        "@DiseaseOrPhenotypicFeatureSrc$": "[DISEASE_SRC]",
        "@/DiseaseOrPhenotypicFeatureSrc$": "[/DISEASE_SRC]",
        "@DiseaseOrPhenotypicFeatureTgt$": "[DISEASE_TGT]",
        "@/DiseaseOrPhenotypicFeatureTgt$": "[/DISEASE_TGT]",
        "@SequenceVariantSrc$": "[ENTITY_SRC]",
        "@/SequenceVariantSrc$": "[/ENTITY_SRC]",
        "@SequenceVariantTgt$": "[ENTITY_TGT]",
        "@/SequenceVariantTgt$": "[/ENTITY_TGT]",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

df = pd.read_csv(IN_FILE, sep="\t", header=None, dtype=str, keep_default_na=False)

out = pd.DataFrame()
out["doc_id"] = df.iloc[:, 0]
out["marked_text"] = df.iloc[:, 7].apply(convert_tags)
out["relation_type"] = df.iloc[:, 8].replace({"None": "No_Relation"})

out.to_csv(OUT_FILE, index=False)

print(f"Saved {len(out)} rows to {OUT_FILE}")
print(out["relation_type"].value_counts())