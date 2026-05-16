import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path
from typing import Dict, List


RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

SPLIT_FILES = {
    "train": "Train.BioC.XML",
    "dev": "Dev.BioC.XML",
    "test": "Test.BioC.XML",
}


def parse_bioc_xml(xml_file: Path) -> List[Dict]:
    """
    Parse BioC XML file and extract one row per relation.
    """
    rows = []

    tree = ET.parse(xml_file)
    root = tree.getroot()

    for doc in root.findall(".//document"):
        doc_id_elem = doc.find("id")
        if doc_id_elem is None:
            continue

        doc_id = doc_id_elem.text

        abstract_text = None
        abstract_offset = 0

        # -------------------
        # Extract abstract text + offset
        # -------------------
        for passage in doc.findall("passage"):
            is_abstract = False

            for infon in passage.findall("infon"):
                if infon.get("key") == "type" and infon.text == "abstract":
                    is_abstract = True
                    break

            if is_abstract:
                text_elem = passage.find("text")
                if text_elem is not None and text_elem.text:
                    abstract_text = text_elem.text.strip()

                location_elem = passage.find("location")
                if location_elem is not None:
                    abstract_offset = int(location_elem.get("offset", 0))

                break

        if abstract_text is None:
            continue

        # -------------------
        # Extract entities
        # -------------------
        entities = {}

        for annotation in doc.findall("passage/annotation"):
            text_elem = annotation.find("text")
            if text_elem is None or not text_elem.text:
                continue

            entity_text = text_elem.text.strip()

            identifier = None
            entity_type = "Unknown"
            offset = None
            length = None

            for infon in annotation.findall("infon"):
                key = infon.get("key")

                if key == "identifier":
                    identifier = infon.text
                elif key == "type":
                    entity_type = infon.text

            location_elem = annotation.find("location")
            if location_elem is not None:
                offset = location_elem.get("offset")
                length = location_elem.get("length")

            if identifier:
                entities[identifier] = {
                    "text": entity_text,
                    "type": entity_type,
                    "identifier": identifier,
                    "offset": int(offset) if offset else None,
                    "length": int(length) if length else None,
                }

        # -------------------
        # Extract relations
        # -------------------
        for relation in doc.findall("relation"):
            infons = {
                infon.get("key"): infon.text
                for infon in relation.findall("infon")
            }

            entity1_id = infons.get("entity1")
            entity2_id = infons.get("entity2")
            relation_type = infons.get("type")

            if not all([entity1_id, entity2_id, relation_type]):
                continue

            if entity1_id not in entities or entity2_id not in entities:
                continue

            entity1 = entities[entity1_id]
            entity2 = entities[entity2_id]

            # -------------------
            # Mark entities
            # -------------------
            if entity1["offset"] is not None and entity2["offset"] is not None:
                local_e1_offset = entity1["offset"] - abstract_offset
                local_e2_offset = entity2["offset"] - abstract_offset

                # If subtracting the offset pushes it negative, the XML offset was likely local!
                if local_e1_offset < 0 or local_e2_offset < 0:
                    local_e1_offset = entity1["offset"]
                    local_e2_offset = entity2["offset"]
                # -----------------------------

                marked_text = mark_entities_by_offsets(
                    abstract_text,
                    local_e1_offset,
                    entity1["text"],
                    local_e2_offset,
                    entity2["text"],
                )
            else:
                marked_text = mark_entities_fallback(
                    abstract_text,
                    entity1["text"],
                    entity2["text"]
                )

            rows.append({
                "doc_id": doc_id,
                "abstract": abstract_text,
                "marked_text": marked_text,
                "entity1": entity1["text"],
                "entity2": entity2["text"],
                "entity1_type": entity1["type"],
                "entity2_type": entity2["type"],
                "entity1_id": entity1["identifier"],
                "entity2_id": entity2["identifier"],
                "relation_type": relation_type,
            })

    return rows


def mark_entities_by_offsets(
    text: str,
    e1_start: int,
    e1_text: str,
    e2_start: int,
    e2_text: str,
) -> str:
    """
    Mark entities using exact offsets.
    Handles overlapping/duplicate names by sorting by descending index.
    """
    entities = [
        ("ENTITY1", e1_start, e1_text),
        ("ENTITY2", e2_start, e2_text),
    ]

    # Sort descending by start offset to prevent indexing shifts
    entities.sort(key=lambda x: x[1], reverse=True)
    marked = text

    success = True
    for tag, start, entity_text in entities:
        end = start + len(entity_text)

        # Verify offset bounds and exact text match
        if 0 <= start <= len(marked) and marked[start:end] == entity_text:
            marked = (
                marked[:start]
                + f"[{tag}] {entity_text} [/{tag}]"
                + marked[end:]
            )
        else:
            success = False
            break  # Break out and trigger fallback

    if success:
        return marked
    
    # If offset matching fails, use standard string matching fallback
    return mark_entities_fallback(text, e1_text, e2_text)


def mark_entities_fallback(text: str, entity1: str, entity2: str) -> str:
    """
    Fallback option that safely marks strings even if offsets are broken.
    """
    marked = text
    
    # use a unique temporary token to avoid accidentally marking 
    # a substring inside an already marked entity.
    if entity1 in marked:
        marked = marked.replace(entity1, f"[ENTITY1] {entity1} [/ENTITY1]", 1)
    if entity2 in marked:
        marked = marked.replace(entity2, f"[ENTITY2] {entity2} [/ENTITY2]", 1)
        
    return marked


def process_split(split_name: str, filename: str):
    """
    Process one dataset split and save CSV.
    """
    input_path = RAW_DATA_DIR / filename
    output_path = PROCESSED_DATA_DIR / f"{split_name}.csv"

    print(f"\nProcessing {split_name}: {input_path}")

    rows = parse_bioc_xml(input_path)

    if not rows:
        print(f"No rows found for {split_name}")
        return

    df = pd.DataFrame(rows)
    df["split"] = split_name

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved {len(df)} samples to {output_path}")
    print(f"Unique documents: {df['doc_id'].nunique()}")
    print(f"Relation classes: {df['relation_type'].nunique()}")

    print("\nRelation distribution:")
    print(df["relation_type"].value_counts())


def main():
    for split_name, filename in SPLIT_FILES.items():
        process_split(split_name, filename)

    print("\nDone preprocessing BioRED.")


if __name__ == "__main__":
    main()