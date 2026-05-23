import os
import csv
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


def merge_csvs_to_df(folder):
    """Merge all CSV files from folder and subfolders."""
    rows = []

    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith('.csv') and file != 'test_summary.csv' and file != 'train_summary.csv':
                path = os.path.join(root, file)
                with open(path, newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        rows.append(row)

    df = pd.DataFrame(rows)

    # Ensure correct types for sorting
    df['doc_id'] = df['doc_id'].astype(str)
    df['entity1_offset'] = pd.to_numeric(df['entity1_offset'])
    df['entity2_offset'] = pd.to_numeric(df['entity2_offset'])

    # Sort by doc_id, then offsets
    df = df.sort_values(by=['doc_id', 'entity1_offset', 'entity2_offset']).reset_index(drop=True)

    # Remove duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)

    return df


def prepare_binary_classification_data(df_train, df_test, dev_size=0.15, seed=42):
    """
    Prepare binary classification data (Relation vs No_Relation).
    Creates train, dev, test splits.
    """
    np.random.seed(seed)

    # Create binary labels
    df_train['binary_label'] = df_train['relation_type'].apply(
        lambda x: 'No_Relation' if x == 'No_Relation' else 'Relation'
    )
    df_test['binary_label'] = df_test['relation_type'].apply(
        lambda x: 'No_Relation' if x == 'No_Relation' else 'Relation'
    )

    # Keep only necessary columns
    cols_to_keep = ['doc_id', 'passage_text', 'entity1_text', 'entity1_type',
                    'entity2_text', 'entity2_type', 'binary_label']

    df_train_binary = df_train[cols_to_keep].copy()
    df_test_binary = df_test[cols_to_keep].copy()

    # Split train into train and dev
    df_train_split, df_dev = train_test_split(
        df_train_binary, test_size=dev_size, random_state=seed,
        stratify=df_train_binary['binary_label']
    )

    # Rename column for consistency
    df_train_split.columns = [c if c != 'binary_label' else 'relation_type' for c in df_train_split.columns]
    df_dev.columns = [c if c != 'binary_label' else 'relation_type' for c in df_dev.columns]
    df_test_binary.columns = [c if c != 'binary_label' else 'relation_type' for c in df_test_binary.columns]

    return df_train_split, df_dev, df_test_binary


def prepare_relation_specific_data(df_train, df_test, dev_size=0.15, seed=42):
    """
    Prepare multi-class classification data (relation types).
    Only uses samples where relation_type != 'No_Relation'.
    """
    np.random.seed(seed)

    # Filter to only relation samples
    df_train_relations = df_train[df_train['relation_type'] != 'No_Relation'].copy()
    df_test_relations = df_test[df_test['relation_type'] != 'No_Relation'].copy()

    # Keep only necessary columns
    cols_to_keep = ['doc_id', 'passage_text', 'entity1_text', 'entity1_type',
                    'entity2_text', 'entity2_type', 'relation_type']

    df_train_relations = df_train_relations[cols_to_keep].copy()
    df_test_relations = df_test_relations[cols_to_keep].copy()

    # Split train into train and dev
    df_train_split, df_dev = train_test_split(
        df_train_relations, test_size=dev_size, random_state=seed,
        stratify=df_train_relations['relation_type']
    )

    return df_train_split, df_dev, df_test_relations


def main():
    # Paths
    sorted_data_path = Path("data/Sorted data/relation_specific")
    output_dir = Path("data/processed/two_stage")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading sorted data...")
    df_train = merge_csvs_to_df(str(sorted_data_path / "relation_specific_train"))
    df_test = merge_csvs_to_df(str(sorted_data_path / "relation_specific_test"))

    print(f"Train shape: {df_train.shape}")
    print(f"Test shape: {df_test.shape}")
    print(f"\nTrain relation distribution:\n{df_train['relation_type'].value_counts()}")
    print(f"\nTest relation distribution:\n{df_test['relation_type'].value_counts()}")

    # Prepare Stage 1: Binary classification
    print("\n" + "="*50)
    print("Preparing Stage 1: Binary Classification")
    print("="*50)

    train_binary, dev_binary, test_binary = prepare_binary_classification_data(df_train, df_test)

    stage1_dir = output_dir / "stage1_binary"
    stage1_dir.mkdir(parents=True, exist_ok=True)

    train_binary.to_csv(stage1_dir / "train.csv", index=False)
    dev_binary.to_csv(stage1_dir / "dev.csv", index=False)
    test_binary.to_csv(stage1_dir / "test.csv", index=False)

    print(f"Train size: {len(train_binary)}")
    print(f"Dev size: {len(dev_binary)}")
    print(f"Test size: {len(test_binary)}")
    print(f"Binary distribution:\n{train_binary['relation_type'].value_counts()}")
    print(f"Saved to: {stage1_dir}")

    # Prepare Stage 2: Relation-specific classification
    print("\n" + "="*50)
    print("Preparing Stage 2: Relation-Specific Classification")
    print("="*50)

    train_relations, dev_relations, test_relations = prepare_relation_specific_data(df_train, df_test)

    stage2_dir = output_dir / "stage2_relation_types"
    stage2_dir.mkdir(parents=True, exist_ok=True)

    train_relations.to_csv(stage2_dir / "train.csv", index=False)
    dev_relations.to_csv(stage2_dir / "dev.csv", index=False)
    test_relations.to_csv(stage2_dir / "test.csv", index=False)

    print(f"Train size: {len(train_relations)}")
    print(f"Dev size: {len(dev_relations)}")
    print(f"Test size: {len(test_relations)}")
    print(f"Relation distribution:\n{train_relations['relation_type'].value_counts()}")
    print(f"Saved to: {stage2_dir}")

    print("\n" + "="*50)
    print("Data preparation complete!")
    print("="*50)


if __name__ == "__main__":
    main()
