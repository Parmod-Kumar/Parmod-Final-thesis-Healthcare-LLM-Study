"""
01_data_loader.py - Data loading and preprocessing
Self-contained version – runs without external modules.

This module standardises healthcare QA datasets into a single schema so the
prompting, fine-tuning, and evaluation scripts can operate on consistent input
records.
"""

import os
import random
import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def load_healthcare_dataset(dataset_name: str = "pubmedqa"):
    """
    Load PubMedQA or MedQA dataset and return a dict with train/validation/test splits.

    The function normalises the data format so downstream modules can treat each
    record as a question-answer pair with a consistent schema.
    """
    if dataset_name == "pubmedqa":
        dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
        data = {"train": [], "validation": [], "test": []}
        # If validation/test are missing, split train artificially
        if "validation" not in dataset or "test" not in dataset:
            all_items = list(dataset["train"])
            train_items, holdout = train_test_split(all_items, test_size=0.30, random_state=42)
            val_items, test_items = train_test_split(holdout, test_size=0.50, random_state=42)
            split_map = {"train": train_items, "validation": val_items, "test": test_items}
        else:
            split_map = {
                "train": dataset["train"],
                "validation": dataset["validation"],
                "test": dataset["test"],
            }

        for split, items in split_map.items():
            for item in items:
                data[split].append({
                    "question": item["question"],
                    "context": item.get("context", ""),
                    "answer": item.get("final_decision", item.get("answer", "")),
                    "label": item.get("final_decision", "")
                })
        return data

    elif dataset_name == "medqa":
        dataset = load_dataset("GBaker/MedQA-USMLE-4-options")
        data = {"train": [], "validation": [], "test": []}
        # Similar fallback
        if "validation" not in dataset:
            train_items = list(dataset["train"])
            if "test" in dataset:
                test_items = list(dataset["test"])
                train_items, val_items = train_test_split(train_items, test_size=0.15, random_state=42)
            else:
                train_items, holdout = train_test_split(train_items, test_size=0.30, random_state=42)
                val_items, test_items = train_test_split(holdout, test_size=0.50, random_state=42)
            split_map = {"train": train_items, "validation": val_items, "test": test_items}
        else:
            split_map = {
                "train": dataset["train"],
                "validation": dataset["validation"],
                "test": dataset["test"],
            }

        for split, items in split_map.items():
            for item in items:
                options = item["options"]
                option_text = " ".join([f"{k}) {v}" for k, v in options.items()])
                data[split].append({
                    "question": f"{item['question']}\n\nOptions: {option_text}",
                    "answer": item["answer_idx"],
                    "label": item["answer_idx"]
                })
        return data

    else:
        raise ValueError(f"Dataset {dataset_name} not supported. Use 'pubmedqa' or 'medqa'.")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
class Config:
    """Simple configuration object for the data-loading pipeline."""
    dataset_name = "pubmedqa"
    data_dir = "./data"
    # You can add more config values as needed

config = Config()

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess_dataset(data: dict) -> dict:
    """Clean and normalise each record before writing it to disk."""
    processed = {}
    for split, items in data.items():
        processed[split] = []
        for item in items:
            question = item["question"].strip()
            answer = item["answer"].strip() if item["answer"] else ""
            processed[split].append({
                "question": question,
                "answer": answer,
                "label": item.get("label", answer)
            })
    return processed

def save_processed_data(data: dict, output_dir: str = "./data") -> None:
    """Save processed data to CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    for split, items in data.items():
        df = pd.DataFrame(items)
        df.to_csv(os.path.join(output_dir, f"{split}.csv"), index=False)
        print(f"Saved {split} split: {len(df)} samples")

# -------------------- Main --------------------
def main():
    print("=" * 60)
    print("DATA LOADING AND PREPROCESSING")
    print("=" * 60)

    # Set seed for reproducibility
    set_seed(42)

    # Load dataset
    print(f"Loading dataset: {config.dataset_name}")
    raw_data = load_healthcare_dataset(config.dataset_name)

    print(f"Train samples: {len(raw_data['train'])}")
    print(f"Validation samples: {len(raw_data['validation'])}")
    print(f"Test samples: {len(raw_data['test'])}")

    # Preprocess
    processed_data = preprocess_dataset(raw_data)

    # Save processed data
    save_processed_data(processed_data, config.data_dir)

    print("\nData loading complete!")

if __name__ == "__main__":
    main()