"""
utils.py - Utility functions for the study
Author: Parmod Kumar
Date: 13/Sept/2026
"""

import os
import json
import random
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_healthcare_dataset(dataset_name: str = "pubmedqa") -> Dict[str, Any]:
    """
    Load healthcare QA dataset.
    
    Supported datasets:
    - pubmedqa: PubMedQA biomedical QA
    - medqa: MedQA USMLE-style questions
    - mimic-qa: MIMIC clinical QA (if available)
    """
    if dataset_name == "pubmedqa":
        # Load PubMedQA dataset
        dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled")

        # Normalise splits across datasets that may not expose validation/test.
        data = {"train": [], "validation": [], "test": []}
        available_splits = list(dataset.keys())

        if "validation" not in available_splits or "test" not in available_splits:
            base_split = "train" if "train" in dataset else available_splits[0]
            all_items = list(dataset[base_split])
            train_items, holdout_items = train_test_split(all_items, test_size=0.30, random_state=42)
            val_items, test_items = train_test_split(holdout_items, test_size=0.50, random_state=42)
            split_map = {
                "train": train_items,
                "validation": val_items,
                "test": test_items,
            }
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
                    "answer": item["final_decision"] if "final_decision" in item else item.get("answer", ""),
                    "label": item.get("final_decision", "")
                })
        
        return data
    
    elif dataset_name == "medqa":
        # Load MedQA dataset
        dataset = load_dataset("GBaker/MedQA-USMLE-4-options")

        data = {"train": [], "validation": [], "test": []}

        available_splits = list(dataset.keys())
        if "validation" not in available_splits:
            base_split = "train" if "train" in dataset else available_splits[0]
            train_items = list(dataset[base_split])
            if "test" in dataset:
                test_items = list(dataset["test"])
                train_items, val_items = train_test_split(train_items, test_size=0.15, random_state=42)
            else:
                train_items, holdout_items = train_test_split(train_items, test_size=0.30, random_state=42)
                val_items, test_items = train_test_split(holdout_items, test_size=0.50, random_state=42)

            split_map = {
                "train": train_items,
                "validation": val_items,
                "test": test_items,
            }
        else:
            split_map = {
                "train": dataset["train"],
                "validation": dataset["validation"],
                "test": dataset["test"],
            }

        for split, items in split_map.items():
            for item in items:
                options = item["options"]
                # Combine options into a single string
                option_text = " ".join([f"{k}) {v}" for k, v in options.items()])
                data[split].append({
                    "question": f"{item['question']}\n\nOptions: {option_text}",
                    "answer": item["answer_idx"],
                    "label": item["answer_idx"]
                })
        
        return data
    
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")


def prepare_prompt_templates() -> Dict[str, str]:
    """Define prompt templates for different prompting strategies."""
    
    templates = {
        "zero_shot": """[Instruction] Answer the following medical question accurately.

[Question] {question}

[Answer]:""",
        
        "few_shot": """[Instruction] Answer the following medical questions. Here are some examples:

Example 1:
Question: {example_question_1}
Answer: {example_answer_1}

Example 2:
Question: {example_question_2}
Answer: {example_answer_2}

Now answer this question:
Question: {question}
Answer:""",
        
        "chain_of_thought": """[Instruction] You are a medical expert. Please reason through the following question step by step before providing your final answer.

[Question] {question}

[Reasoning Steps]:
Step 1: ...
Step 2: ...

[Final Answer]:""",
        
        "role_based": """[Instruction] You are a senior medical expert with specialised knowledge in {domain}. Please answer the following medical question based on your expertise.

[Question] {question}

[Answer]:""",
        
        "prompt_ensemble": """[Instruction] You are a medical expert. Provide a comprehensive answer to the following question.

[Question] {question}

[Answer]:"""
    }
    
    return templates


def format_few_shot_examples(examples: List[Dict], num_shots: int = 3) -> Dict[str, str]:
    """Select and format few-shot examples."""
    selected = random.sample(examples, min(num_shots, len(examples)))
    
    formatted = {}
    for i, ex in enumerate(selected, 1):
        formatted[f"example_question_{i}"] = ex["question"]
        formatted[f"example_answer_{i}"] = ex["answer"]
    
    return formatted


def calculate_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Calculate evaluation metrics."""
    from sklearn.metrics import accuracy_score, f1_score

    if not predictions or not references:
        return {
            "exact_match": 0.0,
            "f1": 0.0
        }
    
    # Exact match accuracy
    exact_match = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip()) / len(predictions)
    
    # F1 score (for classification tasks)
    try:
        f1 = f1_score(references, predictions, average="weighted")
    except:
        f1 = 0.0
    
    return {
        "exact_match": exact_match,
        "f1": f1
    }


def save_results(results: Dict, filename: str, output_dir: str = "./results") -> None:
    """Save results to CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, filename), index=False)
    print(f"Results saved to {os.path.join(output_dir, filename)}")
