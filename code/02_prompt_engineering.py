"""
02_prompt_engineering.py - Prompt Engineering Experiments
Self‑contained version – runs without external modules.

This script evaluates several prompting strategies on a small healthcare-style
QA dataset using a lightweight causal LM. The intent is to compare zero-shot,
few-shot, chain-of-thought, role-based, and ensemble prompting behaviour under
consistent conditions.
"""

import os
import time
import random
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import Counter
import numpy as np

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42):
    """Set deterministic seeds for reproducible prompt-engineering runs."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def save_results(results, filename, output_dir="./results"):
    """Write a list of result dictionaries to a CSV file in the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, filename), index=False)
    print(f"Results saved to {os.path.join(output_dir, filename)}")

def prepare_prompt_templates():
    """Create the reusable prompt templates used across experiments."""
    return {
        "zero_shot": "[Instruction] Answer the following medical question accurately.\n\n[Question] {question}\n\n[Answer]:",
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
        "chain_of_thought": """[Instruction] You are a medical expert. Please reason step by step before providing your final answer.

[Question] {question}

[Reasoning]:
Step 1: ...
Step 2: ...

[Final Answer]:""",
        "role_based": """[Instruction] You are a senior medical expert in {domain}. Answer the following question based on your expertise.

[Question] {question}

[Answer]:""",
    }

def format_few_shot_examples(examples, num_shots=3):
    """Select random examples and return as dict with keys example_question_1, example_answer_1, etc."""
    if not examples:
        return {}
    selected = random.sample(examples, min(num_shots, len(examples)))
    formatted = {}
    for i, ex in enumerate(selected, 1):
        formatted[f"example_question_{i}"] = ex.get("question", "")
        formatted[f"example_answer_{i}"] = ex.get("answer", "")
    return formatted

def load_healthcare_dataset(dataset_name="pubmedqa"):
    """Load a small synthetic dataset for demonstration if real dataset not found."""
    # If you have the real data CSV, you could load it here.
    # For this demo, we create 20 dummy QA pairs with yes/no/maybe answers.
    synthetic = {
        "train": [
            {"question": "Is aspirin effective for headache?", "answer": "yes"},
            {"question": "Does paracetamol cause liver damage?", "answer": "maybe"},
            {"question": "Can antibiotics treat viral infections?", "answer": "no"},
            {"question": "Is exercise beneficial for heart health?", "answer": "yes"},
            {"question": "Does smoking increase lung cancer risk?", "answer": "yes"},
            {"question": "Can meditation reduce blood pressure?", "answer": "maybe"},
            {"question": "Is surgery necessary for all hernias?", "answer": "no"},
            {"question": "Does vitamin C prevent colds?", "answer": "maybe"},
            {"question": "Is coffee harmful during pregnancy?", "answer": "maybe"},
            {"question": "Can stress cause ulcers?", "answer": "yes"},
        ],
        "validation": [
            {"question": "Does sugar cause hyperactivity in children?", "answer": "no"},
            {"question": "Is MRI safe during pregnancy?", "answer": "maybe"},
        ],
        "test": [
            {"question": "Can probiotics treat IBS?", "answer": "maybe"},
            {"question": "Is sunscreen effective against UV rays?", "answer": "yes"},
            {"question": "Does high cholesterol always lead to heart disease?", "answer": "no"},
            {"question": "Can acupuncture relieve back pain?", "answer": "yes"},
            {"question": "Is gluten harmful for everyone?", "answer": "no"},
            {"question": "Can lack of sleep cause weight gain?", "answer": "yes"},
            {"question": "Does drinking water help with kidney stones?", "answer": "yes"},
            {"question": "Is yoga good for flexibility?", "answer": "yes"},
        ]
    }
    return synthetic

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
class Config:
    dataset_name = "pubmedqa"          # not used directly (we use synthetic)
    base_models = ["distilgpt2"]       # small model for quick testing
    max_tokens = 50
    temperature = 0.7
    top_p = 0.9
    output_dir = "./results"

config = Config()

# -------------------- PromptEngineer Class --------------------
class PromptEngineer:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.templates = prepare_prompt_templates()
        self.results = []

    def load_model(self):
        print(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        ).to(self.device)
        self.model.eval()
        print("Model loaded successfully!")

    def generate_response(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Try to extract answer after known markers
        for marker in ["[Answer]:", "[Final Answer]:", "Answer:"]:
            if marker in response:
                response = response.split(marker)[-1].strip()
                break
        return response

    def run_zero_shot(self, questions):
        print("Running zero-shot experiments...")
        results = []
        for i, q in enumerate(questions):
            prompt = self.templates["zero_shot"].format(question=q["question"])
            start = time.time()
            ans = self.generate_response(prompt)
            elapsed = time.time() - start
            results.append({
                "question": q["question"],
                "ground_truth": q.get("answer", ""),
                "prediction": ans,
                "inference_time": elapsed
            })
            if (i+1) % 5 == 0:
                print(f"  Processed {i+1}/{len(questions)}")
        return results

    def run_few_shot(self, questions, train_data, num_shots=3):
        print(f"Running few-shot experiments (shots={num_shots})...")
        results = []
        if not train_data:
            print("  No training data; falling back to zero-shot.")
            return self.run_zero_shot(questions)
        for i, q in enumerate(questions):
            examples = format_few_shot_examples(train_data, num_shots)
            # Ensure required keys exist
            for j in range(1, num_shots+1):
                if f"example_question_{j}" not in examples:
                    examples[f"example_question_{j}"] = q["question"]
                    examples[f"example_answer_{j}"] = ""
            examples["question"] = q["question"]
            prompt = self.templates["few_shot"].format(**examples)
            start = time.time()
            ans = self.generate_response(prompt)
            elapsed = time.time() - start
            results.append({
                "question": q["question"],
                "ground_truth": q.get("answer", ""),
                "prediction": ans,
                "inference_time": elapsed,
                "num_shots": num_shots
            })
            if (i+1) % 5 == 0:
                print(f"  Processed {i+1}/{len(questions)}")
        return results

    def run_chain_of_thought(self, questions):
        print("Running Chain-of-Thought...")
        results = []
        for i, q in enumerate(questions):
            prompt = self.templates["chain_of_thought"].format(question=q["question"])
            start = time.time()
            ans = self.generate_response(prompt)
            elapsed = time.time() - start
            results.append({
                "question": q["question"],
                "ground_truth": q.get("answer", ""),
                "prediction": ans,
                "inference_time": elapsed
            })
            if (i+1) % 5 == 0:
                print(f"  Processed {i+1}/{len(questions)}")
        return results

    def run_role_based(self, questions, domain="cardiology"):
        print("Running role-based experiments...")
        results = []
        for i, q in enumerate(questions):
            prompt = self.templates["role_based"].format(question=q["question"], domain=domain)
            start = time.time()
            ans = self.generate_response(prompt)
            elapsed = time.time() - start
            results.append({
                "question": q["question"],
                "ground_truth": q.get("answer", ""),
                "prediction": ans,
                "inference_time": elapsed,
                "domain": domain
            })
            if (i+1) % 5 == 0:
                print(f"  Processed {i+1}/{len(questions)}")
        return results

    def run_prompt_ensemble(self, questions):
        print("Running prompt ensemble...")
        results = []
        prompt_types = ["zero_shot", "chain_of_thought", "role_based"]
        for i, q in enumerate(questions):
            answers = []
            times = []
            for ptype in prompt_types:
                if ptype == "role_based":
                    prompt = self.templates[ptype].format(question=q["question"], domain="general medicine")
                else:
                    prompt = self.templates[ptype].format(question=q["question"])
                start = time.time()
                ans = self.generate_response(prompt)
                times.append(time.time() - start)
                answers.append(ans)
            # Majority vote
            final = Counter(answers).most_common(1)[0][0]
            avg_time = sum(times)/len(times)
            results.append({
                "question": q["question"],
                "ground_truth": q.get("answer", ""),
                "prediction": final,
                "inference_time": avg_time,
                "ensemble_size": len(prompt_types)
            })
            if (i+1) % 5 == 0:
                print(f"  Processed {i+1}/{len(questions)}")
        return results

    def run_experiments(self, test_data, train_data=None):
        self._add_results(self.run_zero_shot(test_data), "zero_shot")
        self._add_results(self.run_few_shot(test_data, train_data, 3), "few_shot_3")
        self._add_results(self.run_few_shot(test_data, train_data, 5), "few_shot_5")
        self._add_results(self.run_chain_of_thought(test_data), "chain_of_thought")
        self._add_results(self.run_role_based(test_data), "role_based")
        self._add_results(self.run_prompt_ensemble(test_data), "prompt_ensemble")

    def _add_results(self, results, method):
        for r in results:
            r["method"] = method
        self.results.extend(results)

# -------------------- Main --------------------
def main():
    set_seed(42)
    print("="*60)
    print("PROMPT ENGINEERING EXPERIMENTS")
    print("="*60)

    # Load data (synthetic)
    data = load_healthcare_dataset()
    train_data = data.get("train", [])
    val_data = data.get("validation", [])
    test_data = data.get("test", [])
    if not test_data:
        test_data = val_data
    print(f"Train: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")

    for model_name in config.base_models:
        print(f"\n{'='*50}\nModel: {model_name}\n{'='*50}")
        engineer = PromptEngineer(model_name)
        engineer.load_model()
        engineer.run_experiments(test_data, train_data)

        # Save results
        short = model_name.split("/")[-1]
        save_results(engineer.results, f"prompt_engineering_{short}.csv", config.output_dir)

    print("\nAll experiments complete!")

if __name__ == "__main__":
    main()