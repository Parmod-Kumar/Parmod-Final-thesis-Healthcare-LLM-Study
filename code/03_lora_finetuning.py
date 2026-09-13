"""
03_lora_finetuning.py - LoRA Fine-Tuning
Self‑contained version – runs without external modules.

This module demonstrates parameter-efficient fine-tuning with LoRA on a small
medical QA dataset. It focuses on reducing GPU memory usage while preserving
training quality for downstream question-answering tasks.
"""

import os
import time
import random
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import numpy as np

# -------------------- Utilities (inline) --------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def save_results(results, filename, output_dir="./results"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, filename), index=False)
    print(f"Results saved to {os.path.join(output_dir, filename)}")

def load_healthcare_dataset(dataset_name="pubmedqa"):
    """Generate a small synthetic dataset for demonstration."""
    # If you have real CSV data, you can load it here instead.
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
    # Model
    base_models = ["distilgpt2"]          # small model for quick testing
    # LoRA
    lora_r = 4
    lora_alpha = 8
    lora_dropout = 0.1
    # For GPT2-style models, typical target modules are:
    lora_target_modules = ["c_attn", "c_proj"]  # adjust for other architectures
    # Training
    num_epochs = 2
    batch_size = 2
    gradient_accumulation_steps = 1
    learning_rate = 2e-4
    max_tokens = 50
    temperature = 0.7
    top_p = 0.9
    # Paths
    output_dir = "./results"
    model_save_dir = "./models"
    dataset_name = "pubmedqa"  # not used (synthetic)

config = Config()

# -------------------- LoRATrainer Class --------------------
class LoRATrainer:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.peft_model = None

    def load_model(self):
        print(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        ).to(self.device)
        print("Model loaded successfully!")

    def setup_lora(self):
        print("Setting up LoRA configuration...")
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        self.peft_model = get_peft_model(self.model, lora_config)
        self.peft_model.print_trainable_parameters()
        return self.peft_model

    def prepare_dataset(self, data: list) -> Dataset:
        formatted_data = []
        for item in data:
            prompt = f"Question: {item['question']}\nAnswer:"
            response = item["answer"]
            formatted_data.append({"text": prompt + " " + response})

        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=128  # short for speed
            )

        dataset = Dataset.from_list(formatted_data)
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        return tokenized_dataset

    def train(self, train_data: list, val_data: list):
        print("Preparing datasets...")
        train_dataset = self.prepare_dataset(train_data)
        val_dataset = self.prepare_dataset(val_data) if val_data else None

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        training_kwargs = dict(
            output_dir=f"{config.model_save_dir}/lora_{self.model_name.split('/')[-1]}",
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            warmup_steps=10,
            logging_steps=5,
            eval_steps=10,
            save_steps=50,
            save_total_limit=2,
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="eval_loss" if val_dataset else None,
            fp16=torch.cuda.is_available(),
            report_to="none"
        )

        if "eval_strategy" in TrainingArguments.__init__.__code__.co_varnames:
            training_kwargs["eval_strategy"] = "steps" if val_dataset else "no"
        else:
            training_kwargs["evaluation_strategy"] = "steps" if val_dataset else "no"

        training_args = TrainingArguments(**training_kwargs)

        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )

        print("Starting training...")
        start_time = time.time()
        trainer.train()
        training_time = time.time() - start_time
        print(f"Training completed in {training_time/60:.2f} minutes")

        # Save adapters
        self.peft_model.save_pretrained(
            f"{config.model_save_dir}/lora_{self.model_name.split('/')[-1]}"
        )
        self.tokenizer.save_pretrained(
            f"{config.model_save_dir}/lora_{self.model_name.split('/')[-1]}"
        )
        return training_time

    def generate_response(self, question: str) -> tuple:
        prompt = f"Question: {question}\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        start_time = time.time()
        with torch.no_grad():
            outputs = self.peft_model.generate(
                **inputs,
                max_new_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        inference_time = time.time() - start_time

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "Answer:" in response:
            response = response.split("Answer:")[-1].strip()
        return response, inference_time

    def evaluate(self, test_data: list) -> list:
        print(f"Evaluating on {len(test_data)} test samples...")
        results = []
        for i, item in enumerate(test_data):
            prediction, inf_time = self.generate_response(item["question"])
            results.append({
                "question": item["question"],
                "ground_truth": item["answer"],
                "prediction": prediction,
                "inference_time": inf_time
            })
            if (i + 1) % 5 == 0:
                print(f"  Processed {i+1}/{len(test_data)}")
        return results

# -------------------- Main --------------------
def main():
    set_seed(42)
    print("=" * 60)
    print("LORA FINE-TUNING EXPERIMENTS")
    print("=" * 60)

    # Load synthetic data
    data = load_healthcare_dataset()
    train_data = data.get("train", [])
    val_data = data.get("validation", [])
    test_data = data.get("test", [])
    if not test_data:
        test_data = val_data

    print(f"Train: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")

    all_results = []

    for model_name in config.base_models:
        print(f"\n{'='*50}\nModel: {model_name}\n{'='*50}")
        trainer = LoRATrainer(model_name)
        trainer.load_model()
        trainer.setup_lora()

        training_time = trainer.train(train_data, val_data)

        results = trainer.evaluate(test_data)

        for r in results:
            r["model"] = model_name
            r["method"] = "lora"
            r["training_time"] = training_time

        all_results.extend(results)

        short_name = model_name.split("/")[-1]
        save_results(results, f"lora_results_{short_name}.csv", config.output_dir)

    save_results(all_results, "lora_results_combined.csv", config.output_dir)
    print("\nLoRA fine-tuning experiments complete!")

if __name__ == "__main__":
    main()